use crate::aggregate::{cdf_from_segment_votes, compute_change_points_with_votes};
use crate::bootstrap::{compute_tapered_block_bootstrap_threshold_with, ThresholdScratch};
use crate::refine::refine_for_change_type;
use crate::stats::PrefixStats;
use crate::types::{ChangeType, ScanResult, ScanRustResult, WindowScanResult};
use crate::validation::{validate_series, validate_window_sizes};
use crate::wasserstein::wasserstein_1d_with_scratch;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Localized regions are only refined in parallel once there are at least this
/// many of them; below that the sequential pass is cheaper.
const PAR_REGIONS_MIN: usize = 8;

/// Scan one chosen window size over the series.
///
/// The function compares every adjacent pair of non-overlapping windows and
/// computes a tapered block bootstrap threshold for the first comparison,
/// reuses it for screening, and refreshes it after a rejected split.
/// Adjacent rejected splits are merged before each consolidated region is
/// refined to a single candidate change-point. The detailed vectors are
/// intentionally kept for the Python research API.
#[allow(clippy::too_many_arguments)]
pub fn detect_for_window(
    series: &[f64],
    prefix: &PrefixStats,
    w: usize,
    n_boot: usize,
    alpha_q_percent: f64,
    seed: u64,
    change_type: ChangeType,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> ScanRustResult<(usize, WindowScanResult)> {
    // A direct call is not competing with sibling windows for the pool, so the
    // bootstrap is free to use it.
    detect_for_window_with_policy(
        series,
        prefix,
        w,
        n_boot,
        alpha_q_percent,
        seed,
        change_type,
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
        true,
    )
}

/// [`detect_for_window`] with explicit control over nested parallelism.
///
/// `parallel_bootstrap` only affects scheduling. Every replication is seeded
/// from its own index, so the numbers produced are identical either way.
#[allow(clippy::too_many_arguments)]
fn detect_for_window_with_policy(
    series: &[f64],
    prefix: &PrefixStats,
    w: usize,
    n_boot: usize,
    alpha_q_percent: f64,
    seed: u64,
    change_type: ChangeType,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
    parallel_bootstrap: bool,
) -> ScanRustResult<(usize, WindowScanResult)> {
    let n = series.len();
    let n_splits = n
        .checked_div(w)
        .and_then(|n_windows| n_windows.checked_sub(1))
        .unwrap_or(0);

    if n_splits == 0 {
        return Ok((
            w,
            WindowScanResult {
                change_points: Vec::new(),
                starts: Vec::new(),
                statistics: Vec::new(),
                tapered_block_bootstrap_threshold: Vec::new(),
                localized_regions: Vec::new(),
            },
        ));
    }

    // Algorithm 2 applies its multiplicity correction over the M splits for
    // this window size, rather than over the number of window sizes scanned.
    let corrected_q = alpha_q_percent / n_splits as f64;

    let mut starts = Vec::with_capacity(n_splits);
    let mut statistics = Vec::with_capacity(n_splits);
    let mut tapered_block_bootstrap_threshold_values = Vec::with_capacity(n_splits);
    let mut rejected_splits: Vec<usize> = Vec::new();

    // Buffers live across the whole split sweep: the bootstrap scratch (which
    // also caches this window's taper) and the two sorted copies used for the
    // observed split statistic.
    let mut scratch = ThresholdScratch::new(parallel_bootstrap);
    let mut left_sorted: Vec<f64> = Vec::with_capacity(w);
    let mut right_sorted: Vec<f64> = Vec::with_capacity(w);

    // Bootstrap the first split, then reuse that threshold until a rejection.
    // A rejection causes the threshold to be refreshed at the next split.
    let mut threshold = f64::NAN;
    let mut refresh_threshold = true;

    for m_idx in 0..n_splits {
        let start = m_idx * w;
        let split = start + w;
        let end = split + w;

        if refresh_threshold {
            threshold = compute_tapered_block_bootstrap_threshold_with(
                &mut scratch,
                series,
                prefix,
                start,
                w,
                w,
                n_boot,
                seed,
                corrected_q,
                b,
                taper_ratio,
                center,
                eps,
                batch_size,
            )?;
        }

        let statistic = wasserstein_1d_with_scratch(
            &series[start..split],
            &series[split..end],
            &mut left_sorted,
            &mut right_sorted,
        );
        let rejected = statistic > threshold;
        refresh_threshold = rejected;

        starts.push(start);
        statistics.push(statistic);
        tapered_block_bootstrap_threshold_values.push(threshold);
        if rejected {
            rejected_splits.push(m_idx);
        }
    }

    // Consume adjacent rejected splits from left to right in groups of at most
    // two. A change can reject two neighbouring split tests, while a longer
    // run is treated as evidence for additional localization regions. Thus
    // [3, 4, 5, 6, 8] becomes [3, 4], [5, 6], [8].
    let mut localized_regions = Vec::new();
    let mut region_start = 0usize;
    while region_start < rejected_splits.len() {
        let first = rejected_splits[region_start];
        let mut region_end = region_start;

        if region_start + 1 < rejected_splits.len()
            && rejected_splits[region_start + 1] == first + 1
        {
            region_end += 1;
        }

        let last = rejected_splits[region_end];
        localized_regions.push((first * w, (last + 2) * w));

        region_start = region_end + 1;
    }

    // Refining a region only depends on that region's slice, so the regions are
    // independent. Results are collected by index, which keeps both the
    // change-point order and the reported error identical to a serial sweep.
    let refine_region = |&(localization_start, localization_end): &(usize, usize)| {
        let localization_region = &series[localization_start..localization_end];
        refine_for_change_type(localization_region, change_type).map(|local_cp| {
            (localization_start + local_cp).clamp(localization_start + 1, localization_end - 1)
        })
    };

    let refined: Vec<ScanRustResult<usize>> = if localized_regions.len() >= PAR_REGIONS_MIN {
        localized_regions.par_iter().map(refine_region).collect()
    } else {
        localized_regions.iter().map(refine_region).collect()
    };

    let mut change_points = Vec::with_capacity(refined.len());
    for cp in refined {
        change_points.push(cp?);
    }

    Ok((
        w,
        WindowScanResult {
            change_points,
            starts,
            statistics,
            tapered_block_bootstrap_threshold: tapered_block_bootstrap_threshold_values,
            localized_regions,
        },
    ))
}

/// Main Rust engine called by all Python-facing wrappers.
#[allow(clippy::too_many_arguments)]
pub fn run_scan_detector(
    series: Vec<f64>,
    window_sizes: Option<Vec<usize>>,
    n_boot: usize,
    alpha_q: f64,
    seed: u64,
    tol: usize,
    workers: Option<usize>,
    backend: &str,
    change_type: &str,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> ScanRustResult<ScanResult> {
    validate_series(&series)?;

    let window_sizes = window_sizes.unwrap_or_else(|| (10usize..=20usize).collect());
    validate_window_sizes(&window_sizes)?;

    if !backend.eq_ignore_ascii_case("thread") && !backend.eq_ignore_ascii_case("process") {
        return Err(
            "backend must be 'thread' or 'process'. Rust uses Rayon threads internally for both options."
                .to_string(),
        );
    }

    let ct = ChangeType::parse(change_type)?;

    // Accept either 0.01-style or 1.0-style percentage inputs.
    let alpha_percent = if alpha_q <= 1.0 {
        100.0 * alpha_q
    } else {
        alpha_q
    };
    let batch_size = batch_size.max(1);

    let prefix = PrefixStats::from_series(&series);

    let compute = || -> Vec<ScanRustResult<(usize, WindowScanResult)>> {
        // Window sizes are the primary fan-out. Letting each of them also
        // parallelize its bootstrap only pays off when that fan-out cannot fill
        // the pool on its own; once it can, the nested regions are pure
        // overhead. Measured on a 16-thread pool: 8 windows run ~2x faster with
        // nesting, 11 windows run ~2x slower with it.
        let parallel_bootstrap = window_sizes.len() * 2 <= rayon::current_num_threads();

        window_sizes
            .par_iter()
            .map(|&w| {
                detect_for_window_with_policy(
                    &series,
                    &prefix,
                    w,
                    n_boot,
                    alpha_percent,
                    seed,
                    ct,
                    eps,
                    b,
                    taper_ratio,
                    center,
                    batch_size,
                    parallel_bootstrap,
                )
            })
            .collect()
    };

    let results = if let Some(n_threads) = workers.filter(|&n| n > 0) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| format!("failed to build Rayon thread pool: {e}"))?
            .install(compute)
    } else {
        compute()
    };

    let mut cp_dict = BTreeMap::new();
    let mut window_results = BTreeMap::new();

    for item in results {
        let (w, result) = item?;
        cp_dict.insert(w, result.change_points.clone());
        window_results.insert(w, result);
    }

    let segments = compute_change_points_with_votes(&cp_dict, tol);
    let out = cdf_from_segment_votes(&segments, cp_dict.len())?;

    Ok(ScanResult {
        cp_dict,
        window_results,
        segments,
        out,
    })
}
