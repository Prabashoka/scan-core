use crate::aggregate::{cdf_from_segment_votes, compute_change_points_with_votes};
use crate::bootstrap::compute_tapered_block_bootstrap_threshold;
use crate::refine::refine_for_change_type;
use crate::stats::PrefixStats;
use crate::types::{ChangeType, ScanResult, ScanRustResult, WindowScanResult};
use crate::validation::{validate_series, validate_window_sizes};
use crate::wasserstein::wasserstein_1d;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Scan one chosen window size over the series.
///
/// The function compares every adjacent pair of non-overlapping windows and
/// computes a local tapered block bootstrap threshold for each comparison.
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

    // Split tests are independent. Indexed parallel collection preserves their
    // original order and the start-derived bootstrap seeds keep results stable
    // across Rayon scheduling choices.
    let split_results: Vec<(usize, f64, f64, bool)> = (0..n_splits)
        .into_par_iter()
        .map(|m_idx| -> ScanRustResult<(usize, f64, f64, bool)> {
            let start = m_idx * w;
            let split = start + w;
            let end = split + w;

            let threshold = compute_tapered_block_bootstrap_threshold(
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
            let statistic = wasserstein_1d(&series[start..split], &series[split..end]);

            Ok((start, statistic, threshold, statistic > threshold))
        })
        .collect::<ScanRustResult<Vec<_>>>()?;

    let mut starts = Vec::with_capacity(n_splits);
    let mut statistics = Vec::with_capacity(n_splits);
    let mut tapered_block_bootstrap_threshold_values = Vec::with_capacity(n_splits);
    let mut rejected_splits = Vec::new();

    for (m_idx, (start, statistic, threshold, rejected)) in split_results.into_iter().enumerate() {
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
    let mut change_points = Vec::new();
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
        let localization_start = first * w;
        let localization_end = (last + 2) * w;
        let localization_region = &series[localization_start..localization_end];
        localized_regions.push((localization_start, localization_end));
        let local_cp = refine_for_change_type(localization_region, change_type)?;
        let cp =
            (localization_start + local_cp).clamp(localization_start + 1, localization_end - 1);
        change_points.push(cp);

        region_start = region_end + 1;
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

    let backend_lower = backend.to_ascii_lowercase();
    if backend_lower != "thread" && backend_lower != "process" {
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
        window_sizes
            .par_iter()
            .map(|&w| {
                detect_for_window(
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
