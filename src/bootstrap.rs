use crate::stats::PrefixStats;
use crate::types::ScanRustResult;
use crate::wasserstein::{sort_f64, wasserstein_1d_sorted};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Bootstrap replications are only spread across threads above this count;
/// below it the coordination costs more than the work saved.
const PAR_REPS_MIN: usize = 64;

/// Upper bound on the number of tasks one replication sweep is split into.
/// Replications are individually tiny, so a coarse split keeps scheduling
/// overhead well under the work being distributed.
const PAR_REPS_TASKS: usize = 8;

/// Default bootstrap block size used when the user does not provide `b`.
pub fn default_block_size(m: usize) -> usize {
    let b = if m < 100 {
        (m as f64).sqrt().round() as usize
    } else {
        (m as f64).powf(1.0 / 3.0).round() as usize
    };
    usize::max(8, usize::min(b, m))
}

/// Construct a taper window for tapered block bootstrap resampling.
pub fn create_taper_window(length: usize, ratio: f64) -> Vec<f64> {
    let mut taper = vec![1.0; length];
    let slope_len = ((length as f64) * ratio / 2.0).floor() as usize;

    if slope_len > 0 {
        let scale = 1.0 / (slope_len + 1) as f64;
        for i in 0..slope_len {
            let value = (i + 1) as f64 * scale;
            taper[i] = value;
            taper[length - 1 - i] = value;
        }
    }

    taper
}

/// Linear-interpolated percentile.
///
/// Only the one or two order statistics straddling the requested percentile are
/// needed, so the slice is partitioned around them (`O(n)`) rather than fully
/// sorted. The slice is still reordered in place; the returned value is
/// identical to the sort-then-index formulation.
pub fn percentile_linear(values: &mut [f64], percent: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let p = percent.clamp(0.0, 100.0) * 0.01;
    let n = values.len();

    if n == 1 {
        return values[0];
    }

    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;

    // After this, `at_lo` holds exactly the element a full sort would place at
    // index `lo`, and `above` holds the elements a full sort would place after it.
    let (_, at_lo, above) = values.select_nth_unstable_by(lo, f64::total_cmp);
    let value_lo = *at_lo;

    if lo == hi {
        return value_lo;
    }

    // `hi` is always `lo + 1` here, so the element at `hi` is the smallest of
    // the upper partition.
    let value_hi = above
        .iter()
        .copied()
        .reduce(|a, b| if b.total_cmp(&a).is_lt() { b } else { a })
        .unwrap_or(value_lo);

    let weight = h - lo as f64;
    value_lo.mul_add(1.0 - weight, value_hi * weight)
}

/// Deterministically mix seed components so each task gets its own RNG stream.
#[inline]
pub fn seed_from_parts(seed: u64, start: usize, w: usize, salt: u64) -> u64 {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    x ^= (start as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= (w as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= salt.wrapping_mul(0xD6E8_FD50_1B3D_F1AB);
    x
}

/// Precomputed taper parameters so they are not recalculated for every
/// bootstrap replication.
pub struct TaperParams {
    window: Vec<f64>,
    norm_factor: f64,
}

impl TaperParams {
    pub fn new(block_len: usize, taper_ratio: f64) -> Self {
        let window = create_taper_window(block_len, taper_ratio);
        let taper_norm = window.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_factor = (block_len as f64).sqrt() / taper_norm;
        Self {
            window,
            norm_factor,
        }
    }
}

/// Generate many tapered-block-bootstrap Wasserstein distances, writing into a
/// caller-owned buffer.
///
/// Each replication is seeded purely from `(seed, rep_id)`, so replications are
/// independent and the produced sequence does not depend on how the work is
/// scheduled — `parallel` changes only the speed, never the output.
#[allow(clippy::too_many_arguments)]
fn tbb_distances_into(
    pooled: &[f64],
    w: usize,
    b_reps: usize,
    block_len: usize,
    taper: &TaperParams,
    seed: u64,
    parallel: bool,
    out: &mut Vec<f64>,
) -> ScanRustResult<()> {
    let n_views = pooled
        .len()
        .checked_sub(block_len)
        .map(|v| v + 1)
        .ok_or_else(|| "reference series is shorter than block length".to_string())?;

    if n_views == 0 {
        return Err("reference series is shorter than block length".to_string());
    }

    let total_len = 2 * w;
    let k = total_len.div_ceil(block_len);

    // One replication: draw `k` tapered blocks, then compare the two halves.
    let replicate = |z: &mut Vec<f64>, rep_id: usize| -> f64 {
        let rep_seed = seed_from_parts(seed, rep_id, w, 10_007);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(rep_seed);

        z.clear();
        for _ in 0..k {
            // Draw for every block even when the last one is partly unused, so
            // the RNG stream stays aligned with the replication index.
            let idx = rng.random_range(0..n_views);
            // The tail of the final block would be discarded, so never build it.
            let take = block_len.min(total_len - z.len());
            let block = &pooled[idx..idx + take];
            for (&value, &weight) in block.iter().zip(&taper.window[..take]) {
                z.push(value * weight * taper.norm_factor);
            }
        }

        // `z` is owned scratch, so the halves are sorted in place instead of
        // being copied into two fresh vectors per replication.
        let (left, right) = z.split_at_mut(w);
        sort_f64(left);
        sort_f64(right);
        wasserstein_1d_sorted(left, right)
    };

    out.clear();
    out.reserve(b_reps);

    if parallel && b_reps >= PAR_REPS_MIN {
        (0..b_reps)
            .into_par_iter()
            .with_min_len(b_reps.div_ceil(PAR_REPS_TASKS))
            .map_init(|| Vec::with_capacity(total_len), &replicate)
            // Writes straight into `out`'s spare capacity, in index order.
            .collect_into_vec(out);
    } else {
        let mut z: Vec<f64> = Vec::with_capacity(total_len);
        out.extend((0..b_reps).map(|rep_id| replicate(&mut z, rep_id)));
    }

    Ok(())
}

/// Buffers, precomputed taper, and execution policy reused across the splits of
/// one window size.
///
/// The taper depends only on `(block_len, taper_ratio)`, both of which are
/// constant while scanning a single window size, so it is built once per
/// window instead of once per threshold evaluation.
pub struct ThresholdScratch {
    pooled: Vec<f64>,
    dists: Vec<f64>,
    taper: Option<(usize, u64, TaperParams)>,
    parallel_reps: bool,
}

impl ThresholdScratch {
    /// `parallel_reps` spreads each replication sweep across the Rayon pool.
    /// Callers that already fan out over the pool at a coarser level should
    /// pass `false`: the nested region then costs more than it saves.
    pub fn new(parallel_reps: bool) -> Self {
        Self {
            pooled: Vec::new(),
            dists: Vec::new(),
            taper: None,
            parallel_reps,
        }
    }
}

/// Compute the local tapered block bootstrap detection threshold for one window
/// pair, reusing a caller-owned scratch (buffers, cached taper, and the
/// nested-parallelism policy — see [`ThresholdScratch::new`]).
#[allow(clippy::too_many_arguments)]
pub fn compute_tapered_block_bootstrap_threshold_with(
    scratch: &mut ThresholdScratch,
    series: &[f64],
    prefix: &PrefixStats,
    start: usize,
    w: usize,
    delta: usize,
    b_reps: usize,
    seed: u64,
    q_percent: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    eps: f64,
    _batch_size: usize,
) -> ScanRustResult<f64> {
    if delta != w {
        return Err("this implementation assumes delta == w".to_string());
    }

    let total_len = w + delta;
    if start + total_len > series.len() {
        return Ok(f64::INFINITY);
    }

    let left_start = start;
    let right_start = start + w;

    let (left_mean, left_std) = prefix.mean_std(left_start, w, eps);
    let (right_mean, right_std) = prefix.mean_std(right_start, delta, eps);

    let left_std_inv = left_std.recip();
    let right_std_inv = right_std.recip();

    // Split the scratch fields up front so the buffers can be borrowed
    // independently below.
    let ThresholdScratch {
        pooled,
        dists,
        taper: cached_taper,
        parallel_reps,
    } = scratch;

    pooled.clear();
    pooled.reserve(total_len);

    let left = &series[left_start..left_start + w];
    let right = &series[right_start..right_start + delta];

    if center {
        pooled.extend(left.iter().map(|v| (v - left_mean) * left_std_inv));
        pooled.extend(right.iter().map(|v| (v - right_mean) * right_std_inv));
    } else {
        pooled.extend(left.iter().map(|v| v * left_std_inv));
        pooled.extend(right.iter().map(|v| v * right_std_inv));
    }

    let m = pooled.len();
    let block_len = match b {
        Some(value) => usize::max(3, usize::min(value, m)),
        None => default_block_size(m),
    };

    let ratio_bits = taper_ratio.to_bits();
    if !matches!(cached_taper, Some((len, bits, _)) if *len == block_len && *bits == ratio_bits) {
        *cached_taper = Some((
            block_len,
            ratio_bits,
            TaperParams::new(block_len, taper_ratio),
        ));
    }
    let taper = &cached_taper.as_ref().expect("taper just populated").2;

    let bootstrap_seed = seed_from_parts(seed, start, w, 999);

    tbb_distances_into(
        pooled,
        w,
        b_reps,
        block_len,
        taper,
        bootstrap_seed,
        *parallel_reps,
        dists,
    )?;

    // Rescale the standardized bootstrap distances back to the local data scale.
    let local_scale = (0.5 * (left_std.powi(2) + right_std.powi(2))).sqrt();
    for value in dists.iter_mut() {
        *value *= local_scale;
    }

    Ok(percentile_linear(dists, 100.0 - q_percent))
}
