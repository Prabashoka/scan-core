use crate::stats::PrefixStats;
use crate::types::ScanRustResult;
use crate::wasserstein::wasserstein_1d;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

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

/// Linear-interpolated percentile after sorting in place.
pub fn percentile_linear(values: &mut [f64], percent: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    values.sort_unstable_by(|a, b| a.total_cmp(b));

    let p = percent.clamp(0.0, 100.0) * 0.01;
    let n = values.len();

    if n == 1 {
        return values[0];
    }

    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;

    if lo == hi {
        values[lo]
    } else {
        let weight = h - lo as f64;
        values[lo].mul_add(1.0 - weight, values[hi] * weight)
    }
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

/// Precomputed taper parameters so they are not recalculated inside the
/// parallel closure on every bootstrap replication.
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

/// Generate many tapered-block-bootstrap Wasserstein distances.
#[allow(clippy::too_many_arguments)]
pub fn batched_tbb_distances(
    pooled: &[f64],
    w: usize,
    b_reps: usize,
    block_len: usize,
    taper: &TaperParams,
    seed: u64,
    batch_size: usize,
) -> ScanRustResult<Vec<f64>> {
    let n_views = pooled
        .len()
        .checked_sub(block_len)
        .map(|v| v + 1)
        .ok_or_else(|| "reference series is shorter than block length".to_string())?;

    if n_views == 0 {
        return Err("reference series is shorter than block length".to_string());
    }

    let total_len = 2 * w;
    let k = (total_len + block_len - 1) / block_len;
    let batch_size = batch_size.max(1);
    let z_cap = k * block_len;

    let mut dists = vec![0.0; b_reps];

    dists
        .par_chunks_mut(batch_size)
        .enumerate()
        .for_each(|(chunk_id, chunk)| {
            let base_rep = chunk_id * batch_size;

            // Allocate the bootstrap sample buffer once per Rayon chunk/thread
            // and reuse it across bootstrap replications.
            let mut z: Vec<f64> = Vec::with_capacity(z_cap);

            for (offset, out) in chunk.iter_mut().enumerate() {
                let rep_id = base_rep + offset;

                let rep_seed = seed_from_parts(seed, rep_id, w, 10_007);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rep_seed);

                z.clear();

                for _ in 0..k {
                    let idx = rng.random_range(0..n_views);
                    for j in 0..block_len {
                        z.push(pooled[idx + j] * taper.window[j] * taper.norm_factor);
                    }
                }

                z.truncate(total_len);

                // Use the general Wasserstein function. It works for the two
                // equal-length bootstrap windows without needing a separate
                // equal-length helper.
                *out = wasserstein_1d(&z[..w], &z[w..2 * w]);
            }
        });

    Ok(dists)
}

/// Compute the local tapered block bootstrap detection threshold for one window pair.
#[allow(clippy::too_many_arguments)]
pub fn compute_tapered_block_bootstrap_threshold(
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
    batch_size: usize,
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

    let mut pooled: Vec<f64> = Vec::with_capacity(total_len);

    if center {
        pooled.extend(
            series[left_start..left_start + w]
                .iter()
                .map(|v| (v - left_mean) * left_std_inv),
        );
        pooled.extend(
            series[right_start..right_start + delta]
                .iter()
                .map(|v| (v - right_mean) * right_std_inv),
        );
    } else {
        pooled.extend(
            series[left_start..left_start + w]
                .iter()
                .map(|v| v * left_std_inv),
        );
        pooled.extend(
            series[right_start..right_start + delta]
                .iter()
                .map(|v| v * right_std_inv),
        );
    }

    let m = pooled.len();
    let block_len = match b {
        Some(value) => usize::max(3, usize::min(value, m)),
        None => default_block_size(m),
    };

    let taper = TaperParams::new(block_len, taper_ratio);
    let bootstrap_seed = seed_from_parts(seed, start, w, 999);

    let mut dists = batched_tbb_distances(
        &pooled,
        w,
        b_reps,
        block_len,
        &taper,
        bootstrap_seed,
        batch_size,
    )?;

    // Rescale the standardized bootstrap distances back to the local data scale.
    let local_scale = (0.5 * (left_std.powi(2) + right_std.powi(2))).sqrt();
    for value in &mut dists {
        *value *= local_scale;
    }

    Ok(percentile_linear(&mut dists, 100.0 - q_percent))
}
