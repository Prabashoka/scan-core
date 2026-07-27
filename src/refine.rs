use crate::stats::mean;
use crate::types::{ChangeType, ScanRustResult};
use crate::wasserstein::{sort_f64, wasserstein_1d_sorted};

/// CUSUM-style localization used for pure mean changes.
pub fn refine_cp_cusum(y: &[f64]) -> ScanRustResult<usize> {
    if y.len() < 3 {
        return Err("need at least 3 points".to_string());
    }

    let y_mean = mean(y);
    let mut running = 0.0f64;
    let mut best_idx = 0usize;
    let mut best_abs = f64::NEG_INFINITY;

    for (i, value) in y[..y.len() - 1].iter().enumerate() {
        running += y_mean - value;
        let score = running.abs();
        if score > best_abs {
            best_abs = score;
            best_idx = i;
        }
    }

    Ok(best_idx + 1)
}

/// Insert `value` into an ascending (`total_cmp`) slice, keeping it sorted.
#[inline]
fn insert_sorted(sorted: &mut Vec<f64>, value: f64) {
    let pos = sorted.partition_point(|probe| probe.total_cmp(&value).is_lt());
    sorted.insert(pos, value);
}

/// Remove one occurrence of `value` from an ascending (`total_cmp`) slice.
///
/// `value` is required to be present. Two `f64`s that compare equal under
/// `total_cmp` are bit-identical, so which occurrence is dropped does not matter.
#[inline]
fn remove_sorted(sorted: &mut Vec<f64>, value: f64) {
    let pos = sorted.partition_point(|probe| probe.total_cmp(&value).is_lt());
    debug_assert!(pos < sorted.len() && sorted[pos].to_bits() == value.to_bits());
    sorted.remove(pos);
}

/// Walk every candidate split of `y`, reporting the scaled Wasserstein statistic
/// for each, and return the arg-max.
///
/// Splits are visited left to right, so the two sides can be maintained
/// incrementally: each step moves one point from the right sample to the left
/// one. That replaces the two fresh sorts (and two allocations) that a
/// per-split `wasserstein_1d` call would need.
fn scan_wasserstein_splits(
    y: &[f64],
    mut record: impl FnMut(usize, f64),
) -> ScanRustResult<usize> {
    if y.len() < 3 {
        return Err("need at least 3 points".to_string());
    }

    let n = y.len();
    let n_f64 = n as f64;

    let mut left: Vec<f64> = Vec::with_capacity(n);
    let mut right: Vec<f64> = y.to_vec();
    sort_f64(&mut right);

    let mut best_k = 1usize;
    let mut best_score = f64::NEG_INFINITY;

    for t in 1..n {
        // Move y[t - 1] across the split: left becomes y[..t], right y[t..].
        let moved = y[t - 1];
        insert_sorted(&mut left, moved);
        remove_sorted(&mut right, moved);

        let scale = ((t as f64) * ((n - t) as f64) / n_f64).sqrt();
        let score = scale * wasserstein_1d_sorted(&left, &right);
        record(t, score);

        if score > best_score {
            best_score = score;
            best_k = t;
        }
    }

    Ok(best_k)
}

/// Wasserstein localization used for variance or broader distributional changes.
pub fn refine_cp_wasserstein(y: &[f64]) -> ScanRustResult<(usize, Vec<f64>)> {
    let mut stats = vec![f64::NAN; y.len()];
    let best_k = scan_wasserstein_splits(y, |t, score| stats[t] = score)?;
    Ok((best_k, stats))
}

/// Dispatch to the refinement method matching the requested change type.
pub fn refine_for_change_type(block: &[f64], change_type: ChangeType) -> ScanRustResult<usize> {
    match change_type {
        ChangeType::Mean => refine_cp_cusum(block),
        // The per-split statistics are not needed here, so skip building them.
        ChangeType::Var | ChangeType::Distribution => scan_wasserstein_splits(block, |_, _| {}),
    }
}
