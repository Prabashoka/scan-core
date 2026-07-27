use crate::stats::mean;
use crate::types::{ChangeType, ScanRustResult};
use crate::wasserstein::wasserstein_1d;

/// CUSUM-style localization used for pure mean changes.
pub fn refine_cp_cusum(y: &[f64]) -> ScanRustResult<usize> {
    if y.len() < 3 {
        return Err("need at least 3 points".to_string());
    }

    let y_mean = mean(y);
    let mut running = 0.0f64;
    let mut best_idx = 0usize;
    let mut best_abs = f64::NEG_INFINITY;

    for (i, value) in y.iter().take(y.len() - 1).enumerate() {
        running += y_mean - value;
        let score = running.abs();
        if score > best_abs {
            best_abs = score;
            best_idx = i;
        }
    }

    Ok(best_idx + 1)
}

/// Wasserstein localization used for variance or broader distributional changes.
pub fn refine_cp_wasserstein(y: &[f64]) -> ScanRustResult<(usize, Vec<f64>)> {
    if y.len() < 3 {
        return Err("need at least 3 points".to_string());
    }

    let n = y.len();
    let n_f64 = n as f64;
    let mut stats = vec![f64::NAN; n];
    let mut best_k = 1usize;
    let mut best_score = f64::NEG_INFINITY;

    for t in 1..n {
        let scale = ((t as f64) * ((n - t) as f64) / n_f64).sqrt();
        let score = scale * wasserstein_1d(&y[..t], &y[t..]);
        stats[t] = score;

        if score > best_score {
            best_score = score;
            best_k = t;
        }
    }

    Ok((best_k, stats))
}

/// Dispatch to the refinement method matching the requested change type.
pub fn refine_for_change_type(block: &[f64], change_type: ChangeType) -> ScanRustResult<usize> {
    match change_type {
        ChangeType::Mean => refine_cp_cusum(block),
        ChangeType::Var | ChangeType::Distribution => refine_cp_wasserstein(block).map(|(k, _)| k),
    }
}
