use crate::types::ScanRustResult;

/// Validate the input time series before any scanning/refinement.
pub fn validate_series(series: &[f64]) -> ScanRustResult<()> {
    if series.len() < 3 {
        return Err("series must contain at least 3 values".to_string());
    }
    if series.iter().any(|x| !x.is_finite()) {
        return Err(
            "series contains NaN or infinite values; clean/impute before calling scan".to_string(),
        );
    }
    Ok(())
}

pub fn validate_window_sizes(window_sizes: &[usize]) -> ScanRustResult<()> {
    if window_sizes.is_empty() {
        return Err("window_sizes must not be empty".to_string());
    }
    if let Some(w) = window_sizes.iter().find(|&&w| w == 0) {
        return Err(format!("window_sizes must be positive, got {w}"));
    }
    Ok(())
}
