use serde::Serialize;
use std::collections::BTreeMap;

pub type ScanRustResult<T> = std::result::Result<T, String>;

/// Type of distributional change targeted by the refinement step.
#[derive(Clone, Copy, Debug, Serialize)]
pub enum ChangeType {
    Mean,
    Var,
    Distribution,
}

impl ChangeType {
    pub fn parse(value: &str) -> ScanRustResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "var" => Ok(Self::Var),
            "distribution" => Ok(Self::Distribution),
            other => Err(format!(
                "change_type must be one of {{'mean', 'var', 'distribution'}}, got {other:?}"
            )),
        }
    }
}

/// One merged segment of nearby candidate change-points and their votes.
#[derive(Clone, Debug, Serialize)]
pub struct SegmentInfo {
    pub change_points: Vec<usize>,
    pub votes: BTreeMap<usize, usize>,
    pub segment_vote: usize,
}

/// Detailed output for a single scan window.
#[derive(Clone, Debug, Serialize)]
pub struct WindowScanResult {
    pub change_points: Vec<usize>,
    pub starts: Vec<usize>,
    pub statistics: Vec<f64>,
    pub tapered_block_bootstrap_threshold: Vec<f64>,
    pub localized_regions: Vec<(usize, usize)>,
}

/// Aggregated voting output returned through the R API.
#[derive(Clone, Debug, Serialize)]
pub struct AggregatedOut {
    pub leaders_segment_votes: BTreeMap<usize, usize>,
    pub leaders_scores: BTreeMap<usize, f64>,
    pub leaders_probs: BTreeMap<usize, f64>,
    pub cdf: Vec<(usize, f64)>,
}

/// Full internal scan result before conversion to R-friendly JSON.
#[derive(Clone, Debug, Serialize)]
pub struct ScanResult {
    pub cp_dict: BTreeMap<usize, Vec<usize>>,
    pub window_results: BTreeMap<usize, WindowScanResult>,
    pub segments: BTreeMap<String, SegmentInfo>,
    pub out: AggregatedOut,
}
