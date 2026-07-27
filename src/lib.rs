//! Language-neutral change-point detection implementation shared by scanr and scan-py.

mod aggregate;
mod bootstrap;
mod detect;
mod refine;
mod stats;
mod types;
mod validation;
mod wasserstein;

pub use detect::{detect_for_window, run_scan_detector};
pub use refine::{refine_cp_cusum, refine_cp_wasserstein, refine_for_change_type};
pub use stats::PrefixStats;
pub use types::{
    AggregatedOut, ChangeType, ScanResult, ScanRustResult, SegmentInfo, WindowScanResult,
};
pub use validation::{validate_series, validate_window_sizes};
pub use wasserstein::wasserstein_1d;
