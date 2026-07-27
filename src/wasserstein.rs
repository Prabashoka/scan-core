/// Sort ascending under `f64::total_cmp`.
///
/// Every sort in this crate goes through here so that all callers agree on the
/// ordering, including the placement of `-0.0` and NaN.
#[inline]
pub fn sort_f64(values: &mut [f64]) {
    values.sort_unstable_by(f64::total_cmp);
}

/// General 1D Wasserstein distance between two empirical samples.
///
/// This function works for both:
/// - equal sample sizes, e.g. x.len() == y.len()
/// - unequal sample sizes, e.g. x.len() != y.len()
///
/// Mathematically, in one dimension,
///
/// W_1(F_x, F_y) = ∫ |F_x(z) - F_y(z)| dz,
///
/// where F_x and F_y are the empirical CDFs of the two samples.
///
/// The algorithm:
/// 1. Sort both samples.
/// 2. Move through the combined sorted support points.
/// 3. Track the empirical CDF values of both samples.
/// 4. Accumulate the area between the two CDFs.
pub fn wasserstein_1d(x: &[f64], y: &[f64]) -> f64 {
    // Wasserstein distance is not meaningful if either sample is empty.
    if x.is_empty() || y.is_empty() {
        return f64::NAN;
    }

    // Make sorted copies of the input samples.
    //
    // We do not sort `x` and `y` directly because they are borrowed slices.
    // Sorting them directly would modify the original data, which we do not want.
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();

    sort_f64(&mut xs);
    sort_f64(&mut ys);

    wasserstein_1d_sorted(&xs, &ys)
}

/// Same as [`wasserstein_1d`], but reusing caller-owned scratch buffers.
///
/// Hot loops call this thousands of times in a row; routing them through here
/// keeps the two sorted copies on buffers that are allocated once instead of
/// once per call.
#[inline]
pub fn wasserstein_1d_with_scratch(
    x: &[f64],
    y: &[f64],
    xs: &mut Vec<f64>,
    ys: &mut Vec<f64>,
) -> f64 {
    if x.is_empty() || y.is_empty() {
        return f64::NAN;
    }

    xs.clear();
    xs.extend_from_slice(x);
    ys.clear();
    ys.extend_from_slice(y);

    sort_f64(xs);
    sort_f64(ys);

    wasserstein_1d_sorted(xs, ys)
}

/// Core CDF-merge accumulation for two samples that are **already sorted**
/// ascending under `f64::total_cmp`.
///
/// Callers that own their data (bootstrap replicates, incremental split scans)
/// sort in place and come straight here, skipping the two copies that
/// [`wasserstein_1d`] has to make.
pub fn wasserstein_1d_sorted(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return f64::NAN;
    }

    // Each observation contributes equal probability mass to its empirical CDF.
    let nx_inv = 1.0 / xs.len() as f64;
    let ny_inv = 1.0 / ys.len() as f64;

    // Pointers into the sorted samples.
    let mut i = 0usize;
    let mut j = 0usize;

    let mut cdf_x = 0.0f64;
    let mut cdf_y = 0.0f64;

    // Start from the smallest observed value across both samples.
    let mut prev = xs[0].min(ys[0]);

    // Accumulated Wasserstein distance.
    let mut dist = 0.0f64;

    // Phase 1: both samples still have mass left to place, so each step has to
    // decide which side jumps next.
    while i < xs.len() && j < ys.len() {
        // The next support point where at least one empirical CDF jumps.
        let z = xs[i].min(ys[j]);

        // Between `prev` and `z`, both empirical CDFs are constant.
        //
        // Therefore, the area between the two CDFs on this interval is:
        //
        // |F_x - F_y| * interval length
        //
        // This contributes to the Wasserstein distance.
        dist += (cdf_x - cdf_y).abs() * (z - prev);

        // Move through all x values equal to z.
        //
        // There may be ties, so we use a while loop instead of a single if.
        while i < xs.len() && xs[i] == z {
            cdf_x += nx_inv;
            i += 1;
        }

        // Move through all y values equal to z.
        while j < ys.len() && ys[j] == z {
            cdf_y += ny_inv;
            j += 1;
        }

        // Update the previous support point.
        prev = z;
    }

    // Phase 2: one sample is exhausted, so every remaining support point comes
    // from the other one and the branchy two-sided merge is no longer needed.
    // The other sample's CDF is frozen at its final accumulated value.
    while i < xs.len() {
        let z = xs[i];
        dist += (cdf_x - cdf_y).abs() * (z - prev);
        cdf_x += nx_inv;
        i += 1;
        while i < xs.len() && xs[i] == z {
            cdf_x += nx_inv;
            i += 1;
        }
        prev = z;
    }

    while j < ys.len() {
        let z = ys[j];
        dist += (cdf_x - cdf_y).abs() * (z - prev);
        cdf_y += ny_inv;
        j += 1;
        while j < ys.len() && ys[j] == z {
            cdf_y += ny_inv;
            j += 1;
        }
        prev = z;
    }

    dist
}
