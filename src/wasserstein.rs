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

    xs.sort_unstable_by(|a, b| a.total_cmp(b));
    ys.sort_unstable_by(|a, b| a.total_cmp(b));

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

    while i < xs.len() || j < ys.len() {
        // Next value from sample x, or infinity if x is exhausted.
        let next_x = if i < xs.len() { xs[i] } else { f64::INFINITY };

        // Next value from sample y, or infinity if y is exhausted.
        let next_y = if j < ys.len() { ys[j] } else { f64::INFINITY };

        // The next support point where at least one empirical CDF jumps.
        let z = next_x.min(next_y);

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

    dist
}
