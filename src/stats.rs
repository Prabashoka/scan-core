/// sum[start..end] = prefix_sum[end] - prefix_sum[start]
#[derive(Clone, Debug)]
pub struct PrefixStats {
    sum: Vec<f64>,
    sumsq: Vec<f64>,
}

impl PrefixStats {
    /// Build prefix sums and prefix squared sums from the input series.
    ///
    /// Input:
    /// - `series`: the full time series
    ///
    /// Output:
    /// - a `PrefixStats` object containing cumulative sums and cumulative
    ///   squared sums.
    pub fn from_series(series: &[f64]) -> Self {
        let n = series.len();
        let mut sum = Vec::with_capacity(n + 1);
        let mut sumsq = Vec::with_capacity(n + 1);

        sum.push(0.0);
        sumsq.push(0.0);

        let mut s = 0.0f64;
        let mut ss = 0.0f64;
        // Loop through the time series once.
        // At each step, update both cumulative sums.
        for &v in series {
            s += v;
            ss += v * v;

            // Store cumulative sum up to the current observation.
            sum.push(s);

            // Store cumulative squared sum up to the current observation.
            sumsq.push(ss);
        }
        Self { sum, sumsq }
    }

    pub fn mean_std(&self, start: usize, len: usize, eps: f64) -> (f64, f64) {
        if len == 0 {
            return (0.0, eps);
        }

        let end = start + len;
        let n = len as f64;
        let s = self.sum[end] - self.sum[start];
        let ss = self.sumsq[end] - self.sumsq[start];

        let mu = s / n;

        // If there is only one observation, sample standard deviation
        // is not well-defined because it would divide by n - 1 = 0.
        //
        // So we return eps as a safe standard deviation.
        if len <= 1 {
            return (mu, eps);
        }

        let centered_ss = (ss - (s * s) / n).max(0.0);
        let std = (centered_ss / (n - 1.0)).sqrt().max(eps);

        (mu, std)
    }
}

/// Mean helper used by the CUSUM refinement. This is a simple function that computes the average of a slice.
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}
