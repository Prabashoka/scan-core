use crate::types::ScanRustResult;
use crate::types::{AggregatedOut, SegmentInfo};
use std::collections::{BTreeMap, HashSet};

/// Count how many window sizes produced each candidate change-point.
pub fn compute_cp_counts(
    change_points_dict: &BTreeMap<usize, Vec<usize>>,
) -> BTreeMap<usize, usize> {
    let mut cp_to_count: BTreeMap<usize, usize> = BTreeMap::new();

    for cps in change_points_dict.values() {
        // A single window size should contribute at most one vote to the same index.
        let unique: HashSet<usize> = cps.iter().copied().collect();
        for cp in unique {
            *cp_to_count.entry(cp).or_insert(0) += 1;
        }
    }

    cp_to_count
}

/// Merge nearby candidate change-points into segments and keep vote information.
pub fn compute_change_points_with_votes(
    change_points_dict: &BTreeMap<usize, Vec<usize>>,
    tol: usize,
) -> BTreeMap<String, SegmentInfo> {
    let cp_to_count = compute_cp_counts(change_points_dict);
    let all_cps: Vec<usize> = cp_to_count.keys().copied().collect();

    if all_cps.is_empty() {
        return BTreeMap::new();
    }

    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut cur = vec![all_cps[0]];

    for &cp in all_cps.iter().skip(1) {
        let last_cp = match cur.last() {
            Some(last_cp) => *last_cp,
            None => {
                cur.push(cp);
                continue;
            }
        };

        if cp - last_cp <= tol {
            cur.push(cp);
        } else {
            segments.push(cur);
            cur = vec![cp];
        }
    }

    segments.push(cur);

    let mut out = BTreeMap::new();

    for (i, seg) in segments.into_iter().enumerate() {
        let mut votes = BTreeMap::new();
        let mut segment_vote = 0usize;

        for cp in &seg {
            let v = *cp_to_count.get(cp).unwrap_or(&0);
            votes.insert(*cp, v);
            segment_vote += v;
        }

        out.insert(
            format!("segment_{}", i + 1),
            SegmentInfo {
                change_points: seg,
                votes,
                segment_vote,
            },
        );
    }

    out
}

/// Pick the change-point with the  highest votes within each merged segment.
pub fn leaders_from_segments(segments: &BTreeMap<String, SegmentInfo>) -> BTreeMap<usize, usize> {
    let mut leaders = BTreeMap::new();

    for info in segments.values() {
        let mut best_cp = None::<usize>;
        let mut best_vote = 0usize;

        for (&cp, &vote) in &info.votes {
            let is_better = match best_cp {
                Some(current_best_cp) => {
                    vote > best_vote || (vote == best_vote && cp < current_best_cp)
                }
                None => true,
            };

            if is_better {
                best_cp = Some(cp);
                best_vote = vote;
            }
        }

        if let Some(cp) = best_cp {
            // Store the segment-level total vote at the selected representative point.
            leaders.insert(cp, info.segment_vote);
        }
    }

    leaders
}

/// Convert segment votes into normalized scores, probabilities, and a CDF.
pub fn cdf_from_segment_votes(
    segments: &BTreeMap<String, SegmentInfo>,
    num_windows: usize,
) -> ScanRustResult<AggregatedOut> {
    if num_windows == 0 {
        return Err("number of windows must be positive".to_string());
    }

    let leaders_segment_votes = leaders_from_segments(segments);
    let num_windows_f64 = num_windows as f64;

    let mut leaders_scores = BTreeMap::new();
    for (&cp, &v) in &leaders_segment_votes {
        leaders_scores.insert(cp, (v as f64 / num_windows_f64).min(1.0));
    }

    let total = leaders_scores.values().sum::<f64>();

    let mut leaders_probs = BTreeMap::new();
    if total > 0.0 {
        let total_inv = total.recip();
        for (&cp, &score) in &leaders_scores {
            leaders_probs.insert(cp, score * total_inv);
        }
    } else if !leaders_scores.is_empty() {
        let k_inv = 1.0 / leaders_scores.len() as f64;
        for &cp in leaders_scores.keys() {
            leaders_probs.insert(cp, k_inv);
        }
    }

    let mut cdf = Vec::with_capacity(leaders_probs.len());
    let mut cum = 0.0f64;
    for (&cp, &prob) in &leaders_probs {
        cum += prob;
        cdf.push((cp, cum));
    }

    Ok(AggregatedOut {
        leaders_segment_votes,
        leaders_scores,
        leaders_probs,
        cdf,
    })
}
