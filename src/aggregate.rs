use crate::types::ScanRustResult;
use crate::types::{AggregatedOut, SegmentInfo};
use std::collections::{BTreeMap, HashSet};

/// Count how many window sizes produced each candidate change-point.
pub fn compute_cp_counts(
    change_points_dict: &BTreeMap<usize, Vec<usize>>,
) -> BTreeMap<usize, usize> {
    let mut cp_to_count: BTreeMap<usize, usize> = BTreeMap::new();
    // One set, cleared per window size, instead of one allocation per window.
    let mut unique: HashSet<usize> = HashSet::new();

    for cps in change_points_dict.values() {
        // A single window size should contribute at most one vote to the same index.
        unique.clear();
        for &cp in cps {
            if unique.insert(cp) {
                *cp_to_count.entry(cp).or_insert(0) += 1;
            }
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

    let mut iter = cp_to_count.iter();
    let Some((&first_cp, &first_count)) = iter.next() else {
        return BTreeMap::new();
    };

    // Walk the ascending candidates once, cutting a new segment whenever the
    // gap to the previous candidate exceeds `tol`. Votes are accumulated in the
    // same pass, so `cp_to_count` is never looked up again.
    let mut out = BTreeMap::new();
    let mut segment_index = 1usize;
    let mut change_points = vec![first_cp];
    let mut votes = BTreeMap::from([(first_cp, first_count)]);
    let mut segment_vote = first_count;
    let mut last_cp = first_cp;

    for (&cp, &count) in iter {
        if cp - last_cp <= tol {
            change_points.push(cp);
            votes.insert(cp, count);
            segment_vote += count;
        } else {
            out.insert(
                format!("segment_{segment_index}"),
                SegmentInfo {
                    change_points: std::mem::replace(&mut change_points, vec![cp]),
                    votes: std::mem::replace(&mut votes, BTreeMap::from([(cp, count)])),
                    segment_vote,
                },
            );
            segment_index += 1;
            segment_vote = count;
        }
        last_cp = cp;
    }

    out.insert(
        format!("segment_{segment_index}"),
        SegmentInfo {
            change_points,
            votes,
            segment_vote,
        },
    );

    out
}

/// Pick the change-point with the  highest votes within each merged segment.
pub fn leaders_from_segments(segments: &BTreeMap<String, SegmentInfo>) -> BTreeMap<usize, usize> {
    let mut leaders = BTreeMap::new();

    for info in segments.values() {
        // `votes` is a BTreeMap, so candidates arrive in ascending order: the
        // first strict maximum is also the lowest-index maximum.
        let best = info
            .votes
            .iter()
            .fold(None::<(usize, usize)>, |best, (&cp, &vote)| match best {
                Some((_, best_vote)) if vote <= best_vote => best,
                _ => Some((cp, vote)),
            });

        if let Some((cp, _)) = best {
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
