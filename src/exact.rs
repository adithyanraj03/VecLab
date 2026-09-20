//! Brute-force k-nearest neighbours: the ground truth every approximation is
//! measured against.
//!
//! `knn` returns the `k` closest corpus points to `query`, as (index,
//! distance) pairs sorted by (distance, index) — the index tiebreak makes the
//! result fully deterministic even for exactly equidistant points.

use crate::vector::Metric;

/// The `k` nearest corpus points to `query`, as (index, distance) pairs.
///
/// O(n * d). Deterministic: ties are broken by ascending corpus index.
pub fn knn(query: &[f32], corpus: &[Vec<f32>], k: usize, metric: Metric) -> Vec<(usize, f32)> {
    if corpus.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, metric.distance(query, v)))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k.min(scored.len()));
    scored
}

/// Fraction of the exact top-`k` set that `approx` (a set of indices) covers.
///
/// This is the recall@k the benchmark harness reports.
pub fn recall_at_k(exact: &[(usize, f32)], approx: &[usize], k: usize) -> f32 {
    let k = k.min(exact.len());
    if k == 0 {
        return 1.0;
    }
    let truth: std::collections::BTreeSet<usize> = exact.iter().take(k).map(|(i, _)| *i).collect();
    let hit = approx.iter().take(k).filter(|i| truth.contains(i)).count();
    hit as f32 / k as f32
}
