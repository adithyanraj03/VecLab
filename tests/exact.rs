//! Brute-force kNN: correctness, ordering, tie-breaking, recall.

use veclab::exact::{knn, recall_at_k};
use veclab::vector::Metric;

/// A 2-D corpus with a deliberate distance tie (indices 0 and 3) so the
/// ascending-index tiebreak is exercised.
fn corpus() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0],
        vec![3.0, 0.0],
        vec![0.0, 4.0],
        vec![1.0, 1.0],
        vec![10.0, 0.0],
    ]
}

#[test]
fn knn_returns_k_pairs() {
    let c = corpus();
    let r = knn(&[0.5, 0.5], &c, 3, Metric::L2);
    assert_eq!(r.len(), 3);
}

#[test]
fn knn_sorted_by_distance() {
    let c = corpus();
    let r = knn(&[0.5, 0.5], &c, 5, Metric::L2);
    for w in r.windows(2) {
        assert!(w[0].1 <= w[1].1, "not sorted: {r:?}");
    }
}

#[test]
fn knn_tie_broken_by_ascending_index() {
    let c = corpus();
    // indices 0 and 3 are both at distance sqrt(0.5) from the query.
    let r = knn(&[0.5, 0.5], &c, 2, Metric::L2);
    assert_eq!(r[0].0, 0);
    assert_eq!(r[1].0, 3);
    assert!((r[0].1 - r[1].1).abs() < 1e-6, "tie distances differ: {r:?}");
}

#[test]
fn knn_known_ordering() {
    let c = corpus();
    // Expected order: 0 (0.7071), 3 (0.7071), 1 (2.5495), 2 (3.5355), 4 (9.513)
    let r = knn(&[0.5, 0.5], &c, 5, Metric::L2);
    let order: Vec<usize> = r.iter().map(|p| p.0).collect();
    assert_eq!(order, vec![0, 3, 1, 2, 4]);
}

#[test]
fn knn_larger_than_corpus_returns_all() {
    let c = corpus();
    let r = knn(&[0.5, 0.5], &c, 100, Metric::L2);
    assert_eq!(r.len(), c.len());
}

#[test]
fn knn_k_zero_returns_empty() {
    let c = corpus();
    assert!(knn(&[0.5, 0.5], &c, 0, Metric::L2).is_empty());
}

#[test]
fn knn_empty_corpus_returns_empty() {
    let c: Vec<Vec<f32>> = Vec::new();
    assert!(knn(&[0.5, 0.5], &c, 3, Metric::L2).is_empty());
}

#[test]
fn knn_cosine_metric_ordering() {
    let c = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
    let r = knn(&[1.0, 0.0], &c, 3, Metric::Cosine);
    let order: Vec<usize> = r.iter().map(|p| p.0).collect();
    assert_eq!(order, vec![0, 1, 2]);
    assert!(r[0].1.abs() < 1e-6);
}

#[test]
fn knn_deterministic() {
    let c = corpus();
    let a = knn(&[0.5, 0.5], &c, 5, Metric::L2);
    let b = knn(&[0.5, 0.5], &c, 5, Metric::L2);
    assert_eq!(a, b);
}

#[test]
fn recall_perfect_is_one() {
    let exact = vec![(0, 0.1), (1, 0.2), (2, 0.3)];
    assert_eq!(recall_at_k(&exact, &[0, 1, 2], 3), 1.0);
}

#[test]
fn recall_disjoint_is_zero() {
    let exact = vec![(0, 0.1), (1, 0.2), (2, 0.3)];
    assert_eq!(recall_at_k(&exact, &[3, 4, 5], 3), 0.0);
}

#[test]
fn recall_partial_fraction() {
    let exact = vec![(0, 0.1), (1, 0.2), (2, 0.3)];
    // only index 0 overlaps
    assert!((recall_at_k(&exact, &[0, 3, 4], 3) - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn recall_k_zero_is_one() {
    let exact = vec![(0, 0.1)];
    assert_eq!(recall_at_k(&exact, &[0], 0), 1.0);
}

#[test]
fn recall_ignores_approx_beyond_k() {
    let exact = vec![(0, 0.1), (1, 0.2), (2, 0.3)];
    // approx lists extras but we only count the first k=3.
    assert!((recall_at_k(&exact, &[0, 1, 2, 9, 9], 3) - 1.0).abs() < 1e-6);
}
