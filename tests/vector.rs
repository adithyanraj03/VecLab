//! Vector algebra and metric properties.

use veclab::vector::{cosine, l2, l2_sq, norm_sq, parse_metric, Metric};

#[test]
fn l2_sq_of_zero() {
    let a = [1.0f32, 2.0, 3.0];
    assert_eq!(l2_sq(&a, &a), 0.0);
}

#[test]
fn l2_sq_known_value() {
    // (0-3)^2 + (4-0)^2 + (0-0)^2 = 9 + 16 = 25
    let a = [0.0f32, 4.0, 0.0];
    let b = [3.0f32, 0.0, 0.0];
    assert!((l2_sq(&a, &b) - 25.0).abs() < 1e-5);
}

#[test]
fn l2_is_sqrt_of_l2_sq() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 0.0, 0.0];
    assert!((l2(&a, &b) - l2_sq(&a, &b).sqrt()).abs() < 1e-6);
}

#[test]
fn norm_sq_known() {
    let a = [3.0f32, 4.0];
    assert!((norm_sq(&a) - 25.0).abs() < 1e-6);
}

#[test]
fn cosine_identical_unit_is_one() {
    let a = [1.0f32, 0.0, 0.0];
    assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_orthogonal_is_zero() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    assert!(cosine(&a, &b).abs() < 1e-6);
}

#[test]
fn cosine_opposite_is_minus_one() {
    let a = [1.0f32, 2.0];
    let b = [-1.0f32, -2.0];
    assert!((cosine(&a, &b) + 1.0).abs() < 1e-5);
}

#[test]
fn cosine_zero_vector_is_zero() {
    let a = [0.0f32, 0.0];
    let b = [1.0f32, 1.0];
    assert_eq!(cosine(&a, &b), 0.0);
    assert_eq!(cosine(&b, &a), 0.0);
}

#[test]
fn cosine_invariant_to_scale() {
    let a = [2.0f32, 3.0];
    let b = [20.0f32, 30.0]; // 10x a
    assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
}

#[test]
fn metric_l2_distance_is_euclidean() {
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    assert!((Metric::L2.distance(&a, &b) - 5.0).abs() < 1e-5);
}

#[test]
fn metric_cosine_distance_is_one_minus_sim() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    // orthogonal: similarity 0 -> distance 1
    assert!((Metric::Cosine.distance(&a, &b) - 1.0).abs() < 1e-6);
    // identical: similarity 1 -> distance 0
    assert!(Metric::Cosine.distance(&a, &a).abs() < 1e-6);
}

#[test]
fn parse_metric_aliases() {
    assert_eq!(parse_metric("l2"), Some(Metric::L2));
    assert_eq!(parse_metric("euclidean"), Some(Metric::L2));
    assert_eq!(parse_metric("cosine"), Some(Metric::Cosine));
    assert_eq!(parse_metric("minkowski"), None);
    assert_eq!(parse_metric(""), None);
}

#[test]
fn metric_names() {
    assert_eq!(Metric::L2.name(), "l2");
    assert_eq!(Metric::Cosine.name(), "cosine");
}

#[test]
fn distance_symmetric() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert!((Metric::L2.distance(&a, &b) - Metric::L2.distance(&b, &a)).abs() < 1e-6);
    assert!((Metric::Cosine.distance(&a, &b) - Metric::Cosine.distance(&b, &a)).abs() < 1e-6);
}

#[test]
fn triangle_inequality_l2() {
    let a = [0.0f32, 0.0];
    let b = [1.0f32, 2.0];
    let c = [3.0f32, -1.0];
    let lhs = l2(&a, &c);
    let rhs = l2(&a, &b) + l2(&b, &c);
    assert!(lhs <= rhs + 1e-6);
}
