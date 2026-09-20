//! Vector algebra over f32 slices: the whole geometry of the project.
//!
//! Plain f32 arithmetic, no SIMD intrinsics, no `num` crate. Deterministic by
//! construction: the same inputs and the same binary give the same outputs.

/// Euclidean squared distance between `a` and `b`.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// Euclidean distance between `a` and `b`.
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_sq(a, b).sqrt()
}

/// Squared L2 norm of `a`.
pub fn norm_sq(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// Cosine similarity of `a` and `b`, in [-1, 1]. Zero vectors yield 0.0.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm_sq(a).sqrt();
    let nb = norm_sq(b).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        / (na * nb)
}

/// Distance used by the metric; lower is better for both metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance.
    L2,
    /// 1 - cosine similarity (0..=2).
    Cosine,
}

impl Metric {
    /// Distance between `a` and `b` under this metric.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2(a, b),
            Metric::Cosine => 1.0 - cosine(a, b),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::Cosine => "cosine",
        }
    }
}

/// Parse a metric name ("l2" | "cosine").
pub fn parse_metric(s: &str) -> Option<Metric> {
    match s {
        "l2" | "euclidean" => Some(Metric::L2),
        "cosine" => Some(Metric::Cosine),
        _ => None,
    }
}
