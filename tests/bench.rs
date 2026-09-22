//! Benchmark harness: dataset properties, config, determinism.

use veclab::bench::{
    fmt_recall, make_corpus, make_queries, standard_suite, run, BenchConfig, Dataset, EfPoint,
};

#[test]
fn corpus_deterministic_same_seed() {
    let a = make_corpus(Dataset::Gaussians, 500, 16, 7);
    let b = make_corpus(Dataset::Gaussians, 500, 16, 7);
    assert_eq!(a, b);
}

#[test]
fn corpus_differs_by_seed() {
    let a = make_corpus(Dataset::Gaussians, 500, 16, 7);
    let b = make_corpus(Dataset::Gaussians, 500, 16, 8);
    assert_ne!(a, b);
}

#[test]
fn corpus_shape_and_finite() {
    for kind in [Dataset::Gaussians, Dataset::Spherical, Dataset::Blocks] {
        let c = make_corpus(kind, 300, 32, 7);
        assert_eq!(c.len(), 300);
        for v in &c {
            assert_eq!(v.len(), 32);
            assert!(v.iter().all(|x| x.is_finite()), "non-finite coordinate");
        }
    }
}

#[test]
fn spherical_corpus_is_unit_norm() {
    let c = make_corpus(Dataset::Spherical, 500, 32, 7);
    for v in &c {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4, "norm {n} != 1");
    }
}

#[test]
fn blocks_corpus_even_coords_are_block_codes() {
    let c = make_corpus(Dataset::Blocks, 500, 32, 7);
    let allowed: Vec<f32> = (0..5).map(|b| b as f32 * 2.0).collect();
    for v in &c {
        for d in 0..32 {
            if d % 2 == 0 {
                assert!(allowed.contains(&v[d]), "even coord {} not a block code", v[d]);
            } else {
                assert!(v[d].abs() <= 8.0, "odd coord not small-noise-bounded");
            }
        }
    }
}

#[test]
fn gaussians_coordinates_bounded() {
    // lattice position in [-4, 3.5] + clamped gaussian in [-8, 8]
    let c = make_corpus(Dataset::Gaussians, 500, 32, 7);
    for v in &c {
        for x in v {
            assert!(x.abs() <= 12.0, "coordinate {x} out of bounds");
        }
    }
}

#[test]
fn queries_are_not_the_corpus() {
    let corpus = make_corpus(Dataset::Gaussians, 100, 16, 7);
    let queries = make_queries(Dataset::Gaussians, 100, 16, 7);
    assert_ne!(corpus, queries);
}

#[test]
fn dataset_parse_and_name_roundtrip() {
    for name in ["gaussians", "spherical", "blocks"] {
        let d = Dataset::parse(name).unwrap();
        assert_eq!(d.name(), name);
    }
    assert!(Dataset::parse("nope").is_none());
}

#[test]
fn dataset_metric_assignment() {
    assert_eq!(Dataset::Gaussians.metric(), veclab::vector::Metric::L2);
    assert_eq!(Dataset::Spherical.metric(), veclab::vector::Metric::Cosine);
    assert_eq!(Dataset::Blocks.metric(), veclab::vector::Metric::L2);
}

#[test]
fn config_defaults_match_documented() {
    let c = BenchConfig::default();
    assert_eq!(c.dataset, Dataset::Gaussians);
    assert_eq!(c.n, 4000);
    assert_eq!(c.dim, 32);
    assert_eq!(c.k, 10);
    assert_eq!(c.ef_sweep, vec![16, 32, 64, 128, 256]);
    assert_eq!(c.m, 16);
    assert_eq!(c.m0, 32);
    assert_eq!(c.ef_construction, 64);
    assert_eq!(c.seed, 7);
    assert_eq!(c.n_queries, 200);
}

#[test]
fn run_produces_one_point_per_ef() {
    let cfg = BenchConfig {
        dataset: Dataset::Gaussians,
        n: 300,
        dim: 16,
        k: 5,
        ef_sweep: vec![16, 32, 64],
        m: 8,
        m0: 16,
        ef_construction: 32,
        seed: 7,
        n_queries: 20,
    };
    let r = run(&cfg);
    assert_eq!(r.points.len(), 3);
    assert_eq!(
        r.points.iter().map(|p| p.ef).collect::<Vec<_>>(),
        vec![16, 32, 64]
    );
    for p in &r.points {
        assert!((0.0..=1.0).contains(&p.recall));
    }
}

#[test]
fn run_is_deterministic() {
    let cfg = BenchConfig {
        dataset: Dataset::Gaussians,
        n: 300,
        dim: 16,
        k: 5,
        ef_sweep: vec![16, 64],
        m: 8,
        m0: 16,
        ef_construction: 32,
        seed: 7,
        n_queries: 20,
    };
    let a = run(&cfg);
    let b = run(&cfg);
    assert_eq!(a.points, b.points);
    assert_eq!(a.max_level, b.max_level);
    assert_eq!(a.edges_per_level, b.edges_per_level);
}

#[test]
fn standard_suite_is_three_datasets_in_order_with_valid_recall() {
    let suite = standard_suite();
    assert_eq!(suite.len(), 3);
    let names: Vec<&str> = suite.iter().map(|r| r.config.dataset.name()).collect();
    assert_eq!(names, vec!["gaussians", "spherical", "blocks"]);
    for r in &suite {
        assert_eq!(r.points.len(), 5);
        for p in &r.points {
            assert!((0.0..=1.0).contains(&p.recall));
        }
        assert!(r.best_recall() > 0.0);
    }
}

#[test]
fn fmt_recall_fixed_four() {
    assert_eq!(fmt_recall(1.0), "1.0000");
    assert_eq!(fmt_recall(0.0), "0.0000");
    assert_eq!(fmt_recall(0.99996), "1.0000");
    assert_eq!(fmt_recall(0.9665), "0.9665");
}

#[test]
fn ef_point_and_result_accessors() {
    let cfg = BenchConfig {
        n: 100,
        dim: 8,
        k: 3,
        ef_sweep: vec![8, 16],
        m: 4,
        m0: 8,
        ef_construction: 8,
        n_queries: 5,
        ..BenchConfig::default()
    };
    let r = run(&cfg);
    assert_eq!(r.recall_at(8), Some(r.points[0].recall));
    assert_eq!(r.recall_at(999), None);
    let mut best = EfPoint {
        ef: 0,
        recall: 0.0,
    };
    for p in &r.points {
        if p.recall > best.recall {
            best = *p;
        }
    }
    assert_eq!(r.best_recall(), best.recall);
}
