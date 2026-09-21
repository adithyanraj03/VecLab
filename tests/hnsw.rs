//! HNSW: structural invariants, determinism contract, recall thresholds.

use veclab::bench::{make_corpus, make_queries, Dataset};
use veclab::hnsw::{build, recall_at_k, corpus_of, HnswParams};
use veclab::vector::Metric;

fn params() -> HnswParams {
    HnswParams {
        m: 16,
        m0: 32,
        ef_construction: 64,
        metric: Metric::L2,
        seed: 7,
    }
}

fn gauss(n: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    (
        make_corpus(Dataset::Gaussians, n, dim, 7),
        make_queries(Dataset::Gaussians, 100, dim, 7),
    )
}

#[test]
fn empty_index() {
    let idx = build(&[], &params());
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
    assert_eq!(idx.stats().nodes, 0);
    assert!(idx.search(&[0.0, 0.0], 3, 16).is_empty());
}

#[test]
fn single_node_index() {
    let corpus = vec![vec![1.0, 2.0]];
    let idx = build(&corpus, &params());
    assert_eq!(idx.len(), 1);
    let r = idx.search(&[1.0, 2.0], 1, 8);
    assert_eq!(r, vec![(0, 0.0)]);
}

#[test]
fn corpus_preserved_in_insertion_order() {
    let (corpus, _) = gauss(200, 8);
    let idx = build(&corpus, &params());
    assert_eq!(corpus_of(&idx), &corpus);
}

#[test]
fn no_self_loops_at_any_level() {
    let (corpus, _) = gauss(400, 8);
    let idx = build(&corpus, &params());
    let max_l = idx.max_level();
    for i in 0..idx.len() {
        for level in 0..=max_l {
            assert!(
                !idx.neighbors_at(i, level).contains(&i),
                "self loop at node {i} level {level}"
            );
        }
    }
}

#[test]
fn neighbor_indices_in_range() {
    let (corpus, _) = gauss(400, 8);
    let idx = build(&corpus, &params());
    let n = idx.len();
    let max_l = idx.max_level();
    for i in 0..n {
        for level in 0..=max_l {
            for &nb in idx.neighbors_at(i, level) {
                assert!(nb < n, "neighbor {nb} out of range");
            }
        }
    }
}

#[test]
fn degree_capped_at_m0_and_m() {
    let (corpus, _) = gauss(800, 8);
    let idx = build(&corpus, &params());
    for i in 0..idx.len() {
        assert!(idx.neighbors_at(i, 0).len() <= 32, "L0 degree > M0");
        let l = idx.node_level(i);
        for level in 1..l {
            assert!(idx.neighbors_at(i, level).len() <= 16, "L{level} degree > M");
        }
    }
}

#[test]
fn neighbors_beyond_node_level_are_empty() {
    let (corpus, _) = gauss(300, 8);
    let idx = build(&corpus, &params());
    for i in 0..idx.len() {
        let l = idx.node_level(i);
        assert!(idx.neighbors_at(i, l).is_empty());
        assert!(idx.neighbors_at(i, l + 1).is_empty());
    }
}

#[test]
fn stats_consistency() {
    let (corpus, _) = gauss(1000, 16);
    let idx = build(&corpus, &params());
    let st = idx.stats();
    assert_eq!(st.nodes, 1000);
    assert_eq!(st.nodes_per_level.len(), st.max_level + 1);
    assert_eq!(st.edges_per_level.len(), st.max_level + 1);
    // every node occupies layer 0
    assert_eq!(st.nodes_per_level[0], 1000);
    // the counts of "occupies at least level L" are non-increasing
    for w in st.nodes_per_level.windows(2) {
        assert!(w[0] >= w[1], "not non-increasing: {:?}", st.nodes_per_level);
    }
}

#[test]
fn graph_is_deterministic_across_builds() {
    let (corpus, _) = gauss(500, 16);
    let a = build(&corpus, &params());
    let b = build(&corpus, &params());
    assert_eq!(a.max_level(), b.max_level());
    let max_l = a.max_level();
    for i in 0..a.len() {
        for level in 0..=max_l {
            assert_eq!(
                a.neighbors_at(i, level),
                b.neighbors_at(i, level),
                "graph differs at node {i} level {level}"
            );
        }
    }
}

#[test]
fn search_is_deterministic_across_builds() {
    let (corpus, queries) = gauss(500, 16);
    let a = build(&corpus, &params());
    let b = build(&corpus, &params());
    for q in &queries {
        assert_eq!(a.search(q, 10, 64), b.search(q, 10, 64));
    }
}

#[test]
fn different_seed_changes_graph() {
    let (corpus, _) = gauss(500, 16);
    let a = build(&corpus, &params());
    let p2 = HnswParams { seed: 8, ..params() };
    let b = build(&corpus, &p2);
    let differs = (0..a.len())
        .any(|i| a.neighbors_at(i, 0) != b.neighbors_at(i, 0));
    assert!(differs, "graph identical under a different seed");
}

#[test]
fn search_returns_k_sorted_by_distance() {
    let (corpus, queries) = gauss(300, 16);
    let idx = build(&corpus, &params());
    for q in queries.iter().take(20) {
        let r = idx.search(q, 10, 64);
        assert_eq!(r.len(), 10);
        for w in r.windows(2) {
            assert!(w[0].1 <= w[1].1, "distances not sorted");
        }
    }
}

#[test]
fn search_k_zero_empty() {
    let (corpus, queries) = gauss(100, 8);
    let idx = build(&corpus, &params());
    assert!(idx.search(&queries[0], 0, 16).is_empty());
}

#[test]
fn search_k_larger_than_corpus_clamps() {
    let corpus = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let idx = build(&corpus, &params());
    assert_eq!(idx.search(&[0.0, 0.0], 10, 16).len(), 2);
}

#[test]
fn insert_one_by_one_matches_batch_build() {
    let (corpus, _) = gauss(150, 8);
    let batch = build(&corpus, &params());
    let mut one_by_one = veclab::hnsw::Hnsw::new(&params());
    for v in &corpus {
        one_by_one.insert(v);
    }
    let max_l = batch.max_level();
    assert_eq!(one_by_one.max_level(), max_l);
    for i in 0..corpus.len() {
        for level in 0..=max_l {
            assert_eq!(one_by_one.neighbors_at(i, level), batch.neighbors_at(i, level));
        }
    }
}

#[test]
fn large_build_no_panic_and_findable() {
    // Regression: levels above the running max_level must not break insert.
    let (corpus, queries) = gauss(2000, 16);
    let idx = build(&corpus, &params());
    assert!(idx.max_level() >= 1, "expected a multi-level graph");
    let mut r1 = 0.0f32;
    for q in &queries {
        r1 += recall_at_k(&idx, q, 10, 64);
    }
    assert!(r1 / queries.len() as f32 > 0.9, "recall@1 too low: {r1}");
}

#[test]
fn recall_threshold_gaussians_ef64() {
    let (corpus, queries) = gauss(1000, 16);
    let idx = build(&corpus, &params());
    let mut tot = 0.0f32;
    for q in &queries {
        tot += recall_at_k(&idx, q, 10, 64);
    }
    let recall = tot / queries.len() as f32;
    assert!(recall >= 0.99, "recall@10 at ef=64 too low: {recall}");
}

#[test]
fn recall_threshold_cosine_spherical() {
    let corpus = make_corpus(Dataset::Spherical, 1000, 16, 7);
    let queries = make_queries(Dataset::Spherical, 100, 16, 7);
    let p = HnswParams {
        metric: Metric::Cosine,
        ..params()
    };
    let idx = build(&corpus, &p);
    let mut tot = 0.0f32;
    for q in &queries {
        tot += recall_at_k(&idx, q, 10, 64);
    }
    let recall = tot / queries.len() as f32;
    assert!(recall >= 0.99, "cosine recall@10 at ef=64 too low: {recall}");
}

#[test]
fn recall_monotone_in_ef() {
    let (corpus, queries) = gauss(500, 16);
    let idx = build(&corpus, &params());
    let mut prev = 0.0f32;
    for &ef in &[16usize, 32, 64, 128] {
        let mut tot = 0.0f32;
        for q in &queries {
            tot += recall_at_k(&idx, q, 10, ef);
        }
        let r = tot / queries.len() as f32;
        assert!(r + 1e-6 >= prev, "recall decreased from ef {ef:?}: {r} < {prev}");
        prev = r;
    }
}

#[test]
fn search_from_entry_helper() {
    let (corpus, queries) = gauss(300, 16);
    let idx = build(&corpus, &params());
    let exact = veclab::exact::knn(&queries[0], &corpus, 1, Metric::L2);
    let r = idx.search_from(exact[0].0, &queries[0], 10, 64);
    assert_eq!(r.len(), 10);
    assert_eq!(r[0].0, exact[0].0, "beam from the true NN should find it first");
}

#[test]
fn high_ef_recall_at_one_is_perfect_on_medium_corpus() {
    let (corpus, queries) = gauss(1000, 16);
    let idx = build(&corpus, &params());
    let mut hits = 0usize;
    for q in &queries {
        let exact = veclab::exact::knn(q, &corpus, 1, Metric::L2);
        let approx = idx.search(q, 1, 512);
        if approx.first().map(|p| p.0) == Some(exact[0].0) {
            hits += 1;
        }
    }
    assert!(hits as f32 / queries.len() as f32 >= 0.99, "recall@1: {hits}/{}", queries.len());
}
