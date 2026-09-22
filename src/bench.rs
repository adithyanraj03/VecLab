//! Seeded benchmark harness: corpus generation, ground-truth recall measurement.
//!
//! No wall-clock time anywhere in the results: recall is pure arithmetic over
//! a seeded corpus, so two runs with the same config produce identical numbers
//! and a byte-identical report.

use crate::hnsw::{build, recall_at_k, HnswParams};
use crate::rng::Rng;
use crate::vector::Metric;

/// Synthetic dataset families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// 12 Gaussian clusters on a lattice in d dims — the classic HNSW workload.
    Gaussians,
    /// Uniform points on a (d-1)-sphere — the cosine-metric workload.
    Spherical,
    /// 20 shared coordinate blocks with small noise — a duplicate-heavy workload.
    Blocks,
}

impl Dataset {
    pub fn name(&self) -> &'static str {
        match self {
            Dataset::Gaussians => "gaussians",
            Dataset::Spherical => "spherical",
            Dataset::Blocks => "blocks",
        }
    }

    /// The metric each family is meant to be searched under.
    pub fn metric(&self) -> Metric {
        match self {
            Dataset::Spherical => Metric::Cosine,
            _ => Metric::L2,
        }
    }

    pub fn parse(s: &str) -> Option<Dataset> {
        match s {
            "gaussians" => Some(Dataset::Gaussians),
            "spherical" => Some(Dataset::Spherical),
            "blocks" => Some(Dataset::Blocks),
            _ => None,
        }
    }

    /// Accent color used for this family in the report.
    pub fn color(&self) -> &'static str {
        match self {
            Dataset::Gaussians => "#22AC80",
            Dataset::Spherical => "#5B51C7",
            Dataset::Blocks => "#A74221",
        }
    }
}

/// The 12 cluster centers of the Gaussian family.
const CLUSTERS: usize = 12;

/// Generate a deterministic corpus of `n` vectors of dimension `dim`.
pub fn make_corpus(kind: Dataset, n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v = match kind {
            Dataset::Gaussians => {
                let center = rng.next_usize_below(CLUSTERS);
                let mut v = Vec::with_capacity(dim);
                for d in 0..dim {
                    // A fixed lattice position per (cluster, coord), noise on top.
                    let c = (((center * 31 + d * 7) % 17) as f32) * 0.5 - 4.0;
                    v.push(c + rng.next_gaussian_f32());
                }
                v
            }
            Dataset::Spherical => {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.next_gaussian_f32()).collect();
                let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if nrm > 0.0 {
                    for x in v.iter_mut() {
                        *x /= nrm;
                    }
                }
                v
            }
            Dataset::Blocks => {
                // Half the coordinates are block codes (shared by many points),
                // half are near-zero noise: a heavy-tail, duplicate-rich case.
                let mut v = vec![0.0f32; dim];
                let block = rng.next_usize_below(20);
                for d in 0..dim {
                    if d % 2 == 0 {
                        v[d] = (block % 5) as f32 * 2.0;
                    } else {
                        v[d] = rng.next_gaussian_f32() * 0.1;
                    }
                }
                v
            }
        };
        out.push(v);
    }
    out
}

/// In-distribution queries (standard ANN practice: queries from the corpus law).
pub fn make_queries(kind: Dataset, n_queries: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    make_corpus(kind, n_queries, dim, seed.wrapping_add(0x9E37))
}

/// One benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub dataset: Dataset,
    pub n: usize,
    pub dim: usize,
    pub k: usize,
    pub ef_sweep: Vec<usize>,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub seed: u64,
    pub n_queries: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            dataset: Dataset::Gaussians,
            n: 4000,
            dim: 32,
            k: 10,
            ef_sweep: vec![16, 32, 64, 128, 256],
            m: 16,
            m0: 32,
            ef_construction: 64,
            seed: 7,
            n_queries: 200,
        }
    }
}

/// One (ef, mean recall@k) point of a sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfPoint {
    pub ef: usize,
    pub recall: f32,
}

/// The full outcome of one seeded benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub config: BenchConfig,
    pub points: Vec<EfPoint>,
    pub max_level: usize,
    pub edges_per_level: Vec<usize>,
    pub nodes_per_level: Vec<usize>,
}

impl BenchResult {
    pub fn recall_at(&self, ef: usize) -> Option<f32> {
        self.points.iter().find(|p| p.ef == ef).map(|p| p.recall)
    }

    pub fn best_recall(&self) -> f32 {
        self.points.iter().map(|p| p.recall).fold(0.0f32, f32::max)
    }
}

/// Run one seeded benchmark: build the index once, sweep ef, measure recall.
pub fn run(config: &BenchConfig) -> BenchResult {
    let corpus = make_corpus(config.dataset, config.n, config.dim, config.seed);
    let queries = make_queries(config.dataset, config.n_queries, config.dim, config.seed);
    let params = HnswParams {
        m: config.m,
        m0: config.m0,
        ef_construction: config.ef_construction,
        metric: config.dataset.metric(),
        seed: config.seed,
    };
    let idx = build(&corpus, &params);
    let stats = idx.stats();
    let mut points = Vec::new();
    for &ef in &config.ef_sweep {
        let mut total = 0.0f32;
        for q in &queries {
            total += recall_at_k(&idx, q, config.k, ef);
        }
        points.push(EfPoint {
            ef,
            recall: total / queries.len() as f32,
        });
    }
    BenchResult {
        config: config.clone(),
        points,
        max_level: stats.max_level,
        edges_per_level: stats.edges_per_level,
        nodes_per_level: stats.nodes_per_level,
    }
}

/// The standard three-dataset sweep the report renders.
pub fn standard_suite() -> Vec<BenchResult> {
    [Dataset::Gaussians, Dataset::Spherical, Dataset::Blocks]
        .iter()
        .map(|ds| run(&BenchConfig { dataset: *ds, ..BenchConfig::default() }))
        .collect()
}

/// Fixed-point formatting shared by terminal and report.
pub fn fmt_recall(r: f32) -> String {
    format!("{:.4}", r)
}
