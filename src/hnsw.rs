//! HNSW (Hierarchical Navigable Small World) — the approximate index.
//!
//! Implementation of the Malkov & Yashunin (2016) graph: a multi-layer
//! proximity structure where high layers are sparse expressways and layer 0
//! holds the full graph. Insertion descends greedily from the top and runs an
//! `efConstruction` beam search on every layer the node occupies; query does
//! the same descent then an `ef` beam search on layer 0.
//!
//! Determinism contract: no `HashMap`, no thread races, every tie broken by
//! ascending node index, all RNG drawn from one seeded SplitMix64 stream.
//! Identical corpus + parameters + seed ⇒ bit-identical graph and results.

use crate::rng::Rng;
use crate::vector::Metric;
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// A (distance, index) pair with total ordering by (distance, index).
#[derive(Debug, Clone, Copy)]
pub struct Scored {
    pub d: f32,
    pub i: usize,
}

impl PartialEq for Scored {
    fn eq(&self, o: &Scored) -> bool {
        self.d == o.d && self.i == o.i
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, o: &Scored) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Scored {
    fn cmp(&self, o: &Scored) -> Ordering {
        self.d
            .partial_cmp(&o.d)
            .unwrap_or(Ordering::Equal)
            .then(self.i.cmp(&o.i))
    }
}

/// Index parameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Max neighbors per node at layers 1+ (`M`).
    pub m: usize,
    /// Max neighbors per node at layer 0 (`M0 = 2M`).
    pub m0: usize,
    /// Beam width used during insertion.
    pub ef_construction: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Seed for the level-assignment stream.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 64,
            metric: Metric::L2,
            seed: 7,
        }
    }
}

/// Per-level + aggregate index statistics, for the report.
#[derive(Debug, Clone)]
pub struct HnswStats {
    pub nodes: usize,
    pub max_level: usize,
    /// Directed edges per level (each undirected edge counted twice).
    pub edges_per_level: Vec<usize>,
    /// How many nodes occupy at least that level (length = max_level + 1).
    pub nodes_per_level: Vec<usize>,
}

pub struct Hnsw {
    m: usize,
    m0: usize,
    ml: f64,
    ef_construction: usize,
    metric: Metric,
    vecs: Vec<Vec<f32>>,
    /// node -> level -> neighbor indices (layer 0 = full graph)
    levels: Vec<Vec<Vec<usize>>>,
    entry: Option<usize>,
    max_level: usize,
    rng: Rng,
}

impl Hnsw {
    /// Build an empty index. `params.m` must be >= 2.
    pub fn new(params: &HnswParams) -> Self {
        assert!(params.m >= 2, "HNSW requires M >= 2");
        assert!(params.ef_construction >= 1, "efConstruction must be >= 1");
        Self {
            m: params.m,
            m0: if params.m0 >= params.m { params.m0 } else { 2 * params.m },
            ml: 1.0 / (params.m as f64).ln(),
            ef_construction: params.ef_construction,
            metric: params.metric,
            vecs: Vec::new(),
            levels: Vec::new(),
            entry: None,
            max_level: 0,
            rng: Rng::new(params.seed),
        }
    }

    pub fn len(&self) -> usize {
        self.vecs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vecs.is_empty()
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Neighbor list of `node` at `level` (empty if the node doesn't occupy it).
    pub fn neighbors_at(&self, node: usize, level: usize) -> &[usize] {
        if level < self.levels[node].len() {
            &self.levels[node][level]
        } else {
            &[]
        }
    }

    /// Number of layers the node occupies.
    pub fn node_level(&self, node: usize) -> usize {
        self.levels[node].len()
    }

    fn dist(&self, a: usize, b: &[f32]) -> f32 {
        self.metric.distance(&self.vecs[a], b)
    }

    /// Draw the next node's level: `floor(-ln(u) * ml)`, u uniform in (0,1).
    fn next_level(&mut self) -> usize {
        let u = self.rng.next_f64();
        (-u.ln() * self.ml).floor() as usize
    }

    fn level_cap(&self, level: usize) -> usize {
        if level == 0 {
            self.m0
        } else {
            self.m
        }
    }

    /// Keep only the `cap` closest neighbors of `node` at `level` (ties → lower index).
    fn prune(&mut self, node: usize, level: usize, cap: usize) {
        let list = &mut self.levels[node][level];
        let mut scored: Vec<Scored> = list
            .iter()
            .map(|&n| Scored {
                d: self.metric.distance(&self.vecs[node], &self.vecs[n]),
                i: n,
            })
            .collect();
        scored.sort();
        scored.truncate(cap);
        *list = scored.into_iter().map(|s| s.i).collect();
    }

    /// Heuristic neighbor selection (Malkov & Yashunin, §2.4, "rule 1").
    ///
    /// `candidates` arrive sorted by distance to `node`. A candidate is kept
    /// only if it is closer to `node` than to EVERY already-accepted neighbor
    /// — i.e. it is not inside the shadow of one. This keeps neighborhoods
    /// diverse (bridges between clusters) instead of cliques of the closest
    /// points. Degrees intentionally vary: a node may end up with fewer than
    /// `cap` neighbors.
    fn select_neighbors(&self, node: usize, candidates: &[usize], cap: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::new();
        for &c in candidates {
            if selected.len() >= cap {
                break;
            }
            let dc_node = self.metric.distance(&self.vecs[node], &self.vecs[c]);
            let shadowed = selected
                .iter()
                .any(|&s| self.metric.distance(&self.vecs[c], &self.vecs[s]) < dc_node);
            if !shadowed {
                selected.push(c);
            }
        }
        selected
    }

    /// Beam search at one layer: up to `ef` closest nodes to `q`, starting from `ep`.
    fn search_layer(&self, ep: usize, level: usize, ef: usize, q: &[f32]) -> Vec<Scored> {
        let mut candidates: std::collections::BinaryHeap<std::cmp::Reverse<Scored>> =
            std::collections::BinaryHeap::new();
        let mut results: std::collections::BinaryHeap<Scored> = std::collections::BinaryHeap::new();
        let mut visited: BTreeSet<usize> = BTreeSet::new();
        let mut result_idx: BTreeSet<usize> = BTreeSet::new();

        let s_ep = Scored { d: self.dist(ep, q), i: ep };
        candidates.push(std::cmp::Reverse(s_ep));
        visited.insert(ep);
        results.push(s_ep);
        result_idx.insert(ep);

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            let worst = results.peek().map(|w| w.d).unwrap_or(f32::INFINITY);
            if results.len() >= ef && c.d > worst {
                break;
            }
            for &n in self.neighbors_at(c.i, level) {
                if result_idx.contains(&n) || visited.contains(&n) {
                    continue;
                }
                visited.insert(n);
                let d = self.dist(n, q);
                let worst = results.peek().map(|w| w.d).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(std::cmp::Reverse(Scored { d, i: n }));
                    if results.len() < ef {
                        results.push(Scored { d, i: n });
                        result_idx.insert(n);
                    } else if d < worst {
                        if let Some(w) = results.pop() {
                            result_idx.remove(&w.i);
                        }
                        results.push(Scored { d, i: n });
                        result_idx.insert(n);
                    }
                }
            }
        }

        let mut out: Vec<Scored> = results.into_iter().collect();
        out.sort();
        out
    }

    /// Insert a vector into the index.
    pub fn insert(&mut self, v: &[f32]) {
        let node = self.vecs.len();
        self.vecs.push(v.to_vec());
        let l = self.next_level();
        self.levels.push(vec![Vec::new(); l + 1]);

        match self.entry {
            None => {
                self.entry = Some(node);
                self.max_level = l;
                return;
            }
            Some(ep0) => {
                // Greedy descent from the top down to (not including) level l.
                let mut ep = ep0;
                let mut cur = self.max_level;
                while cur > l {
                    let mut found = false;
                    while !found {
                        let mut best: Option<usize> = None;
                        let mut best_d = self.dist(ep, v);
                        for &n in self.neighbors_at(ep, cur) {
                            let d = self.dist(n, v);
                            if d < best_d {
                                best_d = d;
                                best = Some(n);
                            }
                        }
                        match best {
                            Some(q) => ep = q,
                            None => found = true,
                        }
                    }
                    cur -= 1;
                }

                // Beam search each layer the node occupies, wiring edges up.
                // Levels above the current max_level have no other nodes, so
                // their lists stay empty (and the node becomes the new entry).
                let top = l.min(self.max_level);
                for level in (0..=top).rev() {
                    let found = self.search_layer(ep, level, self.ef_construction, v);
                    let cap = self.level_cap(level);
                    let cand: Vec<usize> = found.iter().map(|s| s.i).collect();
                    // Heuristic selection (paper §2.4).
                    let sel = self.select_neighbors(node, &cand, cap);
                    for &n in &sel {
                        self.levels[node][level].push(n);
                        self.levels[n][level].push(node);
                        if self.levels[n][level].len() > self.level_cap(level) {
                            self.prune(n, level, self.level_cap(level));
                        }
                    }
                    if let Some(&closest) = sel.first() {
                        ep = closest;
                    }
                }

                if l > self.max_level {
                    self.max_level = l;
                    self.entry = Some(node);
                }
            }
        }
    }

    /// Level-0 beam search started from a specific entry node.
    ///
    /// Useful for benchmarking entry-point quality: pass the brute-force
    /// nearest node and see how far the graph search gets on its own.
    pub fn search_from(&self, ep: usize, q: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.vecs.is_empty() || k == 0 {
            return Vec::new();
        }
        let ef = ef.max(k).max(1);
        self.search_layer(ep, 0, ef, q)
            .into_iter()
            .take(k)
            .map(|s| (s.i, s.d))
            .collect()
    }

    /// Approximate top-`k`: `ef` must be >= `k` for meaningful recall.
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.vecs.is_empty() || k == 0 {
            return Vec::new();
        }
        let ef = ef.max(k).max(1);
        let (mut ep, mut cur) = (self.entry.unwrap(), self.max_level);
        while cur > 0 {
            let mut found = false;
            while !found {
                let mut best: Option<usize> = None;
                let mut best_d = self.dist(ep, q);
                for &n in self.neighbors_at(ep, cur) {
                    let d = self.dist(n, q);
                    if d < best_d {
                        best_d = d;
                        best = Some(n);
                    }
                }
                match best {
                    Some(qn) => ep = qn,
                    None => found = true,
                }
            }
            cur -= 1;
        }
        self.search_layer(ep, 0, ef, q)
            .into_iter()
            .take(k)
            .map(|s| (s.i, s.d))
            .collect()
    }

    /// The stored corpus vectors, in insertion order.
    pub fn corpus(&self) -> &Vec<Vec<f32>> {
        &self.vecs
    }

    /// Aggregate index statistics.
    pub fn stats(&self) -> HnswStats {
        let mut edges_per_level = Vec::new();
        let mut nodes_per_level = Vec::new();
        for level in 0..=self.max_level {
            let mut edges = 0usize;
            let mut nodes = 0usize;
            for lv in &self.levels {
                if lv.len() > level {
                    nodes += 1;
                    edges += lv[level].len();
                }
            }
            edges_per_level.push(edges);
            nodes_per_level.push(nodes);
        }
        HnswStats {
            nodes: self.vecs.len(),
            max_level: self.max_level,
            edges_per_level,
            nodes_per_level,
        }
    }
}

/// Convenience: build an index over `corpus` with the given parameters.
pub fn build(corpus: &[Vec<f32>], params: &HnswParams) -> Hnsw {
    let mut idx = Hnsw::new(params);
    for v in corpus {
        idx.insert(v);
    }
    idx
}

/// The stored corpus (read-only).
pub fn corpus_of(idx: &Hnsw) -> &Vec<Vec<f32>> {
    &idx.vecs
}

/// Recall@k of `idx.search(q, k, ef)` against brute-force ground truth.
pub fn recall_at_k(idx: &Hnsw, q: &[f32], k: usize, ef: usize) -> f32 {
    let exact = crate::exact::knn(q, corpus_of(idx), k, idx.metric());
    let approx = idx.search(q, k, ef);
    let approx_idx: Vec<usize> = approx.iter().map(|(i, _)| *i).collect();
    crate::exact::recall_at_k(&exact, &approx_idx, k)
}
