//! VecLab — approximate nearest neighbor search, built and benchmarked from scratch.
//!
//! Zero crates: this links only the Rust standard library — no `rand`, no
//! `rayon`, not even a CLI parser. Every public result is deterministic for
//! identical inputs (seeded corpus, fixed parameters), and the HTML report is
//! byte-identical on re-render.


pub const VERSION: &str = env!("CARGO_PKG_VERSION");
