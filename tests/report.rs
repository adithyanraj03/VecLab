//! Report rendering: structure, determinism, byte-identical re-render.

use veclab::bench::{run, BenchConfig, Dataset};
use veclab::report::render;
use veclab::VERSION;

fn small_suite() -> Vec<veclab::bench::BenchResult> {
    [Dataset::Gaussians, Dataset::Spherical, Dataset::Blocks]
        .iter()
        .map(|ds| {
            run(&BenchConfig {
                dataset: *ds,
                n: 300,
                dim: 16,
                k: 5,
                ef_sweep: vec![16, 32, 64],
                m: 8,
                m0: 16,
                ef_construction: 32,
                seed: 7,
                n_queries: 20,
            })
        })
        .collect()
}

#[test]
fn render_is_valid_html_document() {
    let html = render(&small_suite(), "VecLab — ANN Benchmark Report");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("<html lang=\"en\">"));
    assert!(html.contains("</html>"));
    assert!(html.contains("<title>VecLab — ANN Benchmark Report</title>"));
}

#[test]
fn render_contains_all_dataset_cards() {
    let html = render(&small_suite(), "VecLab — ANN Benchmark Report");
    for name in ["GAUSSIANS", "SPHERICAL", "BLOCKS"] {
        assert!(html.contains(name), "missing card {name}");
    }
}

#[test]
fn render_single_dataset_omits_others() {
    let one = small_suite();
    let only = vec![one[0].clone()];
    let html = render(&only, "t");
    assert!(html.contains("GAUSSIANS"));
    assert!(!html.contains("SPHERICAL"));
}

#[test]
fn render_has_charts_and_tables() {
    let html = render(&small_suite(), "t");
    let svgs = html.matches("<svg").count();
    let polylines = html.matches("<polyline").count();
    let tables = html.matches("<table>").count();
    assert!(svgs >= 4, "expected 3 line charts + 1 comparison, got {svgs}");
    assert!(polylines >= 6, "expected 6 polylines, got {polylines}");
    assert!(tables >= 4, "expected 4 tables, got {tables}");
}

#[test]
fn render_meta_line_lists_datasets() {
    let html = render(&small_suite(), "t");
    assert!(html.contains("3 DATASETS"));
    assert!(html.contains("100% DETERMINISTIC"));
}

#[test]
fn render_footer_has_version_and_copyright() {
    let html = render(&small_suite(), "t");
    assert!(html.contains(&format!("veclab v{VERSION}")));
    assert!(html.contains("© 2026 adithyanraj03"));
}

#[test]
fn render_is_byte_identical_across_calls() {
    let suite = small_suite();
    let a = render(&suite, "VecLab — ANN Benchmark Report");
    let b = render(&suite, "VecLab — ANN Benchmark Report");
    assert_eq!(a, b, "report is not byte-identical on re-render");
}

#[test]
fn render_contains_index_anatomy() {
    let html = render(&small_suite(), "t");
    assert!(html.contains("Index anatomy"));
    assert!(html.contains("edges per level"));
}

#[test]
fn render_contains_methodology() {
    let html = render(&small_suite(), "t");
    assert!(html.contains("Ground truth"));
    assert!(html.contains("Zero crates"));
    assert!(html.contains("SplitMix64"));
}

#[test]
fn render_reflects_the_numbers() {
    let suite = small_suite();
    let html = render(&suite, "t");
    // every reported recall value must appear verbatim in the tables
    for r in &suite {
        for p in &r.points {
            let s = veclab::bench::fmt_recall(p.recall);
            assert!(html.contains(&s), "missing {s} for ef {}", p.ef);
        }
    }
}
