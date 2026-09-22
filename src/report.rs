//! Single-file Atelier-style HTML report.
//!
//! Pure string building — no templates, no timestamps, no randomness — so
//! identical inputs render byte-identical output, on every machine.

use crate::bench::{BenchResult, fmt_recall};
use crate::VERSION;

const INK: &str = "#252524";
const MUTED: &str = "#676662";
const HAIRLINE: &str = "#DCDAD1";
const GREEN: &str = "#22AC80";
const INDIGO: &str = "#5B51C7";
const ORANGE: &str = "#E8833A";

/// Render the report for a suite of benchmark results.
pub fn render(results: &[BenchResult], title: &str) -> String {
    let c = &results[0].config;
    let meta = format!(
        "{} DATASETS · D={} · N={} · K={} · {} QUERIES · 100% DETERMINISTIC",
        results.len(),
        c.dim,
        c.n,
        c.k,
        c.n_queries
    );

    let mut s = String::new();
    s.push_str(&format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ margin:0; background:#F7F6F1; color:{ink}; font-family:"Segoe UI",system-ui,sans-serif; }}
  .page {{ max-width:980px; margin:0 auto; padding:48px 40px 28px; }}
  h1 {{ font-family:Georgia,"Times New Roman",serif; font-size:34px; margin:0 0 6px; font-weight:700; }}
  .meta {{ font-size:11px; letter-spacing:0.14em; color:{muted}; text-transform:uppercase; margin-bottom:34px; }}
  h2 {{ font-size:12px; letter-spacing:0.14em; text-transform:uppercase; color:{muted}; margin:38px 0 12px; font-weight:600; }}
  .card {{ background:#fff; border:1px solid {hair}; border-radius:10px; padding:20px 22px; margin-bottom:18px; box-shadow:0 1px 2px rgba(37,37,36,0.04); }}
  .card h3 {{ font-family:Georgia,serif; font-size:19px; margin:0 0 2px; }}
  .card .sub {{ font-size:12px; color:{muted}; margin-bottom:14px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th {{ text-align:left; font-size:10.5px; letter-spacing:0.12em; text-transform:uppercase; color:{muted}; font-weight:600; padding:6px 10px; border-bottom:1.5px solid {hair}; }}
  td {{ padding:6px 10px; border-bottom:1px solid {hair}; font-family:Consolas,monospace; }}
  td.lbl {{ font-family:"Segoe UI",sans-serif; }}
  .hi {{ color:{green}; font-weight:600; }}
  .mid {{ color:{indigo}; font-weight:600; }}
  .lo {{ color:{orange}; font-weight:600; }}
  .legend {{ font-size:12px; color:{muted}; margin-top:8px; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
  ul {{ margin:8px 0 0; padding-left:20px; font-size:13.5px; line-height:1.65; color:#3c3c3a; }}
  .foot {{ margin-top:40px; padding-top:14px; border-top:1.5px solid {hair}; font-size:11.5px; color:{muted}; }}
</style>
</head>
<body>
<div class="page">
<h1>{title}</h1>
<div class="meta">{meta}</div>
"##,
        title = title,
        ink = INK,
        muted = MUTED,
        hair = HAIRLINE,
        green = GREEN,
        indigo = INDIGO,
        orange = ORANGE,
        meta = meta,
    ));

    // ---- 1 · per-dataset cards ----
    for (i, r) in results.iter().enumerate() {
        let cfg = &r.config;
        let color = cfg.dataset.color();
        let pts: Vec<(f32, f32)> = r.points.iter().map(|p| (p.ef as f32, p.recall)).collect();
        s.push_str(&format!(
            r##"<div class="card"><h3>{}. {} ({})</h3><div class="sub">recall@{} vs efSearch · M={} · M0={} · efC={} · seed {}</div>{chart}"##,
            i + 1,
            cfg.dataset.name().to_uppercase(),
            cfg.dataset.metric().name(),
            cfg.k,
            cfg.m,
            cfg.m0,
            cfg.ef_construction,
            cfg.seed,
            chart = line_chart(&pts, color, 920, 190),
        ));
        s.push_str("<table><tr><th>efSearch</th>");
        for p in &r.points {
            s.push_str(&format!("<th>@{}</th>", p.ef));
        }
        s.push_str("</tr><tr><td class=\"lbl\">recall</td>");
        for p in &r.points {
            let cls = if p.recall >= 0.99 {
                "hi"
            } else if p.recall >= 0.95 {
                "mid"
            } else {
                "lo"
            };
            s.push_str(&format!("<td class=\"{}\">{}</td>", cls, fmt_recall(p.recall)));
        }
        s.push_str("</tr></table></div>");
    }

    // ---- 2 · cross-dataset comparison ----
    let (w, h) = (920, 230);
    let all: Vec<f32> = results
        .iter()
        .flat_map(|r| r.points.iter().map(|p| p.recall))
        .collect();
    let y_min = all.iter().cloned().fold(f32::INFINITY, f32::min) - 0.01;
    let y_max = 1.001f32;
    let efs: Vec<f32> = results[0].points.iter().map(|p| p.ef as f32).collect();
    let x_of = |i: usize| 60.0 + (i as f32) * ((w as f32 - 100.0) / (efs.len() - 1).max(1) as f32);
    let y_of = |v: f32| {
        (h as f32 - 40.0) - ((v - y_min) / (y_max - y_min)) * (h as f32 - 80.0)
    };

    let mut cmp = String::new();
    cmp.push_str(&format!(
        r##"<div class="card"><h3>2 · Cross-dataset comparison</h3><div class="sub">recall@{} by efSearch — the same index parameters, three workloads</div>"##,
        results[0].config.k
    ));
    let mut grid = String::new();
    for g in [0.90f32, 0.95, 0.99, 1.00] {
        if g > y_min && g < y_max {
            let y = y_of(g);
            grid.push_str(&format!(
                r##"<line x1="60" y1="{y:.1}" x2="{x2}" y2="{y:.1}" stroke="#DCDAD1" stroke-dasharray="3,4"/><text x="52" y="{y:.1}" font-size="10.5" fill="#676662" text-anchor="end" font-family="Consolas,monospace">{g:.2}</text>"##,
                y = y,
                x2 = w - 40,
                g = g,
            ));
        }
    }
    let mut ticks = String::new();
    for (i, ef) in efs.iter().enumerate() {
        ticks.push_str(&format!(
            r##"<text x="{x:.1}" y="{y}" font-size="11" fill="#676662" text-anchor="middle" font-family="Consolas,monospace">{ef}</text>"##,
            x = x_of(i),
            y = h - 8,
            ef = *ef as i32
        ));
    }
    let mut polylines = String::new();
    for r in results {
        let color = r.config.dataset.color();
        let pts: String = r
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| format!("{:.1},{:.1}", x_of(i), y_of(p.recall)))
            .collect::<Vec<_>>()
            .join(" ");
        polylines.push_str(&format!(
            r##"<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>"##,
            pts = pts,
            color = color
        ));
    }
    cmp.push_str(&format!(
        r##"<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{grid}{ticks}{polylines}</svg>"##,
        w = w,
        h = h,
        grid = grid,
        ticks = ticks,
        polylines = polylines
    ));
    cmp.push_str("<div class=\"legend\">");
    for r in results {
        cmp.push_str(&format!(
            "<span><span class=\"dot\" style=\"background:{}\"></span>{}</span>&nbsp;&nbsp;",
            r.config.dataset.color(),
            r.config.dataset.name()
        ));
    }
    cmp.push_str("</div></div>");
    s.push_str(&cmp);

    // ---- 3 · index anatomy ----
    s.push_str(
        "<h2>3 · Index anatomy</h2><div class=\"card\"><table><tr><th>dataset</th><th>nodes</th><th>max level</th><th>edges per level (L0→top)</th><th>nodes per level</th></tr>",
    );
    for r in results {
        let edges = r
            .edges_per_level
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(" · ");
        let nodes = r
            .nodes_per_level
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(" · ");
        s.push_str(&format!(
            "<tr><td class=\"lbl\">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            r.config.dataset.name(),
            r.config.n,
            r.max_level,
            edges,
            nodes
        ));
    }
    s.push_str("</table></div>");

    // ---- 4 · methodology ----
    s.push_str(&format!(
        r##"<h2>4 · Methodology</h2><div class="card"><ul>
<li><b>Ground truth:</b> every query is answered by brute-force exact kNN over the full corpus; recall@{k} is set agreement between the exact and approximate top-{k} (by index).</li>
<li><b>Determinism:</b> corpus and queries come from a seeded SplitMix64 stream; HNSW layer assignments draw from the same stream; every tie is broken by ascending index. Two runs are bit-identical, and this report is byte-identical on re-render.</li>
<li><b>Zero crates:</b> std only — no rand, no rayon, no serde, no CLI parser, no template engine.</li>
<li><b>Metrics:</b> L2 for gaussians and blocks, cosine for spherical (unit vectors).</li>
</ul></div>"##,
        k = results[0].config.k,
    ));

    // ---- footer ----
    s.push_str(&format!(
        r##"<div class="foot">veclab v{v} · generated deterministically · MIT © 2026 adithyanraj03</div>
</div>
</body>
</html>
"##,
        v = VERSION
    ));
    s
}

/// One recall-vs-ef polyline chart for a single dataset.
fn line_chart(pts: &[(f32, f32)], color: &str, w: u32, h: u32) -> String {
    if pts.is_empty() {
        return String::new();
    }
    let y_min = pts.iter().map(|p| p.1).fold(f32::INFINITY, f32::min) - 0.005;
    let y_max = 1.001f32;
    let (x0, x1) = (60.0f32, w as f32 - 40.0);
    let (y0, y1) = (h as f32 - 36.0, 36.0);
    let x_of = |i: usize, n: usize| {
        if n <= 1 {
            x0
        } else {
            x0 + (i as f32) * ((x1 - x0) / (n - 1) as f32)
        }
    };
    let y_of = |v: f32| y0 - ((v - y_min) / (y_max - y_min)) * (y0 - y1);

    let mut grid = String::new();
    for g in [0.90f32, 0.95, 0.99, 1.00] {
        if g > y_min && g < y_max {
            let y = y_of(g);
            grid.push_str(&format!(
                r##"<line x1="{x0:.0}" y1="{y:.1}" x2="{x1:.0}" y2="{y:.1}" stroke="#DCDAD1" stroke-dasharray="3,4"/><text x="52" y="{y:.1}" font-size="10.5" fill="#676662" text-anchor="end" font-family="Consolas,monospace">{g:.2}</text>"##,
                x0 = x0,
                y = y,
                x1 = x1,
                g = g,
            ));
        }
    }
    let mut poly = String::new();
    let mut dots = String::new();
    for (i, (ef, rec)) in pts.iter().enumerate() {
        let x = x_of(i, pts.len());
        let y = y_of(*rec);
        if i > 0 {
            poly.push(' ');
        }
        poly.push_str(&format!("{:.1},{:.1}", x, y));
        dots.push_str(&format!(
            r##"<circle cx="{x:.1}" cy="{y:.1}" r="3" fill="{color}"/><text x="{x:.1}" y="{yy:.1}" font-size="10.5" fill="#676662" text-anchor="middle" font-family="Consolas,monospace">{ef}</text>"##,
            x = x,
            y = y,
            color = color,
            yy = y0 + 14.0,
            ef = *ef as i32,
        ));
    }
    format!(
        r##"<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{grid}<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>{dots}</svg>"##,
        w = w,
        h = h,
        grid = grid,
        poly = poly,
        color = color,
        dots = dots,
    )
}
