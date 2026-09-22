//! VecLab CLI — hand-rolled argument parsing (no `clap` crate).
//!
//! ```text
//! veclab bench   [--dataset g|s|b|all] [--n 4000] [--dim 32] [--k 10]
//!                [--m 16] [--efc 64] [--ef-sweep 16,32,64,128,256]
//!                [--seed 7] [--queries 200]
//! veclab report  [--out report.html] [same options]
//! veclab demo    (tiny fast run)
//! ```

use std::process::ExitCode;

use veclab::bench::{self, BenchConfig, Dataset, fmt_recall};
use veclab::report;
use veclab::VERSION;

struct Flags {
    dataset: Vec<Dataset>,
    n: usize,
    dim: usize,
    k: usize,
    m: usize,
    m0: usize,
    efc: usize,
    ef_sweep: Vec<usize>,
    seed: u64,
    queries: usize,
    out: Option<String>,
}

impl Default for Flags {
    fn default() -> Self {
        Self {
            dataset: vec![
                Dataset::Gaussians,
                Dataset::Spherical,
                Dataset::Blocks,
            ],
            n: 4000,
            dim: 32,
            k: 10,
            m: 16,
            m0: 32,
            efc: 64,
            ef_sweep: vec![16, 32, 64, 128, 256],
            seed: 7,
            queries: 200,
            out: None,
        }
    }
}

fn parse_flags(args: &[String]) -> Result<Flags, String> {
    let mut f = Flags::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let mut next = || {
            i += 1;
            args.get(i).cloned().ok_or_else(|| format!("{} needs a value", a))
        };
        match a.as_str() {
            "--dataset" | "-d" => {
                let v = next()?;
                f.dataset = if v == "all" {
                    vec![
                        Dataset::Gaussians,
                        Dataset::Spherical,
                        Dataset::Blocks,
                    ]
                } else {
                    vec![Dataset::parse(&v).ok_or_else(|| format!("unknown dataset '{}'", v))?]
                };
            }
            "--n" | "-n" => f.n = num(&next()?, "n")?,
            "--dim" => f.dim = num(&next()?, "dim")?,
            "--k" => f.k = num(&next()?, "k")?,
            "--m" => f.m = num(&next()?, "m")?,
            "--efc" => f.efc = num(&next()?, "efc")?,
            "--ef-sweep" => {
                f.ef_sweep = next()?
                    .split(',')
                    .map(|s| num(s.trim(), "ef-sweep"))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--seed" => f.seed = num(&next()?, "seed")?,
            "--queries" => f.queries = num(&next()?, "queries")?,
            "--out" | "-o" => f.out = Some(next()?),
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag '{}'", other)),
        }
        // `next()` leaves `i` pointing at the value; skip past it.
        i += 1;
    }
    Ok(f)
}

fn num<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
    s.parse::<T>().map_err(|_| format!("{}: not a number: '{}'", what, s))
}

fn print_usage() {
    println!(
        "veclab v{} — approximate nearest neighbor search, built and benchmarked from scratch.

USAGE:
  veclab bench   [options]     run the seeded recall benchmark, print a table
  veclab report  [options]     run it and render the single-file HTML report
  veclab demo                  a tiny fast run
  veclab -h                    this help

OPTIONS:
  -d, --dataset <name>   gaussians | spherical | blocks | all   (default all)
  -n, --n <int>          corpus size                              (default 4000)
  --dim <int>            vector dimensionality                    (default 32)
  --k <int>              neighbors per query                      (default 10)
  --m <int>              HNSW M                                   (default 16)
  --efc <int>            efConstruction                           (default 64)
  --ef-sweep <csv>       ef values to sweep, csv                  (default 16,32,64,128,256)
  --seed <int>           RNG seed                                 (default 7)
  --queries <int>        query count per dataset                  (default 200)
  -o, --out <path>       report output path (report only)         (default report.html)
",
        VERSION
    );
}

fn demo_config() -> BenchConfig {
    BenchConfig {
        dataset: Dataset::Gaussians,
        n: 600,
        dim: 16,
        k: 5,
        ef_sweep: vec![8, 16, 32, 64],
        m: 8,
        m0: 16,
        ef_construction: 32,
        seed: 7,
        n_queries: 40,
    }
}

fn run_bench(args: &[String], render_report: bool) -> ExitCode {
    let mut f = if args.is_empty() && render_report {
        Flags::default()
    } else {
        match parse_flags(args) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("veclab: {}", e);
                print_usage();
                return ExitCode::from(2);
            }
        }
    };
    if render_report && f.out.is_none() {
        f.out = Some("report.html".to_string());
    }

    let mut results = Vec::new();
    for ds in &f.dataset {
        let cfg = BenchConfig {
            dataset: *ds,
            n: f.n,
            dim: f.dim,
            k: f.k,
            ef_sweep: f.ef_sweep.clone(),
            m: f.m,
            m0: f.m0,
            ef_construction: f.efc,
            seed: f.seed,
            n_queries: f.queries,
        };
        results.push(bench::run(&cfg));
    }

    // ---- terminal table ----
    println!("veclab v{} · seeded ANN benchmark · seed {}", VERSION, f.seed);
    println!("{}", "─".repeat(66));
    println!(
        "{:<11} {:>6} {:>4} {:>8} {:>3} {:>6}  {:>10}",
        "dataset", "n", "d", "metric", "k", "ef", "recall@k"
    );
    for r in &results {
        for p in &r.points {
            println!(
                "{:<11} {:>6} {:>4} {:>8} {:>3} {:>6}  {:>10}",
                r.config.dataset.name(),
                r.config.n,
                r.config.dim,
                r.config.dataset.metric().name(),
                r.config.k,
                p.ef,
                fmt_recall(p.recall)
            );
        }
    }
    println!("{}", "─".repeat(66));
    for r in &results {
        println!(
            "{}: best recall@{} = {} (ef={})",
            r.config.dataset.name(),
            r.config.k,
            fmt_recall(r.best_recall()),
            r.points
                .iter()
                .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap())
                .map(|p| p.ef)
                .unwrap_or(0)
        );
    }

    if render_report {
        let out = f.out.clone().unwrap();
        let html = report::render(&results, "VecLab — ANN Benchmark Report");
        if let Some(parent) = std::path::Path::new(&out).parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("veclab: cannot create {}: {}", parent.display(), e);
                    return ExitCode::from(1);
                }
            }
        }
        if let Err(e) = std::fs::write(&out, html) {
            eprintln!("veclab: cannot write {}: {}", out, e);
            return ExitCode::from(1);
        }
        println!(
            "report → {} ({} datasets · {} queries · byte-identical on re-run)",
            out,
            results.len(),
            f.queries
        );
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("bench") => run_bench(&args[1..], false),
        Some("report") => run_bench(&args[1..], true),
        Some("demo") => {
            let cfg = demo_config();
            let r = bench::run(&cfg);
            println!("veclab v{} · demo · seed {}", VERSION, cfg.seed);
            println!(
                "{:<11} {:>6} {:>4} {:>8} {:>3} {:>6}  {:>10}",
                "dataset", "n", "d", "metric", "k", "ef", "recall@k"
            );
            for p in &r.points {
                println!(
                    "{:<11} {:>6} {:>4} {:>8} {:>3} {:>6}  {:>10}",
                    cfg.dataset.name(),
                    cfg.n,
                    cfg.dim,
                    cfg.dataset.metric().name(),
                    cfg.k,
                    p.ef,
                    fmt_recall(p.recall)
                );
            }
            println!("demo: best recall@{} = {}", cfg.k, fmt_recall(r.best_recall()));
            ExitCode::SUCCESS
        }
        Some("-h") | Some("--help") | None => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("veclab: unknown command '{}'", other);
            print_usage();
            ExitCode::from(2)
        }
    }
}
