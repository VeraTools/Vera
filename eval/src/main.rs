//! Vera Evaluation Harness
//!
//! Single-command benchmark runner that produces structured JSON results
//! alongside human-readable summaries.
//!
//! Usage:
//!   vera-eval run [--tasks-dir <path>] [--output <path>] [--tool <name>]
//!   vera-eval verify-corpus [--corpus <path>]

mod lanes;
mod loader;
mod metrics;
mod output;
mod runner;
mod types;
mod vera_adapter;

use anyhow::{Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
#[command(name = "vera-eval", about = "Vera evaluation harness")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the benchmark suite and produce evaluation report.
    Run {
        /// Path to the tasks directory (default: eval/tasks/).
        #[arg(long, default_value = "eval/tasks")]
        tasks_dir: PathBuf,

        /// Path to the corpus manifest (default: eval/corpus.toml).
        #[arg(long, default_value = "eval/corpus.toml")]
        corpus: PathBuf,

        /// Output file path for JSON report (default: stdout).
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Tool adapter to use. `vera-bm25` is the real regression lane; mock tools are for harness self-tests.
        #[arg(long, default_value = "vera-bm25")]
        tool: String,

        /// JSON or TOML file containing one or more model lanes.
        #[arg(long = "lane-spec", alias = "lanes")]
        lane_spec: Option<PathBuf>,

        /// Run only these task IDs. Repeat the flag or separate IDs with commas.
        #[arg(long = "task-id", value_delimiter = ',', action = ArgAction::Append)]
        task_ids: Vec<String>,

        /// Run only these task categories. Repeat the flag or separate values with commas.
        #[arg(long = "category", value_delimiter = ',', action = ArgAction::Append)]
        categories: Vec<String>,

        /// Suppress human-readable summary (JSON only).
        #[arg(long)]
        json_only: bool,
    },
    /// Verify that corpus repos are cloned at correct SHAs.
    VerifyCorpus {
        /// Path to the corpus manifest.
        #[arg(long, default_value = "eval/corpus.toml")]
        corpus: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            tasks_dir,
            corpus,
            output,
            tool,
            lane_spec,
            task_ids,
            categories,
            json_only,
        } => cmd_run(
            &tasks_dir,
            &corpus,
            output.as_deref(),
            &tool,
            lane_spec.as_deref(),
            &task_ids,
            &categories,
            json_only,
        ),
        Commands::VerifyCorpus { corpus } => cmd_verify_corpus(&corpus),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    tasks_dir: &Path,
    corpus_path: &Path,
    output_path: Option<&Path>,
    tool_name: &str,
    lane_spec_path: Option<&Path>,
    task_ids: &[String],
    categories: &[String],
    json_only: bool,
) -> Result<()> {
    // Load tasks
    let tasks = loader::load_tasks(tasks_dir)
        .with_context(|| format!("Failed to load tasks from {}", tasks_dir.display()))?;

    if tasks.is_empty() {
        anyhow::bail!("No benchmark tasks found in {}", tasks_dir.display());
    }

    let tasks = filter_tasks(tasks, task_ids, categories)?;

    eprintln!("Loaded {} benchmark tasks", tasks.len());

    let reports = if let Some(path) = lane_spec_path {
        if tool_name != "vera-bm25" {
            anyhow::bail!("--lane-spec cannot be combined with --tool {tool_name}");
        }
        let specs = lanes::load_file(path)?;
        let resolved = lanes::resolve_specs(specs)?;
        resolved
            .iter()
            .map(|lane| run_lane(tasks.clone(), corpus_path, lane))
            .collect::<Result<Vec<_>>>()?
    } else if let Some(spec) = lanes::preset(tool_name) {
        let lane = lanes::resolve(spec)?;
        vec![run_lane(tasks, corpus_path, &lane)?]
    } else if matches!(tool_name, "mock-perfect" | "mock-partial") {
        vec![run_mock(tasks, tool_name)?]
    } else {
        anyhow::bail!(
            "Unknown tool '{}'. Available: vera-bm25, vera-cuda, vera-cpu, vera-potion, mock-perfect, mock-partial; or pass --lane-spec.",
            tool_name
        );
    };

    // Output JSON
    if let Some(path) = output_path {
        output::write_json_reports(&reports, path)?;
        eprintln!("JSON report written to {}", path.display());
    } else if json_only {
        let json = output::reports_to_json(&reports)?;
        println!("{json}");
    }

    // Print human-readable summary
    if !json_only {
        for (index, report) in reports.iter().enumerate() {
            if index > 0 {
                eprintln!("\n{}\n", "=".repeat(80));
            }
            output::print_summary(report, &mut std::io::stderr())?;
        }
    }

    Ok(())
}

fn cmd_verify_corpus(corpus_path: &Path) -> Result<()> {
    let manifest = loader::load_corpus(corpus_path)?;
    let repo_root = std::env::current_dir()?;
    let issues = loader::verify_corpus(&manifest, &repo_root)?;

    if issues.is_empty() {
        println!(
            "✓ All {} repos verified at correct SHAs",
            manifest.repos.len()
        );
        for repo in &manifest.repos {
            println!(
                "  {} ({}) → {}",
                repo.name,
                repo.language,
                &repo.commit[..12]
            );
        }
        Ok(())
    } else {
        eprintln!("✗ Corpus verification failed:");
        for issue in &issues {
            eprintln!("  - {issue}");
        }
        eprintln!("\nRun eval/setup-corpus.sh to fix.");
        std::process::exit(1);
    }
}

/// Repo paths, SHAs, and benchmark_root scopes from the corpus manifest.
struct VerifiedCorpus {
    repo_paths: HashMap<String, String>,
    repo_shas: HashMap<String, String>,
    benchmark_roots: HashMap<String, String>,
}

fn load_verified_corpus(corpus_path: &Path) -> Result<VerifiedCorpus> {
    if !corpus_path.exists() {
        anyhow::bail!("Corpus manifest not found at {}", corpus_path.display());
    }

    let manifest = loader::load_corpus(corpus_path)?;
    let repo_root = std::env::current_dir()?;
    let issues = loader::verify_corpus(&manifest, &repo_root)?;
    if !issues.is_empty() {
        anyhow::bail!(
            "Corpus verification failed:\n{}",
            issues
                .into_iter()
                .map(|issue| format!("  - {issue}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let repo_paths = vera_adapter::repo_paths_from_manifest(&repo_root, &manifest);
    let benchmark_roots = vera_adapter::benchmark_roots_from_manifest(&manifest);
    let repo_shas = manifest
        .repos
        .iter()
        .map(|repo| (repo.name.clone(), repo.commit.clone()))
        .collect();

    Ok(VerifiedCorpus {
        repo_paths,
        repo_shas,
        benchmark_roots,
    })
}

/// Filter tasks to only those whose repos are in the corpus manifest.
///
/// When using a subset corpus, tasks for missing repos are dropped so the eval
/// harness can run against any corpus subset. An empty filtered suite is an
/// error because it cannot produce meaningful metrics.
fn filter_tasks_to_corpus(
    tasks: Vec<types::BenchmarkTask>,
    repo_paths: &HashMap<String, String>,
) -> Result<Vec<types::BenchmarkTask>> {
    let before = tasks.len();
    let filtered: Vec<_> = tasks
        .into_iter()
        .filter(|task| repo_paths.contains_key(&task.repo))
        .collect();
    let skipped = before - filtered.len();
    if skipped > 0 {
        eprintln!(
            "Skipped {skipped} tasks referencing repos not in corpus ({} tasks remaining)",
            filtered.len()
        );
    }
    if filtered.is_empty() {
        anyhow::bail!("No benchmark tasks remain after filtering to the corpus");
    }
    Ok(filtered)
}

fn run_lane(
    tasks: Vec<types::BenchmarkTask>,
    corpus_path: &Path,
    lane: &lanes::ResolvedLane,
) -> Result<types::EvalReport> {
    let _env = lanes::apply_environment(lane);
    let corpus = load_verified_corpus(corpus_path)?;
    let tasks = filter_tasks_to_corpus(tasks, &corpus.repo_paths)?;
    let (mut report, evaluated_tasks) = if lane.is_bm25() {
        let vera = vera_adapter::VeraBm25Adapter::new()?;
        let report = runner::run_benchmark_scoped(
            &vera,
            &tasks,
            &corpus.repo_paths,
            &corpus.repo_shas,
            &corpus.benchmark_roots,
        );
        (report, tasks)
    } else {
        let backend = lane.backend.expect("non-BM25 lane must have a backend");
        let vera =
            vera_adapter::VeraFullAdapter::new_with_options(backend, lane.rerank(), lane.name())?;
        let report = runner::run_benchmark_scoped(
            &vera,
            &tasks,
            &corpus.repo_paths,
            &corpus.repo_shas,
            &corpus.benchmark_roots,
        );
        (report, tasks)
    };

    let provenance = lane.provenance()?;
    runner::attach_provenance(
        &mut report,
        Some(provenance.clone()),
        lanes::task_set_identity(&evaluated_tasks),
        lane.config_map(&provenance),
        lanes::environment_summary(lane),
        vera_git_sha(),
        std::env::args().collect(),
    );
    Ok(report)
}

fn run_mock(tasks: Vec<types::BenchmarkTask>, tool_name: &str) -> Result<types::EvalReport> {
    let mut report = match tool_name {
        "mock-perfect" => runner::run_benchmark_with_mock(&runner::MockAdapter::perfect(), &tasks),
        "mock-partial" => {
            runner::run_benchmark_with_mock(&runner::MockAdapter::partial(0.7), &tasks)
        }
        _ => unreachable!("mock tool validated by caller"),
    };
    runner::attach_provenance(
        &mut report,
        None,
        lanes::task_set_identity(&tasks),
        BTreeMap::new(),
        lanes::process_environment_summary(),
        vera_git_sha(),
        std::env::args().collect(),
    );
    Ok(report)
}

fn filter_tasks(
    tasks: Vec<types::BenchmarkTask>,
    task_ids: &[String],
    categories: &[String],
) -> Result<Vec<types::BenchmarkTask>> {
    if task_ids.is_empty() && categories.is_empty() {
        return Ok(tasks);
    }

    let requested_ids: HashSet<&str> = task_ids.iter().map(String::as_str).collect();
    let requested_categories = categories
        .iter()
        .map(|category| parse_category(category))
        .collect::<Result<HashSet<_>>>()?;
    let known_ids: HashSet<&str> = tasks.iter().map(|task| task.id.as_str()).collect();
    let unknown_ids: Vec<_> = requested_ids.difference(&known_ids).copied().collect();
    if !unknown_ids.is_empty() {
        anyhow::bail!("unknown task ID(s): {}", unknown_ids.join(", "));
    }

    let filtered: Vec<_> = tasks
        .into_iter()
        .filter(|task| {
            let id_match = requested_ids.is_empty() || requested_ids.contains(task.id.as_str());
            let category_match =
                requested_categories.is_empty() || requested_categories.contains(&task.category);
            id_match && category_match
        })
        .collect();
    if filtered.is_empty() {
        anyhow::bail!("No benchmark tasks match the requested filters");
    }
    Ok(filtered)
}

fn parse_category(value: &str) -> Result<types::TaskCategory> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "symbol_lookup" => Ok(types::TaskCategory::SymbolLookup),
        "intent" => Ok(types::TaskCategory::Intent),
        "cross_file" => Ok(types::TaskCategory::CrossFile),
        "config" => Ok(types::TaskCategory::Config),
        "disambiguation" => Ok(types::TaskCategory::Disambiguation),
        other => anyhow::bail!(
            "unknown task category '{other}'; expected symbol_lookup, intent, cross_file, config, or disambiguation"
        ),
    }
}

fn vera_git_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!sha.is_empty()).then_some(sha)
}
