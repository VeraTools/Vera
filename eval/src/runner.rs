//! Benchmark runner: executes tools against tasks and collects results.
//!
//! Provides a `ToolAdapter` trait that tool integrations implement,
//! plus a mock adapter for harness self-testing.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use crate::metrics;
use crate::types::{
    AggregateMetrics, BenchmarkTask, EvalReport, LaneProvenance, PerformanceMetrics,
    RetrievalResult, SembleSnapshot, TaskEvaluation, TaskResult, TaskSetIdentity, VersionInfo,
};

pub const METRIC_CONTRACT: &str = "vera-graded-2-1-task-mean-v1";

pub struct ReportProvenance {
    pub lane: Option<LaneProvenance>,
    pub task_set: TaskSetIdentity,
    pub config: BTreeMap<String, String>,
    pub environment: BTreeMap<String, String>,
    pub vera_git_sha: Option<String>,
    pub command: Vec<String>,
    pub corpus_version: u32,
    pub semble: Option<SembleSnapshot>,
}

/// Trait for tool adapters that execute search queries.
///
/// Each retrieval tool (Vera, ripgrep, grepai, etc.) implements this trait
/// to integrate with the benchmark runner.
#[allow(dead_code)]
pub trait ToolAdapter {
    /// Tool name for reporting.
    fn name(&self) -> &str;

    /// Tool version string.
    fn version(&self) -> String;

    /// Execute a search query and return ranked results.
    /// `path_scope` optionally restricts results to a subdirectory (e.g. "src/flask").
    fn search(
        &self,
        query: &str,
        repo_path: &str,
        path_scope: Option<&str>,
    ) -> Vec<RetrievalResult>;

    /// Index a repository (if the tool requires pre-indexing).
    /// Returns (index_time_secs, storage_size_bytes).
    fn index(&self, repo_path: &str) -> (f64, u64);
}

/// Add invocation metadata to a completed report.
pub fn attach_provenance(report: &mut EvalReport, provenance: ReportProvenance) {
    report.version_info.lane = provenance.lane;
    report.version_info.task_set = Some(provenance.task_set);
    report.version_info.environment = provenance.environment;
    report.version_info.vera_git_sha = provenance.vera_git_sha;
    report.version_info.host_cpu_model = Some(crate::lanes::host_cpu_model());
    report.version_info.command = provenance.command;
    report.version_info.corpus_version = provenance.corpus_version;
    report.version_info.metric_contract = METRIC_CONTRACT.to_string();
    report.version_info.semble = provenance.semble;
    report.version_info.config.extend(provenance.config);
}

/// Run benchmark with optional per-repo path scopes (benchmark_root).
pub fn run_benchmark_scoped(
    adapter: &dyn ToolAdapter,
    tasks: &[BenchmarkTask],
    repo_paths: &HashMap<String, String>,
    corpus_shas: &HashMap<String, String>,
    benchmark_roots: &HashMap<String, String>,
) -> EvalReport {
    let timestamp = chrono::Utc::now().to_rfc3339();

    // Index all repos and measure time/storage
    let mut total_index_time = 0.0;
    let mut total_storage = 0u64;
    for path in repo_paths.values() {
        let (time, size) = adapter.index(path);
        total_index_time += time;
        total_storage += size;
    }

    // Run all tasks and collect results
    let task_results: Vec<TaskResult> = tasks
        .iter()
        .map(|task| {
            let repo_path = repo_paths.get(&task.repo).map(String::as_str).unwrap_or("");
            let scope = benchmark_roots.get(&task.repo).map(String::as_str);

            let start = Instant::now();
            let results = adapter.search(&task.query, repo_path, scope);
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

            TaskResult {
                task_id: task.id.clone(),
                results,
                latency_ms,
            }
        })
        .collect();

    // Evaluate all tasks
    let per_task = metrics::evaluate_tasks(tasks, &task_results);

    // Compute performance metrics
    let performance =
        metrics::compute_performance_metrics(&task_results, total_index_time, total_storage);

    // Compute per-category aggregates
    let per_category = compute_category_aggregates(&per_task, &performance);

    // Compute overall aggregate
    let aggregate = metrics::aggregate_metrics(&per_task, performance);

    EvalReport {
        tool_name: adapter.name().to_string(),
        timestamp,
        version_info: VersionInfo {
            tool_version: adapter.version(),
            corpus_version: 1,
            metric_contract: METRIC_CONTRACT.to_string(),
            semble: None,
            repo_shas: corpus_shas.clone(),
            config: HashMap::new(),
            lane: None,
            task_set: None,
            vera_git_sha: None,
            host_cpu_model: None,
            command: Vec::new(),
            environment: BTreeMap::new(),
        },
        per_task,
        per_category,
        aggregate,
    }
}

/// Compute aggregate metrics per task category.
fn compute_category_aggregates(
    evaluations: &[TaskEvaluation],
    performance: &PerformanceMetrics,
) -> HashMap<String, AggregateMetrics> {
    let mut by_category: HashMap<String, Vec<&TaskEvaluation>> = HashMap::new();

    for eval in evaluations {
        by_category
            .entry(eval.category.to_string())
            .or_default()
            .push(eval);
    }

    by_category
        .into_iter()
        .map(|(cat, evals)| {
            let owned: Vec<TaskEvaluation> = evals.into_iter().cloned().collect();
            let agg = metrics::aggregate_metrics(&owned, performance.clone());
            (cat, agg)
        })
        .collect()
}

/// Mock tool adapter for harness self-testing.
///
/// Returns deterministic results based on task ground truth,
/// simulating a tool with configurable accuracy.
pub struct MockAdapter {
    /// Name of this mock configuration.
    pub name: String,
    /// Fraction of ground truth entries to include in results (0.0 to 1.0).
    pub accuracy: f64,
    /// Number of noise results to add.
    pub noise_results: usize,
    /// Simulated latency in milliseconds.
    pub simulated_latency_ms: f64,
}

impl Default for MockAdapter {
    fn default() -> Self {
        Self {
            name: "mock-perfect".to_string(),
            accuracy: 1.0,
            noise_results: 0,
            simulated_latency_ms: 5.0,
        }
    }
}

impl MockAdapter {
    /// Create a perfect mock that returns all ground truth results.
    pub fn perfect() -> Self {
        Self::default()
    }

    /// Create a partial mock that returns some ground truth results.
    pub fn partial(accuracy: f64) -> Self {
        Self {
            name: format!("mock-{:.0}pct", accuracy * 100.0),
            accuracy,
            noise_results: 3,
            simulated_latency_ms: 10.0,
        }
    }
}

impl ToolAdapter for MockAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> String {
        "mock-1.0.0".to_string()
    }

    fn search(&self, _query: &str, _repo_path: &str, _scope: Option<&str>) -> Vec<RetrievalResult> {
        Vec::new()
    }

    fn index(&self, _repo_path: &str) -> (f64, u64) {
        (1.5, 1024 * 1024) // 1.5s, 1MB
    }
}

/// Run benchmark with a mock adapter that uses task ground truth for results.
///
/// This is the primary self-testing function: it creates deterministic results
/// based on each task's ground truth, allowing us to verify metric computation.
pub fn run_benchmark_with_mock(mock: &MockAdapter, tasks: &[BenchmarkTask]) -> EvalReport {
    let timestamp = chrono::Utc::now().to_rfc3339();

    // Create mock task results based on ground truth
    let task_results: Vec<TaskResult> = tasks
        .iter()
        .map(|task| {
            let mut results = Vec::new();

            // Add noise results first (wrong file)
            for i in 0..mock.noise_results {
                results.push(RetrievalResult {
                    file_path: format!("noise/file_{}.rs", i),
                    line_start: 1,
                    line_end: 10,
                    score: 0.1,
                });
            }

            // Add ground truth results based on accuracy
            let gt_count = (task.ground_truth.len() as f64 * mock.accuracy).ceil() as usize;
            for gt in task.ground_truth.iter().take(gt_count) {
                results.push(RetrievalResult {
                    file_path: gt.file_path.clone(),
                    line_start: gt.line_start,
                    line_end: gt.line_end,
                    score: 0.9,
                });
            }

            TaskResult {
                task_id: task.id.clone(),
                results,
                latency_ms: mock.simulated_latency_ms,
            }
        })
        .collect();

    // Evaluate tasks
    let per_task = metrics::evaluate_tasks(tasks, &task_results);

    let performance = metrics::compute_performance_metrics(&task_results, 1.5, 1024 * 1024);

    let per_category = compute_category_aggregates(&per_task, &performance);
    let aggregate = metrics::aggregate_metrics(&per_task, performance);

    EvalReport {
        tool_name: mock.name.clone(),
        timestamp,
        version_info: VersionInfo {
            tool_version: mock.version(),
            corpus_version: 1,
            metric_contract: METRIC_CONTRACT.to_string(),
            semble: None,
            repo_shas: HashMap::new(),
            config: HashMap::from([
                ("accuracy".to_string(), mock.accuracy.to_string()),
                ("noise_results".to_string(), mock.noise_results.to_string()),
            ]),
            lane: None,
            task_set: None,
            vera_git_sha: None,
            host_cpu_model: None,
            command: Vec::new(),
            environment: BTreeMap::new(),
        },
        per_task,
        per_category,
        aggregate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GroundTruthEntry, TaskCategory};

    fn sample_tasks() -> Vec<BenchmarkTask> {
        vec![
            BenchmarkTask {
                id: "sym-001".to_string(),
                query: "find Config struct".to_string(),
                category: TaskCategory::SymbolLookup,
                repo: "ripgrep".to_string(),
                ground_truth: vec![GroundTruthEntry {
                    file_path: "crates/core/config.rs".to_string(),
                    line_start: 10,
                    line_end: 50,
                    relevance: 1,
                }],
                description: "Find the Config struct definition".to_string(),
            },
            BenchmarkTask {
                id: "intent-001".to_string(),
                query: "error handling patterns".to_string(),
                category: TaskCategory::Intent,
                repo: "flask".to_string(),
                ground_truth: vec![
                    GroundTruthEntry {
                        file_path: "src/flask/app.py".to_string(),
                        line_start: 100,
                        line_end: 150,
                        relevance: 1,
                    },
                    GroundTruthEntry {
                        file_path: "src/flask/errors.py".to_string(),
                        line_start: 1,
                        line_end: 80,
                        relevance: 1,
                    },
                ],
                description: "Find error handling code".to_string(),
            },
        ]
    }

    #[test]
    fn test_mock_perfect_adapter() {
        let mock = MockAdapter::perfect();
        let tasks = sample_tasks();
        let report = run_benchmark_with_mock(&mock, &tasks);

        assert_eq!(report.tool_name, "mock-perfect");
        assert_eq!(report.per_task.len(), 2);

        // Perfect mock should find all ground truth
        for eval in &report.per_task {
            assert_eq!(
                eval.retrieval_metrics.recall_at_10, 1.0,
                "Perfect mock should have Recall@10 = 1.0 for task {}",
                eval.task_id
            );
        }
    }

    #[test]
    fn test_mock_partial_adapter() {
        let mock = MockAdapter::partial(0.5);
        let tasks = sample_tasks();
        let report = run_benchmark_with_mock(&mock, &tasks);

        // With noise results, some metrics will be lower
        assert!(
            report.aggregate.retrieval.mrr < 1.0,
            "Partial mock with noise should have MRR < 1.0"
        );
    }

    #[test]
    fn test_report_has_categories() {
        let mock = MockAdapter::perfect();
        let tasks = sample_tasks();
        let report = run_benchmark_with_mock(&mock, &tasks);

        assert!(report.per_category.contains_key("symbol_lookup"));
        assert!(report.per_category.contains_key("intent"));
    }

    #[test]
    fn test_report_has_version_info() {
        let mock = MockAdapter::perfect();
        let tasks = sample_tasks();
        let report = run_benchmark_with_mock(&mock, &tasks);

        assert_eq!(report.version_info.tool_version, "mock-1.0.0");
        assert_eq!(report.version_info.corpus_version, 1);
    }

    #[test]
    fn test_consecutive_runs_stable() {
        let mock = MockAdapter::perfect();
        let tasks = sample_tasks();

        let report1 = run_benchmark_with_mock(&mock, &tasks);
        let report2 = run_benchmark_with_mock(&mock, &tasks);

        // Retrieval metrics should be identical (deterministic mock)
        for (e1, e2) in report1.per_task.iter().zip(report2.per_task.iter()) {
            let diff =
                (e1.retrieval_metrics.recall_at_10 - e2.retrieval_metrics.recall_at_10).abs();
            assert!(
                diff < 0.001,
                "Consecutive runs should produce identical retrieval metrics, diff={diff}"
            );
        }
    }
}
