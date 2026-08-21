//! Core data types for the Vera evaluation harness.
//!
//! Defines the structures for benchmark tasks, tool results, computed metrics,
//! and the overall evaluation report.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// A single benchmark task: a query with ground truth expectations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    /// Unique task identifier (e.g., "symbol-lookup-001").
    pub id: String,
    /// The search query to execute.
    pub query: String,
    /// Category of the task (symbol_lookup, intent, cross_file, config, disambiguation).
    pub category: TaskCategory,
    /// Which corpus repo this task targets.
    pub repo: String,
    /// Ground truth: expected results that a correct retrieval should find.
    pub ground_truth: Vec<GroundTruthEntry>,
    /// Optional description of what the task tests.
    #[serde(default)]
    pub description: String,
}

/// Categories of benchmark tasks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TaskCategory {
    SymbolLookup,
    Intent,
    CrossFile,
    Config,
    Disambiguation,
}

impl std::fmt::Display for TaskCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskCategory::SymbolLookup => write!(f, "symbol_lookup"),
            TaskCategory::Intent => write!(f, "intent"),
            TaskCategory::CrossFile => write!(f, "cross_file"),
            TaskCategory::Config => write!(f, "config"),
            TaskCategory::Disambiguation => write!(f, "disambiguation"),
        }
    }
}

/// A single ground truth entry: an expected file + line range in results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthEntry {
    /// Relative file path within the repo.
    pub file_path: String,
    /// Start line (1-based, inclusive).
    pub line_start: usize,
    /// End line (1-based, inclusive).
    pub line_end: usize,
    /// Relevance rank (1 = most relevant). Used for nDCG.
    #[serde(default = "default_relevance")]
    pub relevance: u32,
}

fn default_relevance() -> u32 {
    1
}

/// A single result returned by a retrieval tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// File path (relative to repo root).
    pub file_path: String,
    /// Start line (1-based).
    pub line_start: usize,
    /// End line (1-based).
    pub line_end: usize,
    /// Relevance score from the tool (higher = more relevant).
    #[serde(default)]
    pub score: f64,
}

/// Results from running a single tool on a single task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID.
    pub task_id: String,
    /// Retrieved results (in ranked order).
    pub results: Vec<RetrievalResult>,
    /// Query latency in milliseconds.
    pub latency_ms: f64,
}

/// Computed retrieval quality metrics for a single task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg: f64,
}

impl Default for RetrievalMetrics {
    fn default() -> Self {
        Self {
            recall_at_1: 0.0,
            recall_at_5: 0.0,
            recall_at_10: 0.0,
            mrr: 0.0,
            ndcg: 0.0,
        }
    }
}

/// Performance metrics from a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Median query latency in milliseconds.
    pub latency_p50_ms: f64,
    /// 95th percentile query latency in milliseconds.
    pub latency_p95_ms: f64,
    /// Index build time in seconds.
    pub index_time_secs: f64,
    /// Index storage size in bytes.
    pub storage_size_bytes: u64,
    /// Total output token count across all results.
    pub total_token_count: u64,
}

/// Per-task evaluation result: metrics for one task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvaluation {
    pub task_id: String,
    pub category: TaskCategory,
    pub retrieval_metrics: RetrievalMetrics,
    pub latency_ms: f64,
    pub result_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<RetrievalResult>,
}

/// Aggregate metrics across all tasks or a category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub retrieval: RetrievalMetrics,
    pub performance: PerformanceMetrics,
    pub task_count: usize,
}

/// Complete evaluation report from a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Tool/configuration being evaluated.
    pub tool_name: String,
    /// Timestamp of the run.
    pub timestamp: String,
    /// Version info for reproducibility.
    pub version_info: VersionInfo,
    /// Per-task evaluation results.
    pub per_task: Vec<TaskEvaluation>,
    /// Per-category aggregate metrics.
    pub per_category: HashMap<String, AggregateMetrics>,
    /// Overall aggregate metrics.
    pub aggregate: AggregateMetrics,
}

/// Version and configuration info for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Tool version string.
    pub tool_version: String,
    /// Corpus manifest version.
    pub corpus_version: u32,
    /// Repo SHAs used (name -> SHA).
    pub repo_shas: HashMap<String, String>,
    /// Configuration parameters (key -> value).
    #[serde(default)]
    pub config: HashMap<String, String>,
    /// Resolved model lane configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<LaneProvenance>,
    /// Identity of the task slice evaluated by this report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_set: Option<TaskSetIdentity>,
    /// Vera source revision used for the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vera_git_sha: Option<String>,
    /// Process arguments used to invoke the evaluator.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command: Vec<String>,
    /// Relevant environment values after lane overrides, with secrets redacted.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub environment: BTreeMap<String, String>,
}

/// Resolved configuration for one benchmark lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneProvenance {
    pub name: String,
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_data_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooling: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
    pub rerank: bool,
}

/// Stable identity for the selected task IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSetIdentity {
    pub count: usize,
    pub task_ids_sha256: String,
}

/// Corpus manifest parsed from corpus.toml.
#[derive(Debug, Clone, Deserialize)]
pub struct CorpusManifest {
    pub corpus: CorpusMetadata,
    pub repos: Vec<RepoEntry>,
}

/// Metadata section of corpus.toml.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct CorpusMetadata {
    pub version: u32,
    pub description: String,
    pub clone_root: String,
}

/// A single repo entry in corpus.toml.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct RepoEntry {
    pub name: String,
    pub url: String,
    pub commit: String,
    pub language: String,
    pub description: String,
    /// Optional subdirectory scope for benchmarks (e.g. "src/flask").
    /// When set, search results are filtered to only include files under this path.
    #[serde(default)]
    pub benchmark_root: Option<String>,
}
