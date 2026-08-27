//! Shared helper functions for CLI command implementations.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use clap::Args;
use vera_core::presentation::{CompactResult, truncate_to_budget};

/// Install the process interrupt handler and return a future for its first event.
#[cfg(unix)]
pub fn wait_for_interrupt(
    runtime: &tokio::runtime::Handle,
) -> std::io::Result<impl std::future::Future<Output = ()> + Send + 'static> {
    let _guard = runtime.enter();
    let mut signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    Ok(async move {
        signal.recv().await;
    })
}

/// Install the process interrupt handler and return a future for its first event.
#[cfg(windows)]
pub fn wait_for_interrupt(
    runtime: &tokio::runtime::Handle,
) -> std::io::Result<impl std::future::Future<Output = ()> + Send + 'static> {
    let _guard = runtime.enter();
    let mut signal = tokio::signal::windows::ctrl_c()?;
    Ok(async move {
        signal.recv().await;
    })
}

/// Cancel a spawned operation when signalled, then wait for it to stop safely.
///
/// The operation runs separately so the signal handler is active during synchronous discovery
/// and parsing. If publication has already started, this waits for and returns its real result.
pub async fn cancel_task_on_signal<T, Signal>(
    mut task: tokio::task::JoinHandle<anyhow::Result<T>>,
    signal: Signal,
    cancellation: vera_core::CancellationToken,
    operation_name: &str,
) -> anyhow::Result<T>
where
    Signal: std::future::Future<Output = ()>,
{
    tokio::pin!(signal);

    let result = tokio::select! {
        biased;
        result = &mut task => result,
        _ = &mut signal => {
            cancellation.cancel();
            task.await
        },
    };

    result.with_context(|| format!("{operation_name} task failed"))?
}

pub fn warn_if_index_stale(repo_path: &Path, indexing_config: &vera_core::config::IndexingConfig) {
    match vera_core::indexing::detect_staleness(repo_path, indexing_config) {
        Ok(freshness) => {
            if let Some(warning) = freshness.stale_warning() {
                let stderr = std::io::stderr();
                let mut err = stderr.lock();
                let _ = writeln!(err, "{warning}");
            }
        }
        Err(err) => {
            tracing::debug!(error = %err, "failed to check index freshness");
        }
    }
}

#[derive(Debug, Clone, Default, Args)]
pub struct SearchFilterArgs {
    /// Filter by programming language (case-insensitive).
    #[arg(long)]
    pub lang: Option<String>,
    /// Filter by file path glob pattern (e.g., "src/**/*.rs"). Repeatable;
    /// patterns are combined with OR semantics.
    #[arg(long)]
    pub path: Vec<String>,
    /// Filter by symbol type.
    /// Note: function and method are treated as aliases.
    #[arg(long, rename_all = "snake_case")]
    pub r#type: Option<String>,
    /// Restrict results to a coarse corpus scope.
    #[arg(long, value_parser = ["source", "docs", "runtime", "all"])]
    pub scope: Option<String>,
    /// Include generated or minified files such as dist bundles.
    #[arg(long)]
    pub include_generated: bool,
}

impl SearchFilterArgs {
    pub fn to_filters(&self) -> vera_core::types::SearchFilters {
        vera_core::types::SearchFilters {
            language: self.lang.clone(),
            path_glob: self.path.clone(),
            exact_paths: None,
            symbol_type: self.r#type.clone(),
            scope: self.scope.as_deref().and_then(|value| value.parse().ok()),
            include_generated: Some(self.include_generated),
        }
    }
}

#[derive(Debug, Clone, Default, Args)]
pub struct LocalBackendFlags {
    /// Use Potion Code static embeddings on CPU (the default backend).
    #[arg(long = "potion-code", visible_alias = "potion-cpu", group = "backend")]
    pub potion_code: bool,
    /// Use local ONNX models on CPU.
    #[arg(long = "onnx-jina-cpu", group = "backend")]
    pub onnx_jina_cpu: bool,
    /// Use local ONNX models with CUDA (NVIDIA GPU).
    #[arg(long = "onnx-jina-cuda", group = "backend")]
    pub onnx_jina_cuda: bool,
    /// Use local ONNX models with ROCm (AMD GPU, Linux only).
    #[arg(long = "onnx-jina-rocm", group = "backend")]
    pub onnx_jina_rocm: bool,
    /// Use local ONNX models with DirectML (Windows GPU).
    #[arg(long = "onnx-jina-directml", group = "backend")]
    pub onnx_jina_directml: bool,
    /// Use local ONNX models with CoreML (Apple Silicon).
    #[arg(long = "onnx-jina-coreml", group = "backend")]
    pub onnx_jina_coreml: bool,
    /// Use local ONNX models with OpenVINO (Intel GPU/iGPU, Linux only).
    #[arg(long = "onnx-jina-openvino", group = "backend")]
    pub onnx_jina_openvino: bool,
    /// Alias for --onnx-jina-cpu (backwards compatibility).
    #[arg(long, group = "backend", hide = true)]
    pub local: bool,
}

#[derive(Debug, Clone, Default, Args)]
pub struct GitScopeFlags {
    /// Limit results to modified, staged, and untracked files.
    #[arg(long, group = "git_scope")]
    pub changed: bool,
    /// Limit results to files changed since the given revision.
    #[arg(long, value_name = "REV", group = "git_scope")]
    pub since: Option<String>,
    /// Limit results to files changed since merge-base(HEAD, REV).
    #[arg(long, value_name = "REV", group = "git_scope")]
    pub base: Option<String>,
}

impl GitScopeFlags {
    pub fn resolve(&self) -> Option<vera_core::git_scope::GitScope> {
        if self.changed {
            Some(vera_core::git_scope::GitScope::Changed)
        } else if let Some(rev) = self.since.as_ref() {
            Some(vera_core::git_scope::GitScope::Since(rev.clone()))
        } else {
            self.base
                .as_ref()
                .map(|rev| vera_core::git_scope::GitScope::Base(rev.clone()))
        }
    }
}

pub fn prepare_indexed_repo(
    indexing_config: &vera_core::config::IndexingConfig,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current directory: {e}"))?;
    let index_dir = vera_core::indexing::index_dir(&cwd);
    if !index_dir.exists() {
        anyhow::bail!(MISSING_INDEX_MESSAGE);
    }
    warn_if_index_stale(&cwd, indexing_config);
    Ok((cwd, index_dir))
}

pub const MISSING_INDEX_MESSAGE: &str = "no index found in current directory.\n\
Hint: run `vera index <path>` first to create an index.";

pub fn should_offer_auto_index(json_output: bool, is_terminal: bool) -> bool {
    !json_output && is_terminal
}

pub fn apply_git_scope(
    cwd: &Path,
    filters: &vera_core::types::SearchFilters,
    git_scope: Option<&vera_core::git_scope::GitScope>,
) -> anyhow::Result<vera_core::types::SearchFilters> {
    let mut filters = filters.clone();
    if let Some(scope) = git_scope {
        filters.exact_paths = Some(Arc::new(vera_core::git_scope::resolve_scope(cwd, scope)?));
    }
    Ok(filters)
}

pub fn prepare_indexed_search(
    indexing_config: &vera_core::config::IndexingConfig,
    filters: &vera_core::types::SearchFilters,
    git_scope: Option<&vera_core::git_scope::GitScope>,
) -> anyhow::Result<(PathBuf, vera_core::types::SearchFilters)> {
    let (cwd, index_dir) = prepare_indexed_repo(indexing_config)?;
    let filters = apply_git_scope(&cwd, filters, git_scope)?;
    Ok((index_dir, filters))
}

impl LocalBackendFlags {
    pub fn any_set(&self) -> bool {
        self.potion_code
            || self.onnx_jina_cpu
            || self.onnx_jina_cuda
            || self.onnx_jina_rocm
            || self.onnx_jina_directml
            || self.onnx_jina_coreml
            || self.onnx_jina_openvino
            || self.local
    }

    pub fn explicit_backend(&self) -> Option<vera_core::config::InferenceBackend> {
        self.any_set().then(|| resolve_backend_flags(self))
    }

    pub fn resolve(&self) -> vera_core::config::InferenceBackend {
        resolve_backend_flags(self)
    }
}

#[derive(Debug, Clone, Default, Args)]
pub struct LocalEmbeddingModelFlags {
    /// Use CodeRankEmbed instead of Vera's default Jina ONNX embedding model.
    #[arg(
        long = "code-rank-embed",
        alias = "coderankembed",
        group = "local_embedding_source"
    )]
    pub code_rank_embed: bool,
    /// Hugging Face repo id or full Hugging Face URL for a custom local embedding model.
    #[arg(
        long = "embedding-repo",
        value_name = "REPO_OR_URL",
        group = "local_embedding_source"
    )]
    pub embedding_repo: Option<String>,
    /// Local directory containing a custom ONNX embedding model.
    #[arg(
        long = "embedding-dir",
        value_name = "DIR",
        group = "local_embedding_source"
    )]
    pub embedding_dir: Option<String>,
    /// Relative path to the ONNX file inside the selected repo or directory.
    #[arg(long = "embedding-onnx-file", value_name = "PATH")]
    pub embedding_onnx_file: Option<String>,
    /// Relative path to the ONNX external data file inside the selected repo or directory.
    #[arg(
        long = "embedding-onnx-data-file",
        value_name = "PATH",
        conflicts_with = "embedding_no_onnx_data"
    )]
    pub embedding_onnx_data_file: Option<String>,
    /// Use models that do not require an ONNX external data file.
    #[arg(long = "embedding-no-onnx-data")]
    pub embedding_no_onnx_data: bool,
    /// Relative path to the tokenizer file inside the selected repo or directory.
    #[arg(long = "embedding-tokenizer-file", value_name = "PATH")]
    pub embedding_tokenizer_file: Option<String>,
    /// Embedding dimension the model returns.
    #[arg(long = "embedding-dim", value_name = "DIM")]
    pub embedding_dim: Option<usize>,
    /// Pooling strategy for token-level output models.
    #[arg(long = "embedding-pooling", value_name = "POOLING", value_parser = ["mean", "cls", "last-token"])]
    pub embedding_pooling: Option<String>,
    /// Tokenizer truncation length for local embedding inference.
    #[arg(long = "embedding-max-length", value_name = "TOKENS")]
    pub embedding_max_length: Option<usize>,
    /// Optional asymmetric query prefix for models that require it.
    #[arg(long = "embedding-query-prefix", value_name = "TEXT")]
    pub embedding_query_prefix: Option<String>,
    /// Optional asymmetric document prefix for models that require it.
    #[arg(long = "embedding-document-prefix", value_name = "TEXT")]
    pub embedding_document_prefix: Option<String>,
}

impl LocalEmbeddingModelFlags {
    pub fn any_set(&self) -> bool {
        self.code_rank_embed
            || self.embedding_repo.is_some()
            || self.embedding_dir.is_some()
            || self.embedding_onnx_file.is_some()
            || self.embedding_onnx_data_file.is_some()
            || self.embedding_no_onnx_data
            || self.embedding_tokenizer_file.is_some()
            || self.embedding_dim.is_some()
            || self.embedding_pooling.is_some()
            || self.embedding_max_length.is_some()
            || self.embedding_query_prefix.is_some()
            || self.embedding_document_prefix.is_some()
    }
}

/// Resolve an `InferenceBackend` from the per-command boolean flags.
pub fn resolve_backend_flags(flags: &LocalBackendFlags) -> vera_core::config::InferenceBackend {
    use vera_core::config::{InferenceBackend, OnnxExecutionProvider};
    let explicit = if flags.potion_code {
        Some(InferenceBackend::PotionCode)
    } else if flags.onnx_jina_cpu || flags.local {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu))
    } else if flags.onnx_jina_cuda {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda))
    } else if flags.onnx_jina_rocm {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm))
    } else if flags.onnx_jina_directml {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl))
    } else if flags.onnx_jina_coreml {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl))
    } else if flags.onnx_jina_openvino {
        Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino))
    } else {
        None
    };
    vera_core::config::resolve_backend(explicit)
}

/// Output search results with a total character budget.
///
/// Priority: `--json` compact JSON > `--raw` verbose > default markdown codeblocks.
/// When `budget` is non-zero, output is truncated so it stays within the budget:
/// markdown and raw mode spend it progressively across results (lower-ranked
/// results are dropped first), JSON mode truncates the serialized document.
/// When `compact` is true, function/class bodies are stripped to show only signatures.
pub fn output_results(
    results: &[vera_core::types::SearchResult],
    json_output: bool,
    raw: bool,
    compact: bool,
    budget: usize,
) {
    use vera_core::parsing::signatures::extract_signature_for_path;

    // When compact mode is on, pre-compute signature-only content for each result.
    let compacted: Vec<String> = if compact {
        results
            .iter()
            .map(|r| extract_signature_for_path(&r.content, r.language, &r.file_path))
            .collect()
    } else {
        Vec::new()
    };
    let contents: Vec<&str> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            if compact {
                compacted[i].as_str()
            } else {
                r.content.as_str()
            }
        })
        .collect();

    if json_output {
        println!("{}", json_within_budget(results, &contents, budget));
    } else if raw {
        if results.is_empty() {
            println!("No results found.");
        } else {
            print!("{}", format_raw_results(results, &contents, budget));
        }
    } else {
        // Default: markdown codeblocks (most token-efficient for LLM agents).
        let mut remaining = budget;
        for (i, r) in results.iter().enumerate() {
            if budget > 0 && remaining == 0 {
                break;
            }
            if i > 0 {
                println!();
            }
            println!("```{}", result_info_line(r));
            let content = budget_slice(contents[i], budget, &mut remaining);
            print!("{}", content);
            if !content.ends_with('\n') {
                println!();
            }
            println!("```");
        }
    }
}

fn result_info_line(r: &vera_core::types::SearchResult) -> String {
    let mut info = format!("{}:{}-{}", r.file_path, r.line_start, r.line_end);
    if let (Some(stype), Some(name)) = (&r.symbol_type, &r.symbol_name) {
        info.push_str(&format!(" {stype}:{name}"));
    }
    info
}

/// Spend `remaining` on this chunk of content when a budget is set, returning
/// the part that fits.
fn budget_slice<'a>(
    content: &'a str,
    budget: usize,
    remaining: &mut usize,
) -> std::borrow::Cow<'a, str> {
    if budget == 0 {
        return std::borrow::Cow::Borrowed(content);
    }
    let c = truncate_to_budget(content, *remaining);
    *remaining = remaining.saturating_sub(c.len());
    c
}

fn json_results_string(results: &[vera_core::types::SearchResult], contents: &[&str]) -> String {
    let json_results: Vec<CompactResult> = results
        .iter()
        .zip(contents)
        .map(|(r, content)| {
            let mut cr = CompactResult::from_search_result(r);
            cr.content = std::borrow::Cow::Borrowed(content);
            cr
        })
        .collect();
    serde_json::to_string(&json_results)
        .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {e}\"}}"))
}

/// Serialize results so the document stays valid JSON within `budget`.
///
/// Cutting the serialized text at a byte offset splits string literals and
/// leaves arrays unclosed, which breaks every programmatic consumer. Whole
/// results are dropped instead, and a lone oversized result has its content
/// shortened before serialization.
fn json_within_budget(
    results: &[vera_core::types::SearchResult],
    contents: &[&str],
    budget: usize,
) -> String {
    let json = json_results_string(results, contents);
    if budget == 0 || json.len() <= budget || results.is_empty() {
        return json;
    }
    for count in (1..results.len()).rev() {
        let shorter = json_results_string(&results[..count], &contents[..count]);
        if shorter.len() <= budget {
            return shorter;
        }
    }
    // One result still exceeds the budget: shorten its content instead.
    let head = &results[..1];
    let overhead = json_results_string(head, &[""]).len();
    let room = budget.saturating_sub(overhead).max(1);
    let trimmed = truncate_to_budget(contents[0], room);
    json_results_string(head, &[trimmed.as_ref()])
}

/// Numbered verbose listing, one block per result. With a budget, each result's
/// content spends from a shared allowance like markdown mode does; once it runs
/// out, lower-ranked results are dropped (headers never consume budget).
fn format_raw_results(
    results: &[vera_core::types::SearchResult],
    contents: &[&str],
    budget: usize,
) -> String {
    let mut out = String::new();
    let mut remaining = budget;
    for (i, result) in results.iter().enumerate() {
        if budget > 0 && remaining == 0 {
            break;
        }
        out.push_str(&format!(
            "{}. {} (lines {}-{}, {})\n",
            i + 1,
            result.file_path,
            result.line_start,
            result.line_end,
            result.language,
        ));
        if let Some(ref name) = result.symbol_name {
            match &result.symbol_type {
                Some(stype) => out.push_str(&format!("   {stype} {name}\n")),
                None => out.push_str(&format!("   {name}\n")),
            }
        }
        out.push_str(&format!("   score: {:.6}\n", result.score));
        let content = budget_slice(contents[i], budget, &mut remaining);
        for line in content.lines().take(3) {
            out.push_str("   │ ");
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// Format a byte count as a compact human-readable string (e.g. "1.2 MB").
fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1_024.0;
    const MB: f64 = 1_024.0 * KB;
    const GB: f64 = 1_024.0 * MB;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else {
        format!("{:.1} KB", b / KB)
    }
}

/// Print a human-readable summary of the indexing run.
///
/// When `verbose` is true, individual file paths are listed for skipped-file
/// categories. Otherwise only counts are shown with a hint to rerun with `-v`.
pub fn print_human_summary(summary: &vera_core::indexing::IndexSummary, verbose: bool) {
    println!("Indexing complete!");
    println!();
    println!("  Files parsed:        {}", summary.files_parsed);
    println!("  Chunks created:      {}", summary.chunks_created);
    println!("  Embeddings generated: {}", summary.embeddings_generated);
    println!("  Elapsed time:        {:.2}s", summary.elapsed_secs);

    if summary.files_with_tree_sitter_errors > 0 || summary.files_using_tier0_fallback > 0 {
        println!();
        println!("  Index health:");
        if summary.files_with_tree_sitter_errors > 0 {
            println!(
                "    Tree-sitter errors: {}",
                summary.files_with_tree_sitter_errors
            );
        }
        if summary.files_using_tier0_fallback > 0 {
            println!(
                "    Tier 0 fallback:    {}",
                summary.files_using_tier0_fallback
            );
        }
    }

    // Report skipped files if any.
    let skipped_total = summary.binary_skipped + summary.large_skipped + summary.error_skipped;
    if skipped_total > 0 {
        println!();
        println!("  Skipped files:");
        if summary.binary_skipped > 0 {
            println!("    Binary:     {}", summary.binary_skipped);
        }
        if summary.large_skipped > 0 {
            println!("    Too large:  {}", summary.large_skipped);
            if verbose {
                for (path, size) in &summary.large_skipped_paths {
                    println!("      - {path} ({size})", size = format_bytes(*size));
                }
            }
        }
        if summary.error_skipped > 0 {
            println!("    Read errors: {}", summary.error_skipped);
        }
        if !verbose && !summary.large_skipped_paths.is_empty() {
            println!();
            println!("  Rerun with --verbose (-v) to see skipped file paths.");
        }
    }

    // Report parse errors if any.
    if !summary.parse_errors.is_empty() {
        println!();
        println!("  Parse errors ({}):", summary.parse_errors.len());
        for err in &summary.parse_errors {
            println!("    {}: {}", err.file_path, err.error);
        }
    }

    // Special message for empty repos.
    if summary.files_parsed == 0 && summary.chunks_created == 0 {
        println!();
        println!("  No source files found to index.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vera_core::types::Language;

    /// Two results whose combined content far exceeds a small budget.
    fn two_results() -> Vec<vera_core::types::SearchResult> {
        let content = "fn a() {\n    body\n}\n".repeat(20);
        vec![
            vera_core::types::SearchResult {
                file_path: "src/a.rs".to_string(),
                line_start: 1,
                line_end: 3,
                content: content.clone(),
                language: Language::Rust,
                score: 0.9,
                symbol_name: Some("a".to_string()),
                symbol_type: None,
            },
            vera_core::types::SearchResult {
                file_path: "src/b.rs".to_string(),
                line_start: 10,
                line_end: 12,
                content,
                language: Language::Rust,
                score: 0.5,
                symbol_name: None,
                symbol_type: None,
            },
        ]
    }

    #[test]
    fn json_output_stays_parseable_within_the_character_budget() {
        let results = two_results();
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        let full = json_results_string(&results, &contents);
        assert!(full.len() > 300, "fixture must exceed the budget below");

        let budgeted = json_within_budget(&results, &contents, 300);
        let parsed: serde_json::Value =
            serde_json::from_str(&budgeted).expect("budgeted JSON must parse");
        assert!(parsed.as_array().is_some_and(|a| !a.is_empty()));
        assert!(
            budgeted.len() < full.len(),
            "budget must drop or shorten results"
        );

        // Without a budget the document is untouched.
        assert_eq!(json_within_budget(&results, &contents, 0), full);
    }

    #[test]
    fn json_output_shortens_a_single_oversized_result() {
        let results = two_results();
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        let budgeted = json_within_budget(&results[..1], &contents[..1], 120);
        let parsed: serde_json::Value =
            serde_json::from_str(&budgeted).expect("budgeted JSON must parse");
        assert_eq!(parsed.as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn raw_output_spends_the_budget_across_results_and_drops_lower_ranked_ones() {
        // Single-line contents: truncation cannot stop early at a line
        // boundary, so each result spends its whole remaining allowance.
        let results = two_results();
        let results: Vec<vera_core::types::SearchResult> = results
            .into_iter()
            .map(|mut r| {
                r.content = "x".repeat(400);
                r
            })
            .collect();
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();

        // No budget: both blocks print in full.
        let unlimited = format_raw_results(&results, &contents, 0);
        assert_eq!(unlimited.matches("(lines ").count(), 2);
        assert!(unlimited.contains("src/b.rs"));

        // A budget the first result exhausts exactly: its header stays
        // visible, but the second result is dropped entirely.
        let first_only = format_raw_results(&results, &contents, 200);
        assert!(first_only.contains("src/a.rs"));
        assert!(!first_only.contains("src/b.rs"), "{first_only}");
    }

    #[test]
    fn raw_output_format_is_stable_without_a_budget() {
        let mut result = two_results().remove(0);
        result.content = "line one\nline two\nline three\nline four\n".to_string();
        let out = format_raw_results(std::slice::from_ref(&result), &[result.content.as_str()], 0);
        assert_eq!(
            out,
            "1. src/a.rs (lines 1-3, rust)\n   \
             a\n   score: 0.900000\n   \
             │ line one\n   │ line two\n   │ line three\n\n"
        );
    }

    #[test]
    fn every_local_embedding_flag_on_its_own_counts_as_set() {
        // `any_set` decides whether `vera setup` runs unattended and whether a
        // non-ONNX backend rejects the flags. A flag missing from it is
        // silently ignored when it is the only one passed, so each one is
        // checked alone rather than in combination.
        type FlagMutator = fn(&mut LocalEmbeddingModelFlags);
        let mutators: Vec<(&str, FlagMutator)> = vec![
            ("--code-rank-embed", |f| f.code_rank_embed = true),
            ("--embedding-repo", |f| {
                f.embedding_repo = Some("org/repo".to_string())
            }),
            ("--embedding-dir", |f| {
                f.embedding_dir = Some("/models".to_string())
            }),
            ("--embedding-onnx-file", |f| {
                f.embedding_onnx_file = Some("onnx/model.onnx".to_string())
            }),
            ("--embedding-onnx-data-file", |f| {
                f.embedding_onnx_data_file = Some("onnx/model.onnx_data".to_string())
            }),
            ("--embedding-no-onnx-data", |f| {
                f.embedding_no_onnx_data = true
            }),
            ("--embedding-tokenizer-file", |f| {
                f.embedding_tokenizer_file = Some("tokenizer.json".to_string())
            }),
            ("--embedding-dim", |f| f.embedding_dim = Some(768)),
            ("--embedding-pooling", |f| {
                f.embedding_pooling = Some("cls".to_string())
            }),
            ("--embedding-max-length", |f| {
                f.embedding_max_length = Some(512)
            }),
            ("--embedding-query-prefix", |f| {
                f.embedding_query_prefix = Some("Query:".to_string())
            }),
            ("--embedding-document-prefix", |f| {
                f.embedding_document_prefix = Some("Document:".to_string())
            }),
        ];

        assert!(!LocalEmbeddingModelFlags::default().any_set());
        for (flag, set_it) in mutators {
            let mut flags = LocalEmbeddingModelFlags::default();
            set_it(&mut flags);
            assert!(flags.any_set(), "{flag} alone was not treated as set");
        }
    }

    #[test]
    fn index_freshness_summary_formats_nonzero_counts() {
        let freshness = vera_core::indexing::IndexFreshness {
            files_added: 2,
            files_modified: 1,
            files_deleted: 3,
        };
        assert_eq!(freshness.summary(), "2 added, 1 modified, 3 deleted");
    }

    #[test]
    fn auto_index_offer_requires_human_output_and_a_terminal() {
        assert!(should_offer_auto_index(false, true));
        assert!(!should_offer_auto_index(true, true));
        assert!(!should_offer_auto_index(false, false));
        assert!(!should_offer_auto_index(true, false));
    }

    #[test]
    fn missing_index_message_preserves_the_cli_contract() {
        assert_eq!(
            MISSING_INDEX_MESSAGE,
            "no index found in current directory.\nHint: run `vera index <path>` first to create an index."
        );
    }

    #[tokio::test]
    async fn ready_operation_error_wins_over_ready_signal() {
        for _ in 0..64 {
            let task = tokio::spawn(async { Err::<(), _>(anyhow::anyhow!("provider failed")) });
            tokio::task::yield_now().await;
            let error = cancel_task_on_signal(
                task,
                std::future::ready(()),
                vera_core::CancellationToken::new(),
                "test operation",
            )
            .await
            .unwrap_err();

            assert_eq!(error.to_string(), "provider failed");
        }
    }

    #[tokio::test]
    async fn cancellation_waits_for_the_operation_to_stop() {
        let cancellation = vera_core::CancellationToken::new();
        let operation_cancellation = cancellation.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let task = tokio::spawn(async move {
            let _ = started_tx.send(());
            while !operation_cancellation.is_cancelled() {
                tokio::task::yield_now().await;
            }
            Err::<(), _>(anyhow::anyhow!("operation cancelled safely"))
        });
        let signal = async move {
            let _ = started_rx.await;
        };

        let error = cancel_task_on_signal(task, signal, cancellation, "test operation")
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "operation cancelled safely");
    }
}

/// Warn when a `--path` pattern matched none of the indexed files.
///
/// Empty results look the same whether the query found nothing or the filter
/// excluded everything, and the two want different fixes. The common case is a
/// directory pattern carrying a wildcard: `--path 'crates/*/src'` matches no
/// *file*, while the wildcard-free `--path crates/vera-core/src` is treated as
/// a directory prefix and matches everything beneath it.
///
/// Returns `None` when every pattern matched something, so a genuinely empty
/// result set stays quiet.
pub fn path_filter_hint(
    index_dir: &std::path::Path,
    filters: &vera_core::types::SearchFilters,
) -> Option<String> {
    if filters.path_glob.is_empty() {
        return None;
    }

    // Best effort: this runs only on an empty result set, and a hint is not
    // worth failing a command over.
    let store =
        vera_core::storage::metadata::MetadataStore::open(&index_dir.join("metadata.db")).ok()?;
    let files = store.indexed_files().ok()?;
    let unmatched = filters.path_patterns_matching_nothing(&files);
    // `path_glob` is OR-combined, so one working pattern still admits files and
    // the empty result is then a genuine miss rather than the filter's doing.
    if unmatched.is_empty() || unmatched.len() != filters.path_glob.len() {
        return None;
    }

    let quoted: Vec<String> = unmatched.iter().map(|p| format!("`{p}`")).collect();
    let suggestions: Vec<String> = unmatched
        .iter()
        .copied()
        .filter_map(directory_pattern_suggestion)
        .collect();

    let mut hint = format!(
        "note: no indexed file matches {}; the path filter excluded everything, so this is not necessarily an empty search",
        quoted.join(", ")
    );
    if !suggestions.is_empty() {
        hint.push_str(&format!(
            "\n      a directory pattern containing a wildcard matches no file on its own; try {}",
            suggestions.join(", ")
        ));
    }
    Some(hint)
}

/// The `/**` spelling to suggest for an unmatched pattern, if one makes sense.
///
/// Only patterns that look like a directory prefix get one. Appending `/**` to
/// anything else produces a suggestion that cannot match: `*.rs/**` and
/// `Makefile*/**` both ask for files beneath a directory of that name, and
/// `src/**/` already ends in `**` once the trailing separator is normalized.
///
/// Separate from `path_filter_hint` so the classification can be tested
/// without an index behind it.
fn directory_pattern_suggestion(pattern: &str) -> Option<String> {
    let trimmed = pattern.trim_end_matches(['/', '\\']);
    let last_segment = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);

    let looks_like_a_directory_prefix = trimmed.contains('*')
        && !trimmed.ends_with("**")
        // A literal directory name: no extension, and not itself a glob.
        && !last_segment.contains('.')
        && !last_segment.contains('*');

    looks_like_a_directory_prefix.then(|| format!("`{trimmed}/**`"))
}

#[cfg(test)]
mod path_hint_tests {
    use super::directory_pattern_suggestion;

    #[test]
    fn only_directory_shaped_patterns_get_a_suggestion() {
        // The case the hint exists for: a wildcarded directory prefix, which
        // matches no file on its own.
        assert_eq!(
            directory_pattern_suggestion("crates/*/src").as_deref(),
            Some("`crates/*/src/**`")
        );
        // A trailing separator is normalized, not carried into the suggestion.
        assert_eq!(
            directory_pattern_suggestion("crates/*/src/").as_deref(),
            Some("`crates/*/src/**`")
        );

        // Everything below would produce a suggestion that cannot match.
        for pattern in [
            "*.rs",      // extension glob: `*.rs/**` wants files under a dir named `*.rs`
            "src/*.ts",  // same, with a prefix
            "Makefile*", // extensionless file glob, still not a directory
            "src/**",    // already recursive
            "src/**/",   // already recursive, with a trailing separator
            "crates/*",  // last segment is itself a glob, not a directory name
            "src",       // no wildcard: the prefix fallback already covers it
        ] {
            assert_eq!(
                directory_pattern_suggestion(pattern),
                None,
                "{pattern} must not get a `/**` suggestion"
            );
        }
    }
}
