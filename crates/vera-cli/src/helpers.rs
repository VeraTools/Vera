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

/// Load the effective runtime configuration.
pub fn load_runtime_config() -> anyhow::Result<vera_core::config::VeraConfig> {
    crate::state::load_runtime_config()
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
    /// Use Potion Code static embeddings on CPU.
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
        anyhow::bail!(
            "no index found in current directory.\n\
             Hint: run `vera index <path>` first to create an index."
        );
    }
    warn_if_index_stale(&cwd, indexing_config);
    Ok((cwd, index_dir))
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
    /// Use CodeRankEmbed instead of Vera's default Jina local embedding model.
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
/// When `budget` is non-zero, output is progressively truncated so the combined
/// content stays within the budget. Lower-ranked results are truncated first.
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
    // Helper: pick compacted or original content by index.
    macro_rules! content_for {
        ($i:expr, $r:expr) => {
            if compact {
                compacted[$i].as_str()
            } else {
                $r.content.as_str()
            }
        };
    }

    if json_output {
        let json_results: Vec<CompactResult> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut cr = CompactResult::from_search_result(r);
                if compact {
                    cr.content = std::borrow::Cow::Borrowed(compacted[i].as_str());
                }
                cr
            })
            .collect();
        let json = serde_json::to_string(&json_results)
            .unwrap_or_else(|e| format!("{{\"error\": \"failed to serialize: {e}\"}}"));
        println!("{json}");
    } else if raw {
        if results.is_empty() {
            println!("No results found.");
        } else {
            for (i, result) in results.iter().enumerate() {
                println!(
                    "{}. {} (lines {}-{}, {})",
                    i + 1,
                    result.file_path,
                    result.line_start,
                    result.line_end,
                    result.language,
                );
                if let Some(ref name) = result.symbol_name {
                    if let Some(ref stype) = result.symbol_type {
                        println!("   {stype} {name}");
                    } else {
                        println!("   {name}");
                    }
                }
                println!("   score: {:.6}", result.score);
                let display_content = content_for!(i, result);
                let preview: String = display_content
                    .lines()
                    .take(3)
                    .map(|l| format!("   │ {l}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                println!("{preview}");
                println!();
            }
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
            let mut info = format!("{}:{}-{}", r.file_path, r.line_start, r.line_end);
            if let (Some(stype), Some(name)) = (&r.symbol_type, &r.symbol_name) {
                info.push_str(&format!(" {stype}:{name}"));
            }
            println!("```{info}");
            let base_content = content_for!(i, r);
            let content = if budget > 0 {
                let c = truncate_to_budget(base_content, remaining);
                remaining = remaining.saturating_sub(c.len());
                c
            } else {
                std::borrow::Cow::Borrowed(base_content)
            };
            print!("{}", content);
            if !content.ends_with('\n') {
                println!();
            }
            println!("```");
        }
    }
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

    #[test]
    fn index_freshness_summary_formats_nonzero_counts() {
        let freshness = vera_core::indexing::IndexFreshness {
            files_added: 2,
            files_modified: 1,
            files_deleted: 3,
        };
        assert_eq!(freshness.summary(), "2 added, 1 modified, 3 deleted");
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
