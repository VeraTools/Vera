//! `vera update <path>` — Incrementally update the index.

use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};
use vera_core::config::InferenceBackend;
use vera_core::indexing::{UpdateOptions, UpdateProgress};

use crate::helpers::{cancel_task_on_signal, load_runtime_config, wait_for_interrupt};

pub struct CommandOptions {
    pub backend: InferenceBackend,
    pub exclude: Vec<String>,
    pub no_ignore: bool,
    pub no_default_excludes: bool,
    pub no_progress: bool,
    pub max_files: Option<usize>,
}

/// Run the `vera update <path>` command.
pub fn run(path: &str, json_output: bool, options: CommandOptions) -> anyhow::Result<()> {
    let CommandOptions {
        backend,
        exclude,
        no_ignore,
        no_default_excludes,
        no_progress,
        max_files,
    } = options;
    let repo_path = Path::new(path);

    if !repo_path.exists() {
        bail!(
            "path does not exist: {path}\n\
             Hint: check the path and try again."
        );
    }
    if !repo_path.is_dir() {
        bail!(
            "path is not a directory: {path}\n\
             Hint: vera update expects a directory path, not a file."
        );
    }

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;

    let mut config = load_runtime_config()?;
    config.adjust_for_backend(backend);
    config.indexing.extra_excludes = exclude;
    config.indexing.no_ignore = no_ignore;
    config.indexing.no_default_excludes = no_default_excludes;

    let (provider, model_name) = rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    ))?;

    // Check metadata mismatch
    let metadata_path = repo_path.join(".vera").join("metadata.db");
    if let Ok(metadata_store) = vera_core::storage::metadata::MetadataStore::open(&metadata_path) {
        if let (Some(s_model), Some(s_dim)) = (
            metadata_store.get_index_meta("model_name").unwrap_or(None),
            metadata_store
                .get_index_meta("embedding_dim")
                .unwrap_or(None),
        ) {
            if !vera_core::config::model_names_match_with_aliases(
                &s_model,
                &model_name,
                &config.embedding.model_aliases,
            ) {
                bail!(
                    "Index was created with model '{}' ({} dimensions), but you are using model '{}'. Please re-index with matching provider.",
                    s_model,
                    s_dim,
                    model_name
                );
            }
            if let Ok(dim) = s_dim.parse::<usize>() {
                use vera_core::embedding::EmbeddingProvider;
                if let Some(provider_dim) = provider.expected_dim() {
                    if provider_dim < dim {
                        bail!(
                            "Dimension mismatch: index has {} dimensions but active provider only returns {}. Please re-index with matching provider.",
                            dim,
                            provider_dim
                        );
                    }
                }
            }
        }
    }

    let show_progress = !json_output && !no_progress && std::io::stderr().is_terminal();
    let options = UpdateOptions { max_files };
    let cancellation = vera_core::CancellationToken::new();
    let operation_cancellation = cancellation.clone();
    let summary = if show_progress {
        let multi = cliclack::multi_progress("Updating...");
        let spinner = Arc::new(multi.add(cliclack::spinner()));
        spinner.start("Discovering files...");
        let embed_bar: Arc<cliclack::ProgressBar> = Arc::new(multi.add(cliclack::progress_bar(0)));
        let embed_started = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let spinner_ref = Arc::clone(&spinner);
        let embed_bar_ref = embed_bar.clone();
        let embed_started_ref = embed_started.clone();

        let on_progress = move |event: UpdateProgress| match event {
            UpdateProgress::DiscoveryDone { file_count } => {
                spinner_ref.stop(format!("Discovered {file_count} files"));
                spinner_ref.start("Classifying changes...");
            }
            UpdateProgress::ClassificationDone {
                modified,
                added,
                deleted,
                unchanged,
                deferred,
            } => {
                spinner_ref.stop(format!(
                    "Changes: {added} added, {modified} modified, {deleted} deleted; \
                     {unchanged} unchanged, {deferred} deferred"
                ));
                spinner_ref.start("Parsing changed files...");
            }
            UpdateProgress::ParsingDone {
                file_count,
                chunk_count,
            } => {
                spinner_ref.stop(format!(
                    "Parsed {file_count} changed files into {chunk_count} chunks"
                ));
            }
            UpdateProgress::EmbeddingProgress { done, total } => {
                if !embed_started_ref.load(std::sync::atomic::Ordering::Relaxed) {
                    embed_started_ref.store(true, std::sync::atomic::Ordering::Relaxed);
                    embed_bar_ref.set_length(total as u64);
                    embed_bar_ref.start("Generating embeddings...");
                }
                embed_bar_ref.set_position(done as u64);
                embed_bar_ref.set_message(format!("Generating embeddings ({done}/{total})"));
            }
            UpdateProgress::EmbeddingDone { count } => {
                if embed_started_ref.load(std::sync::atomic::Ordering::Relaxed) {
                    embed_bar_ref.stop(format!("Generated {count} embeddings"));
                }
                spinner_ref.start("Writing index updates...");
            }
            UpdateProgress::StorageDone => {
                spinner_ref.stop("Wrote index updates");
            }
        };

        let task_repo_path = repo_path.to_path_buf();
        let signal = wait_for_interrupt(rt.handle())?;
        let task = rt.handle().spawn(async move {
            vera_core::indexing::update_repository_with_options_and_progress_and_cancellation(
                &task_repo_path,
                &provider,
                &config,
                &model_name,
                &options,
                on_progress,
                &operation_cancellation,
            )
            .await
        });
        let result = rt.block_on(cancel_task_on_signal(task, signal, cancellation, "update"));
        multi.stop();
        result.context("update failed")?
    } else {
        let task_repo_path = repo_path.to_path_buf();
        let signal = wait_for_interrupt(rt.handle())?;
        let task = rt.handle().spawn(async move {
            vera_core::indexing::update_repository_with_options_and_progress_and_cancellation(
                &task_repo_path,
                &provider,
                &config,
                &model_name,
                &options,
                |_| {},
                &operation_cancellation,
            )
            .await
        });
        rt.block_on(cancel_task_on_signal(task, signal, cancellation, "update"))
            .context("update failed")?
    };

    // Output results.
    if json_output {
        let json = serde_json::to_string_pretty(&summary)
            .map_err(|e| anyhow::anyhow!("failed to serialize summary: {e}"))?;
        println!("{json}");
    } else {
        print_update_summary(&summary);
    }

    Ok(())
}

/// Print a human-readable summary of the update run.
fn print_update_summary(summary: &vera_core::indexing::UpdateSummary) {
    println!("Update complete!");
    println!();
    println!("  Files modified:  {}", summary.files_modified);
    println!("  Files added:     {}", summary.files_added);
    println!("  Files deleted:   {}", summary.files_deleted);
    println!("  Files unchanged: {}", summary.files_unchanged);
    if summary.files_with_tree_sitter_errors > 0 || summary.files_using_tier0_fallback > 0 {
        println!(
            "  Tree-sitter errors: {}",
            summary.files_with_tree_sitter_errors
        );
        println!(
            "  Tier 0 fallback:    {}",
            summary.files_using_tier0_fallback
        );
    }
    println!("  Files deferred:  {}", summary.files_deferred);
    println!("  Total chunks:    {}", summary.total_chunks);
    println!("  Elapsed time:    {:.2}s", summary.elapsed_secs);

    if !summary.parse_errors.is_empty() {
        println!();
        println!("  Parse errors ({}):", summary.parse_errors.len());
        for err in &summary.parse_errors {
            println!("    {}: {}", err.file_path, err.error);
        }
    }

    let total_pending = summary.files_modified
        + summary.files_added
        + summary.files_deleted
        + summary.files_deferred;
    if total_pending == 0 && summary.parse_errors.is_empty() {
        println!();
        println!("  Index is up to date — no changes detected.");
    }
}
