//! `vera update <path>` — Incrementally update the index.

use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};
use vera_core::config::InferenceBackend;
use vera_core::indexing::{UpdateOptions, UpdateProgress};

use crate::helpers::{cancel_task_on_signal, wait_for_interrupt};
use crate::state;

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

    let mut config = state::load_runtime_config()?;
    config.adjust_for_backend(backend);
    config.indexing.extra_excludes = exclude;
    config.indexing.no_ignore = no_ignore;
    config.indexing.no_default_excludes = no_default_excludes;

    let (provider, model_name) = rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    ))?;

    // Check metadata mismatch. Read-only open: a missing database means
    // "no index yet", which the update itself will create.
    let metadata_path = repo_path.join(".vera").join("metadata.db");
    if let Ok(metadata_store) =
        vera_core::storage::metadata::MetadataStore::open_existing(&metadata_path)
        && let (Some(s_model), Some(s_dim)) = (
            metadata_store.get_index_meta("model_name").unwrap_or(None),
            metadata_store
                .get_index_meta("embedding_dim")
                .unwrap_or(None),
        )
    {
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
            if let Some(provider_dim) = provider.expected_dim()
                && provider_dim < dim
            {
                bail!(
                    "Dimension mismatch: index has {} dimensions but active provider only returns {}. Please re-index with matching provider.",
                    dim,
                    provider_dim
                );
            }
        }
    }

    let show_progress = !json_output && !no_progress && std::io::stderr().is_terminal();
    let options = UpdateOptions { max_files };
    let cancellation = vera_core::CancellationToken::new();
    let operation_cancellation = cancellation.clone();
    let summary = if show_progress {
        let multi = cliclack::multi_progress("Updating...");
        let parse_spinner = Arc::new(multi.add(cliclack::spinner()));
        parse_spinner.start("Discovering files...");

        // Honest denominator for update as well: reuse the same
        // open-ended vs fixed-total contract. The incremental pipeline
        // normally parses all changed files before embedding, so the
        // first embedding is already determinate, but the tracker handles
        // the general case (and documents the contract) without ever
        // showing a growing denominator as a fixed total.
        let tracker = Arc::new(std::sync::Mutex::new(
            vera_core::indexing::progress::UpdateProgressTracker::new(),
        ));
        let embed_spinner: Arc<std::sync::Mutex<Option<Arc<cliclack::ProgressBar>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let embed_bar: Arc<std::sync::Mutex<Option<Arc<cliclack::ProgressBar>>>> =
            Arc::new(std::sync::Mutex::new(None));

        let parse_spinner_ref = Arc::clone(&parse_spinner);
        let tracker_ref = Arc::clone(&tracker);
        let embed_spinner_ref = Arc::clone(&embed_spinner);
        let embed_bar_ref = Arc::clone(&embed_bar);
        let multi_ref = multi.clone();

        let on_progress = move |event: UpdateProgress| {
            let display = {
                let mut guard = tracker_ref.lock().unwrap();
                guard.handle(&event)
            };
            match event {
                UpdateProgress::DiscoveryDone { file_count } => {
                    parse_spinner_ref.stop(format!("Discovered {file_count} files"));
                    parse_spinner_ref.start("Classifying changes...");
                }
                UpdateProgress::ClassificationDone {
                    modified,
                    added,
                    deleted,
                    unchanged,
                    deferred,
                } => {
                    parse_spinner_ref.stop(format!(
                        "Changes: {added} added, {modified} modified, {deleted} deleted; \
                         {unchanged} unchanged, {deferred} deferred"
                    ));
                    parse_spinner_ref.start("Parsing changed files...");
                }
                UpdateProgress::ParsingDone {
                    file_count,
                    chunk_count,
                } => {
                    parse_spinner_ref.stop(format!(
                        "Parsed {file_count} changed files into {chunk_count} chunks"
                    ));
                    if let Some(spinner_widget) = embed_spinner_ref.lock().unwrap().take() {
                        spinner_widget.stop(format!(
                            "Parsed {file_count} changed files into {chunk_count} chunks — finalizing embeddings..."
                        ));
                    }
                }
                UpdateProgress::EmbeddingProgress { .. } => match display {
                    Some(vera_core::indexing::progress::EmbedDisplay::Indeterminate { done }) => {
                        let mut guard = embed_spinner_ref.lock().unwrap();
                        if guard.is_none() {
                            let w = Arc::new(multi_ref.add(cliclack::spinner()));
                            w.start(format!("Generating embeddings ({} chunks so far)", done));
                            *guard = Some(w);
                        } else if let Some(w) = guard.as_ref() {
                            w.set_message(format!(
                                "Generating embeddings ({} chunks so far)",
                                done
                            ));
                        }
                    }
                    Some(vera_core::indexing::progress::EmbedDisplay::Determinate {
                        done,
                        total,
                    }) => {
                        if let Some(spinner_widget) = embed_spinner_ref.lock().unwrap().take() {
                            spinner_widget
                                .stop(format!("Generating embeddings ({}/{})", done, total));
                        }
                        let mut guard = embed_bar_ref.lock().unwrap();
                        if guard.is_none() {
                            let w = Arc::new(multi_ref.add(cliclack::progress_bar(total as u64)));
                            w.start(format!("Generating embeddings ({}/{})", done, total));
                            w.set_position(done as u64);
                            *guard = Some(w);
                        } else if let Some(w) = guard.as_ref() {
                            w.set_position(done as u64);
                            w.set_message(format!("Generating embeddings ({}/{})", done, total));
                        }
                    }
                    Some(vera_core::indexing::progress::EmbedDisplay::Done { .. }) => {}
                    None => {}
                },
                UpdateProgress::EmbeddingDone { .. } => {
                    if let Some(vera_core::indexing::progress::EmbedDisplay::Done { count }) =
                        display
                    {
                        if let Some(bar) = embed_bar_ref.lock().unwrap().take() {
                            bar.stop(format!("Generated {} embeddings", count));
                        } else if let Some(spinner_widget) =
                            embed_spinner_ref.lock().unwrap().take()
                        {
                            spinner_widget.stop(format!("Generated {} embeddings", count));
                        }
                        parse_spinner_ref.start("Writing index updates...");
                    }
                }
                UpdateProgress::StorageDone => {
                    parse_spinner_ref.stop("Wrote index updates");
                }
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
        // Honest display for update as well: cancelled / error are not shown as success.
        let is_cancel = result
            .as_ref()
            .is_err_and(|err| err.to_string().to_lowercase().contains("cancel"));
        if is_cancel {
            if let Some(bar) = embed_bar.lock().unwrap().take() {
                bar.cancel("Cancelled");
            }
            if let Some(spinner_widget) = embed_spinner.lock().unwrap().take() {
                spinner_widget.cancel("Cancelled");
            }
            parse_spinner.cancel("Cancelled");
            multi.cancel();
        } else if let Err(err) = &result {
            let msg = err.to_string();
            if let Some(bar) = embed_bar.lock().unwrap().take() {
                bar.error(format!("Failed: {msg}"));
            }
            if let Some(spinner_widget) = embed_spinner.lock().unwrap().take() {
                spinner_widget.error(format!("Failed: {msg}"));
            }
            parse_spinner.error(format!("Failed: {msg}"));
            multi.error(&msg);
        } else {
            multi.stop();
        }
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
