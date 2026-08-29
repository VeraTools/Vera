//! `vera index <path>` — Index a codebase for search.

use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};
use vera_core::config::InferenceBackend;
use vera_core::indexing::IndexProgress;

use crate::helpers::{cancel_task_on_signal, print_human_summary, wait_for_interrupt};
use crate::state;

/// Run the `vera index <path>` command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    path: &str,
    json_output: bool,
    backend: InferenceBackend,
    exclude: Vec<String>,
    no_ignore: bool,
    no_default_excludes: bool,
    no_progress: bool,
    verbose: bool,
    low_vram: bool,
) -> anyhow::Result<()> {
    let summary = execute(
        path,
        json_output,
        backend,
        exclude,
        no_ignore,
        no_default_excludes,
        no_progress,
        low_vram,
    )?;

    if json_output {
        let json = serde_json::to_string_pretty(&summary)
            .map_err(|e| anyhow::anyhow!("failed to serialize summary: {e}"))?;
        println!("{json}");
    } else {
        print_human_summary(&summary, verbose);
    }

    Ok(())
}

/// Index a repository and return the resulting summary.
#[allow(clippy::too_many_arguments)]
pub fn execute(
    path: &str,
    json_output: bool,
    backend: InferenceBackend,
    exclude: Vec<String>,
    no_ignore: bool,
    no_default_excludes: bool,
    no_progress: bool,
    low_vram: bool,
) -> anyhow::Result<vera_core::indexing::IndexSummary> {
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
             Hint: vera index expects a directory path, not a file."
        );
    }

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;

    let mut config = state::load_runtime_config()?;
    if low_vram {
        config.embedding.low_vram = true;
    }
    config.adjust_for_backend(backend);
    config.indexing.extra_excludes = exclude;
    config.indexing.no_ignore = no_ignore;
    config.indexing.no_default_excludes = no_default_excludes;

    let (provider, model_name) = rt.block_on(vera_core::embedding::create_dynamic_provider(
        &config, backend,
    ))?;

    // Show the progress display only for interactive, non-JSON runs. Mirrors
    // the same decision in `vera update` so both commands behave identically
    // when piped, redirected, or run with --no-progress.
    let show_progress = !json_output && !no_progress && std::io::stderr().is_terminal();
    if !show_progress {
        let cancellation = vera_core::CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let task_repo_path = repo_path.to_path_buf();
        let signal = wait_for_interrupt(rt.handle())?;
        let task = rt.handle().spawn(async move {
            vera_core::indexing::pipeline::index_repository_with_cancellation(
                &task_repo_path,
                &provider,
                &config,
                &model_name,
                &task_cancellation,
            )
            .await
        });
        let summary = rt
            .block_on(cancel_task_on_signal(
                task,
                signal,
                cancellation,
                "indexing",
            ))
            .context("indexing failed")?;
        return Ok(summary);
    }

    let multi = cliclack::multi_progress("Indexing...");
    let parse_spinner = Arc::new(multi.add(cliclack::spinner()));
    parse_spinner.start("Discovering files...");

    // Honest denominator: while parsing is still open, the embedding indicator
    // is open-ended (count without total / spinner) because cliclack 0.5.6 has
    // no unset-length API. Once `ParsingDone` arrives we switch to a fixed
    // total. The pure state machine lives in `vera_core::indexing::progress`.
    let tracker = Arc::new(std::sync::Mutex::new(
        vera_core::indexing::progress::HonestProgressTracker::new(),
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

    let on_progress = move |event: IndexProgress| {
        // Update the pure tracker and decide what to render.
        let display = {
            let mut guard = tracker_ref.lock().unwrap();
            guard.handle(&event)
        };
        match event {
            IndexProgress::DiscoveryDone { file_count } => {
                parse_spinner_ref.stop(format!("Discovered {file_count} files"));
                parse_spinner_ref.start("Parsing files...");
            }
            IndexProgress::ParsingDone { chunk_count } => {
                parse_spinner_ref.stop(format!("Parsed into {chunk_count} chunks"));
                // If we were showing an indeterminate spinner for embeddings,
                // transition to a determinate bar on the next embedding event.
                // We stop the spinner here so it does not linger as a separate
                // line while the bar takes over.
                if let Some(spinner_widget) = embed_spinner_ref.lock().unwrap().take() {
                    spinner_widget.stop(format!(
                        "Parsed into {chunk_count} chunks — finalizing embeddings..."
                    ));
                }
            }
            IndexProgress::EmbeddingProgress { .. } => {
                match display {
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
                        // If an indeterminate spinner is still around (race where
                        // ParsingDone and first determinate embedding interleave),
                        // stop it before showing the bar.
                        if let Some(spinner_widget) = embed_spinner_ref.lock().unwrap().take() {
                            spinner_widget.stop(format!(
                                "Parsed into {total} chunks — finalizing embeddings..."
                            ));
                        }
                        let mut guard = embed_bar_ref.lock().unwrap();
                        if guard.is_none() {
                            let w = Arc::new(multi_ref.add(cliclack::progress_bar(total as u64)));
                            w.start(format!("Generating embeddings ({}/{})", done, total));
                            w.set_position(done as u64);
                            *guard = Some(w);
                        } else if let Some(w) = guard.as_ref() {
                            // Length is fixed on creation; only position/message change.
                            w.set_position(done as u64);
                            w.set_message(format!("Generating embeddings ({}/{})", done, total));
                        }
                    }
                    Some(vera_core::indexing::progress::EmbedDisplay::Done { .. }) => {}
                    None => {}
                }
            }
            IndexProgress::EmbeddingDone { .. } => {
                if let Some(vera_core::indexing::progress::EmbedDisplay::Done { count }) = display {
                    if let Some(bar) = embed_bar_ref.lock().unwrap().take() {
                        bar.stop(format!("Generated {} embeddings", count));
                    } else if let Some(spinner_widget) = embed_spinner_ref.lock().unwrap().take() {
                        spinner_widget.stop(format!("Generated {} embeddings", count));
                    } else {
                        // No embed widget was ever created (e.g. tiny repo
                        // with no embedding progress events?); create a
                        // short-lived bar just to show completion.
                        let w = multi_ref.add(cliclack::progress_bar(count as u64));
                        w.start(format!("Generated {} embeddings", count));
                        w.stop(format!("Generated {} embeddings", count));
                    }
                }
            }
            IndexProgress::StorageDone => {}
        }
    };

    let cancellation = vera_core::CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let task_repo_path = repo_path.to_path_buf();
    let signal = wait_for_interrupt(rt.handle())?;
    let task = rt.handle().spawn(async move {
        vera_core::indexing::pipeline::index_repository_with_progress_and_cancellation(
            &task_repo_path,
            &provider,
            &config,
            &model_name,
            on_progress,
            &task_cancellation,
        )
        .await
    });
    let result = rt.block_on(cancel_task_on_signal(
        task,
        signal,
        cancellation,
        "indexing",
    ));
    // Honest display: cancellation mid-index and mid-run embedding failure are
    // displayed as cancelled/error states, not a frozen or 100% bar.
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

    result.context("indexing failed")
}
