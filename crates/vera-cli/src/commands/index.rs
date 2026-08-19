//! `vera index <path>` — Index a codebase for search.

use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};
use vera_core::config::InferenceBackend;
use vera_core::indexing::IndexProgress;

use crate::helpers::{
    cancel_task_on_signal, load_runtime_config, print_human_summary, wait_for_interrupt,
};

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

    let mut config = load_runtime_config()?;
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
    let spinner = Arc::new(multi.add(cliclack::spinner()));
    spinner.start("Discovering files...");
    let embed_bar: Arc<cliclack::ProgressBar> = Arc::new(multi.add(cliclack::progress_bar(0)));
    let embed_started = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let spinner_ref = Arc::clone(&spinner);
    let embed_bar_ref = embed_bar.clone();
    let embed_started_ref = embed_started.clone();

    let on_progress = move |event: IndexProgress| match event {
        IndexProgress::DiscoveryDone { file_count } => {
            spinner_ref.stop(format!("Discovered {file_count} files"));
        }
        IndexProgress::ParsingDone { chunk_count } => {
            spinner_ref.start(format!("Parsed into {chunk_count} chunks"));
            spinner_ref.stop(format!("Parsed into {chunk_count} chunks"));
        }
        IndexProgress::EmbeddingProgress { done, total } => {
            if !embed_started_ref.load(std::sync::atomic::Ordering::Relaxed) {
                embed_started_ref.store(true, std::sync::atomic::Ordering::Relaxed);
                embed_bar_ref.set_length(total as u64);
                embed_bar_ref.start("Generating embeddings...");
            }
            embed_bar_ref.set_position(done as u64);
            embed_bar_ref.set_message(format!("Generating embeddings ({done}/{total})"));
        }
        IndexProgress::EmbeddingDone { count } => {
            embed_bar_ref.stop(format!("Generated {count} embeddings"));
        }
        IndexProgress::StorageDone => {}
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
    multi.stop();

    result.context("indexing failed")
}
