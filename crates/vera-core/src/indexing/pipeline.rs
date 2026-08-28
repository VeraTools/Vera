//! Indexing pipeline orchestrator.
//!
//! Coordinates file discovery, parsing, chunking, embedding, and storage
//! into a single `index_repository` entry point. Produces an [`IndexSummary`]
//! describing the work performed.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::CancellationToken;
use crate::config::VeraConfig;
use crate::discovery::{self, DiscoveryResult};
use crate::embedding::{
    EmbeddingError, EmbeddingProvider, embed_chunks_concurrent_with_progress_and_cancellation,
};
use crate::indexing::update::{content_hash, detect_language_for_path};
use crate::parsing;
use crate::parsing::references::RawReference;
use crate::parsing::type_relations::RawTypeRelation;
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::{FileIndexState, FileIndexStatus, MetadataStore};
use crate::storage::vector::VectorStore;
use crate::types::{Chunk, Language};

// ── Index summary ────────────────────────────────────────────────────

/// Summary of an indexing run, suitable for display to the user.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexSummary {
    /// Number of source files parsed.
    pub files_parsed: usize,
    /// Number of chunks created from parsed files.
    pub chunks_created: usize,
    /// Number of embedding vectors generated.
    pub embeddings_generated: usize,
    /// Number of binary files skipped.
    pub binary_skipped: usize,
    /// Number of files skipped due to size threshold.
    pub large_skipped: usize,
    /// Relative paths and sizes (bytes) of files skipped due to size threshold.
    pub large_skipped_paths: Vec<(String, u64)>,
    /// Number of files skipped due to permission or read errors.
    pub error_skipped: usize,
    /// Number of successfully indexed files whose parse trees contained errors.
    pub files_with_tree_sitter_errors: usize,
    /// Number of successfully indexed files that fell back to Tier 0 chunking.
    pub files_using_tier0_fallback: usize,
    /// Files that had parse errors (path + error message).
    pub parse_errors: Vec<FileError>,
    /// Wall-clock elapsed time in seconds.
    pub elapsed_secs: f64,
}

/// A file-level error encountered during indexing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FileError {
    pub file_path: String,
    pub error: String,
}

// ── Progress reporting ───────────────────────────────────────────────

/// Progress events emitted during indexing.
#[derive(Debug, Clone)]
pub enum IndexProgress {
    /// File discovery complete.
    DiscoveryDone { file_count: usize },
    /// Parsing and chunking complete.
    ParsingDone { chunk_count: usize },
    /// An embedding batch finished. `done` is cumulative chunks embedded so far.
    EmbeddingProgress { done: usize, total: usize },
    /// All embeddings generated.
    EmbeddingDone { count: usize },
    /// Index artifacts written to disk.
    StorageDone,
}

// ── Index directory layout ───────────────────────────────────────────

/// Default index directory name (placed inside the indexed repo).
pub(crate) const INDEX_DIR_NAME: &str = ".vera";

const INDEX_BUILD_SUFFIX: &str = "build";
const INDEX_OLD_SUFFIX: &str = "old";
pub(crate) const INDEX_STAGING_SUFFIXES: [&str; 2] = [INDEX_BUILD_SUFFIX, INDEX_OLD_SUFFIX];

/// Subdirectory for BM25 (Tantivy) index files.
const BM25_SUBDIR: &str = "bm25";

/// Filename for SQLite metadata + vector databases.
const METADATA_DB: &str = "metadata.db";
const VECTOR_DB: &str = "vectors.db";

/// Maximum number of parsed chunks held by the full-index pipeline at once.
///
/// The actual bound is approximate when a parse group or a single file
/// produces more chunks than the target. With the default embedding byte
/// limit, this bounds live chunk text to roughly `target * max_chunk_bytes`.
const WINDOW_CHUNK_TARGET: usize = 2048;

/// Keep each rayon parse group bounded while a window is being assembled.
const MAX_PARSE_FILE_GROUP: usize = 64;

/// Resolve the index directory for a given repository root.
pub fn index_dir(repo_root: &Path) -> std::path::PathBuf {
    repo_root.join(INDEX_DIR_NAME)
}

/// True when `path` lives inside the live index directory or one of the
/// staging siblings (`.vera.build`, `.vera.old`) a build swaps in and out.
/// Watchers and file discovery must treat all three as internal artifacts:
/// reacting to staging writes re-triggers watchers, and indexing them
/// duplicates index content as source.
pub fn path_in_index_artifacts(idx_dir: &Path, path: &Path) -> bool {
    path.starts_with(idx_dir)
        || INDEX_STAGING_SUFFIXES
            .iter()
            .any(|suffix| path.starts_with(sibling_index_dir(idx_dir, suffix)))
}

// ── Pipeline entry point ─────────────────────────────────────────────

/// Index a repository: discover files, parse, chunk, embed, and store.
///
/// This is the main orchestrator for `vera index <path>`. It:
/// 1. Validates the input path
/// 2. Discovers source files (respecting .gitignore and exclusions)
/// 3. Parses and chunks each file
/// 4. Generates embeddings via the provider
/// 5. Stores metadata, vectors, and BM25 index on disk
///
/// # Arguments
/// - `repo_path` — Path to the repository to index
/// - `provider` — Embedding provider (API-backed or mock)
/// - `config` — Pipeline configuration
///
/// # Errors
/// Returns an error if the path is invalid, not a directory, or storage fails.
pub async fn index_repository<P: EmbeddingProvider>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
) -> Result<IndexSummary> {
    index_repository_with_cancellation(
        repo_path,
        provider,
        config,
        model_name,
        &CancellationToken::new(),
    )
    .await
}

/// Index a repository while cooperatively observing cancellation.
pub async fn index_repository_with_cancellation<P: EmbeddingProvider>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
    cancellation: &CancellationToken,
) -> Result<IndexSummary> {
    index_repository_with_progress_and_cancellation(
        repo_path,
        provider,
        config,
        model_name,
        |_| {},
        cancellation,
    )
    .await
}

/// Index a repository and report progress to the supplied callback.
pub async fn index_repository_with_progress<P, F>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
    on_progress: F,
) -> Result<IndexSummary>
where
    P: EmbeddingProvider,
    F: Fn(IndexProgress) + Send + Sync,
{
    index_repository_with_progress_and_cancellation(
        repo_path,
        provider,
        config,
        model_name,
        on_progress,
        &CancellationToken::new(),
    )
    .await
}

/// Index a repository with progress reporting and cooperative cancellation.
pub async fn index_repository_with_progress_and_cancellation<P, F>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
    on_progress: F,
    cancellation: &CancellationToken,
) -> Result<IndexSummary>
where
    P: EmbeddingProvider,
    F: Fn(IndexProgress) + Send + Sync,
{
    index_repository_with_progress_and_cancellation_with_window_target(
        repo_path,
        provider,
        config,
        model_name,
        on_progress,
        cancellation,
        WINDOW_CHUNK_TARGET,
    )
    .await
}

/// Test seam for running the full-index pipeline with a smaller chunk window.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn index_repository_with_progress_and_cancellation_with_window_target<P, F>(
    repo_path: &Path,
    provider: &P,
    config: &VeraConfig,
    model_name: &str,
    on_progress: F,
    cancellation: &CancellationToken,
    window_chunk_target: usize,
) -> Result<IndexSummary>
where
    P: EmbeddingProvider,
    F: Fn(IndexProgress) + Send + Sync,
{
    let start = Instant::now();
    cancellation.check()?;
    let window_chunk_target = window_chunk_target.max(1);

    // ── 1. Validate path ─────────────────────────────────────────
    if !repo_path.exists() {
        bail!("path does not exist: {}", repo_path.display());
    }
    if !repo_path.is_dir() {
        bail!("path is not a directory: {}", repo_path.display());
    }

    let repo_root = repo_path
        .canonicalize()
        .with_context(|| format!("failed to resolve path: {}", repo_path.display()))?;

    info!(path = %repo_root.display(), "starting indexing");

    let idx_dir = index_dir(&repo_root);
    // Serialize concurrent writers for this repo. The lock is held for the
    // entire build so a concurrent `reuse_index` check sees a held lock and
    // refuses to reuse a half-written live index. `flock` releases
    // automatically on process exit, so a crashed writer cannot leave a stale
    // lock that blocks forever (unlike a plain `.lock` file).
    let _index_lock = crate::indexing::lock::IndexLock::acquire_blocking_for_index_dir(&idx_dir)
        .context("failed to acquire index lock")?;
    recover_index_directories(&idx_dir).context("failed to recover index directories")?;

    // ── 2. Discover files ────────────────────────────────────────
    let discovery =
        discovery::discover_files_with_cancellation(&repo_root, &config.indexing, cancellation)
            .context("file discovery failed")?;

    if discovery.files.is_empty() {
        return Ok(IndexSummary {
            files_parsed: 0,
            chunks_created: 0,
            embeddings_generated: 0,
            binary_skipped: discovery.binary_skipped,
            large_skipped: discovery.large_skipped,
            large_skipped_paths: discovery.large_skipped_paths.clone(),
            error_skipped: discovery.error_skipped,
            files_with_tree_sitter_errors: 0,
            files_using_tier0_fallback: 0,
            parse_errors: Vec::new(),
            elapsed_secs: start.elapsed().as_secs_f64(),
        });
    }

    info!(
        files = discovery.files.len(),
        binary_skipped = discovery.binary_skipped,
        large_skipped = discovery.large_skipped,
        error_skipped = discovery.error_skipped,
        "file discovery complete"
    );
    on_progress(IndexProgress::DiscoveryDone {
        file_count: discovery.files.len(),
    });

    // Build into a sibling directory. The guard removes it on every error or
    // cancellation, leaving the previous live index untouched until swap.
    let mut staging = StagingIndex::new(&idx_dir).context("failed to create staging index")?;

    // Store windows on a dedicated blocking thread. Parsing, embedding, and
    // storing are each CPU-heavy, so serializing stores behind embedding
    // idles whole stages at every window boundary. The worker owns the
    // staging stores, applies windows in send order, and reports back through
    // the Finish command, so store order and the crash contract are
    // unchanged: a failed build discards the staging directory either way.
    let mut stores = StoreHandle::spawn(
        staging.build_dir.clone(),
        model_name.to_string(),
        provider.document_prefix_identity(),
    );

    // ── 3. Parse, embed, and store bounded windows ───────────────
    let (batch_size, max_concurrent_requests) = config.embedding.bounded_parallelism();
    if batch_size != config.embedding.batch_size
        || max_concurrent_requests != config.embedding.max_concurrent_requests
    {
        info!(
            configured_batch_size = config.embedding.batch_size,
            configured_concurrency = config.embedding.max_concurrent_requests,
            max_in_flight_inputs = config.embedding.max_in_flight_inputs,
            batch_size,
            max_concurrent_requests,
            "clamped embedding parallelism to the in-flight input bound"
        );
    }

    let parse_group_size = window_chunk_target.min(MAX_PARSE_FILE_GROUP);
    let discovery = Arc::new(discovery);
    let repo_root = Arc::new(repo_root);
    let mut parsed_chunk_count = 0;
    let mut embedded_count = 0;
    let mut parse_errors = Vec::new();
    let mut file_hashes = Vec::new();
    let mut file_states = Vec::new();

    // Parse one window ahead of the embed+store stage. Parsing is pure (it
    // only reads source files), so running window N+1 on a blocking thread
    // while window N embeds and stores keeps the CPU busy across window
    // boundaries without changing store order or the crash contract. A build
    // that fails or is cancelled drops the in-flight handle; the detached
    // task only reads sources, observes the cancellation token, and exits.
    let spawn_parse = |start_file_index: usize| {
        let discovery = Arc::clone(&discovery);
        let repo_root = Arc::clone(&repo_root);
        let config = config.clone();
        let cancellation = cancellation.clone();
        tokio::task::spawn_blocking(move || {
            parse_window(
                &discovery,
                start_file_index,
                window_chunk_target,
                parse_group_size,
                &repo_root,
                &config,
                &cancellation,
            )
        })
    };

    let total_files = discovery.files.len();
    let mut parse_ahead = (0 < total_files).then(|| spawn_parse(0));
    while let Some(handle) = parse_ahead.take() {
        cancellation.check()?;
        let window = handle
            .await
            .map_err(|error| anyhow::anyhow!("parse task panicked: {error}"))??;
        let next_file_index = window.next_file_index;
        parse_ahead = (next_file_index < total_files).then(|| spawn_parse(next_file_index));

        parsed_chunk_count += window.chunks.len();
        if next_file_index == total_files {
            info!(
                chunks = parsed_chunk_count,
                parse_errors = window.parse_errors.len() + parse_errors.len(),
                "parsing complete"
            );
            on_progress(IndexProgress::ParsingDone {
                chunk_count: parsed_chunk_count,
            });
        }

        parse_errors.extend(window.parse_errors);
        file_hashes.extend(window.file_hashes);
        file_states.extend(window.file_states.iter().cloned());

        cancellation.check()?;
        if !window.chunks.is_empty() {
            let embedded_before_window = embedded_count;
            let parsed_through_window = parsed_chunk_count;
            let window_batch_size = batch_size.min(window_chunk_target);
            let progress_cb = |done: usize, _total: usize| {
                on_progress(IndexProgress::EmbeddingProgress {
                    done: embedded_before_window + done,
                    total: parsed_through_window,
                });
            };
            let embedding_result = embed_chunks_concurrent_with_progress_and_cancellation(
                provider,
                &window.chunks,
                window_batch_size,
                max_concurrent_requests,
                config.indexing.max_chunk_bytes,
                cancellation.as_async_token(),
                progress_cb,
            )
            .await;
            let mut embeddings = match embedding_result {
                Ok(embeddings) => embeddings,
                Err(error) => {
                    // A completed provider error outranks a simultaneous cancellation,
                    // mirroring the biased select in the CLI's cancel_task_on_signal.
                    if matches!(error, EmbeddingError::Cancelled) {
                        cancellation.check()?;
                    }
                    stores.abort().await;
                    return Err(error).context("embedding generation failed");
                }
            };
            cancellation.check()?;

            let window_dim =
                super::truncate_embeddings(&mut embeddings, config.embedding.max_stored_dim);
            embedded_count += embeddings.len();

            stores
                .send_window(WindowStore {
                    chunks: window.chunks,
                    embeddings,
                    window_dim,
                    file_states: window.file_states,
                    refs: window.refs,
                    type_relations: window.type_relations,
                })
                .await?;
        } else {
            // Parse-error-only windows still need their file state and parse
            // artifacts persisted if a later window contains chunks.
            stores
                .send_window(WindowStore {
                    chunks: Vec::new(),
                    embeddings: Vec::new(),
                    window_dim: 0,
                    file_states: window.file_states,
                    refs: window.refs,
                    type_relations: window.type_relations,
                })
                .await?;
        }

        cancellation.check()?;
    }

    if parsed_chunk_count == 0 {
        stores.abort().await;
        return Ok(IndexSummary {
            files_parsed: discovery.files.len() - parse_errors.len(),
            chunks_created: 0,
            embeddings_generated: 0,
            binary_skipped: discovery.binary_skipped,
            large_skipped: discovery.large_skipped,
            large_skipped_paths: discovery.large_skipped_paths.clone(),
            error_skipped: discovery.error_skipped,
            files_with_tree_sitter_errors: count_tree_sitter_error_files(&file_states),
            files_using_tier0_fallback: count_tier0_fallback_files(&file_states),
            parse_errors,
            elapsed_secs: start.elapsed().as_secs_f64(),
        });
    }

    on_progress(IndexProgress::EmbeddingDone {
        count: embedded_count,
    });

    // Publication is synchronous, so cancellation must win before any artifact is replaced.
    cancellation.check()?;
    // The worker commits the BM25 index, applies the empty-vector
    // dimensionality fallback, and certifies the build before replying.
    stores
        .finish(parsed_chunk_count > 0, file_hashes, config.indexing.clone())
        .await?;

    let files_parsed = discovery.files.len() - parse_errors.len();
    let summary = IndexSummary {
        files_parsed,
        chunks_created: parsed_chunk_count,
        embeddings_generated: embedded_count,
        binary_skipped: discovery.binary_skipped,
        large_skipped: discovery.large_skipped,
        large_skipped_paths: discovery.large_skipped_paths.clone(),
        error_skipped: discovery.error_skipped,
        files_with_tree_sitter_errors: count_tree_sitter_error_files(&file_states),
        files_using_tier0_fallback: count_tier0_fallback_files(&file_states),
        parse_errors,
        elapsed_secs: start.elapsed().as_secs_f64(),
    };

    swap_staging_index(&idx_dir, &staging.build_dir, &staging.old_dir)
        .context("failed to publish staged index")?;
    staging.committed = true;

    info!(index_dir = %idx_dir.display(), "index artifacts written");
    on_progress(IndexProgress::StorageDone);

    Ok(summary)
}

// ── Internal helpers ─────────────────────────────────────────────────

/// One parsed window of the full-index pipeline.
struct ParsedWindow {
    chunks: Vec<Chunk>,
    parse_errors: Vec<FileError>,
    file_hashes: Vec<(String, String)>,
    refs: Vec<(String, Vec<RawReference>)>,
    type_relations: Vec<(String, Vec<RawTypeRelation>)>,
    file_states: Vec<FileIndexState>,
    next_file_index: usize,
}

/// Parse file groups into a window holding at least `window_chunk_target`
/// chunks (bounded above by the target plus one group). Pure with respect to
/// the index: it only reads source files, so it can run ahead of storage on a
/// blocking thread.
fn parse_window(
    discovery: &DiscoveryResult,
    start_file_index: usize,
    window_chunk_target: usize,
    parse_group_size: usize,
    repo_root: &Path,
    config: &VeraConfig,
    cancellation: &CancellationToken,
) -> Result<ParsedWindow> {
    let mut window = ParsedWindow {
        chunks: Vec::new(),
        parse_errors: Vec::new(),
        file_hashes: Vec::new(),
        refs: Vec::new(),
        type_relations: Vec::new(),
        file_states: Vec::new(),
        next_file_index: start_file_index,
    };

    while window.next_file_index < discovery.files.len()
        && (window.next_file_index == start_file_index || window.chunks.len() < window_chunk_target)
    {
        let group_end = (window.next_file_index + parse_group_size).min(discovery.files.len());
        let (
            chunks,
            group_parse_errors,
            group_file_hashes,
            group_refs,
            group_type_relations,
            group_file_states,
        ) = parse_discovered_files_parallel(
            discovery,
            &discovery.files[window.next_file_index..group_end],
            repo_root,
            config,
            cancellation,
        )?;
        window.next_file_index = group_end;
        window.chunks.extend(chunks);
        window.parse_errors.extend(group_parse_errors);
        window.file_hashes.extend(group_file_hashes);
        window.refs.extend(group_refs);
        window.type_relations.extend(group_type_relations);
        window.file_states.extend(group_file_states);
    }

    Ok(window)
}

/// One parsed and embedded window handed to the store worker.
struct WindowStore {
    chunks: Vec<Chunk>,
    embeddings: Vec<(String, Vec<f32>)>,
    window_dim: usize,
    file_states: Vec<FileIndexState>,
    refs: Vec<(String, Vec<RawReference>)>,
    type_relations: Vec<(String, Vec<RawTypeRelation>)>,
}

/// Commands for the staging store worker.
enum StoreCommand {
    Window(WindowStore),
    Finish {
        /// Set when the build parsed chunks but no embedding ever produced a
        /// stored dimension, triggering the empty-vector fallback store.
        create_fallback_vector_store: bool,
        file_hashes: Vec<(String, String)>,
        indexing_config: crate::config::IndexingConfig,
    },
}

/// Sending end of the staging store worker plus its join handle.
///
/// Dropping the handle without `finish` or `abort` closes the channel; the
/// worker drains the bounded queue, observes the hangup, and exits without
/// committing or certifying anything. In-flight work is bounded by the
/// channel capacity, so a dropped worker terminates on its own while the
/// staging guard removes the build directory.
struct StoreHandle {
    tx: tokio::sync::mpsc::Sender<StoreCommand>,
    worker: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl StoreHandle {
    fn spawn(build_dir: PathBuf, model_name: String, document_prefix: String) -> Self {
        // Capacity bounds in-flight windows to one being stored plus one
        // queued, applying backpressure to the embed stage if stores lag.
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let worker = tokio::task::spawn_blocking(move || {
            store_worker(&build_dir, &model_name, &document_prefix, rx)
        });
        Self {
            tx,
            worker: Some(worker),
        }
    }

    async fn send_window(&mut self, job: WindowStore) -> Result<()> {
        if self.tx.send(StoreCommand::Window(job)).await.is_err() {
            // The worker stopped; surface its real error rather than the
            // channel's.
            self.join().await?;
            anyhow::bail!("store worker stopped without reporting an error");
        }
        Ok(())
    }

    async fn finish(
        mut self,
        create_fallback_vector_store: bool,
        file_hashes: Vec<(String, String)>,
        indexing_config: crate::config::IndexingConfig,
    ) -> Result<()> {
        self.tx
            .send(StoreCommand::Finish {
                create_fallback_vector_store,
                file_hashes,
                indexing_config,
            })
            .await
            .map_err(|_| anyhow::anyhow!("store worker stopped before finalization"))?;
        self.join().await
    }

    /// Close the channel and wait for the worker to drain and exit. Used on
    /// error paths; the worker's own result is superseded by the error the
    /// caller is already returning.
    async fn abort(mut self) {
        drop(self.tx);
        if let Some(worker) = self.worker.take() {
            let _ = worker.await;
        }
    }

    async fn join(&mut self) -> Result<()> {
        if let Some(worker) = self.worker.take() {
            worker
                .await
                .map_err(|error| anyhow::anyhow!("store worker panicked: {error}"))??;
        }
        Ok(())
    }
}

/// Apply parsed and embedded windows to the staging stores in order.
///
/// Owns every staging store so window storage overlaps parsing and embedding
/// on other threads. Replies through the join handle only at `Finish` — after
/// the single BM25 commit and index certification — or at the first failure.
/// A channel hangup without `Finish` means the build was abandoned; nothing
/// is committed and the staging guard removes the directory.
fn store_worker(
    build_dir: &Path,
    model_name: &str,
    document_prefix: &str,
    mut rx: tokio::sync::mpsc::Receiver<StoreCommand>,
) -> Result<()> {
    let metadata_store = MetadataStore::open(&build_dir.join(METADATA_DB))
        .context("failed to open staging metadata store")?;
    metadata_store
        .set_index_meta("model_name", model_name)
        .context("failed to store model_name")?;
    metadata_store
        .set_index_meta("document_prefix", document_prefix)
        .context("failed to store document_prefix")?;
    let bm25_index = Bm25Index::open(&build_dir.join(BM25_SUBDIR))
        .context("failed to open staging BM25 index")?;
    // One writer for the whole build: a writer per window would pay creation,
    // commit, and a merge-thread join for every window and serialize segment
    // merges with embedding.
    let bm25_writer = bm25_index
        .begin_bulk_build()
        .context("failed to open bulk BM25 writer")?;
    let mut vector_store = None;
    let mut stored_dim = None;

    while let Some(command) = rx.blocking_recv() {
        match command {
            StoreCommand::Window(job) => {
                if job.chunks.is_empty() {
                    // Parse-error-only windows still need their file state and
                    // parse artifacts persisted if a later window has chunks.
                    metadata_store
                        .insert_file_states(&job.file_states)
                        .context("failed to store file index states")?;
                    metadata_store
                        .insert_parse_artifacts_batch(&job.refs, &job.type_relations)
                        .context("failed to store references and type relations")?;
                    continue;
                }
                metadata_store
                    .insert_chunks(&job.chunks)
                    .context("failed to insert chunk metadata")?;
                metadata_store
                    .insert_file_states(&job.file_states)
                    .context("failed to store file index states")?;
                metadata_store
                    .insert_parse_artifacts_batch(&job.refs, &job.type_relations)
                    .context("failed to store references and type relations")?;

                if !job.embeddings.is_empty() {
                    if let Some(existing_dim) = stored_dim {
                        anyhow::ensure!(
                            existing_dim == job.window_dim,
                            "embedding dimension changed between windows: expected {}, got {}",
                            existing_dim,
                            job.window_dim
                        );
                    } else {
                        let store = VectorStore::open(&build_dir.join(VECTOR_DB), job.window_dim)
                            .context("failed to open staging vector store")?;
                        metadata_store
                            .set_index_meta("embedding_dim", &job.window_dim.to_string())
                            .context("failed to store embedding_dim")?;
                        vector_store = Some(store);
                        stored_dim = Some(job.window_dim);
                    }
                }
                if let Some(vector_store) = vector_store.as_ref() {
                    let batch: Vec<(&str, &[f32])> = job
                        .embeddings
                        .iter()
                        .map(|(id, vector)| (id.as_str(), vector.as_slice()))
                        .collect();
                    // The staging store is freshly created, so every chunk id
                    // is new and the re-insert defenses of insert_batch
                    // cannot match.
                    vector_store
                        .insert_batch_fresh(&batch)
                        .context("failed to insert vectors")?;
                }
                bm25_writer
                    .insert_chunks(&job.chunks)
                    .context("failed to insert BM25 documents")?;
            }
            StoreCommand::Finish {
                create_fallback_vector_store,
                file_hashes,
                indexing_config,
            } => {
                // A provider returning no vectors is not expected, but
                // preserve the empty-vector dimensionality fallback for a
                // non-empty parsed corpus.
                if create_fallback_vector_store && stored_dim.is_none() {
                    let fallback_dim = 4096;
                    let _store = VectorStore::open(&build_dir.join(VECTOR_DB), fallback_dim)
                        .context("failed to open staging vector store")?;
                    metadata_store
                        .set_index_meta("embedding_dim", &fallback_dim.to_string())
                        .context("failed to store embedding_dim")?;
                }
                // Commit the BM25 index once for the whole build.
                // Certification stays last: it attests that every staged
                // insert, BM25 included, landed.
                bm25_writer
                    .finish()
                    .context("failed to commit BM25 index")?;
                publish_index_certification(&metadata_store, &file_hashes, &indexing_config)
                    .context("failed to publish index freshness metadata")?;
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Parse all discovered files in parallel using rayon and collect chunks.
///
/// Each file is read and parsed on a rayon thread pool worker. Results
/// are collected and flattened. Files that fail parsing are recorded as
/// errors but do not abort the pipeline. Also computes content hashes
/// for incremental indexing support.
#[allow(clippy::type_complexity)]
fn parse_discovered_files_parallel(
    discovery: &DiscoveryResult,
    files: &[discovery::DiscoveredFile],
    repo_root: &Path,
    config: &VeraConfig,
    cancellation: &CancellationToken,
) -> Result<(
    Vec<Chunk>,
    Vec<FileError>,
    Vec<(String, String)>,
    Vec<(String, Vec<RawReference>)>,
    Vec<(String, Vec<RawTypeRelation>)>,
    Vec<FileIndexState>,
)> {
    let config = Arc::new(config.clone());
    let repo_root = Arc::new(repo_root.to_path_buf());

    struct ParsedFileResult {
        chunks: Vec<Chunk>,
        parse_error: Option<FileError>,
        file_hash: Option<(String, String)>,
        refs: Option<(String, Vec<RawReference>)>,
        type_relations: Option<(String, Vec<RawTypeRelation>)>,
        file_state: Option<FileIndexState>,
    }

    let results: Vec<ParsedFileResult> = files
        .par_iter()
        .map(|file| {
            if cancellation.is_cancelled() {
                return ParsedFileResult {
                    chunks: Vec::new(),
                    parse_error: None,
                    file_hash: None,
                    refs: None,
                    type_relations: None,
                    file_state: None,
                };
            }

            let source = match crate::discovery::read_source_lossy_at(
                &discovery.root_dir,
                Path::new(&file.relative_path),
            ) {
                Ok(source) => source,
                Err(err) => {
                    warn!(
                        file = %file.relative_path,
                        error = %err,
                        "failed to read file for parsing"
                    );
                    return ParsedFileResult {
                        chunks: Vec::new(),
                        parse_error: Some(FileError {
                            file_path: file.relative_path.clone(),
                            error: err.to_string(),
                        }),
                        file_hash: None,
                        refs: None,
                        type_relations: None,
                        file_state: None,
                    };
                }
            };

            let language = detect_language_for_path(&file.absolute_path);

            // RST files need preprocessing before chunking, but refs
            // come from the raw source, so they can't share a single parse.
            // The hash is computed before the parse so the error branch can
            // store the hash of the source that was actually attempted;
            // otherwise a parse-failing RST file looks modified on every
            // update and is re-parsed forever.
            let hash;
            let parsed = if language == Language::Rst {
                let refs = parsing::parse_and_extract_references(&source, language);
                let normalized_source = match parsing::sphinx::preprocess_rst_with_limit(
                    &source,
                    &file.absolute_path,
                    repo_root.as_path(),
                    config.indexing.max_file_size_bytes,
                ) {
                    Ok(preprocessed) => Some(preprocessed),
                    Err(err) => {
                        warn!(
                            file = %file.relative_path,
                            error = %err,
                            "failed to preprocess rst; falling back to raw source"
                        );
                        None
                    }
                };
                let src = normalized_source.as_deref().unwrap_or(&source);
                hash = content_hash(src);
                parsing::parse_file_with_diagnostics(
                    src,
                    &file.relative_path,
                    language,
                    &config.indexing,
                )
                .map(|(chunks, _ignored_refs, diagnostics)| (chunks, refs, diagnostics))
            } else {
                hash = content_hash(&source);
                parsing::parse_file_with_diagnostics(
                    &source,
                    &file.relative_path,
                    language,
                    &config.indexing,
                )
            };

            match parsed {
                Ok((chunks, refs, diagnostics)) => {
                    let chunk_count = chunks.len() as u64;
                    let type_relations = parsing::type_relations::extract_type_relations(&chunks);
                    debug!(
                        file = %file.relative_path,
                        chunks = chunk_count,
                        refs = refs.len(),
                        type_relations = type_relations.len(),
                        "parsed file"
                    );
                    ParsedFileResult {
                        chunks,
                        parse_error: None,
                        file_hash: Some((file.relative_path.clone(), hash)),
                        refs: (!refs.is_empty()).then_some((file.relative_path.clone(), refs)),
                        type_relations: (!type_relations.is_empty())
                            .then_some((file.relative_path.clone(), type_relations)),
                        file_state: Some(FileIndexState {
                            file_path: file.relative_path.clone(),
                            language: language.to_string(),
                            status: FileIndexStatus::Indexed,
                            tree_has_error: diagnostics.tree_has_error,
                            tier0_fallback: diagnostics.used_tier0_fallback,
                            chunk_count,
                        }),
                    }
                }
                Err(err) => {
                    warn!(
                        file = %file.relative_path,
                        error = %err,
                        "parse error"
                    );
                    ParsedFileResult {
                        chunks: Vec::new(),
                        parse_error: Some(FileError {
                            file_path: file.relative_path.clone(),
                            error: err.to_string(),
                        }),
                        file_hash: Some((file.relative_path.clone(), hash)),
                        refs: None,
                        type_relations: None,
                        file_state: Some(FileIndexState {
                            file_path: file.relative_path.clone(),
                            language: language.to_string(),
                            status: FileIndexStatus::ParseError,
                            tree_has_error: false,
                            tier0_fallback: false,
                            chunk_count: 0,
                        }),
                    }
                }
            }
        })
        .collect();

    cancellation.check()?;

    // Flatten results into chunks, errors, file hashes, and references.
    let mut all_chunks = Vec::new();
    let mut parse_errors = Vec::new();
    let mut file_hashes = Vec::new();
    let mut all_refs = Vec::new();
    let mut all_type_relations = Vec::new();
    let mut file_states = Vec::new();
    for result in results {
        all_chunks.extend(result.chunks);
        if let Some(error) = result.parse_error {
            parse_errors.push(error);
        }
        if let Some(file_hash) = result.file_hash {
            file_hashes.push(file_hash);
        }
        if let Some(file_refs) = result.refs {
            all_refs.push(file_refs);
        }
        if let Some(type_relations) = result.type_relations {
            all_type_relations.push(type_relations);
        }
        if let Some(file_state) = result.file_state {
            file_states.push(file_state);
        }
    }

    Ok((
        all_chunks,
        parse_errors,
        file_hashes,
        all_refs,
        all_type_relations,
        file_states,
    ))
}

struct StagingIndex {
    build_dir: PathBuf,
    old_dir: PathBuf,
    committed: bool,
}

impl StagingIndex {
    fn new(idx_dir: &Path) -> Result<Self> {
        let build_dir = sibling_index_dir(idx_dir, INDEX_BUILD_SUFFIX);
        let old_dir = sibling_index_dir(idx_dir, INDEX_OLD_SUFFIX);
        if build_dir.exists() {
            std::fs::remove_dir_all(&build_dir).with_context(|| {
                format!(
                    "failed to remove stale staging dir: {}",
                    build_dir.display()
                )
            })?;
        }
        std::fs::create_dir_all(&build_dir)
            .with_context(|| format!("failed to create staging dir: {}", build_dir.display()))?;
        Ok(Self {
            build_dir,
            old_dir,
            committed: false,
        })
    }
}

impl Drop for StagingIndex {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_dir_all(&self.build_dir);
        }
    }
}

fn sibling_index_dir(idx_dir: &Path, suffix: &str) -> PathBuf {
    let name = idx_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(INDEX_DIR_NAME);
    idx_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{name}.{suffix}"))
}

fn recover_index_directories(idx_dir: &Path) -> Result<()> {
    let build_dir = sibling_index_dir(idx_dir, INDEX_BUILD_SUFFIX);
    let old_dir = sibling_index_dir(idx_dir, INDEX_OLD_SUFFIX);
    if !idx_dir.exists() && old_dir.exists() {
        std::fs::rename(&old_dir, idx_dir).with_context(|| {
            format!(
                "failed to restore previous index from {}",
                old_dir.display()
            )
        })?;
    }
    if build_dir.exists() {
        std::fs::remove_dir_all(&build_dir).with_context(|| {
            format!(
                "failed to remove stale staging dir: {}",
                build_dir.display()
            )
        })?;
    }
    if old_dir.exists() {
        std::fs::remove_dir_all(&old_dir)
            .with_context(|| format!("failed to remove stale old index: {}", old_dir.display()))?;
    }
    Ok(())
}

fn swap_staging_index(idx_dir: &Path, build_dir: &Path, old_dir: &Path) -> Result<()> {
    if old_dir.exists() {
        std::fs::remove_dir_all(old_dir)
            .with_context(|| format!("failed to remove old index: {}", old_dir.display()))?;
    }
    if idx_dir.exists() {
        std::fs::rename(idx_dir, old_dir).with_context(|| {
            format!(
                "failed to move live index to temporary path: {}",
                old_dir.display()
            )
        })?;
    }
    if let Err(error) = std::fs::rename(build_dir, idx_dir) {
        if old_dir.exists() && !idx_dir.exists() {
            let _ = std::fs::rename(old_dir, idx_dir);
        }
        return Err(error)
            .with_context(|| format!("failed to publish staging index as {}", idx_dir.display()));
    }
    if old_dir.exists() {
        std::fs::remove_dir_all(old_dir)
            .with_context(|| format!("failed to remove old index: {}", old_dir.display()))?;
    }
    Ok(())
}

fn publish_index_certification(
    metadata_store: &MetadataStore,
    file_hashes: &[(String, String)],
    indexing_config: &crate::config::IndexingConfig,
) -> Result<()> {
    // Hashes and the freshness stamp certify the index current, so they are
    // written only after every staged metadata, vector, and BM25 insert has
    // completed. A failure before this publication leaves the staging build
    // uncertified and the previous live index untouched. The swap happens
    // only after these writes succeed.
    metadata_store
        .set_file_hashes_batch(file_hashes)
        .context("failed to store file hashes")?;
    super::freshness::record_index_snapshot(metadata_store, indexing_config)
        .context("failed to store index freshness metadata")?;
    Ok(())
}

pub(crate) fn count_tree_sitter_error_files(file_states: &[FileIndexState]) -> usize {
    file_states
        .iter()
        .filter(|state| state.status == FileIndexStatus::Indexed && state.tree_has_error)
        .count()
}

pub(crate) fn count_tier0_fallback_files(file_states: &[FileIndexState]) -> usize {
    file_states
        .iter()
        .filter(|state| state.status == FileIndexStatus::Indexed && state.tier0_fallback)
        .count()
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
