//! Indexing pipeline orchestrator.
//!
//! Coordinates file discovery, parsing, chunking, embedding, and storage
//! into a single `index_repository` entry point. Produces an [`IndexSummary`]
//! describing the work performed.

use std::path::Path;
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
use crate::indexing::update::content_hash;
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
const INDEX_DIR_NAME: &str = ".vera";

/// Subdirectory for BM25 (Tantivy) index files.
const BM25_SUBDIR: &str = "bm25";

/// Filename for SQLite metadata + vector databases.
const METADATA_DB: &str = "metadata.db";
const VECTOR_DB: &str = "vectors.db";

/// Resolve the index directory for a given repository root.
pub fn index_dir(repo_root: &Path) -> std::path::PathBuf {
    repo_root.join(INDEX_DIR_NAME)
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
    let start = Instant::now();
    cancellation.check()?;

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

    // ── 3. Parse and chunk each file (parallelized with rayon) ──
    let (all_chunks, parse_errors, file_hashes, all_refs, all_type_relations, file_states) =
        parse_discovered_files_parallel(&discovery, &repo_root, config, cancellation)?;

    info!(
        chunks = all_chunks.len(),
        parse_errors = parse_errors.len(),
        "parsing complete"
    );
    on_progress(IndexProgress::ParsingDone {
        chunk_count: all_chunks.len(),
    });

    if all_chunks.is_empty() {
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

    // ── 4. Generate embeddings (concurrent batches) ──────────────
    cancellation.check()?;
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

    let progress_cb = |done: usize, total: usize| {
        on_progress(IndexProgress::EmbeddingProgress { done, total });
    };
    let embedding_result = embed_chunks_concurrent_with_progress_and_cancellation(
        provider,
        &all_chunks,
        batch_size,
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
            return Err(error).context("embedding generation failed");
        }
    };
    cancellation.check()?;

    // Truncate vectors if max_stored_dim is configured.
    let stored_dim = super::truncate_embeddings(&mut embeddings, config.embedding.max_stored_dim);

    info!(
        embeddings = embeddings.len(),
        stored_dim, "embeddings generated"
    );
    on_progress(IndexProgress::EmbeddingDone {
        count: embeddings.len(),
    });

    // ── 5. Store everything on disk ──────────────────────────────
    // Publication is synchronous, so cancellation must win before any artifact is replaced.
    cancellation.check()?;
    let idx_dir = index_dir(&repo_root);
    store_index(
        &idx_dir,
        &all_chunks,
        &embeddings,
        &file_hashes,
        &all_refs,
        &all_type_relations,
        IndexBuildMetadata {
            file_states: &file_states,
            indexing_config: &config.indexing,
            model_name,
        },
    )
    .context("failed to write index artifacts")?;

    info!(index_dir = %idx_dir.display(), "index artifacts written");
    on_progress(IndexProgress::StorageDone);

    let files_parsed = discovery.files.len() - parse_errors.len();

    Ok(IndexSummary {
        files_parsed,
        chunks_created: all_chunks.len(),
        embeddings_generated: embeddings.len(),
        binary_skipped: discovery.binary_skipped,
        large_skipped: discovery.large_skipped,
        large_skipped_paths: discovery.large_skipped_paths,
        error_skipped: discovery.error_skipped,
        files_with_tree_sitter_errors: count_tree_sitter_error_files(&file_states),
        files_using_tier0_fallback: count_tier0_fallback_files(&file_states),
        parse_errors,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Parse all discovered files in parallel using rayon and collect chunks.
///
/// Each file is read and parsed on a rayon thread pool worker. Results
/// are collected and flattened. Files that fail parsing are recorded as
/// errors but do not abort the pipeline. Also computes content hashes
/// for incremental indexing support.
#[allow(clippy::type_complexity)]
fn parse_discovered_files_parallel(
    discovery: &DiscoveryResult,
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

    let results: Vec<ParsedFileResult> = discovery
        .files
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

            let source = match crate::discovery::read_source_lossy(&file.absolute_path) {
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

            let language = file
                .absolute_path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(Language::from_filename)
                .unwrap_or_else(|| {
                    let ext = file
                        .absolute_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    Language::from_extension(ext)
                });

            // RST files need preprocessing before chunking, but refs
            // come from the raw source, so they can't share a single parse.
            let parsed = if language == Language::Rst {
                let refs = parsing::parse_and_extract_references(&source, language);
                let normalized_source = match parsing::sphinx::preprocess_rst(
                    &source,
                    &file.absolute_path,
                    repo_root.as_path(),
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
                let hash = content_hash(src);
                parsing::parse_file_with_diagnostics(
                    src,
                    &file.relative_path,
                    language,
                    &config.indexing,
                )
                .map(|(chunks, _ignored_refs, diagnostics)| (chunks, refs, hash, diagnostics))
            } else {
                let hash = content_hash(&source);
                parsing::parse_file_with_diagnostics(
                    &source,
                    &file.relative_path,
                    language,
                    &config.indexing,
                )
                .map(|(chunks, refs, diagnostics)| (chunks, refs, hash, diagnostics))
            };

            match parsed {
                Ok((chunks, refs, hash, diagnostics)) => {
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
                        file_hash: Some((file.relative_path.clone(), content_hash(&source))),
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

/// Write chunks, embeddings, BM25 index, file hashes, and references to disk.
struct IndexBuildMetadata<'a> {
    file_states: &'a [FileIndexState],
    indexing_config: &'a crate::config::IndexingConfig,
    model_name: &'a str,
}

fn store_index(
    idx_dir: &Path,
    chunks: &[Chunk],
    embeddings: &[(String, Vec<f32>)],
    file_hashes: &[(String, String)],
    file_refs: &[(String, Vec<RawReference>)],
    file_type_relations: &[(String, Vec<RawTypeRelation>)],
    metadata: IndexBuildMetadata<'_>,
) -> Result<()> {
    // Ensure index directory exists.
    std::fs::create_dir_all(idx_dir)
        .with_context(|| format!("failed to create index dir: {}", idx_dir.display()))?;

    // Determine vector dimensionality from the first embedding.
    let dim = embeddings.first().map(|(_, v)| v.len()).unwrap_or(4096);

    // ── Metadata store ───────────────────────────────────────────
    let metadata_path = idx_dir.join(METADATA_DB);
    let metadata_store =
        MetadataStore::open(&metadata_path).context("failed to open metadata store")?;
    // Clear previous data (fresh index).
    metadata_store
        .clear()
        .context("failed to clear metadata store")?;
    metadata_store
        .insert_chunks(chunks)
        .context("failed to insert chunk metadata")?;

    metadata_store
        .insert_file_states(metadata.file_states)
        .context("failed to store file index states")?;

    // Store call-site references and type relations for call graph analysis,
    // batched into a single transaction instead of up to two commits per file.
    metadata_store
        .insert_parse_artifacts_batch(file_refs, file_type_relations)
        .context("failed to store references and type relations")?;

    metadata_store
        .set_index_meta("model_name", metadata.model_name)
        .context("failed to store model_name")?;
    metadata_store
        .set_index_meta("embedding_dim", &dim.to_string())
        .context("failed to store embedding_dim")?;

    debug!(chunks = chunks.len(), "metadata stored");

    // ── Vector store ─────────────────────────────────────────────
    let vector_path = idx_dir.join(VECTOR_DB);
    if vector_path.exists() {
        std::fs::remove_file(&vector_path)
            .with_context(|| format!("failed to reset vector db: {}", vector_path.display()))?;
    }
    let vector_store =
        VectorStore::open(&vector_path, dim).context("failed to open vector store")?;
    vector_store
        .clear()
        .context("failed to clear vector store")?;

    let batch: Vec<(&str, &[f32])> = embeddings
        .iter()
        .map(|(id, vec)| (id.as_str(), vec.as_slice()))
        .collect();
    vector_store
        .insert_batch(&batch)
        .context("failed to insert vectors")?;

    debug!(vectors = embeddings.len(), "vectors stored");

    // ── BM25 index ───────────────────────────────────────────────
    let bm25_dir = idx_dir.join(BM25_SUBDIR);
    if bm25_dir.exists() {
        std::fs::remove_dir_all(&bm25_dir)
            .with_context(|| format!("failed to reset BM25 dir: {}", bm25_dir.display()))?;
    }
    let bm25_index = Bm25Index::open(&bm25_dir).context("failed to open BM25 index")?;

    bm25_index
        .insert_chunks(chunks)
        .context("failed to insert BM25 documents")?;

    debug!(docs = chunks.len(), "BM25 index built");

    // Hashes and the freshness stamp are what certify the index current, so
    // they are written only once every store has been rebuilt. Both stores are
    // deleted and repopulated above; committing the hashes first would leave a
    // crash or a Ctrl-C in that window with a complete metadata store, current
    // hashes and a missing or half-built vector/BM25 store, which
    // `detect_staleness` cannot see and `vera update` skips. `metadata_store`
    // was cleared above, hashes and index metadata included, so a failure
    // before final publication leaves both absent, every file reads as new and
    // the next run reprocesses. A failure during the final metadata writes can
    // happen only after both stores are complete. `update.rs` keeps the same
    // order.
    metadata_store
        .set_file_hashes_batch(file_hashes)
        .context("failed to store file hashes")?;
    super::freshness::record_index_snapshot(&metadata_store, metadata.indexing_config)
        .context("failed to store index freshness metadata")?;

    Ok(())
}

fn count_tree_sitter_error_files(file_states: &[FileIndexState]) -> usize {
    file_states
        .iter()
        .filter(|state| state.status == FileIndexStatus::Indexed && state.tree_has_error)
        .count()
}

fn count_tier0_fallback_files(file_states: &[FileIndexState]) -> usize {
    file_states
        .iter()
        .filter(|state| state.status == FileIndexStatus::Indexed && state.tier0_fallback)
        .count()
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
