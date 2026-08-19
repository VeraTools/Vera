//! Index construction and maintenance.
//!
//! This module is responsible for:
//! - Orchestrating the indexing pipeline (discover → parse → chunk → embed → store)
//! - Building BM25 full-text indexes via Tantivy
//! - Building vector indexes via sqlite-vec
//! - Incremental update logic (detect changed files, re-index only those)

pub mod freshness;
pub mod pipeline;
pub mod update;

pub use freshness::{IndexFreshness, detect_staleness};
pub use pipeline::{
    FileError, IndexProgress, IndexSummary, index_dir, index_repository,
    index_repository_with_progress,
};
pub use update::{
    UpdateOptions, UpdateProgress, UpdateSummary, content_hash, update_repository,
    update_repository_with_options_and_progress,
    update_repository_with_options_and_progress_and_cancellation, update_repository_with_progress,
};

/// Truncate embedding vectors to at most `max_dim` dimensions.
///
/// This supports Matryoshka-style dimension reduction: Qwen3 embeddings
/// retain good retrieval quality at lower dimensions, and truncation
/// dramatically reduces index storage size.
///
/// Returns the actual stored dimensionality.
pub(crate) fn truncate_embeddings(embeddings: &mut [(String, Vec<f32>)], max_dim: usize) -> usize {
    if max_dim == 0 || embeddings.is_empty() {
        return embeddings.first().map(|(_, v)| v.len()).unwrap_or(0);
    }

    let original_dim = embeddings.first().map(|(_, v)| v.len()).unwrap_or(0);
    if original_dim <= max_dim {
        return original_dim;
    }

    tracing::debug!(
        original_dim,
        truncated_dim = max_dim,
        "truncating embedding vectors"
    );

    for (_, vec) in embeddings.iter_mut() {
        vec.truncate(max_dim);
    }

    max_dim
}
