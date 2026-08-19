//! Hybrid retrieval engine combining BM25 and vector search via RRF fusion.
//!
//! Runs both BM25 keyword search and vector similarity search in parallel,
//! then merges the results using Reciprocal Rank Fusion (RRF). Items appearing
//! in both result sets rank higher than single-source results.
//!
//! RRF score: `score = sum(1 / (k + rank_i))` where `k` is a constant
//! (typically 60) and `rank_i` is the 1-based rank of the item in each
//! source result list.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::embedding::EmbeddingProvider;
use crate::retrieval::bm25::search_bm25_with_stores_and_filters;
use crate::retrieval::graph_augmentation::augment_pool;
use crate::retrieval::query_classifier::{QueryType, classify_query, params_for_query_type};
use crate::retrieval::query_utils::result_key;
use crate::retrieval::ranking::is_path_weighted_query;
use crate::retrieval::reranker::{Reranker, rerank_results};
use crate::retrieval::vector::{VectorSearchError, search_vector_with_stores};
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;
use crate::types::{SearchFilters, SearchResult};

/// Errors specific to hybrid search.
#[derive(Debug, thiserror::Error)]
pub enum HybridSearchError {
    /// Both BM25 and vector search failed.
    #[error("both BM25 and vector search failed: bm25={bm25_error}, vector={vector_error}")]
    BothFailed {
        bm25_error: String,
        vector_error: String,
    },

    /// Storage or pipeline error.
    #[error("{0}")]
    PipelineError(#[from] anyhow::Error),
}

/// Compute the number of vector candidates to fetch for a given limit and
/// query type multiplier. Ensures at least 50 candidates for any limit.
pub fn compute_vector_candidates(limit: usize, multiplier: usize) -> usize {
    limit.saturating_mul(multiplier).max(50)
}

fn compute_bm25_candidates(query: &str, limit: usize) -> usize {
    let query_type = classify_query(query);
    let token_count = query.split_whitespace().count();

    if is_path_weighted_query(query) {
        return limit.saturating_mul(5).max(100);
    }
    if query_type == QueryType::NaturalLanguage {
        return limit.saturating_mul(4).max(limit + 40);
    }
    if token_count <= 2 {
        return limit.saturating_mul(4).max(80);
    }

    limit.saturating_mul(3).max(limit + 20)
}

/// Per-stage timing data from hybrid search.
#[derive(Debug, Default)]
pub struct HybridTimings {
    pub embedding: Option<Duration>,
    pub bm25: Option<Duration>,
    pub vector: Option<Duration>,
    pub fusion: Option<Duration>,
    pub reranking: Option<Duration>,
}

/// Perform hybrid search combining BM25 and vector retrieval via RRF fusion.
///
/// Runs BM25 and vector search concurrently, then merges the results using
/// Reciprocal Rank Fusion (RRF). BM25 runs in a blocking task with its own
/// database connections while vector search (embedding + nearest-neighbor)
/// runs on the async runtime. If vector search fails (e.g., embedding API
/// unavailable), falls back to BM25-only results with a warning.
#[allow(clippy::too_many_arguments)]
pub async fn search_hybrid(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    vector_candidates: usize,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    let query_type = classify_query(bm25_query);
    let query_params = params_for_query_type(query_type);
    let bm25_candidates = compute_bm25_candidates(bm25_query, limit);
    let mut timings = HybridTimings::default();

    // Spawn BM25 search in a blocking task with its own database connections.
    // This runs concurrently with vector search (embedding + nearest-neighbor).
    let bm25_dir = index_dir.join("bm25");
    let metadata_path = index_dir.join("metadata.db");
    let bm25_query = bm25_query.to_string();
    let bm25_filters = filters.clone();
    let bm25_handle = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let result = Bm25Index::open(&bm25_dir)
            .context("failed to open BM25 index for search")
            .and_then(|index| {
                let store = MetadataStore::open(&metadata_path)
                    .context("failed to open metadata store for BM25 search")?;
                search_bm25_with_stores_and_filters(
                    &index,
                    &store,
                    &bm25_query,
                    &bm25_filters,
                    bm25_candidates,
                )
            });
        (result, start.elapsed())
    });

    // Run vector search concurrently on the async runtime.
    let vector_metadata_path = index_dir.join("metadata.db");
    // Filters are applied after the kNN fetch, so a selective filter can
    // discard every hit in the default window. Over-fetch when filters are
    // active so filtered chunks beyond the window can still reach fusion.
    let vector_fetch = if filters.is_empty() {
        vector_candidates
    } else {
        vector_candidates.saturating_mul(4).max(200)
    };
    let embed_start = Instant::now();
    let vector_results = match VectorStore::open(&index_dir.join("vectors.db"), stored_dim) {
        Ok(vector_store) => {
            let vector_store_result =
                MetadataStore::open(&vector_metadata_path).context("failed to open metadata store");
            match vector_store_result {
                Ok(vector_metadata) => {
                    match search_vector_with_stores(
                        &vector_store,
                        &vector_metadata,
                        provider,
                        vector_query,
                        vector_fetch,
                    )
                    .await
                    {
                        Ok(mut results) => {
                            if !filters.is_empty() {
                                results.retain(|result| filters.matches(result));
                            }
                            Ok(results)
                        }
                        Err(err) => Err(err),
                    }
                }
                Err(err) => Err(VectorSearchError::StorageError(err)),
            }
        }
        Err(err) => Err(VectorSearchError::StorageError(
            err.context("failed to open vector store"),
        )),
    };
    let vector_elapsed = embed_start.elapsed();
    timings.embedding = Some(vector_elapsed);
    timings.vector = Some(vector_elapsed);

    // Await the BM25 result (should already be done or nearly done).
    let (bm25_results, bm25_elapsed) = bm25_handle.await.map_err(|e| {
        HybridSearchError::PipelineError(anyhow::anyhow!("BM25 task panicked: {e}"))
    })?;
    timings.bm25 = Some(bm25_elapsed);

    match (bm25_results, vector_results) {
        (Ok(bm25), Ok(vector)) => {
            debug!(
                bm25_count = bm25.len(),
                vector_count = vector.len(),
                query_type = ?query_type,
                bm25_weight = query_params.bm25_weight,
                vector_weight = query_params.vector_weight,
                "merging BM25 and vector results via weighted RRF"
            );
            let fusion_start = Instant::now();
            let fused = fuse_rrf_weighted(
                &bm25,
                &vector,
                rrf_k,
                limit,
                query_params.bm25_weight,
                query_params.vector_weight,
            );
            timings.fusion = Some(fusion_start.elapsed());
            Ok((fused, timings))
        }
        (Ok(bm25), Err(vec_err)) => {
            warn!(
                error = %vec_err,
                "vector search failed, falling back to BM25-only results"
            );
            let mut results = bm25;
            results.truncate(limit);
            Ok((results, timings))
        }
        (Err(bm25_err), Ok(vector)) => {
            warn!(
                error = %bm25_err,
                "BM25 search failed, falling back to vector-only results"
            );
            let mut results = vector;
            results.truncate(limit);
            Ok((results, timings))
        }
        (Err(bm25_err), Err(vec_err)) => Err(HybridSearchError::BothFailed {
            bm25_error: format!("{bm25_err:#}"),
            vector_error: format!("{vec_err:#}"),
        }),
    }
}

/// Perform hybrid search with cross-encoder reranking.
///
/// Runs the full hybrid pipeline (BM25 + vector → RRF fusion), then
/// sends the top candidates to a cross-encoder reranker for more accurate
/// relevance scoring.
///
/// **Graceful degradation:**
/// - If the reranker API is unavailable (timeout, 5xx, connection error),
///   returns unreranked results with a warning logged to stderr.
/// - If the embedding API is unavailable, falls back to BM25-only results
///   (handled by the inner `search_hybrid` call).
///
/// # Arguments
/// - `index_dir` — Path to the `.vera` index directory
/// - `provider` — Embedding provider for vector search
/// - `reranker` — Reranker for result refinement
/// - `bm25_query` — Raw query text for the BM25 side (never intent-prefixed)
/// - `vector_query` — Query text for embedding and reranking (may carry intent)
/// - `filters` — Post-retrieval filters applied before fusion and reranking
/// - `fetch_limit` — Maximum number of candidates to retain through fusion
/// - `result_limit` — Maximum number of results to return to the user
/// - `rrf_k` — RRF constant (typically 60.0)
/// - `stored_dim` — Dimensionality of stored vectors
/// - `rerank_candidates` — Number of candidates to send to the reranker
/// - `vector_candidates` — Number of vector candidates to fetch (query-type-aware)
#[allow(clippy::too_many_arguments)]
pub async fn search_hybrid_reranked(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    reranker: &impl Reranker,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    rerank_candidates: usize,
    vector_candidates: usize,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    search_hybrid_reranked_with_augmentation(
        index_dir,
        provider,
        reranker,
        bm25_query,
        vector_query,
        filters,
        fetch_limit,
        result_limit,
        rrf_k,
        stored_dim,
        rerank_candidates,
        vector_candidates,
        false,
    )
    .await
}

/// Perform hybrid search with optional experimental graph augmentation.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn search_hybrid_reranked_with_augmentation(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    reranker: &impl Reranker,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    rerank_candidates: usize,
    vector_candidates: usize,
    graph_augmentation_enabled: bool,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    let fusion_limit = rerank_candidates.max(fetch_limit);

    let (mut hybrid_results, mut timings) = search_hybrid(
        index_dir,
        provider,
        bm25_query,
        vector_query,
        filters,
        fusion_limit,
        rrf_k,
        stored_dim,
        vector_candidates,
    )
    .await?;

    let augmented_count = if graph_augmentation_enabled {
        augment_pool(index_dir, &mut hybrid_results, filters)
    } else {
        0
    };

    if hybrid_results.is_empty() {
        return Ok((hybrid_results, timings));
    }

    if hybrid_results.len() <= result_limit && augmented_count == 0 {
        return Ok((hybrid_results, timings));
    }

    let rerank_start = Instant::now();
    // When graph candidates were appended, score the whole expanded pool so
    // candidates beyond the normal rerank prefix still compete semantically.
    let rerank_limit = if augmented_count > 0 {
        hybrid_results.len()
    } else {
        rerank_candidates
    };
    match rerank_results(reranker, vector_query, &hybrid_results, rerank_limit).await {
        Ok(mut reranked) => {
            timings.reranking = Some(rerank_start.elapsed());
            info!(
                query = vector_query,
                candidates = hybrid_results.len(),
                reranked = reranked.len(),
                "reranking complete"
            );
            let mut score_ceiling = reranked.last().map(|result| result.score);
            for mut result in hybrid_results.into_iter().skip(rerank_limit) {
                // RRF and reranker scores have different scales. Keep the untouched tail below
                // the reranked prefix so the public score order still matches the result order.
                if let Some(ceiling) = score_ceiling {
                    result.score = result.score.min(ceiling);
                }
                score_ceiling = Some(result.score);
                reranked.push(result);
            }
            reranked.truncate(fetch_limit);
            Ok((reranked, timings))
        }
        Err(rerank_err) => {
            timings.reranking = Some(rerank_start.elapsed());
            warn!(
                error = %rerank_err,
                "reranker unavailable, returning unreranked results"
            );
            eprintln!(
                "Warning: reranker unavailable ({rerank_err}), returning unreranked results."
            );
            let mut results = hybrid_results;
            results.truncate(fetch_limit);
            Ok((results, timings))
        }
    }
}

/// Perform hybrid search using pre-opened stores (useful for testing).
///
/// Takes pre-computed BM25 and vector results and fuses them via RRF.
/// This is the core fusion logic, separated from I/O for testability.
pub fn fuse_rrf(
    bm25_results: &[SearchResult],
    vector_results: &[SearchResult],
    rrf_k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    fuse_rrf_multi_weighted(&[bm25_results, vector_results], &[1.0, 1.0], rrf_k, limit)
}

/// Fuse BM25 and vector results with explicit per-source weights.
///
/// Identifier queries pass a higher BM25 weight (2.5) so lexical matches
/// dominate; NL queries use equal weights (1.0, 1.0).
pub fn fuse_rrf_weighted(
    bm25_results: &[SearchResult],
    vector_results: &[SearchResult],
    rrf_k: f64,
    limit: usize,
    bm25_weight: f64,
    vector_weight: f64,
) -> Vec<SearchResult> {
    fuse_rrf_multi_weighted(
        &[bm25_results, vector_results],
        &[bm25_weight, vector_weight],
        rrf_k,
        limit,
    )
}

/// Fuse multiple ranked result lists with weighted reciprocal rank fusion.
///
/// Each result set has an associated weight that scales its RRF contribution.
/// A weight of 2.0 means that set's scores count double in the final ranking.
pub fn fuse_rrf_multi_weighted(
    result_sets: &[&[SearchResult]],
    weights: &[f64],
    rrf_k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    let mut fused: HashMap<String, (f64, SearchResult)> = HashMap::new();

    for (set_idx, result_set) in result_sets.iter().enumerate() {
        let weight = weights.get(set_idx).copied().unwrap_or(1.0);
        for (rank_0, result) in result_set.iter().enumerate() {
            let key = result_key(result);
            let rrf_score = weight / (rrf_k + (rank_0 + 1) as f64);

            fused
                .entry(key)
                .and_modify(|(score, _)| *score += rrf_score)
                .or_insert_with(|| (rrf_score, result.clone()));
        }
    }

    let mut ranked: Vec<(f64, SearchResult)> = fused.into_values().collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    ranked
        .into_iter()
        .take(limit)
        .map(|(rrf_score, mut result)| {
            result.score = rrf_score;
            result
        })
        .collect()
}

#[cfg(test)]
#[path = "hybrid_tests.rs"]
mod tests;
