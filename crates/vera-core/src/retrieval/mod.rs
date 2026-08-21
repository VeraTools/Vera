//! Hybrid retrieval pipeline: BM25 + vector search, RRF fusion, reranking.
//!
//! This module is responsible for:
//! - BM25 keyword search via Tantivy
//! - Vector similarity search via sqlite-vec
//! - Reciprocal Rank Fusion (RRF) for merging results
//! - Cross-encoder reranking via external API
//! - Post-retrieval filtering by language, path glob, and symbol type
//! - Graceful degradation when services are unavailable

pub mod bm25;
pub(crate) mod exact_matches;
pub(crate) mod graph_augmentation;
pub mod hybrid;
pub mod query_classifier;
pub mod ranking;
pub mod references;
pub mod reranker;
pub mod search_service;
pub mod type_relations;
pub mod vector;

pub use bm25::{search_bm25, search_bm25_with_stores, search_bm25_with_stores_and_filters};
pub use hybrid::{
    HybridSearchError, HybridTimings, fuse_rrf, fuse_rrf_multi_weighted, search_hybrid,
    search_hybrid_reranked,
};
pub use reranker::{
    ApiReranker, RerankScore, Reranker, RerankerConfig, RerankerError, rerank_results,
};

pub mod dynamic_reranker;
pub use dynamic_reranker::{DynamicReranker, create_dynamic_reranker};

pub mod completion_client;
pub(crate) mod file_scan;
pub mod iterative_search;
pub mod local_reranker;
pub(crate) mod query_utils;
pub mod rag_fusion;
pub mod regex_search;
pub mod structural;

pub use local_reranker::LocalReranker;
pub use references::search_callers;
pub use regex_search::search_regex;
pub use structural::{StructuralSearchKind, search_structural};

pub use vector::{VectorSearchError, search_vector_with_stores};

use crate::types::{SearchFilters, SearchResult};

#[cfg(test)]
#[path = "search_quality_tests.rs"]
mod search_quality_tests;

/// Apply search filters to a list of results, preserving order and limit.
///
/// Filters are applied post-retrieval: results that don't match all active
/// filters are removed. The `limit` parameter caps the final result count.
pub fn apply_filters(
    results: Vec<SearchResult>,
    filters: &SearchFilters,
    limit: usize,
) -> Vec<SearchResult> {
    if filters.is_empty() {
        let mut results = results;
        results.truncate(limit);
        return results;
    }

    results
        .into_iter()
        .filter(|r| filters.matches(r))
        .take(limit)
        .collect()
}

/// Collapse whitespace and drop empty or repeated subqueries.
///
/// Fusion counts every subquery, so the same query arriving twice would
/// otherwise weigh double against the ones it was meant to complement.
pub fn normalize_queries(queries: &[String]) -> Vec<String> {
    let mut normalized = Vec::with_capacity(queries.len());
    let mut seen = std::collections::HashSet::new();

    for query in queries {
        let collapsed = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            continue;
        }
        if seen.insert(collapsed.to_ascii_lowercase()) {
            normalized.push(collapsed);
        }
    }

    normalized
}

/// Candidate width each subquery keeps before multi-query fusion.
///
/// A subquery has to over-fetch relative to the caller's limit, or fusion has
/// nothing to merge: the first subquery alone fills the final window and every
/// later one is truncated away.
pub fn multi_query_candidate_limit(result_limit: usize) -> usize {
    result_limit
        .saturating_mul(2)
        .max(result_limit.saturating_add(10))
        .max(20)
}

/// Merge per-subquery result sets: equally weighted RRF, then exact-match
/// augmentation, then the cut to `result_limit`.
///
/// `fuse_limit` is a parameter rather than a constant because the two callers
/// disagree on it and are meant to. MCP passes `multi_query_candidate_limit`,
/// so augmentation still has candidates to displace; `vera search` passes
/// `result_limit`, truncating before augmenting, which is issue #121 and is
/// left alone here. Collapsing them would silently change CLI results.
pub fn fuse_and_augment_multi_query(
    index_dir: &std::path::Path,
    queries: &[String],
    result_sets: &[Vec<SearchResult>],
    filters: &SearchFilters,
    rrf_k: f64,
    fuse_limit: usize,
    result_limit: usize,
) -> anyhow::Result<Vec<SearchResult>> {
    let slices: Vec<&[SearchResult]> = result_sets.iter().map(Vec::as_slice).collect();
    let weights = vec![1.0; result_sets.len()];
    let fused = fuse_rrf_multi_weighted(&slices, &weights, rrf_k, fuse_limit);
    exact_matches::augment_multi_query_exact_matches(
        index_dir,
        queries,
        fused,
        filters,
        result_limit,
    )
}
