//! Vector similarity search over indexed chunks.
//!
//! Provides semantic search using the embedding pipeline and sqlite-vec
//! vector store. Generates a query embedding via the configured API provider,
//! then performs nearest-neighbor lookup, and hydrates results from the
//! metadata store. Finds semantically related code even when query terms
//! don't appear literally in results (e.g., "memory allocation" finds `alloc`).

use anyhow::Result;
use tracing::{debug, warn};

use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::{MAX_KNN_K, VectorStore};
use crate::types::SearchResult;

/// Errors specific to vector search.
#[derive(Debug, thiserror::Error)]
pub enum VectorSearchError {
    /// The embedding API is unavailable (connection, auth, etc.).
    #[error("embedding API unavailable: {source}")]
    EmbeddingUnavailable {
        #[from]
        source: EmbeddingError,
    },

    /// Storage or metadata error.
    #[error("storage error: {0:#}")]
    StorageError(#[from] anyhow::Error),
}

/// Perform vector search using pre-opened stores (useful for testing and reuse).
///
/// Generates a query embedding, runs nearest-neighbor search in the vector
/// store, then hydrates each result with full chunk metadata. Results are
/// returned sorted by similarity score in descending order.
///
/// The similarity score is derived from the distance: `score = 1.0 / (1.0 + distance)`.
/// This transforms the raw distance (lower = closer) into a 0–1 similarity score
/// (higher = more similar), which is consistent with other scoring conventions
/// in the retrieval pipeline.
pub async fn search_vector_with_stores(
    vector_store: &VectorStore,
    metadata_store: &MetadataStore,
    provider: &impl EmbeddingProvider,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchResult>, VectorSearchError> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    // 1. Generate query embedding.
    let query_embedding = generate_query_embedding(provider, query, vector_store.dim()).await?;

    debug!(
        query = query,
        dim = query_embedding.len(),
        "generated query embedding"
    );

    // 2. Search the vector store for nearest neighbors.
    let (requested, candidates) = candidate_pool(limit);
    if requested > candidates {
        // Deliberately without the query text: the surrounding `debug!` calls
        // carry it, but this one is on by default, and a search query is user
        // content that should not be raised into default-visible logs.
        warn!(
            requested,
            fetched = candidates,
            "candidate pool exceeds the sqlite-vec KNN cap; the filter and \
             metadata over-fetch are inert above it"
        );
    }

    let vector_results = vector_store
        .search(&query_embedding, candidates)
        .map_err(|e| VectorSearchError::StorageError(e.context("vector search failed")))?;

    debug!(
        query = query,
        raw_results = vector_results.len(),
        "vector search returned candidates"
    );

    // 3. Hydrate results from metadata store and convert distances to scores.
    // Fetched in a single batch query instead of one query per candidate
    // (N+1). SQLite does not preserve `IN (...)` row order, so results are
    // re-projected back into the original vector-distance order below.
    let ids: Vec<String> = vector_results
        .iter()
        .map(|vr| vr.chunk_id.clone())
        .collect();
    let mut chunk_by_id = metadata_store.get_chunks_by_ids(&ids).map_err(|e| {
        VectorSearchError::StorageError(e.context("failed to batch-fetch chunk metadata"))
    })?;

    let mut results = Vec::with_capacity(vector_results.len());

    for vr in &vector_results {
        let Some(chunk) = chunk_by_id.remove(&vr.chunk_id) else {
            debug!(
                chunk_id = %vr.chunk_id,
                "chunk metadata not found, skipping"
            );
            continue;
        };

        // Convert distance to similarity score: higher is better.
        let score = distance_to_similarity(vr.distance);

        results.push(chunk.into_search_result(score));

        if results.len() >= limit {
            break;
        }
    }

    debug!(
        query = query,
        returned = results.len(),
        "vector search complete"
    );

    Ok(results)
}

/// Size the KNN request for a caller-selected pool, bounded by the backend cap.
///
/// Returns `(requested, fetched)`. Extra candidates are fetched to absorb chunks
/// whose metadata has gone missing, without doubling an already-large pool.
///
/// The two values differ once the compounded over-fetch runs past
/// [`MAX_KNN_K`]. Callers stack several multipliers before reaching here (query
/// type, then a filter over-fetch, then this one), so a natural-language query
/// with an active filter can ask for more than sqlite-vec will serve. Bounding
/// it here rather than letting the storage layer clamp keeps the ceiling
/// visible to the one place that can report it; the same `k` reaches sqlite-vec
/// either way, so ranking is unchanged.
fn candidate_pool(limit: usize) -> (usize, usize) {
    let requested = limit
        .saturating_add(limit / 2)
        .max(limit.saturating_add(10));
    (requested, requested.min(MAX_KNN_K))
}

/// Generate a query embedding, truncating to match stored dimensionality.
///
/// The embedding provider may return vectors of higher dimensionality than
/// what is stored (e.g., Qwen3 produces 4096-dim but we store 1024-dim
/// via Matryoshka truncation). This function truncates the query vector
/// to match the stored vector dimensionality.
async fn generate_query_embedding(
    provider: &impl EmbeddingProvider,
    query: &str,
    stored_dim: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    let texts = vec![provider.prepare_query_text(query)];
    let mut vectors = provider.embed_batch(&texts).await?;

    if vectors.is_empty() {
        return Err(EmbeddingError::ResponseError {
            message: "embedding API returned no vectors for query".to_string(),
        });
    }

    let mut vector = vectors.swap_remove(0);

    // Truncate to stored dimensionality if needed (Matryoshka embedding).
    if vector.len() > stored_dim {
        vector.truncate(stored_dim);
    }

    // Validate dimensionality matches.
    if vector.len() != stored_dim {
        return Err(EmbeddingError::ResponseError {
            message: format!(
                "query embedding dimension {} does not match stored dimension {}",
                vector.len(),
                stored_dim
            ),
        });
    }

    Ok(vector)
}
/// Convert a distance score to a similarity score.
///
/// sqlite-vec returns L2 (Euclidean) distances where lower is better.
/// We convert to a similarity score in (0, 1] where higher is better:
///   similarity = 1.0 / (1.0 + distance)
///
/// This is a standard transformation that:
/// - Maps distance=0 to similarity=1 (perfect match)
/// - Maps distance→∞ to similarity→0 (no similarity)
/// - Is monotonically decreasing (preserves ranking)
fn distance_to_similarity(distance: f64) -> f64 {
    1.0 / (1.0 + distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::test_helpers::MockProvider;
    use crate::types::{Chunk, Language, SymbolType};
    use std::collections::HashMap;

    /// The pool must never ask sqlite-vec for more than it will serve, and must
    /// report the shortfall so a degraded pool is diagnosable rather than silent.
    #[test]
    fn candidate_pool_is_bounded_by_the_knn_cap() {
        // Below the cap the metadata over-fetch is untouched.
        assert_eq!(candidate_pool(10), (20, 20));
        assert_eq!(candidate_pool(100), (150, 150));

        // At and above it, the request is reported but not made.
        let (requested, fetched) = candidate_pool(MAX_KNN_K);
        assert!(
            requested > MAX_KNN_K,
            "over-fetch should exceed the cap here"
        );
        assert_eq!(fetched, MAX_KNN_K);

        // No caller can push the fetch past the cap, and none can overflow it.
        assert_eq!(candidate_pool(usize::MAX).1, MAX_KNN_K);
    }

    /// Create sample chunks with semantic variety for testing.
    fn sample_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                id: "src/alloc.rs:0".to_string(),
                file_path: "src/alloc.rs".to_string(),
                line_start: 1,
                line_end: 15,
                content: "pub fn alloc_page(size: usize) -> *mut u8 {\n    \
                           let layout = Layout::from_size_align(size, 4096).unwrap();\n    \
                           unsafe { std::alloc::alloc(layout) }\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("alloc_page".to_string()),
            },
            Chunk {
                id: "src/alloc.rs:1".to_string(),
                file_path: "src/alloc.rs".to_string(),
                line_start: 17,
                line_end: 30,
                content: "pub fn dealloc_page(ptr: *mut u8, size: usize) {\n    \
                           let layout = Layout::from_size_align(size, 4096).unwrap();\n    \
                           unsafe { std::alloc::dealloc(ptr, layout) }\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("dealloc_page".to_string()),
            },
            Chunk {
                id: "src/auth.rs:0".to_string(),
                file_path: "src/auth.rs".to_string(),
                line_start: 1,
                line_end: 12,
                content: "pub fn authenticate(user: &str, password: &str) -> Result<Token> {\n    \
                           let hash = hash_password(password);\n    \
                           verify_credentials(user, &hash)\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("authenticate".to_string()),
            },
            Chunk {
                id: "src/db.py:0".to_string(),
                file_path: "src/db.py".to_string(),
                line_start: 1,
                line_end: 10,
                content: "class DatabaseConnection:\n    \
                           def __init__(self, host, port):\n        \
                           self.conn = psycopg2.connect(host=host, port=port)\n    \
                           def execute_query(self, sql):\n        \
                           return self.conn.execute(sql)"
                    .to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Class),
                symbol_name: Some("DatabaseConnection".to_string()),
            },
            Chunk {
                id: "src/cache.go:0".to_string(),
                file_path: "src/cache.go".to_string(),
                line_start: 1,
                line_end: 15,
                content: "func NewLRUCache(capacity int) *LRUCache {\n    \
                           return &LRUCache{\n        \
                           capacity: capacity,\n        \
                           items: make(map[string]*list.Element),\n        \
                           order: list.New(),\n    }\n}"
                    .to_string(),
                language: Language::Go,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("NewLRUCache".to_string()),
            },
            Chunk {
                id: "src/server.ts:0".to_string(),
                file_path: "src/server.ts".to_string(),
                line_start: 1,
                line_end: 10,
                content: "function handleRequest(req: Request): Response {\n    \
                           const auth = authenticate(req.headers);\n    \
                           if (!auth) return new Response('Unauthorized', { status: 401 });\n    \
                           return processRequest(req);\n}"
                    .to_string(),
                language: Language::TypeScript,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("handleRequest".to_string()),
            },
        ]
    }

    /// Embed sample chunks with MockProvider and store in vector + metadata stores.
    async fn setup_test_stores(dim: usize) -> (VectorStore, MetadataStore) {
        let chunks = sample_chunks();
        let provider = MockProvider::new(dim);

        // Store metadata.
        let metadata_store = MetadataStore::open_in_memory().unwrap();
        metadata_store.insert_chunks(&chunks).unwrap();

        // Generate embeddings and store vectors.
        let vector_store = VectorStore::open_in_memory(dim).unwrap();
        crate::embedding::test_helpers::embed_and_insert_vectors(&vector_store, &provider, &chunks)
            .await;

        (vector_store, metadata_store)
    }

    // ── distance_to_similarity tests ─────────────────────────────────

    #[test]
    fn distance_zero_gives_similarity_one() {
        let score = distance_to_similarity(0.0);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn distance_positive_gives_score_between_zero_and_one() {
        let score = distance_to_similarity(1.0);
        assert!(score > 0.0);
        assert!(score < 1.0);
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn distance_large_gives_score_near_zero() {
        let score = distance_to_similarity(1000.0);
        assert!(score > 0.0);
        assert!(score < 0.01);
    }

    #[test]
    fn similarity_is_monotonically_decreasing() {
        let distances = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
        let scores: Vec<f64> = distances
            .iter()
            .map(|d| distance_to_similarity(*d))
            .collect();
        for i in 1..scores.len() {
            assert!(
                scores[i - 1] > scores[i],
                "similarity must decrease as distance increases: {} > {} at index {i}",
                scores[i - 1],
                scores[i],
            );
        }
    }

    #[test]
    fn storage_error_display_includes_error_chain() {
        let error = anyhow::anyhow!("root cause").context("outer context");
        let display = VectorSearchError::StorageError(error).to_string();

        assert!(display.contains("outer context"));
        assert!(display.contains("root cause"));
    }

    // ── generate_query_embedding tests ───────────────────────────────

    #[tokio::test]
    async fn query_embedding_has_correct_dimension() {
        let provider = MockProvider::new(8);
        let embedding = generate_query_embedding(&provider, "test query", 8)
            .await
            .unwrap();
        assert_eq!(embedding.len(), 8);
    }

    #[tokio::test]
    async fn query_embedding_truncates_to_stored_dim() {
        // Provider produces 16-dim, but stored is 8-dim.
        let provider = MockProvider::new(16);
        let embedding = generate_query_embedding(&provider, "test query", 8)
            .await
            .unwrap();
        assert_eq!(embedding.len(), 8);
    }

    #[tokio::test]
    async fn query_embedding_rejects_dim_mismatch() {
        // Provider produces 4-dim, but stored is 8-dim.
        let provider = MockProvider::new(4);
        let result = generate_query_embedding(&provider, "test query", 8).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("dimension"),
            "error should mention dimension: {err}"
        );
    }

    #[tokio::test]
    async fn query_embedding_propagates_auth_error() {
        let provider = MockProvider::failing(EmbeddingError::AuthError {
            message: "invalid key".to_string(),
        });
        let result = generate_query_embedding(&provider, "test query", 4).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), EmbeddingError::AuthError { .. }),
            "should propagate auth error"
        );
    }

    #[tokio::test]
    async fn query_embedding_propagates_connection_error() {
        let provider = MockProvider::failing(EmbeddingError::ConnectionError {
            message: "unreachable".to_string(),
        });
        let result = generate_query_embedding(&provider, "test query", 4).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), EmbeddingError::ConnectionError { .. }),
            "should propagate connection error"
        );
    }

    // ── search_vector_with_stores tests ──────────────────────────────

    #[tokio::test]
    async fn search_returns_results_for_indexed_content() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "alloc", 10)
                .await
                .unwrap();

        assert!(!results.is_empty(), "should find results for 'alloc'");
    }

    #[tokio::test]
    async fn results_sorted_by_score_descending() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results = search_vector_with_stores(
            &vector_store,
            &metadata_store,
            &provider,
            "database connection query",
            10,
        )
        .await
        .unwrap();

        assert!(
            results.len() >= 2,
            "need multiple results to verify ordering"
        );
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "scores must be descending: {} >= {} at position {i}",
                results[i - 1].score,
                results[i].score,
            );
        }
    }

    #[tokio::test]
    async fn results_include_full_metadata() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "cache", 10)
                .await
                .unwrap();

        assert!(!results.is_empty());
        let top = &results[0];
        // Every result should have required fields populated.
        assert!(!top.file_path.is_empty(), "file_path should be set");
        assert!(top.line_start > 0, "line_start should be 1-based");
        assert!(top.line_end >= top.line_start, "line_end >= line_start");
        assert!(!top.content.is_empty(), "content should be present");
        assert!(top.score > 0.0, "score should be positive");
        assert!(top.score <= 1.0, "similarity score should be <= 1.0");
    }

    #[tokio::test]
    async fn search_respects_limit() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "function", 2)
                .await
                .unwrap();

        assert!(results.len() <= 2, "results should respect the limit of 2");
    }

    #[tokio::test]
    async fn search_zero_limit_skips_embedding() {
        let dim = 8;
        let vector_store = VectorStore::open_in_memory(dim).unwrap();
        let metadata_store = MetadataStore::open_in_memory().unwrap();
        let provider = MockProvider::failing(EmbeddingError::ConnectionError {
            message: "should not be called".to_string(),
        });

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "function", 0)
                .await
                .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn scores_are_positive_and_bounded() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results = search_vector_with_stores(
            &vector_store,
            &metadata_store,
            &provider,
            "authenticate",
            10,
        )
        .await
        .unwrap();

        for result in &results {
            assert!(result.score > 0.0, "score should be positive");
            assert!(result.score <= 1.0, "score should be <= 1.0");
        }
    }

    #[tokio::test]
    async fn search_with_embedding_error_returns_error() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::failing(EmbeddingError::ConnectionError {
            message: "API down".to_string(),
        });

        let result =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "test", 10).await;

        assert!(result.is_err(), "should return error when embedding fails");
        assert!(
            matches!(
                result.unwrap_err(),
                VectorSearchError::EmbeddingUnavailable { .. }
            ),
            "should be an embedding unavailable error"
        );
    }

    #[tokio::test]
    async fn search_returns_results_from_multiple_languages() {
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        // Get all results.
        let results = search_vector_with_stores(
            &vector_store,
            &metadata_store,
            &provider,
            "code function",
            10,
        )
        .await
        .unwrap();

        let languages: std::collections::HashSet<_> = results.iter().map(|r| r.language).collect();
        assert!(
            languages.len() >= 2,
            "should return results from multiple languages, got: {languages:?}"
        );
    }

    #[tokio::test]
    async fn search_with_truncation() {
        // Provider returns 16-dim vectors, but store uses 8-dim.
        let dim = 8;
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        // Use a provider that produces larger vectors than stored.
        let query_provider = MockProvider::new(16);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &query_provider, "cache", 10)
                .await
                .unwrap();

        assert!(
            !results.is_empty(),
            "should work with truncated query embeddings"
        );
    }

    #[tokio::test]
    async fn empty_vector_store_returns_empty_results() {
        let dim = 8;
        let vector_store = VectorStore::open_in_memory(dim).unwrap();
        let metadata_store = MetadataStore::open_in_memory().unwrap();
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "anything", 10)
                .await
                .unwrap();

        assert!(results.is_empty(), "empty store should return no results");
    }

    #[tokio::test]
    async fn batch_chunk_fetch_preserves_vector_distance_order() {
        // Regression test for the N+1 -> batch fetch change: insert chunks
        // in an order different from the requested id order, and assert the
        // search results come back in vector-distance order (the order `vr`
        // came in), not whatever order SQLite happens to return for
        // `IN (...)` (unspecified) or insertion order.
        let dim = 8;
        let metadata_store = MetadataStore::open_in_memory().unwrap();

        // Inserted in id order, which is also the order `WHERE id IN (...)`
        // returns rows in (the chunks table keys on id). The mock provider
        // ranks these contents c, a, b by distance, so the expected result
        // order differs from both — which is what makes the assertions below
        // able to fail.
        let chunks = vec![
            Chunk {
                id: "a".to_string(),
                file_path: "src/a.rs".to_string(),
                line_start: 1,
                line_end: 2,
                content: "fn a() {}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("a".to_string()),
            },
            Chunk {
                id: "b".to_string(),
                file_path: "src/b.rs".to_string(),
                line_start: 1,
                line_end: 2,
                content: "fn b() {}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("b".to_string()),
            },
            Chunk {
                id: "c".to_string(),
                file_path: "src/c.rs".to_string(),
                line_start: 1,
                line_end: 2,
                content: "fn c() {}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("c".to_string()),
            },
        ];
        metadata_store.insert_chunks(&chunks).unwrap();

        let vector_store = VectorStore::open_in_memory(dim).unwrap();
        let provider = MockProvider::new(dim);
        crate::embedding::test_helpers::embed_and_insert_vectors(&vector_store, &provider, &chunks)
            .await;

        // Ground truth: the distance order the vector store itself reports.
        // Every candidate must come back paired with its own distance.
        let query_embedding = generate_query_embedding(&provider, "fn", dim)
            .await
            .unwrap();
        let vector_results = vector_store.search(&query_embedding, chunks.len()).unwrap();
        assert_eq!(vector_results.len(), chunks.len());

        let chunk_by_id: HashMap<&str, &Chunk> =
            chunks.iter().map(|c| (c.id.as_str(), c)).collect();
        let expected: Vec<(&str, &str, f64)> = vector_results
            .iter()
            .map(|vr| {
                let chunk = chunk_by_id[vr.chunk_id.as_str()];
                (
                    chunk.file_path.as_str(),
                    chunk.content.as_str(),
                    distance_to_similarity(vr.distance),
                )
            })
            .collect();

        // Guard that the fixture actually discriminates. The chunks are
        // inserted in id order, so insertion order and the `IN (...)`
        // primary-key order are the same sequence here; one comparison covers
        // both. Without this, a fixture whose distance order matched storage
        // order would satisfy the assertions below without testing anything —
        // which is exactly what the first version of this test did.
        let expected_paths: Vec<&str> = expected.iter().map(|(path, _, _)| *path).collect();
        let storage_order_paths: Vec<&str> = chunks.iter().map(|c| c.file_path.as_str()).collect();
        assert_ne!(
            expected_paths, storage_order_paths,
            "fixture is not discriminating: distance order equals insertion and primary-key order"
        );

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "fn", 10)
                .await
                .unwrap();

        // Assert the pairing, not just that scores descend. Scores descend for
        // any implementation that emits one result per candidate in candidate
        // order, because the vector store already returns rows sorted by
        // distance — so a monotonicity check cannot catch chunks being paired
        // with the wrong neighbour's distance.
        assert_eq!(
            results.len(),
            expected.len(),
            "batch fetch dropped candidates"
        );
        let actual: Vec<(&str, &str, f64)> = results
            .iter()
            .map(|r| (r.file_path.as_str(), r.content.as_str(), r.score))
            .collect();
        assert_eq!(
            actual, expected,
            "each chunk must come back paired with its own distance, in vector-distance order"
        );
    }

    #[tokio::test]
    async fn result_content_matches_source_chunk() {
        let dim = 8;
        let chunks = sample_chunks();
        let (vector_store, metadata_store) = setup_test_stores(dim).await;
        let provider = MockProvider::new(dim);

        let results =
            search_vector_with_stores(&vector_store, &metadata_store, &provider, "alloc page", 10)
                .await
                .unwrap();

        assert!(!results.is_empty());
        // Find a result from alloc.rs and verify content matches original.
        let alloc_result = results.iter().find(|r| r.file_path == "src/alloc.rs");
        assert!(alloc_result.is_some(), "should find result from alloc.rs");
        let alloc_result = alloc_result.unwrap();
        // Verify content matches one of the original chunks.
        let matching_chunk = chunks
            .iter()
            .find(|c| c.file_path == alloc_result.file_path && c.content == alloc_result.content);
        assert!(
            matching_chunk.is_some(),
            "result content should match a source chunk"
        );
    }
}
