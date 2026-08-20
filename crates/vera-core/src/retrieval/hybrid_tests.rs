use super::*;
use crate::retrieval::reranker::{RerankScore, Reranker, RerankerError};
use crate::types::{Language, SymbolType};

struct NegativeReranker;

impl Reranker for NegativeReranker {
    async fn rerank(
        &self,
        _query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        Ok(documents
            .iter()
            .enumerate()
            .map(|(index, _)| RerankScore {
                index,
                relevance_score: -1.0 - index as f64,
            })
            .collect())
    }
}

/// Helper to create a SearchResult with given parameters.
fn make_result(
    file: &str,
    line_start: u32,
    line_end: u32,
    score: f64,
    symbol_name: Option<&str>,
) -> SearchResult {
    SearchResult {
        file_path: file.to_string(),
        line_start,
        line_end,
        content: format!("content of {file}:{line_start}"),
        language: Language::Rust,
        score,
        symbol_name: symbol_name.map(|s| s.to_string()),
        symbol_type: Some(SymbolType::Function),
    }
}

// ── RRF score calculation tests ─────────────────────────────────

#[test]
fn rrf_single_source_bm25_only() {
    let bm25 = vec![
        make_result("a.rs", 1, 10, 5.0, Some("func_a")),
        make_result("b.rs", 1, 10, 3.0, Some("func_b")),
    ];
    let vector: Vec<SearchResult> = vec![];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    assert_eq!(results.len(), 2);
    // First BM25 result: 1/(60+1) ≈ 0.01639
    let expected_score_1 = 1.0 / 61.0;
    assert!(
        (results[0].score - expected_score_1).abs() < 1e-10,
        "first result RRF score: got {}, expected {}",
        results[0].score,
        expected_score_1
    );
    assert_eq!(results[0].file_path, "a.rs");
}

#[test]
fn rrf_single_source_vector_only() {
    let bm25: Vec<SearchResult> = vec![];
    let vector = vec![
        make_result("c.rs", 1, 10, 0.9, Some("func_c")),
        make_result("d.rs", 1, 10, 0.7, Some("func_d")),
    ];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    assert_eq!(results.len(), 2);
    let expected_score_1 = 1.0 / 61.0;
    assert!(
        (results[0].score - expected_score_1).abs() < 1e-10,
        "first result RRF score: got {}, expected {}",
        results[0].score,
        expected_score_1
    );
    assert_eq!(results[0].file_path, "c.rs");
}

#[test]
fn rrf_overlapping_results_rank_higher() {
    // Result "shared.rs:1:10" appears in both BM25 (rank 2) and vector (rank 1).
    // Result "bm25_only.rs:1:10" appears only in BM25 (rank 1).
    // Result "vector_only.rs:1:10" appears only in vector (rank 2).
    let bm25 = vec![
        make_result("bm25_only.rs", 1, 10, 5.0, Some("bm25_func")),
        make_result("shared.rs", 1, 10, 3.0, Some("shared_func")),
    ];
    let vector = vec![
        make_result("shared.rs", 1, 10, 0.9, Some("shared_func")),
        make_result("vector_only.rs", 1, 10, 0.7, Some("vector_func")),
    ];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    // shared.rs appears in both: RRF = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    let shared_score = 1.0 / 62.0 + 1.0 / 61.0;
    // bm25_only.rs: RRF = 1/(60+1) = 1/61
    let bm25_only_score = 1.0 / 61.0;
    // vector_only.rs: RRF = 1/(60+2) = 1/62
    let _vector_only_score = 1.0 / 62.0;

    assert!(
        shared_score > bm25_only_score,
        "overlapping result should have higher RRF score"
    );

    // shared.rs should be the top result.
    assert_eq!(
        results[0].file_path, "shared.rs",
        "result appearing in both lists should rank first"
    );
    assert!(
        (results[0].score - shared_score).abs() < 1e-10,
        "shared score: got {}, expected {}",
        results[0].score,
        shared_score
    );
}

#[test]
fn rrf_scores_are_descending() {
    let bm25 = vec![
        make_result("a.rs", 1, 10, 5.0, Some("func_a")),
        make_result("b.rs", 1, 10, 3.0, Some("func_b")),
        make_result("c.rs", 1, 10, 1.0, Some("func_c")),
    ];
    let vector = vec![
        make_result("d.rs", 1, 10, 0.9, Some("func_d")),
        make_result("a.rs", 1, 10, 0.8, Some("func_a")),
        make_result("e.rs", 1, 10, 0.5, Some("func_e")),
    ];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "scores must be descending: {} >= {} at position {i}",
            results[i - 1].score,
            results[i].score,
        );
    }
}

#[test]
fn rrf_respects_limit() {
    let bm25 = vec![
        make_result("a.rs", 1, 10, 5.0, None),
        make_result("b.rs", 1, 10, 3.0, None),
        make_result("c.rs", 1, 10, 1.0, None),
    ];
    let vector = vec![
        make_result("d.rs", 1, 10, 0.9, None),
        make_result("e.rs", 1, 10, 0.7, None),
    ];

    let results = fuse_rrf(&bm25, &vector, 60.0, 2);
    assert_eq!(results.len(), 2, "should respect the limit");
}

#[test]
fn rrf_empty_inputs_return_empty() {
    let results = fuse_rrf(&[], &[], 60.0, 10);
    assert!(results.is_empty(), "no inputs should give no results");
}

#[test]
fn rrf_preserves_metadata_from_first_seen() {
    let bm25 = vec![make_result("shared.rs", 1, 10, 5.0, Some("func"))];
    let vector = vec![make_result("shared.rs", 1, 10, 0.9, Some("func"))];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    assert_eq!(results.len(), 1);
    let result = &results[0];
    assert_eq!(result.file_path, "shared.rs");
    assert_eq!(result.line_start, 1);
    assert_eq!(result.line_end, 10);
    assert_eq!(result.symbol_name.as_deref(), Some("func"));
    assert_eq!(result.symbol_type, Some(SymbolType::Function));
    assert_eq!(result.language, Language::Rust);
}

#[test]
fn rrf_k_parameter_affects_scores() {
    let bm25 = vec![make_result("a.rs", 1, 10, 5.0, None)];
    let vector: Vec<SearchResult> = vec![];

    // With k=60: score = 1/61 ≈ 0.01639
    let results_k60 = fuse_rrf(&bm25, &vector, 60.0, 10);
    // With k=1: score = 1/2 = 0.5
    let results_k1 = fuse_rrf(&bm25, &vector, 1.0, 10);

    assert!(
        results_k1[0].score > results_k60[0].score,
        "lower k should produce higher scores: k1={}, k60={}",
        results_k1[0].score,
        results_k60[0].score
    );
}

#[test]
fn rrf_with_known_inputs_produces_exact_scores() {
    // RRF with k=60:
    // Item A: BM25 rank 1, vector rank 3 → 1/61 + 1/63
    // Item B: BM25 rank 2, vector rank 1 → 1/62 + 1/61
    // Item C: BM25 rank 3, vector rank 2 → 1/63 + 1/62
    let bm25 = vec![
        make_result("a.rs", 1, 10, 5.0, Some("a")),
        make_result("b.rs", 1, 10, 3.0, Some("b")),
        make_result("c.rs", 1, 10, 1.0, Some("c")),
    ];
    let vector = vec![
        make_result("b.rs", 1, 10, 0.9, Some("b")),
        make_result("c.rs", 1, 10, 0.8, Some("c")),
        make_result("a.rs", 1, 10, 0.5, Some("a")),
    ];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    let score_a = 1.0 / 61.0 + 1.0 / 63.0;
    let score_b = 1.0 / 62.0 + 1.0 / 61.0;
    let score_c = 1.0 / 63.0 + 1.0 / 62.0;

    // B should rank first (highest combined score)
    assert_eq!(results[0].file_path, "b.rs");
    assert!(
        (results[0].score - score_b).abs() < 1e-10,
        "B score: got {}, expected {score_b}",
        results[0].score
    );

    // A should rank second
    assert_eq!(results[1].file_path, "a.rs");
    assert!(
        (results[1].score - score_a).abs() < 1e-10,
        "A score: got {}, expected {score_a}",
        results[1].score
    );

    // C should rank third
    assert_eq!(results[2].file_path, "c.rs");
    assert!(
        (results[2].score - score_c).abs() < 1e-10,
        "C score: got {}, expected {score_c}",
        results[2].score
    );
}

#[test]
fn rrf_distinct_results_no_overlap() {
    // When BM25 and vector have completely different results,
    // BM25 rank-1 and vector rank-1 should tie (same RRF score).
    let bm25 = vec![
        make_result("bm25_1.rs", 1, 10, 5.0, None),
        make_result("bm25_2.rs", 1, 10, 3.0, None),
    ];
    let vector = vec![
        make_result("vec_1.rs", 1, 10, 0.9, None),
        make_result("vec_2.rs", 1, 10, 0.7, None),
    ];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    assert_eq!(results.len(), 4);
    // The top two should have score 1/61, the bottom two should have score 1/62.
    let expected_top = 1.0 / 61.0;
    let expected_bottom = 1.0 / 62.0;

    assert!((results[0].score - expected_top).abs() < 1e-10);
    assert!((results[1].score - expected_top).abs() < 1e-10);
    assert!((results[2].score - expected_bottom).abs() < 1e-10);
    assert!((results[3].score - expected_bottom).abs() < 1e-10);
}

#[test]
fn rrf_different_chunks_same_file_are_separate() {
    // Two different chunks from the same file should be treated as separate.
    let bm25 = vec![
        make_result("lib.rs", 1, 10, 5.0, Some("func_1")),
        make_result("lib.rs", 20, 30, 3.0, Some("func_2")),
    ];
    let vector = vec![make_result("lib.rs", 1, 10, 0.9, Some("func_1"))];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    // lib.rs:1:10 appears in both → higher score
    // lib.rs:20:30 appears only in BM25 → lower score
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].line_start, 1,
        "overlapping chunk should rank first"
    );
    assert_eq!(
        results[1].line_start, 20,
        "single-source chunk should rank second"
    );
}

#[test]
fn rrf_scores_are_positive() {
    let bm25 = vec![
        make_result("a.rs", 1, 10, 5.0, None),
        make_result("b.rs", 1, 10, 3.0, None),
    ];
    let vector = vec![
        make_result("c.rs", 1, 10, 0.9, None),
        make_result("a.rs", 1, 10, 0.8, None),
    ];

    let results = fuse_rrf(&bm25, &vector, 60.0, 10);

    for result in &results {
        assert!(result.score > 0.0, "RRF scores should be positive");
    }
}

// ── result_key tests ────────────────────────────────────────────

#[test]
fn result_key_is_unique_for_different_chunks() {
    let r1 = make_result("a.rs", 1, 10, 1.0, None);
    let r2 = make_result("a.rs", 20, 30, 1.0, None);
    let r3 = make_result("b.rs", 1, 10, 1.0, None);

    assert_ne!(result_key(&r1), result_key(&r2));
    assert_ne!(result_key(&r1), result_key(&r3));
    assert_ne!(result_key(&r2), result_key(&r3));
}

#[test]
fn result_key_is_same_for_same_chunk() {
    let r1 = make_result("a.rs", 1, 10, 5.0, Some("func"));
    let r2 = make_result("a.rs", 1, 10, 0.9, Some("func"));

    assert_eq!(result_key(&r1), result_key(&r2));
}

// ── Integration tests for search_hybrid_reranked ─────────────────

/// Set up a temp index directory with BM25 + vector + metadata stores.
/// Returns (index_dir_path, stored_dim) for use in integration tests.
async fn setup_test_index(tmp: &std::path::Path) -> (std::path::PathBuf, usize) {
    use crate::config::VeraConfig;
    use crate::embedding::test_helpers::MockProvider;
    use crate::parsing;
    use crate::storage::bm25::Bm25Index;
    use crate::storage::metadata::MetadataStore;
    use crate::storage::vector::VectorStore;

    let dim = 8;
    let provider = MockProvider::new(dim);
    let config = VeraConfig::default();

    // Create sample source files.
    let repo_dir = tmp.join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(
        repo_dir.join("auth.rs"),
        "pub fn authenticate(user: &str, pass: &str) -> Result<Token, Error> {\n    \
             let hash = hash_password(pass);\n    verify_credentials(user, &hash)\n}\n\n\
             pub fn authorize(token: &Token, resource: &str) -> bool {\n    \
             token.has_permission(resource)\n}\n",
    )
    .unwrap();
    std::fs::write(
        repo_dir.join("cache.rs"),
        "pub fn get_cached(key: &str) -> Option<Value> {\n    \
             CACHE.lock().unwrap().get(key).cloned()\n}\n\n\
             pub fn set_cached(key: &str, value: Value) {\n    \
             CACHE.lock().unwrap().insert(key.to_string(), value);\n}\n",
    )
    .unwrap();

    // Parse and chunk.
    let mut all_chunks = Vec::new();
    for file in ["auth.rs", "cache.rs"] {
        let source = std::fs::read_to_string(repo_dir.join(file)).unwrap();
        let lang = crate::types::Language::Rust;
        let chunks = parsing::parse_and_chunk(&source, file, lang, &config.indexing).unwrap();
        all_chunks.extend(chunks);
    }

    // Create index directory and stores.
    let index_dir = repo_dir.join(".vera");
    std::fs::create_dir_all(&index_dir).unwrap();

    // Metadata store.
    let metadata_path = index_dir.join("metadata.db");
    let metadata_store = MetadataStore::open(&metadata_path).unwrap();
    metadata_store.insert_chunks(&all_chunks).unwrap();

    // Vector store.
    let vector_path = index_dir.join("vectors.db");
    let vector_store = VectorStore::open(&vector_path, dim).unwrap();
    crate::embedding::test_helpers::embed_and_insert_vectors(&vector_store, &provider, &all_chunks)
        .await;

    // BM25 index.
    let bm25_dir = index_dir.join("bm25");
    let bm25 = Bm25Index::open(&bm25_dir).unwrap();
    bm25.insert_chunks(&all_chunks).unwrap();

    (index_dir, dim)
}

// Regression for issue #20: the intent-prefixed query must only reach the
// vector side. BM25 receives the raw query, so an `intent: ... |` prefix in
// `vector_query` never makes Tantivy parse `intent:` as a missing field.
#[tokio::test]
async fn search_hybrid_keeps_intent_out_of_bm25() {
    use crate::embedding::EmbeddingError;
    use crate::embedding::test_helpers::MockProvider;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;
    // Fail vector search so a successful result can only come from BM25.
    let provider = MockProvider::failing(EmbeddingError::ConnectionError {
        message: "vector search disabled for this regression test".to_string(),
    });

    // bm25_query is the raw user query; vector_query carries the intent prefix.
    let (results, _timings) = search_hybrid(
        &index_dir,
        &provider,
        "authenticate",
        "intent: find auth handlers | authenticate",
        &SearchFilters::default(),
        5,
        60.0,
        dim,
        50,
    )
    .await
    .expect("hybrid search must not error when intent prefix is present");

    assert!(
        results.iter().any(|result| {
            result.file_path == "auth.rs" && result.content.contains("authenticate")
        }),
        "BM25 should return the authenticate chunk for the raw query"
    );
}

/// An embedding provider that spends a known amount of time in `embed_batch`,
/// so the model cost dominates the span the storage stages share with it and
/// its attribution is decidable from the reported stages alone.
struct SlowProvider {
    inner: crate::embedding::test_helpers::MockProvider,
    delay: std::time::Duration,
}

impl crate::embedding::EmbeddingProvider for SlowProvider {
    async fn embed_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, crate::embedding::EmbeddingError> {
        tokio::time::sleep(self.delay).await;
        self.inner.embed_batch(texts).await
    }

    fn expected_dim(&self) -> Option<usize> {
        self.inner.expected_dim()
    }
}

// Regression for issue #105: `embedding` and `vector` were both assigned the
// span that covers the embedding, the store opens, the KNN query and
// hydration, so `--timing` printed one measurement twice.
//
// The split itself is asserted by `charge_vector_span_partitions_the_span`
// below, on values rather than on a clock. It has to be, because the span the
// two stages are cut from is internal to `search_hybrid` and no measurement
// available to this test stands in for it. The wall time of the whole call
// does not: BM25 runs concurrently with the vector arm and is awaited after
// it, so that wall time is the slower arm's. Guarding on the reported BM25
// stage does not repair the substitution, because `bm25_elapsed` starts inside
// the `spawn_blocking` closure and so measures the arm's duration, not when it
// finished relative to the vector arm.
//
// Nor is the storage stage small enough to be bounded by the injected model
// cost. Measured on this fixture, `vector` is 1.6 to 2.9 ms standalone but
// 262.7 ms inside the 794-test suite, against a 300 ms sleep: a 1.14x margin.
// Every bound of that family, absolute or relative to a second run, sits on
// the wrong side of that number, so this test asserts none of them.
//
// What is left here is machine-independent and worth keeping: the provider
// cost reaches `embedding`, the storage cost still reaches `vector`, and the
// two are no longer one value reported twice, which is the symptom #105 was
// filed for.
#[tokio::test]
async fn search_hybrid_charges_embedding_delay_to_embedding_not_vector() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;

    let delay = std::time::Duration::from_millis(300);
    let provider = SlowProvider {
        inner: crate::embedding::test_helpers::MockProvider::new(dim),
        delay,
    };

    let (_results, timings) = search_hybrid(
        &index_dir,
        &provider,
        "authenticate",
        "authenticate",
        &SearchFilters::default(),
        5,
        60.0,
        dim,
        50,
    )
    .await
    .unwrap();

    let embedding = timings.embedding.expect("embedding stage must be reported");
    let vector = timings.vector.expect("vector stage must be reported");

    // `sleep` never returns early, so this is a property of the provider rather
    // than of the machine: the model cost reaches `embedding`.
    assert!(
        embedding >= delay,
        "embedding must carry the provider cost: {embedding:?} < {delay:?}"
    );

    // Taking the embedding out must not empty the stage: the store opens, the
    // KNN query and hydration are still charged to `vector`. This presence has
    // to be asserted before the non-identity below, which a zeroed `vector`
    // would otherwise satisfy.
    assert!(
        vector > std::time::Duration::ZERO,
        "vector must still carry the storage cost: {vector:?}"
    );

    // The reported symptom. Under the bug both stages were assigned the same
    // `Duration` value, so this fails on the bug whatever the machine is doing;
    // reaching it legitimately would need the storage cost to equal the model
    // cost to the nanosecond.
    assert_ne!(
        embedding, vector,
        "the two stages must not be one measurement printed twice"
    );
}

// The split itself, on values. `search_hybrid` cannot expose the span it cuts,
// so this asserts the cut where it is decidable: given a span and the embedding
// measurement taken inside it, the two stages must partition the span. Under
// the bug each stage got the whole span, which this rejects without a clock.
//
// `charge_vector_span` writes both `HybridTimings` fields rather than returning
// two values for the caller to assign, so this test covers the assignment as
// well as the arithmetic. That is deliberate. While the caller did the
// assigning, a caller that kept the remainder on `vector` but charged the whole
// span to `embedding` passed this test and the one above together, which is a
// second way to make `embedding` unattributable. The three durations here are
// distinct so neither field can hold the whole span by coincidence.
#[test]
fn charge_vector_span_partitions_the_span() {
    let span = std::time::Duration::from_millis(500);
    let embed = std::time::Duration::from_millis(300);

    let mut timings = HybridTimings::default();
    charge_vector_span(&mut timings, span, Some(embed));

    assert_eq!(
        (timings.embedding, timings.vector),
        (Some(embed), Some(std::time::Duration::from_millis(200))),
        "embedding takes the model cost and vector takes the remainder"
    );

    // A vector search that failed reports no embedding, so the whole span is
    // storage cost rather than being dropped.
    let mut failed = HybridTimings::default();
    charge_vector_span(&mut failed, span, None);

    assert_eq!(
        (failed.embedding, failed.vector),
        (None, Some(span)),
        "an unmeasured embedding leaves the whole span on vector"
    );
}

#[tokio::test]
async fn search_hybrid_reranked_returns_reranked_results() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::reranker::test_helpers::MockReranker;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;

    let provider = MockProvider::new(dim);
    let reranker = MockReranker::new();

    let (results, _timings) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "authenticate",
        "authenticate",
        &SearchFilters::default(),
        5,
        5,
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();

    assert!(
        !results.is_empty(),
        "should find results for 'authenticate'"
    );

    // Results should be sorted by reranker scores (descending).
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "reranked scores must be descending: {} >= {}",
            results[i - 1].score,
            results[i].score,
        );
    }
}

#[tokio::test]
async fn reranked_tail_scores_remain_descending_across_score_domains() {
    use crate::embedding::test_helpers::MockProvider;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;
    let provider = MockProvider::new(dim);

    let (results, _) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &NegativeReranker,
        "function",
        "function",
        &SearchFilters::default(),
        10,
        1,
        60.0,
        dim,
        1,
        50,
    )
    .await
    .unwrap();

    assert!(results.len() > 1);
    assert!(
        results
            .windows(2)
            .all(|pair| pair[0].score >= pair[1].score)
    );
}

#[tokio::test]
async fn search_hybrid_reranked_skips_without_surplus_and_runs_with_surplus() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::reranker::RerankerError;
    use crate::retrieval::reranker::test_helpers::MockReranker;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;
    let provider = MockProvider::new(dim);
    let reranker = MockReranker::failing(RerankerError::ConnectionError {
        message: "reranker should only be called for a surplus pool".to_string(),
    });
    let fetch_limit = 10;

    let (fused_results, _) = search_hybrid(
        &index_dir,
        &provider,
        "function",
        "function",
        &SearchFilters::default(),
        fetch_limit,
        60.0,
        dim,
        50,
    )
    .await
    .unwrap();
    assert!(!fused_results.is_empty());

    let (_, timings) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "function",
        "function",
        &SearchFilters::default(),
        fetch_limit,
        fused_results.len(),
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();
    assert!(
        timings.reranking.is_none(),
        "reranker should not be called when the fused pool fits the result limit"
    );

    let (_, timings) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "function",
        "function",
        &SearchFilters::default(),
        fetch_limit,
        fused_results.len() - 1,
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();
    assert!(
        timings.reranking.is_some(),
        "reranker should be called when the fused pool exceeds the result limit"
    );
}

#[tokio::test]
async fn search_hybrid_reranked_degrades_on_reranker_failure() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::reranker::RerankerError;
    use crate::retrieval::reranker::test_helpers::MockReranker;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;

    let provider = MockProvider::new(dim);
    let reranker = MockReranker::failing(RerankerError::ConnectionError {
        message: "reranker timeout".to_string(),
    });

    // Should NOT return an error — graceful degradation returns unreranked results.
    let (results, timings) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "authenticate",
        "authenticate",
        &SearchFilters::default(),
        5,
        1,
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();

    assert!(
        timings.reranking.is_some(),
        "the failing reranker must be attempted when the result limit is below the candidate pool"
    );
    assert!(
        !results.is_empty(),
        "should return unreranked results when reranker fails"
    );
}

#[tokio::test]
async fn search_hybrid_reranked_degrades_on_embedding_failure() {
    use crate::embedding::EmbeddingError;
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::reranker::test_helpers::MockReranker;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;

    // Embedding provider that always fails.
    let provider = MockProvider::failing(EmbeddingError::ConnectionError {
        message: "embedding API down".to_string(),
    });
    let reranker = MockReranker::new();

    // Should fall back to BM25-only results (not crash/hang).
    let (results, _timings) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "authenticate",
        "authenticate",
        &SearchFilters::default(),
        5,
        5,
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();

    // BM25 fallback should still find keyword matches.
    assert!(
        !results.is_empty(),
        "should return BM25-only results when embedding API fails"
    );
}

#[tokio::test]
async fn search_hybrid_reranked_respects_limit() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::reranker::test_helpers::MockReranker;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;

    let provider = MockProvider::new(dim);
    let reranker = MockReranker::new();

    let (results, _timings) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "function",
        "function",
        &SearchFilters::default(),
        2,
        2,
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();

    assert!(results.len() <= 2, "should respect the limit of 2");
}

#[tokio::test]
async fn reranking_preserves_unreranked_tail_for_later_filtering() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::apply_filters;
    use crate::retrieval::reranker::test_helpers::MockReranker;
    use crate::types::SearchFilters;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;
    let provider = MockProvider::new(dim);
    let reranker = MockReranker::new();

    let (hybrid_results, _) = search_hybrid(
        &index_dir,
        &provider,
        "function",
        "function",
        &SearchFilters::default(),
        10,
        60.0,
        dim,
        50,
    )
    .await
    .unwrap();
    let first = hybrid_results
        .first()
        .expect("test index should have results");
    let tail_candidate = hybrid_results
        .iter()
        .skip(1)
        .find(|candidate| candidate.file_path != first.file_path)
        .expect("test index should have a tail candidate from another file")
        .clone();

    let (reranked_results, _) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "function",
        "function",
        &SearchFilters::default(),
        10,
        10,
        60.0,
        dim,
        1,
        50,
    )
    .await
    .unwrap();
    let filtered = apply_filters(
        reranked_results,
        &SearchFilters {
            path_glob: vec![tail_candidate.file_path.clone()],
            ..Default::default()
        },
        10,
    );

    assert!(filtered.iter().any(|result| {
        result.file_path == tail_candidate.file_path
            && result.line_start == tail_candidate.line_start
            && result.line_end == tail_candidate.line_end
    }));
}

// ── compute_vector_candidates tests ─────────────────────────────

#[test]
fn compute_vector_candidates_minimum_50() {
    // Even with small limit and multiplier, floor is 50.
    assert!(compute_vector_candidates(5, 3) >= 50);
    assert!(compute_vector_candidates(1, 1) >= 50);
    assert!(compute_vector_candidates(10, 3) >= 50);
}

#[test]
fn compute_vector_candidates_default_limit_10() {
    // For default limit=10 with identifier multiplier (3): max(30, 50) = 50
    assert_eq!(compute_vector_candidates(10, 3), 50);
    // For default limit=10 with NL multiplier (5): max(50, 50) = 50
    assert_eq!(compute_vector_candidates(10, 5), 50);
}

#[test]
fn compute_vector_candidates_scales_with_limit() {
    // For large limit, multiplier should dominate.
    assert_eq!(compute_vector_candidates(100, 3), 300);
    assert_eq!(compute_vector_candidates(100, 5), 500);
}

#[test]
fn path_queries_expand_bm25_candidate_pool() {
    assert_eq!(
        compute_bm25_candidates("turbo.json pipeline configuration", 20),
        100
    );
}

#[test]
fn natural_language_queries_expand_bm25_candidate_pool() {
    assert_eq!(
        compute_bm25_candidates("request validation and schema enforcement", 20),
        80
    );
}

#[test]
fn short_identifier_queries_expand_bm25_candidate_pool() {
    assert_eq!(compute_bm25_candidates("Config", 20), 80);
}

// ── Integration: NL vs identifier query produces different fusion ──

#[tokio::test]
async fn nl_query_uses_different_rrf_k_than_identifier() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::query_classifier::{QueryType, classify_query, params_for_query_type};
    use crate::retrieval::reranker::test_helpers::MockReranker;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;

    let provider = MockProvider::new(dim);
    let reranker = MockReranker::new();

    // Identifier query → uses default k=60.
    let id_query = "authenticate";
    let id_type = classify_query(id_query);
    assert_eq!(id_type, QueryType::Identifier);
    let id_params = params_for_query_type(id_type);

    let (id_results, _) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        id_query,
        id_query,
        &SearchFilters::default(),
        5,
        5,
        id_params.rrf_k,
        dim,
        10,
        compute_vector_candidates(5, id_params.vector_candidate_multiplier),
    )
    .await
    .unwrap();

    // NL query → uses lower k=20.
    let nl_query = "how is authentication handled";
    let nl_type = classify_query(nl_query);
    assert_eq!(nl_type, QueryType::NaturalLanguage);
    let nl_params = params_for_query_type(nl_type);

    let (nl_results, _) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        nl_query,
        nl_query,
        &SearchFilters::default(),
        5,
        5,
        nl_params.rrf_k,
        dim,
        10,
        compute_vector_candidates(5, nl_params.vector_candidate_multiplier),
    )
    .await
    .unwrap();

    // Both should return results (the index has auth content).
    assert!(
        !id_results.is_empty(),
        "identifier query should find results"
    );
    assert!(!nl_results.is_empty(), "NL query should find results");

    // The key assertion: different RRF k was used.
    assert!(
        id_params.rrf_k > nl_params.rrf_k,
        "identifier k ({}) should be greater than NL k ({})",
        id_params.rrf_k,
        nl_params.rrf_k
    );
}
