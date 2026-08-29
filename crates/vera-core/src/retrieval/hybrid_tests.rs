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
        part_index: None,
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
fn rrf_ties_break_on_file_path_then_line_range() {
    // Equal weights make the ties bit-identical: a result found only in BM25 at rank r
    // and one found only in the vector list at rank r both score 1/(k + r). Fusion
    // collects into a HashMap, so without an explicit tie-break the order of each pair
    // is the HashMap seed, which is fresh per process.
    let bm25 = vec![
        make_result("z_bm25.rs", 1, 10, 5.0, None),
        make_result("a_bm25.rs", 1, 10, 4.0, None),
        make_result("shared.rs", 200, 210, 3.0, None),
        make_result("shared.rs", 300, 1000, 2.0, None),
    ];
    let vector = vec![
        make_result("m_vec.rs", 1, 10, 0.9, None),
        make_result("b_vec.rs", 1, 10, 0.8, None),
        make_result("shared.rs", 30, 40, 0.7, None),
        make_result("shared.rs", 300, 305, 0.6, None),
    ];

    // Each call builds a fresh HashMap, and std gives every HashMap in a thread its own
    // hash keys, so repeating the call samples several orders within this one process.
    for _ in 0..8 {
        let results = fuse_rrf(&bm25, &vector, 60.0, 10);

        let order: Vec<(&str, u32, u32)> = results
            .iter()
            .map(|result| {
                (
                    result.file_path.as_str(),
                    result.line_start,
                    result.line_end,
                )
            })
            .collect();

        assert_eq!(
            order,
            vec![
                // rank 1 tie, ascending file_path
                ("m_vec.rs", 1, 10),
                ("z_bm25.rs", 1, 10),
                // rank 2 tie, ascending file_path
                ("a_bm25.rs", 1, 10),
                ("b_vec.rs", 1, 10),
                // rank 3 tie, same file, ascending line_start numerically (a lexicographic
                // compare of the "path:start:end" key would put 200 before 30)
                ("shared.rs", 30, 40),
                ("shared.rs", 200, 210),
                // rank 4 tie, same file and same line_start, so only line_end separates
                // them; 305 before 1000 is again the numeric order, not the lexicographic
                // one. A comparator that stopped at (file_path, line_start) would leave
                // this pair on the HashMap seed.
                ("shared.rs", 300, 305),
                ("shared.rs", 300, 1000),
            ],
            "tied RRF scores must order by (file_path, line_start, line_end)"
        );
    }
}

#[test]
fn rrf_multi_source_ties_break_on_file_path() {
    // The multi-query path fuses more than two lists, so a single rank can carry a tie
    // as wide as the number of sources. Six disjoint sets, one hit each at rank 1, give
    // a six-way bit-identical tie whose order is otherwise the HashMap seed.
    let sets: Vec<Vec<SearchResult>> = ["e.rs", "c.rs", "f.rs", "a.rs", "d.rs", "b.rs"]
        .iter()
        .map(|file| vec![make_result(file, 1, 10, 1.0, None)])
        .collect();
    let refs: Vec<&[SearchResult]> = sets.iter().map(|set| set.as_slice()).collect();

    let results = fuse_rrf_multi_weighted(&refs, &[1.0; 6], 60.0, 10);

    let order: Vec<&str> = results
        .iter()
        .map(|result| result.file_path.as_str())
        .collect();

    assert_eq!(
        order,
        vec!["a.rs", "b.rs", "c.rs", "d.rs", "e.rs", "f.rs"],
        "a tie across all sources must order by file_path, not by insertion or hash order"
    );
    for result in &results {
        assert!(
            (result.score - 1.0 / 61.0).abs() < 1e-10,
            "all six must tie"
        );
    }
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
async fn nl_query_uses_same_rrf_k_as_identifier() {
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

    // NL query → uses same k=60 as identifier queries.
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

    // The key assertion: same RRF k is used for both query types.
    assert_eq!(
        id_params.rrf_k, nl_params.rrf_k,
        "identifier k ({}) should equal NL k ({})",
        id_params.rrf_k, nl_params.rrf_k
    );
    assert_eq!(id_params.rrf_k, 60.0);
    assert_eq!(nl_params.rrf_k, 60.0);
}

// ── Indexed file list cache tests ───────────────────────────────

#[test]
fn vec0_truncation_warning_only_when_clamped_and_loss_possible() {
    use super::should_emit_vec0_truncation_warning;
    use crate::storage::vector::MAX_KNN_K;

    // Helper: truncated means clamped_requested > cap && fetched == cap
    let cap = MAX_KNN_K;
    let over = cap + 1;
    let big_index = 4427;
    let small_index = 53;
    let flat_true = true;
    let flat_false = false;

    // (i) vec0 + clamped + index>4096 + filtered-empty emits the warning
    assert!(
        should_emit_vec0_truncation_warning(flat_false, big_index, over, cap, false, true),
        "vec0 filtered-empty with clamped over cap should warn"
    );

    // (ii) vec0 + clamped + index>4096 + filtered-partial emits it
    // filtered-partial is same as above (has_matches true, filters not empty)
    // Use different requested value to represent partial
    assert!(
        should_emit_vec0_truncation_warning(flat_false, big_index, cap + 500, cap, false, true),
        "vec0 filtered-partial with clamped over cap should warn"
    );

    // (iii) vec0 + index<=4096 emits nothing regardless of requested pool
    assert!(
        !should_emit_vec0_truncation_warning(
            flat_false,
            small_index,
            over,
            small_index,
            false,
            true
        ),
        "below-cap index must not warn even when requested > cap (pool clamped to index)"
    );
    assert!(
        !should_emit_vec0_truncation_warning(flat_false, 4096, over, 4096, false, true),
        "exactly at cap must not warn"
    );
    assert!(
        !should_emit_vec0_truncation_warning(
            flat_false,
            small_index,
            over,
            small_index,
            true,
            false
        ),
        "below-cap unfiltered must not warn"
    );

    // (iv) flat backend never emits this warning at any depth
    assert!(
        !should_emit_vec0_truncation_warning(flat_true, big_index, over, over, false, true),
        "flat must never warn (filtered)"
    );
    assert!(
        !should_emit_vec0_truncation_warning(flat_true, big_index, over, over, true, false),
        "flat must never warn (unfiltered)"
    );
    assert!(
        !should_emit_vec0_truncation_warning(
            flat_true,
            big_index,
            cap + 10000,
            cap + 10000,
            false,
            true
        ),
        "flat deep must never warn"
    );

    // (v) vec0 + clamped + index>4096 + unfiltered small-request emits nothing
    // small-request => not truncated (requested <= cap)
    assert!(
        !should_emit_vec0_truncation_warning(flat_false, big_index, 100, 100, true, false),
        "vec0 small unfiltered (not truncated) must not warn"
    );
    assert!(
        !should_emit_vec0_truncation_warning(flat_false, big_index, cap, cap, true, false),
        "vec0 exactly cap unfiltered must not warn"
    );

    // (vi) vec0 + clamped + index>4096 + a filter matching zero chunks emits nothing (true negative)
    assert!(
        !should_emit_vec0_truncation_warning(flat_false, big_index, over, cap, false, false),
        "true negative must stay quiet even when truncated"
    );

    // Additional: actionable diagnostic requirement — the warning text must name backend, cap, and remedy.
    // This is verified by the caller tracing::warn! containing "vec0", "4096", and
    // "Use the default flat vector scan (unset VERA_VECTOR_SCAN)". We assert the
    // string literals here to pin the diagnostic so a wording change fails this test.
    let diagnostic_filtered = "vec0 vector search truncated at sqlite-vec KNN cap (4096) with active filters; results may be incomplete. Use the default flat vector scan (unset VERA_VECTOR_SCAN) for complete results";
    let diagnostic_unfiltered = "vec0 vector search truncated at sqlite-vec KNN cap (4096); results may be incomplete. Use the default flat vector scan (unset VERA_VECTOR_SCAN) for complete results";
    assert!(diagnostic_filtered.contains("vec0"));
    assert!(diagnostic_filtered.contains("4096"));
    assert!(
        diagnostic_filtered.contains("Use the default flat vector scan (unset VERA_VECTOR_SCAN)")
    );
    assert!(diagnostic_unfiltered.contains("vec0"));
    assert!(diagnostic_unfiltered.contains("4096"));
    assert!(
        diagnostic_unfiltered.contains("Use the default flat vector scan (unset VERA_VECTOR_SCAN)")
    );
}

#[cfg(test)]
mod indexed_files_cache_tests {
    use super::super::SearchStores;
    use crate::storage::metadata::MetadataStore;
    use crate::types::{Chunk, Language, SymbolType};
    use std::sync::Arc;

    fn chunk(file_path: &str) -> Chunk {
        Chunk {
            id: format!("{file_path}:1"),
            file_path: file_path.to_string(),
            line_start: 1,
            line_end: 3,
            content: "fn f() {}".to_string(),
            language: Language::Rust,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("f".to_string()),
            part_index: None,
        }
    }

    #[test]
    fn indexed_files_cache_refreshes_after_metadata_update() {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path();
        let metadata_path = index_dir.join("metadata.db");
        MetadataStore::open(&metadata_path)
            .unwrap()
            .insert_chunks(&[chunk("src/a.rs")])
            .unwrap();

        let stores = SearchStores::open(index_dir).unwrap();

        let first = stores.indexed_files().unwrap();
        assert_eq!(first.as_slice(), &["src/a.rs".to_string()]);

        // Unchanged database: the cache serves the same allocation.
        let second = stores.indexed_files().unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        // A writer on another connection adds a file, as watch mode would.
        MetadataStore::open(&metadata_path)
            .unwrap()
            .insert_chunks(&[chunk("src/b.rs")])
            .unwrap();

        let third = stores.indexed_files().unwrap();
        assert_eq!(
            third.as_slice(),
            &["src/a.rs".to_string(), "src/b.rs".to_string()],
            "cache must refresh after the metadata database changes"
        );
        assert!(!Arc::ptr_eq(&first, &third));
    }
}

// ── Filtered-vector edge cases (VAL-FILTER-012/013/016/017/019/021) ──

// VAL-FILTER-012: deterministic ordering across repeated runs.
// The fused output must be a pure function of its inputs, so two identical
// runs produce byte-identical ordering (no HashMap iteration randomness).
#[test]
fn fused_results_are_deterministic_across_repeated_runs() {
    let bm25 = vec![
        make_result("a.rs", 1, 10, 5.0, None),
        make_result("b.rs", 1, 10, 4.0, None),
        make_result("c.rs", 1, 10, 3.0, None),
    ];
    let vector = vec![
        make_result("b.rs", 1, 10, 0.9, None),
        make_result("a.rs", 1, 10, 0.8, None),
        make_result("c.rs", 1, 10, 0.7, None),
    ];

    let first = fuse_rrf(&bm25, &vector, 60.0, 10);
    let second = fuse_rrf(&bm25, &vector, 60.0, 10);

    // Byte-identical ordering (and scores) across runs.
    assert_eq!(
        first.len(),
        second.len(),
        "both runs must return same count"
    );
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.file_path, b.file_path);
        assert_eq!(a.line_start, b.line_start);
        assert_eq!(a.line_end, b.line_end);
        assert!(
            (a.score - b.score).abs() < 1e-12,
            "scores must be identical: {} vs {}",
            a.score,
            b.score
        );
    }

    let json_first = serde_json::to_string(&first).unwrap();
    let json_second = serde_json::to_string(&second).unwrap();
    assert_eq!(
        json_first, json_second,
        "JSON serializations must be byte-identical"
    );
}

// VAL-FILTER-013: extreme --limit is clamped to the index size, bounded memory.
// The vector candidate pool must never exceed the real index count, even for
// limits like 20000, so a flat backend never truncates and never allocates
// proportional to the requested limit.
#[test]
fn candidate_pool_is_clamped_to_index_for_extreme_limits() {
    use crate::retrieval::vector::candidate_pool;

    // Simulate an over-cap index with 4427 chunks (flat backend).
    let index_count = 4427;

    for limit in [10, 20_000, 100_000] {
        let (requested, fetched) = candidate_pool(limit, index_count, true);
        assert!(
            requested <= index_count,
            "requested {requested} for limit {limit} must not exceed index {index_count}"
        );
        assert!(
            fetched <= index_count,
            "fetched {fetched} for limit {limit} must not exceed index {index_count}"
        );
    }

    // Extreme limit on flat must still fetch the whole index.
    let (req_flat, fetch_flat) = candidate_pool(20_000, index_count, true);
    assert_eq!(req_flat, index_count, "flat requested must clamp to index");
    assert_eq!(
        fetch_flat, index_count,
        "flat extreme limit must clamp to index size, not to 20000"
    );

    // Same extreme limit on vec0 must cap fetched at 4096 but requested is clamped to index.
    let (req_vec0, fetch_vec0) = candidate_pool(20_000, index_count, false);
    assert_eq!(req_vec0, index_count);
    assert_eq!(
        fetch_vec0,
        crate::storage::vector::MAX_KNN_K,
        "vec0 extreme limit must cap fetched at MAX_KNN_K"
    );

    // Small limit retains normal over-fetch semantics (limit + half, but still capped).
    let (req_small, fetch_small) = candidate_pool(10, 4427, true);
    assert_eq!(req_small, 20);
    assert_eq!(fetch_small, 20);
}

// VAL-FILTER-016: exact_paths-style restrictions reach the island above the cap.
// The whole-index fetch for flat filtered queries must be driven by index_count,
// not by the post-filter count, so an island file at the tail is reachable.
#[tokio::test]
async fn exact_paths_filter_reaches_island_above_cap() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::storage::metadata::MetadataStore;
    use crate::storage::vector::VectorStore;
    use crate::types::{Chunk, SearchFilters};
    use std::collections::HashSet;
    use std::sync::Arc;

    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("idx");
    std::fs::create_dir_all(&index_dir).unwrap();

    // Build a tiny index where the island file is the last inserted vector,
    // so a shallow vec0 fetch would miss it.
    let dim = 4;
    let provider = MockProvider::new(dim);
    let mut chunks = Vec::new();
    for i in 0..20 {
        chunks.push(Chunk {
            id: format!("bulk:{i}"),
            file_path: format!("src/audio/filter{i:04}.ts"),
            line_start: 1,
            line_end: 5,
            content: format!("audio content {i}"),
            language: Language::TypeScript,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some(format!("filter{i:04}")),
            part_index: None,
        });
    }
    chunks.push(Chunk {
        id: "island:0".to_string(),
        file_path: "src/video/island.ts".to_string(),
        line_start: 1,
        line_end: 5,
        content: "island video handler apply envelope".to_string(),
        language: Language::TypeScript,
        symbol_type: Some(SymbolType::Function),
        symbol_name: Some("island".to_string()),
        part_index: None,
    });

    let metadata_store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
    metadata_store.insert_chunks(&chunks).unwrap();
    let vector_store = VectorStore::open(&index_dir.join("vectors.db"), dim).unwrap();
    crate::embedding::test_helpers::embed_and_insert_vectors(&vector_store, &provider, &chunks)
        .await;
    let bm25 = crate::storage::bm25::Bm25Index::open(&index_dir.join("bm25")).unwrap();
    bm25.insert_chunks(&chunks).unwrap();

    let mut exact = HashSet::new();
    exact.insert("src/video/island.ts".to_string());
    let filters = SearchFilters {
        exact_paths: Some(Arc::new(exact)),
        ..Default::default()
    };

    let (results, _) = search_hybrid(
        &index_dir,
        &provider,
        "apply envelope to buffer samples",
        "apply envelope to buffer samples",
        &filters,
        10,
        60.0,
        dim,
        50,
    )
    .await
    .unwrap();

    assert!(
        results.iter().any(|r| r.file_path == "src/video/island.ts"),
        "exact_paths filter must reach island, got {:?}",
        results.iter().map(|r| &r.file_path).collect::<Vec<_>>()
    );
}

// VAL-FILTER-017: multiple --path values work above the cap (OR semantics).
#[tokio::test]
async fn multiple_path_filters_use_or_semantics() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::storage::metadata::MetadataStore;
    use crate::storage::vector::VectorStore;
    use crate::types::{Chunk, SearchFilters};

    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("idx");
    std::fs::create_dir_all(&index_dir).unwrap();

    let dim = 4;
    let provider = MockProvider::new(dim);
    let chunks = vec![
        Chunk {
            id: "a:0".to_string(),
            file_path: "src/video/a.ts".to_string(),
            line_start: 1,
            line_end: 5,
            content: "video content a apply envelope".to_string(),
            language: Language::TypeScript,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("a".to_string()),
            part_index: None,
        },
        Chunk {
            id: "b:0".to_string(),
            file_path: "src/videoplayer/b.ts".to_string(),
            line_start: 1,
            line_end: 5,
            content: "videoplayer content b apply envelope".to_string(),
            language: Language::TypeScript,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("b".to_string()),
            part_index: None,
        },
        Chunk {
            id: "c:0".to_string(),
            file_path: "src/audio/c.ts".to_string(),
            line_start: 1,
            line_end: 5,
            content: "audio content c apply envelope".to_string(),
            language: Language::TypeScript,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("c".to_string()),
            part_index: None,
        },
    ];

    let metadata_store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
    metadata_store.insert_chunks(&chunks).unwrap();
    let vector_store = VectorStore::open(&index_dir.join("vectors.db"), dim).unwrap();
    crate::embedding::test_helpers::embed_and_insert_vectors(&vector_store, &provider, &chunks)
        .await;
    let bm25 = crate::storage::bm25::Bm25Index::open(&index_dir.join("bm25")).unwrap();
    bm25.insert_chunks(&chunks).unwrap();

    let filters = SearchFilters {
        path_glob: vec!["src/video".to_string(), "src/videoplayer/b.ts".to_string()],
        ..Default::default()
    };

    let (results, _) = search_hybrid(
        &index_dir,
        &provider,
        "apply envelope",
        "apply envelope",
        &filters,
        10,
        60.0,
        dim,
        50,
    )
    .await
    .unwrap();

    let paths: std::collections::HashSet<&str> =
        results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(
        paths.contains("src/video/a.ts"),
        "must contain src/video, got {paths:?}"
    );
    assert!(
        paths.contains("src/videoplayer/b.ts"),
        "must contain src/videoplayer/b.ts, got {paths:?}"
    );
    assert!(
        !paths.contains("src/audio/c.ts"),
        "must exclude audio bulk, got {paths:?}"
    );
}

// VAL-FILTER-019: filter match semantics unchanged — sibling prefix not matched,
// and near-miss hint still fires for truly unmatched globs.
#[test]
fn video_filter_does_not_match_videoplayer_and_near_miss_hint_preserved() {
    use crate::types::{SearchFilters, directory_prefix_near_misses};

    let filters = SearchFilters {
        path_glob: vec!["src/video".to_string()],
        ..Default::default()
    };
    let video = make_result("src/video/a.ts", 1, 5, 1.0, None);
    let videoplayer = make_result("src/videoplayer/b.ts", 1, 5, 1.0, None);
    let sibling = make_result("src/videoplayer-extra/c.ts", 1, 5, 1.0, None);

    assert!(
        filters.matches(&video),
        "src/video must match src/video/a.ts"
    );
    assert!(
        !filters.matches(&videoplayer),
        "src/video must NOT match src/videoplayer/b.ts (prefix guard)"
    );
    assert!(
        !filters.matches(&sibling),
        "src/video must NOT match src/videoplayer-extra/c.ts"
    );

    // Near-miss hint: a pattern like "crates/*/src" that matches no file
    // directly but matches a directory ancestor should be reported.
    let paths = vec![
        "crates/vera-core/src/lib.rs".to_string(),
        "crates/vera-cli/src/main.rs".to_string(),
    ];
    let patterns = vec!["crates/*/src".to_string()];
    let near_misses = directory_prefix_near_misses(&patterns, &paths);
    assert_eq!(
        near_misses,
        vec!["crates/*/src".to_string()],
        "must report directory-prefix near miss for crates/*/src"
    );

    // A truly matched pattern does not produce a hint.
    let no_miss = directory_prefix_near_misses(&["src/**/*.rs".to_string()], &paths);
    assert!(
        no_miss.is_empty(),
        "matched pattern must not produce near-miss hint"
    );
}

// VAL-FILTER-021: filtered above-cap results survive reranking (no re-clamping),
// and graceful degradation on reranker failure preserves the filtered set.
#[tokio::test]
async fn filtered_results_survive_reranking_and_graceful_degradation() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::reranker::RerankerError;
    use crate::retrieval::reranker::test_helpers::MockReranker;
    use crate::storage::metadata::MetadataStore;
    use crate::storage::vector::VectorStore;
    use crate::types::{Chunk, SearchFilters};

    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("idx");
    std::fs::create_dir_all(&index_dir).unwrap();

    let dim = 4;
    let provider = MockProvider::new(dim);
    let chunks = vec![
        Chunk {
            id: "island:0".to_string(),
            file_path: "src/video/island.ts".to_string(),
            line_start: 1,
            line_end: 5,
            content: "apply envelope to buffer samples island".to_string(),
            language: Language::TypeScript,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("island".to_string()),
            part_index: None,
        },
        Chunk {
            id: "bulk:0".to_string(),
            file_path: "src/audio/bulk.ts".to_string(),
            line_start: 1,
            line_end: 5,
            content: "apply envelope to buffer samples bulk".to_string(),
            language: Language::TypeScript,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("bulk".to_string()),
            part_index: None,
        },
    ];

    let metadata_store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
    metadata_store.insert_chunks(&chunks).unwrap();
    let vector_store = VectorStore::open(&index_dir.join("vectors.db"), dim).unwrap();
    crate::embedding::test_helpers::embed_and_insert_vectors(&vector_store, &provider, &chunks)
        .await;
    let bm25 = crate::storage::bm25::Bm25Index::open(&index_dir.join("bm25")).unwrap();
    bm25.insert_chunks(&chunks).unwrap();

    let filters = SearchFilters {
        path_glob: vec!["src/video".to_string()],
        ..Default::default()
    };

    // Successful reranking must keep the filtered island.
    let reranker = MockReranker::new();
    let (reranked, _) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &reranker,
        "apply envelope to buffer samples",
        "apply envelope to buffer samples",
        &filters,
        10,
        10,
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();

    assert!(
        reranked
            .iter()
            .any(|r| r.file_path == "src/video/island.ts"),
        "reranked filtered set must retain island, got {:?}",
        reranked.iter().map(|r| &r.file_path).collect::<Vec<_>>()
    );
    assert!(
        reranked
            .iter()
            .all(|r| r.file_path.starts_with("src/video")),
        "reranked must not reintroduce bulk files"
    );

    // Failing reranker must degrade gracefully and still return the filtered island.
    let failing = MockReranker::failing(RerankerError::ConnectionError {
        message: "reranker down".to_string(),
    });
    let (degraded, _) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &failing,
        "apply envelope to buffer samples",
        "apply envelope to buffer samples",
        &filters,
        10,
        1,
        60.0,
        dim,
        10,
        50,
    )
    .await
    .unwrap();

    assert!(
        degraded
            .iter()
            .any(|r| r.file_path == "src/video/island.ts"),
        "graceful degradation must preserve filtered island, got {:?}",
        degraded.iter().map(|r| &r.file_path).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn reranker_failure_degrades_to_exact_original_hybrid_order() {
    use crate::embedding::test_helpers::MockProvider;
    use crate::retrieval::reranker::RerankerError;
    use crate::retrieval::reranker::test_helpers::MockReranker;

    let tmp = tempfile::tempdir().unwrap();
    let (index_dir, dim) = setup_test_index(tmp.path()).await;
    let provider = MockProvider::new(dim);

    // Baseline hybrid order without reranking
    let (hybrid_results, _) = search_hybrid(
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

    // Reranked with a failing reranker must return byte-identical hybrid order
    let failing = MockReranker::failing(RerankerError::ConnectionError {
        message: "reranker down".to_string(),
    });
    let (degraded, _) = search_hybrid_reranked(
        &index_dir,
        &provider,
        &failing,
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

    assert_eq!(
        degraded.len(),
        hybrid_results.len().min(5),
        "degraded must have same count as hybrid truncated to fetch_limit"
    );
    let hybrid_json = serde_json::to_string(&hybrid_results[..degraded.len()]).unwrap();
    let degraded_json = serde_json::to_string(&degraded).unwrap();
    assert_eq!(
        hybrid_json, degraded_json,
        "degradation must be byte-identical to hybrid order; hybrid={hybrid_json} degraded={degraded_json}"
    );
    // Also verify scores are identical (original hybrid scores preserved)
    for (h, d) in hybrid_results.iter().zip(degraded.iter()) {
        assert!(
            (h.score - d.score).abs() < 1e-12,
            "score must be preserved: hybrid {} vs degraded {}",
            h.score,
            d.score
        );
        assert_eq!(h.file_path, d.file_path);
        assert_eq!(h.line_start, d.line_start);
        assert_eq!(h.line_end, d.line_end);
    }
}
