use super::*;
use crate::test_env::EnvVarGuard;
use crate::types::{Language, SymbolType};

/// Helper to create a SearchResult with given parameters.
fn make_result(
    file: &str,
    line_start: u32,
    line_end: u32,
    score: f64,
    symbol_name: Option<&str>,
    content: &str,
) -> SearchResult {
    SearchResult {
        file_path: file.to_string(),
        line_start,
        line_end,
        content: content.to_string(),
        language: Language::Rust,
        score,
        symbol_name: symbol_name.map(|s| s.to_string()),
        symbol_type: Some(SymbolType::Function),
    }
}

// ── RerankerConfig tests ─────────────────────────────────────────

#[test]
fn config_from_values() {
    let config = RerankerConfig::new(
        "https://api.example.com/v1".to_string(),
        "model-1".to_string(),
        "key-123".to_string(),
    );
    assert_eq!(config.base_url, "https://api.example.com/v1");
    assert_eq!(config.model_id, "model-1");
    assert_eq!(config.timeout, Duration::from_secs(30));
    assert_eq!(config.max_retries, 2);
}

#[test]
fn config_with_timeout() {
    let config = RerankerConfig::new(
        "https://api.example.com/v1".to_string(),
        "model-1".to_string(),
        "key-123".to_string(),
    )
    .with_timeout(Duration::from_secs(10));
    assert_eq!(config.timeout, Duration::from_secs(10));
}

#[test]
fn api_reranker_legacy_constructor_uses_environment_batch_size() {
    let _env = EnvVarGuard::set(&[("VERA_MAX_RERANK_BATCH", "7")]);
    let config = RerankerConfig::new(
        "https://api.example.com/v1".to_string(),
        "model-1".to_string(),
        "key-123".to_string(),
    );

    let reranker = ApiReranker::new(config).unwrap();

    assert_eq!(reranker.max_rerank_batch, 7);
}

#[test]
fn config_with_max_retries() {
    let config = RerankerConfig::new(
        "https://api.example.com/v1".to_string(),
        "model-1".to_string(),
        "key-123".to_string(),
    )
    .with_max_retries(5);
    assert_eq!(config.max_retries, 5);
}

// ── MockReranker tests ───────────────────────────────────────────

#[tokio::test]
async fn mock_reranker_returns_scores_for_all_documents() {
    let reranker = test_helpers::MockReranker::new();
    let docs = vec![
        "doc 1".to_string(),
        "doc 2".to_string(),
        "doc 3".to_string(),
    ];

    let scores = reranker.rerank("query", &docs).await.unwrap();

    assert_eq!(scores.len(), 3);
}

#[tokio::test]
async fn mock_reranker_scores_are_descending() {
    let reranker = test_helpers::MockReranker::new();
    let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let scores = reranker.rerank("query", &docs).await.unwrap();

    for i in 1..scores.len() {
        assert!(
            scores[i - 1].relevance_score >= scores[i].relevance_score,
            "scores must be descending"
        );
    }
}

#[tokio::test]
async fn mock_reranker_empty_documents() {
    let reranker = test_helpers::MockReranker::new();

    let scores = reranker.rerank("query", &[]).await.unwrap();

    assert!(scores.is_empty());
}

#[tokio::test]
async fn mock_reranker_connection_error() {
    let reranker = test_helpers::MockReranker::failing(RerankerError::ConnectionError {
        message: "timeout".to_string(),
    });

    let result = reranker.rerank("query", &["doc".to_string()]).await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RerankerError::ConnectionError { .. }
    ));
}

#[tokio::test]
async fn mock_reranker_auth_error() {
    let reranker = test_helpers::MockReranker::failing(RerankerError::AuthError {
        message: "invalid key".to_string(),
    });

    let result = reranker.rerank("query", &["doc".to_string()]).await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RerankerError::AuthError { .. }
    ));
}

// ── rerank_results tests ─────────────────────────────────────────

#[tokio::test]
async fn rerank_results_reorders_by_reranker_scores() {
    let reranker = test_helpers::MockReranker::new();
    let results = vec![
        make_result("a.rs", 1, 10, 0.5, Some("func_a"), "fn func_a() {}"),
        make_result("b.rs", 1, 10, 0.4, Some("func_b"), "fn func_b() {}"),
        make_result("c.rs", 1, 10, 0.3, Some("func_c"), "fn func_c() {}"),
    ];

    let reranked = rerank_results(&reranker, "test query", &results, 10)
        .await
        .unwrap();

    assert_eq!(reranked.len(), 3);
    // Scores should be descending.
    for i in 1..reranked.len() {
        assert!(
            reranked[i - 1].score >= reranked[i].score,
            "reranked scores must be descending"
        );
    }
}

#[tokio::test]
async fn rerank_results_replaces_original_scores() {
    let reranker = test_helpers::MockReranker::new();
    let results = vec![
        make_result("a.rs", 1, 10, 100.0, None, "fn a() {}"),
        make_result("b.rs", 1, 10, 50.0, None, "fn b() {}"),
    ];

    let reranked = rerank_results(&reranker, "query", &results, 10)
        .await
        .unwrap();

    // Original scores (100.0, 50.0) should be replaced by reranker scores.
    for result in &reranked {
        assert!(
            result.score <= 1.0,
            "reranker scores should replace original (was {})",
            result.score
        );
    }
}

#[tokio::test]
async fn rerank_results_preserves_metadata() {
    let reranker = test_helpers::MockReranker::new();
    let results = vec![make_result(
        "auth.rs",
        5,
        20,
        0.8,
        Some("authenticate"),
        "fn authenticate() {}",
    )];

    let reranked = rerank_results(&reranker, "auth", &results, 10)
        .await
        .unwrap();

    assert_eq!(reranked.len(), 1);
    let r = &reranked[0];
    assert_eq!(r.file_path, "auth.rs");
    assert_eq!(r.line_start, 5);
    assert_eq!(r.line_end, 20);
    assert_eq!(r.symbol_name.as_deref(), Some("authenticate"));
    assert_eq!(r.symbol_type, Some(SymbolType::Function));
    assert_eq!(r.language, Language::Rust);
    assert!(!r.content.is_empty());
}

#[tokio::test]
async fn rerank_results_respects_top_n() {
    let reranker = test_helpers::MockReranker::new();
    let results = vec![
        make_result("a.rs", 1, 10, 0.9, None, "fn a() {}"),
        make_result("b.rs", 1, 10, 0.8, None, "fn b() {}"),
        make_result("c.rs", 1, 10, 0.7, None, "fn c() {}"),
        make_result("d.rs", 1, 10, 0.6, None, "fn d() {}"),
        make_result("e.rs", 1, 10, 0.5, None, "fn e() {}"),
    ];

    // Only rerank top 2 candidates.
    let reranked = rerank_results(&reranker, "query", &results, 2)
        .await
        .unwrap();

    assert_eq!(
        reranked.len(),
        2,
        "should only return top_n reranked results"
    );
}

#[tokio::test]
async fn rerank_results_empty_input() {
    let reranker = test_helpers::MockReranker::new();

    let reranked = rerank_results(&reranker, "query", &[], 10).await.unwrap();

    assert!(reranked.is_empty());
}

#[tokio::test]
async fn rerank_results_propagates_connection_error() {
    let reranker = test_helpers::MockReranker::failing(RerankerError::ConnectionError {
        message: "timeout".to_string(),
    });
    let results = vec![make_result("a.rs", 1, 10, 0.5, None, "fn a() {}")];

    let result = rerank_results(&reranker, "query", &results, 10).await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RerankerError::ConnectionError { .. }
    ));
}

#[tokio::test]
async fn rerank_results_propagates_api_error() {
    let reranker = test_helpers::MockReranker::failing(RerankerError::ApiError {
        status: 500,
        message: "internal error".to_string(),
    });
    let results = vec![make_result("a.rs", 1, 10, 0.5, None, "fn a() {}")];

    let result = rerank_results(&reranker, "query", &results, 10).await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RerankerError::ApiError { status: 500, .. }
    ));
}

// ── format_for_reranker tests ────────────────────────────────────

#[test]
fn format_includes_symbol_info() {
    let result = make_result("lib.rs", 1, 10, 0.5, Some("my_func"), "fn my_func() {}");
    let formatted = format_for_reranker(&result);

    assert!(formatted.contains("Symbol: my_func"));
    assert!(formatted.contains("Symbol type: function"));
    assert!(formatted.contains("Filename: lib.rs"));
    assert!(formatted.contains("File: lib.rs"));
    assert!(formatted.contains("fn my_func() {}"));
}

#[test]
fn format_without_symbol_info() {
    let mut result = make_result("lib.rs", 1, 10, 0.5, None, "some code");
    result.symbol_type = None;
    let formatted = format_for_reranker(&result);

    assert!(formatted.contains("Filename: lib.rs"));
    assert!(formatted.contains("File: lib.rs"));
    assert!(formatted.contains("some code"));
    assert!(!formatted.contains("Symbol type:"));
}

// ── sanitize_error_message tests ─────────────────────────────────

#[test]
fn sanitize_truncates_long_messages() {
    let long_msg = "a".repeat(1000);
    let sanitized = sanitize_error_message(&long_msg);
    assert!(sanitized.len() <= 500);
}

#[test]
fn sanitize_multibyte_utf8_boundary() {
    // Create a string with multi-byte chars near the 500-byte boundary.
    // Each '🦀' is 4 bytes. 125 crabs = 500 bytes exactly, but place
    // the boundary right in the middle of a multi-byte sequence.
    let msg = "a".repeat(499) + "🦀"; // 499 + 4 = 503 bytes
    let sanitized = sanitize_error_message(&msg);
    // Should truncate before the crab emoji, not panic.
    assert!(sanitized.len() <= 500);
    assert!(sanitized.is_char_boundary(sanitized.len()));
}

#[test]
fn sanitize_empty_message() {
    let sanitized = sanitize_error_message("");
    assert_eq!(sanitized, "no details available");
}

// ── ApiReranker endpoint URL tests ───────────────────────────────

#[test]
fn endpoint_url_builds_correctly() {
    let config = RerankerConfig::new(
        "https://api.siliconflow.com/v1".to_string(),
        "model".to_string(),
        "key".to_string(),
    );
    let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();
    assert_eq!(
        reranker.endpoint_url(),
        "https://api.siliconflow.com/v1/rerank"
    );
}

#[test]
fn endpoint_url_strips_trailing_slash() {
    let config = RerankerConfig::new(
        "https://api.siliconflow.com/v1/".to_string(),
        "model".to_string(),
        "key".to_string(),
    );
    let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();
    assert_eq!(
        reranker.endpoint_url(),
        "https://api.siliconflow.com/v1/rerank"
    );
}

// ── truncate_document tests ─────────────────────────────────────

#[test]
fn truncate_document_short_passthrough() {
    assert_eq!(truncate_document("hello", 100), "hello");
}

#[test]
fn truncate_document_cuts_at_newline() {
    let doc = "line1\nline2\nline3\nline4";
    let result = truncate_document(doc, 15);
    assert_eq!(result, "line1\nline2");
}

#[test]
fn truncate_document_no_newline() {
    let doc = "abcdefghij";
    let result = truncate_document(doc, 5);
    assert_eq!(result, "abcde");
}

#[test]
fn truncate_document_zero_max_passthrough() {
    assert_eq!(truncate_document("hello", 0), "hello");
}

#[tokio::test]
async fn api_reranker_unreachable_endpoint() {
    let config = RerankerConfig::new(
        "http://127.0.0.1:19999".to_string(),
        "model".to_string(),
        "key".to_string(),
    )
    .with_timeout(Duration::from_millis(500))
    .with_max_retries(0);

    let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();
    let result = reranker.rerank("test", &["document".to_string()]).await;

    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), RerankerError::ConnectionError { .. }),
        "unreachable endpoint should return connection error"
    );
}

// ── Voyage wire-format compatibility ────────────────────────────

#[test]
fn voyage_base_url_detection() {
    assert!(is_voyage_base_url("https://api.voyageai.com/v1"));
    assert!(is_voyage_base_url("https://api.voyageai.com/v1/"));
    assert!(is_voyage_base_url("https://api.voyageai.com"));
    assert!(!is_voyage_base_url("https://api.siliconflow.com/v1"));
    assert!(!is_voyage_base_url("https://api.jina.ai/v1"));
    assert!(!is_voyage_base_url("https://api.cohere.ai/v1"));
    assert!(!is_voyage_base_url(""));
}

#[test]
fn api_reranker_detects_voyage() {
    let voyage = ApiReranker::new_with_max_rerank_batch(
        RerankerConfig::new(
            "https://api.voyageai.com/v1".to_string(),
            "rerank-2".to_string(),
            "k".to_string(),
        ),
        20,
    )
    .unwrap();
    assert!(voyage.is_voyage);

    let other = ApiReranker::new_with_max_rerank_batch(
        RerankerConfig::new(
            "https://api.siliconflow.com/v1".to_string(),
            "Qwen/Qwen3-Reranker-8B".to_string(),
            "k".to_string(),
        ),
        20,
    )
    .unwrap();
    assert!(!other.is_voyage);
}

#[test]
fn rerank_request_serializes_top_n_for_non_voyage() {
    let docs = vec!["a".to_string()];
    let body = RerankRequest {
        model: "Qwen/Qwen3-Reranker-8B",
        query: "q",
        documents: &docs,
        top: TopLimit::TopN { top_n: 5 },
        return_documents: Some(false),
    };
    let json = serde_json::to_value(&body).unwrap();
    assert_eq!(json["top_n"], serde_json::json!(5));
    assert!(
        json.get("top_k").is_none(),
        "non-voyage providers must not receive top_k: {json}"
    );
}

#[test]
fn rerank_request_serializes_top_k_for_voyage() {
    let docs = vec!["a".to_string()];
    let body = RerankRequest {
        model: "rerank-2",
        query: "q",
        documents: &docs,
        top: TopLimit::TopK { top_k: 5 },
        return_documents: Some(false),
    };
    let json = serde_json::to_value(&body).unwrap();
    assert_eq!(json["top_k"], serde_json::json!(5));
    assert!(
        json.get("top_n").is_none(),
        "voyage must not receive top_n: {json}"
    );
}

#[test]
fn rerank_response_accepts_results_field() {
    let payload = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
    let resp: RerankResponse = serde_json::from_str(payload).unwrap();
    assert_eq!(resp.results.len(), 1);
    assert_eq!(resp.results[0].index, 0);
}

#[test]
fn rerank_response_accepts_data_field_alias() {
    let payload = r#"{"data":[{"index":2,"relevance_score":0.42}]}"#;
    let resp: RerankResponse = serde_json::from_str(payload).unwrap();
    assert_eq!(resp.results.len(), 1);
    assert_eq!(resp.results[0].index, 2);
}
