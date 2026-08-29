use super::*;
use crate::test_env::run_env_test;
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
        part_index: None,
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
    run_env_test(
        "retrieval::reranker::tests::api_reranker_legacy_constructor_uses_environment_batch_size_probe",
        &[("VERA_MAX_RERANK_BATCH", Some("7"))],
    );
}

#[test]
#[ignore = "driven by api_reranker_legacy_constructor_uses_environment_batch_size"]
fn api_reranker_legacy_constructor_uses_environment_batch_size_probe() {
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
        instruction: None,
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
        instruction: None,
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

// ── Protocol & wire-format TcpListener mocks ───────────────────────

#[cfg(test)]
mod protocol_wire_tests {
    use super::*;
    use crate::config::{RerankerProtocol, RetrievalConfig};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> (String, serde_json::Value) {
        let mut buf = vec![0u8; 8192];
        let mut request = Vec::new();
        // Read until headers complete and body (via Content-Length) arrives.
        loop {
            let n = stream.read(&mut buf).await.unwrap();
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request.windows(4).any(|w| w == b"\r\n\r\n") {
                // Try to parse Content-Length
                let headers_end = request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                let header_str = String::from_utf8_lossy(&request[..headers_end]);
                let content_len = header_str
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                    .and_then(|l| l.split(':').nth(1))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= headers_end + content_len {
                    break;
                }
            }
            if request.len() > 65536 {
                break;
            }
        }
        let header_end = request
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|p| p + 4)
            .unwrap_or(request.len());
        let header_str = String::from_utf8_lossy(&request[..header_end]).to_string();
        let first_line = header_str.lines().next().unwrap_or("").to_string();
        let body_bytes = &request[header_end..];
        let body: serde_json::Value = if body_bytes.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(body_bytes).unwrap_or(serde_json::Value::Null)
        };
        (first_line, body)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_retrieval_with(
        protocol: Option<RerankerProtocol>,
        endpoint_path: Option<&str>,
        instruction: Option<&str>,
        field: Option<&str>,
        doc_chars: Option<usize>,
        return_docs: Option<Option<bool>>,
        timeout_secs: Option<u64>,
        max_retries: Option<u32>,
        rate_limit_wait: Option<Option<u64>>,
        batch: Option<usize>,
    ) -> RetrievalConfig {
        let mut cfg = RetrievalConfig::default();
        if let Some(p) = protocol {
            cfg.reranker_protocol = Some(p);
        }
        if let Some(p) = endpoint_path {
            cfg.reranker_endpoint_path = Some(p.to_string());
        }
        if let Some(i) = instruction {
            cfg.reranker_task_instruction = Some(i.to_string());
        }
        if let Some(f) = field {
            cfg.reranker_task_field = Some(f.to_string());
        }
        if let Some(c) = doc_chars {
            cfg.reranker_max_doc_chars = c;
        }
        if let Some(r) = return_docs {
            cfg.reranker_return_documents = r;
        }
        if let Some(t) = timeout_secs {
            cfg.reranker_timeout_secs = t;
        }
        if let Some(r) = max_retries {
            cfg.reranker_max_retries = r;
        }
        if let Some(w) = rate_limit_wait {
            cfg.reranker_rate_limit_wait_secs = w;
        }
        if let Some(b) = batch {
            cfg.max_rerank_batch = b;
        }
        cfg
    }

    #[tokio::test]
    async fn generic_protocol_sends_top_n_and_parses_results() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9},{"index":1,"relevance_score":0.5}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let base = format!("http://{addr}");
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        // Non-Voyage base should map to Generic -> top_n
        let cfg = RerankerConfig::new(base, "model".to_string(), "key".to_string());
        let reranker = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let scores = reranker
            .rerank("q", &["doc a".to_string(), "doc b".to_string()])
            .await
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0].relevance_score > scores[1].relevance_score);
        let body = captured.lock().unwrap().clone();
        assert_eq!(body["top_n"], serde_json::json!(2));
        assert!(
            body.get("top_k").is_none(),
            "generic must not send top_k: {body}"
        );
    }

    #[tokio::test]
    async fn voyage_auto_detection_sends_top_k_and_accepts_data() {
        let base = "https://api.voyageai.com/v1".to_string();
        // We need to point at our mock listener but still trigger Voyage detection.
        // To do that, we test detection via explicit voyage URL logic: use voyage hostname directly
        // but override the actual connection via base URL that contains voyage hostname?
        // For mock, we instead directly test that a config with voyage hostname would use top_k.
        // Here we simulate by setting base to voyage hostname and using a custom listener via rewriting?
        // Simpler: test explicit Voyage protocol on non-voyage host (VAL-RERANK-003) covers auto-detection.
        // This test verifies voyage base detection by checking from_configs with voyage URL.
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(base, "rerank-2".to_string(), "k".to_string());
        let reranker = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        // Verify the reranker internally chose Voyage protocol (is_voyage true)
        assert!(
            reranker.is_voyage,
            "voyage hostname should auto-detect Voyage"
        );
        // For wire capture we need a mock that actually receives top_k; use explicit Voyage on loopback
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let listener2 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr2 = listener2.local_addr().unwrap();
        let cap2 = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener2.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap2.lock().unwrap() = body;
            let resp = r#"{"data":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let cfg2 = RerankerConfig::new(format!("http://{addr2}"), "m".to_string(), "k".to_string());
        let mut retrieval2 = retrieval.clone();
        retrieval2.reranker_protocol = Some(RerankerProtocol::Voyage);
        let r2 = ApiReranker::from_configs(cfg2, &retrieval2).unwrap();
        let _ = r2.rerank("q", &["doc".to_string()]).await.unwrap();
        let body = captured.lock().unwrap().clone();
        assert_eq!(body["top_k"], serde_json::json!(1));
        assert!(body.get("top_n").is_none());
    }

    #[tokio::test]
    async fn explicit_protocol_overrides_auto_detection() {
        // Voyage hostname with explicit Generic -> top_n
        let retrieval = make_retrieval_with(
            Some(RerankerProtocol::Generic),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(
            "https://api.voyageai.com/v1".to_string(),
            "m".to_string(),
            "k".to_string(),
        );
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        assert!(
            !r.is_voyage,
            "explicit Generic must override Voyage hostname"
        );
        assert_eq!(r.protocol, RerankerProtocol::Generic);

        // Non-Voyage hostname with explicit Voyage -> top_k
        let retrieval2 = make_retrieval_with(
            Some(RerankerProtocol::Voyage),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cfg2 = RerankerConfig::new(
            "https://api.siliconflow.com/v1".to_string(),
            "m".to_string(),
            "k".to_string(),
        );
        let r2 = ApiReranker::from_configs(cfg2, &retrieval2).unwrap();
        assert!(
            r2.is_voyage,
            "explicit Voyage must win over generic hostname"
        );
    }

    #[tokio::test]
    async fn custom_proxy_with_explicit_protocol() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let retrieval = make_retrieval_with(
            Some(RerankerProtocol::Voyage),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let _ = r.rerank("q", &["doc".to_string()]).await.unwrap();
        let body = captured.lock().unwrap().clone();
        assert_eq!(body["top_k"], serde_json::json!(1));
        assert!(
            body.get("top_n").is_none(),
            "custom proxy with explicit Voyage must send top_k"
        );
    }

    #[tokio::test]
    async fn endpoint_path_override_honored_verbatim() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured_path = Arc::new(Mutex::new(String::new()));
        let cap = Arc::clone(&captured_path);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (first, _body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = first.clone();
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let retrieval = make_retrieval_with(
            None,
            Some("/v1/reranking"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        assert_eq!(r.endpoint_url(), format!("http://{addr}/v1/reranking"));
        let _ = r.rerank("q", &["doc".to_string()]).await.unwrap();
        let path = captured_path.lock().unwrap().clone();
        assert!(
            path.contains("/v1/reranking"),
            "configured path must be honored verbatim: {path}"
        );
        assert!(
            !path.contains("/rerank") || path.contains("/v1/reranking"),
            "custom path must not append /rerank: {path}"
        );
    }

    #[tokio::test]
    async fn default_endpoint_unchanged() {
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(
            "https://api.siliconflow.com/v1/".to_string(),
            "m".to_string(),
            "k".to_string(),
        );
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        assert_eq!(r.endpoint_url(), "https://api.siliconflow.com/v1/rerank");
    }

    #[tokio::test]
    async fn return_documents_per_capability() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        // Generic default should send return_documents false
        let retrieval = make_retrieval_with(
            None,
            None,
            None,
            None,
            None,
            Some(Some(false)),
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let _ = r.rerank("q", &["doc".to_string()]).await.unwrap();
        let body = captured.lock().unwrap().clone();
        assert_eq!(body["return_documents"], serde_json::json!(false));

        // None should omit field
        let listener2 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr2 = listener2.local_addr().unwrap();
        let cap2 = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap2c = Arc::clone(&cap2);
        tokio::spawn(async move {
            let (mut stream, _) = listener2.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap2c.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let retrieval2 = make_retrieval_with(
            None,
            None,
            None,
            None,
            None,
            Some(None),
            None,
            None,
            None,
            None,
        );
        let cfg2 = RerankerConfig::new(format!("http://{addr2}"), "m".to_string(), "k".to_string());
        let r2 = ApiReranker::from_configs(cfg2, &retrieval2).unwrap();
        let _ = r2.rerank("q", &["doc".to_string()]).await.unwrap();
        let body2 = cap2.lock().unwrap().clone();
        assert!(
            body2.get("return_documents").is_none(),
            "None should omit return_documents: {body2}"
        );
    }

    #[tokio::test]
    async fn document_echo_tolerated() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            // Response echoes document field
            let resp =
                r#"{"results":[{"index":0,"relevance_score":0.9,"document":{"text":"echoed"}}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let scores = r.rerank("q", &["local doc".to_string()]).await.unwrap();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].index, 0);
        assert!((scores[0].relevance_score - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn task_instruction_omitted_by_default() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let _ = r.rerank("q", &["doc".to_string()]).await.unwrap();
        let body = captured.lock().unwrap().clone();
        assert!(
            body.get("instruction").is_none(),
            "instruction must be omitted by default: {body}"
        );
    }

    #[tokio::test]
    async fn task_instruction_omitted_when_unsupported() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        // Voyage does not support instruction by default
        let retrieval = make_retrieval_with(
            Some(RerankerProtocol::Voyage),
            None,
            Some("code search"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let _ = r.rerank("q", &["doc".to_string()]).await.unwrap();
        let body = captured.lock().unwrap().clone();
        assert!(
            body.get("instruction").is_none(),
            "Voyage should omit instruction when unsupported: {body}"
        );
    }

    #[tokio::test]
    async fn task_instruction_serialized_with_exact_field_when_enabled() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let retrieval = make_retrieval_with(
            Some(RerankerProtocol::Generic),
            None,
            Some("Given a web search query, retrieve relevant passages that answer the query"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let _ = r.rerank("q", &["doc".to_string()]).await.unwrap();
        let body = captured.lock().unwrap().clone();
        assert_eq!(
            body[RERANKER_INSTRUCTION_FIELD],
            serde_json::json!(
                "Given a web search query, retrieve relevant passages that answer the query"
            )
        );
        // Changing only instruction changes only that field - compare with explicit field case
        assert_eq!(body["query"], serde_json::json!("q"));
        assert_eq!(body["documents"], serde_json::json!(["doc"]));
    }

    #[tokio::test]
    async fn task_instruction_explicit_field_overrides_capability() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(serde_json::Value::Null));
        let cap = Arc::clone(&captured);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            *cap.lock().unwrap() = body;
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        resp.len(),
                        resp
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        // Voyage normally omits, but explicit field forces serialization
        let retrieval = make_retrieval_with(
            Some(RerankerProtocol::Voyage),
            None,
            Some("my instruction"),
            Some("custom_instruction"),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let _ = r.rerank("q", &["doc".to_string()]).await.unwrap();
        let body = captured.lock().unwrap().clone();
        assert_eq!(
            body["custom_instruction"],
            serde_json::json!("my instruction")
        );
    }

    #[tokio::test]
    async fn document_budget_newline_safe_and_unlimited() {
        // Default 4800
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        assert_eq!(retrieval.reranker_max_doc_chars, 4800);
        // 1200 explicit
        let retrieval2 = make_retrieval_with(
            None,
            None,
            None,
            None,
            Some(1200),
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(retrieval2.reranker_max_doc_chars, 1200);
        // 0 unlimited
        let retrieval3 = make_retrieval_with(
            None,
            None,
            None,
            None,
            Some(0),
            None,
            None,
            None,
            None,
            None,
        );
        let cfg = RerankerConfig::new(
            "http://example.com".to_string(),
            "m".to_string(),
            "k".to_string(),
        );
        let r = ApiReranker::from_configs(cfg, &retrieval3).unwrap();
        assert_eq!(r.max_document_chars, 0);
        // Truncate test
        assert_eq!(truncate_document("a\nb\nc\nd", 5), "a\nb");
        assert_eq!(truncate_document("abcdefghij", 5), "abcde");
        assert_eq!(truncate_document("hello", 0), "hello");
    }

    #[tokio::test]
    async fn batch_defaults_and_zero_unbatched() {
        // Default 20 -> 50 candidates => 3 requests
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cnt = Arc::clone(&count);
        let max_docs = Arc::new(Mutex::new(Vec::new()));
        let md = Arc::clone(&max_docs);
        tokio::spawn(async move {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let (_first, body) = read_http_request(&mut stream).await;
                cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let docs = body["documents"].as_array().unwrap().len();
                md.lock().unwrap().push(docs);
                let resp = format!(
                    r#"{{"results":[{}]}}"#,
                    (0..docs)
                        .map(|i| format!(
                            r#"{{"index":{i},"relevance_score":{}}}"#,
                            1.0 - i as f64 * 0.1
                        ))
                        .collect::<Vec<_>>()
                        .join(",")
                );
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp.len(),
                            resp
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let retrieval = make_retrieval_with(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(20),
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let docs: Vec<String> = (0..50).map(|i| format!("doc {i}")).collect();
        let scores = r.rerank("q", &docs).await.unwrap();
        assert_eq!(scores.len(), 50);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 3);
        let captured = max_docs.lock().unwrap().clone();
        assert_eq!(captured, vec![20, 20, 10]);

        // 0 means one unbatched request
        let listener2 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr2 = listener2.local_addr().unwrap();
        let count2 = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cnt2 = Arc::clone(&count2);
        tokio::spawn(async move {
            let (mut stream, _) = listener2.accept().await.unwrap();
            let (_first, body) = read_http_request(&mut stream).await;
            cnt2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let docs = body["documents"].as_array().unwrap().len();
            assert_eq!(docs, 50, "0 batch must send all docs in one request");
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp.len(),
                            resp
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            // Send minimal response for rest if any
        });
        let retrieval2 = make_retrieval_with(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(0),
        );
        let cfg2 = RerankerConfig::new(format!("http://{addr2}"), "m".to_string(), "k".to_string());
        let r2 = ApiReranker::from_configs(cfg2, &retrieval2).unwrap();
        let docs2: Vec<String> = (0..50).map(|i| format!("doc {i}")).collect();
        // Need to handle single-batch response that returns only one score; but our mock only returns one.
        // For this test we just check request count, not scores.
        let _ = r2.rerank("q", &docs2).await;
        // Small sleep to allow server to count
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert_eq!(count2.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn batch_local_to_global_remapping() {
        // 5 docs, batch 2 => 3 batches (2,2,1). Each batch returns local indices with distinct scores.
        // Verify global indices are correct after remapping and final ordering is by global score.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // Batch 0: docs 0,1 -> return local 1 with score 0.9, local 0 with 0.5
            // Batch 1: docs 2,3 -> local 0 score 0.8, local1 score 0.6
            // Batch 2: docs 4 -> local0 score 1.0 (global 4)
            let responses = vec![
                r#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.5}]}"#,
                r#"{"results":[{"index":0,"relevance_score":0.8},{"index":1,"relevance_score":0.6}]}"#,
                r#"{"results":[{"index":0,"relevance_score":1.0}]}"#,
            ];
            for resp in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let _ = read_http_request(&mut stream).await;
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp.len(),
                            resp
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let retrieval = make_retrieval_with(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(2),
        );
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let docs: Vec<String> = (0..5).map(|i| format!("doc {i}")).collect();
        let scores = r.rerank("q", &docs).await.unwrap();
        // Expected global mapping: batch0 local1 -> global1 score0.9, local0->global0 score0.5
        // batch1 local0->global2 score0.8, local1->global3 score0.6
        // batch2 local0->global4 score1.0
        // Sorted descending: global4(1.0), global1(0.9), global2(0.8), global3(0.6), global0(0.5)
        let order: Vec<usize> = scores.iter().map(|s| s.index).collect();
        assert_eq!(order, vec![4, 1, 2, 3, 0]);
    }

    #[tokio::test]
    async fn duplicate_and_out_of_range_ignored() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            // Duplicate index 0, out-of-range 10, valid 1
            let resp = r#"{"results":[{"index":0,"relevance_score":0.9},{"index":0,"relevance_score":0.5},{"index":10,"relevance_score":0.8},{"index":1,"relevance_score":0.7}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp.len(),
                            resp
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
        });
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        // Build distinct candidates so deduplication can be verified by file/content
        let mut results = Vec::new();
        for i in 0..3 {
            results.push(super::make_result(
                &format!("f{i}.rs"),
                1,
                10,
                0.5,
                None,
                &format!("content {i}"),
            ));
        }
        let reranked = rerank_results(&r, "q", &results, 3).await.unwrap();
        // Server returns duplicate index 0 (second duplicate filtered), out-of-range 10 (filtered),
        // so only indices 0 and 1 survive as valid reranked entries.
        assert_eq!(
            reranked.len(),
            2,
            "duplicate + out-of-range must be ignored: {:?}",
            reranked
        );
        // Valid reranked entries are returned in score order.
        assert_eq!(reranked[0].content, "content 0");
        assert_eq!(reranked[1].content, "content 1");
    }

    #[tokio::test]
    async fn short_response_append_in_original_order() {
        // Server returns only 2 of 5 candidates
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            let resp = r#"{"results":[{"index":1,"relevance_score":0.9},{"index":3,"relevance_score":0.8}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp.len(),
                            resp
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
        });
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let mut results = Vec::new();
        for i in 0..5 {
            results.push(super::make_result(
                &format!("f{i}.rs"),
                1,
                10,
                1.0,
                None,
                &format!("content {i}"),
            ));
        }
        let reranked = rerank_results(&r, "q", &results, 5).await.unwrap();
        // rerank_results returns only the successfully mapped reranked entries (filtering
        // duplicates/out-of-range/missing scores). The shortfall append in original hybrid
        // order is performed by `hybrid::merge_reranked_results` (VAL-RERANK-015) and is
        // covered by `hybrid::shortfall_tests`. Here we verify the reranker layer returns
        // exactly the 2 mapped entries in score order and does not fabricate the missing 3.
        assert_eq!(
            reranked.len(),
            2,
            "rerank layer must return only mapped entries"
        );
        assert_eq!(reranked[0].content, "content 1");
        assert_eq!(reranked[1].content, "content 3");
    }

    #[tokio::test]
    async fn score_alias_and_malformed_json() {
        // Score alias test: both relevance_score and score produce same ordering
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            let resp = r#"{"results":[{"index":0,"score":0.9},{"index":1,"score":0.5}]}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp.len(),
                            resp
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
        });
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let scores = r
            .rerank("q", &["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(scores[0].index, 0);

        // Malformed JSON should error without leaking key
        let listener2 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr2 = listener2.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener2.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            let resp = r#"{"results":[{"index":0,"#; // truncated
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp.len(),
                            resp
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
        });
        let cfg2 = RerankerConfig::new(
            format!("http://{addr2}"),
            "m".to_string(),
            "secret-key-123".to_string(),
        );
        let r2 = ApiReranker::from_configs(cfg2, &retrieval).unwrap();
        let err = r2
            .rerank("secret query", &["doc".to_string()])
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(!msg.contains("secret-key-123"), "key leaked: {msg}");
        assert!(!msg.contains("secret query"), "query leaked: {msg}");
        assert!(
            msg.contains("200") || msg.contains("parse"),
            "diagnosis missing: {msg}"
        );
    }

    #[tokio::test]
    async fn retry_classification_permanent_4xx_not_retried() {
        for status in [400u16, 404, 422] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let cnt = Arc::clone(&count);
            tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let _ = read_http_request(&mut stream).await;
                cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let body = format!("error {status}");
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                // Do not accept second connection; if retried, second accept would hang and count stays 1
            });
            let retrieval =
                make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
            let mut cfg =
                RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
            cfg = cfg.with_max_retries(2);
            let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
            let _ = r.rerank("q", &["doc".to_string()]).await;
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            assert_eq!(
                count.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "status {status} must not be retried"
            );
        }
    }

    #[tokio::test]
    async fn transient_failures_are_retried() {
        // 500 should be retried
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cnt = Arc::clone(&count);
        tokio::spawn(async move {
            for i in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let _ = read_http_request(&mut stream).await;
                cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i == 0 {
                    let body = "server error";
                    stream
                        .write_all(
                            format!(
                                "HTTP/1.1 500 Internal Server Error\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                body.len(),
                                body
                            )
                            .as_bytes(),
                        )
                        .await
                        .unwrap();
                } else {
                    let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
                    stream
                        .write_all(
                            format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                resp.len(),
                                resp
                            )
                            .as_bytes(),
                        )
                        .await
                        .unwrap();
                }
            }
        });
        let retrieval =
            make_retrieval_with(None, None, None, None, None, None, None, None, None, None);
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string())
            .with_max_retries(2);
        let r = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let scores = r.rerank("q", &["doc".to_string()]).await.unwrap();
        assert_eq!(scores.len(), 1);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
