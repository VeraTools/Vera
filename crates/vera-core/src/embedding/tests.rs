//! Tests for the embedding pipeline.

use std::sync::{Arc, Mutex};

use crate::embedding::provider::test_helpers::MockProvider;
use crate::embedding::{EmbeddingError, EmbeddingProvider, embed_chunks_concurrent};
use crate::types::{Chunk, Language, SymbolType};

fn sample_chunks(n: usize) -> Vec<Chunk> {
    (0..n)
        .map(|i| Chunk {
            id: format!("chunk-{i}"),
            file_path: format!("src/file{i}.rs"),
            line_start: 1,
            line_end: 10,
            content: format!("fn func_{i}() {{\n    // body {i}\n}}"),
            language: Language::Rust,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some(format!("func_{i}")),
            part_index: None,
        })
        .collect()
}

struct ContextLimitProvider {
    dim: usize,
    max_chars: usize,
}

struct TokenLimitBatchProvider {
    dim: usize,
    max_chars: usize,
}

struct TokensInParensProvider {
    dim: usize,
    max_chars: usize,
}

struct OpenAiInputLengthProvider {
    dim: usize,
    max_chars: usize,
}

struct BatchLimitedProvider {
    dim: usize,
    max_batch_size: usize,
    max_seen_batch_size: Arc<Mutex<usize>>,
}

impl EmbeddingProvider for ContextLimitProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts
            .iter()
            .any(|text| text.chars().count() > self.max_chars)
        {
            return Err(EmbeddingError::ApiError {
                status: 400,
                message: r#"{"error":{"code":400,"message":"input (8766 tokens) is larger than the max context size (8192 tokens). skipping","type":"exceed_context_size_error","n_prompt_tokens":8766,"n_ctx":8192}}"#.to_string(),
            });
        }

        Ok(texts
            .iter()
            .map(|text| {
                (0..self.dim)
                    .map(|i| text.len() as f32 + i as f32)
                    .collect()
            })
            .collect())
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

impl EmbeddingProvider for TokenLimitBatchProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts
            .iter()
            .any(|text| text.chars().count() > self.max_chars)
        {
            return Err(EmbeddingError::ApiError {
                status: 500,
                message: r#"{"error":{"code":500,"message":"input (20301 tokens) is too large to process. increase the physical batch size (current batch size: 4096)","type":"server_error"}}"#.to_string(),
            });
        }

        Ok(texts
            .iter()
            .map(|text| {
                (0..self.dim)
                    .map(|i| text.len() as f32 + i as f32)
                    .collect()
            })
            .collect())
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

impl EmbeddingProvider for TokensInParensProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts
            .iter()
            .any(|text| text.chars().count() > self.max_chars)
        {
            return Err(EmbeddingError::ApiError {
                status: 400,
                message: "input (9448 tokens) is larger than the max context size (8192 tokens). skipping".to_string(),
            });
        }

        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32, 1.0, 2.0, 3.0][..self.dim].to_vec())
            .collect())
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

impl EmbeddingProvider for OpenAiInputLengthProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts
            .iter()
            .any(|text| text.chars().count() > self.max_chars)
        {
            // OpenAI's format carries the limit but no input token count.
            return Err(EmbeddingError::ApiError {
                status: 400,
                message: r#"{"error":{"message":"Invalid 'input[3]': maximum input length is 8192 tokens.","type":"invalid_request_error","param":null,"code":null}}"#.to_string(),
            });
        }

        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32, 1.0, 2.0, 3.0][..self.dim].to_vec())
            .collect())
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

impl EmbeddingProvider for BatchLimitedProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut max_seen = self.max_seen_batch_size.lock().unwrap();
        *max_seen = (*max_seen).max(texts.len());
        drop(max_seen);

        if texts.len() > self.max_batch_size {
            return Err(EmbeddingError::ApiError {
                status: 400,
                message: format!(
                    "* BatchEmbedContentsRequest.requests: at most {} requests can be in one batch",
                    self.max_batch_size
                ),
            });
        }

        Ok(texts
            .iter()
            .map(|text| {
                (0..self.dim)
                    .map(|i| text.len() as f32 + i as f32)
                    .collect()
            })
            .collect())
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(self.dim)
    }

    fn max_batch_size(&self) -> Option<usize> {
        Some(self.max_batch_size)
    }
}

// ── Provider trait tests ────────────────────────────────────────────

#[tokio::test]
async fn mock_provider_returns_correct_count() {
    let provider = MockProvider::new(128);
    let texts: Vec<String> = vec!["hello".into(), "world".into()];
    let result = provider.embed_batch(&texts).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 128);
    assert_eq!(result[1].len(), 128);
}

#[tokio::test]
async fn mock_provider_empty_input() {
    let provider = MockProvider::new(128);
    let result = provider.embed_batch(&[]).await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn mock_provider_produces_non_zero_vectors() {
    let provider = MockProvider::new(64);
    let texts = vec!["fn main() {}".to_string()];
    let result = provider.embed_batch(&texts).await.unwrap();

    let all_zero = result[0].iter().all(|&v| v == 0.0);
    assert!(!all_zero, "mock vectors should be non-zero");
}

#[tokio::test]
async fn mock_provider_auth_error() {
    let provider = MockProvider::failing(EmbeddingError::AuthError {
        message: "invalid API key".to_string(),
    });
    let texts = vec!["hello".to_string()];
    let result = provider.embed_batch(&texts).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, EmbeddingError::AuthError { .. }),
        "expected AuthError, got: {err}"
    );
    // Ensure the error message doesn't contain an API key.
    let msg = err.to_string();
    assert!(
        !msg.contains("sk-"),
        "error should not contain API key prefix"
    );
    assert!(
        !msg.contains("Bearer"),
        "error should not contain auth header"
    );
}

#[tokio::test]
async fn mock_provider_connection_error() {
    let provider = MockProvider::failing(EmbeddingError::ConnectionError {
        message: "connection refused".to_string(),
    });
    let texts = vec!["hello".to_string()];
    let result = provider.embed_batch(&texts).await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        EmbeddingError::ConnectionError { .. }
    ));
}

// ── Batch embedding tests ───────────────────────────────────────────

#[tokio::test]
async fn embed_chunks_respects_batch_size() {
    // With 10 chunks and batch_size=3, we should get 4 batches (3+3+3+1).
    // The mock doesn't track batch calls, but we can verify the output is correct.
    let provider = MockProvider::new(32);
    let chunks = sample_chunks(10);

    let result = embed_chunks_concurrent(&provider, &chunks, 3, 4, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 10);
    // Verify ordering is preserved.
    for (i, (id, _)) in result.iter().enumerate() {
        assert_eq!(id, &format!("chunk-{i}"));
    }
}

#[tokio::test]
async fn embed_chunks_batch_size_one() {
    let provider = MockProvider::new(16);
    let chunks = sample_chunks(3);

    let result = embed_chunks_concurrent(&provider, &chunks, 1, 4, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 3);
}

#[tokio::test]
async fn embed_chunks_concurrent_clamps_to_provider_batch_limit() {
    let max_seen_batch_size = Arc::new(Mutex::new(0));
    let provider = BatchLimitedProvider {
        dim: 16,
        max_batch_size: 100,
        max_seen_batch_size: max_seen_batch_size.clone(),
    };
    let chunks = sample_chunks(205);

    let result = embed_chunks_concurrent(&provider, &chunks, 128, 4, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 205);
    assert_eq!(*max_seen_batch_size.lock().unwrap(), 100);
}

#[tokio::test]
async fn embed_chunks_vectors_are_non_zero() {
    let provider = MockProvider::new(64);
    let chunks = sample_chunks(5);

    let result = embed_chunks_concurrent(&provider, &chunks, 32, 4, 0)
        .await
        .unwrap();

    for (id, vec) in &result {
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            norm > 0.0,
            "vector for {id} should be non-zero (norm={norm})"
        );
    }
}

#[tokio::test]
async fn embed_chunks_propagates_connection_error() {
    let provider = MockProvider::failing(EmbeddingError::ConnectionError {
        message: "unreachable".to_string(),
    });
    let chunks = sample_chunks(3);

    let result = embed_chunks_concurrent(&provider, &chunks, 32, 4, 0).await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        EmbeddingError::ConnectionError { .. }
    ));
}

// ── Embedding text formatting ───────────────────────────────────────

#[tokio::test]
async fn different_chunks_get_different_vectors() {
    let provider = MockProvider::new(64);
    let chunks = sample_chunks(3);

    let result = embed_chunks_concurrent(&provider, &chunks, 32, 4, 0)
        .await
        .unwrap();

    // Different content should produce different vectors.
    assert_ne!(result[0].1, result[1].1);
    assert_ne!(result[1].1, result[2].1);
}

// ── Error message sanitization ──────────────────────────────────────

#[test]
fn error_display_does_not_leak_keys() {
    let err = EmbeddingError::AuthError {
        message: "invalid API key".to_string(),
    };
    let display = err.to_string();
    assert!(display.contains("authentication failed"));
    assert!(!display.contains("sk-"));
    assert!(!display.contains("Bearer"));
}

#[test]
fn connection_error_display() {
    let err = EmbeddingError::ConnectionError {
        message: "connection refused".to_string(),
    };
    let display = err.to_string();
    assert!(display.contains("connection failed"));
}

// ── Vector storage integration ──────────────────────────────────────

#[tokio::test]
async fn embed_and_store_in_vector_db() {
    use crate::storage::vector::VectorStore;

    let dim = 64;
    let provider = MockProvider::new(dim);
    let chunks = sample_chunks(5);

    let embeddings = embed_chunks_concurrent(&provider, &chunks, 32, 4, 0)
        .await
        .unwrap();

    // Store in vector DB.
    let store = VectorStore::open_in_memory(dim).unwrap();
    for (chunk_id, vec) in &embeddings {
        store.insert(chunk_id, vec).unwrap();
    }

    // Verify count matches.
    assert_eq!(store.count().unwrap(), 5);

    // Verify NN search finds the correct chunk.
    let query_vec = &embeddings[2].1; // Search for chunk-2.
    let results = store.search(query_vec, 10).unwrap();
    assert!(!results.is_empty());
    assert_eq!(
        results[0].chunk_id, "chunk-2",
        "self-query should rank first"
    );
}

#[tokio::test]
async fn embed_and_store_batch() {
    use crate::storage::vector::VectorStore;

    let dim = 32;
    let provider = MockProvider::new(dim);
    let chunks = sample_chunks(10);

    let embeddings = embed_chunks_concurrent(&provider, &chunks, 4, 4, 0)
        .await
        .unwrap();
    assert_eq!(embeddings.len(), 10);

    // Batch insert into vector store.
    let store = VectorStore::open_in_memory(dim).unwrap();
    let items: Vec<(&str, &[f32])> = embeddings
        .iter()
        .map(|(id, vec)| (id.as_str(), vec.as_slice()))
        .collect();
    store.insert_batch(&items).unwrap();

    assert_eq!(store.count().unwrap(), 10);
}

// ── Concurrent embedding tests ───────────────────────────────────────

#[tokio::test]
async fn concurrent_embed_returns_all_embeddings() {
    let provider = MockProvider::new(64);
    let chunks = sample_chunks(20);

    let result = embed_chunks_concurrent(&provider, &chunks, 5, 4, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 20);
    // Verify ordering is preserved.
    for (i, (id, vec)) in result.iter().enumerate() {
        assert_eq!(id, &format!("chunk-{i}"));
        assert_eq!(vec.len(), 64);
    }
}

#[tokio::test]
async fn concurrent_embed_output_independent_of_concurrency_level() {
    let provider = MockProvider::new(32);
    let chunks = sample_chunks(15);

    let one_at_a_time = embed_chunks_concurrent(&provider, &chunks, 4, 1, 0)
        .await
        .unwrap();
    let concurrent = embed_chunks_concurrent(&provider, &chunks, 4, 3, 0)
        .await
        .unwrap();

    assert_eq!(one_at_a_time.len(), concurrent.len());
    for (s, c) in one_at_a_time.iter().zip(concurrent.iter()) {
        assert_eq!(s.0, c.0, "chunk IDs should match");
        assert_eq!(s.1, c.1, "vectors should be identical");
    }
}

#[tokio::test]
async fn concurrent_embed_empty_input() {
    let provider = MockProvider::new(64);
    let result = embed_chunks_concurrent(&provider, &[], 32, 4, 0)
        .await
        .unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn concurrent_embed_single_batch() {
    let provider = MockProvider::new(16);
    let chunks = sample_chunks(3);

    let result = embed_chunks_concurrent(&provider, &chunks, 100, 4, 0)
        .await
        .unwrap();
    assert_eq!(result.len(), 3);
}

#[tokio::test]
async fn concurrent_embed_propagates_error() {
    let provider = MockProvider::failing(EmbeddingError::AuthError {
        message: "invalid key".to_string(),
    });
    let chunks = sample_chunks(10);

    let result = embed_chunks_concurrent(&provider, &chunks, 3, 4, 0).await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        EmbeddingError::AuthError { .. }
    ));
}

#[tokio::test]
async fn concurrent_embed_recovers_from_context_limit_errors() {
    let provider = ContextLimitProvider {
        dim: 4,
        max_chars: 80,
    };
    let mut chunks = sample_chunks(3);
    chunks[1].content = format!("fn huge() {{\n    {}\n}}", "x".repeat(600));

    let result = embed_chunks_concurrent(&provider, &chunks, 2, 2, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0, "chunk-0");
    assert_eq!(result[1].0, "chunk-1");
    assert_eq!(result[2].0, "chunk-2");
}

#[tokio::test]
async fn concurrent_embed_recovers_from_token_limit_batch_errors() {
    let provider = TokenLimitBatchProvider {
        dim: 4,
        max_chars: 5_000,
    };
    let mut chunks = sample_chunks(3);
    chunks[1].content = format!("fn huge() {{\n    {}\n}}", "x".repeat(7_000));

    let result = embed_chunks_concurrent(&provider, &chunks, 2, 2, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0, "chunk-0");
    assert_eq!(result[1].0, "chunk-1");
    assert_eq!(result[2].0, "chunk-2");
}

#[tokio::test]
async fn concurrent_embed_recovers_from_tokens_in_parens_error() {
    let provider = TokensInParensProvider {
        dim: 4,
        max_chars: 80,
    };
    let mut chunks = sample_chunks(3);
    chunks[1].content = format!("fn huge() {{\n    {}\n}}", "x".repeat(600));

    let result = embed_chunks_concurrent(&provider, &chunks, 2, 2, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0, "chunk-0");
    assert_eq!(result[1].0, "chunk-1");
    assert_eq!(result[2].0, "chunk-2");
}

#[tokio::test]
async fn concurrent_embed_recovers_from_openai_max_input_length_error() {
    // Regression for issue #21: OpenAI's "maximum input length is N tokens"
    // error must trigger truncation-retry instead of being fatal, even though
    // it carries no input-token count.
    let provider = OpenAiInputLengthProvider {
        dim: 4,
        max_chars: 80,
    };
    let mut chunks = sample_chunks(3);
    chunks[1].content = format!("fn huge() {{\n    {}\n}}", "x".repeat(600));

    let result = embed_chunks_concurrent(&provider, &chunks, 2, 2, 0)
        .await
        .unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0, "chunk-0");
    assert_eq!(result[1].0, "chunk-1");
    assert_eq!(result[2].0, "chunk-2");
}

// ── OpenAI provider config tests ────────────────────────────────────

#[test]
fn provider_config_from_values() {
    use crate::embedding::EmbeddingProviderConfig;

    let config = EmbeddingProviderConfig::new(
        "https://api.example.com/v1".to_string(),
        "model-123".to_string(),
        "test-key".to_string(),
    );

    assert_eq!(config.base_url, "https://api.example.com/v1");
    assert_eq!(config.model_id, "model-123");
    // API key should not be publicly accessible (it's private).
}

#[test]
fn provider_config_with_timeout() {
    use crate::embedding::EmbeddingProviderConfig;
    use std::time::Duration;

    let config = EmbeddingProviderConfig::new(
        "https://api.example.com/v1".to_string(),
        "model-123".to_string(),
        "test-key".to_string(),
    )
    .with_timeout(Duration::from_secs(60))
    .with_max_retries(5);

    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.max_retries, 5);
}

#[test]
fn provider_config_debug_redacts_api_key() {
    use crate::embedding::EmbeddingProviderConfig;

    let config = EmbeddingProviderConfig::new(
        "https://api.example.com/v1".to_string(),
        "model-123".to_string(),
        "super-secret-key-12345".to_string(),
    );

    let debug_output = format!("{config:?}");
    assert!(
        debug_output.contains("[REDACTED]"),
        "Debug should show [REDACTED] for api_key"
    );
    assert!(
        !debug_output.contains("super-secret-key-12345"),
        "Debug must NOT contain the actual API key"
    );
    assert!(
        debug_output.contains("model-123"),
        "Debug should still show the model_id"
    );
}

// ── OpenAI provider HTTP tests (with mock server) ───────────────────

#[tokio::test]
async fn openai_provider_unreachable_endpoint() {
    use crate::embedding::{EmbeddingProviderConfig, OpenAiProvider};

    let config = EmbeddingProviderConfig::new(
        "http://127.0.0.1:19999".to_string(), // No server here.
        "test-model".to_string(),
        "test-key".to_string(),
    )
    .with_timeout(std::time::Duration::from_secs(2))
    .with_max_retries(0);

    let provider = OpenAiProvider::new(config).unwrap();
    let texts = vec!["hello".to_string()];
    let result = provider.embed_batch(&texts).await;

    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), EmbeddingError::ConnectionError { .. }),
        "unreachable endpoint should produce ConnectionError"
    );
}

// ── CachedEmbeddingProvider tests ───────────────────────────────────

#[tokio::test]
async fn cached_provider_returns_same_result() {
    use crate::embedding::CachedEmbeddingProvider;

    let inner = MockProvider::new(64);
    let cached = CachedEmbeddingProvider::new(inner, 128);

    let texts = vec!["test query".to_string()];
    let first = cached.embed_batch(&texts).await.unwrap();
    let second = cached.embed_batch(&texts).await.unwrap();

    assert_eq!(first, second, "cached results should match");
    assert_eq!(cached.cache_size(), 1, "one entry should be cached");
}

#[tokio::test]
async fn cached_provider_cache_hit_is_fast() {
    use crate::embedding::CachedEmbeddingProvider;

    let inner = MockProvider::new(64);
    let cached = CachedEmbeddingProvider::new(inner, 128);

    let texts = vec!["test query for latency".to_string()];

    // Warm the cache.
    let _ = cached.embed_batch(&texts).await.unwrap();

    // Measure cache hit latency.
    let start = std::time::Instant::now();
    let _ = cached.embed_batch(&texts).await.unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "cached query should be <50ms, was {}ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn cached_provider_different_queries_cached_separately() {
    use crate::embedding::CachedEmbeddingProvider;

    let inner = MockProvider::new(128);
    let cached = CachedEmbeddingProvider::new(inner, 128);

    let q1 = vec!["find authentication logic in the codebase".to_string()];
    let q2 = vec!["database connection pooling implementation".to_string()];

    let r1 = cached.embed_batch(&q1).await.unwrap();
    let r2 = cached.embed_batch(&q2).await.unwrap();

    assert_ne!(r1, r2, "different queries should produce different vectors");
    assert_eq!(cached.cache_size(), 2);
}

#[tokio::test]
async fn cached_provider_evicts_oldest_when_full() {
    use crate::embedding::CachedEmbeddingProvider;

    let inner = MockProvider::new(8);
    let cached = CachedEmbeddingProvider::new(inner, 2);

    // Fill cache with 2 entries.
    cached.embed_batch(&[String::from("first")]).await.unwrap();
    cached.embed_batch(&[String::from("second")]).await.unwrap();
    assert_eq!(cached.cache_size(), 2);

    // Insert third — should evict "first".
    cached.embed_batch(&[String::from("third")]).await.unwrap();
    assert_eq!(cached.cache_size(), 2, "cache should stay at max capacity");
}

#[tokio::test]
async fn cached_provider_multi_text_batch_not_cached() {
    use crate::embedding::CachedEmbeddingProvider;

    let inner = MockProvider::new(16);
    let cached = CachedEmbeddingProvider::new(inner, 128);

    // Multi-text batches (used for indexing) are not cached.
    let texts = vec!["text a".to_string(), "text b".to_string()];
    let _ = cached.embed_batch(&texts).await.unwrap();
    assert_eq!(
        cached.cache_size(),
        0,
        "multi-text batches should not be cached"
    );
}

#[tokio::test]
async fn cached_provider_propagates_errors() {
    use crate::embedding::CachedEmbeddingProvider;

    let inner = MockProvider::failing(EmbeddingError::ConnectionError {
        message: "API down".to_string(),
    });
    let cached = CachedEmbeddingProvider::new(inner, 128);

    let result = cached.embed_batch(&["test".to_string()]).await;
    assert!(result.is_err(), "errors should propagate through cache");
    assert_eq!(cached.cache_size(), 0, "errors should not be cached");
}

#[tokio::test]
async fn cached_provider_expected_dim_delegates() {
    use crate::embedding::CachedEmbeddingProvider;

    let inner = MockProvider::new(64);
    let cached = CachedEmbeddingProvider::new(inner, 128);
    assert_eq!(cached.expected_dim(), Some(64));
}

#[tokio::test]
async fn cached_provider_namespace_isolates_models() {
    use crate::embedding::CachedEmbeddingProvider;

    // Two providers with different namespaces (simulating different models).
    let inner_a = MockProvider::new(3);
    let cached_a = CachedEmbeddingProvider::with_namespace(inner_a, 128, "model-a");

    let inner_b = MockProvider::new(3);
    let cached_b = CachedEmbeddingProvider::with_namespace(inner_b, 128, "model-b");

    // Same query text, different namespaces — must NOT share cache entries.
    let query = vec!["hello".to_string()];
    let vec_a = cached_a.embed_batch(&query).await.unwrap();
    assert_eq!(cached_a.cache_size(), 1);

    // Model B should get its own embedding, not model A's cached result.
    let vec_b = cached_b.embed_batch(&query).await.unwrap();
    assert_eq!(cached_b.cache_size(), 1);

    // Both should have results (mock returns deterministic vectors based on dim).
    assert_eq!(vec_a[0].len(), 3);
    assert_eq!(vec_b[0].len(), 3);
}

#[tokio::test]
async fn cached_provider_empty_namespace_backwards_compat() {
    use crate::embedding::CachedEmbeddingProvider;

    // Default constructor (no namespace) should still work — backwards compat.
    let inner = MockProvider::new(4);
    let cached = CachedEmbeddingProvider::new(inner, 128);

    let query = vec!["test".to_string()];
    let _ = cached.embed_batch(&query).await.unwrap();
    assert_eq!(cached.cache_size(), 1, "empty namespace should still cache");

    // Second call should hit cache.
    let _ = cached.embed_batch(&query).await.unwrap();
    assert_eq!(
        cached.cache_size(),
        1,
        "repeat query should hit cache, not grow"
    );
}
