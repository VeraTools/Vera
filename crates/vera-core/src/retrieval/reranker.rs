//! Cross-encoder reranker for refining retrieval results.
//!
//! After hybrid retrieval produces candidates via RRF fusion, the reranker
//! sends the top-N candidates along with the query to a cross-encoder API
//! (e.g. Qwen3-Reranker via SiliconFlow). The API scores each query-document
//! pair independently, producing more accurate relevance scores than the
//! fast-but-approximate RRF fusion.
//!
//! Graceful degradation: if the reranker API is unavailable (timeout, 5xx,
//! connection error), the pipeline returns unreranked results with a warning.

use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::chunk_text::{file_name, normalize_path_tokens};
use crate::retrieval::ranking::file_role_label;
use crate::types::SearchResult;

// ── Error types ──────────────────────────────────────────────────────

/// Errors specific to the reranking pipeline.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RerankerError {
    /// Authentication failure (invalid or missing API key).
    #[error("reranker API authentication failed: {message}")]
    AuthError { message: String },

    /// Cannot reach the reranker endpoint.
    #[error("reranker API connection failed: {message}")]
    ConnectionError { message: String },

    /// The API returned a non-auth, non-connection error.
    #[error("reranker API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    /// Rate limit exceeded.
    #[error("reranker API rate limit exceeded: {message}")]
    RateLimitError { message: String },

    /// Unexpected response format.
    #[error("unexpected reranker API response: {message}")]
    ResponseError { message: String },

    /// Request was cancelled because the client disconnected.
    #[error("rerank cancelled")]
    Cancelled,
}

// ── Reranker trait ───────────────────────────────────────────────────

/// Trait abstracting a reranker provider.
///
/// Implementations take a query and a set of document texts, returning
/// relevance scores for each document. The scores are used to reorder
/// search results after initial retrieval.
#[allow(async_fn_in_trait)]
pub trait Reranker: Send + Sync {
    /// Score each document against the query.
    ///
    /// Returns a vector of `(original_index, relevance_score)` pairs,
    /// sorted by relevance_score descending.
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError>;

    /// Like `rerank`, but aborts between batches if `cancel` is fired.
    ///
    /// The default implementation ignores the token and delegates to `rerank`.
    async fn rerank_cancellable(
        &self,
        query: &str,
        documents: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        let _ = cancel;
        self.rerank(query, documents).await
    }
}

/// A single reranking score for a document.
#[derive(Debug, Clone)]
pub struct RerankScore {
    /// Original index in the input documents array.
    pub index: usize,
    /// Relevance score from the reranker (higher is better).
    pub relevance_score: f64,
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for an API-based reranker.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Base URL for the API (e.g. "https://api.siliconflow.com/v1").
    pub base_url: String,
    /// Model identifier (e.g. "Qwen/Qwen3-Reranker-8B").
    pub model_id: String,
    /// API key (never logged or exposed).
    api_key: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retries on transient errors.
    pub max_retries: u32,
}

impl RerankerConfig {
    /// Create a new config. The API key is stored opaquely and never exposed.
    pub fn new(base_url: String, model_id: String, api_key: String) -> Self {
        Self {
            base_url,
            model_id,
            api_key,
            timeout: Duration::from_secs(30),
            max_retries: 2,
        }
    }

    /// Create config from environment variables.
    ///
    /// Reads:
    /// - `RERANKER_MODEL_BASE_URL`
    /// - `RERANKER_MODEL_ID`
    /// - `RERANKER_MODEL_API_KEY`
    pub fn from_env() -> Result<Self> {
        let base_url =
            std::env::var("RERANKER_MODEL_BASE_URL").context("RERANKER_MODEL_BASE_URL not set")?;
        let model_id = std::env::var("RERANKER_MODEL_ID").context("RERANKER_MODEL_ID not set")?;
        let api_key =
            std::env::var("RERANKER_MODEL_API_KEY").context("RERANKER_MODEL_API_KEY not set")?;

        Ok(Self::new(base_url, model_id, api_key))
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the maximum retry count.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}

// ── API-based reranker ───────────────────────────────────────────────

/// Reranker that calls an external cross-encoder API (SiliconFlow-compatible).
///
/// Sends query + documents to the `/rerank` endpoint and returns scored
/// results. Compatible with SiliconFlow, Jina, Cohere, and similar APIs
/// that implement the `/v1/rerank` endpoint format.
pub struct ApiReranker {
    client: reqwest::Client,
    config: RerankerConfig,
    pub(crate) max_rerank_batch: usize,
    max_document_chars: usize,
    /// Whether the configured base URL points at Voyage AI.
    ///
    /// Voyage's `/rerank` accepts `top_k` instead of the `top_n` field
    /// used by SiliconFlow, Jina, Cohere, and the rest of the OpenAI-style
    /// rerank ecosystem. Detected once at construction; mirrors the
    /// embedding-side detection in `embedding/provider.rs`.
    is_voyage: bool,
}

impl ApiReranker {
    /// Create a new API-based reranker from configuration.
    ///
    /// This preserves the original constructor contract by resolving the
    /// batch size from `VERA_MAX_RERANK_BATCH` (defaulting to 20).
    pub fn new(config: RerankerConfig) -> Result<Self> {
        let max_rerank_batch = std::env::var("VERA_MAX_RERANK_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        Self::new_with_max_rerank_batch(config, max_rerank_batch)
    }

    /// Create a new API-based reranker with an explicit batch size.
    ///
    /// `max_rerank_batch` is the caller's resolved `retrieval.max_rerank_batch`.
    /// It is a parameter rather than an environment lookup so that the value
    /// in `~/.vera/config.json` is the one actually used; 0 disables batching.
    pub fn new_with_max_rerank_batch(
        config: RerankerConfig,
        max_rerank_batch: usize,
    ) -> Result<Self> {
        crate::init_tls();
        let max_document_chars = std::env::var("VERA_MAX_RERANK_DOC_CHARS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4800);
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to create HTTP client for reranker")?;

        let is_voyage = is_voyage_base_url(&config.base_url);

        Ok(Self {
            client,
            config,
            max_rerank_batch,
            max_document_chars,
            is_voyage,
        })
    }

    /// Build the rerank endpoint URL.
    fn endpoint_url(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}/rerank")
    }

    /// Execute a rerank API call with retry logic.
    async fn call_api(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
        cancel: &CancellationToken,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        let url = self.endpoint_url();
        let top = if self.is_voyage {
            TopLimit::TopK { top_k: top_n }
        } else {
            TopLimit::TopN { top_n }
        };
        let body = RerankRequest {
            model: &self.config.model_id,
            query,
            documents,
            top,
            return_documents: Some(false),
        };

        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            if cancel.is_cancelled() {
                return Err(RerankerError::Cancelled);
            }

            if attempt > 0 {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt.min(4) - 1));
                debug!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    "retrying reranker API"
                );
                tokio::select! {
                    _ = cancel.cancelled() => return Err(RerankerError::Cancelled),
                    _ = tokio::time::sleep(delay) => {}
                }
            }

            let request = tokio::select! {
                _ = cancel.cancelled() => return Err(RerankerError::Cancelled),
                result = self.send_request(&url, &body) => result,
            };

            match request {
                Ok(scores) => return Ok(scores),
                Err(e) => {
                    // Don't retry auth errors.
                    if matches!(e, RerankerError::AuthError { .. }) {
                        return Err(e);
                    }
                    warn!(
                        attempt = attempt + 1,
                        max = self.config.max_retries + 1,
                        error = %e,
                        "reranker API call failed"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| RerankerError::ApiError {
            status: 0,
            message: "all retries exhausted".to_string(),
        }))
    }

    /// Send a single HTTP request and parse the response.
    async fn send_request(
        &self,
        url: &str,
        body: &RerankRequest<'_>,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() || e.is_timeout() {
                    RerankerError::ConnectionError {
                        message: format!("failed to connect to reranker API: {e}"),
                    }
                } else {
                    RerankerError::ConnectionError {
                        message: format!("request failed: {e}"),
                    }
                }
            })?;

        let status = response.status().as_u16();

        if status == 401 || status == 403 {
            let text = response.text().await.unwrap_or_default();
            return Err(RerankerError::AuthError {
                message: sanitize_error_message(&text),
            });
        }

        if status == 429 {
            let text = response.text().await.unwrap_or_default();
            return Err(RerankerError::RateLimitError {
                message: sanitize_error_message(&text),
            });
        }

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(RerankerError::ApiError {
                status,
                message: sanitize_error_message(&text),
            });
        }

        let resp: RerankResponse =
            response
                .json()
                .await
                .map_err(|e| RerankerError::ResponseError {
                    message: format!("failed to parse reranker response: {e}"),
                })?;

        // Convert to RerankScore, sorted by relevance_score descending.
        let mut scores: Vec<RerankScore> = resp
            .results
            .into_iter()
            .map(|r| RerankScore {
                index: r.index,
                relevance_score: r.relevance_score,
            })
            .collect();

        scores.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scores)
    }

    async fn rerank_inner(
        &self,
        query: &str,
        documents: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if cancel.is_cancelled() {
            return Err(RerankerError::Cancelled);
        }

        // Truncate documents that exceed the reranker's context window.
        let truncated: Vec<String> = documents
            .iter()
            .map(|d| truncate_document(d, self.max_document_chars))
            .collect();
        let documents = &truncated;

        let batch_size = self.max_rerank_batch;
        if batch_size == 0 || documents.len() <= batch_size {
            return self
                .call_api(query, documents, documents.len(), cancel)
                .await;
        }

        // Partition into batches, rerank each, merge with corrected indices.
        let mut all_scores = Vec::with_capacity(documents.len());
        for (batch_idx, batch) in documents.chunks(batch_size).enumerate() {
            if cancel.is_cancelled() {
                return Err(RerankerError::Cancelled);
            }
            let offset = batch_idx * batch_size;
            let scores = self.call_api(query, batch, batch.len(), cancel).await?;
            for s in scores {
                all_scores.push(RerankScore {
                    index: s.index + offset,
                    relevance_score: s.relevance_score,
                });
            }
        }

        all_scores.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(all_scores)
    }
}

impl Reranker for ApiReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        self.rerank_inner(query, documents, &CancellationToken::new())
            .await
    }

    async fn rerank_cancellable(
        &self,
        query: &str,
        documents: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        self.rerank_inner(query, documents, cancel).await
    }
}

// ── Rerank search results ────────────────────────────────────────────

/// Rerank a set of search results using a cross-encoder reranker.
///
/// Sends the top-N candidates to the reranker API, reorders them by
/// reranker scores, and returns the reordered results. The reranker
/// score replaces the original score in each result.
///
/// If the reranker fails, returns `Err` — the caller decides whether
/// to fall back to unreranked results.
pub async fn rerank_results(
    reranker: &impl Reranker,
    query: &str,
    results: &[SearchResult],
    top_n: usize,
) -> Result<Vec<SearchResult>, RerankerError> {
    if results.is_empty() {
        return Ok(Vec::new());
    }

    // Take only top_n candidates for reranking.
    let candidates: Vec<&SearchResult> = results.iter().take(top_n).collect();

    // Extract document texts for the reranker.
    let documents: Vec<String> = candidates.iter().map(|r| format_for_reranker(r)).collect();

    debug!(
        query = query,
        candidates = candidates.len(),
        "sending candidates to reranker"
    );

    // Call the reranker.
    let scores = reranker.rerank(query, &documents).await?;

    debug!(
        query = query,
        scored = scores.len(),
        "received reranker scores"
    );

    // Reorder results by reranker scores.
    let mut reranked: Vec<SearchResult> = scores
        .iter()
        .filter_map(|score| {
            if score.index < candidates.len() {
                let mut result = candidates[score.index].clone();
                result.score = score.relevance_score;
                Some(result)
            } else {
                warn!(
                    index = score.index,
                    candidates = candidates.len(),
                    "reranker returned out-of-bounds index, skipping"
                );
                None
            }
        })
        .collect();

    // Ensure results are sorted by score descending (should already be from the API).
    reranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(reranked)
}

/// Format a search result for the reranker API.
///
/// Includes metadata context to help the cross-encoder understand the code.
fn format_for_reranker(result: &SearchResult) -> String {
    let mut parts = Vec::new();
    let filename = file_name(&result.file_path);

    if let Some(ref sym_name) = result.symbol_name {
        parts.push(format!("Symbol: {sym_name}"));
    }
    if let Some(ref sym_type) = result.symbol_type {
        parts.push(format!("Symbol type: {sym_type}"));
    }

    parts.push(format!("Filename: {filename}"));
    parts.push(format!("File: {} ({})", result.file_path, result.language));
    parts.push(format!(
        "Path tokens: {}",
        normalize_path_tokens(&result.file_path)
    ));
    parts.push(format!(
        "Role: {}",
        file_role_label(&result.file_path, result.language)
    ));
    parts.push(format!("Lines: {}-{}", result.line_start, result.line_end));

    parts.push(format!("Code:\n{}", result.content));

    parts.join("\n")
}

// ── Sanitization ─────────────────────────────────────────────────────

/// Remove any potential API key fragments from error messages.
fn sanitize_error_message(msg: &str) -> String {
    // Truncate at a safe char boundary to avoid panicking on multi-byte UTF-8.
    let truncated = if msg.len() > 500 {
        let end = msg
            .char_indices()
            .take_while(|(i, _)| *i < 500)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        &msg[..end]
    } else {
        msg
    };
    let sanitized = truncated
        .replace(|c: char| !c.is_ascii_graphic() && c != ' ', " ")
        .trim()
        .to_string();
    if sanitized.is_empty() {
        "no details available".to_string()
    } else {
        sanitized
    }
}

// ── Document truncation ──────────────────────────────────────────────

/// Truncate a document to at most `max_chars` characters, cutting at the last
/// newline boundary to avoid splitting mid-line. Documents within the limit
/// are returned as-is (zero-copy path).
fn truncate_document(doc: &str, max_chars: usize) -> String {
    if max_chars == 0 || doc.len() <= max_chars {
        return doc.to_string();
    }
    // Find the last char boundary at or before max_chars.
    let mut end = max_chars.min(doc.len());
    while end > 0 && !doc.is_char_boundary(end) {
        end -= 1;
    }
    let slice = &doc[..end];
    match slice.rfind('\n') {
        Some(pos) => slice[..pos].to_string(),
        None => slice.to_string(),
    }
}

// ── API request/response types ───────────────────────────────────────

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    #[serde(flatten)]
    top: TopLimit,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
}

/// Per-provider field name for the top-results limit.
///
/// Voyage AI's `/rerank` requires `top_k`. SiliconFlow / Jina / Cohere
/// and other OpenAI-style rerank endpoints use `top_n`. The wire format is
/// otherwise identical, so this is flattened into the request body.
#[derive(Serialize)]
#[serde(untagged)]
enum TopLimit {
    TopN { top_n: usize },
    TopK { top_k: usize },
}

#[derive(Deserialize)]
struct RerankResponse {
    /// Voyage wraps results in `data` instead of `results`. Tolerate both.
    #[serde(alias = "data")]
    results: Vec<RerankResult>,
}

/// Returns `true` if the configured base URL points at Voyage AI.
///
/// Voyage's `/rerank` accepts `top_k` instead of `top_n`. Mirrors the
/// embedding-side detection in `embedding/provider.rs` (substring match on
/// the canonical hostname). A custom Voyage proxy needs the same hostname.
fn is_voyage_base_url(base_url: &str) -> bool {
    base_url.contains("api.voyageai.com")
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    #[serde(alias = "score")]
    relevance_score: f64,
}

// ── Test helpers ─────────────────────────────────────────────────────

/// Mock reranker for unit testing.
///
/// Returns scores based on document length or simulates errors.
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    /// A mock reranker that returns deterministic scores.
    ///
    /// Scores each document based on a simple heuristic (shorter documents
    /// score higher, simulating "precision" preference). Can also be
    /// configured to fail with a specific error.
    pub struct MockReranker {
        pub fail_with: Option<RerankerError>,
    }

    impl MockReranker {
        pub fn new() -> Self {
            Self { fail_with: None }
        }

        pub fn failing(error: RerankerError) -> Self {
            Self {
                fail_with: Some(error),
            }
        }
    }

    impl Reranker for MockReranker {
        async fn rerank(
            &self,
            _query: &str,
            documents: &[String],
        ) -> Result<Vec<RerankScore>, RerankerError> {
            if let Some(ref err) = self.fail_with {
                return Err(err.clone());
            }

            // Deterministic scoring: reverse order of input (last doc scores highest).
            let total = documents.len();
            let mut scores: Vec<RerankScore> = documents
                .iter()
                .enumerate()
                .map(|(i, _)| RerankScore {
                    index: i,
                    relevance_score: (total - i) as f64 / total as f64,
                })
                .collect();

            scores.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            Ok(scores)
        }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::Notify;

    #[tokio::test]
    async fn api_reranker_honors_pre_cancelled_token() {
        let config = RerankerConfig::new(
            "http://127.0.0.1:19999".to_string(),
            "model".to_string(),
            "key".to_string(),
        )
        .with_max_retries(2);
        let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = reranker
            .rerank_cancellable("query", &["document".to_string()], &cancel)
            .await;

        assert!(matches!(result, Err(RerankerError::Cancelled)));
    }

    #[tokio::test]
    async fn api_reranker_honors_cancellation_during_retry_backoff() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let first_response = Arc::new(Notify::new());
        let server_request_count = Arc::clone(&request_count);
        let server_first_response = Arc::clone(&first_response);

        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                server_request_count.fetch_add(1, Ordering::SeqCst);
                let response_ready = Arc::clone(&server_first_response);
                tokio::spawn(async move {
                    let mut request = [0_u8; 4096];
                    let _ = stream.read(&mut request).await;
                    stream
                        .write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                    response_ready.notify_one();
                });
            }
        });

        let config = RerankerConfig::new(
            format!("http://{address}"),
            "model".to_string(),
            "key".to_string(),
        )
        .with_timeout(Duration::from_secs(2))
        .with_max_retries(3);
        let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            reranker
                .rerank_cancellable("query", &["document".to_string()], &task_cancel)
                .await
        });

        tokio::time::timeout(Duration::from_secs(1), first_response.notified())
            .await
            .expect("reranker should receive the first failed response");
        cancel.cancel();

        let result = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("cancellation should stop retry backoff")
            .expect("reranker task should not panic");
        assert!(matches!(result, Err(RerankerError::Cancelled)));
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        server.abort();
    }
}

#[cfg(test)]
#[path = "reranker_tests.rs"]
mod tests;
