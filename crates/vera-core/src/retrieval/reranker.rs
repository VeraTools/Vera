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

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::chunk_text::{file_name, normalize_path_tokens};
use crate::config::{RerankerProtocol, RetrievalConfig};
use crate::retrieval::ranking::file_role_label;
use crate::types::SearchResult;

/// The exact wire field name for reranker task instructions on supported
/// protocols (vLLM `/rerank` and OpenRouter Qwen reranker). Pinned by
/// pre-coding research against vLLM 2024+ docs (`/rerank` `instruction`
/// parameter, see `docs.vllm.ai/models/pooling_models/scoring`
/// Cohere Rerank API `instruction` field).
pub const RERANKER_INSTRUCTION_FIELD: &str = "instruction";

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
    RateLimitError {
        message: String,
        /// Server-provided wait hint (`Retry-After` header or a rate-limit
        /// reset timestamp embedded in the error body), when available.
        retry_after: Option<Duration>,
    },

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
    /// Cap on how long a 429 rate-limit wait may last before retrying.
    ///
    /// `None` (the default) keeps the short generic backoff, which favors
    /// fast degradation for interactive searches. Set this (or
    /// `VERA_RERANK_RATE_LIMIT_WAIT_SECS`) when completing the rerank
    /// matters more than latency, e.g. batch runs against free-tier
    /// endpoints with per-minute quotas.
    pub rate_limit_wait_cap: Option<Duration>,
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
            rate_limit_wait_cap: None,
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

        let mut config = Self::new(base_url, model_id, api_key);
        if let Some(secs) = std::env::var("VERA_RERANK_RATE_LIMIT_WAIT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|secs| *secs > 0)
        {
            config.rate_limit_wait_cap = Some(Duration::from_secs(secs));
        }
        Ok(config)
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

    /// Set the rate-limit wait cap (see the field documentation).
    pub fn with_rate_limit_wait_cap(mut self, cap: Duration) -> Self {
        self.rate_limit_wait_cap = Some(cap);
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
    protocol: RerankerProtocol,
    endpoint_path: Option<String>,
    task_instruction: Option<String>,
    task_field: Option<String>,
    return_documents: Option<bool>,
    /// Legacy `is_voyage` view for backward compat tests; kept in sync with `protocol`.
    #[allow(dead_code)]
    pub(crate) is_voyage: bool,
}

impl ApiReranker {
    /// Create a new API-based reranker from configuration.
    ///
    /// This preserves the original constructor contract by resolving
    /// configuration from `VeraConfig::default()` (which itself reads the
    /// legacy `VERA_MAX_RERANK_*` env overrides for its defaults). Explicit
    /// `RetrievalConfig` values remain authoritative per `aae94f7`.
    pub fn new(config: RerankerConfig) -> Result<Self> {
        let retrieval = crate::config::VeraConfig::default().retrieval;
        Self::from_configs(config, &retrieval)
    }

    /// Create a new API-based reranker with an explicit batch size.
    ///
    /// `max_rerank_batch` is the caller's resolved `retrieval.max_rerank_batch`.
    /// It is a parameter rather than an environment lookup so that the value
    /// in `~/.vera/config.json` is the one actually used; 0 disables batching.
    /// Other retrieval fields are sourced from `VeraConfig::default()` for the
    /// legacy static path, preserving env-honored defaults but keeping config
    /// authoritative when an explicit retrieval config is used via
    /// `from_configs`.
    pub fn new_with_max_rerank_batch(
        config: RerankerConfig,
        max_rerank_batch: usize,
    ) -> Result<Self> {
        let mut retrieval = crate::config::VeraConfig::default().retrieval;
        retrieval.max_rerank_batch = max_rerank_batch;
        Self::from_configs(config, &retrieval)
    }

    /// Create an API reranker from both the low-level `RerankerConfig`
    /// (credentials + base URL) and the persisted `RetrievalConfig`
    /// (protocol, endpoint path, instruction, budgets, timeouts).
    ///
    /// This is the authoritative construction path for the dynamic
    /// retrieval pipeline (`retrieval.max_rerank_batch` per `aae94f7` and
    /// all other retrieval reranker keys config-authoritative, env only
    /// via serde defaults).
    pub fn from_configs(mut config: RerankerConfig, retrieval: &RetrievalConfig) -> Result<Self> {
        crate::init_tls();

        // Retrieval config is the source of truth for these tunables;
        // apply them to the low-level config so retries/timeouts honour
        // persisted settings. Explicit retrieval values win over any
        // env-derived defaults inside `RerankerConfig::from_env`.
        config.timeout = Duration::from_secs(retrieval.reranker_timeout_secs);
        config.max_retries = retrieval.reranker_max_retries;
        config.rate_limit_wait_cap = retrieval
            .reranker_rate_limit_wait_secs
            .map(Duration::from_secs);

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to create HTTP client for reranker")?;

        let protocol = resolve_protocol(retrieval, &config.base_url);
        let is_voyage = protocol == RerankerProtocol::Voyage;

        Ok(Self {
            client,
            config,
            max_rerank_batch: retrieval.max_rerank_batch,
            max_document_chars: retrieval.reranker_max_doc_chars,
            protocol,
            endpoint_path: retrieval.reranker_endpoint_path.clone(),
            task_instruction: retrieval.reranker_task_instruction.clone(),
            task_field: retrieval.reranker_task_field.clone(),
            return_documents: retrieval.reranker_return_documents,
            is_voyage,
        })
    }

    /// Build the rerank endpoint URL.
    fn endpoint_url(&self) -> String {
        if let Some(path) = &self.endpoint_path {
            let base = self.config.base_url.trim_end_matches('/');
            // Configured path is used verbatim (leading `/` required).
            // If a hand-edited config omits the `/`, normalize defensively
            // so the URL does not become `{base}v1/rerank`.
            let normalized = if path.starts_with('/') {
                path.clone()
            } else {
                format!("/{path}")
            };
            return format!("{base}{normalized}");
        }
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
        let top = match self.protocol {
            RerankerProtocol::Voyage => TopLimit::TopK { top_k: top_n },
            RerankerProtocol::Generic => TopLimit::TopN { top_n },
        };
        let instruction = self.effective_instruction();
        let body = RerankRequest {
            model: &self.config.model_id,
            query,
            documents,
            top,
            return_documents: self.return_documents,
            instruction,
        };

        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            if cancel.is_cancelled() {
                return Err(RerankerError::Cancelled);
            }

            if attempt > 0 {
                let delay = self.retry_delay(attempt, last_err.as_ref());
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
                    // Don't retry auth errors or permanent 4xx (400,401,403,404,422).
                    // Only connection/timeout, 408, 429, and 5xx are transient.
                    if matches!(e, RerankerError::AuthError { .. }) || !is_retryable_error(&e) {
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

    /// Compute the backoff before the next attempt.
    ///
    /// A 429 with a server-provided wait hint sleeps until the quota window
    /// resets (bounded by `rate_limit_wait_cap`) instead of the short generic
    /// backoff, which cannot outlast a per-minute quota. Without a configured
    /// cap or a hint, the generic backoff applies unchanged.
    fn retry_delay(&self, attempt: u32, last_err: Option<&RerankerError>) -> Duration {
        if let (
            Some(cap),
            Some(RerankerError::RateLimitError {
                retry_after: Some(wait),
                ..
            }),
        ) = (self.config.rate_limit_wait_cap, last_err)
        {
            return (*wait).min(cap);
        }
        Duration::from_millis(500 * 2u64.pow(attempt.min(4) - 1))
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
            let retry_after = parse_retry_after_header(response.headers());
            let text = response.text().await.unwrap_or_default();
            let retry_after = retry_after.or_else(|| parse_rate_limit_reset(&text));
            return Err(RerankerError::RateLimitError {
                message: sanitize_error_message(&text),
                retry_after,
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

        // Convert to RerankScore, skipping entries without a recognized score field
        // (VAL-RERANK-016: missing score handled via shortfall path, not error).
        // Echoed document field is tolerated and never replaces local documents.
        let mut scores: Vec<RerankScore> = resp
            .results
            .into_iter()
            .filter_map(|r| {
                let score = r.relevance_score?;
                Some(RerankScore {
                    index: r.index,
                    relevance_score: score,
                })
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
    let mut seen_indices = HashSet::new();
    let mut reranked: Vec<SearchResult> = scores
        .iter()
        .filter_map(|score| {
            if score.index < candidates.len() {
                if !seen_indices.insert(score.index) {
                    warn!(
                        index = score.index,
                        "reranker returned duplicate index, skipping"
                    );
                    return None;
                }
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

    if reranked.len() < candidates.len() {
        warn!(
            candidates = candidates.len(),
            scored = reranked.len(),
            "reranker returned fewer scores than candidates"
        );
    }

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

// ── Rate-limit wait hints ────────────────────────────────────────────

/// Parse the standard `Retry-After` header (whole seconds).
fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Extract the `X-RateLimit-Reset` unix-ms timestamp embedded in an
/// OpenRouter-style 429 body and convert it to a wait duration.
fn parse_rate_limit_reset(body: &str) -> Option<Duration> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let reset_ms: u64 = value
        .pointer("/error/metadata/headers/X-RateLimit-Reset")?
        .as_str()?
        .parse()
        .ok()?;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    Some(Duration::from_millis(reset_ms.saturating_sub(now_ms)))
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
/// are returned as-is (zero-copy path). `max_chars` is a **char** budget
/// (Unicode scalar count), matching `RetrievalConfig::reranker_max_doc_chars`.
fn truncate_document(doc: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return doc.to_string();
    }
    // Single-pass traversal: capture the byte offset of the truncation
    // boundary and remember the last newline within the budget so we
    // avoid a second scan and an extra `chars().count()`.
    let mut last_nl: Option<usize> = None;
    let mut byte_end: Option<usize> = None;
    for (char_idx, (byte_idx, ch)) in doc.char_indices().enumerate() {
        if char_idx >= max_chars {
            byte_end = Some(byte_idx);
            break;
        }
        if ch == '\n' {
            last_nl = Some(byte_idx);
        }
    }
    let Some(end) = byte_end else {
        // `char_count == actual char length <= max_chars` — no truncation.
        return doc.to_string();
    };
    // `end` is the byte offset of the first char beyond the budget.
    debug_assert!(doc.is_char_boundary(end));
    if let Some(pos) = last_nl {
        // `last_nl` was recorded within the budget, so `pos < end`.
        doc[..pos].to_string()
    } else {
        doc[..end].to_string()
    }
}

// ── API request/response types ───────────────────────────────────────

struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    top: TopLimit,
    return_documents: Option<bool>,
    instruction: Option<Instruction<'a>>,
}

struct Instruction<'a> {
    field: &'a str,
    value: &'a str,
}

impl<'a> serde::Serialize for RerankRequest<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut len = 4; // model, query, documents, top
        if self.return_documents.is_some() {
            len += 1;
        }
        if self.instruction.is_some() {
            len += 1;
        }
        let mut map = serializer.serialize_map(Some(len))?;
        map.serialize_entry("model", self.model)?;
        map.serialize_entry("query", self.query)?;
        map.serialize_entry("documents", self.documents)?;
        match &self.top {
            TopLimit::TopN { top_n } => map.serialize_entry("top_n", top_n)?,
            TopLimit::TopK { top_k } => map.serialize_entry("top_k", top_k)?,
        }
        if let Some(v) = &self.return_documents {
            map.serialize_entry("return_documents", v)?;
        }
        if let Some(instr) = &self.instruction {
            map.serialize_entry(instr.field, instr.value)?;
        }
        map.end()
    }
}

/// Per-provider field name for the top-results limit.
///
/// Voyage AI's `/rerank` requires `top_k`. SiliconFlow / Jina / Cohere
/// and other OpenAI-style rerank endpoints use `top_n`. The wire format is
/// otherwise identical.
#[derive(Debug)]
enum TopLimit {
    TopN { top_n: usize },
    TopK { top_k: usize },
}

#[derive(Deserialize)]
struct RerankResponse {
    /// Voyage wraps results in `data` instead of `results`. Tolerate both.
    #[serde(alias = "data")]
    results: Vec<RerankResult>,
    // Tolerate provider extras like id/model/usage and echoed documents.
}

/// Returns `true` if the configured base URL points at Voyage AI.
///
/// Voyage's `/rerank` accepts `top_k` instead of `top_n`. Mirrors the
/// embedding-side detection in `embedding/provider.rs` (substring match on
/// the canonical hostname). A custom Voyage proxy needs the same hostname.
fn is_voyage_base_url(base_url: &str) -> bool {
    base_url.contains("api.voyageai.com")
}

fn resolve_protocol(retrieval: &RetrievalConfig, base_url: &str) -> RerankerProtocol {
    if let Some(p) = retrieval.reranker_protocol {
        p
    } else if is_voyage_base_url(base_url) {
        RerankerProtocol::Voyage
    } else {
        RerankerProtocol::Generic
    }
}

impl ApiReranker {
    fn effective_instruction(&self) -> Option<Instruction<'_>> {
        let value = self.task_instruction.as_deref()?;
        if value.trim().is_empty() {
            return None;
        }
        if let Some(field) = self.task_field.as_deref() {
            if field.trim().is_empty() {
                return None;
            }
            return Some(Instruction { field, value });
        }
        // No explicit field: use protocol default if supported, otherwise omit.
        let field = match self.protocol {
            RerankerProtocol::Generic => RERANKER_INSTRUCTION_FIELD,
            RerankerProtocol::Voyage => return None,
        };
        Some(Instruction { field, value })
    }
}

/// Whether an error should be retried (transient) vs failed fast (permanent 4xx).
fn is_retryable_error(err: &RerankerError) -> bool {
    match err {
        RerankerError::ConnectionError { .. } => true,
        RerankerError::RateLimitError { .. } => true,
        RerankerError::ApiError { status, .. } => {
            // Retry 408 and 5xx; do not retry other 4xx (400,401,403,404,422 etc).
            *status == 408 || (500..=599).contains(status)
        }
        RerankerError::AuthError { .. } => false,
        RerankerError::ResponseError { .. } => false,
        RerankerError::Cancelled => false,
    }
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    #[serde(default, alias = "score")]
    relevance_score: Option<f64>,
    // Echoed document field tolerated (string or object, per provider)
    #[serde(default)]
    #[allow(dead_code)]
    document: Option<serde_json::Value>,
    // Tolerate extra provider fields like id, etc.
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
    use std::time::Instant;
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
    async fn api_reranker_cancels_in_flight_request_promptly() {
        // Server accepts but delays response body for 10s; client cancels after 100ms
        // and must return Cancelled quickly without waiting for server.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    // Delay response well beyond cancellation window
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
                    let _ = stream
                        .write_all(
                            format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                resp.len(),
                                resp
                            )
                            .as_bytes(),
                        )
                        .await;
                });
            }
        });
        let config =
            RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string())
                .with_timeout(Duration::from_secs(5))
                .with_max_retries(0);
        let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let start = Instant::now();
        let task = tokio::spawn(async move {
            reranker
                .rerank_cancellable("query", &["document".to_string()], &cancel_clone)
                .await
        });
        // Give request time to be in-flight
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("cancellation of in-flight must be prompt")
            .unwrap();
        assert!(
            matches!(result, Err(RerankerError::Cancelled)),
            "in-flight cancel must yield Cancelled, got {result:?}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "in-flight cancellation should not wait for server response"
        );
    }

    #[tokio::test]
    async fn api_reranker_cancels_during_rate_limit_wait() {
        // 429 with Retry-After 10s, cap 10s -> retry_delay 10s; cancel during wait
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let cnt = Arc::clone(&count);
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let c = Arc::clone(&cnt);
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    let _ = stream
                        .write_all(b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 10\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                        .await;
                });
            }
        });
        let config =
            RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string())
                .with_timeout(Duration::from_secs(5))
                .with_max_retries(2)
                .with_rate_limit_wait_cap(Duration::from_secs(10));
        let reranker = ApiReranker::new_with_max_rerank_batch(config, 1).unwrap();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let start = Instant::now();
        let task = tokio::spawn(async move {
            reranker
                .rerank_cancellable("query", &["document".to_string()], &cancel_clone)
                .await
        });
        // Wait for first request to complete and enter backoff wait
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("cancel during rate-limit wait must be prompt")
            .unwrap();
        assert!(matches!(result, Err(RerankerError::Cancelled)));
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "cancel should interrupt rate-limit wait"
        );
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "cancel during wait must prevent second request"
        );
    }

    #[tokio::test]
    async fn api_reranker_cancels_between_batches_prevents_next_batch() {
        // 3 docs, batch 1 => 3 batches. First succeeds, cancel before second.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let cnt = Arc::clone(&count);
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let c = Arc::clone(&cnt);
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    // Read full request (may need multiple reads)
                    let mut request = Vec::new();
                    loop {
                        let n = stream.read(&mut buf).await.unwrap();
                        if n == 0 {
                            break;
                        }
                        request.extend_from_slice(&buf[..n]);
                        if request.windows(4).any(|w| w == b"\r\n\r\n") {
                            let end =
                                request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                            let header = String::from_utf8_lossy(&request[..end]);
                            if let Some(cl) = header
                                .lines()
                                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                            {
                                let len: usize =
                                    cl.split(':').nth(1).unwrap().trim().parse().unwrap_or(0);
                                if request.len() >= end + len {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    c.fetch_add(1, Ordering::SeqCst);
                    // Slow second batch handling to expose window, but first is quick
                    // For first request, respond immediately
                    let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
                    // Small delay to let cancel fire between batches
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    let _ = stream
                        .write_all(
                            format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                resp.len(),
                                resp
                            )
                            .as_bytes(),
                        )
                        .await;
                });
            }
        });
        let config =
            RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string())
                .with_timeout(Duration::from_secs(5))
                .with_max_retries(0);
        let reranker = ApiReranker::new_with_max_rerank_batch(config, 1).unwrap();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        // Spawn with 3 docs (batch 1 -> 3 requests). Cancel after first batch completes.
        let task = tokio::spawn(async move {
            let docs = vec!["d0".to_string(), "d1".to_string(), "d2".to_string()];
            reranker
                .rerank_cancellable("query", &docs, &cancel_clone)
                .await
        });
        // Wait for first batch to be counted
        for _ in 0..20 {
            if count.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // Cancel between batches
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("inter-batch cancel must be prompt")
            .unwrap();
        assert!(matches!(result, Err(RerankerError::Cancelled)));
        // Must not have issued all 3 batches
        let final_count = count.load(Ordering::SeqCst);
        assert!(
            final_count < 3,
            "cancel between batches must prevent remaining batches, got {final_count}"
        );
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
mod rate_limit_tests {
    use super::*;
    use crate::config::RetrievalConfig;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn retry_after_header_parses_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "3".parse().unwrap());
        assert_eq!(
            parse_retry_after_header(&headers),
            Some(Duration::from_secs(3))
        );
        let empty = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after_header(&empty), None);
    }

    #[test]
    fn openrouter_reset_timestamp_becomes_wait_duration() {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let reset_ms = now_ms + 30_000;
        let body = format!(
            r#"{{"error":{{"message":"Rate limit exceeded","code":429,"metadata":{{"headers":{{"X-RateLimit-Limit":"20","X-RateLimit-Remaining":"0","X-RateLimit-Reset":"{reset_ms}"}}}}}}}}"#
        );
        let wait = parse_rate_limit_reset(&body).expect("reset hint should parse");
        assert!(wait <= Duration::from_secs(30) && wait >= Duration::from_secs(25));
        assert_eq!(parse_rate_limit_reset("not json"), None);
        assert_eq!(parse_rate_limit_reset(r#"{"error":{"message":"x"}}"#), None);
    }

    /// Server that answers the first request with a 429 (`Retry-After: 1`)
    /// and subsequent requests with a valid rerank response.
    async fn spawn_rate_limit_server() -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let server_count = Arc::clone(&request_count);
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let count = Arc::clone(&server_count);
                tokio::spawn(async move {
                    let mut request = [0_u8; 4096];
                    let _ = stream.read(&mut request).await;
                    // No Content-Length: the close-delimited body avoids
                    // hand-counting bytes.
                    let response = if count.fetch_add(1, Ordering::SeqCst) == 0 {
                        "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nConnection: close\r\n\r\n{}".to_string()
                    } else {
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"results\":[{\"index\":0,\"relevance_score\":0.9}]}".to_string()
                    };
                    stream.write_all(response.as_bytes()).await.unwrap();
                });
            }
        });
        (format!("http://{address}"), request_count)
    }

    #[tokio::test]
    async fn waits_out_rate_limit_window_when_cap_configured() {
        let (url, request_count) = spawn_rate_limit_server().await;
        let retrieval = RetrievalConfig {
            reranker_timeout_secs: 5,
            reranker_max_retries: 2,
            reranker_rate_limit_wait_secs: Some(10),
            max_rerank_batch: 20,
            ..Default::default()
        };
        let cfg = RerankerConfig::new(url, "model".to_string(), "key".to_string());
        let reranker = ApiReranker::from_configs(cfg, &retrieval).unwrap();

        let scores = reranker.rerank("query", &["document".to_string()]).await;

        assert!(scores.is_ok(), "429 should be retried after the wait hint");
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_after_wait_is_clamped_by_cap() {
        // Server returns 429 with Retry-After: 4, cap 2 -> should wait ~2 not 4
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let cnt = Arc::clone(&count);
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let c = Arc::clone(&cnt);
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        let _ = stream
                            .write_all(
                                b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 4\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                            )
                            .await;
                    } else {
                        let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
                        let _ = stream
                            .write_all(
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                    resp.len(),
                                    resp
                                )
                                .as_bytes(),
                            )
                            .await;
                    }
                });
            }
        });
        let retrieval = RetrievalConfig {
            reranker_timeout_secs: 5,
            reranker_max_retries: 2,
            reranker_rate_limit_wait_secs: Some(2),
            max_rerank_batch: 20,
            ..Default::default()
        };
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let reranker = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let start = Instant::now();
        let scores = reranker
            .rerank("query", &["document".to_string()])
            .await
            .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(scores.len(), 1);
        assert_eq!(count.load(Ordering::SeqCst), 2);
        assert!(
            elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(4),
            "Retry-After 4 with cap 2 should wait ~2s, got {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn openrouter_body_reset_wait_is_honored_and_clamped_by_cap_end_to_end() {
        // Exact bca26e9 body shape: {"error":{"message":"...","code":429,"metadata":{"headers":{"X-RateLimit-Reset":"<unix-ms>"}}}}
        // Server returns 429 with body containing future reset ~3s from now, cap 2 -> should wait ~2
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let cnt = Arc::clone(&count);
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let c = Arc::clone(&cnt);
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    let mut request = Vec::new();
                    loop {
                        let n = stream.read(&mut buf).await.unwrap();
                        if n == 0 {
                            break;
                        }
                        request.extend_from_slice(&buf[..n]);
                        if request.windows(4).any(|w| w == b"\r\n\r\n") {
                            let end =
                                request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                            let header = String::from_utf8_lossy(&request[..end]);
                            if let Some(cl) = header
                                .lines()
                                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                            {
                                let len: usize =
                                    cl.split(':').nth(1).unwrap().trim().parse().unwrap_or(0);
                                if request.len() >= end + len {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;
                        // Future reset ~ 4s from now
                        let reset_ms = now_ms + 4000;
                        let body = format!(
                            r#"{{"error":{{"message":"Rate limit exceeded","code":429,"metadata":{{"headers":{{"X-RateLimit-Limit":"20","X-RateLimit-Remaining":"0","X-RateLimit-Reset":"{reset_ms}"}}}}}}}}"#
                        );
                        let header = format!(
                            "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(header.as_bytes()).await;
                        let _ = stream.write_all(body.as_bytes()).await;
                    } else {
                        let resp = r#"{"results":[{"index":0,"relevance_score":0.9}]}"#;
                        let _ = stream
                            .write_all(
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                    resp.len(),
                                    resp
                                )
                                .as_bytes(),
                            )
                            .await;
                    }
                });
            }
        });
        let retrieval = RetrievalConfig {
            reranker_timeout_secs: 10,
            reranker_max_retries: 2,
            reranker_rate_limit_wait_secs: Some(2),
            max_rerank_batch: 20,
            ..Default::default()
        };
        let cfg = RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string());
        let reranker = ApiReranker::from_configs(cfg, &retrieval).unwrap();
        let start = Instant::now();
        let scores = reranker
            .rerank("query", &["document".to_string()])
            .await
            .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(scores.len(), 1);
        assert_eq!(count.load(Ordering::SeqCst), 2);
        assert!(
            elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(4),
            "OpenRouter reset ~4s with cap 2 should clamp to ~2s, got {elapsed:?}"
        );
        // Verify body shape parsing: reset hint must be string unix-ms at correct pointer
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let future_reset = now_ms + 5000;
        let body = format!(
            r#"{{"error":{{"message":"Rate limit exceeded","code":429,"metadata":{{"headers":{{"X-RateLimit-Reset":"{future_reset}"}}}}}}}}"#
        );
        let parsed = parse_rate_limit_reset(&body).unwrap();
        assert!(parsed <= Duration::from_secs(6) && parsed >= Duration::from_secs(2));
    }

    #[tokio::test]
    async fn openrouter_reset_without_cap_degrades_quickly_end_to_end() {
        // Without cap, a long OpenRouter reset (30s) must not be waited out; should degrade quickly via generic backoff
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    let mut request = Vec::new();
                    loop {
                        let n = stream.read(&mut buf).await.unwrap();
                        if n == 0 {
                            break;
                        }
                        request.extend_from_slice(&buf[..n]);
                        if request.windows(4).any(|w| w == b"\r\n\r\n") {
                            let end =
                                request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                            let header = String::from_utf8_lossy(&request[..end]);
                            if let Some(cl) = header
                                .lines()
                                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                            {
                                let len: usize =
                                    cl.split(':').nth(1).unwrap().trim().parse().unwrap_or(0);
                                if request.len() >= end + len {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64;
                    let reset_ms = now_ms + 30_000;
                    let body = format!(
                        r#"{{"error":{{"message":"Rate limit exceeded","code":429,"metadata":{{"headers":{{"X-RateLimit-Reset":"{reset_ms}"}}}}}}}}"#
                    );
                    let header = format!(
                        "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(header.as_bytes()).await;
                    let _ = stream.write_all(body.as_bytes()).await;
                });
            }
        });
        let config =
            RerankerConfig::new(format!("http://{addr}"), "m".to_string(), "k".to_string())
                .with_timeout(Duration::from_secs(5))
                .with_max_retries(1);
        // No cap configured (default None)
        let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();
        let start = Instant::now();
        let result = reranker.rerank("query", &["document".to_string()]).await;
        let elapsed = start.elapsed();
        assert!(matches!(result, Err(RerankerError::RateLimitError { .. })));
        assert!(
            elapsed < Duration::from_secs(5),
            "without cap, long OpenRouter reset must degrade quickly, elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn rate_limit_degrades_quickly_without_wait_cap() {
        // Server that always answers 429 with a long Retry-After hint.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut request = [0_u8; 4096];
                    let _ = stream.read(&mut request).await;
                    stream
                        .write_all(b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 30\r\nConnection: close\r\n\r\n{}")
                        .await
                        .unwrap();
                });
            }
        });
        let config = RerankerConfig::new(url, "model".to_string(), "key".to_string())
            .with_timeout(Duration::from_secs(5))
            .with_max_retries(1);
        let reranker = ApiReranker::new_with_max_rerank_batch(config, 20).unwrap();

        let started = Instant::now();
        let result = reranker.rerank("query", &["document".to_string()]).await;

        assert!(matches!(result, Err(RerankerError::RateLimitError { .. })));
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "without a wait cap the Retry-After hint must not delay retries"
        );
    }
}

#[cfg(test)]
mod shortfall_tests {
    use super::*;
    use crate::types::Language;

    fn result(index: usize) -> SearchResult {
        SearchResult {
            file_path: format!("candidate-{index}.rs"),
            line_start: 1,
            line_end: 1,
            content: format!("candidate {index}"),
            language: Language::Rust,
            score: 1.0,
            symbol_name: None,
            symbol_type: None,
            part_index: None,
        }
    }

    #[tokio::test]
    async fn duplicate_reranker_indices_are_skipped() {
        struct DuplicateReranker;

        impl Reranker for DuplicateReranker {
            async fn rerank(
                &self,
                _query: &str,
                _documents: &[String],
            ) -> Result<Vec<RerankScore>, RerankerError> {
                Ok(vec![
                    RerankScore {
                        index: 0,
                        relevance_score: 1.0,
                    },
                    RerankScore {
                        index: 0,
                        relevance_score: 0.5,
                    },
                ])
            }
        }

        let results = [result(0), result(1)];
        let reranked = rerank_results(&DuplicateReranker, "query", &results, 2)
            .await
            .unwrap();

        assert_eq!(reranked.len(), 1);
        assert_eq!(reranked[0].content, "candidate 0");
    }
}

#[cfg(test)]
#[path = "reranker_tests.rs"]
mod tests;
