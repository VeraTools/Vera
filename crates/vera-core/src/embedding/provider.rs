//! Embedding provider abstraction and OpenAI-compatible implementation.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::chunk_text;
use crate::local_models::CODERANK_QUERY_PREFIX;
use crate::types::Chunk;

// ── Error types ──────────────────────────────────────────────────────

/// Errors specific to the embedding pipeline.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Authentication failure (invalid or missing API key).
    #[error("embedding API authentication failed: {message}")]
    AuthError { message: String },

    /// Cannot reach the embedding endpoint.
    #[error("embedding API connection failed: {message}")]
    ConnectionError { message: String },

    /// The request exceeded its client-side timeout. Retrying could duplicate
    /// work when an upstream proxy continues processing after disconnect.
    #[error("embedding API request timed out: {message}")]
    TimeoutError { message: String },

    /// The API returned a non-auth, non-connection error.
    #[error("embedding API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    /// Rate limit exceeded.
    #[error("embedding API rate limit exceeded: {message}")]
    RateLimitError { message: String },

    /// Unexpected response format.
    #[error("unexpected embedding API response: {message}")]
    ResponseError { message: String },

    /// Request was cancelled because the client disconnected.
    #[error("embedding cancelled")]
    Cancelled,
}

/// Wrap an internal failure as an `ApiError` with status 500 (local pipeline
/// errors surface through the same variant as remote API errors).
pub(crate) fn api_err(error: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::ApiError {
        status: 500,
        message: error.to_string(),
    }
}

// ── Provider trait ───────────────────────────────────────────────────

/// Trait abstracting an embedding provider.
///
/// Implementations must be able to embed a batch of text inputs and return
/// one vector per input. Vectors must have consistent dimensionality.
#[allow(async_fn_in_trait)]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of text inputs, returning one vector per input.
    ///
    /// The returned vectors must all have the same dimensionality.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Return the expected vector dimensionality (if known ahead of time).
    fn expected_dim(&self) -> Option<usize>;

    /// Rewrite query text for providers that require asymmetric query prefixes.
    fn prepare_query_text(&self, query: &str) -> String {
        query.to_string()
    }

    /// Rewrite passage text for providers that require asymmetric document
    /// prefixes. Applied on the indexing path only.
    fn prepare_document_text(&self, document: &str) -> String {
        document.to_string()
    }

    /// Return the maximum number of inputs the provider accepts per request.
    ///
    /// `None` means Vera should use the configured batch size as-is.
    fn max_batch_size(&self) -> Option<usize> {
        None
    }

    /// Like `embed_batch`, but aborts between sub-batches if `cancel` is fired.
    ///
    /// The default implementation ignores the token and delegates to `embed_batch`.
    /// Providers that run blocking inference in sub-batch loops should override this
    /// to check `cancel.is_cancelled()` between iterations so client disconnects
    /// stop GPU work without waiting for the entire request to finish.
    async fn embed_batch_cancellable(
        &self,
        texts: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let _ = cancel;
        self.embed_batch(texts).await
    }
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for an OpenAI-compatible embedding provider.
#[derive(Clone)]
pub struct EmbeddingProviderConfig {
    /// Base URL for the API (e.g. "https://api.openai.com/v1").
    pub base_url: String,
    /// Model identifier (e.g. "Qwen/Qwen3-Embedding-8B").
    pub model_id: String,
    /// API key (never logged or exposed).
    api_key: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retries on transient errors.
    pub max_retries: u32,
    /// Optional prefix prepended to query text for asymmetric embedding models.
    /// Read from `EMBEDDING_QUERY_PREFIX` env var.
    pub query_prefix: Option<String>,
}

impl std::fmt::Debug for EmbeddingProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingProviderConfig")
            .field("base_url", &self.base_url)
            .field("model_id", &self.model_id)
            .field("api_key", &"[REDACTED]")
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl EmbeddingProviderConfig {
    /// Create a new config. The API key is stored opaquely and never exposed.
    pub fn new(base_url: String, model_id: String, api_key: String) -> Self {
        Self {
            base_url,
            model_id,
            api_key,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            query_prefix: None,
        }
    }

    /// Create config from environment variables.
    ///
    /// Reads:
    /// - `EMBEDDING_MODEL_BASE_URL`
    /// - `EMBEDDING_MODEL_ID`
    /// - `EMBEDDING_MODEL_API_KEY`
    /// - `EMBEDDING_QUERY_PREFIX` (optional override; auto-detected from model ID if unset)
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("EMBEDDING_MODEL_BASE_URL")
            .context("EMBEDDING_MODEL_BASE_URL not set")?;
        let model_id = std::env::var("EMBEDDING_MODEL_ID").context("EMBEDDING_MODEL_ID not set")?;
        let api_key =
            std::env::var("EMBEDDING_MODEL_API_KEY").context("EMBEDDING_MODEL_API_KEY not set")?;
        let query_prefix = std::env::var("EMBEDDING_QUERY_PREFIX")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| default_query_prefix_for_model(&model_id));

        let mut config = Self::new(base_url, model_id, api_key);
        config.query_prefix = query_prefix;
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
}

// ── Query prefix auto-detection ──────────────────────────────────────

/// Auto-detect the query prefix for known asymmetric embedding model families.
///
/// Returns `None` for symmetric models or unrecognized model IDs.
/// Users can always override via `EMBEDDING_QUERY_PREFIX` env var.
fn default_query_prefix_for_model(model_id: &str) -> Option<String> {
    let id = model_id.to_lowercase();
    if id.contains("qwen3-embedding") || id.contains("qwen3_embedding") {
        Some("Instruct: Given a code search query, retrieve relevant code snippets that match the query\nQuery: ".into())
    } else if id.contains("coderankembed") {
        // `prepare_query_text` concatenates without a separator, so the
        // trailing space belongs here rather than in the shared constant.
        Some(format!("{CODERANK_QUERY_PREFIX} "))
    } else if id.contains("e5-") || id.contains("e5_") {
        Some("query: ".into())
    } else if id.contains("bge-") || id.contains("bge_") {
        Some("Represent this sentence for searching relevant passages: ".into())
    } else {
        // Unrecognized model: try fetching prefix from HuggingFace.
        fetch_query_prefix_from_hf(model_id)
    }
}

/// Try to fetch the default query prompt from a model's HuggingFace `tokenizer_config.json`.
///
/// Many HuggingFace models store a default retrieval prompt under
/// `prompts.retrieval` or `default_prompt` in their tokenizer config.
/// This is a best-effort fallback; returns `None` on any failure.
fn fetch_query_prefix_from_hf(model_id: &str) -> Option<String> {
    // Only attempt if model_id looks like a HuggingFace repo (contains '/').
    if !model_id.contains('/') {
        return None;
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/main/tokenizer_config.json",
        model_id
    );
    debug!(model_id, url = %url, "fetching query prefix from HuggingFace");

    let body = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?
        .get(&url)
        .send()
        .ok()?
        .text()
        .ok()?;

    let config: serde_json::Value = serde_json::from_str(&body).ok()?;

    // Check common locations for a retrieval/query prompt.
    let prompt = config
        .get("prompts")
        .and_then(|p| p.get("retrieval").or_else(|| p.get("query")))
        .and_then(|v| v.as_str())
        .or_else(|| config.get("default_prompt").and_then(|v| v.as_str()));

    prompt.map(|p| {
        debug!(model_id, prefix = %p, "auto-detected query prefix from HuggingFace");
        format!("{p} ")
    })
}

// ── OpenAI-compatible provider ───────────────────────────────────────

/// OpenAI-compatible embedding provider.
///
/// Works with any API that implements the OpenAI `/v1/embeddings` endpoint,
/// including Nebius, Together, Fireworks, vLLM, etc.
pub struct OpenAiProvider {
    client: reqwest::Client,
    config: EmbeddingProviderConfig,
}

impl OpenAiProvider {
    /// Create a new provider from configuration.
    pub fn new(config: EmbeddingProviderConfig) -> Result<Self> {
        crate::init_tls();
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to create HTTP client")?;

        Ok(Self { client, config })
    }

    /// Build the embeddings endpoint URL.
    fn endpoint_url(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}/embeddings")
    }

    /// Execute a single embedding API call with retry logic.
    ///
    /// Rate limit errors (429) get extra retries with longer backoffs
    /// to respect API quotas while still completing the operation.
    async fn call_api(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = self.endpoint_url();
        let body = EmbeddingRequest {
            model: &self.config.model_id,
            input: texts,
        };

        let mut last_err = None;
        let mut retries = 0;
        loop {
            if retries > 0 {
                let is_rate_limit = matches!(last_err, Some(EmbeddingError::RateLimitError { .. }));
                let delay = if is_rate_limit {
                    // Rate limit: wait 2-4s with exponential backoff.
                    Duration::from_secs(2 + u64::from(retries.min(2)))
                } else {
                    Duration::from_millis(500 * 2u64.pow(retries.min(5) - 1))
                };
                debug!(
                    attempt = retries + 1,
                    delay_ms = delay.as_millis(),
                    rate_limited = is_rate_limit,
                    "retrying embedding API"
                );
                tokio::time::sleep(delay).await;
            }

            match self.send_request(&url, &body).await {
                Ok(vectors) => return Ok(vectors),
                Err(e) => {
                    // A timed-out request may still be running behind a proxy,
                    // so replaying it can create an abandoned-work queue.
                    // Context-size errors are handled by embed_batch_resilient.
                    if !is_retryable_error(&e) {
                        return Err(e);
                    }

                    let retry_limit = retry_limit_for_error(&e, self.config.max_retries);
                    warn!(
                        attempt = retries + 1,
                        max = retry_limit + 1,
                        error = %e,
                        "embedding API call failed"
                    );
                    if retries >= retry_limit {
                        return Err(e);
                    }
                    last_err = Some(e);
                    retries += 1;
                }
            }
        }
    }

    /// Send a single HTTP request and parse the response.
    async fn send_request(
        &self,
        url: &str,
        body: &EmbeddingRequest<'_>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EmbeddingError::TimeoutError {
                        message: format!("request to embedding API timed out: {e}"),
                    }
                } else if e.is_connect() {
                    EmbeddingError::ConnectionError {
                        message: format!("failed to connect to embedding API: {e}"),
                    }
                } else {
                    EmbeddingError::ConnectionError {
                        message: format!("request failed: {e}"),
                    }
                }
            })?;

        let status = response.status().as_u16();

        if status == 401 || status == 403 {
            let text =
                read_error_response_text(response, "failed to read authentication error response")
                    .await?;
            return Err(EmbeddingError::AuthError {
                message: sanitize_error_message(&text),
            });
        }

        if status == 429 {
            let text =
                read_error_response_text(response, "failed to read rate limit response").await?;
            return Err(EmbeddingError::RateLimitError {
                message: sanitize_error_message(&text),
            });
        }

        if !response.status().is_success() {
            let text =
                read_error_response_text(response, "failed to read embedding error response")
                    .await?;
            // Some providers return 400 with "Unable to process" for transient
            // overload conditions. Treat these as rate limits so they get retried.
            if status == 400 && text.contains("Unable to process") {
                return Err(EmbeddingError::RateLimitError {
                    message: sanitize_error_message(&text),
                });
            }
            return Err(EmbeddingError::ApiError {
                status,
                message: sanitize_error_message(&text),
            });
        }

        let resp: EmbeddingResponse = response
            .json()
            .await
            .map_err(|error| response_read_error(error, "failed to parse embedding response"))?;

        // Sort by index to ensure correct ordering.
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);

        let vectors: Vec<Vec<f32>> = data.into_iter().map(|d| d.embedding).collect();

        if vectors.len() != body.input.len() {
            return Err(EmbeddingError::ResponseError {
                message: format!(
                    "expected {} embeddings, got {}",
                    body.input.len(),
                    vectors.len()
                ),
            });
        }

        Ok(vectors)
    }
}

fn response_read_error(error: reqwest::Error, context: &str) -> EmbeddingError {
    if error.is_timeout() {
        EmbeddingError::TimeoutError {
            message: format!("{context}: {error}"),
        }
    } else {
        EmbeddingError::ResponseError {
            message: format!("{context}: {error}"),
        }
    }
}

async fn read_error_response_text(
    response: reqwest::Response,
    context: &str,
) -> Result<String, EmbeddingError> {
    match response.text().await {
        Ok(text) => Ok(text),
        Err(error) if error.is_timeout() => Err(response_read_error(error, context)),
        Err(error) => Ok(format!("{context}: {error}")),
    }
}

impl EmbeddingProvider for OpenAiProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(texts).await
    }

    fn expected_dim(&self) -> Option<usize> {
        None
    }

    fn prepare_query_text(&self, query: &str) -> String {
        match &self.config.query_prefix {
            Some(prefix) => format!("{prefix}{query}"),
            None => query.to_string(),
        }
    }

    fn max_batch_size(&self) -> Option<usize> {
        provider_batch_limit(&self.config)
    }
}

// ── Cached embedding provider ────────────────────────────────────────

/// An in-memory LRU-style cache wrapper around any `EmbeddingProvider`.
///
/// Caches `query_text → embedding_vector` so that repeated identical queries
/// (common during interactive search sessions) skip the API call entirely.
/// The first query pays full API cost; subsequent identical queries resolve
/// in microseconds from cache.
///
/// The cache uses a bounded `HashMap` with capacity-based eviction: when the
/// cache exceeds `max_entries`, the oldest entry (by insertion time) is removed.
pub struct CachedEmbeddingProvider<P> {
    inner: P,
    cache: Mutex<LruCache>,
    /// Namespace prefix for cache keys (typically the model id or provider URL).
    /// Prevents cross-model cache poisoning when the same query text is embedded
    /// by different models producing incompatible vectors.
    namespace: String,
}

/// Simple bounded cache with insertion-order eviction.
struct LruCache {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
}

struct CacheEntry {
    vector: Vec<f32>,
    inserted_at: Instant,
}

impl LruCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries),
            max_entries: max_entries.max(1),
        }
    }

    fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.entries.get(key).map(|e| &e.vector)
    }

    fn insert(&mut self, key: String, vector: Vec<f32>) {
        // Evict oldest entry if at capacity and this is a new key.
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.inserted_at)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&oldest_key);
            }
        }
        self.entries.insert(
            key,
            CacheEntry {
                vector,
                inserted_at: Instant::now(),
            },
        );
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<P: EmbeddingProvider> CachedEmbeddingProvider<P> {
    /// Create a new cached provider wrapping the given inner provider.
    ///
    /// `max_entries` controls the maximum number of cached embeddings.
    /// A reasonable default for interactive use is 256–1024.
    ///
    /// `namespace` is a stable identifier for the embedding model (e.g.,
    /// model name, provider URL, or config key). It prefixes every cache key
    /// so that switching models invalidates the cache instead of silently
    /// returning vectors from the wrong model.
    pub fn new(inner: P, max_entries: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(max_entries)),
            namespace: String::new(),
        }
    }

    /// Create a cached provider with an explicit namespace for cache key isolation.
    pub fn with_namespace(inner: P, max_entries: usize, namespace: impl Into<String>) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(max_entries)),
            namespace: namespace.into(),
        }
    }

    /// Return the number of currently cached entries.
    #[cfg(test)]
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Build a cache key that includes the namespace (model identity) so that
    /// switching embedding models invalidates the cache rather than silently
    /// returning vectors from a different model.
    fn make_cache_key(&self, query: &str) -> String {
        if self.namespace.is_empty() {
            query.to_string()
        } else {
            format!("{}\0{}", self.namespace, query)
        }
    }
}

impl<P: EmbeddingProvider> EmbeddingProvider for CachedEmbeddingProvider<P> {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // For single-text batches (query embedding), check cache first.
        if texts.len() == 1 {
            let key = self.make_cache_key(&texts[0]);
            if let Some(cached) = self.cache.lock().unwrap().get(&key) {
                debug!("embedding cache hit for query");
                return Ok(vec![cached.clone()]);
            }
        }

        // Cache miss — delegate to inner provider.
        let vectors = self.inner.embed_batch(texts).await?;

        // Cache single-text results (query embeddings).
        if texts.len() == 1 && vectors.len() == 1 {
            let key = self.make_cache_key(&texts[0]);
            let vector = vectors[0].clone();
            self.cache.lock().unwrap().insert(key, vector);
            debug!("embedding cached for query");
        }

        Ok(vectors)
    }

    async fn embed_batch_cancellable(
        &self,
        texts: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Query cache hits do not perform provider work, so they can return
        // immediately even when cancellation has already fired.
        if texts.len() == 1 {
            let key = self.make_cache_key(&texts[0]);
            if let Some(cached) = self.cache.lock().unwrap().get(&key) {
                debug!("embedding cache hit for query");
                return Ok(vec![cached.clone()]);
            }
        }

        // Preserve cancellation on cache misses so wrapped local providers
        // can stop between their own inference sub-batches.
        let vectors = self.inner.embed_batch_cancellable(texts, cancel).await?;

        if texts.len() == 1 && vectors.len() == 1 {
            let key = self.make_cache_key(&texts[0]);
            let vector = vectors[0].clone();
            self.cache.lock().unwrap().insert(key, vector);
            debug!("embedding cached for query");
        }

        Ok(vectors)
    }

    fn expected_dim(&self) -> Option<usize> {
        self.inner.expected_dim()
    }

    fn prepare_query_text(&self, query: &str) -> String {
        self.inner.prepare_query_text(query)
    }

    fn prepare_document_text(&self, document: &str) -> String {
        self.inner.prepare_document_text(document)
    }

    fn max_batch_size(&self) -> Option<usize> {
        self.inner.max_batch_size()
    }
}

// ── Batch embedding orchestrator ─────────────────────────────────────

const TRUNCATED_EMBEDDING_MARKER: &str = "\n\n[truncated for embedding]";

fn provider_batch_limit(config: &EmbeddingProviderConfig) -> Option<usize> {
    let base_url = config.base_url.to_ascii_lowercase();
    let model_id = config.model_id.to_ascii_lowercase();

    if base_url.contains("generativelanguage.googleapis.com")
        || base_url.contains("ai.google.dev")
        || (base_url.contains("googleapis.com") && model_id.contains("gemini"))
        || model_id.contains("gemini")
    {
        return Some(100);
    }

    // Voyage AI enforces both a per-batch input count cap (128) and a
    // per-batch token cap (120k for voyage-code-3). The token cap is handled
    // adaptively via context_size_info; this caps input count as a safety net.
    if base_url.contains("api.voyageai.com") || model_id.starts_with("voyage-") {
        return Some(128);
    }

    None
}

fn effective_batch_size<P: EmbeddingProvider>(provider: &P, configured_batch_size: usize) -> usize {
    let configured_batch_size = configured_batch_size.max(1);
    match provider.max_batch_size().filter(|limit| *limit > 0) {
        Some(provider_limit) if provider_limit < configured_batch_size => {
            debug!(
                configured_batch_size,
                provider_limit,
                effective_batch_size = provider_limit,
                "clamped embedding batch size to provider limit"
            );
            provider_limit
        }
        _ => configured_batch_size,
    }
}

fn retry_limit_for_error(error: &EmbeddingError, configured_retries: u32) -> u32 {
    if matches!(error, EmbeddingError::RateLimitError { .. }) {
        configured_retries.saturating_add(4)
    } else {
        configured_retries
    }
}

fn is_retryable_error(error: &EmbeddingError) -> bool {
    !matches!(
        error,
        EmbeddingError::AuthError { .. } | EmbeddingError::TimeoutError { .. }
    ) && !is_context_size_error(error)
}

#[derive(Clone)]
struct EmbeddingBatchItem {
    original_index: usize,
    chunk_id: String,
    text: String,
}

fn embedding_error_message(error: &EmbeddingError) -> &str {
    match error {
        EmbeddingError::AuthError { message }
        | EmbeddingError::ConnectionError { message }
        | EmbeddingError::TimeoutError { message }
        | EmbeddingError::ApiError { message, .. }
        | EmbeddingError::RateLimitError { message }
        | EmbeddingError::ResponseError { message } => message,
        EmbeddingError::Cancelled => "embedding cancelled",
    }
}

struct ContextSizeInfo {
    max_tokens: usize,
    input_tokens: Option<usize>,
}

fn context_size_info(error: &EmbeddingError) -> Option<ContextSizeInfo> {
    let message = embedding_error_message(error);
    let lower = message.to_ascii_lowercase();
    if !lower.contains("context size")
        && !lower.contains("exceed_context_size_error")
        && !lower.contains("\"n_ctx\"")
        && !lower.contains("too large to process")
        && !lower.contains("max allowed tokens per submitted batch")
        && !lower.contains("maximum input length")
    {
        return None;
    }

    static N_CTX_RE: OnceLock<Regex> = OnceLock::new();
    static MAX_CONTEXT_RE: OnceLock<Regex> = OnceLock::new();
    static BATCH_SIZE_RE: OnceLock<Regex> = OnceLock::new();
    static INPUT_TOKENS_RE: OnceLock<Regex> = OnceLock::new();
    static N_PROMPT_RE: OnceLock<Regex> = OnceLock::new();
    static MAX_BATCH_TOKENS_RE: OnceLock<Regex> = OnceLock::new();
    static BATCH_TOKENS_RE: OnceLock<Regex> = OnceLock::new();
    static MAX_INPUT_LENGTH_RE: OnceLock<Regex> = OnceLock::new();
    let n_ctx_re = N_CTX_RE.get_or_init(|| Regex::new(r#""n_ctx"\s*:\s*(\d+)"#).unwrap());
    let max_context_re =
        MAX_CONTEXT_RE.get_or_init(|| Regex::new(r"max context size \((\d+)").unwrap());
    let batch_size_re = BATCH_SIZE_RE.get_or_init(|| Regex::new(r"batch size:\s*(\d+)").unwrap());
    let input_tokens_re =
        INPUT_TOKENS_RE.get_or_init(|| Regex::new(r"input \((\d+) tokens?\)").unwrap());
    let n_prompt_re =
        N_PROMPT_RE.get_or_init(|| Regex::new(r#""n_prompt_tokens"\s*:\s*(\d+)"#).unwrap());
    let max_batch_tokens_re = MAX_BATCH_TOKENS_RE
        .get_or_init(|| Regex::new(r"max allowed tokens per submitted batch is (\d+)").unwrap());
    let batch_tokens_re =
        BATCH_TOKENS_RE.get_or_init(|| Regex::new(r"your batch has (\d+) tokens?").unwrap());
    // OpenAI: `Invalid 'input[3]': maximum input length is 8192 tokens.`
    let max_input_length_re =
        MAX_INPUT_LENGTH_RE.get_or_init(|| Regex::new(r"maximum input length is (\d+)").unwrap());

    let max_tokens = n_ctx_re
        .captures(&lower)
        .and_then(|caps| caps.get(1))
        .or_else(|| max_context_re.captures(&lower).and_then(|caps| caps.get(1)))
        .or_else(|| batch_size_re.captures(&lower).and_then(|caps| caps.get(1)))
        .or_else(|| {
            max_batch_tokens_re
                .captures(&lower)
                .and_then(|caps| caps.get(1))
        })
        .or_else(|| {
            max_input_length_re
                .captures(&lower)
                .and_then(|caps| caps.get(1))
        })
        .and_then(|capture| capture.as_str().parse::<usize>().ok())
        .or(Some(8192))?;

    let input_tokens = n_prompt_re
        .captures(&lower)
        .and_then(|caps| caps.get(1))
        .or_else(|| {
            input_tokens_re
                .captures(&lower)
                .and_then(|caps| caps.get(1))
        })
        .or_else(|| {
            batch_tokens_re
                .captures(&lower)
                .and_then(|caps| caps.get(1))
        })
        .and_then(|capture| capture.as_str().parse::<usize>().ok());

    Some(ContextSizeInfo {
        max_tokens,
        input_tokens,
    })
}

fn is_context_size_error(error: &EmbeddingError) -> bool {
    context_size_info(error).is_some()
}

fn truncate_to_char_boundary(text: &str, max_chars: usize) -> &str {
    if text.chars().count() <= max_chars {
        return text;
    }

    text.char_indices()
        .nth(max_chars)
        .map(|(idx, _)| &text[..idx])
        .unwrap_or(text)
}

fn shrink_text_for_context_limit(
    text: &str,
    max_tokens: usize,
    input_tokens: Option<usize>,
) -> String {
    let current_chars = text.chars().count();
    if current_chars <= 1 {
        return text.to_string();
    }

    let marker_chars = TRUNCATED_EMBEDDING_MARKER.chars().count();

    let target_chars = if let Some(actual_tokens) = input_tokens.filter(|&t| t > 0) {
        // We know exactly how many tokens the current text produced.
        // Compute the real chars-per-token ratio and truncate precisely,
        // targeting 85% of the context limit as a safety margin.
        let ratio = current_chars as f64 / actual_tokens as f64;
        let safe_limit = (max_tokens as f64 * 0.85) as usize;
        (safe_limit as f64 * ratio) as usize
    } else {
        // No actual token count available. Use 75% of current length as
        // a conservative fallback (always makes progress).
        current_chars.saturating_mul(3) / 4
    }
    .max(1)
    .saturating_sub(marker_chars);

    if target_chars >= current_chars {
        return text.to_string();
    }

    let truncated = truncate_to_char_boundary(text, target_chars).trim_end();
    if truncated.is_empty() {
        return text.to_string();
    }

    let mut shrunk = truncated.to_string();
    shrunk.push_str(TRUNCATED_EMBEDDING_MARKER);
    shrunk
}

async fn embed_batch_resilient<P: EmbeddingProvider>(
    provider: &P,
    items: Vec<EmbeddingBatchItem>,
    cancel: &CancellationToken,
) -> Result<Vec<(usize, String, Vec<f32>)>, EmbeddingError> {
    let mut pending = vec![items];
    let mut completed = Vec::new();

    while let Some(batch) = pending.pop() {
        if cancel.is_cancelled() {
            return Err(EmbeddingError::Cancelled);
        }
        let texts: Vec<String> = batch.iter().map(|item| item.text.clone()).collect();

        let result = tokio::select! {
            biased;
            result = provider.embed_batch_cancellable(&texts, cancel) => result,
            _ = cancel.cancelled() => Err(EmbeddingError::Cancelled),
        };

        match result {
            Ok(vectors) => {
                completed.extend(
                    batch
                        .into_iter()
                        .zip(vectors)
                        .map(|(item, vector)| (item.original_index, item.chunk_id, vector)),
                );
            }
            Err(error) => {
                let Some(info) = context_size_info(&error) else {
                    return Err(error);
                };

                if batch.len() > 1 {
                    let mid = batch.len() / 2;
                    debug!(
                        batch_size = batch.len(),
                        token_limit = info.max_tokens,
                        "embedding batch exceeded provider context limit, retrying in smaller batches"
                    );
                    pending.push(batch[mid..].to_vec());
                    pending.push(batch[..mid].to_vec());
                    continue;
                }

                let item = batch.into_iter().next().expect("single-item batch");
                let shrunk_text =
                    shrink_text_for_context_limit(&item.text, info.max_tokens, info.input_tokens);
                if shrunk_text == item.text {
                    return Err(error);
                }

                warn!(
                    chunk_id = %item.chunk_id,
                    token_limit = info.max_tokens,
                    input_tokens = ?info.input_tokens,
                    original_chars = item.text.chars().count(),
                    shrunk_chars = shrunk_text.chars().count(),
                    "embedding input exceeded provider context limit; retrying with a truncated text"
                );
                pending.push(vec![EmbeddingBatchItem {
                    text: shrunk_text,
                    ..item
                }]);
            }
        }
    }

    completed.sort_by_key(|(original_index, _, _)| *original_index);
    Ok(completed)
}

/// Embed all chunks using the given provider with concurrent batch processing.
///
/// Splits chunks into batches and sends up to `max_concurrent` batches
/// simultaneously. This significantly reduces wall-clock time for large
/// repositories where the embedding API is the bottleneck.
///
/// Returns a vector of `(chunk_id, embedding)` pairs in the same order
/// as the input chunks.
pub async fn embed_chunks_concurrent<P: EmbeddingProvider>(
    provider: &P,
    chunks: &[Chunk],
    batch_size: usize,
    max_concurrent: usize,
    max_chunk_bytes: usize,
) -> Result<Vec<(String, Vec<f32>)>, EmbeddingError> {
    embed_chunks_concurrent_with_progress(
        provider,
        chunks,
        batch_size,
        max_concurrent,
        max_chunk_bytes,
        |_, _| {},
    )
    .await
}

/// Like `embed_chunks_concurrent` but calls `on_progress(done, total)` after each batch.
pub async fn embed_chunks_concurrent_with_progress<P, F>(
    provider: &P,
    chunks: &[Chunk],
    batch_size: usize,
    max_concurrent: usize,
    max_chunk_bytes: usize,
    on_progress: F,
) -> Result<Vec<(String, Vec<f32>)>, EmbeddingError>
where
    P: EmbeddingProvider,
    F: Fn(usize, usize),
{
    embed_chunks_concurrent_with_progress_and_cancellation(
        provider,
        chunks,
        batch_size,
        max_concurrent,
        max_chunk_bytes,
        &CancellationToken::new(),
        on_progress,
    )
    .await
}

/// Embed chunks concurrently while allowing the caller to cancel active batches.
///
/// The token remains owned by the indexing operation. Cancellation drops every
/// active provider future and returns [`EmbeddingError::Cancelled`] without
/// reporting further progress.
pub(crate) async fn embed_chunks_concurrent_with_progress_and_cancellation<P, F>(
    provider: &P,
    chunks: &[Chunk],
    batch_size: usize,
    max_concurrent: usize,
    max_chunk_bytes: usize,
    cancel: &CancellationToken,
    on_progress: F,
) -> Result<Vec<(String, Vec<f32>)>, EmbeddingError>
where
    P: EmbeddingProvider,
    F: Fn(usize, usize),
{
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = effective_batch_size(provider, batch_size);
    let max_concurrent = max_concurrent.max(1);
    let total = chunks.len();
    let total_batches = total.div_ceil(batch_size);

    debug!(
        total_chunks = total,
        batch_size, total_batches, max_concurrent, "starting concurrent embedding"
    );

    let mut indexed_chunks: Vec<(usize, &Chunk)> = chunks.iter().enumerate().collect();
    indexed_chunks.sort_by_key(|(_, c)| c.content.len());

    let body_budget = budget_after_prefix(max_chunk_bytes, document_prefix_overhead(provider));

    let batch_inputs: Vec<Vec<EmbeddingBatchItem>> = indexed_chunks
        .chunks(batch_size)
        .map(|batch| {
            batch
                .iter()
                .map(|(orig_idx, chunk)| EmbeddingBatchItem {
                    original_index: *orig_idx,
                    chunk_id: chunk.id.clone(),
                    text: provider
                        .prepare_document_text(&chunk_to_embedding_text(chunk, body_budget)),
                })
                .collect()
        })
        .collect();

    let mut all_results: Vec<(usize, String, Vec<f32>)> = Vec::with_capacity(total);
    let mut done_count: usize = 0;

    for group_start in (0..batch_inputs.len()).step_by(max_concurrent) {
        let group_end = (group_start + max_concurrent).min(batch_inputs.len());
        let group = &batch_inputs[group_start..group_end];

        let futures: Vec<_> = group
            .iter()
            .enumerate()
            .map(|(i, items)| {
                let batch_idx = group_start + i;
                async move {
                    debug!(batch = batch_idx + 1, total_batches, "embedding batch");
                    embed_batch_resilient(provider, items.clone(), cancel).await
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        for result in results {
            // Batch errors (real provider failures or Cancelled) win over a
            // pending cancellation so the caller sees the actual failure.
            let batch_results = result?;
            if cancel.is_cancelled() {
                return Err(EmbeddingError::Cancelled);
            }
            done_count += batch_results.len();
            all_results.extend(batch_results);
            on_progress(done_count, total);
        }
    }

    all_results.sort_by_key(|(orig_idx, _, _)| *orig_idx);

    let results: Vec<(String, Vec<f32>)> = all_results
        .into_iter()
        .map(|(_, id, vec)| (id, vec))
        .collect();

    Ok(results)
}

/// Format a chunk's content for embedding.
///
/// Prepends metadata context (language, symbol info) to help the model
/// produce more code-aware embeddings. When `max_bytes > 0`, the code
/// content is truncated so the total text fits within the budget.
fn chunk_to_embedding_text(chunk: &Chunk, max_bytes: usize) -> String {
    if max_bytes > 0 {
        chunk_text::build_embedding_text_bounded(chunk, max_bytes)
    } else {
        chunk_text::build_embedding_text(chunk)
    }
}

/// Measure how many bytes `prepare_document_text` adds to a passage.
///
/// Read off the provider rather than off the model config, so it stays correct
/// for any implementation of the hook, including the identity default.
fn document_prefix_overhead<P: EmbeddingProvider>(provider: &P) -> usize {
    const PROBE: &str = "x";
    provider
        .prepare_document_text(PROBE)
        .len()
        .saturating_sub(PROBE.len())
}

/// Shrink the chunk byte budget so the document prefix fits inside it.
///
/// The prefix spends the same context window the budget exists to protect, so
/// it has to be reserved before truncation rather than added after it.
///
/// `0` means "unbounded" to [`chunk_to_embedding_text`], so a prefix at least
/// as large as the budget floors at 1 instead of saturating to 0, which would
/// silently switch truncation off entirely.
fn budget_after_prefix(max_chunk_bytes: usize, prefix_overhead: usize) -> usize {
    if max_chunk_bytes == 0 {
        return 0;
    }
    max_chunk_bytes.saturating_sub(prefix_overhead).max(1)
}

// ── Sanitization ─────────────────────────────────────────────────────

/// Remove any potential API key fragments from error messages.
///
/// API error bodies sometimes echo back parts of the request. This
/// ensures we never propagate credential material in error messages.
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
    // Strip anything that looks like a bearer token or key.
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

// ── API request/response types ───────────────────────────────────────

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    /// A mock embedding provider for unit testing.
    ///
    /// Returns deterministic vectors based on the input text length.
    pub struct MockProvider {
        pub dim: usize,
        pub fail_with: Option<EmbeddingError>,
    }

    impl MockProvider {
        pub fn new(dim: usize) -> Self {
            Self {
                dim,
                fail_with: None,
            }
        }

        pub fn failing(error: EmbeddingError) -> Self {
            Self {
                dim: 4,
                fail_with: Some(error),
            }
        }
    }

    impl EmbeddingProvider for MockProvider {
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if let Some(ref err) = self.fail_with {
                // Re-create the error since EmbeddingError is not Clone.
                return Err(match err {
                    EmbeddingError::AuthError { message } => EmbeddingError::AuthError {
                        message: message.clone(),
                    },
                    EmbeddingError::ConnectionError { message } => {
                        EmbeddingError::ConnectionError {
                            message: message.clone(),
                        }
                    }
                    EmbeddingError::TimeoutError { message } => EmbeddingError::TimeoutError {
                        message: message.clone(),
                    },
                    EmbeddingError::ApiError { status, message } => EmbeddingError::ApiError {
                        status: *status,
                        message: message.clone(),
                    },
                    EmbeddingError::RateLimitError { message } => EmbeddingError::RateLimitError {
                        message: message.clone(),
                    },
                    EmbeddingError::ResponseError { message } => EmbeddingError::ResponseError {
                        message: message.clone(),
                    },
                    EmbeddingError::Cancelled => EmbeddingError::Cancelled,
                });
            }

            Ok(texts
                .iter()
                .map(|text| {
                    // Deterministic hash-based seed from text content.
                    let mut hash: u64 = 5381;
                    for byte in text.bytes() {
                        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
                    }
                    let seed = hash as f32;
                    (0..self.dim)
                        .map(|i| {
                            let x = seed * (i as f32 + 1.0) * 0.001;
                            (x.sin() + 1.0) / 2.0
                        })
                        .collect()
                })
                .collect())
        }

        fn expected_dim(&self) -> Option<usize> {
            Some(self.dim)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn provider_with_truncated_error_body(
        status: &'static str,
    ) -> (OpenAiProvider, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request).await;
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: 32\r\nConnection: close\r\n\r\nshort"
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        });
        let config = EmbeddingProviderConfig::new(
            format!("http://{address}"),
            "test-model".into(),
            "test-key".into(),
        )
        .with_max_retries(0);
        (OpenAiProvider::new(config).unwrap(), server)
    }

    /// Cancels the token mid-request but still returns vectors, modelling an
    /// embedding response that completes at the same moment cancellation fires.
    struct CancelThenSucceedProvider;

    impl EmbeddingProvider for CancelThenSucceedProvider {
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(vec![vec![0.0; 8]; texts.len()])
        }

        async fn embed_batch_cancellable(
            &self,
            texts: &[String],
            cancel: &CancellationToken,
        ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            cancel.cancel();
            self.embed_batch(texts).await
        }

        fn expected_dim(&self) -> Option<usize> {
            Some(8)
        }
    }

    struct CancellationAwareProvider;

    impl EmbeddingProvider for CancellationAwareProvider {
        async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Err(EmbeddingError::ResponseError {
                message: "non-cancellable path used".to_string(),
            })
        }

        async fn embed_batch_cancellable(
            &self,
            _texts: &[String],
            cancel: &CancellationToken,
        ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if cancel.is_cancelled() {
                return Err(EmbeddingError::Cancelled);
            }
            Ok(vec![vec![1.0]])
        }

        fn expected_dim(&self) -> Option<usize> {
            Some(1)
        }
    }

    #[test]
    fn prepare_query_text_with_prefix() {
        let mut config = EmbeddingProviderConfig::new("http://x".into(), "m".into(), "k".into());
        config.query_prefix = Some("Instruct: search\nQuery: ".into());
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(
            provider.prepare_query_text("find foo"),
            "Instruct: search\nQuery: find foo"
        );
    }

    #[test]
    fn prepare_query_text_without_prefix() {
        let config = EmbeddingProviderConfig::new("http://x".into(), "m".into(), "k".into());
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(provider.prepare_query_text("find foo"), "find foo");
    }

    #[tokio::test]
    async fn cached_provider_forwards_cancellation_on_cache_miss() {
        let cached = CachedEmbeddingProvider::new(CancellationAwareProvider, 8);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = cached
            .embed_batch_cancellable(&["uncached query".to_string()], &cancel)
            .await;

        assert!(matches!(result, Err(EmbeddingError::Cancelled)));
        assert_eq!(cached.cache_size(), 0);
    }

    #[test]
    fn auto_detect_qwen3_prefix() {
        let prefix = default_query_prefix_for_model("Qwen/Qwen3-Embedding-8B");
        assert!(prefix.is_some());
        assert!(prefix.unwrap().contains("Query: "));
    }

    #[test]
    fn auto_detect_coderankembed_prefix() {
        let prefix = default_query_prefix_for_model("krlvi/CodeRankEmbed");
        // Exact: the published prompt in the model's
        // `config_sentence_transformers.json`, trailing space included.
        assert_eq!(
            prefix.as_deref(),
            Some("Represent this query for searching relevant code: ")
        );
    }

    /// The local ONNX path and the API path must embed the same query
    /// identically. They apply the prefix differently (`query_text` trims the
    /// *prefix* and rejoins it with one space, `prepare_query_text`
    /// concatenates a prefix that already carries its own trailing space), so
    /// this pins the resulting text rather than the constant.
    ///
    /// Neither path touches the query itself, so an un-normalized query has to
    /// survive verbatim and identically on both sides; that case is covered
    /// here so a one-sided `trim()` cannot be added without failing.
    #[test]
    fn coderankembed_query_text_matches_across_local_and_api_paths() {
        let cases = [
            (
                "find router code",
                "Represent this query for searching relevant code: find router code",
            ),
            (
                "  find router code  ",
                "Represent this query for searching relevant code:   find router code  ",
            ),
        ];

        for (query, expected) in cases {
            let local =
                crate::local_models::LocalEmbeddingModelConfig::coderankembed().query_text(query);

            let mut config = EmbeddingProviderConfig::new(
                "http://x".into(),
                "krlvi/CodeRankEmbed".into(),
                "k".into(),
            );
            config.query_prefix = default_query_prefix_for_model(&config.model_id);
            let api = OpenAiProvider::new(config)
                .unwrap()
                .prepare_query_text(query);

            assert_eq!(local, api, "paths diverged on {query:?}");
            assert_eq!(local, expected, "unexpected prefixed text for {query:?}");
        }
    }

    #[test]
    fn auto_detect_e5_prefix() {
        let prefix = default_query_prefix_for_model("intfloat/e5-large-v2");
        assert!(prefix.is_some());
        assert_eq!(prefix.unwrap(), "query: ");
    }

    #[test]
    fn auto_detect_bge_prefix() {
        let prefix = default_query_prefix_for_model("BAAI/bge-large-en-v1.5");
        assert!(prefix.is_some());
        assert!(prefix.unwrap().contains("Represent this sentence"));
    }

    #[test]
    fn auto_detect_unknown_model_no_prefix() {
        // Unknown model without '/' won't attempt HF fetch.
        let prefix = default_query_prefix_for_model("some-unknown-model");
        assert!(prefix.is_none());
    }

    #[test]
    fn detect_gemini_batch_limit_from_base_url() {
        let config = EmbeddingProviderConfig::new(
            "https://generativelanguage.googleapis.com/v1beta/openai".into(),
            "text-embedding-004".into(),
            "k".into(),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(provider.max_batch_size(), Some(100));
    }

    #[test]
    fn detect_gemini_batch_limit_from_model_id() {
        let config = EmbeddingProviderConfig::new(
            "http://localhost:4000/v1".into(),
            "gemini-embedding-001".into(),
            "k".into(),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(provider.max_batch_size(), Some(100));
    }

    #[test]
    fn non_gemini_provider_has_no_batch_limit_override() {
        let config = EmbeddingProviderConfig::new(
            "https://api.openai.com/v1".into(),
            "text-embedding-3-small".into(),
            "k".into(),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(provider.max_batch_size(), None);
    }

    #[test]
    fn detect_voyage_batch_limit_from_base_url() {
        let config = EmbeddingProviderConfig::new(
            "https://api.voyageai.com/v1".into(),
            "voyage-code-3".into(),
            "k".into(),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(provider.max_batch_size(), Some(128));
    }

    #[test]
    fn detect_voyage_batch_limit_from_model_id() {
        let config = EmbeddingProviderConfig::new(
            "http://localhost:4000/v1".into(),
            "voyage-code-3".into(),
            "k".into(),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(provider.max_batch_size(), Some(128));
    }

    #[test]
    fn context_size_info_parses_voyage_batch_token_error() {
        let err = EmbeddingError::ApiError {
            status: 400,
            message: "{\"detail\":\"Request to model 'voyage-code-3' failed. The max allowed tokens per submitted batch is 120000. Your batch has 124417 tokens after truncation. Please lower the number of tokens in the batch.\"}".to_string(),
        };
        let info = context_size_info(&err).expect("should recognize voyage batch token error");
        assert_eq!(info.max_tokens, 120000);
        assert_eq!(info.input_tokens, Some(124417));
        assert!(is_context_size_error(&err));
    }

    #[test]
    fn context_size_info_parses_openai_max_input_length_error() {
        // OpenAI reports only the limit, not the input token count. See issue #21.
        let err = EmbeddingError::ApiError {
            status: 400,
            message: "{\"error\":{\"message\":\"Invalid 'input[3]': maximum input length is 8192 tokens.\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":null}}".to_string(),
        };
        let info = context_size_info(&err).expect("should recognize openai max input length error");
        assert_eq!(info.max_tokens, 8192);
        assert_eq!(info.input_tokens, None);
        assert!(is_context_size_error(&err));
    }

    #[test]
    fn rate_limits_receive_only_the_documented_extra_retries() {
        let error = EmbeddingError::RateLimitError {
            message: "busy".into(),
        };
        assert_eq!(retry_limit_for_error(&error, 3), 7);
    }

    #[test]
    fn ordinary_failures_do_not_receive_rate_limit_retries() {
        let error = EmbeddingError::ConnectionError {
            message: "refused".into(),
        };
        assert_eq!(retry_limit_for_error(&error, 3), 3);
    }

    #[test]
    fn timeouts_are_not_retried() {
        let timeout = EmbeddingError::TimeoutError {
            message: "upstream may still be working".into(),
        };
        let connection = EmbeddingError::ConnectionError {
            message: "connection refused".into(),
        };

        assert!(!is_retryable_error(&timeout));
        assert!(is_retryable_error(&connection));
    }

    #[tokio::test]
    async fn delayed_rate_limit_body_is_a_non_retryable_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let server_request_count = request_count.clone();

        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                server_request_count.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut request = [0_u8; 2048];
                    let _ = stream.read(&mut request).await;
                    stream
                        .write_all(
                            b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\nConnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    let _ = stream.write_all(b"slow").await;
                });
            }
        });

        let config = EmbeddingProviderConfig::new(
            format!("http://{address}"),
            "test-model".into(),
            "test-key".into(),
        )
        .with_timeout(Duration::from_millis(50))
        .with_max_retries(3);
        let provider = OpenAiProvider::new(config).unwrap();

        let error = tokio::time::timeout(
            Duration::from_secs(1),
            provider.embed_batch(&["input".to_string()]),
        )
        .await
        .expect("a body timeout must not enter rate-limit backoff")
        .unwrap_err();

        assert!(matches!(error, EmbeddingError::TimeoutError { .. }));
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        server.abort();
    }

    #[tokio::test]
    async fn truncated_auth_body_preserves_authentication_classification() {
        let (provider, server) = provider_with_truncated_error_body("401 Unauthorized").await;
        let texts = ["input".to_string()];
        let body = EmbeddingRequest {
            model: &provider.config.model_id,
            input: &texts,
        };

        let error = provider
            .send_request(&provider.endpoint_url(), &body)
            .await
            .unwrap_err();

        assert!(matches!(error, EmbeddingError::AuthError { .. }));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn truncated_rate_limit_body_preserves_rate_limit_classification() {
        let (provider, server) = provider_with_truncated_error_body("429 Too Many Requests").await;
        let texts = ["input".to_string()];
        let body = EmbeddingRequest {
            model: &provider.config.model_id,
            input: &texts,
        };

        let error = provider
            .send_request(&provider.endpoint_url(), &body)
            .await
            .unwrap_err();

        assert!(matches!(error, EmbeddingError::RateLimitError { .. }));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn cancellation_after_completed_batch_suppresses_further_progress() {
        let chunks = vec![crate::types::Chunk {
            id: "chunk-0".to_string(),
            file_path: "src/main.rs".to_string(),
            line_start: 1,
            line_end: 1,
            content: "fn main() {}".to_string(),
            language: crate::types::Language::Rust,
            symbol_type: None,
            symbol_name: None,
        }];
        let progress_events = AtomicUsize::new(0);

        let result = embed_chunks_concurrent_with_progress_and_cancellation(
            &CancelThenSucceedProvider,
            &chunks,
            1,
            1,
            0,
            &CancellationToken::new(),
            |_, _| {
                progress_events.fetch_add(1, Ordering::SeqCst);
            },
        )
        .await;

        assert!(matches!(result, Err(EmbeddingError::Cancelled)));
        assert_eq!(progress_events.load(Ordering::SeqCst), 0);
    }

    /// Records the texts it is asked to embed and prefixes passages, so the
    /// test can prove the indexing funnel actually calls the document hook
    /// rather than merely defining it.
    struct RecordingProvider {
        seen: Mutex<Vec<String>>,
    }

    impl EmbeddingProvider for RecordingProvider {
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            self.seen.lock().unwrap().extend(texts.iter().cloned());
            Ok(texts.iter().map(|_| vec![0.0]).collect())
        }

        fn expected_dim(&self) -> Option<usize> {
            Some(1)
        }

        fn prepare_document_text(&self, document: &str) -> String {
            format!("Document: {document}")
        }

        fn prepare_query_text(&self, query: &str) -> String {
            format!("Query: {query}")
        }
    }

    #[tokio::test]
    async fn indexing_applies_the_document_prefix_and_not_the_query_prefix() {
        let chunks = vec![crate::types::Chunk {
            id: "chunk-0".to_string(),
            file_path: "src/main.rs".to_string(),
            line_start: 1,
            line_end: 1,
            content: "fn main() {}".to_string(),
            language: crate::types::Language::Rust,
            symbol_type: None,
            symbol_name: None,
        }];
        let provider = RecordingProvider {
            seen: Mutex::new(Vec::new()),
        };

        embed_chunks_concurrent_with_progress_and_cancellation(
            &provider,
            &chunks,
            1,
            1,
            0,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("embedding should succeed");

        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "expected exactly one embedded passage");
        assert!(
            seen[0].starts_with("Document: "),
            "indexing path did not apply the document prefix: {:?}",
            seen[0]
        );
        assert!(
            !seen[0].contains("Query: "),
            "indexed passage was given the query prefix: {:?}",
            seen[0]
        );
        assert!(
            seen[0].contains("fn main() {}"),
            "prefix replaced the chunk body instead of prefixing it: {:?}",
            seen[0]
        );
    }

    #[tokio::test]
    async fn the_document_prefix_fits_inside_the_chunk_byte_budget() {
        const BUDGET: usize = 200;
        let chunks = vec![crate::types::Chunk {
            id: "chunk-0".to_string(),
            file_path: "src/main.rs".to_string(),
            line_start: 1,
            line_end: 200,
            // Far past the budget, so the bounded builder is the thing that
            // decides the length. Short lines keep the line-boundary cut from
            // landing well inside the budget and hiding the overshoot.
            content: "ab\n".repeat(400),
            language: crate::types::Language::Rust,
            symbol_type: None,
            symbol_name: None,
        }];
        let provider = RecordingProvider {
            seen: Mutex::new(Vec::new()),
        };

        embed_chunks_concurrent_with_progress_and_cancellation(
            &provider,
            &chunks,
            1,
            1,
            BUDGET,
            &CancellationToken::new(),
            |_, _| {},
        )
        .await
        .expect("embedding should succeed");

        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "expected exactly one embedded passage");
        assert!(
            seen[0].starts_with("Document: "),
            "the passage lost its document prefix: {:?}",
            seen[0]
        );
        assert!(
            seen[0].contains("Code:\nab\nab"),
            "the reservation ate the chunk body: {:?}",
            seen[0]
        );
        assert!(
            seen[0].len() <= BUDGET,
            "prefixed passage is {} bytes against a {BUDGET}-byte budget",
            seen[0].len()
        );
    }

    #[test]
    fn a_prefix_larger_than_the_budget_still_leaves_a_budget() {
        // 0 means "unbounded" to chunk_to_embedding_text, so saturating to it
        // would turn an over-long prefix into no truncation at all.
        assert_eq!(budget_after_prefix(8, 10), 1);
        assert_eq!(budget_after_prefix(8, 8), 1);
        assert_eq!(budget_after_prefix(200, 10), 190);
        // An unbounded budget stays unbounded.
        assert_eq!(budget_after_prefix(0, 10), 0);
    }

    #[test]
    fn the_prefix_overhead_is_measured_from_the_provider() {
        let provider = RecordingProvider {
            seen: Mutex::new(Vec::new()),
        };
        assert_eq!(document_prefix_overhead(&provider), "Document: ".len());
        // A provider that takes the identity default reserves nothing.
        assert_eq!(document_prefix_overhead(&CancelThenSucceedProvider), 0);
    }
}
