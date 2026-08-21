//! Vera HTTP API server.
//!
//! Exposes OpenAI-compatible inference endpoints for standard vera clients:
//!
//! ```text
//! POST /v1/embeddings   OpenAI format  (EMBEDDING_MODEL_BASE_URL)
//! POST /v1/rerank       Cohere/Jina format  (RERANKER_MODEL_BASE_URL)
//! GET  /v1/health       liveness + model info
//! ```
//!
//! A regular vera client configured with `vera setup --api` pointing at
//! `http://host:port/v1` will work without any modifications.

mod handlers;
mod provider_cache;
pub mod types;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use vera_core::config::{InferenceBackend, VeraConfig};
use vera_core::embedding::DynamicProvider;
use vera_core::retrieval::DynamicReranker;

pub use provider_cache::CacheMode;
use provider_cache::ModelSlot;

/// Shared state injected into every handler.
pub struct AppState {
    pub api_key: Option<String>,
    /// Config used to create providers on-demand.
    pub config: VeraConfig,
    pub backend: InferenceBackend,
    /// Human-readable model name reported in /v1/health and embeddings responses.
    pub model_name: String,
    /// Whether a reranker is available (probed at startup).
    pub reranker_available: bool,
    /// The embedding model, cached independently of the reranker.
    pub(crate) embedding: Arc<ModelSlot<DynamicProvider>>,
    /// The reranker, cached independently of the embedding model.
    pub(crate) reranker: Arc<ModelSlot<DynamicReranker>>,
}

/// Start the Vera HTTP server.
///
/// Probes the embedding model and reranker at startup to validate the config,
/// then listens for connections on `host:port`. The embedding probe is kept and
/// becomes the first resident model; the reranker is loaded on its first request,
/// since holding it costs memory a server that only embeds never uses. Each model
/// is then held as `cache_mode` dictates.
///
/// - `config`     — vera retrieval/embedding config
/// - `backend`    — compute backend (API, CPU, GPU)
/// - `api_key`    — optional bearer token; `None` disables auth
/// - `host`       — bind address (e.g. `"127.0.0.1"` or `"0.0.0.0"`)
/// - `port`       — TCP port to listen on
/// - `cache_mode` — how long a loaded model stays resident
pub async fn run_server(
    config: VeraConfig,
    backend: InferenceBackend,
    api_key: Option<String>,
    host: &str,
    port: u16,
    cache_mode: CacheMode,
) -> Result<()> {
    eprintln!(
        "vera serve: initializing {} backend…",
        backend_label(backend)
    );

    // Probe-load to validate the config and obtain the model name.
    let (probe, model_name) = vera_core::embedding::create_dynamic_provider(&config, backend)
        .await
        .map_err(|e| anyhow::anyhow!("failed to load embedding model: {e}"))?;

    eprintln!("vera serve: embedding model ready ({})", model_name);

    let embedding = Arc::new(ModelSlot::new(cache_mode));
    let reranker: Arc<ModelSlot<DynamicReranker>> = Arc::new(ModelSlot::new(cache_mode));

    // The probe already paid for a full load, so hand it to the slot rather than
    // dropping it and making the first request load the same model again. Seeded
    // here, before the reranker probe, so `PerRequest` — where `seed` is a no-op —
    // still releases it before anything else is loaded.
    embedding.seed(Arc::new(probe)).await;

    // The reranker probe is deliberately not seeded: it is built here, read for
    // `reranker_available`, and dropped, so the first /v1/rerank loads it again.
    // Keeping it would cost ~670 MB resident on a server that only answers
    // /v1/embeddings and never reranks at all.
    let reranker_available = vera_core::retrieval::create_dynamic_reranker(&config, backend)
        .await
        .unwrap_or_else(|e| {
            eprintln!("vera serve: reranker unavailable ({e}), reranking disabled");
            None
        })
        .is_some();

    if reranker_available {
        eprintln!("vera serve: reranker ready");
    }

    let api_key = api_key.filter(|k| !k.is_empty());

    if api_key.is_some() {
        eprintln!("vera serve: API key authentication enabled");
    } else {
        eprintln!("vera serve: no API key set — unauthenticated access allowed");
    }

    match cache_mode {
        CacheMode::PerRequest => eprintln!(
            "vera serve: model cache disabled (--idle-timeout 0); every request rebuilds the model"
        ),
        CacheMode::Forever => eprintln!("vera serve: models stay loaded (no idle timeout)"),
        CacheMode::Idle(d) => eprintln!(
            "vera serve: models unload after {}s of inactivity",
            d.as_secs()
        ),
    }

    // Background task: unload each model after `cache_mode`'s idle timeout.
    if let CacheMode::Idle(timeout) = cache_mode {
        let embedding = Arc::clone(&embedding);
        let reranker = Arc::clone(&reranker);
        tokio::spawn(async move {
            let check_interval = (timeout / 4).max(Duration::from_secs(1));
            loop {
                tokio::time::sleep(check_interval).await;
                if embedding.evict_if_idle().await {
                    eprintln!("vera serve: embedding model unloaded (idle timeout reached)");
                }
                if reranker.evict_if_idle().await {
                    eprintln!("vera serve: reranker unloaded (idle timeout reached)");
                }
            }
        });
    }

    let state = Arc::new(AppState {
        api_key,
        config,
        backend,
        model_name,
        reranker_available,
        embedding,
        reranker,
    });

    let app = Router::new()
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/rerank", post(handlers::rerank))
        .route("/v1/health", get(handlers::health))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("vera serve: listening on http://{addr}");
    eprintln!();
    eprintln!("  Client setup:");
    eprintln!("    vera setup --api  (then set EMBEDDING_MODEL_BASE_URL=http://{addr}/v1)");
    axum::serve(listener, app).await?;
    Ok(())
}

fn backend_label(backend: InferenceBackend) -> &'static str {
    use vera_core::config::OnnxExecutionProvider;
    match backend {
        InferenceBackend::Api => "api",
        InferenceBackend::PotionCode => "potion-code (CPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu) => "cpu",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda) => "cuda (GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm) => "rocm (AMD GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl) => "directml (GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl) => "coreml (Apple GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino) => "openvino (Intel GPU)",
    }
}
