use crate::config::{InferenceBackend, VeraConfig};
use crate::embedding::local_provider::LocalEmbeddingProvider;
use crate::embedding::model2vec_provider::Model2VecProvider;
use crate::embedding::provider::{
    EmbeddingError, EmbeddingProvider, EmbeddingProviderConfig, OpenAiProvider,
};
use crate::local_models::configured_local_model_name;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub enum DynamicProvider {
    Api(OpenAiProvider),
    Local(LocalEmbeddingProvider),
    Model2Vec(Model2VecProvider),
}

impl EmbeddingProvider for DynamicProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match self {
            Self::Api(p) => p.embed_batch(texts).await,
            Self::Local(p) => p.embed_batch(texts).await,
            Self::Model2Vec(p) => p.embed_batch(texts).await,
        }
    }

    fn expected_dim(&self) -> Option<usize> {
        match self {
            Self::Api(p) => p.expected_dim(),
            Self::Local(p) => p.expected_dim(),
            Self::Model2Vec(p) => p.expected_dim(),
        }
    }

    fn prepare_document_text(&self, document: &str) -> String {
        match self {
            Self::Api(p) => p.prepare_document_text(document),
            Self::Local(p) => p.prepare_document_text(document),
            Self::Model2Vec(p) => p.prepare_document_text(document),
        }
    }

    fn prepare_query_text(&self, query: &str) -> String {
        match self {
            Self::Api(p) => p.prepare_query_text(query),
            Self::Local(p) => p.prepare_query_text(query),
            Self::Model2Vec(p) => p.prepare_query_text(query),
        }
    }

    fn max_batch_size(&self) -> Option<usize> {
        match self {
            Self::Api(p) => p.max_batch_size(),
            Self::Local(p) => p.max_batch_size(),
            Self::Model2Vec(p) => p.max_batch_size(),
        }
    }

    async fn embed_batch_cancellable(
        &self,
        texts: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match self {
            Self::Api(p) => p.embed_batch_cancellable(texts, cancel).await,
            Self::Local(p) => p.embed_batch_cancellable(texts, cancel).await,
            Self::Model2Vec(p) => p.embed_batch_cancellable(texts, cancel).await,
        }
    }
}

pub async fn create_dynamic_provider(
    config: &VeraConfig,
    backend: InferenceBackend,
) -> anyhow::Result<(DynamicProvider, String)> {
    match backend {
        InferenceBackend::OnnxJina(ep) => {
            let gpu_mem_limit_mb = config.embedding.gpu_mem_limit_mb;
            let p = LocalEmbeddingProvider::new_with_ep_and_mem_limit(ep, gpu_mem_limit_mb).await.map_err(|e| {
                anyhow::anyhow!("Failed to initialize local embedding provider: {e}\nHint: check network connection or manually place model at ~/.vera/models/")
            })?;
            Ok((DynamicProvider::Local(p), configured_local_model_name()))
        }
        InferenceBackend::PotionCode => {
            let p = Model2VecProvider::new_potion_code().await.map_err(|e| {
                anyhow::anyhow!("Failed to initialize potion-code provider: {e}\nHint: run `vera repair --potion-code` to fetch missing model assets.")
            })?;
            Ok((
                DynamicProvider::Model2Vec(p),
                crate::local_models::potion_code_model_name().to_string(),
            ))
        }
        InferenceBackend::Api => {
            let provider_config = EmbeddingProviderConfig::from_env()
                .map_err(|err| anyhow::anyhow!("embedding API not configured: {err}\nHint: set EMBEDDING_MODEL_BASE_URL, EMBEDDING_MODEL_ID, and EMBEDDING_MODEL_API_KEY environment variables, or use --potion-code for local CPU inference."))?;
            let model_name = provider_config.model_id.clone();
            let provider_config = provider_config
                .with_timeout(Duration::from_secs(config.embedding.timeout_secs))
                .with_max_retries(config.embedding.max_retries);
            let p = OpenAiProvider::new(provider_config)
                .map_err(|err| anyhow::anyhow!("failed to initialize embedding provider: {err}"))?;
            Ok((DynamicProvider::Api(p), model_name))
        }
    }
}
