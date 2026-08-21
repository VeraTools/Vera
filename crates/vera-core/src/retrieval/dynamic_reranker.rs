use crate::config::{InferenceBackend, VeraConfig};
use crate::retrieval::local_reranker::LocalReranker;
use crate::retrieval::reranker::{
    ApiReranker, RerankScore, Reranker, RerankerConfig, RerankerError,
};
use anyhow::Result;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub enum DynamicReranker {
    Api(ApiReranker),
    Local(LocalReranker),
}

impl Reranker for DynamicReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<RerankScore>, RerankerError> {
        match self {
            Self::Api(p) => p.rerank(query, documents).await,
            Self::Local(p) => p.rerank(query, documents).await,
        }
    }

    async fn rerank_cancellable(
        &self,
        query: &str,
        documents: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<RerankScore>, RerankerError> {
        match self {
            Self::Api(p) => p.rerank_cancellable(query, documents, cancel).await,
            Self::Local(p) => p.rerank_cancellable(query, documents, cancel).await,
        }
    }
}

pub async fn create_dynamic_reranker(
    config: &VeraConfig,
    backend: InferenceBackend,
) -> anyhow::Result<Option<DynamicReranker>> {
    if !config.retrieval.reranking_enabled {
        return Ok(None);
    }

    match backend {
        InferenceBackend::OnnxJina(ep) => {
            // Prefer a configured API reranker over the local ONNX one: hosted
            // rerankers are typically higher quality, and setting
            // RERANKER_MODEL_* signals explicit intent to use one.
            if let Ok(cfg) = RerankerConfig::from_env() {
                let cfg = cfg
                    .with_timeout(Duration::from_secs(30))
                    .with_max_retries(2);
                let p =
                    ApiReranker::new_with_max_rerank_batch(cfg, config.retrieval.max_rerank_batch)
                        .map_err(|err| anyhow::anyhow!("failed to init reranker: {err}"))?;
                return Ok(Some(DynamicReranker::Api(p)));
            }
            let p = LocalReranker::new_with_ep(ep)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize local reranker: {e}\nHint: check network connection or manually place model at ~/.vera/models/"))?;
            Ok(Some(DynamicReranker::Local(p)))
        }
        InferenceBackend::PotionCode => Ok(None),
        InferenceBackend::Api => match RerankerConfig::from_env() {
            Ok(cfg) => {
                let cfg = cfg
                    .with_timeout(Duration::from_secs(30))
                    .with_max_retries(2);
                let p =
                    ApiReranker::new_with_max_rerank_batch(cfg, config.retrieval.max_rerank_batch)
                        .map_err(|err| anyhow::anyhow!("failed to init reranker: {err}"))?;
                Ok(Some(DynamicReranker::Api(p)))
            }
            Err(_) => Ok(None),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::EnvVarGuard;

    #[tokio::test]
    async fn api_reranker_batches_by_the_configured_value_not_the_environment() {
        // Held across the await and dropped on any unwind, so the credentials
        // and the batch value never outlive the test.
        let _env = EnvVarGuard::set(&[
            ("RERANKER_MODEL_BASE_URL", "http://127.0.0.1:19998/v1"),
            ("RERANKER_MODEL_ID", "test-model"),
            ("RERANKER_MODEL_API_KEY", "test-key"),
            // The value the old second env lookup would have produced.
            ("VERA_MAX_RERANK_BATCH", "20"),
        ]);

        let mut config = VeraConfig::default();
        config.retrieval.reranking_enabled = true;
        config.retrieval.max_rerank_batch = 8;

        let reranker = create_dynamic_reranker(&config, InferenceBackend::Api)
            .await
            .unwrap();

        let Some(DynamicReranker::Api(api)) = reranker else {
            panic!("expected an API reranker for InferenceBackend::Api");
        };
        assert_eq!(
            api.max_rerank_batch, 8,
            "retrieval.max_rerank_batch must reach the reranker"
        );
    }
}
