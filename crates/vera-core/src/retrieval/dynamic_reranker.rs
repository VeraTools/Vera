use crate::config::{InferenceBackend, VeraConfig};
use crate::retrieval::local_reranker::LocalReranker;
use crate::retrieval::reranker::{
    ApiReranker, RerankScore, Reranker, RerankerConfig, RerankerError,
};
use anyhow::Result;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RerankerSource {
    Api,
    Local(crate::config::OnnxExecutionProvider),
    None,
}

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
    let source = reranker_source(
        backend,
        config.retrieval.reranking_enabled,
        RerankerConfig::from_env().is_ok(),
    );

    match source {
        RerankerSource::Api => {
            let cfg = RerankerConfig::from_env()
                .map_err(|err| anyhow::anyhow!("failed to configure reranker: {err}"))?;
            let p = ApiReranker::from_configs(cfg, &config.retrieval)
                .map_err(|err| anyhow::anyhow!("failed to init reranker: {err}"))?;
            Ok(Some(DynamicReranker::Api(p)))
        }
        RerankerSource::Local(ep) => {
            let p = LocalReranker::new_with_ep(ep).await.map_err(|e| {
                anyhow::anyhow!("Failed to initialize local reranker: {e}\nHint: check network connection or manually place model at ~/.vera/models/")
            })?;
            Ok(Some(DynamicReranker::Local(p)))
        }
        RerankerSource::None => Ok(None),
    }
}

fn reranker_source(
    backend: InferenceBackend,
    reranking_enabled: bool,
    api_configured: bool,
) -> RerankerSource {
    if !reranking_enabled {
        return RerankerSource::None;
    }

    if api_configured {
        return RerankerSource::Api;
    }

    match backend {
        InferenceBackend::OnnxJina(ep) => RerankerSource::Local(ep),
        // Potion Code uses Model2Vec on CPU rather than an ONNX embedding
        // provider, so its local reranker has no alternate EP to inherit.
        InferenceBackend::PotionCode => {
            RerankerSource::Local(crate::config::OnnxExecutionProvider::Cpu)
        }
        InferenceBackend::Api => RerankerSource::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{InferenceBackend, OnnxExecutionProvider};
    use crate::test_env::run_env_test;

    #[test]
    fn reranker_source_respects_potion_gating_and_api_preference() {
        assert_eq!(
            reranker_source(InferenceBackend::PotionCode, false, false),
            RerankerSource::None
        );
        assert_eq!(
            reranker_source(InferenceBackend::PotionCode, true, false),
            RerankerSource::Local(OnnxExecutionProvider::Cpu)
        );
        assert_eq!(
            reranker_source(InferenceBackend::PotionCode, true, true),
            RerankerSource::Api
        );
    }

    #[test]
    fn reranker_source_preserves_onnx_jina_selection() {
        assert_eq!(
            reranker_source(
                InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl),
                false,
                false,
            ),
            RerankerSource::None
        );
        assert_eq!(
            reranker_source(
                InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda),
                true,
                false,
            ),
            RerankerSource::Local(OnnxExecutionProvider::Cuda)
        );
        assert_eq!(
            reranker_source(
                InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda),
                true,
                true,
            ),
            RerankerSource::Api
        );
    }

    #[test]
    fn api_reranker_batches_by_the_configured_value_not_the_environment() {
        run_env_test(
            "retrieval::dynamic_reranker::tests::api_reranker_batches_by_the_configured_value_not_the_environment_probe",
            &[
                ("RERANKER_MODEL_BASE_URL", Some("http://127.0.0.1:19998/v1")),
                ("RERANKER_MODEL_ID", Some("test-model")),
                ("RERANKER_MODEL_API_KEY", Some("test-key")),
                // The value the old second env lookup would have produced.
                ("VERA_MAX_RERANK_BATCH", Some("20")),
            ],
        );
    }

    #[tokio::test]
    #[ignore = "driven by api_reranker_batches_by_the_configured_value_not_the_environment"]
    async fn api_reranker_batches_by_the_configured_value_not_the_environment_probe() {
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
