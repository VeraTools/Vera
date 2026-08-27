use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use tokio::runtime::Runtime;
use vera_core::config::{InferenceBackend, VeraConfig};
use vera_core::embedding::{EmbeddingError, EmbeddingProvider};
use vera_core::indexing::{index_dir, index_repository};
use vera_core::retrieval::search_service::SearchContext;
use vera_core::types::SearchFilters;

use crate::lanes::LaneSpec;
use crate::runner::ToolAdapter;
use crate::types::RetrievalResult;

const EMBEDDING_DIM: usize = 64;
const MODEL_NAME: &str = "eval-hash-bm25-v1";
const RESULT_LIMIT: usize = 10;

/// Deterministic Vera adapter for regression testing.
///
/// This indexes real corpora with a lightweight hash embedding so the eval
/// harness can exercise Vera end-to-end without model downloads or API keys.
/// Query-time search uses the normal BM25 fallback path from `execute_search`,
/// which keeps the ranking and augmentation logic close to the CLI.
pub struct VeraBm25Adapter {
    runtime: Runtime,
    config: VeraConfig,
    search_context: SearchContext,
}

impl VeraBm25Adapter {
    pub fn new() -> anyhow::Result<Self> {
        let mut config = VeraConfig::default();
        config.retrieval.reranking_enabled = false;
        config.embedding.max_stored_dim = EMBEDDING_DIM;
        Ok(Self {
            runtime: Runtime::new()?,
            config,
            search_context: SearchContext::bm25_only(),
        })
    }
}

impl ToolAdapter for VeraBm25Adapter {
    fn name(&self) -> &str {
        "vera-bm25"
    }

    fn version(&self) -> String {
        format!("{MODEL_NAME}/{}", env!("CARGO_PKG_VERSION"))
    }

    fn search(
        &self,
        query: &str,
        repo_path: &str,
        path_scope: Option<&str>,
    ) -> Vec<RetrievalResult> {
        search_with(
            &self.runtime,
            &self.search_context,
            &self.config,
            query,
            repo_path,
            path_scope,
            "vera-bm25",
        )
    }

    fn index(&self, repo_path: &str) -> (f64, u64) {
        index_with(
            &self.runtime,
            &self.config,
            Path::new(repo_path),
            &HashEmbeddingProvider,
            MODEL_NAME,
            "vera-bm25",
        )
    }
}

struct HashEmbeddingProvider;

impl EmbeddingProvider for HashEmbeddingProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Ok(texts.iter().map(|text| hash_embedding(text)).collect())
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(EMBEDDING_DIM)
    }
}

fn hash_embedding(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIM];

    for token in tokenize(text) {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();
        let idx = (hash as usize) % EMBEDDING_DIM;
        let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
        vector[idx] += sign;
    }

    normalize(&mut vector);
    vector
}

fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|token| token.len() >= 2)
        .map(|token| token.to_ascii_lowercase())
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn search_with(
    runtime: &Runtime,
    search_context: &SearchContext,
    config: &VeraConfig,
    query: &str,
    repo_path: &str,
    path_scope: Option<&str>,
    label: &str,
) -> Vec<RetrievalResult> {
    let repo_path = Path::new(repo_path);
    if repo_path.as_os_str().is_empty() {
        return Vec::new();
    }

    let mut filters = SearchFilters::default();
    if let Some(scope) = path_scope {
        filters.path_glob = vec![format!("{scope}/**")];
    }

    match runtime.block_on(search_context.search(
        &index_dir(repo_path),
        query,
        None,
        config,
        &filters,
        RESULT_LIMIT,
    )) {
        Ok((results, _)) => results.into_iter().map(into_retrieval_result).collect(),
        // Abort rather than return empty results: a failed search would
        // otherwise be scored as a legitimate zero and skew aggregates.
        Err(err) => panic!("{label} search failed for {}: {err}", repo_path.display()),
    }
}

fn index_with<P: EmbeddingProvider>(
    runtime: &Runtime,
    config: &VeraConfig,
    repo_path: &Path,
    provider: &P,
    model_name: &str,
    label: &str,
) -> (f64, u64) {
    match runtime.block_on(index_repository(repo_path, provider, config, model_name)) {
        Ok(summary) => (summary.elapsed_secs, dir_size(&index_dir(repo_path))),
        Err(err) => panic!("{label} index failed for {}: {err:#}", repo_path.display()),
    }
}

fn into_retrieval_result(result: vera_core::types::SearchResult) -> RetrievalResult {
    RetrievalResult {
        file_path: result.file_path,
        line_start: result.line_start as usize,
        line_end: result.line_end as usize,
        score: result.score,
    }
}

fn dir_size(path: &Path) -> u64 {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        Ok(metadata) if metadata.is_dir() => fs::read_dir(path)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.filter_map(Result::ok))
            .map(|entry| dir_size(&entry.path()))
            .sum(),
        _ => 0,
    }
}

/// Full-pipeline Vera adapter that uses real ONNX models with a specified backend.
///
/// Unlike `VeraBm25Adapter`, this runs the complete hybrid search pipeline
/// (embedding + BM25 + RRF fusion + optional reranking) via `execute_search`.
/// Use `InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda)` for GPU
/// acceleration.
pub struct VeraFullAdapter {
    runtime: Runtime,
    config: VeraConfig,
    backend: InferenceBackend,
    name: String,
    search_context: SearchContext,
}

impl VeraFullAdapter {
    pub fn new_with_options(
        backend: InferenceBackend,
        reranking_enabled: bool,
        name: impl Into<String>,
        lane: &LaneSpec,
    ) -> anyhow::Result<Self> {
        let mut config = VeraConfig::default();
        config.retrieval.reranking_enabled = reranking_enabled;
        config.adjust_for_backend(backend);
        if let Some(batch_size) = lane.batch_size {
            config.embedding.batch_size = batch_size;
        }
        if let Some(max_concurrent) = lane.max_concurrent_requests {
            config.embedding.max_concurrent_requests = max_concurrent;
        }
        if let Some(timeout_secs) = lane.timeout_secs {
            config.embedding.timeout_secs = timeout_secs;
        }
        if let Some(max_retries) = lane.max_retries {
            config.embedding.max_retries = max_retries;
        }
        let runtime = Runtime::new()?;
        let search_context = runtime.block_on(SearchContext::new(&config, backend));
        if search_context.embedding_provider().is_none() {
            anyhow::bail!("failed to initialize the embedding provider for backend {backend}");
        }
        Ok(Self {
            runtime,
            config,
            backend,
            name: name.into(),
            search_context,
        })
    }
}

impl ToolAdapter for VeraFullAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> String {
        format!("vera-full-{}/{}", self.backend, env!("CARGO_PKG_VERSION"))
    }

    fn search(
        &self,
        query: &str,
        repo_path: &str,
        path_scope: Option<&str>,
    ) -> Vec<RetrievalResult> {
        search_with(
            &self.runtime,
            &self.search_context,
            &self.config,
            query,
            repo_path,
            path_scope,
            "vera-full",
        )
    }

    fn index(&self, repo_path: &str) -> (f64, u64) {
        let Some(provider) = self.search_context.embedding_provider() else {
            panic!("embedding provider unavailable for {}", self.backend);
        };
        let Some(model_name) = self.search_context.model_name() else {
            panic!("embedding model name unavailable for {}", self.backend);
        };
        index_with(
            &self.runtime,
            &self.config,
            Path::new(repo_path),
            provider,
            model_name,
            "vera-full",
        )
    }
}

pub fn repo_paths_from_manifest(
    repo_root: &Path,
    manifest: &crate::types::CorpusManifest,
) -> std::collections::HashMap<String, String> {
    let clone_root = resolve_clone_root(repo_root, &manifest.corpus.clone_root);
    manifest
        .repos
        .iter()
        .map(|repo| {
            (
                repo.name.clone(),
                clone_root.join(&repo.name).display().to_string(),
            )
        })
        .collect()
}

/// Extract benchmark_root scopes from corpus manifest.
/// Returns repo_name -> benchmark_root (e.g. "fastapi" -> "fastapi").
pub fn benchmark_roots_from_manifest(
    manifest: &crate::types::CorpusManifest,
) -> std::collections::HashMap<String, String> {
    manifest
        .repos
        .iter()
        .filter_map(|repo| {
            repo.benchmark_root
                .as_ref()
                .map(|root| (repo.name.clone(), root.clone()))
        })
        .collect()
}

fn resolve_clone_root(repo_root: &Path, clone_root: &str) -> PathBuf {
    let clone_root = PathBuf::from(clone_root);
    if clone_root.is_absolute() {
        clone_root
    } else {
        repo_root.join(clone_root)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn vera_bm25_indexes_and_searches_small_repo() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("auth.rs"),
            "pub fn authenticate_token(token: &str) -> bool { !token.is_empty() }\n",
        )
        .unwrap();

        let adapter = VeraBm25Adapter::new().unwrap();
        let (index_time, size) = adapter.index(dir.path().to_str().unwrap());
        assert!(index_time >= 0.0);
        assert!(size > 0);

        let results = adapter.search("authenticate token", dir.path().to_str().unwrap(), None);
        assert!(
            results.iter().any(|result| result.file_path == "auth.rs"),
            "expected auth.rs in results, got {results:?}"
        );
    }
}
