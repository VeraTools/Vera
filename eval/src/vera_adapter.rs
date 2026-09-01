use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use tokio::runtime::Runtime;
use vera_core::config::{InferenceBackend, VeraConfig, model_names_match_with_aliases};
use vera_core::embedding::{EmbeddingError, EmbeddingProvider};
use vera_core::indexing::{detect_staleness, index_dir, index_repository};
use vera_core::retrieval::search_service::SearchContext;
use vera_core::storage::metadata::MetadataStore;
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
            false,
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
    reuse_index: bool,
) -> (f64, u64) {
    if reuse_index && index_is_current(config, repo_path, provider, model_name) {
        let size = dir_size(&index_dir(repo_path));
        eprintln!("{label}: reusing current index for {}", repo_path.display());
        return (0.0, size);
    }
    match runtime.block_on(index_repository(repo_path, provider, config, model_name)) {
        Ok(summary) => (summary.elapsed_secs, dir_size(&index_dir(repo_path))),
        Err(err) => panic!("{label} index failed for {}: {err:#}", repo_path.display()),
    }
}

/// True when the on-disk index was built with the same embedding identity
/// (model, document prefix, embedding_dim, indexing config, format version,
/// and completeness marker) and the working tree has not drifted since.
/// Also refuses to reuse while another process holds the index write lock.
/// Any doubt returns false so the caller re-indexes.
fn index_is_current<P: EmbeddingProvider>(
    config: &VeraConfig,
    repo_path: &Path,
    provider: &P,
    model_name: &str,
) -> bool {
    // If another process is currently writing this repo's index, do not reuse
    // a half-written live directory. The writer holds an exclusive flock for
    // the entire build; a non-blocking check here avoids observing partial
    // state. The caller will then re-index and block on the same lock.
    if vera_core::indexing::lock::IndexLock::is_locked_for_index_dir(&index_dir(repo_path)) {
        return false;
    }
    let metadata_path = index_dir(repo_path).join("metadata.db");
    if !metadata_path.exists() {
        return false;
    }
    let Ok(metadata_store) = MetadataStore::open(&metadata_path) else {
        return false;
    };
    if !index_complete_matches(&metadata_store) {
        return false;
    }
    let identity_matches = metadata_store
        .get_index_meta("model_name")
        .unwrap_or(None)
        .is_some_and(|stored| {
            model_names_match_with_aliases(&stored, model_name, &config.embedding.model_aliases)
        });
    let prefix_matches = metadata_store
        .get_index_meta("document_prefix")
        .unwrap_or(None)
        .unwrap_or_default()
        == provider.document_prefix_identity();
    if !identity_matches || !prefix_matches {
        return false;
    }
    if !embedding_dim_matches(&metadata_store, provider) {
        return false;
    }
    if !indexing_config_matches(&metadata_store, config) {
        return false;
    }
    if !index_format_matches(&metadata_store) {
        return false;
    }
    match detect_staleness(repo_path, &config.indexing) {
        Ok(freshness) => !freshness.is_stale(),
        Err(_) => false,
    }
}

fn embedding_dim_matches<P: EmbeddingProvider>(store: &MetadataStore, provider: &P) -> bool {
    let Some(stored) = store.get_index_meta("embedding_dim").unwrap_or(None) else {
        return false;
    };
    let Some(expected) = provider.expected_dim() else {
        // API providers do not report an expected dimension ahead of time
        // (OpenAI-compatible embeddings vary by model). The stored
        // `embedding_dim` was written at index time from the actual
        // embedding vectors, so any parseable stored value is sufficient to
        // allow reuse — the model identity check already gates on the model
        // name/alias. Treat missing expected as compatible for API lanes,
        // but keep the mismatch for local BM25 hash providers where the
        // dimension is part of the index identity.
        return stored.parse::<usize>().is_ok();
    };
    stored == expected.to_string()
}

fn indexing_config_matches(store: &MetadataStore, config: &VeraConfig) -> bool {
    // Content-affecting indexing keys: max_chunk_lines, max_chunk_bytes,
    // max_file_size_bytes, and embedding max_length. Any mismatch forces a
    // full re-index because chunk boundaries or file inclusion change.
    // Throughput-only keys (batch_size, max_concurrent_requests,
    // timeout_secs, max_retries, max_in_flight_inputs) are intentionally
    // NOT part of the identity and are allowed to differ without
    // invalidating a reusable index.
    let Some(encoded) = store
        .get_index_meta(vera_core::indexing::freshness::INDEXING_CONFIG_KEY)
        .unwrap_or(None)
    else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&encoded) else {
        return false;
    };
    if value.get("max_chunk_lines").and_then(|v| v.as_u64())
        != Some(config.indexing.max_chunk_lines as u64)
    {
        return false;
    }
    if value.get("max_chunk_bytes").and_then(|v| v.as_u64())
        != Some(config.indexing.max_chunk_bytes as u64)
    {
        return false;
    }
    // Chunk char budget is content-affecting (~750-char hypothesis). Stored
    // under "chunk_max_chars" (new) with alias "max_chunk_chars" for compat.
    // Old indexes lack the key; treat missing as 0 (DEFAULT OFF) so
    // byte-identical default chunking still reuses old indexes.
    let stored_chunk_chars = value
        .get("chunk_max_chars")
        .or_else(|| value.get("max_chunk_chars"))
        .or_else(|| value.get("max_chunk_characters"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if stored_chunk_chars != config.indexing.chunk_max_chars_effective() as u64 {
        return false;
    }
    if value.get("max_file_size_bytes").and_then(|v| v.as_u64())
        != Some(config.indexing.max_file_size_bytes)
    {
        return false;
    }
    // Embedding max_length is content-affecting but historically not stored
    // in the indexing_config JSON (it lives in the local model config).
    // If a future pipeline stores it under "max_length", verify it here.
    // Missing key is considered compatible for backward compatibility.
    if let Some(stored_max_length) = value.get("max_length").and_then(|v| v.as_u64()) {
        // Effective max_length comes from the local model env; fall back to
        // default local model length when not overridden.
        if let Some(current) = std::env::var("VERA_LOCAL_EMBEDDING_MAX_LENGTH")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            && stored_max_length != current
        {
            return false;
        }
    }
    true
}

fn index_format_matches(store: &MetadataStore) -> bool {
    let Some(stored) = store
        .get_index_meta(vera_core::indexing::freshness::INDEX_FORMAT_VERSION_KEY)
        .unwrap_or(None)
    else {
        return false;
    };
    stored == vera_core::indexing::freshness::INDEX_FORMAT_VERSION
}

fn index_complete_matches(store: &MetadataStore) -> bool {
    let Some(stored) = store
        .get_index_meta(vera_core::indexing::freshness::INDEX_COMPLETE_KEY)
        .unwrap_or(None)
    else {
        return false;
    };
    stored == vera_core::indexing::freshness::INDEX_COMPLETE_VALUE
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
    reuse_index: bool,
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
            reuse_index: lane.reuse_index.unwrap_or(false),
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
            self.reuse_index,
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
    use std::path::Path;

    use super::*;
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    use vera_core::config::VeraConfig;
    use vera_core::indexing::freshness::{
        INDEX_COMPLETE_KEY, INDEX_COMPLETE_VALUE, INDEX_FORMAT_VERSION, INDEX_FORMAT_VERSION_KEY,
        INDEXING_CONFIG_KEY,
    };
    use vera_core::indexing::{index_dir, index_repository};
    use vera_core::storage::metadata::MetadataStore;

    fn test_runtime() -> Runtime {
        Runtime::new().unwrap()
    }

    fn write_repo_file(dir: &Path, relative: &str, content: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn index_once(
        runtime: &Runtime,
        dir: &Path,
        config: &VeraConfig,
        provider: &HashEmbeddingProvider,
        model_name: &str,
    ) {
        runtime
            .block_on(index_repository(dir, provider, config, model_name))
            .unwrap();
    }

    fn metadata_store(dir: &Path) -> MetadataStore {
        let path = index_dir(dir).join("metadata.db");
        MetadataStore::open(&path).unwrap()
    }

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

    #[test]
    fn reuse_current_index_reports_zero_time() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(
            dir.path(),
            "auth.rs",
            "pub fn authenticate_token(token: &str) -> bool { !token.is_empty() }\n",
        );
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-reuse-current";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        let size_before = dir_size(&index_dir(dir.path()));
        assert!(size_before > 0);
        // Second call with reuse_index=true should reuse
        let (elapsed, size) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert_eq!(elapsed, 0.0, "reuse should report 0.0 index time");
        assert!(size > 0, "reuse must still report nonzero size");
        assert_eq!(size, size_before, "reuse size should equal first run size");
    }

    #[test]
    fn reuse_missing_metadata_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-missing-meta";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        // Remove metadata.db
        let meta_path = index_dir(dir.path()).join("metadata.db");
        fs::remove_file(&meta_path).unwrap();
        let is_current = index_is_current(&config, dir.path(), &provider, model_name);
        assert!(!is_current, "missing metadata.db should not be current");
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0, "missing metadata should force re-index");
    }

    #[test]
    fn reuse_corrupt_metadata_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-corrupt";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        let meta_path = index_dir(dir.path()).join("metadata.db");
        fs::write(&meta_path, b"corrupt garbage not sqlite").unwrap();
        let is_current = index_is_current(&config, dir.path(), &provider, model_name);
        assert!(
            !is_current,
            "corrupt metadata should not be current and must not panic"
        );
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0, "corrupt metadata should force re-index");
    }

    #[test]
    fn reuse_missing_model_meta_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-missing-key";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        let store = metadata_store(dir.path());
        // Delete model_name key via SQL
        store.get_index_meta("model_name").unwrap();
        // Direct SQL delete
        let conn_path = index_dir(dir.path()).join("metadata.db");
        let conn = rusqlite::Connection::open(&conn_path).unwrap();
        conn.execute("DELETE FROM index_metadata WHERE key = 'model_name'", [])
            .unwrap();
        drop(conn);
        let is_current = index_is_current(&config, dir.path(), &provider, model_name);
        assert!(!is_current, "missing model_name key should force re-index");
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0);
    }

    #[test]
    fn reuse_model_mismatch_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let stored_model = "model-a";
        let lane_model = "model-b";
        index_once(&runtime, dir.path(), &config, &provider, stored_model);
        let is_current = index_is_current(&config, dir.path(), &provider, lane_model);
        assert!(!is_current, "different model name should not be current");
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            lane_model,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0);
        // Verify stored model updated after re-index
        let store = metadata_store(dir.path());
        let new_stored = store.get_index_meta("model_name").unwrap().unwrap();
        assert_eq!(new_stored, lane_model);
    }

    #[test]
    fn reuse_config_alias_reuses() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let mut config = VeraConfig::default();
        config.embedding.model_aliases = vec![vec!["canonical".to_string(), "alias".to_string()]];
        let provider = HashEmbeddingProvider;
        // Store canonical, lane uses alias
        index_once(&runtime, dir.path(), &config, &provider, "canonical");
        let is_current = index_is_current(&config, dir.path(), &provider, "alias");
        assert!(
            is_current,
            "config alias should allow reuse canonical->alias"
        );
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            "alias",
            "test-label",
            true,
        );
        assert_eq!(elapsed, 0.0);

        // Other direction: store alias, lane uses canonical
        let dir2 = tempfile::tempdir().unwrap();
        write_repo_file(dir2.path(), "lib.rs", "pub fn hello() {}\n");
        index_once(&runtime, dir2.path(), &config, &provider, "alias");
        let is_current2 = index_is_current(&config, dir2.path(), &provider, "canonical");
        assert!(is_current2, "alias should match in both directions");
        let (elapsed2, _) = index_with(
            &runtime,
            &config,
            dir2.path(),
            &provider,
            "canonical",
            "test-label",
            true,
        );
        assert_eq!(elapsed2, 0.0);
    }

    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn reuse_env_alias_reuses() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev = std::env::var("VERA_EMBEDDING_MODEL_ALIASES").ok();
        unsafe {
            std::env::set_var(
                "VERA_EMBEDDING_MODEL_ALIASES",
                "canonical,alias;other,other-alias",
            );
        }
        let result = std::panic::catch_unwind(|| {
            let dir = tempfile::tempdir().unwrap();
            write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
            let runtime = test_runtime();
            let config = VeraConfig::default();
            let provider = HashEmbeddingProvider;
            index_once(&runtime, dir.path(), &config, &provider, "canonical");
            let is_current = index_is_current(&config, dir.path(), &provider, "alias");
            assert!(is_current, "env alias should allow reuse canonical->alias");
            let (elapsed, _) = index_with(
                &runtime,
                &config,
                dir.path(),
                &provider,
                "alias",
                "test-label",
                true,
            );
            assert_eq!(elapsed, 0.0);

            let dir2 = tempfile::tempdir().unwrap();
            write_repo_file(dir2.path(), "lib.rs", "pub fn hello() {}\n");
            index_once(&runtime, dir2.path(), &config, &provider, "alias");
            let is_current2 = index_is_current(&config, dir2.path(), &provider, "canonical");
            assert!(is_current2, "env alias should match both directions");
            let (elapsed2, _) = index_with(
                &runtime,
                &config,
                dir2.path(),
                &provider,
                "canonical",
                "test-label",
                true,
            );
            assert_eq!(elapsed2, 0.0);
        });
        // Restore
        unsafe {
            match prev {
                Some(v) => std::env::set_var("VERA_EMBEDDING_MODEL_ALIASES", v),
                None => std::env::remove_var("VERA_EMBEDDING_MODEL_ALIASES"),
            }
        }
        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
    }

    #[test]
    fn reuse_prefix_mismatch_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        // Create a provider with custom prefix identity
        struct PrefixedProvider;
        impl vera_core::embedding::EmbeddingProvider for PrefixedProvider {
            async fn embed_batch(
                &self,
                texts: &[String],
            ) -> Result<Vec<Vec<f32>>, vera_core::embedding::EmbeddingError> {
                Ok(texts.iter().map(|_| vec![0.0; 64]).collect())
            }
            fn expected_dim(&self) -> Option<usize> {
                Some(64)
            }
            fn document_prefix_identity(&self) -> String {
                "Document: ".to_string()
            }
        }
        let prefixed = PrefixedProvider;
        let model_name = "test-model-prefix";
        // Index with empty prefix (Hash provider)
        let hash_provider = HashEmbeddingProvider;
        index_once(&runtime, dir.path(), &config, &hash_provider, model_name);
        // Now check with prefixed provider should mismatch
        let is_current = index_is_current(&config, dir.path(), &prefixed, model_name);
        assert!(!is_current, "prefix mismatch should force re-index");
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &prefixed,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0);

        // Also missing prefix key against provider with nonempty identity should mismatch
        let dir2 = tempfile::tempdir().unwrap();
        write_repo_file(dir2.path(), "lib.rs", "pub fn hello() {}\n");
        index_once(&runtime, dir2.path(), &config, &hash_provider, model_name);
        // Manually delete document_prefix meta
        let conn_path = index_dir(dir2.path()).join("metadata.db");
        let conn = rusqlite::Connection::open(&conn_path).unwrap();
        conn.execute("DELETE FROM index_metadata WHERE key='document_prefix'", [])
            .unwrap();
        drop(conn);
        let is_current2 = index_is_current(&config, dir2.path(), &prefixed, model_name);
        assert!(
            !is_current2,
            "missing prefix vs nonempty identity should mismatch"
        );
    }

    #[test]
    fn reuse_stale_tree_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "src/lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-stale";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        assert!(index_is_current(&config, dir.path(), &provider, model_name));

        // Add file
        write_repo_file(dir.path(), "src/new.rs", "pub fn added() {}\n");
        assert!(
            !index_is_current(&config, dir.path(), &provider, model_name),
            "added file should make stale"
        );
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0);
        // After re-index, should be current again
        assert!(index_is_current(&config, dir.path(), &provider, model_name));

        // Modify file content
        write_repo_file(dir.path(), "src/lib.rs", "pub fn hello_modified() {}\n");
        assert!(!index_is_current(
            &config,
            dir.path(),
            &provider,
            model_name
        ));
        let (elapsed2, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed2 > 0.0);
        assert!(index_is_current(&config, dir.path(), &provider, model_name));

        // Delete file
        fs::remove_file(dir.path().join("src/new.rs")).unwrap();
        assert!(!index_is_current(
            &config,
            dir.path(),
            &provider,
            model_name
        ));
        let (elapsed3, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed3 > 0.0);
    }

    #[test]
    fn bm25_never_reuses_even_with_reuse_true() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "auth.rs", "pub fn hello() {}\n");
        let adapter = VeraBm25Adapter::new().unwrap();
        let (t1, _s1) = adapter.index(dir.path().to_str().unwrap());
        assert!(t1 >= 0.0);
        // Second index via adapter still does full index (hard-coded false)
        let (t2, s2) = adapter.index(dir.path().to_str().unwrap());
        assert!(
            t2 > 0.0 || t2 == 0.0 && s2 > 0,
            "BM25 should not report 0 time even with existing index"
        );
        // More directly: even if we call index_with with reuse true, BM25 path hard-codes false.
        // Simulate by checking that index_is_current would be true but adapter still reindexes.
        // Build index via hash provider to make it current, then BM25 adapter should still reindex.
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let hash_provider = HashEmbeddingProvider;
        let model = "eval-hash-bm25-v1";
        // Ensure index is current for hash provider
        index_once(&runtime, dir.path(), &config, &hash_provider, model);
        assert!(index_is_current(&config, dir.path(), &hash_provider, model));
        // BM25 adapter's index_with is called with false, so it will not reuse. Verify time >0.
        let (t3, _) = adapter.index(dir.path().to_str().unwrap());
        assert!(t3 > 0.0, "BM25 lane must never reuse");
    }

    #[test]
    fn reuse_default_full_reindex() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-default";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        // With reuse false, should always re-index even if current
        let (elapsed_false, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            false,
        );
        assert!(elapsed_false > 0.0, "reuse false should force re-index");
        // Flag absent is equivalent to false; index_is_current true but caller passes false
        assert!(index_is_current(&config, dir.path(), &provider, model_name));
        let (elapsed_absent, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            false,
        );
        assert!(elapsed_absent > 0.0);
    }

    #[test]
    fn reuse_embedding_dim_mismatch_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-dim";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        // Mutate stored embedding_dim
        let store = metadata_store(dir.path());
        store.set_index_meta("embedding_dim", "9999").unwrap();
        assert!(!index_is_current(
            &config,
            dir.path(),
            &provider,
            model_name
        ));
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0);

        // Missing embedding_dim also forces re-index
        let dir2 = tempfile::tempdir().unwrap();
        write_repo_file(dir2.path(), "lib.rs", "pub fn hello() {}\n");
        index_once(&runtime, dir2.path(), &config, &provider, model_name);
        let conn_path = index_dir(dir2.path()).join("metadata.db");
        let conn = rusqlite::Connection::open(&conn_path).unwrap();
        conn.execute("DELETE FROM index_metadata WHERE key='embedding_dim'", [])
            .unwrap();
        drop(conn);
        assert!(!index_is_current(
            &config,
            dir2.path(),
            &provider,
            model_name
        ));
        let (elapsed2, _) = index_with(
            &runtime,
            &config,
            dir2.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed2 > 0.0);
    }

    #[test]
    fn reuse_indexing_config_mismatch_forces_reindex_and_throughput_allowed() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let mut config = VeraConfig::default();
        config.indexing.max_chunk_lines = 200;
        config.indexing.max_chunk_bytes = 24_576;
        config.indexing.max_file_size_bytes = 1_000_000;
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-indexing";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        assert!(index_is_current(&config, dir.path(), &provider, model_name));

        // Mutate stored indexing_config to change max_chunk_lines (content-affecting)
        let store = metadata_store(dir.path());
        let mut stored_cfg: serde_json::Value =
            serde_json::from_str(&store.get_index_meta(INDEXING_CONFIG_KEY).unwrap().unwrap())
                .unwrap();
        stored_cfg["max_chunk_lines"] = serde_json::json!(999);
        store
            .set_index_meta(
                INDEXING_CONFIG_KEY,
                &serde_json::to_string(&stored_cfg).unwrap(),
            )
            .unwrap();
        assert!(
            !index_is_current(&config, dir.path(), &provider, model_name),
            "max_chunk_lines mismatch should force re-index"
        );
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0);

        // Throughput-only difference should still allow reuse: change batch_size in config
        // (not part of indexing_config identity)
        let dir2 = tempfile::tempdir().unwrap();
        write_repo_file(dir2.path(), "lib.rs", "pub fn hello() {}\n");
        index_once(&runtime, dir2.path(), &config, &provider, model_name);
        let mut config_throughput = config.clone();
        config_throughput.embedding.batch_size = 999;
        config_throughput.embedding.max_concurrent_requests = 99;
        config_throughput.embedding.timeout_secs = 999;
        config_throughput.embedding.max_retries = 99;
        assert!(
            index_is_current(&config_throughput, dir2.path(), &provider, model_name),
            "throughput-only changes should still allow reuse"
        );
        let (elapsed2, _) = index_with(
            &runtime,
            &config_throughput,
            dir2.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert_eq!(elapsed2, 0.0, "throughput change must not force re-index");
    }

    #[test]
    fn reuse_format_version_mismatch_forces_reindex() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-format";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        // Stored version should be current
        assert!(index_is_current(&config, dir.path(), &provider, model_name));
        // Mutate to wrong version
        let store = metadata_store(dir.path());
        store
            .set_index_meta(INDEX_FORMAT_VERSION_KEY, "999")
            .unwrap();
        assert!(!index_is_current(
            &config,
            dir.path(),
            &provider,
            model_name
        ));
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed > 0.0);

        // Missing version also forces re-index
        let dir2 = tempfile::tempdir().unwrap();
        write_repo_file(dir2.path(), "lib.rs", "pub fn hello() {}\n");
        index_once(&runtime, dir2.path(), &config, &provider, model_name);
        let conn_path = index_dir(dir2.path()).join("metadata.db");
        let conn = rusqlite::Connection::open(&conn_path).unwrap();
        conn.execute(
            "DELETE FROM index_metadata WHERE key=?1",
            [INDEX_FORMAT_VERSION_KEY],
        )
        .unwrap();
        drop(conn);
        assert!(!index_is_current(
            &config,
            dir2.path(),
            &provider,
            model_name
        ));
        let (elapsed2, _) = index_with(
            &runtime,
            &config,
            dir2.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(elapsed2 > 0.0);
    }

    #[test]
    fn reuse_identity_survives_update() {
        // Verify that vera update preserves identity keys so a subsequent reuse can succeed.
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        write_repo_file(dir.path(), "other.rs", "pub fn other() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-update";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        assert!(index_is_current(&config, dir.path(), &provider, model_name));
        // Run incremental update with a changed file
        write_repo_file(dir.path(), "lib.rs", "pub fn hello_modified() {}\n");
        runtime
            .block_on(vera_core::indexing::update_repository(
                dir.path(),
                &provider,
                &config,
                model_name,
            ))
            .unwrap();
        // After update, identity still matches and tree is fresh, so reuse should succeed
        assert!(index_is_current(&config, dir.path(), &provider, model_name));
        let (elapsed, size) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert_eq!(elapsed, 0.0);
        assert!(size > 0);

        // Now mutate model_name after update to ensure mismatch still forces re-index
        let store = metadata_store(dir.path());
        store
            .set_index_meta("model_name", "different-model")
            .unwrap();
        assert!(!index_is_current(
            &config,
            dir.path(),
            &provider,
            model_name
        ));
    }

    #[test]
    fn reuse_incomplete_marker_forces_reindex() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-complete-marker";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        // Marker should be present after a successful index
        let store = metadata_store(dir.path());
        assert_eq!(
            store.get_index_meta(INDEX_COMPLETE_KEY).unwrap().unwrap(),
            INDEX_COMPLETE_VALUE
        );
        assert!(index_is_current(&config, dir.path(), &provider, model_name));

        // Simulate interrupted write: delete the completeness marker
        let conn_path = index_dir(dir.path()).join("metadata.db");
        let conn = rusqlite::Connection::open(&conn_path).unwrap();
        conn.execute(
            "DELETE FROM index_metadata WHERE key=?1",
            [INDEX_COMPLETE_KEY],
        )
        .unwrap();
        drop(conn);

        // Reuse must now refuse
        assert!(
            !index_is_current(&config, dir.path(), &provider, model_name),
            "missing completeness marker should force re-index"
        );
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert!(
            elapsed > 0.0,
            "interrupted-write simulation (marker absent) must trigger full re-index"
        );
        // After re-index, marker restored and reuse succeeds
        assert!(index_is_current(&config, dir.path(), &provider, model_name));
        let (elapsed2, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert_eq!(elapsed2, 0.0, "after repair, reuse should succeed");
        let store2 = metadata_store(dir.path());
        assert_eq!(
            store2.get_index_meta(INDEX_COMPLETE_KEY).unwrap().unwrap(),
            INDEX_COMPLETE_VALUE
        );
        assert_eq!(
            store2
                .get_index_meta(INDEX_FORMAT_VERSION_KEY)
                .unwrap()
                .unwrap(),
            INDEX_FORMAT_VERSION
        );
    }

    #[test]
    fn reuse_while_locked_forces_reindex() {
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "lib.rs", "pub fn hello() {}\n");
        let runtime = test_runtime();
        let config = VeraConfig::default();
        let provider = HashEmbeddingProvider;
        let model_name = "test-model-locked-marker";
        index_once(&runtime, dir.path(), &config, &provider, model_name);
        assert!(index_is_current(&config, dir.path(), &provider, model_name));

        // Hold the per-repo index lock and verify reuse is refused
        let idx_dir = index_dir(dir.path());
        let _lock = vera_core::indexing::lock::IndexLock::acquire_blocking_for_index_dir(&idx_dir)
            .expect("acquire lock for test");
        assert!(
            !index_is_current(&config, dir.path(), &provider, model_name),
            "held write lock should make index not current"
        );
        // is_locked helper must agree
        assert!(
            vera_core::indexing::lock::IndexLock::is_locked_for_index_dir(&idx_dir),
            "is_locked should be true while writer holds lock"
        );
        // While lock held, try_acquire should fail
        let try_lock =
            vera_core::indexing::lock::IndexLock::try_acquire_for_index_dir(&idx_dir).unwrap();
        assert!(
            try_lock.is_none(),
            "try_acquire should return None while lock held"
        );
        drop(_lock);
        // After release, reuse should succeed again
        assert!(!vera_core::indexing::lock::IndexLock::is_locked_for_index_dir(&idx_dir));
        assert!(index_is_current(&config, dir.path(), &provider, model_name));
        let (elapsed, _) = index_with(
            &runtime,
            &config,
            dir.path(),
            &provider,
            model_name,
            "test-label",
            true,
        );
        assert_eq!(elapsed, 0.0);
    }

    struct SlowHashProvider;

    impl vera_core::embedding::EmbeddingProvider for SlowHashProvider {
        async fn embed_batch(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f32>>, vera_core::embedding::EmbeddingError> {
            // Simulate a slow embedding service so concurrent builds overlap
            tokio::time::sleep(Duration::from_millis(80)).await;
            Ok(texts
                .iter()
                .map(|text| {
                    let mut hasher = DefaultHasher::new();
                    text.hash(&mut hasher);
                    let h = hasher.finish();
                    // deterministic 64-dim vector from hash
                    let v = (h as f32 % 997.0) / 997.0;
                    vec![v; 64]
                })
                .collect())
        }

        fn expected_dim(&self) -> Option<usize> {
            Some(64)
        }

        fn document_prefix_identity(&self) -> String {
            String::new()
        }
    }

    #[test]
    fn concurrent_same_repo_never_observes_partial_index() {
        // Single cargo process spawns two harness threads (sanctioned exception
        // to the one-indexing-process rule: local-only, no paid endpoints).
        // Barrier makes the two writes overlap deterministically; the lock
        // guarantees the second blocks until the first finishes, so neither
        // observes a half-written live index.
        let dir = tempfile::tempdir().unwrap();
        write_repo_file(dir.path(), "a.rs", "pub fn a() -> i32 { 41 }\n");
        write_repo_file(dir.path(), "b.rs", "pub fn b() -> i32 { 42 }\n");
        write_repo_file(dir.path(), "c.rs", "pub fn c() -> i32 { 43 }\n");

        let config = VeraConfig::default();
        let model_name = "concurrent-race-model";

        // Seed a valid initial index
        let runtime = test_runtime();
        index_once(
            &runtime,
            dir.path(),
            &config,
            &HashEmbeddingProvider,
            model_name,
        );
        assert!(index_is_current(
            &config,
            dir.path(),
            &HashEmbeddingProvider,
            model_name
        ));

        // Mutate one file so both concurrent writers have work to do
        write_repo_file(dir.path(), "c.rs", "pub fn c_modified() -> i32 { 99 }\n");

        let barrier = Arc::new(Barrier::new(2));
        let dir1 = dir.path().to_path_buf();
        let dir2 = dir.path().to_path_buf();
        let cfg1 = config.clone();
        let cfg2 = config.clone();
        let b1 = Arc::clone(&barrier);
        let b2 = Arc::clone(&barrier);

        // Spawn two threads that run the full indexing pipeline concurrently
        // on the same repository. Each uses its own Tokio runtime so they
        // behave like separate harness processes.
        let h1 = std::thread::spawn(move || {
            let rt = test_runtime();
            b1.wait();
            rt.block_on(index_repository(
                &dir1,
                &SlowHashProvider,
                &cfg1,
                model_name,
            ))
            .expect("concurrent index 1 failed");
            // While holding the lock already released after index, the index
            // must be complete. Verify marker and non-empty stores.
            let store = MetadataStore::open(&index_dir(&dir1).join("metadata.db")).unwrap();
            assert_eq!(
                store.get_index_meta(INDEX_COMPLETE_KEY).unwrap().unwrap(),
                INDEX_COMPLETE_VALUE
            );
        });

        let h2 = std::thread::spawn(move || {
            let rt = test_runtime();
            b2.wait();
            rt.block_on(index_repository(
                &dir2,
                &SlowHashProvider,
                &cfg2,
                model_name,
            ))
            .expect("concurrent index 2 failed");
            let store = MetadataStore::open(&index_dir(&dir2).join("metadata.db")).unwrap();
            assert_eq!(
                store.get_index_meta(INDEX_COMPLETE_KEY).unwrap().unwrap(),
                INDEX_COMPLETE_VALUE
            );
        });

        h1.join().expect("thread 1 panicked");
        h2.join().expect("thread 2 panicked");

        // Final index must be fully valid: marker, version, and searchable
        let store = MetadataStore::open(&index_dir(dir.path()).join("metadata.db")).unwrap();
        assert_eq!(
            store.get_index_meta(INDEX_COMPLETE_KEY).unwrap().unwrap(),
            INDEX_COMPLETE_VALUE
        );
        assert_eq!(
            store
                .get_index_meta(INDEX_FORMAT_VERSION_KEY)
                .unwrap()
                .unwrap(),
            INDEX_FORMAT_VERSION
        );
        // Chunk count must be consistent with storage (open succeeds and rows parse)
        let idx_dir = index_dir(dir.path());
        let vector_db = idx_dir.join("vectors.db");
        let bm25_dir = idx_dir.join("bm25");
        assert!(
            vector_db.exists(),
            "vectors.db must exist after concurrent builds"
        );
        assert!(bm25_dir.exists(), "bm25 must exist after concurrent builds");
        // No leftover staging directories
        assert!(
            !idx_dir.with_extension("build").exists(),
            "no .vera.build must remain after concurrent builds"
        );
        assert!(
            !idx_dir.with_extension("old").exists(),
            "no .vera.old must remain after concurrent builds"
        );
        // After both writers finish, reuse should succeed
        let runtime2 = test_runtime();
        assert!(index_is_current(
            &config,
            dir.path(),
            &HashEmbeddingProvider,
            model_name
        ));
        let (elapsed, size) = index_with(
            &runtime2,
            &config,
            dir.path(),
            &HashEmbeddingProvider,
            model_name,
            "test-label",
            true,
        );
        assert_eq!(
            elapsed, 0.0,
            "after concurrent builds, reuse must report 0.0"
        );
        assert!(size > 0);
    }

    #[test]
    fn concurrent_stress_loop_no_partial_observation() {
        // Repeated stress loop (5 iterations) to catch flaky races
        for iteration in 0..5 {
            let dir = tempfile::tempdir().unwrap();
            write_repo_file(
                dir.path(),
                "lib.rs",
                &format!("pub fn f{iteration}() {{}}\n"),
            );
            for i in 0..3 {
                write_repo_file(
                    dir.path(),
                    &format!("file{i}.rs"),
                    &format!("pub fn func{i}() {{}}\n"),
                );
            }
            let config = VeraConfig::default();
            let model = format!("stress-model-{iteration}");
            let runtime = test_runtime();
            index_once(
                &runtime,
                dir.path(),
                &config,
                &HashEmbeddingProvider,
                &model,
            );

            let barrier = Arc::new(Barrier::new(3));
            let mut handles = Vec::new();
            for _ in 0..3 {
                let dir_clone = dir.path().to_path_buf();
                let cfg_clone = config.clone();
                let b_clone = Arc::clone(&barrier);
                let model_clone = model.clone();
                handles.push(std::thread::spawn(move || {
                    let rt = test_runtime();
                    b_clone.wait();
                    rt.block_on(index_repository(
                        &dir_clone,
                        &SlowHashProvider,
                        &cfg_clone,
                        &model_clone,
                    ))
                    .unwrap();
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            // Verify final index integrity each iteration
            let store = MetadataStore::open(&index_dir(dir.path()).join("metadata.db")).unwrap();
            assert_eq!(
                store.get_index_meta(INDEX_COMPLETE_KEY).unwrap().unwrap(),
                INDEX_COMPLETE_VALUE,
                "iteration {iteration}: completeness marker must be present"
            );
            assert_eq!(
                store
                    .get_index_meta(INDEX_FORMAT_VERSION_KEY)
                    .unwrap()
                    .unwrap(),
                INDEX_FORMAT_VERSION
            );
        }
    }
}
