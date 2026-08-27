//! Benchmark lane specifications and their runtime configuration.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::types::{BenchmarkTask, LaneProvenance, TaskSetIdentity};
use vera_core::config::{InferenceBackend, OnnxExecutionProvider};
use vera_core::local_models::{
    LOCAL_EMBEDDING_DIM_ENV, LOCAL_EMBEDDING_DIR_ENV, LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV,
    LOCAL_EMBEDDING_MAX_LENGTH_ENV, LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV,
    LOCAL_EMBEDDING_ONNX_FILE_ENV, LOCAL_EMBEDDING_POOLING_ENV, LOCAL_EMBEDDING_QUERY_PREFIX_ENV,
    LOCAL_EMBEDDING_REPO_ENV, LOCAL_EMBEDDING_REVISION_ENV, LOCAL_EMBEDDING_TOKENIZER_FILE_ENV,
    LocalEmbeddingModelConfig, LocalEmbeddingSource, POTION_CODE_REPO, POTION_CODE_REVISION,
};

const LOCAL_MODEL_ENV_KEYS: &[&str] = &[
    LOCAL_EMBEDDING_REPO_ENV,
    LOCAL_EMBEDDING_DIR_ENV,
    LOCAL_EMBEDDING_REVISION_ENV,
    LOCAL_EMBEDDING_ONNX_FILE_ENV,
    LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV,
    LOCAL_EMBEDDING_TOKENIZER_FILE_ENV,
    LOCAL_EMBEDDING_DIM_ENV,
    LOCAL_EMBEDDING_POOLING_ENV,
    LOCAL_EMBEDDING_MAX_LENGTH_ENV,
    LOCAL_EMBEDDING_QUERY_PREFIX_ENV,
    LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV,
];

const PROVENANCE_ENV_KEYS: &[&str] = &[
    "VERA_BACKEND",
    "VERA_LOCAL",
    "EMBEDDING_MODEL_BASE_URL",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_MODEL_API_KEY",
    "EMBEDDING_QUERY_PREFIX",
    "RERANKER_MODEL_BASE_URL",
    "RERANKER_MODEL_ID",
    "RERANKER_MODEL_API_KEY",
    "VERA_MAX_RERANK_BATCH",
];

/// One benchmark lane. The same shape can be supplied in JSON or TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneSpec {
    /// Stable name used in reports and output filenames.
    pub name: String,
    /// `api`, `potion`, `onnx-jina-*`, `custom-onnx`, or the legacy `bm25`.
    pub backend: String,
    /// API model identifier, when using the API backend.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Hugging Face model repository for local ONNX models.
    #[serde(default, alias = "model_repo")]
    pub repo: Option<String>,
    /// Local directory containing the ONNX model and tokenizer.
    #[serde(default, alias = "model_dir")]
    pub dir: Option<PathBuf>,
    /// Hugging Face model revision for local ONNX models. Omitted means `main`.
    #[serde(default)]
    pub revision: Option<String>,
    /// ONNX execution provider for `custom-onnx`; defaults to CPU.
    #[serde(default, alias = "provider")]
    pub execution_provider: Option<String>,
    /// Relative ONNX file inside the repository or directory.
    #[serde(default)]
    pub onnx_file: Option<String>,
    /// Relative ONNX external-data file. Set `no_onnx_data` for models
    /// without an external-data sidecar.
    #[serde(default)]
    pub onnx_data_file: Option<String>,
    #[serde(default)]
    pub no_onnx_data: bool,
    /// Relative tokenizer file inside the repository or directory.
    #[serde(default)]
    pub tokenizer_file: Option<String>,
    /// Token pooling mode: `mean`, `cls`, or `last-token`.
    #[serde(default)]
    pub pooling: Option<String>,
    #[serde(default)]
    pub query_prefix: Option<String>,
    #[serde(default)]
    pub document_prefix: Option<String>,
    #[serde(default)]
    pub dim: Option<usize>,
    #[serde(default)]
    pub max_length: Option<usize>,
    /// Whether to enable Vera's reranker for this lane.
    #[serde(default = "default_rerank")]
    pub rerank: bool,
    /// Optional embedding batch size override (API lanes).
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Optional cap on concurrent embedding requests (API lanes).
    #[serde(default)]
    pub max_concurrent_requests: Option<usize>,
    /// Optional embedding request timeout in seconds (API lanes).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Optional embedding max-retries override (API lanes).
    #[serde(default)]
    pub max_retries: Option<u32>,
    /// Additional environment overrides, useful for API endpoints and keys.
    /// Secret values are redacted in report provenance.
    #[serde(default, alias = "env")]
    pub environment: BTreeMap<String, String>,
}

fn default_rerank() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct LaneFile {
    lanes: Vec<LaneSpec>,
}

/// A resolved lane with the backend understood by `vera-core`.
#[derive(Debug, Clone)]
pub struct ResolvedLane {
    pub spec: LaneSpec,
    pub backend: Option<InferenceBackend>,
}

impl ResolvedLane {
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    pub fn is_bm25(&self) -> bool {
        self.backend.is_none()
    }

    pub fn rerank(&self) -> bool {
        self.spec.rerank
            && matches!(
                self.backend,
                Some(InferenceBackend::Api | InferenceBackend::OnnxJina(_))
            )
    }

    /// Resolve the effective model settings after [`LaneEnvGuard`] is active.
    pub fn provenance(&self) -> Result<LaneProvenance> {
        let backend = self.spec.backend.trim().to_ascii_lowercase();
        if self.is_bm25() {
            return Ok(LaneProvenance {
                name: self.spec.name.clone(),
                backend,
                execution_provider: None,
                model_id: None,
                model_repo: None,
                model_dir: None,
                model_revision: self.spec.revision.clone(),
                onnx_file: None,
                onnx_data_file: None,
                tokenizer_file: None,
                pooling: None,
                query_prefix: None,
                document_prefix: None,
                dim: None,
                max_length: None,
                rerank: false,
            });
        }

        if matches!(self.backend, Some(InferenceBackend::PotionCode)) {
            return Ok(LaneProvenance {
                name: self.spec.name.clone(),
                backend,
                execution_provider: None,
                model_id: Some(POTION_CODE_REPO.to_string()),
                model_repo: Some(POTION_CODE_REPO.to_string()),
                model_dir: None,
                model_revision: Some(POTION_CODE_REVISION.to_string()),
                onnx_file: None,
                onnx_data_file: None,
                tokenizer_file: None,
                pooling: None,
                query_prefix: None,
                document_prefix: None,
                dim: Some(vera_core::local_models::POTION_CODE_DIM),
                max_length: Some(vera_core::local_models::POTION_CODE_MAX_LENGTH),
                rerank: false,
            });
        }

        if matches!(self.backend, Some(InferenceBackend::Api)) {
            return Ok(LaneProvenance {
                name: self.spec.name.clone(),
                backend,
                execution_provider: None,
                model_id: std::env::var("EMBEDDING_MODEL_ID")
                    .ok()
                    .or_else(|| self.spec.model_id.clone()),
                model_repo: self.spec.repo.clone(),
                model_dir: self
                    .spec
                    .dir
                    .as_ref()
                    .map(|path| path.display().to_string()),
                model_revision: self.spec.revision.clone(),
                onnx_file: None,
                onnx_data_file: None,
                tokenizer_file: None,
                pooling: None,
                query_prefix: self
                    .spec
                    .query_prefix
                    .clone()
                    .or_else(|| std::env::var("EMBEDDING_QUERY_PREFIX").ok()),
                document_prefix: None,
                dim: self.spec.dim,
                max_length: self.spec.max_length,
                rerank: self.rerank(),
            });
        }

        let mut model = LocalEmbeddingModelConfig::from_env()
            .context("failed to resolve local embedding model provenance")?;
        if let Some(provider) = self.backend.and_then(InferenceBackend::execution_provider) {
            model.adjust_for_gpu(provider);
        }
        let (model_repo, model_dir, model_revision) = match &model.source {
            LocalEmbeddingSource::HuggingFace { repo } => (
                Some(repo.clone()),
                None,
                Some(model.revision.clone().unwrap_or_else(|| "main".to_string())),
            ),
            LocalEmbeddingSource::Directory { path } => {
                (None, Some(path.display().to_string()), None)
            }
        };
        let onnx_file = model.onnx_file.clone();
        let onnx_data_file = model.onnx_data_file.clone();
        Ok(LaneProvenance {
            name: self.spec.name.clone(),
            backend,
            execution_provider: self
                .backend
                .and_then(InferenceBackend::execution_provider)
                .map(|provider| provider.to_string()),
            model_id: Some(model.display_name()),
            model_repo,
            model_dir,
            model_revision,
            onnx_file: Some(onnx_file),
            onnx_data_file,
            tokenizer_file: Some(model.tokenizer_file),
            pooling: Some(model.pooling.to_string()),
            query_prefix: model.query_prefix,
            document_prefix: model.document_prefix,
            dim: Some(model.embedding_dim),
            max_length: Some(model.max_length),
            rerank: self.rerank(),
        })
    }

    pub fn config_map(&self, provenance: &LaneProvenance) -> BTreeMap<String, String> {
        let mut config = BTreeMap::new();
        config.insert("lane.name".to_string(), provenance.name.clone());
        config.insert("lane.backend".to_string(), provenance.backend.clone());
        config.insert("lane.rerank".to_string(), provenance.rerank.to_string());
        insert_optional(
            &mut config,
            "lane.execution_provider",
            &provenance.execution_provider,
        );
        insert_optional(&mut config, "lane.model_id", &provenance.model_id);
        insert_optional(&mut config, "lane.model_repo", &provenance.model_repo);
        insert_optional(&mut config, "lane.model_dir", &provenance.model_dir);
        insert_optional(
            &mut config,
            "lane.model_revision",
            &provenance.model_revision,
        );
        insert_optional(&mut config, "lane.onnx_file", &provenance.onnx_file);
        insert_optional(
            &mut config,
            "lane.onnx_data_file",
            &provenance.onnx_data_file,
        );
        insert_optional(
            &mut config,
            "lane.tokenizer_file",
            &provenance.tokenizer_file,
        );
        insert_optional(&mut config, "lane.pooling", &provenance.pooling);
        insert_optional(&mut config, "lane.query_prefix", &provenance.query_prefix);
        insert_optional(
            &mut config,
            "lane.document_prefix",
            &provenance.document_prefix,
        );
        if let Some(dim) = provenance.dim {
            config.insert("lane.dim".to_string(), dim.to_string());
        }
        if let Some(max_length) = provenance.max_length {
            config.insert("lane.max_length".to_string(), max_length.to_string());
        }
        if let Some(batch_size) = self.spec.batch_size {
            config.insert("lane.batch_size".to_string(), batch_size.to_string());
        }
        if let Some(max_concurrent) = self.spec.max_concurrent_requests {
            config.insert(
                "lane.max_concurrent_requests".to_string(),
                max_concurrent.to_string(),
            );
        }
        if let Some(timeout_secs) = self.spec.timeout_secs {
            config.insert("lane.timeout_secs".to_string(), timeout_secs.to_string());
        }
        if let Some(max_retries) = self.spec.max_retries {
            config.insert("lane.max_retries".to_string(), max_retries.to_string());
        }
        config
    }
}

fn insert_optional(map: &mut BTreeMap<String, String>, key: &str, value: &Option<String>) {
    if let Some(value) = value {
        map.insert(key.to_string(), value.clone());
    }
}

/// Load lanes from a JSON or TOML file.
pub fn load_file(path: &Path) -> Result<Vec<LaneSpec>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read lane spec {}", path.display()))?;
    let extension = path.extension().and_then(|value| value.to_str());
    let lanes = match extension {
        Some("toml") => {
            toml::from_str::<LaneFile>(&content)
                .with_context(|| format!("failed to parse lane TOML {}", path.display()))?
                .lanes
        }
        _ => match serde_json::from_str::<LaneFile>(&content) {
            Ok(file) => file.lanes,
            Err(wrapper_error) => {
                serde_json::from_str::<Vec<LaneSpec>>(&content).with_context(|| {
                    format!(
                        "failed to parse lane JSON {}: {wrapper_error}",
                        path.display()
                    )
                })?
            }
        },
    };
    if lanes.is_empty() {
        anyhow::bail!("lane spec {} does not contain any lanes", path.display());
    }
    Ok(lanes)
}

/// Resolve a spec into the backend enum and validate unsupported combinations.
pub fn resolve(mut spec: LaneSpec) -> Result<ResolvedLane> {
    if spec.name.trim().is_empty() {
        anyhow::bail!("lane name cannot be empty");
    }
    if spec.repo.is_some() && spec.dir.is_some() {
        anyhow::bail!("lane '{}' cannot set both repo and dir", spec.name);
    }
    if let Some(revision) = spec.revision.as_deref() {
        spec.revision = Some(vera_core::local_models::normalize_model_revision(revision)?);
        if spec.dir.is_some() {
            anyhow::bail!(
                "lane '{}' cannot set revision with a directory source",
                spec.name
            );
        }
    }

    let backend_name = spec.backend.trim().to_ascii_lowercase();
    let backend = match backend_name.as_str() {
        "bm25" => {
            reject_local_fields(&spec, "bm25")?;
            if spec.repo.is_some() || spec.dir.is_some() {
                anyhow::bail!("lane '{}' BM25 backend cannot set repo or dir", spec.name);
            }
            None
        }
        "api" => {
            reject_local_fields(&spec, "api")?;
            if spec.repo.is_some() || spec.dir.is_some() {
                anyhow::bail!("lane '{}' API backend cannot set repo or dir", spec.name);
            }
            Some(InferenceBackend::Api)
        }
        "potion" | "potion-code" | "potion-code-cpu" | "potion-cpu" => {
            reject_local_fields(&spec, "potion")?;
            Some(InferenceBackend::PotionCode)
        }
        "custom-onnx" => {
            if spec.repo.is_none() && spec.dir.is_none() {
                anyhow::bail!(
                    "lane '{}' custom-onnx backend requires repo or dir",
                    spec.name
                );
            }
            let provider = parse_execution_provider(spec.execution_provider.as_deref())?;
            Some(InferenceBackend::OnnxJina(provider))
        }
        value if value.starts_with("onnx-jina-") => {
            let backend = InferenceBackend::from_str(value).map_err(anyhow::Error::msg)?;
            if spec.execution_provider.is_some() {
                anyhow::bail!(
                    "lane '{}' already selects an execution provider in backend '{}'; omit execution_provider",
                    spec.name,
                    spec.backend
                );
            }
            Some(backend)
        }
        other => anyhow::bail!(
            "unknown lane backend '{other}' for '{}'; expected api, potion, onnx-jina-*, custom-onnx, or bm25",
            spec.name
        ),
    };

    Ok(ResolvedLane { spec, backend })
}

fn reject_local_fields(spec: &LaneSpec, backend: &str) -> Result<()> {
    let has_local_model_fields = spec.pooling.is_some()
        || spec.document_prefix.is_some()
        || spec.dim.is_some()
        || spec.max_length.is_some()
        || spec.onnx_file.is_some()
        || spec.onnx_data_file.is_some()
        || spec.no_onnx_data
        || spec.tokenizer_file.is_some()
        || spec.revision.is_some();
    if has_local_model_fields {
        anyhow::bail!(
            "lane '{}' backend '{backend}' only supports model_id and query_prefix; local ONNX fields require custom-onnx or onnx-jina-*",
            spec.name
        );
    }
    Ok(())
}

fn parse_execution_provider(value: Option<&str>) -> Result<OnnxExecutionProvider> {
    let value = value.map(str::trim).map(str::to_ascii_lowercase);
    let value = value.as_deref().unwrap_or("cpu");
    let backend = if value.starts_with("onnx-jina-") {
        value.to_string()
    } else {
        format!("onnx-jina-{value}")
    };
    match InferenceBackend::from_str(&backend).map_err(anyhow::Error::msg)? {
        InferenceBackend::OnnxJina(provider) => Ok(provider),
        _ => unreachable!("onnx-jina backend parser returned a non-ONNX backend"),
    }
}

/// Keep the four historical Vera lanes as named presets.
pub fn preset(name: &str) -> Option<LaneSpec> {
    let (backend, rerank) = match name {
        "vera-bm25" => ("bm25", false),
        "vera-cuda" => ("onnx-jina-cuda", true),
        "vera-cpu" => ("onnx-jina-cpu", true),
        "vera-potion" => ("potion", false),
        _ => return None,
    };
    Some(LaneSpec {
        name: name.to_string(),
        backend: backend.to_string(),
        model_id: None,
        repo: None,
        dir: None,
        revision: None,
        execution_provider: None,
        onnx_file: None,
        onnx_data_file: None,
        no_onnx_data: false,
        tokenizer_file: None,
        pooling: None,
        query_prefix: None,
        document_prefix: None,
        dim: None,
        max_length: None,
        rerank,
        batch_size: None,
        max_concurrent_requests: None,
        timeout_secs: None,
        max_retries: None,
        environment: BTreeMap::new(),
    })
}

/// Resolve named presets or a lane spec file, rejecting duplicate names.
pub fn resolve_specs(specs: Vec<LaneSpec>) -> Result<Vec<ResolvedLane>> {
    let mut names = HashSet::new();
    specs
        .into_iter()
        .map(resolve)
        .collect::<Result<Vec<_>>>()
        .and_then(|lanes| {
            for lane in &lanes {
                if !names.insert(lane.name().to_string()) {
                    anyhow::bail!("duplicate lane name '{}'", lane.name());
                }
            }
            Ok(lanes)
        })
}

/// Apply one lane's model settings to Vera's environment. The guard restores
/// the caller's environment when the lane completes.
pub fn apply_environment(lane: &ResolvedLane) -> LaneEnvGuard {
    let mut keys: Vec<String> = LOCAL_MODEL_ENV_KEYS
        .iter()
        .chain(PROVENANCE_ENV_KEYS.iter())
        .map(|key| (*key).to_string())
        .chain(lane.spec.environment.keys().cloned())
        .collect();
    keys.sort();
    keys.dedup();
    let guard = LaneEnvGuard::new(keys);

    for (key, value) in &lane.spec.environment {
        set_env(key, Some(value));
    }

    let backend_name = lane.spec.backend.trim().to_ascii_lowercase();
    let backend_value = lane.spec.backend.trim().to_string();
    set_env("VERA_BACKEND", Some(&backend_value));
    set_env(
        "VERA_LOCAL",
        Some(if lane.backend.is_some_and(InferenceBackend::is_local) {
            "1"
        } else {
            "0"
        }),
    );

    if lane.backend.is_some_and(InferenceBackend::is_onnx)
        && (backend_name == "custom-onnx" || has_local_model_overrides(lane))
    {
        for key in LOCAL_MODEL_ENV_KEYS {
            set_env(key, None);
        }
        if let Some(repo) = lane.spec.repo.as_deref() {
            set_env(LOCAL_EMBEDDING_REPO_ENV, Some(repo));
        }
        if let Some(dir) = lane.spec.dir.as_ref() {
            set_env(LOCAL_EMBEDDING_DIR_ENV, Some(&dir.display().to_string()));
        }
        set_env(
            LOCAL_EMBEDDING_ONNX_FILE_ENV,
            lane.spec.onnx_file.as_deref(),
        );
        if lane.spec.no_onnx_data {
            set_env(LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV, Some(""));
        } else {
            set_env(
                LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV,
                lane.spec.onnx_data_file.as_deref(),
            );
        }
        set_env(
            LOCAL_EMBEDDING_TOKENIZER_FILE_ENV,
            lane.spec.tokenizer_file.as_deref(),
        );
        set_env(
            LOCAL_EMBEDDING_DIM_ENV,
            lane.spec.dim.map(|value| value.to_string()).as_deref(),
        );
        set_env(LOCAL_EMBEDDING_POOLING_ENV, lane.spec.pooling.as_deref());
        set_env(
            LOCAL_EMBEDDING_MAX_LENGTH_ENV,
            lane.spec
                .max_length
                .map(|value| value.to_string())
                .as_deref(),
        );
        set_env(
            LOCAL_EMBEDDING_QUERY_PREFIX_ENV,
            lane.spec.query_prefix.as_deref(),
        );
        set_env(
            LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV,
            lane.spec.document_prefix.as_deref(),
        );
    } else if lane.backend == Some(InferenceBackend::Api) {
        if let Some(model_id) = lane.spec.model_id.as_deref() {
            set_env("EMBEDDING_MODEL_ID", Some(model_id));
        }
        if lane.spec.query_prefix.is_some() {
            set_env("EMBEDDING_QUERY_PREFIX", lane.spec.query_prefix.as_deref());
        }
    }

    // Set after the override block, which clears every LOCAL_MODEL_ENV_KEYS
    // entry (including this one) before re-applying lane fields.
    if lane.backend.is_some_and(InferenceBackend::is_onnx) {
        set_env(LOCAL_EMBEDDING_REVISION_ENV, lane.spec.revision.as_deref());
    }

    guard
}

fn has_local_model_overrides(lane: &ResolvedLane) -> bool {
    lane.spec.repo.is_some()
        || lane.spec.dir.is_some()
        || lane.spec.onnx_file.is_some()
        || lane.spec.onnx_data_file.is_some()
        || lane.spec.no_onnx_data
        || lane.spec.tokenizer_file.is_some()
        || lane.spec.pooling.is_some()
        || lane.spec.query_prefix.is_some()
        || lane.spec.document_prefix.is_some()
        || lane.spec.dim.is_some()
        || lane.spec.max_length.is_some()
        || lane.spec.revision.is_some()
}

pub struct LaneEnvGuard {
    values: Vec<(String, Option<OsString>)>,
}

impl LaneEnvGuard {
    fn new(keys: Vec<String>) -> Self {
        Self {
            values: keys
                .into_iter()
                .map(|key| {
                    let value = std::env::var_os(&key);
                    (key, value)
                })
                .collect(),
        }
    }
}

impl Drop for LaneEnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.values {
            set_env_os(key, value.as_ref());
        }
    }
}

fn set_env(key: &str, value: Option<&str>) {
    set_env_os(key, value.map(OsString::from).as_ref());
}

fn set_env_os(key: &str, value: Option<&OsString>) {
    unsafe {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

/// Summarize relevant runtime environment without emitting credentials.
pub fn environment_summary(lane: &ResolvedLane) -> BTreeMap<String, String> {
    let keys: HashSet<&str> = PROVENANCE_ENV_KEYS
        .iter()
        .copied()
        .chain(lane.spec.environment.keys().map(String::as_str))
        .chain(LOCAL_MODEL_ENV_KEYS.iter().copied())
        .collect();
    environment_summary_for_keys(keys)
}

/// Summarize the evaluator's relevant process environment for non-model lanes.
pub fn process_environment_summary() -> BTreeMap<String, String> {
    environment_summary_for_keys(
        PROVENANCE_ENV_KEYS
            .iter()
            .copied()
            .chain(LOCAL_MODEL_ENV_KEYS.iter().copied())
            .collect(),
    )
}

fn environment_summary_for_keys(keys: HashSet<&str>) -> BTreeMap<String, String> {
    let mut summary = BTreeMap::new();
    for key in keys {
        let value = std::env::var(key).unwrap_or_else(|_| "<unset>".to_string());
        let value = if key.contains("KEY") || key.contains("TOKEN") || key.contains("SECRET") {
            if value == "<unset>" {
                value
            } else {
                "<redacted>".to_string()
            }
        } else {
            value
        };
        summary.insert(key.to_string(), value);
    }
    summary
}

/// Hash the sorted task IDs, not their ground truth, so slicing never changes
/// the benchmark definition.
pub fn task_set_identity(tasks: &[BenchmarkTask]) -> TaskSetIdentity {
    let mut ids: Vec<&str> = tasks.iter().map(|task| task.id.as_str()).collect();
    ids.sort_unstable();
    let mut hasher = Sha256::new();
    for id in &ids {
        hasher.update(id.as_bytes());
        hasher.update(b"\n");
    }
    let digest = hasher.finalize();
    TaskSetIdentity {
        count: ids.len(),
        task_ids_sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct RevisionEnvGuard {
        previous: Option<OsString>,
        _lock: MutexGuard<'static, ()>,
    }

    impl RevisionEnvGuard {
        fn set(value: Option<&str>) -> Self {
            let lock = ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var_os(LOCAL_EMBEDDING_REVISION_ENV);
            unsafe {
                match value {
                    Some(value) => std::env::set_var(LOCAL_EMBEDDING_REVISION_ENV, value),
                    None => std::env::remove_var(LOCAL_EMBEDDING_REVISION_ENV),
                }
            }
            Self {
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for RevisionEnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(LOCAL_EMBEDDING_REVISION_ENV, value),
                    None => std::env::remove_var(LOCAL_EMBEDDING_REVISION_ENV),
                }
            }
        }
    }

    #[test]
    fn presets_keep_legacy_lane_names() {
        for name in ["vera-bm25", "vera-cuda", "vera-cpu", "vera-potion"] {
            let lane = resolve(preset(name).unwrap()).unwrap();
            assert_eq!(lane.name(), name);
        }
    }

    #[test]
    fn custom_onnx_defaults_to_cpu() {
        let lane = resolve(LaneSpec {
            name: "candidate".to_string(),
            backend: "custom-onnx".to_string(),
            repo: Some("org/model".to_string()),
            rerank: false,
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();
        assert_eq!(
            lane.backend,
            Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu))
        );
    }

    #[test]
    fn revision_is_rejected_for_bm25_lanes() {
        let lane = LaneSpec {
            name: "bm25-pinned".to_string(),
            backend: "bm25".to_string(),
            revision: Some("abc123".to_string()),
            ..preset("vera-bm25").unwrap()
        };

        assert!(resolve(lane).is_err());
    }

    #[test]
    fn custom_onnx_rejects_directory_revision_combinations() {
        let lane = LaneSpec {
            name: "directory-pinned".to_string(),
            backend: "custom-onnx".to_string(),
            dir: Some(std::path::PathBuf::from("/tmp/model")),
            revision: Some("abc123".to_string()),
            ..preset("vera-cpu").unwrap()
        };

        assert!(resolve(lane).is_err());
    }

    #[test]
    fn custom_onnx_revision_is_normalized_during_resolution() {
        let lane = resolve(LaneSpec {
            name: "repo-pinned".to_string(),
            backend: "custom-onnx".to_string(),
            repo: Some("org/model".to_string()),
            revision: Some(" abc123 ".to_string()),
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();

        assert_eq!(lane.spec.revision.as_deref(), Some("abc123"));
    }

    #[test]
    fn apply_environment_sets_and_restores_revision_for_custom_and_preset_onnx() {
        let _env = RevisionEnvGuard::set(Some("ambient"));
        let custom = resolve(LaneSpec {
            name: "custom-pinned".to_string(),
            backend: "custom-onnx".to_string(),
            repo: Some("org/model".to_string()),
            revision: Some("abc123".to_string()),
            onnx_file: Some("model.onnx".to_string()),
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();

        {
            let _guard = apply_environment(&custom);
            assert_eq!(
                std::env::var(LOCAL_EMBEDDING_REVISION_ENV),
                Ok("abc123".to_string())
            );
        }
        assert_eq!(
            std::env::var(LOCAL_EMBEDDING_REVISION_ENV),
            Ok("ambient".to_string())
        );

        let preset = resolve(LaneSpec {
            revision: Some("abc123".to_string()),
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();
        {
            let _guard = apply_environment(&preset);
            assert_eq!(
                std::env::var(LOCAL_EMBEDDING_REVISION_ENV),
                Ok("abc123".to_string())
            );
        }
        assert_eq!(
            std::env::var(LOCAL_EMBEDDING_REVISION_ENV),
            Ok("ambient".to_string())
        );
    }

    #[test]
    fn provenance_reports_pinned_and_main_revisions() {
        let _env = RevisionEnvGuard::set(None);
        let pinned = resolve(LaneSpec {
            name: "custom-pinned".to_string(),
            backend: "custom-onnx".to_string(),
            repo: Some("org/model".to_string()),
            revision: Some("abc123".to_string()),
            onnx_file: Some("model.onnx".to_string()),
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();

        {
            let _guard = apply_environment(&pinned);
            let provenance = pinned.provenance().unwrap();
            assert_eq!(provenance.model_revision.as_deref(), Some("abc123"));
        }

        let unpinned = resolve(LaneSpec {
            name: "custom-main".to_string(),
            backend: "custom-onnx".to_string(),
            repo: Some("org/model".to_string()),
            onnx_file: Some("model.onnx".to_string()),
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();
        {
            let _guard = apply_environment(&unpinned);
            let provenance = unpinned.provenance().unwrap();
            assert_eq!(provenance.model_revision.as_deref(), Some("main"));
        }
    }

    #[test]
    fn potion_lane_provenance_reports_the_pinned_revision() {
        let lane = resolve(preset("vera-potion").unwrap()).unwrap();
        let provenance = lane.provenance().unwrap();
        assert_eq!(provenance.model_repo.as_deref(), Some(POTION_CODE_REPO));
        assert_eq!(
            provenance.model_revision.as_deref(),
            Some(POTION_CODE_REVISION)
        );
        assert_eq!(
            provenance.dim,
            Some(vera_core::local_models::POTION_CODE_DIM)
        );
        assert_eq!(
            provenance.max_length,
            Some(vera_core::local_models::POTION_CODE_MAX_LENGTH)
        );
        assert!(!provenance.rerank);
    }

    #[test]
    fn task_identity_ignores_ground_truth_and_order() {
        let mut a = BenchmarkTask {
            id: "b".to_string(),
            query: String::new(),
            category: crate::types::TaskCategory::Intent,
            repo: "repo".to_string(),
            ground_truth: Vec::new(),
            description: String::new(),
        };
        let mut b = a.clone();
        b.id = "a".to_string();
        let first = task_set_identity(&[a.clone(), b.clone()]);
        a.ground_truth.push(crate::types::GroundTruthEntry {
            file_path: "different".to_string(),
            line_start: 1,
            line_end: 1,
            relevance: 1,
        });
        let second = task_set_identity(&[b, a]);
        assert_eq!(first.count, 2);
        assert_eq!(first.task_ids_sha256, second.task_ids_sha256);
    }

    #[test]
    fn parses_json_wrapper_and_toml_shape() {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("lanes.json");
        std::fs::write(
            &json_path,
            r#"{"lanes":[{"name":"cpu","backend":"onnx-jina-cpu","rerank":false}]}"#,
        )
        .unwrap();
        assert_eq!(load_file(&json_path).unwrap().len(), 1);

        let toml_path = dir.path().join("lanes.toml");
        std::fs::write(
            &toml_path,
            "[[lanes]]\nname = \"cpu\"\nbackend = \"onnx-jina-cpu\"\nrerank = false\n",
        )
        .unwrap();
        assert_eq!(load_file(&toml_path).unwrap().len(), 1);
    }
}
