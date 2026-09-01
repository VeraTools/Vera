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
    "VERA_RANKING_FILENAME_STEM_BOOST",
    "VERA_RANKING_DEFINITION_BOOST",
    "VERA_RANKING_RECALL_POOL_EXPANSION",
    "VERA_RANKING_MULTIPLICATIVE_PATH_PENALTY",
    "VERA_RANKING_CANDIDATE_POOL_MULTIPLIER",
    "VERA_INDEXING_CHUNK_MAX_CHARS",
];

/// Key used to record the host CPU model in the environment provenance block.
pub const HOST_CPU_MODEL_KEY: &str = "host.cpu_model";

/// Parse the CPU model name from `/proc/cpuinfo` content.
///
/// Returns the first `model name` value (trimmed) if present.
pub fn parse_cpu_model(cpuinfo: &str) -> Option<String> {
    for line in cpuinfo.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("model name")
            && let Some(colon) = trimmed.find(':')
        {
            let value = trimmed[colon + 1..].trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Return the host CPU model from raw cpuinfo content, or a placeholder when unavailable.
#[allow(dead_code)]
pub fn host_cpu_model_from_content(content: &str) -> String {
    parse_cpu_model(content).unwrap_or_else(|| "unknown".to_string())
}

/// Read the host CPU model from a specific path (useful for testing fallback).
pub fn host_cpu_model_from_path(path: &Path) -> String {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|content| parse_cpu_model(&content))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Read the host CPU model from `/proc/cpuinfo` with graceful fallback.
///
/// On Linux the model name is extracted from the `model name` field. On
/// non-Linux or when the file is unavailable an opaque placeholder is
/// returned without panicking. No new dependency is introduced.
pub fn host_cpu_model() -> String {
    host_cpu_model_from_path(Path::new("/proc/cpuinfo"))
}

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
    /// Skip indexing for repos whose on-disk index matches the lane's
    /// embedding identity and whose working tree has not drifted. Reports
    /// then show near-zero index time; use for query-phase reruns on a
    /// pinned corpus, never for cold-index measurements.
    #[serde(default)]
    pub reuse_index: Option<bool>,
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
        if let Some(reuse_index) = self.spec.reuse_index {
            config.insert("lane.reuse_index".to_string(), reuse_index.to_string());
        }
        if provenance.rerank {
            // Explicit reranker provenance (replaces run-note fallback)
            let retrieval = vera_core::config::VeraConfig::default().retrieval;
            config.insert(
                "retrieval.rerank_candidates".to_string(),
                retrieval.rerank_candidates.to_string(),
            );
            config.insert(
                "retrieval.reranker_max_doc_chars".to_string(),
                retrieval.reranker_max_doc_chars.to_string(),
            );
            config.insert(
                "retrieval.max_rerank_batch".to_string(),
                retrieval.max_rerank_batch.to_string(),
            );
            // Bare keys per the milestone spec shorthand
            config.insert(
                "rerank_candidates".to_string(),
                retrieval.rerank_candidates.to_string(),
            );
            config.insert(
                "reranker_max_doc_chars".to_string(),
                retrieval.reranker_max_doc_chars.to_string(),
            );
            config.insert(
                "max_rerank_batch".to_string(),
                retrieval.max_rerank_batch.to_string(),
            );
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
        reuse_index: None,
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
    // Host CPU model is derived from /proc/cpuinfo, not an env var, but
    // recorded alongside the environment block so hardware changes are
    // detectable from the artifact alone.
    summary.insert(HOST_CPU_MODEL_KEY.to_string(), host_cpu_model());
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

    #[test]
    fn lane_reuse_index_provenance_present_when_set() {
        let lane_true = resolve(LaneSpec {
            reuse_index: Some(true),
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();
        let prov = lane_true.provenance().unwrap();
        let map = lane_true.config_map(&prov);
        assert_eq!(map.get("lane.reuse_index"), Some(&"true".to_string()));

        let lane_false = resolve(LaneSpec {
            reuse_index: Some(false),
            ..preset("vera-cpu").unwrap()
        })
        .unwrap();
        let prov2 = lane_false.provenance().unwrap();
        let map2 = lane_false.config_map(&prov2);
        assert_eq!(map2.get("lane.reuse_index"), Some(&"false".to_string()));
    }

    #[test]
    fn lane_reuse_index_provenance_absent_when_unset() {
        let lane = resolve(preset("vera-cpu").unwrap()).unwrap();
        let prov = lane.provenance().unwrap();
        let map = lane.config_map(&prov);
        assert!(
            !map.contains_key("lane.reuse_index"),
            "absent flag should not create provenance key"
        );
    }

    #[test]
    fn bm25_lane_records_reuse_index_provenance_but_adapter_never_reuses() {
        // Provenance must still record the spec value even though the adapter hard-codes never-reuse.
        let lane = resolve(LaneSpec {
            reuse_index: Some(true),
            ..preset("vera-bm25").unwrap()
        })
        .unwrap();
        let prov = lane.provenance().unwrap();
        let map = lane.config_map(&prov);
        assert_eq!(map.get("lane.reuse_index"), Some(&"true".to_string()));
        // BM25 lane is identified as bm25 (backend None) and should never reuse; the adapter's
        // hard-coded false is verified in vera_adapter tests.
        assert!(lane.is_bm25());
    }

    #[test]
    fn lane_reranker_provenance_emitted_when_rerank_enabled() {
        // Explicit reranker provenance replaces the run-note fallback
        // (VAL-SCREEN-011): version_info.config must contain candidate depth,
        // doc budget, and batch when reranking is enabled.
        let base = preset("vera-bm25").unwrap();
        let rerank_spec = LaneSpec {
            name: "api-rerank".to_string(),
            backend: "api".to_string(),
            rerank: true,
            model_id: Some("test-model".to_string()),
            ..base.clone()
        };
        let lane = resolve(rerank_spec).unwrap();
        let prov = lane.provenance().unwrap();
        assert!(prov.rerank, "api lane with rerank=true must be reranking");
        let map = lane.config_map(&prov);
        for key in [
            "retrieval.rerank_candidates",
            "retrieval.reranker_max_doc_chars",
            "retrieval.max_rerank_batch",
            "rerank_candidates",
            "reranker_max_doc_chars",
            "max_rerank_batch",
        ] {
            assert!(
                map.contains_key(key),
                "reranker key {key} must be present when rerank enabled"
            );
        }
        assert_eq!(map.get("retrieval.rerank_candidates").unwrap(), "50");
        assert_eq!(map.get("retrieval.reranker_max_doc_chars").unwrap(), "4800");
        assert_eq!(map.get("retrieval.max_rerank_batch").unwrap(), "20");
        assert_eq!(map.get("rerank_candidates").unwrap(), "50");
        assert_eq!(map.get("reranker_max_doc_chars").unwrap(), "4800");
        assert_eq!(map.get("max_rerank_batch").unwrap(), "20");

        let no_rerank_spec = LaneSpec {
            name: "api-no-rerank".to_string(),
            backend: "api".to_string(),
            rerank: false,
            model_id: Some("test-model".to_string()),
            ..base
        };
        let lane2 = resolve(no_rerank_spec).unwrap();
        let prov2 = lane2.provenance().unwrap();
        assert!(!prov2.rerank);
        let map2 = lane2.config_map(&prov2);
        for key in [
            "retrieval.rerank_candidates",
            "retrieval.reranker_max_doc_chars",
            "retrieval.max_rerank_batch",
            "rerank_candidates",
            "reranker_max_doc_chars",
            "max_rerank_batch",
        ] {
            assert!(
                !map2.contains_key(key),
                "reranker key {key} must be absent when rerank disabled"
            );
        }
    }

    #[test]
    fn cpuinfo_fixture_parsing_extracts_model_name() {
        let fixture = "\
processor\t: 0
vendor_id\t: AuthenticAMD
cpu family\t: 26
model\t\t: 68
model name\t: AMD Ryzen 7 9800X3D 8-Core Processor
stepping\t: 0
microcode\t: 0xb404038
";
        assert_eq!(
            parse_cpu_model(fixture).as_deref(),
            Some("AMD Ryzen 7 9800X3D 8-Core Processor")
        );
        assert_eq!(
            host_cpu_model_from_content(fixture),
            "AMD Ryzen 7 9800X3D 8-Core Processor"
        );
        // Only the first model name is returned even with multiple processors.
        let multi = format!("{fixture}processor\t: 1\nmodel name\t: Other CPU\n");
        assert_eq!(
            parse_cpu_model(&multi).as_deref(),
            Some("AMD Ryzen 7 9800X3D 8-Core Processor")
        );
    }

    #[test]
    fn cpuinfo_fallback_returns_placeholder_without_panic() {
        // Empty content yields placeholder.
        assert_eq!(parse_cpu_model(""), None);
        assert_eq!(host_cpu_model_from_content(""), "unknown");
        // Missing file yields placeholder without panicking.
        let missing = std::path::Path::new("/nonexistent/proc/cpuinfo/fixture");
        assert_eq!(host_cpu_model_from_path(missing), "unknown");
        // Real host path should not panic; on this Linux host it returns the actual model.
        let real = host_cpu_model();
        assert!(
            !real.is_empty(),
            "host_cpu_model should return non-empty placeholder or real model"
        );
        // Placeholder contract: unknown or empty are both acceptable, but we choose unknown.
        assert_ne!(
            real, "",
            "empty string not expected for fallback contract; use unknown"
        );
    }

    #[test]
    fn environment_block_records_ranking_overrides_with_effective_values() {
        // Guard the three ranking env keys plus the host key via the shared ENV_LOCK.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Remember prior values to restore deterministically.
        let keys = [
            "VERA_RANKING_FILENAME_STEM_BOOST",
            "VERA_RANKING_DEFINITION_BOOST",
            "VERA_RANKING_RECALL_POOL_EXPANSION",
        ];
        let prev: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|k| (k.to_string(), std::env::var_os(k)))
            .collect();

        // Case 1: when set to 0, the environment block records "0".
        for k in keys {
            unsafe { std::env::set_var(k, "0") };
        }
        let lane = resolve(preset("vera-potion").unwrap()).unwrap();
        let env = environment_summary(&lane);
        for k in keys {
            assert_eq!(
                env.get(k).map(String::as_str),
                Some("0"),
                "ranking key {k} should be 0 when set"
            );
        }
        // Host CPU is always present alongside the ranking keys.
        assert!(
            env.contains_key(HOST_CPU_MODEL_KEY),
            "environment must contain {HOST_CPU_MODEL_KEY}"
        );
        assert!(!env[HOST_CPU_MODEL_KEY].is_empty());

        // Case 2: when unset, the block records "<unset>" (existing pattern).
        for k in keys {
            unsafe { std::env::remove_var(k) };
        }
        let env2 = environment_summary(&lane);
        for k in keys {
            assert_eq!(
                env2.get(k).map(String::as_str),
                Some("<unset>"),
                "ranking key {k} should be <unset> when not set"
            );
        }
        assert!(
            env2.contains_key(HOST_CPU_MODEL_KEY),
            "host CPU key must still be present when ranking keys unset"
        );

        // Restore.
        for (k, v) in prev {
            unsafe {
                match v {
                    Some(val) => std::env::set_var(&k, val),
                    None => std::env::remove_var(&k),
                }
            }
        }
    }

    #[test]
    fn environment_block_follows_existing_key_value_pattern() {
        // Verifies the new ranking keys use the exact same "<unset>"/value
        // contract as the existing PROVENANCE_ENV_KEYS, and that the host CPU
        // key follows the dot-notation precedent.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let lane = resolve(preset("vera-bm25").unwrap()).unwrap();
        let env = environment_summary(&lane);
        // Existing key still present and uses "<unset>" when not set.
        assert!(env.contains_key("VERA_BACKEND"));
        // New keys exist.
        for k in [
            "VERA_RANKING_FILENAME_STEM_BOOST",
            "VERA_RANKING_DEFINITION_BOOST",
            "VERA_RANKING_RECALL_POOL_EXPANSION",
            HOST_CPU_MODEL_KEY,
        ] {
            assert!(
                env.contains_key(k),
                "expected provenance key {k} in environment block"
            );
        }
        // Host key uses dot notation consistent with config_map precedent.
        assert!(HOST_CPU_MODEL_KEY.contains('.'));
    }

    #[test]
    fn old_result_json_still_parses_and_no_existing_key_changes() {
        // Legacy JSON without the new host CPU or ranking fields must still deserialize.
        let legacy = r#"{
            "tool_version": "legacy",
            "corpus_version": 1,
            "repo_shas": {},
            "config": {"lane.name": "vera-potion", "lane.backend": "potion"},
            "environment": {"VERA_BACKEND": "potion", "VERA_LOCAL": "1"}
        }"#;
        let version: crate::types::VersionInfo = serde_json::from_str(legacy).unwrap();
        // New field defaults to None (additive, not required).
        assert!(version.host_cpu_model.is_none());
        // Existing keys unchanged.
        assert_eq!(version.tool_version, "legacy");
        assert_eq!(version.metric_contract, "unknown-legacy");
        // Serializing and deserializing again preserves the new optional field.
        let json = serde_json::to_string(&version).unwrap();
        let round: crate::types::VersionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(round.tool_version, "legacy");
    }

    #[test]
    fn new_field_is_additive_host_cpu_present_in_provenance() {
        let lane = resolve(preset("vera-potion").unwrap()).unwrap();
        let env = environment_summary(&lane);
        let cpu = env
            .get(HOST_CPU_MODEL_KEY)
            .expect("host CPU in environment");
        assert!(!cpu.is_empty());
        assert_ne!(cpu, "<unset>");
        // On this host it should match the real CPU model.
        // Gracefully accept "unknown" on non-Linux, but on Linux check substring.
        if cpu != "unknown" {
            assert!(
                cpu.contains("AMD") || cpu.contains("Intel") || cpu.contains("Ryzen"),
                "host CPU model should look like a real CPU string, got: {cpu}"
            );
        }
        // Also verify the top-level VersionInfo field via host_cpu_model().
        let top = host_cpu_model();
        assert!(!top.is_empty());
        assert_eq!(
            cpu, &top,
            "environment and direct host_cpu_model should agree"
        );
    }
}
