use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;

pub(super) const HUB_URL: &str = "https://huggingface.co";
pub(super) const EMBEDDING_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
pub(super) const EMBEDDING_ONNX_FILE: &str = "onnx/model_quantized.onnx";
pub(super) const EMBEDDING_ONNX_DATA_FILE: &str = "onnx/model_quantized.onnx_data";
/// FP16 model for GPU backends (quantized INT8 ops lack CUDA kernels,
/// causing ORT to silently fall back to CPU).
pub(super) const EMBEDDING_ONNX_GPU_FILE: &str = "onnx/model_fp16.onnx";
pub(super) const EMBEDDING_ONNX_GPU_DATA_FILE: &str = "onnx/model_fp16.onnx_data";
pub(super) const EMBEDDING_TOKENIZER_FILE: &str = "tokenizer.json";
pub(super) const EMBEDDING_DIM: usize = 768;
pub(super) const EMBEDDING_MAX_LENGTH: usize = 512;
pub(super) const ONNX_HEADER_MAX_BYTES: usize = 4 * 1024;
pub(super) const ONNX_HEADER_MAX_FIELDS: usize = 256;

/// jina-embeddings-v5-text-nano-retrieval is asymmetric: `config_sentence_transformers.json`
/// declares `{"query": "Query: ", "document": "Document: "}` and the model card requires both
/// sides for the retrieval variant.
pub(super) const JINA_QUERY_PREFIX: &str = "Query:";
pub(super) const JINA_DOCUMENT_PREFIX: &str = "Document:";

pub(super) const CODERANK_EMBEDDING_REPO: &str = "Zenabius/CodeRankEmbed-onnx";
pub(super) const CODERANK_QUERY_PREFIX: &str = "Represent this query for searching relevant code:";

pub const POTION_CODE_REPO: &str = "minishlab/potion-code-16M";
pub const POTION_CODE_TOKENIZER_FILE: &str = "tokenizer.json";
pub const POTION_CODE_MODEL_FILE: &str = "model.safetensors";
pub const POTION_CODE_CONFIG_FILE: &str = "config.json";
pub const POTION_CODE_DIM: usize = 256;
pub const POTION_CODE_MAX_LENGTH: usize = 512;

pub const LOCAL_EMBEDDING_REPO_ENV: &str = "VERA_LOCAL_EMBEDDING_REPO";
pub const LOCAL_EMBEDDING_DIR_ENV: &str = "VERA_LOCAL_EMBEDDING_DIR";
pub const LOCAL_EMBEDDING_ONNX_FILE_ENV: &str = "VERA_LOCAL_EMBEDDING_ONNX_FILE";
pub const LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV: &str = "VERA_LOCAL_EMBEDDING_ONNX_DATA_FILE";
pub const LOCAL_EMBEDDING_TOKENIZER_FILE_ENV: &str = "VERA_LOCAL_EMBEDDING_TOKENIZER_FILE";
pub const LOCAL_EMBEDDING_DIM_ENV: &str = "VERA_LOCAL_EMBEDDING_DIM";
pub const LOCAL_EMBEDDING_POOLING_ENV: &str = "VERA_LOCAL_EMBEDDING_POOLING";
pub const LOCAL_EMBEDDING_MAX_LENGTH_ENV: &str = "VERA_LOCAL_EMBEDDING_MAX_LENGTH";
pub const LOCAL_EMBEDDING_QUERY_PREFIX_ENV: &str = "VERA_LOCAL_EMBEDDING_QUERY_PREFIX";
pub const LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV: &str = "VERA_LOCAL_EMBEDDING_DOCUMENT_PREFIX";
pub const LEGACY_EMBEDDING_QUERY_PREFIX_ENV: &str = "VERA_EMBEDDING_QUERY_PREFIX";

pub(super) const RERANKER_REPO: &str = "jinaai/jina-reranker-v2-base-multilingual";
/// No prebuilt reranker ONNX export runs on the CoreML GPU: the quantized
/// export contains DynamicQuantizeLinear/MatMulInteger ops the CoreML EP cannot
/// execute, and the fp16 export stores every tensor as float16 which the CoreML
/// EP rejects as an input dtype. The reranker is explicitly pinned to the CPU
/// provider for CoreML via `reranker_execution_provider` because CoreML can
/// accept a fused subgraph and then fail at inference. Since CoreML cannot
/// accelerate the reranker either way, all backends use the quantized INT8
/// export — the fastest CPU path. `vera doctor` surfaces this CPU placement so
/// the all-green probe does not mislead users.
pub const RERANKER_ONNX_FILE: &str = "onnx/model_quantized.onnx";
pub(super) const RERANKER_TOKENIZER_FILE: &str = "tokenizer.json";

/// Execution provider the reranker session must use for a given backend.
///
/// Every backend except CoreML reranks on its own provider. CoreML cannot
/// accelerate the reranker at all (see `RERANKER_ONNX_FILE`), and registering
/// the CoreML EP anyway is not a harmless no-op: ONNX Runtime still assigns a
/// fused subgraph to CoreML, which then fails at inference with "Unable to
/// compute the prediction using a neural network model". Session creation
/// succeeds, so the CPU retry in `LocalReranker::new_with_ep` never fires and
/// the failure reaches the caller. Select CPU up front instead.
pub fn reranker_execution_provider(
    ep: crate::config::OnnxExecutionProvider,
) -> crate::config::OnnxExecutionProvider {
    match ep {
        crate::config::OnnxExecutionProvider::CoreMl => crate::config::OnnxExecutionProvider::Cpu,
        other => other,
    }
}

/// ONNX Runtime version to auto-download. Using 1.24.4 for CUDA 13 support.
/// The `ort` crate (rc.11) uses `load-dynamic` so any ABI-compatible ORT works.
pub(super) const ORT_VERSION: &str = "1.24.4";
pub(super) const DEFAULT_CUDA_MAJOR: u32 = 12;
pub(super) const CUDA_13_ORT_MIN_MAJOR: u32 = 13;
pub(super) const CUDA_RUNTIME_LIBRARY_PREFIXES: [&str; 3] =
    ["libcudart.so.", "libcublas.so.", "libcublasLt.so."];

/// ONNX Runtime 1.24.x dropped macOS x86_64 binaries. 1.23.2 is the last
/// release that ships `onnxruntime-osx-x86_64` archives.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) const ORT_VERSION_MACOS_X86: &str = "1.23.2";

pub(super) static ORT_INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
pub(super) static MODEL_DOWNLOAD_ATTEMPT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocalEmbeddingPooling {
    Mean,
    Cls,
    /// Take the final unpadded token. Required by jina-embeddings-v5, whose
    /// `1_Pooling/config.json` sets `pooling_mode_lasttoken`.
    LastToken,
}

impl fmt::Display for LocalEmbeddingPooling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Cls => write!(f, "cls"),
            Self::LastToken => write!(f, "last-token"),
        }
    }
}

impl std::str::FromStr for LocalEmbeddingPooling {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            "last-token" | "lasttoken" | "last_token" => Ok(Self::LastToken),
            other => Err(format!(
                "invalid pooling mode: {other} (expected `mean`, `cls` or `last-token`)"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "kebab-case")]
pub enum LocalEmbeddingSource {
    HuggingFace { repo: String },
    Directory { path: PathBuf },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalEmbeddingModelConfig {
    pub source: LocalEmbeddingSource,
    pub onnx_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_data_file: Option<String>,
    pub tokenizer_file: String,
    pub embedding_dim: usize,
    pub pooling: LocalEmbeddingPooling,
    #[serde(default = "default_embedding_max_length")]
    pub max_length: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_prefix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LocalEmbeddingAssetPaths {
    pub onnx_path: PathBuf,
    pub onnx_data_path: Option<PathBuf>,
    pub tokenizer_path: PathBuf,
}

impl Default for LocalEmbeddingModelConfig {
    fn default() -> Self {
        Self::jina()
    }
}

impl LocalEmbeddingModelConfig {
    fn preset(
        repo: &str,
        onnx_data_file: Option<&str>,
        pooling: LocalEmbeddingPooling,
        query_prefix: Option<&str>,
        document_prefix: Option<&str>,
    ) -> Self {
        Self {
            source: LocalEmbeddingSource::HuggingFace {
                repo: repo.to_string(),
            },
            onnx_file: EMBEDDING_ONNX_FILE.to_string(),
            onnx_data_file: onnx_data_file.map(str::to_string),
            tokenizer_file: EMBEDDING_TOKENIZER_FILE.to_string(),
            embedding_dim: EMBEDDING_DIM,
            pooling,
            max_length: EMBEDDING_MAX_LENGTH,
            query_prefix: query_prefix.map(str::to_string),
            document_prefix: document_prefix.map(str::to_string),
        }
    }

    /// `jina-embeddings-v5-text-nano-retrieval` pools on the final token, not
    /// the mean: `1_Pooling/config.json` sets `pooling_mode_lasttoken`, and the
    /// ONNX graph carries a matching `lasttoken_squeeze` + normalize path whose
    /// result it exposes as a second `sentence_embedding` output.
    pub fn jina() -> Self {
        Self::preset(
            EMBEDDING_REPO,
            Some(EMBEDDING_ONNX_DATA_FILE),
            LocalEmbeddingPooling::LastToken,
            Some(JINA_QUERY_PREFIX),
            Some(JINA_DOCUMENT_PREFIX),
        )
    }

    pub fn coderankembed() -> Self {
        Self::preset(
            CODERANK_EMBEDDING_REPO,
            None,
            LocalEmbeddingPooling::Cls,
            Some(CODERANK_QUERY_PREFIX),
            None,
        )
    }

    /// The exact model config `vera setup` froze into `config.json` for jina
    /// before the pooling fix.
    ///
    /// Frozen literals, never the `EMBEDDING_*` constants. This describes a
    /// file already sitting on someone's disk, so it must not follow the live
    /// preset: deriving it from the constants means the day one of them moves
    /// — raising `EMBEDDING_MAX_LENGTH` for #67, renaming an ONNX export — the
    /// literal stops matching any real pre-fix config and the migration
    /// silently never fires again, leaving every such install mean-pooled with
    /// no error. `legacy_jina_literal_stays_pinned_when_the_constants_move` is
    /// the tripwire for that.
    fn legacy_jina_before_pooling_fix() -> Self {
        Self {
            source: LocalEmbeddingSource::HuggingFace {
                repo: "jinaai/jina-embeddings-v5-text-nano-retrieval".to_string(),
            },
            onnx_file: "onnx/model_quantized.onnx".to_string(),
            onnx_data_file: Some("onnx/model_quantized.onnx_data".to_string()),
            tokenizer_file: "tokenizer.json".to_string(),
            embedding_dim: 768,
            pooling: LocalEmbeddingPooling::Mean,
            max_length: 512,
            query_prefix: None,
            document_prefix: None,
        }
    }

    /// Repair a stored config that froze jina's old mean-pooling default.
    ///
    /// `vera setup` writes the resolved model config to `config.json` and that
    /// copy wins over the preset, so an install created before this fix would
    /// keep mean-pooling jina forever. Only the exact old preset is upgraded;
    /// a config differing in any field is treated as deliberate and left
    /// alone.
    pub fn repair_stored_defaults(self) -> Self {
        if self == Self::legacy_jina_before_pooling_fix() {
            Self::jina()
        } else {
            self
        }
    }

    pub fn from_huggingface_repo(repo: impl Into<String>) -> Self {
        let source = LocalEmbeddingSource::HuggingFace { repo: repo.into() };
        let mut defaults = Self::defaults_for_source(&source);
        defaults.source = source;
        defaults
    }

    pub fn from_directory(path: PathBuf) -> Self {
        let source = LocalEmbeddingSource::Directory { path };
        let mut defaults = Self::defaults_for_source(&source);
        defaults.source = source;
        defaults
    }

    /// Switch to the FP16 ONNX model when running on a GPU execution provider.
    ///
    /// Quantized INT8 models use operators (QLinearMatMul, MatMulInteger) that
    /// lack CUDA/ROCm/DirectML kernels, so ORT silently falls back to CPU for
    /// those nodes. FP16 runs natively on GPU and is much faster.
    ///
    /// Only applies to the default Jina model; custom user overrides are left
    /// untouched.
    pub fn adjust_for_gpu(&mut self, ep: crate::config::OnnxExecutionProvider) {
        if ep == crate::config::OnnxExecutionProvider::Cpu {
            tracing::debug!("adjust_for_gpu: CPU backend, keeping {}", self.onnx_file);
            return;
        }
        // Only swap if the user hasn't overridden the ONNX file to a
        // non-default value via env vars. Note: the CLI config loader sets
        // this env var from saved config even for default values, so we
        // check the actual value, not just presence.
        if let Some(env_val) = env_override(LOCAL_EMBEDDING_ONNX_FILE_ENV) {
            if env_val != EMBEDDING_ONNX_FILE {
                tracing::debug!(
                    "adjust_for_gpu: user overrode ONNX file via env to {env_val}, skipping swap"
                );
                return;
            }
        }
        if matches!(
            &self.source,
            LocalEmbeddingSource::HuggingFace { repo } if repo == EMBEDDING_REPO
        ) && self.onnx_file == EMBEDDING_ONNX_FILE
        {
            tracing::info!(
                "GPU backend ({ep}): switching from quantized to fp16 model (INT8 ops lack GPU kernels)"
            );
            self.onnx_file = EMBEDDING_ONNX_GPU_FILE.to_string();
            self.onnx_data_file = Some(EMBEDDING_ONNX_GPU_DATA_FILE.to_string());
        } else {
            tracing::debug!(
                "adjust_for_gpu: onnx_file={} is not default quantized, no swap needed",
                self.onnx_file
            );
        }
    }

    pub fn from_env() -> Result<Self> {
        let repo = env_override(LOCAL_EMBEDDING_REPO_ENV);
        let dir = env_override(LOCAL_EMBEDDING_DIR_ENV);

        let source = match (repo, dir) {
            (Some(repo), None) => {
                return Self::apply_env_overrides(Self::from_huggingface_repo(
                    normalize_huggingface_repo(&repo)?,
                ));
            }
            (None, Some(path)) => {
                return Self::apply_env_overrides(Self::from_directory(PathBuf::from(path)));
            }
            (None, None) => Self::default().source,
            (Some(_), Some(_)) => anyhow::bail!(
                "{LOCAL_EMBEDDING_REPO_ENV} and {LOCAL_EMBEDDING_DIR_ENV} cannot both be set"
            ),
        };
        Self::apply_env_overrides(Self::defaults_for_source(&source))
    }

    pub fn display_name(&self) -> String {
        match &self.source {
            LocalEmbeddingSource::HuggingFace { repo } => repo.clone(),
            LocalEmbeddingSource::Directory { path } => path.display().to_string(),
        }
    }

    pub fn model_identity(&self) -> String {
        // Presets keep a readable identity, but pooling has to stay in it.
        // Vectors pooled two different ways are not comparable, so a pooling
        // change must invalidate an existing index rather than silently query
        // mean-pooled rows with last-token vectors.
        if self == &Self::jina() || self == &Self::coderankembed() {
            return format!(
                "{}|pooling={}|qp={}|dp={}",
                self.display_name(),
                self.pooling,
                Self::prefix_identity(self.query_prefix.as_deref()),
                Self::prefix_identity(self.document_prefix.as_deref()),
            );
        }

        let source = match &self.source {
            LocalEmbeddingSource::HuggingFace { repo } => format!("hf:{repo}"),
            LocalEmbeddingSource::Directory { path } => format!("dir:{}", path.display()),
        };
        let onnx_data = self.onnx_data_file.as_deref().unwrap_or("-");
        format!(
            "{source}|onnx={}|onnx_data={onnx_data}|tokenizer={}|pooling={}|dim={}|max_length={}|qp={}|dp={}",
            self.onnx_file,
            self.tokenizer_file,
            self.pooling,
            self.embedding_dim,
            self.max_length,
            Self::prefix_identity(self.query_prefix.as_deref()),
            Self::prefix_identity(self.document_prefix.as_deref()),
        )
    }

    /// The canonical form of a configured prefix: trimmed, and absent once
    /// nothing is left of it.
    ///
    /// Both the text that gets embedded and the identity that guards the index
    /// go through here, so two configs that embed byte-identical text can never
    /// disagree about whether the index is stale.
    fn normalize_prefix(prefix: Option<&str>) -> Option<&str> {
        prefix.map(str::trim).filter(|value| !value.is_empty())
    }

    /// Encode a prefix for `model_identity`.
    ///
    /// Length-delimited, because `|qp=` and `|dp=` are otherwise ordinary text:
    /// a prefix containing one moved the field boundary, so
    /// `qp="a" dp="b|dp=c"` and `qp="a|dp=b" dp="c"` encoded to the same
    /// string. The absent case gets a marker no encoded value can spell, since
    /// a present prefix always starts with its length; `unwrap_or("-")` used to
    /// give an unprefixed config and one prefixed with a literal `-` the same
    /// identity, so the staleness guard let their vectors share a table.
    fn prefix_identity(prefix: Option<&str>) -> String {
        match Self::normalize_prefix(prefix) {
            Some(value) => format!("{}:{value}", value.len()),
            None => "none".to_string(),
        }
    }

    /// Join a configured prefix to a text with exactly one space.
    ///
    /// The prefix is trimmed first, so a trailing space in the configured
    /// value cannot double up. That trim is also why there is no
    /// whitespace-preserving branch: a trimmed prefix can never end in
    /// whitespace by the time it is joined.
    fn apply_prefix(prefix: Option<&str>, text: &str) -> String {
        let Some(prefix) = Self::normalize_prefix(prefix) else {
            return text.to_string();
        };
        format!("{prefix} {text}")
    }

    pub fn query_text(&self, query: &str) -> String {
        Self::apply_prefix(self.query_prefix.as_deref(), query)
    }

    /// Prefix an indexed passage. Mirrors `query_text` and is applied only on
    /// the indexing path, so a query never receives it.
    pub fn document_text(&self, document: &str) -> String {
        Self::apply_prefix(self.document_prefix.as_deref(), document)
    }

    pub fn cached_asset_paths(&self) -> Result<LocalEmbeddingAssetPaths> {
        let base_dir = match &self.source {
            LocalEmbeddingSource::HuggingFace { repo } => {
                vera_home_dir()?.join("models").join(repo)
            }
            LocalEmbeddingSource::Directory { path } => path.clone(),
        };
        Ok(LocalEmbeddingAssetPaths {
            onnx_path: base_dir.join(&self.onnx_file),
            onnx_data_path: self.onnx_data_file.as_ref().map(|path| base_dir.join(path)),
            tokenizer_path: base_dir.join(&self.tokenizer_file),
        })
    }

    fn defaults_for_source(source: &LocalEmbeddingSource) -> Self {
        match source {
            LocalEmbeddingSource::HuggingFace { repo } if repo == CODERANK_EMBEDDING_REPO => {
                Self::coderankembed()
            }
            LocalEmbeddingSource::HuggingFace { repo } if repo == EMBEDDING_REPO => Self::jina(),
            _ => Self::generic_defaults(),
        }
    }

    /// Asset shape for a repo Vera has no preset for: jina's file layout and
    /// dimensions, with mean pooling.
    ///
    /// Held separate from `jina()` so that jina's own pooling can be correct
    /// without changing how every custom repo is pooled. `from_source`
    /// overwrites `source`, so only the non-source fields are inherited.
    fn generic_defaults() -> Self {
        Self::preset(
            EMBEDDING_REPO,
            Some(EMBEDDING_ONNX_DATA_FILE),
            LocalEmbeddingPooling::Mean,
            None,
            None,
        )
    }

    fn apply_env_overrides(defaults: Self) -> Result<Self> {
        let explicit_model_env = model_source_and_onnx_file_are_set();
        Ok(Self {
            source: defaults.source,
            onnx_file: env_override(LOCAL_EMBEDDING_ONNX_FILE_ENV)
                .unwrap_or_else(|| defaults.onnx_file.clone()),
            onnx_data_file: env_optional_override(
                LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV,
                defaults.onnx_data_file.clone(),
                explicit_model_env,
            ),
            tokenizer_file: env_override(LOCAL_EMBEDDING_TOKENIZER_FILE_ENV)
                .unwrap_or_else(|| defaults.tokenizer_file.clone()),
            embedding_dim: parse_env_usize(LOCAL_EMBEDDING_DIM_ENV, defaults.embedding_dim)?,
            pooling: parse_pooling_env(LOCAL_EMBEDDING_POOLING_ENV, defaults.pooling)?,
            max_length: parse_env_usize(LOCAL_EMBEDDING_MAX_LENGTH_ENV, defaults.max_length)?,
            query_prefix: query_prefix_from_env(defaults.query_prefix.clone(), explicit_model_env),
            document_prefix: env_optional_override(
                LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV,
                defaults.document_prefix.clone(),
                explicit_model_env,
            ),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocalModelAssetState {
    Missing,
    Valid,
    Invalid,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalModelAssetStatus {
    pub name: &'static str,
    pub path: PathBuf,
    pub exists: bool,
    pub state: LocalModelAssetState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalModelAssetKind {
    Other,
    Onnx,
}

impl LocalModelAssetStatus {
    pub fn is_valid(&self) -> bool {
        self.state == LocalModelAssetState::Valid
    }

    pub fn is_missing(&self) -> bool {
        self.state == LocalModelAssetState::Missing
    }

    pub fn is_invalid(&self) -> bool {
        self.state == LocalModelAssetState::Invalid
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedLibraryDependencyStatus {
    pub inspected_files: Vec<PathBuf>,
    pub missing_details: Vec<String>,
    pub missing_libraries: Vec<String>,
}

pub(super) fn default_embedding_max_length() -> usize {
    EMBEDDING_MAX_LENGTH
}

pub(super) fn env_override(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_optional_override(
    key: &str,
    default: Option<String>,
    explicit_model_env: bool,
) -> Option<String> {
    let value = std::env::var(key).ok();
    resolve_optional_env_value(value.as_deref(), default, explicit_model_env)
}

/// Resolve a field whose absence is meaningful.
///
/// Unlike `env_override`, a variable that is set but empty is not the same as
/// an unset one: it is the opt-out, and returns `None` without consulting the
/// default. An unset variable falls back to the preset default unless the
/// caller spelled the model out through the environment, in which case nothing
/// is inherited from the preset.
fn resolve_optional_env_value(
    value: Option<&str>,
    default: Option<String>,
    explicit_model_env: bool,
) -> Option<String> {
    match value {
        Some(value) if value.trim().is_empty() => None,
        Some(value) => Some(value.trim().to_string()),
        None if explicit_model_env => None,
        None => default,
    }
}

fn model_source_and_onnx_file_are_set() -> bool {
    (env_override(LOCAL_EMBEDDING_REPO_ENV).is_some()
        || env_override(LOCAL_EMBEDDING_DIR_ENV).is_some())
        && env_override(LOCAL_EMBEDDING_ONNX_FILE_ENV).is_some()
}

pub(super) fn parse_env_usize(key: &str, default: usize) -> Result<usize> {
    match env_override(key) {
        Some(value) => value
            .parse::<usize>()
            .with_context(|| format!("invalid {key}: {value}")),
        None => Ok(default),
    }
}

pub(super) fn parse_pooling_env(
    key: &str,
    default: LocalEmbeddingPooling,
) -> Result<LocalEmbeddingPooling> {
    match env_override(key) {
        Some(value) => value
            .parse::<LocalEmbeddingPooling>()
            .map_err(anyhow::Error::msg),
        None => Ok(default),
    }
}

/// Resolve the query prefix from the canonical variable, then the legacy one.
///
/// The canonical variable is consulted even when it is empty: an empty value is
/// the opt-out, so it has to suppress the legacy variable rather than fall
/// through to it.
pub(super) fn query_prefix_from_env(
    default: Option<String>,
    explicit_model_env: bool,
) -> Option<String> {
    match std::env::var(LOCAL_EMBEDDING_QUERY_PREFIX_ENV) {
        Ok(value) => resolve_optional_env_value(Some(&value), default, explicit_model_env),
        Err(_) => env_optional_override(
            LEGACY_EMBEDDING_QUERY_PREFIX_ENV,
            default,
            explicit_model_env,
        ),
    }
}

pub fn normalize_huggingface_repo(value: &str) -> Result<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        anyhow::bail!("embedding repo cannot be empty");
    }

    if let Some(rest) = trimmed
        .strip_prefix("https://huggingface.co/")
        .or_else(|| trimmed.strip_prefix("http://huggingface.co/"))
    {
        let mut parts = rest.split('/').filter(|part| !part.is_empty());
        let owner = parts
            .next()
            .context("invalid Hugging Face URL: missing owner")?;
        let repo = parts
            .next()
            .context("invalid Hugging Face URL: missing repo")?;
        return Ok(format!("{owner}/{repo}"));
    }

    if trimmed.starts_with("https://") || trimmed.starts_with("http://") {
        anyhow::bail!("unsupported embedding repo URL: {trimmed}");
    }

    Ok(trimmed.to_string())
}

pub(crate) mod assets;
pub(crate) mod cuda;
pub(crate) mod ort;

#[cfg(test)]
mod tests;

pub use assets::{
    configured_local_model_name, ensure_local_embedding_assets, ensure_model_file,
    ensure_potion_code_assets, inspect_local_model_files_for_ep, inspect_potion_code_model_files,
    potion_code_model_dir, potion_code_model_name, prepare_local_models_for_ep,
};
pub use ort::{
    ensure_ort_library_for_ep, ensure_ort_runtime, ensure_provider_dependencies,
    inspect_provider_dependencies, inspect_shared_library_deps, ort_library_path_for_ep,
    refresh_ort_library_for_ep, wrap_ort_error,
};

/// Return Vera's home directory.
///
/// Resolution order:
/// 1. `VERA_HOME` env var (explicit override)
/// 2. `~/.vera` if it already exists (backward compatibility)
/// 3. `$XDG_DATA_HOME/vera` (XDG standard, defaults to `~/.local/share/vera`)
/// 4. `~/.vera` as final fallback
pub fn vera_home_dir() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("VERA_HOME") {
        if !path.trim().is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let home = dirs::home_dir().context("Could not find home directory")?;
    let legacy = home.join(".vera");
    if legacy.exists() {
        return Ok(legacy);
    }

    if let Some(data) = dirs::data_dir() {
        return Ok(data.join("vera"));
    }

    Ok(legacy)
}

#[cfg(test)]
mod finding_tests {
    use super::*;
    use crate::config::OnnxExecutionProvider;

    #[test]
    fn reranker_runs_on_cpu_under_coreml_and_on_its_own_provider_elsewhere() {
        // CoreML cannot execute any prebuilt reranker export. Registering the
        // EP anyway lets ORT fuse a subgraph for it that fails at inference.
        assert_eq!(
            reranker_execution_provider(OnnxExecutionProvider::CoreMl),
            OnnxExecutionProvider::Cpu
        );

        for ep in [
            OnnxExecutionProvider::Cpu,
            OnnxExecutionProvider::Cuda,
            OnnxExecutionProvider::Rocm,
            OnnxExecutionProvider::DirectMl,
            OnnxExecutionProvider::OpenVino,
        ] {
            assert_eq!(reranker_execution_provider(ep), ep);
        }
    }

    #[test]
    fn gpu_adjustment_only_changes_the_default_jina_model() {
        let mut custom_repo = LocalEmbeddingModelConfig::from_huggingface_repo("org/model");
        custom_repo.adjust_for_gpu(OnnxExecutionProvider::Cuda);
        assert_eq!(custom_repo.onnx_file, EMBEDDING_ONNX_FILE);
        assert_eq!(
            custom_repo.onnx_data_file.as_deref(),
            Some(EMBEDDING_ONNX_DATA_FILE)
        );

        let mut jina = LocalEmbeddingModelConfig::jina();
        jina.adjust_for_gpu(OnnxExecutionProvider::Cuda);
        assert_eq!(jina.onnx_file, EMBEDDING_ONNX_GPU_FILE);
        assert_eq!(
            jina.onnx_data_file.as_deref(),
            Some(EMBEDDING_ONNX_GPU_DATA_FILE)
        );
    }

    #[test]
    fn omitted_data_file_is_disabled_for_explicit_model_environment() {
        assert_eq!(
            resolve_optional_env_value(None, Some("default.data".to_string()), true),
            None
        );
        assert_eq!(
            resolve_optional_env_value(None, Some("default.data".to_string()), false).as_deref(),
            Some("default.data")
        );
        assert_eq!(
            resolve_optional_env_value(Some("custom.data"), None, true).as_deref(),
            Some("custom.data")
        );
    }
}
