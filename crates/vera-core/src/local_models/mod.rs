use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;

pub(super) const HUB_URL: &str = "https://huggingface.co";
pub(super) const EMBEDDING_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
pub(super) const EMBEDDING_REVISION: &str = "ac5d898c8d382b17167c33e5c8af644a3519b47d";
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
pub(super) const CODERANK_EMBEDDING_REVISION: &str = "e6a6893986a9aaf09a6aa177f42ebc73bb623cca";
pub(super) const CODERANK_QUERY_PREFIX: &str = "Represent this query for searching relevant code:";

pub const POTION_CODE_REPO: &str = "minishlab/potion-code-16M-v2";
/// Immutable upstream revision the potion assets are pinned to, so a silent
/// upstream update can never swap the weights under an existing index.
pub const POTION_CODE_REVISION: &str = "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b";
pub const POTION_CODE_TOKENIZER_FILE: &str = "tokenizer.json";
pub const POTION_CODE_MODEL_FILE: &str = "model.safetensors";
pub const POTION_CODE_CONFIG_FILE: &str = "config.json";
pub const POTION_CODE_DIM: usize = 256;
pub const POTION_CODE_MAX_LENGTH: usize = 512;

pub const LOCAL_EMBEDDING_REPO_ENV: &str = "VERA_LOCAL_EMBEDDING_REPO";
pub const LOCAL_EMBEDDING_DIR_ENV: &str = "VERA_LOCAL_EMBEDDING_DIR";
pub const LOCAL_EMBEDDING_REVISION_ENV: &str = "VERA_LOCAL_EMBEDDING_REVISION";
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
pub(super) const RERANKER_REVISION: &str = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9";
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
pub const LOCAL_RERANKER_REPO_ENV: &str = "LOCAL_RERANKER_REPO";
pub const LOCAL_RERANKER_REVISION_ENV: &str = "LOCAL_RERANKER_REVISION";
pub const LOCAL_RERANKER_ONNX_FILE_ENV: &str = "LOCAL_RERANKER_ONNX_FILE";
pub const LOCAL_RERANKER_TOKENIZER_FILE_ENV: &str = "LOCAL_RERANKER_TOKENIZER_FILE";

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

/// CodeRankEmbed's CoreML graph has a dynamic output dimension that CoreML
/// accepts during session construction but rejects during inference.
pub fn embedding_execution_provider(
    ep: crate::config::OnnxExecutionProvider,
    model: &LocalEmbeddingModelConfig,
) -> crate::config::OnnxExecutionProvider {
    if ep == crate::config::OnnxExecutionProvider::CoreMl && model.is_coderankembed_preset() {
        crate::config::OnnxExecutionProvider::Cpu
    } else {
        ep
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

pub(super) fn sha256_hex(reader: impl Read) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut reader = reader;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub(super) fn verify_sha256(reader: impl Read, expected: &str, label: &str) -> Result<()> {
    let actual = sha256_hex(reader)?;
    if !actual.eq_ignore_ascii_case(expected) {
        anyhow::bail!("{label} SHA-256 mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

pub(super) fn expected_model_sha256(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<&'static str> {
    let revision = revision?;
    match (repo, revision, file) {
        (EMBEDDING_REPO, EMBEDDING_REVISION, EMBEDDING_ONNX_FILE) => {
            Some("ac93a7417c216e5076e37da2b3599f7ef16513934098a477680440c09f735a08")
        }
        (EMBEDDING_REPO, EMBEDDING_REVISION, EMBEDDING_ONNX_DATA_FILE) => {
            Some("ee7870eb143a7353be08b33f79992a51de3e32b41f684ccd82953a710c2f2f9c")
        }
        (EMBEDDING_REPO, EMBEDDING_REVISION, EMBEDDING_ONNX_GPU_FILE) => {
            Some("028918f8c20e4f858dfcdf41e950f24194862c25422fd2d0299286855b446f06")
        }
        (EMBEDDING_REPO, EMBEDDING_REVISION, EMBEDDING_ONNX_GPU_DATA_FILE) => {
            Some("1564cc224352ff170df0d861bfa4f50f3a8c7f9f88b253420fff286d0ebc3b51")
        }
        (EMBEDDING_REPO, EMBEDDING_REVISION, EMBEDDING_TOKENIZER_FILE) => {
            Some("98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069")
        }
        (CODERANK_EMBEDDING_REPO, CODERANK_EMBEDDING_REVISION, "onnx/model_quantized.onnx") => {
            Some("732d85552abc1c7f3d8d755a7cbb2df1563d8acafb0d42192971ce078409d06c")
        }
        (CODERANK_EMBEDDING_REPO, CODERANK_EMBEDDING_REVISION, "tokenizer.json") => {
            Some("91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854")
        }
        (RERANKER_REPO, RERANKER_REVISION, RERANKER_ONNX_FILE) => {
            Some("c5220cf8fe023f8aa0ed2a3eb787d4451a7f17cf53f6b787e35718dd4b8815c3")
        }
        (RERANKER_REPO, RERANKER_REVISION, RERANKER_TOKENIZER_FILE) => {
            Some("3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20")
        }
        (POTION_CODE_REPO, POTION_CODE_REVISION, POTION_CODE_MODEL_FILE) => {
            Some("75cf7a6c2171b230ad19b1e7d8e0b1aee86da5a02af8e7cacedd9921d227623c")
        }
        (POTION_CODE_REPO, POTION_CODE_REVISION, POTION_CODE_TOKENIZER_FILE) => {
            Some("107bbdcbad4bff1d299b7a4c3a2fb17c52890688b7dd0e4c9deab79d3c4f3d45")
        }
        (POTION_CODE_REPO, POTION_CODE_REVISION, POTION_CODE_CONFIG_FILE) => {
            Some("148e5691a6fcc553437156859701fba017a1ba5d340b170f17e0f3668fb861a7")
        }
        _ => None,
    }
}

pub(super) fn expected_ort_archive_sha256(archive_filename: &str) -> Option<&'static str> {
    match archive_filename {
        "onnxruntime-linux-aarch64-1.24.4.tgz" => {
            Some("866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e")
        }
        "onnxruntime-linux-x64-1.24.4.tgz" => {
            Some("3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747")
        }
        "onnxruntime-linux-x64-gpu-1.24.4.tgz" => {
            Some("c5f804ff5d239b436fa59e9f2fb288a39f7eb9552f6a636c8b71e792e91a8808")
        }
        "onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz" => {
            Some("fdc6eb18317b4eaeda8b3b86595e5da7e853f72bac67ccac9b04ffc20c9f7fe0")
        }
        "onnxruntime-osx-arm64-1.24.4.tgz" => {
            Some("93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989")
        }
        "onnxruntime-osx-x86_64-1.23.2.tgz" => {
            Some("d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6")
        }
        "onnxruntime-win-x64-1.24.4.zip" => {
            Some("d2319fddfb6ea4db99ccc4b60c85c517bcd855721f5daa6a06d40d7cb2ee2357")
        }
        "onnxruntime-win-x64-gpu-1.24.4.zip" => {
            Some("ef3337a0b8184eb8beec310f7c83bd50376b3eefc43aab84ac8e452f6987df0a")
        }
        "onnxruntime-win-x64-gpu_cuda13-1.24.4.zip" => {
            Some("971be8cf984950672934a3173669590a8ece10b44746883420da8066ba836707")
        }
        _ => None,
    }
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
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
        revision: Option<&str>,
        onnx_data_file: Option<&str>,
        pooling: LocalEmbeddingPooling,
        query_prefix: Option<&str>,
        document_prefix: Option<&str>,
    ) -> Self {
        Self {
            source: LocalEmbeddingSource::HuggingFace {
                repo: repo.to_string(),
            },
            revision: revision.map(str::to_string),
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
            Some(EMBEDDING_REVISION),
            Some(EMBEDDING_ONNX_DATA_FILE),
            LocalEmbeddingPooling::LastToken,
            Some(JINA_QUERY_PREFIX),
            Some(JINA_DOCUMENT_PREFIX),
        )
    }

    pub fn coderankembed() -> Self {
        Self::preset(
            CODERANK_EMBEDDING_REPO,
            Some(CODERANK_EMBEDDING_REVISION),
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
            revision: None,
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
        if let Some(env_val) = env_override(LOCAL_EMBEDDING_ONNX_FILE_ENV)
            && env_val != EMBEDDING_ONNX_FILE
        {
            tracing::debug!(
                "adjust_for_gpu: user overrode ONNX file via env to {env_val}, skipping swap"
            );
            return;
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

    fn is_coderankembed_preset(&self) -> bool {
        let mut config = self.clone();
        config.revision = None;
        let mut preset = Self::coderankembed();
        preset.revision = None;
        config == preset
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
        let mut identity_config = self.clone();
        identity_config.revision = None;
        let identity = if identity_config.is_jina_preset()
            || identity_config.is_coderankembed_preset()
        {
            format!(
                "{}|pooling={}|qp={}|dp={}",
                self.display_name(),
                self.pooling,
                Self::prefix_identity(self.query_prefix.as_deref()),
                Self::prefix_identity(self.document_prefix.as_deref()),
            )
        } else {
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
        };

        // Normalize so a hand-edited config agrees with the validated cache
        // path and download URL. An invalid revision fails asset resolution
        // before it can be downloaded, so the trimmed raw value is a safe
        // fallback that keeps such configs distinguishable.
        match self.revision.as_deref().map(str::trim) {
            Some(revision) if !revision.is_empty() && revision != "main" => {
                let revision =
                    normalize_model_revision(revision).unwrap_or_else(|_| revision.to_string());
                format!("{identity}|revision={revision}")
            }
            _ => identity,
        }
    }

    fn is_jina_preset(&self) -> bool {
        let mut config = self.clone();
        config.revision = None;
        let mut preset = Self::jina();
        preset.revision = None;
        config == preset
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
        let revision = normalize_optional_model_revision(self.revision.as_deref())?;
        let base_dir = match &self.source {
            LocalEmbeddingSource::HuggingFace { repo } => {
                model_cache_dir(&vera_home_dir()?, repo, revision.as_deref())?
            }
            LocalEmbeddingSource::Directory { path } => {
                if revision.is_some() {
                    anyhow::bail!(
                        "embedding revision cannot be used with a directory source: {}",
                        path.display()
                    );
                }
                path.clone()
            }
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
            None,
            Some(EMBEDDING_ONNX_DATA_FILE),
            LocalEmbeddingPooling::Mean,
            None,
            None,
        )
    }

    fn apply_env_overrides(defaults: Self) -> Result<Self> {
        let explicit_model_env = model_source_and_onnx_file_are_set();
        let revision = revision_from_env(LOCAL_EMBEDDING_REVISION_ENV, defaults.revision)?;
        if revision.is_some() && matches!(&defaults.source, LocalEmbeddingSource::Directory { .. })
        {
            anyhow::bail!("embedding revision cannot be used with a directory source");
        }
        Ok(Self {
            source: defaults.source,
            revision,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRerankerConfig {
    pub repo: String,
    pub revision: Option<String>,
    pub onnx_file: String,
    pub tokenizer_file: String,
}

#[derive(Debug, Clone)]
pub struct LocalRerankerAssetPaths {
    pub onnx_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

impl Default for LocalRerankerConfig {
    fn default() -> Self {
        Self {
            repo: RERANKER_REPO.to_string(),
            revision: Some(RERANKER_REVISION.to_string()),
            onnx_file: RERANKER_ONNX_FILE.to_string(),
            tokenizer_file: RERANKER_TOKENIZER_FILE.to_string(),
        }
    }
}

impl LocalRerankerConfig {
    pub fn from_env() -> Result<Self> {
        let defaults = Self::default();
        let repo = env_override(LOCAL_RERANKER_REPO_ENV)
            .map(|repo| normalize_huggingface_repo(&repo))
            .transpose()?
            .unwrap_or_else(|| defaults.repo.clone());
        let default_revision = (repo == RERANKER_REPO)
            .then_some(defaults.revision)
            .flatten();
        Ok(Self {
            repo,
            revision: revision_from_env(LOCAL_RERANKER_REVISION_ENV, default_revision)?,
            onnx_file: env_override(LOCAL_RERANKER_ONNX_FILE_ENV).unwrap_or(defaults.onnx_file),
            tokenizer_file: env_override(LOCAL_RERANKER_TOKENIZER_FILE_ENV)
                .unwrap_or(defaults.tokenizer_file),
        })
    }

    pub fn cached_asset_paths(&self) -> Result<LocalRerankerAssetPaths> {
        let revision = normalize_optional_model_revision(self.revision.as_deref())?;
        let model_dir = model_cache_dir(&vera_home_dir()?, &self.repo, revision.as_deref())?;
        Ok(LocalRerankerAssetPaths {
            onnx_path: model_dir.join(&self.onnx_file),
            tokenizer_path: model_dir.join(&self.tokenizer_file),
        })
    }
}

/// Normalize and validate a Hugging Face revision used for local model assets.
///
/// Revisions may contain slash-separated refs such as `refs/pr/123`, but every
/// component must be a normal relative path component so it cannot escape the
/// model cache or alter the Hub URL path. `|` is rejected because
/// `model_identity` uses it as the field delimiter.
pub fn normalize_model_revision(value: &str) -> Result<String> {
    let revision = value.trim();
    if revision.is_empty() {
        anyhow::bail!("embedding revision cannot be empty");
    }
    if revision.starts_with('/')
        || revision.ends_with('/')
        || revision.contains("//")
        || revision.contains('\\')
        || revision.contains('?')
        || revision.contains('#')
        || revision.contains('%')
        || revision.contains('|')
        || revision.chars().any(char::is_control)
    {
        anyhow::bail!("invalid embedding revision: {revision}");
    }

    for component in revision.split('/') {
        if component.is_empty() || component == "." || component == ".." {
            anyhow::bail!("invalid embedding revision: {revision}");
        }
    }

    Ok(revision.to_string())
}

pub(super) fn normalize_optional_model_revision(revision: Option<&str>) -> Result<Option<String>> {
    revision.map(normalize_model_revision).transpose()
}

fn revision_from_env(key: &str, default: Option<String>) -> Result<Option<String>> {
    match std::env::var(key) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => Ok(Some(normalize_model_revision(&value)?)),
        Err(std::env::VarError::NotPresent) => {
            default.as_deref().map(normalize_model_revision).transpose()
        }
        Err(error) => Err(error).with_context(|| format!("failed to read {key}")),
    }
}

pub(super) fn model_cache_dir(
    home_dir: &Path,
    repo: &str,
    revision: Option<&str>,
) -> Result<PathBuf> {
    let base_dir = home_dir.join("models").join(repo);
    let Some(revision) = normalize_optional_model_revision(revision)? else {
        return Ok(base_dir);
    };
    if revision == "main" {
        return Ok(base_dir);
    }

    let mut path = base_dir.join("revisions");
    for component in revision.split('/') {
        path.push(component);
    }
    Ok(path)
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
    configured_local_model_name, ensure_local_embedding_assets, ensure_local_reranker_assets,
    ensure_model_file, ensure_potion_code_assets, inspect_local_model_files_for_ep,
    inspect_potion_code_model_files, potion_code_model_dir, potion_code_model_name,
    prepare_local_models_for_ep,
};
pub use ort::{
    ensure_ort_library_for_ep, ensure_ort_runtime, ensure_provider_dependencies,
    inspect_provider_dependencies, ort_library_path_for_ep, refresh_ort_library_for_ep,
    wrap_ort_error,
};

/// Return Vera's home directory.
///
/// Resolution order:
/// 1. `VERA_HOME` env var (explicit override)
/// 2. `~/.vera` if it holds a real installation (backward compatibility)
/// 3. `$XDG_DATA_HOME/vera` (XDG standard, defaults to `~/.local/share/vera`)
/// 4. `~/.vera` as final fallback
pub fn vera_home_dir() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("VERA_HOME")
        && !path.trim().is_empty()
    {
        return Ok(PathBuf::from(path));
    }

    let home = dirs::home_dir().context("Could not find home directory")?;
    Ok(resolve_vera_home_dir(&home, dirs::data_dir()))
}

/// Files a stray `~/.vera` may contain without counting as an installation.
/// Older releases wrote the update-check cache there unconditionally, which
/// must not shadow a populated XDG data directory.
const LEGACY_HOME_INCIDENTAL_FILES: &[&str] = &["update-check.json"];

fn resolve_vera_home_dir(home: &Path, data_dir: Option<PathBuf>) -> PathBuf {
    let legacy = home.join(".vera");
    if is_legacy_installation(&legacy) {
        return legacy;
    }

    match data_dir {
        Some(data) => data.join("vera"),
        None => legacy,
    }
}

fn is_legacy_installation(legacy: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(legacy) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        !LEGACY_HOME_INCIDENTAL_FILES
            .iter()
            .any(|incidental| name == *incidental)
    })
}

#[cfg(test)]
mod home_dir_tests {
    use super::*;

    #[test]
    fn legacy_dir_with_only_update_cache_does_not_shadow_xdg_dir() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let data = temp.path().join("data");
        std::fs::create_dir_all(home.join(".vera")).unwrap();
        std::fs::write(home.join(".vera").join("update-check.json"), "{}").unwrap();

        assert_eq!(
            resolve_vera_home_dir(&home, Some(data.clone())),
            data.join("vera")
        );
    }

    #[test]
    fn populated_legacy_dir_is_preferred() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let data = temp.path().join("data");
        std::fs::create_dir_all(home.join(".vera")).unwrap();
        std::fs::write(home.join(".vera").join("update-check.json"), "{}").unwrap();
        std::fs::write(home.join(".vera").join("config.json"), "{}").unwrap();

        assert_eq!(resolve_vera_home_dir(&home, Some(data)), home.join(".vera"));
    }

    #[test]
    fn missing_legacy_dir_falls_back_to_xdg_then_legacy() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let data = temp.path().join("data");

        assert_eq!(
            resolve_vera_home_dir(&home, Some(data.clone())),
            data.join("vera")
        );
        assert_eq!(resolve_vera_home_dir(&home, None), home.join(".vera"));
    }
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
    fn coderank_embedding_runs_on_cpu_under_coreml_only() {
        let coderank = LocalEmbeddingModelConfig::coderankembed();
        assert_eq!(
            embedding_execution_provider(OnnxExecutionProvider::CoreMl, &coderank),
            OnnxExecutionProvider::Cpu
        );
        assert_eq!(
            embedding_execution_provider(OnnxExecutionProvider::Cuda, &coderank),
            OnnxExecutionProvider::Cuda
        );

        let jina = LocalEmbeddingModelConfig::jina();
        assert_eq!(
            embedding_execution_provider(OnnxExecutionProvider::CoreMl, &jina),
            OnnxExecutionProvider::CoreMl
        );
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
