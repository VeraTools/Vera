//! Configuration types and defaults for Vera's pipeline.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Top-level configuration for Vera.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VeraConfig {
    /// Indexing configuration.
    pub indexing: IndexingConfig,
    /// Retrieval configuration.
    pub retrieval: RetrievalConfig,
    /// Embedding configuration.
    pub embedding: EmbeddingConfig,
}

/// Configuration for the indexing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Maximum lines for a single chunk before splitting.
    pub max_chunk_lines: u32,
    /// Default path exclusion patterns (in addition to .gitignore).
    pub default_excludes: Vec<String>,
    /// Maximum file size in bytes to index (skip larger files).
    pub max_file_size_bytes: u64,
    /// Extra exclusion globs from CLI `--exclude` flags.
    #[serde(default)]
    pub extra_excludes: Vec<String>,
    /// Disable .gitignore and .veraignore parsing.
    #[serde(default)]
    pub no_ignore: bool,
    /// Disable smart default exclusions.
    #[serde(default)]
    pub no_default_excludes: bool,
    /// Maximum chunk size in bytes for embedding. Chunks exceeding this are
    /// split at line boundaries. 0 disables byte-based splitting.
    /// Default: 24576 (24KB, ~6K-7K tokens, safe for any embedding model).
    #[serde(default = "default_max_chunk_bytes")]
    pub max_chunk_bytes: usize,
}

/// Read a `usize` config override from an environment variable, falling
/// back to `default` when unset or unparseable.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn default_max_chunk_bytes() -> usize {
    env_usize("VERA_MAX_CHUNK_BYTES", 24_576)
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            max_chunk_lines: 200,
            default_excludes: vec![
                ".git".to_string(),
                ".vera".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "build".to_string(),
                "dist".to_string(),
                "__pycache__".to_string(),
                ".venv".to_string(),
            ],
            max_file_size_bytes: 1_000_000, // 1MB
            extra_excludes: Vec::new(),
            no_ignore: false,
            no_default_excludes: false,
            max_chunk_bytes: default_max_chunk_bytes(),
        }
    }
}

/// Configuration for the retrieval pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of results to return by default.
    pub default_limit: usize,
    /// RRF fusion constant (k in 1/(k + rank)).
    pub rrf_k: f64,
    /// Number of candidates to pass to the reranker.
    pub rerank_candidates: usize,
    /// Whether to enable reranking (requires API credentials).
    pub reranking_enabled: bool,
    /// Maximum documents per reranker API call. Larger candidate sets are
    /// partitioned into batches and scores merged. 0 means no batching.
    #[serde(default = "default_max_rerank_batch")]
    pub max_rerank_batch: usize,
    /// Total character budget for search output. Results are progressively
    /// truncated so the combined output stays within this limit.
    /// 0 means unlimited.
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
}

fn default_max_output_chars() -> usize {
    12_000
}

fn default_max_rerank_batch() -> usize {
    env_usize("VERA_MAX_RERANK_BATCH", 20)
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_limit: 5,
            rrf_k: 60.0,
            rerank_candidates: 50,
            reranking_enabled: true,
            max_rerank_batch: default_max_rerank_batch(),
            max_output_chars: 12_000,
        }
    }
}

/// Configuration for the embedding provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Batch size for embedding API calls.
    pub batch_size: usize,
    /// Maximum number of concurrent embedding API requests.
    pub max_concurrent_requests: usize,
    /// Hard limit on the total number of embedding inputs that may be active
    /// across concurrent API requests. This bounds abandoned backend work when
    /// an indexing client disconnects.
    #[serde(default = "default_max_in_flight_inputs")]
    pub max_in_flight_inputs: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum retries on transient errors.
    pub max_retries: u32,
    /// Maximum stored vector dimensionality.
    ///
    /// If the embedding model produces vectors larger than this, they
    /// are truncated to this dimensionality before storage. Qwen3 models
    /// support Matryoshka-style truncation, so lower dimensions still
    /// yield good retrieval quality while dramatically reducing index size.
    /// Set to 0 to store full-dimensionality vectors.
    pub max_stored_dim: usize,
    /// GPU memory limit in MB for ONNX CUDA sessions.
    /// 0 means no limit (ORT default: use all available VRAM).
    #[serde(default)]
    pub gpu_mem_limit_mb: u64,
    /// When true, forces conservative GPU settings (batch_size=1, low mem limit).
    #[serde(default)]
    pub low_vram: bool,
    /// Equivalent embedding model names.
    ///
    /// OpenAI-compatible providers sometimes expose a deployment alias while the
    /// embedding response, stored index metadata, or another compatible gateway
    /// reports the canonical upstream model name. Each inner list is one
    /// equivalence class. When two model names normalize into the same list, Vera
    /// treats them as index-compatible after the existing dimension check passes.
    /// Only alias models you have verified produce compatible embeddings.
    ///
    /// Can also be supplied with `VERA_EMBEDDING_MODEL_ALIASES`, using
    /// semicolon-separated groups of comma-separated aliases:
    /// `text-embedding-3-large,text-embedding-3-large-2;model-a,model-a-prod`
    #[serde(default)]
    pub model_aliases: Vec<Vec<String>>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        let is_local = is_local_mode();
        Self {
            batch_size: if is_local { 4 } else { 128 },
            max_concurrent_requests: if is_local { 1 } else { 8 },
            max_in_flight_inputs: default_max_in_flight_inputs(),
            timeout_secs: 60,
            max_retries: 3,
            max_stored_dim: 1024,
            gpu_mem_limit_mb: 0,
            low_vram: false,
            model_aliases: Vec::new(),
        }
    }
}

fn default_max_in_flight_inputs() -> usize {
    env_usize("VERA_MAX_IN_FLIGHT_INPUTS", 16).max(1)
}

impl EmbeddingConfig {
    /// Clamp configured batching so the product of batch size and concurrency
    /// never exceeds `max_in_flight_inputs`.
    pub fn bounded_parallelism(&self) -> (usize, usize) {
        let max_in_flight = self.max_in_flight_inputs.max(1);
        let batch_size = self.batch_size.max(1).min(max_in_flight);
        let max_concurrent_requests = self
            .max_concurrent_requests
            .max(1)
            .min((max_in_flight / batch_size).max(1));
        (batch_size, max_concurrent_requests)
    }
}

/// ONNX execution provider for local inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OnnxExecutionProvider {
    Cpu,
    Cuda,
    Rocm,
    DirectMl,
    CoreMl,
    OpenVino,
}

impl fmt::Display for OnnxExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Rocm => write!(f, "rocm"),
            Self::DirectMl => write!(f, "directml"),
            Self::CoreMl => write!(f, "coreml"),
            Self::OpenVino => write!(f, "openvino"),
        }
    }
}

/// Inference backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceBackend {
    /// Use external OpenAI-compatible API for embeddings/reranking.
    Api,
    /// Use local ONNX models with the specified execution provider.
    OnnxJina(OnnxExecutionProvider),
    /// Use the CPU-first Potion Code static embedding model.
    PotionCode,
}

impl InferenceBackend {
    /// True if this backend uses local inference.
    pub fn is_local(self) -> bool {
        matches!(self, Self::OnnxJina(_) | Self::PotionCode)
    }

    /// True if this backend uses local ONNX inference.
    pub fn is_onnx(self) -> bool {
        matches!(self, Self::OnnxJina(_))
    }

    /// Get the execution provider (only for local backends).
    pub fn execution_provider(self) -> Option<OnnxExecutionProvider> {
        match self {
            Self::OnnxJina(ep) => Some(ep),
            Self::Api | Self::PotionCode => None,
        }
    }
}

impl fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Api => write!(f, "api"),
            Self::OnnxJina(ep) => write!(f, "onnx-jina-{ep}"),
            Self::PotionCode => write!(f, "potion-code-cpu"),
        }
    }
}

impl FromStr for InferenceBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "api" => Ok(Self::Api),
            "onnx-jina-cpu" => Ok(Self::OnnxJina(OnnxExecutionProvider::Cpu)),
            "onnx-jina-cuda" => Ok(Self::OnnxJina(OnnxExecutionProvider::Cuda)),
            "onnx-jina-rocm" => Ok(Self::OnnxJina(OnnxExecutionProvider::Rocm)),
            "onnx-jina-directml" => Ok(Self::OnnxJina(OnnxExecutionProvider::DirectMl)),
            "onnx-jina-coreml" => Ok(Self::OnnxJina(OnnxExecutionProvider::CoreMl)),
            "onnx-jina-openvino" => Ok(Self::OnnxJina(OnnxExecutionProvider::OpenVino)),
            "potion-code-cpu" | "potion-code" | "potion-cpu" => Ok(Self::PotionCode),
            other => Err(format!("unknown backend: {other}")),
        }
    }
}

/// Check if the local inference mode is active (legacy env var support).
pub fn is_local_mode() -> bool {
    std::env::var("VERA_LOCAL")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

/// Whether experimental structural graph augmentation is enabled.
///
/// This is intentionally an environment-only ablation switch rather than a
/// persisted retrieval setting. Accepted truthy values are `1`, `true`, and
/// `yes`, case-insensitively.
pub fn graph_augmentation_enabled() -> bool {
    std::env::var("VERA_GRAPH_AUGMENT")
        .map(|value| {
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

fn backend_from_env() -> Option<InferenceBackend> {
    std::env::var("VERA_BACKEND")
        .ok()
        .and_then(|value| InferenceBackend::from_str(&value).ok())
}

impl VeraConfig {
    /// Adjust embedding parameters to match the actual backend.
    ///
    /// Saved configs may have API-mode defaults (batch 128, concurrency 8)
    /// even when the user switches to local mode. CPU inference needs small
    /// batches; GPU can handle larger ones. For GPU backends, this picks a
    /// coarse outer batch ceiling from available VRAM. The local ONNX provider
    /// still shapes the actual micro-batches from sequence length at runtime.
    pub fn adjust_for_backend(&mut self, backend: InferenceBackend) {
        match backend {
            InferenceBackend::PotionCode => {
                self.embedding.batch_size = 1024;
                self.embedding.max_concurrent_requests = 1;
                self.embedding.max_stored_dim = self.embedding.max_stored_dim.min(256);
            }
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu) => {
                self.embedding.batch_size = 4;
                self.embedding.max_concurrent_requests = 1;
            }
            InferenceBackend::OnnxJina(ep) => {
                self.embedding.max_concurrent_requests = 1;

                if self.embedding.low_vram {
                    self.embedding.batch_size = 1;
                    if self.embedding.gpu_mem_limit_mb == 0 {
                        self.embedding.gpu_mem_limit_mb = 1024;
                    }
                    tracing::info!(
                        "low-vram mode: batch_size=1, gpu_mem_limit={}MB",
                        self.embedding.gpu_mem_limit_mb
                    );
                    return;
                }

                let gpu_info = detect_gpu_info(ep);
                if let Some(vram) = gpu_info.vram_free_mb {
                    tracing::info!("detected GPU VRAM: {vram}MB");
                    // Auto-scale batch_size based on VRAM.
                    // Prioritize speed: use large batches when VRAM allows.
                    let auto_batch = if vram < 3072 {
                        4
                    } else if vram < 5120 {
                        16
                    } else if vram < 8192 {
                        32
                    } else if vram < 12288 {
                        64
                    } else {
                        128
                    };
                    // Unified memory is shared with macOS and apps; cap the
                    // CoreML batch so large-RAM Macs don't starve the system.
                    if ep == OnnxExecutionProvider::CoreMl {
                        self.embedding.batch_size = auto_batch.min(64);
                    } else {
                        self.embedding.batch_size = auto_batch;
                    }

                    // Set a conservative memory limit only for low-VRAM GPUs
                    // to prevent ORT from grabbing all VRAM. For >=8GB, no limit.
                    if self.embedding.gpu_mem_limit_mb == 0 && vram < 8192 {
                        // Use 80% of available VRAM.
                        self.embedding.gpu_mem_limit_mb = (vram as f64 * 0.8) as u64;
                        tracing::info!(
                            "auto-set gpu_mem_limit={}MB (80% of {vram}MB)",
                            self.embedding.gpu_mem_limit_mb
                        );
                    }
                } else {
                    // Could not detect VRAM; use conservative defaults.
                    // DirectML/CoreML/OpenVINO lack CLI VRAM detection,
                    // so pick a safe batch size that won't OOM on small GPUs.
                    self.embedding.batch_size = 16;
                }
            }
            InferenceBackend::Api => {}
        }
    }
}

/// GPU information collected from a single detection pass.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Free VRAM in MB, if detectable.
    pub vram_free_mb: Option<u64>,
    /// Device fingerprint string for profile keying.
    pub fingerprint: String,
}

/// Detect GPU information (VRAM and device fingerprint) for the given
/// execution provider. Runs the vendor CLI tool once and extracts both
/// pieces of data, avoiding duplicate subprocess calls.
pub fn detect_gpu_info(ep: OnnxExecutionProvider) -> GpuInfo {
    match ep {
        OnnxExecutionProvider::Cuda => detect_nvidia_gpu_info(),
        OnnxExecutionProvider::Rocm => detect_rocm_gpu_info(),
        OnnxExecutionProvider::CoreMl => detect_apple_silicon_mem_info(),
        _ => GpuInfo {
            vram_free_mb: None,
            fingerprint: host_fingerprint(ep),
        },
    }
}

/// Apple Silicon uses unified memory: the GPU shares system RAM. Report half
/// of total RAM (via `sysctl hw.memsize`) as the available pool so batch
/// auto-scaling works; returns None for VRAM off-macOS or on parse failure.
fn detect_apple_silicon_mem_info() -> GpuInfo {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok();
    let total_mb = output
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map(|bytes| bytes / (1024 * 1024));
    // Half of system RAM is a conservative proxy for what the GPU can use
    // while macOS and other apps stay responsive.
    let vram_free_mb = total_mb.map(|mb| mb / 2);
    let brand = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    let fingerprint = match (brand, total_mb) {
        (Some(brand), Some(mb)) => format!("{brand}|{mb}MB-unified"),
        _ => host_fingerprint(OnnxExecutionProvider::CoreMl),
    };
    GpuInfo {
        vram_free_mb,
        fingerprint,
    }
}

fn detect_nvidia_gpu_info() -> GpuInfo {
    // Single nvidia-smi call that returns free VRAM, device name, total VRAM, and driver.
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok();
    let Some(output) = output.filter(|o| o.status.success()) else {
        return GpuInfo {
            vram_free_mb: None,
            fingerprint: host_fingerprint(OnnxExecutionProvider::Cuda),
        };
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().find_map(|line| {
        let trimmed = line.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    });
    let Some(line) = first_line else {
        return GpuInfo {
            vram_free_mb: None,
            fingerprint: host_fingerprint(OnnxExecutionProvider::Cuda),
        };
    };
    // CSV columns: memory.free, name, memory.total, driver_version
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    let vram_free_mb = parts.first().and_then(|s| s.parse::<u64>().ok());
    // Fingerprint from name, total VRAM, driver (columns 1-3).
    let fingerprint = if parts.len() >= 4 {
        format!("{}|{}|{}", parts[1], parts[2], parts[3])
    } else {
        line.replace(", ", "|").replace(',', "|")
    };
    GpuInfo {
        vram_free_mb,
        fingerprint,
    }
}

fn detect_rocm_gpu_info() -> GpuInfo {
    // rocm-smi: get VRAM info and product name in one call.
    let output = std::process::Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
        .output()
        .ok();
    let Some(output) = output.filter(|o| o.status.success()) else {
        return GpuInfo {
            vram_free_mb: None,
            fingerprint: host_fingerprint(OnnxExecutionProvider::Rocm),
        };
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut vram_free_mb = None;
    let mut fingerprint_line = None;
    for line in stdout.lines().skip(1) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("GPU") {
            continue;
        }
        if fingerprint_line.is_none() {
            fingerprint_line = Some(trimmed.replace(", ", "|").replace(',', "|"));
        }
        if vram_free_mb.is_none() {
            // rocm-smi reports bytes; convert to MB.
            vram_free_mb = line
                .split(',')
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .next()
                .map(|bytes| bytes / (1024 * 1024));
        }
    }
    GpuInfo {
        vram_free_mb,
        fingerprint: fingerprint_line
            .unwrap_or_else(|| host_fingerprint(OnnxExecutionProvider::Rocm)),
    }
}

fn host_fingerprint(ep: OnnxExecutionProvider) -> String {
    let host = std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| {
                    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    (!s.is_empty()).then_some(s)
                })
        })
        .unwrap_or_else(|| "unknown-host".to_string());
    format!(
        "{host}|os={}|arch={}|backend={ep}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

/// Resolve the effective inference backend from a CLI flag or environment.
pub fn resolve_backend(backend: Option<InferenceBackend>) -> InferenceBackend {
    if let Some(b) = backend {
        return b;
    }
    if let Some(b) = backend_from_env() {
        return b;
    }
    // Legacy: VERA_LOCAL=1 maps to onnx-jina-cpu
    if is_local_mode() {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu);
    }
    InferenceBackend::Api
}

/// Check whether two model names refer to the same model, using configured
/// alias groups plus aliases supplied by `VERA_EMBEDDING_MODEL_ALIASES`.
///
/// Model names may differ only by an org/repo prefix (e.g.
/// `"jinaai/jina-embeddings-v5-text-nano-retrieval"` vs
/// `"jina-embeddings-v5-text-nano-retrieval"`). Both names are normalised by
/// stripping everything up to and including the last `/` and then compared
/// case-insensitively.
pub fn model_names_match_with_aliases(a: &str, b: &str, aliases: &[Vec<String>]) -> bool {
    let a = normalize_model_name(a);
    let b = normalize_model_name(b);
    a == b || aliases_match(&a, &b, aliases) || aliases_match_env(&a, &b)
}

fn normalize_model_name(s: &str) -> String {
    s.rsplit('/')
        .next()
        .unwrap_or(s)
        .trim()
        .to_ascii_lowercase()
}

fn aliases_match(a: &str, b: &str, aliases: &[Vec<String>]) -> bool {
    aliases.iter().any(|group| {
        let mut has_a = false;
        let mut has_b = false;
        for alias in group {
            let normalized = normalize_model_name(alias);
            has_a |= normalized == a;
            has_b |= normalized == b;
        }
        has_a && has_b
    })
}

fn aliases_match_env(a: &str, b: &str) -> bool {
    std::env::var("VERA_EMBEDDING_MODEL_ALIASES")
        .ok()
        .map(|value| aliases_match(a, b, &parse_model_alias_groups(&value)))
        .unwrap_or(false)
}

fn parse_model_alias_groups(value: &str) -> Vec<Vec<String>> {
    value
        .split(';')
        .filter_map(|group| {
            let aliases = group
                .split(',')
                .map(str::trim)
                .filter(|alias| !alias.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            (aliases.len() >= 2).then_some(aliases)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::EnvVarGuard;

    #[test]
    fn default_config_is_valid() {
        let config = VeraConfig::default();
        assert!(config.indexing.max_chunk_lines > 0);
        assert!(config.retrieval.default_limit > 0);
        assert!(config.retrieval.rrf_k > 0.0);
        assert!(config.embedding.batch_size > 0);
        assert!(config.embedding.max_in_flight_inputs > 0);
        let (batch_size, concurrency) = config.embedding.bounded_parallelism();
        assert!(
            batch_size * concurrency <= config.embedding.max_in_flight_inputs,
            "default embedding parallelism must respect its in-flight bound"
        );
    }

    #[test]
    fn graph_augmentation_env_accepts_only_truthy_values() {
        for value in ["1", "true", "TRUE", "yes", "YeS"] {
            let _guard = EnvVarGuard::set(&[("VERA_GRAPH_AUGMENT", value)]);
            assert!(
                graph_augmentation_enabled(),
                "{value} should enable the flag"
            );
        }

        for value in ["0", "false", "no", "", "on"] {
            let _guard = EnvVarGuard::set(&[("VERA_GRAPH_AUGMENT", value)]);
            assert!(
                !graph_augmentation_enabled(),
                "{value} should disable the flag"
            );
        }
    }

    #[test]
    fn embedding_parallelism_clamps_batch_and_concurrency() {
        let config = EmbeddingConfig {
            batch_size: 128,
            max_concurrent_requests: 8,
            max_in_flight_inputs: 16,
            ..EmbeddingConfig::default()
        };

        assert_eq!(config.bounded_parallelism(), (16, 1));
    }

    #[test]
    fn embedding_parallelism_normalizes_zero_values_to_one() {
        let config = EmbeddingConfig {
            batch_size: 0,
            max_concurrent_requests: 0,
            max_in_flight_inputs: 0,
            ..EmbeddingConfig::default()
        };

        assert_eq!(config.bounded_parallelism(), (1, 1));
    }

    #[test]
    fn max_in_flight_environment_value_normalizes_zero_to_one() {
        let _guard = EnvVarGuard::set(&[("VERA_MAX_IN_FLIGHT_INPUTS", "0")]);

        assert_eq!(default_max_in_flight_inputs(), 1);
    }

    #[test]
    fn config_serialization_round_trip() {
        let config = VeraConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: VeraConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.indexing.max_chunk_lines,
            config.indexing.max_chunk_lines
        );
        assert_eq!(
            deserialized.retrieval.default_limit,
            config.retrieval.default_limit
        );
    }

    #[test]
    fn openvino_backend_round_trip() {
        let backend = InferenceBackend::from_str("onnx-jina-openvino").unwrap();
        assert_eq!(
            backend,
            InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino)
        );
        assert_eq!(backend.to_string(), "onnx-jina-openvino");
        assert!(backend.is_local());
    }

    #[test]
    fn potion_code_backend_round_trip() {
        let backend = InferenceBackend::from_str("potion-code-cpu").unwrap();
        assert_eq!(backend, InferenceBackend::PotionCode);
        assert_eq!(backend.to_string(), "potion-code-cpu");
        assert!(backend.is_local());
        assert!(!backend.is_onnx());
        assert_eq!(backend.execution_provider(), None);
    }

    #[test]
    fn default_excludes_contains_common_dirs() {
        let config = IndexingConfig::default();
        assert!(config.default_excludes.contains(&".git".to_string()));
        assert!(
            config
                .default_excludes
                .contains(&"node_modules".to_string())
        );
        assert!(config.default_excludes.contains(&"target".to_string()));
    }

    #[test]
    fn resolve_backend_prefers_saved_backend_env() {
        let _guard = EnvVarGuard::set(&[("VERA_BACKEND", "onnx-jina-cuda"), ("VERA_LOCAL", "1")]);

        assert_eq!(
            resolve_backend(None),
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda)
        );
    }

    #[test]
    fn resolve_backend_falls_back_to_legacy_local_env() {
        let _guard = EnvVarGuard::apply(&[("VERA_BACKEND", None), ("VERA_LOCAL", Some("1"))]);

        assert_eq!(
            resolve_backend(None),
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu)
        );
    }

    /// Shorthand for matching without configured alias groups.
    fn model_names_match(a: &str, b: &str) -> bool {
        model_names_match_with_aliases(a, b, &[])
    }

    #[test]
    fn model_names_match_exact() {
        assert!(model_names_match(
            "jina-embeddings-v5",
            "jina-embeddings-v5"
        ));
    }

    #[test]
    fn model_names_match_with_org_prefix() {
        assert!(model_names_match(
            "jinaai/jina-embeddings-v5-text-nano-retrieval",
            "jina-embeddings-v5-text-nano-retrieval"
        ));
    }

    #[test]
    fn model_names_match_case_insensitive() {
        assert!(model_names_match(
            "Jina-Embeddings-V5",
            "jina-embeddings-v5"
        ));
    }

    #[test]
    fn model_names_match_different_models() {
        assert!(!model_names_match("jina-embeddings-v5", "jina-reranker-v2"));
    }

    #[test]
    fn model_names_match_configured_alias_group() {
        let aliases = vec![vec![
            "text-embedding-3-large".to_string(),
            "text-embedding-3-large-2".to_string(),
        ]];

        assert!(model_names_match_with_aliases(
            "text-embedding-3-large",
            "text-embedding-3-large-2",
            &aliases
        ));
        assert!(!model_names_match_with_aliases(
            "text-embedding-3-large",
            "text-embedding-3-small",
            &aliases
        ));
    }

    #[test]
    fn model_names_match_env_alias_group() {
        let _guard = EnvVarGuard::set(&[(
            "VERA_EMBEDDING_MODEL_ALIASES",
            "text-embedding-3-large,text-embedding-3-large-2;other,other-prod",
        )]);

        assert!(model_names_match(
            "text-embedding-3-large",
            "text-embedding-3-large-2"
        ));
        assert!(model_names_match("other", "other-prod"));
        assert!(!model_names_match(
            "text-embedding-3-large",
            "text-embedding-3-small"
        ));
    }

    #[test]
    fn model_alias_groups_ignore_single_entry_groups() {
        assert!(parse_model_alias_groups("solo;").is_empty());
        assert_eq!(
            parse_model_alias_groups("a,b; c , d").len(),
            2,
            "whitespace-tolerant groups parse"
        );
    }
}
