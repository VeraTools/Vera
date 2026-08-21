use crate::config::OnnxExecutionProvider;
use crate::embedding::provider::{EmbeddingError, EmbeddingProvider, api_err};
use crate::local_models::{LocalEmbeddingModelConfig, LocalEmbeddingPooling};
use anyhow::{Context, Result};
use ort::session::{Session, builder::GraphOptimizationLevel};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::{Encoding, Tokenizer};
use tokio::task;
use tokio_util::sync::CancellationToken;

const ADAPTIVE_BATCH_SCALER_STATE_VERSION: u32 = 1;

#[derive(Clone)]
pub struct LocalEmbeddingProvider {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    config: Arc<LocalEmbeddingModelConfig>,
    batch_scaler: Option<Arc<PersistedAdaptiveBatchScaler>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct BatchBucketWindow {
    max_safe_batch: Option<usize>,
    min_failed_batch: Option<usize>,
    #[serde(skip)]
    loaded_from_disk: bool,
}

/// Reference model size (MB) for cold-start batch scaling. Models larger than
/// this get proportionally smaller initial batch estimates.
const REFERENCE_MODEL_SIZE_MB: u64 = 100;

#[derive(Debug, Clone, Default)]
struct AdaptiveBatchScaler {
    max_length: usize,
    /// ONNX model file size in MB, used to dampen cold-start batch estimates
    /// for large models.
    model_size_mb: u64,
    buckets: BTreeMap<usize, BatchBucketWindow>,
}

impl AdaptiveBatchScaler {
    fn new(max_length: usize, model_size_mb: u64) -> Self {
        Self {
            max_length: max_length.max(1),
            model_size_mb,
            buckets: BTreeMap::new(),
        }
    }

    fn recommend_batch_len(&self, requested: usize, seq_len: usize) -> usize {
        let requested = requested.max(1);
        let bucket = self.bucket_for(seq_len);
        let seq_guided = self.sequence_guided_batch_len(requested, bucket);
        let learned = match self.buckets.get(&bucket) {
            Some(window) => match (window.max_safe_batch, window.min_failed_batch) {
                (Some(safe), Some(failed)) if safe + 1 < failed => (safe + failed) / 2,
                (Some(safe), Some(_)) => safe,
                (Some(safe), None) if window.loaded_from_disk => safe,
                (Some(safe), None) => safe.saturating_add(safe.div_ceil(2)),
                (None, Some(failed)) => failed.saturating_div(2).max(1),
                (None, None) => seq_guided,
            },
            None => seq_guided,
        };
        learned.clamp(1, requested)
    }

    fn note_success(&mut self, seq_len: usize, batch_len: usize) {
        let window = self.buckets.entry(self.bucket_for(seq_len)).or_default();
        window.loaded_from_disk = false;
        window.max_safe_batch = Some(
            window
                .max_safe_batch
                .map_or(batch_len, |current| current.max(batch_len)),
        );
        if window
            .min_failed_batch
            .is_some_and(|failed| batch_len >= failed)
        {
            window.min_failed_batch = None;
        }
    }

    fn note_failure(&mut self, seq_len: usize, batch_len: usize) {
        let window = self.buckets.entry(self.bucket_for(seq_len)).or_default();
        window.loaded_from_disk = false;
        window.min_failed_batch = Some(
            window
                .min_failed_batch
                .map_or(batch_len, |current| current.min(batch_len)),
        );
        if window.max_safe_batch.is_some_and(|safe| safe >= batch_len) {
            window.max_safe_batch = batch_len.checked_sub(1);
        }
    }

    fn bucket_for(&self, seq_len: usize) -> usize {
        let width = (self.max_length / 8).max(1);
        seq_len.max(1).div_ceil(width) * width
    }

    fn sequence_guided_batch_len(&self, requested: usize, seq_len: usize) -> usize {
        let reference_len = (self.max_length / 2).max(1) as u64;
        let seq_len = seq_len.max(1) as u64;
        let requested = requested.max(1) as u64;
        let scaled = requested
            .saturating_mul(reference_len.saturating_mul(reference_len))
            .div_ceil(seq_len.saturating_mul(seq_len));
        // Dampen for large models: if the model is bigger than the reference
        // size, reduce the cold-start estimate proportionally.
        let scaled = if self.model_size_mb > REFERENCE_MODEL_SIZE_MB {
            let ratio = REFERENCE_MODEL_SIZE_MB.max(1) as f64 / self.model_size_mb as f64;
            (scaled as f64 * ratio).ceil() as u64
        } else {
            scaled
        };
        scaled.clamp(1, requested) as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdaptiveBatchScalerProfile {
    key: String,
    backend: OnnxExecutionProvider,
    device_fingerprint: String,
    model_identity: String,
    max_length: usize,
}

#[derive(Debug, Clone)]
struct AdaptiveBatchScalerPersistenceTarget {
    path: PathBuf,
    profile: AdaptiveBatchScalerProfile,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedAdaptiveBatchScalerRegistry {
    #[serde(default = "default_adaptive_batch_scaler_state_version")]
    version: u32,
    #[serde(default)]
    profiles: BTreeMap<String, PersistedAdaptiveBatchScalerRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedAdaptiveBatchScalerRecord {
    backend: OnnxExecutionProvider,
    device_fingerprint: String,
    model_identity: String,
    max_length: usize,
    updated_at_secs: u64,
    #[serde(default)]
    buckets: BTreeMap<usize, BatchBucketWindow>,
}

#[derive(Debug)]
struct PersistedAdaptiveBatchScaler {
    inner: Mutex<PersistedAdaptiveBatchScalerState>,
}

#[derive(Debug)]
struct PersistedAdaptiveBatchScalerState {
    scaler: AdaptiveBatchScaler,
    persistence: Option<AdaptiveBatchScalerPersistenceTarget>,
    dirty: bool,
}

impl PersistedAdaptiveBatchScaler {
    fn load_or_new(
        max_length: usize,
        model_size_mb: u64,
        persistence: Option<AdaptiveBatchScalerPersistenceTarget>,
    ) -> Self {
        let mut scaler = AdaptiveBatchScaler::new(max_length, model_size_mb);

        if let Some(target) = persistence.as_ref() {
            match load_persisted_batch_scaler(&target.path, &target.profile) {
                Ok(Some(loaded)) => {
                    tracing::info!(
                        backend = %target.profile.backend,
                        device = %target.profile.device_fingerprint,
                        buckets = loaded.buckets.len(),
                        "loaded persisted adaptive batch scaler"
                    );
                    scaler = loaded;
                    scaler.model_size_mb = model_size_mb;
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        path = %target.path.display(),
                        error = %error,
                        "failed to load persisted adaptive batch scaler"
                    );
                }
            }
        }

        Self {
            inner: Mutex::new(PersistedAdaptiveBatchScalerState {
                scaler,
                persistence,
                dirty: false,
            }),
        }
    }

    fn recommend_batch_len(&self, requested: usize, seq_len: usize) -> usize {
        self.inner
            .lock()
            .unwrap()
            .scaler
            .recommend_batch_len(requested, seq_len)
    }

    fn note_success(&self, seq_len: usize, batch_len: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.scaler.note_success(seq_len, batch_len);
        inner.dirty = true;
    }

    fn note_failure(&self, seq_len: usize, batch_len: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.scaler.note_failure(seq_len, batch_len);
        inner.dirty = true;
    }

    fn flush(&self) -> Result<()> {
        let (persistence, scaler, dirty) = {
            let inner = self.inner.lock().unwrap();
            (
                inner.persistence.clone(),
                inner.scaler.clone(),
                inner.dirty && !inner.scaler.buckets.is_empty(),
            )
        };
        if !dirty {
            return Ok(());
        }
        let Some(persistence) = persistence else {
            return Ok(());
        };

        save_persisted_batch_scaler(&persistence.path, &persistence.profile, &scaler)
    }
}

impl Drop for PersistedAdaptiveBatchScaler {
    fn drop(&mut self) {
        if let Err(error) = self.flush() {
            tracing::warn!(error = %error, "failed to save adaptive batch scaler state");
        }
    }
}

fn default_adaptive_batch_scaler_state_version() -> u32 {
    ADAPTIVE_BATCH_SCALER_STATE_VERSION
}

impl LocalEmbeddingProvider {
    pub async fn new_with_ep(ep: OnnxExecutionProvider) -> Result<Self, EmbeddingError> {
        Self::new_with_ep_and_mem_limit(ep, 0).await
    }

    pub async fn new_with_ep_and_mem_limit(
        ep: OnnxExecutionProvider,
        gpu_mem_limit_mb: u64,
    ) -> Result<Self, EmbeddingError> {
        let base_config = LocalEmbeddingModelConfig::from_env().map_err(api_err)?;
        let mut config = base_config.clone();
        config.adjust_for_gpu(ep);
        let mut batch_scaler = if ep == OnnxExecutionProvider::Cpu {
            None
        } else {
            let persistence = match build_batch_scaler_persistence_target(ep, &config) {
                Ok(target) => target,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "failed to configure adaptive batch scaler persistence; continuing with in-memory state"
                    );
                    None
                }
            };
            let model_size_mb = config
                .cached_asset_paths()
                .ok()
                .and_then(|p| fs::metadata(&p.onnx_path).ok())
                .map(|m| effective_model_size_for_adaptive_batching(ep, m.len() / (1024 * 1024)))
                .unwrap_or(0);
            Some(Arc::new(PersistedAdaptiveBatchScaler::load_or_new(
                config.max_length,
                model_size_mb,
                persistence,
            )))
        };
        let ort_path = crate::local_models::ensure_ort_library_for_ep(ep)
            .await
            .map_err(api_err)?;
        crate::local_models::ensure_ort_runtime(Some(&ort_path)).map_err(api_err)?;
        crate::local_models::ensure_provider_dependencies(ep, &ort_path).map_err(api_err)?;
        let components = load_embedding_components(ep, &config, gpu_mem_limit_mb).await;
        let (session, tokenizer) = match components {
            Ok(components) => components,
            Err(error) if ep != OnnxExecutionProvider::Cpu => {
                tracing::warn!(
                    backend = %ep,
                    error = %error,
                    "selected ONNX execution provider failed; retrying embedding session on CPU"
                );
                config = base_config;
                batch_scaler = None;
                load_embedding_components(OnnxExecutionProvider::Cpu, &config, 0)
                    .await
                    .map_err(|fallback_error| {
                        api_err(format!(
                            "{}\nHint: run `vera repair --onnx-jina-{ep}`.\nCPU fallback also failed: {}\nHint: run `vera repair --onnx-jina-cpu`.",
                            crate::local_models::wrap_ort_error(error),
                            crate::local_models::wrap_ort_error(fallback_error),
                        ))
                    })?
            }
            Err(error) => {
                return Err(api_err(format!(
                    "{}\nHint: run `vera repair --onnx-jina-{ep}`.",
                    crate::local_models::wrap_ort_error(error)
                )));
            }
        };

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            config: Arc::new(config),
            batch_scaler,
        })
    }

    pub fn probe_provider_registration(ep: OnnxExecutionProvider) -> Result<()> {
        let builder = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;
        let _ = register_execution_provider(builder, ep, 0)?;
        Ok(())
    }

    pub fn probe_session(ep: OnnxExecutionProvider) -> Result<()> {
        let mut config = LocalEmbeddingModelConfig::from_env()?;
        config.adjust_for_gpu(ep);
        let ort_path = crate::local_models::ort_library_path_for_ep(ep)?;
        crate::local_models::ensure_ort_runtime(Some(&ort_path))?;
        let asset_paths = config.cached_asset_paths()?;
        let _ = build_session(ep, asset_paths.onnx_path, 0)?;
        Ok(())
    }

    pub fn probe_inference(ep: OnnxExecutionProvider) -> Result<()> {
        let mut config = LocalEmbeddingModelConfig::from_env()?;
        config.adjust_for_gpu(ep);
        let ort_path = crate::local_models::ort_library_path_for_ep(ep)?;
        crate::local_models::ensure_ort_runtime(Some(&ort_path))?;
        let asset_paths = config.cached_asset_paths()?;
        let mut session = build_session(ep, asset_paths.onnx_path, 0)?;
        let tokenizer = load_tokenizer(asset_paths.tokenizer_path, config.max_length)?;
        run_probe_inference(&mut session, &tokenizer)
    }

    fn do_embed_cancellable(
        &self,
        texts: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let encodings = self.tokenize_texts(texts).map_err(api_err)?;

        if encodings.is_empty() {
            return Ok(Vec::new());
        }

        if self.batch_scaler.is_none() {
            if cancel.is_cancelled() {
                return Err(EmbeddingError::Cancelled);
            }
            return self.do_embed_once(&encodings).map_err(api_err);
        }

        let mut results = Vec::with_capacity(encodings.len());
        let mut start = 0;
        while start < encodings.len() {
            if cancel.is_cancelled() {
                return Err(EmbeddingError::Cancelled);
            }
            let remaining = &encodings[start..];
            let seq_len = batch_max_len(remaining);
            let planned_batch_len = self
                .batch_scaler
                .as_ref()
                .unwrap()
                .recommend_batch_len(remaining.len(), seq_len)
                .min(remaining.len())
                .max(1);
            let end = start + planned_batch_len;
            tracing::debug!(
                requested_batch_size = remaining.len(),
                planned_batch_size = planned_batch_len,
                seq_len,
                "planning local ONNX embedding sub-batch"
            );
            let batch = self
                .embed_with_adaptive_batching(&encodings[start..end])
                .map_err(api_err)?;
            results.extend(batch);
            start = end;
        }

        Ok(results)
    }

    #[allow(clippy::needless_range_loop)]
    fn do_embed_once(&self, encodings: &[Encoding]) -> Result<Vec<Vec<f32>>> {
        let batch_size = encodings.len();
        let mut max_len = batch_max_len(encodings);
        if max_len == 0 {
            max_len = 1;
        }

        let mut input_ids = ndarray::Array2::<i64>::zeros((batch_size, max_len));
        let mut attention_mask = ndarray::Array2::<i64>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len();
            for j in 0..len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
            }
        }

        let input_ids_tensor = ort::value::Tensor::from_array(input_ids)
            .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;
        let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask.clone())
            .map_err(|e| anyhow::anyhow!("Tensor error: {}", e))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ];

        let t0 = std::time::Instant::now();
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs)?;
        tracing::debug!(
            batch_size,
            seq_len = max_len,
            elapsed_ms = t0.elapsed().as_millis(),
            "ort session.run"
        );

        let output_value = outputs.values().next().unwrap();
        let (shape, data) = output_value.try_extract_tensor::<f32>()?;
        let ndim = shape.len();

        let mut result = Vec::with_capacity(batch_size);

        if ndim == 2 {
            let dim = shape[1] as usize;
            for i in 0..batch_size {
                let start = i * dim;
                let mut emb = data[start..start + dim].to_vec();
                normalize_embedding(&mut emb);
                result.push(emb);
            }
        } else if ndim == 3 {
            let seq_len = shape[1] as usize;
            let dim = shape[2] as usize;
            for i in 0..batch_size {
                let emb = match self.config.pooling {
                    LocalEmbeddingPooling::Cls => {
                        data[i * seq_len * dim..(i * seq_len + 1) * dim].to_vec()
                    }
                    LocalEmbeddingPooling::Mean => {
                        let mut emb = vec![0.0; dim];
                        let mut valid_tokens = 0.0;
                        for j in 0..max_len {
                            if attention_mask[[i, j]] == 1 {
                                valid_tokens += 1.0;
                                for d in 0..dim {
                                    emb[d] += data[i * seq_len * dim + j * dim + d];
                                }
                            }
                        }
                        if valid_tokens > 0.0 {
                            for value in &mut emb {
                                *value /= valid_tokens;
                            }
                        }
                        emb
                    }
                    LocalEmbeddingPooling::LastToken => {
                        let last = last_unpadded_index(&attention_mask, i, max_len);
                        let start = i * seq_len * dim + last * dim;
                        data[start..start + dim].to_vec()
                    }
                };
                let mut emb = emb;
                normalize_embedding(&mut emb);
                result.push(emb);
            }
        } else {
            anyhow::bail!("Unexpected tensor shape: {:?}", shape);
        }

        Ok(result)
    }

    fn tokenize_texts(&self, texts: &[String]) -> Result<Vec<Encoding>> {
        let mut encodings = Vec::with_capacity(texts.len());
        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
            encodings.push(encoding);
        }
        Ok(encodings)
    }

    fn embed_with_adaptive_batching(&self, encodings: &[Encoding]) -> Result<Vec<Vec<f32>>> {
        match self.do_embed_once(encodings) {
            Ok(results) => {
                self.note_batch_success(encodings);
                Ok(results)
            }
            Err(error) if encodings.len() > 1 && is_retryable_onnx_allocation_error(&error) => {
                self.note_batch_failure(encodings);
                let split_at = self.retry_split_at(encodings);
                tracing::warn!(
                    batch_size = encodings.len(),
                    seq_len = batch_max_len(encodings),
                    split_at,
                    error = %error,
                    "embedding batch hit an ONNX allocation error, retrying with smaller batches"
                );
                let mut left = self.embed_with_adaptive_batching(&encodings[..split_at])?;
                let mut right = self.embed_with_adaptive_batching(&encodings[split_at..])?;
                left.append(&mut right);
                Ok(left)
            }
            Err(error) if encodings.len() == 1 && is_retryable_onnx_allocation_error(&error) => {
                self.note_batch_failure(encodings);
                let seq_len = batch_max_len(encodings);
                tracing::error!(
                    seq_len,
                    "GPU out of memory even with batch_size=1. \
                     The model is too large for available VRAM. \
                     Try: (1) use CPU mode instead of GPU, \
                     (2) use a smaller/quantized model, \
                     (3) free GPU memory from other applications"
                );
                Err(error.context(
                    "GPU out of memory with batch_size=1: model too large for available VRAM. \
                     Try CPU mode, a smaller model, or free GPU memory from other applications",
                ))
            }
            Err(error) => Err(error),
        }
    }

    fn retry_split_at(&self, encodings: &[Encoding]) -> usize {
        if encodings.len() <= 1 {
            return 1;
        }
        let seq_len = batch_max_len(encodings);
        let suggested = self
            .batch_scaler
            .as_ref()
            .map(|scaler| scaler.recommend_batch_len(encodings.len() - 1, seq_len))
            .unwrap_or_else(|| encodings.len() / 2);
        suggested.clamp(1, encodings.len() - 1)
    }

    fn note_batch_success(&self, encodings: &[Encoding]) {
        if let Some(scaler) = &self.batch_scaler {
            scaler.note_success(batch_max_len(encodings), encodings.len());
        }
    }

    fn note_batch_failure(&self, encodings: &[Encoding]) {
        if let Some(scaler) = &self.batch_scaler {
            scaler.note_failure(batch_max_len(encodings), encodings.len());
        }
    }
}

async fn load_embedding_components(
    ep: OnnxExecutionProvider,
    config: &LocalEmbeddingModelConfig,
    gpu_mem_limit_mb: u64,
) -> Result<(Session, Tokenizer)> {
    let asset_paths = crate::local_models::ensure_local_embedding_assets(config).await?;
    let onnx_path = asset_paths.onnx_path;
    let tokenizer_path = asset_paths.tokenizer_path;
    let tokenizer_max_length = config.max_length;

    let tokenizer =
        task::spawn_blocking(move || load_tokenizer(tokenizer_path, tokenizer_max_length))
            .await??;
    let session =
        task::spawn_blocking(move || build_session(ep, onnx_path, gpu_mem_limit_mb)).await??;

    Ok((session, tokenizer))
}

fn build_batch_scaler_persistence_target(
    ep: OnnxExecutionProvider,
    config: &LocalEmbeddingModelConfig,
) -> Result<Option<AdaptiveBatchScalerPersistenceTarget>> {
    let path = crate::local_models::vera_home_dir()?.join("adaptive-batch-scaler.json");
    let gpu_info = crate::config::detect_gpu_info(ep);
    let device_fingerprint = gpu_info.fingerprint;
    let model_identity = config.model_identity();
    let max_length = config.max_length;
    let key =
        format!("{ep}|device={device_fingerprint}|model={model_identity}|max_length={max_length}");
    Ok(Some(AdaptiveBatchScalerPersistenceTarget {
        path,
        profile: AdaptiveBatchScalerProfile {
            key,
            backend: ep,
            device_fingerprint,
            model_identity,
            max_length,
        },
    }))
}

fn load_persisted_batch_scaler(
    path: &Path,
    profile: &AdaptiveBatchScalerProfile,
) -> Result<Option<AdaptiveBatchScaler>> {
    if !path.exists() {
        return Ok(None);
    }

    let data = fs::read(path).with_context(|| {
        format!(
            "failed to read adaptive batch scaler state {}",
            path.display()
        )
    })?;
    let registry: PersistedAdaptiveBatchScalerRegistry = serde_json::from_slice(&data)
        .with_context(|| {
            format!(
                "failed to parse adaptive batch scaler state {}",
                path.display()
            )
        })?;
    if registry.version != ADAPTIVE_BATCH_SCALER_STATE_VERSION {
        return Ok(None);
    }

    let Some(record) = registry.profiles.get(&profile.key) else {
        return Ok(None);
    };
    if record.backend != profile.backend
        || record.device_fingerprint != profile.device_fingerprint
        || record.model_identity != profile.model_identity
        || record.max_length != profile.max_length
    {
        return Ok(None);
    }

    Ok(Some(AdaptiveBatchScaler {
        max_length: record.max_length.max(1),
        model_size_mb: 0, // will be overwritten by caller
        buckets: record
            .buckets
            .iter()
            .map(|(bucket, window)| {
                let mut window = window.clone();
                window.loaded_from_disk = true;
                window.max_safe_batch = window.max_safe_batch.map(persisted_batch_margin);
                (*bucket, window)
            })
            .collect(),
    }))
}

fn save_persisted_batch_scaler(
    path: &Path,
    profile: &AdaptiveBatchScalerProfile,
    scaler: &AdaptiveBatchScaler,
) -> Result<()> {
    let mut registry = if path.exists() {
        let data = fs::read(path).with_context(|| {
            format!(
                "failed to read adaptive batch scaler state before saving {}",
                path.display()
            )
        })?;
        serde_json::from_slice::<PersistedAdaptiveBatchScalerRegistry>(&data).with_context(
            || {
                format!(
                    "failed to parse adaptive batch scaler state before saving {}",
                    path.display()
                )
            },
        )?
    } else {
        PersistedAdaptiveBatchScalerRegistry::default()
    };
    registry.version = ADAPTIVE_BATCH_SCALER_STATE_VERSION;
    registry.profiles.insert(
        profile.key.clone(),
        PersistedAdaptiveBatchScalerRecord {
            backend: profile.backend,
            device_fingerprint: profile.device_fingerprint.clone(),
            model_identity: profile.model_identity.clone(),
            max_length: profile.max_length,
            updated_at_secs: now_unix_secs(),
            buckets: scaler.buckets.clone(),
        },
    );

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create adaptive batch scaler directory {}",
                parent.display()
            )
        })?;
    }
    let json = serde_json::to_vec_pretty(&registry)
        .context("failed to serialize adaptive batch scaler state")?;
    fs::write(path, json).with_context(|| {
        format!(
            "failed to write adaptive batch scaler state {}",
            path.display()
        )
    })
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn persisted_batch_margin(batch_len: usize) -> usize {
    let margin = batch_len.div_ceil(8).max(1);
    batch_len.saturating_sub(margin).max(1)
}

// The model-size dampener protects discrete GPU backends from VRAM cold-start
// overestimates. CoreML runs on unified memory, so file size is not a useful
// proxy and can make Apple Silicon indexing start at batch 1 for long chunks.
fn effective_model_size_for_adaptive_batching(
    ep: OnnxExecutionProvider,
    model_size_mb: u64,
) -> u64 {
    if ep == OnnxExecutionProvider::CoreMl {
        0
    } else {
        model_size_mb
    }
}

fn batch_max_len(encodings: &[Encoding]) -> usize {
    encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(0)
}

fn is_retryable_onnx_allocation_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("failed to allocate memory")
        || message.contains("cuda out of memory")
        || message.contains("out of memory")
        || message.contains("bfcarena")
}

fn load_tokenizer(tokenizer_path: std::path::PathBuf, max_length: usize) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer init failed: {}", e))?;
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("Tokenizer truncation init failed: {}", e))?;
    Ok(tokenizer)
}

/// Index of the final unpadded token in row `row` of `attention_mask`.
///
/// Scanning for the highest set position rather than counting ones keeps this
/// correct under both left and right padding. An all-padding row yields 0.
fn last_unpadded_index(attention_mask: &ndarray::Array2<i64>, row: usize, len: usize) -> usize {
    (0..len)
        .rev()
        .find(|&j| attention_mask[[row, j]] == 1)
        .unwrap_or(0)
}

fn normalize_embedding(embedding: &mut [f32]) {
    let norm: f32 = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 1e-6 {
        for value in embedding {
            *value /= norm;
        }
    }
}

fn build_session(
    ep: OnnxExecutionProvider,
    onnx_path: std::path::PathBuf,
    gpu_mem_limit_mb: u64,
) -> Result<Session> {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // CPU EP benefits from all cores; GPU EPs do compute on device,
    // so limit CPU threads to avoid contention.
    let threads = if ep == OnnxExecutionProvider::Cpu {
        available
    } else {
        available.min(4)
    };
    tracing::info!(
        threads,
        available,
        model = %onnx_path.display(),
        "building ONNX session"
    );
    let builder = ort::session::builder::SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?;
    let builder = register_execution_provider(builder, ep, gpu_mem_limit_mb)?;
    builder
        .commit_from_file(&onnx_path)
        .with_context(|| format!("failed to load embedding model {}", onnx_path.display()))
}

fn run_probe_inference(session: &mut Session, tokenizer: &Tokenizer) -> Result<()> {
    let encoding = tokenizer
        .encode("vera doctor probe", true)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

    let ids = encoding.get_ids();
    let mask = encoding.get_attention_mask();
    let max_len = ids.len().max(1);
    let mut input_ids = ndarray::Array2::<i64>::zeros((1, max_len));
    let mut attention_mask = ndarray::Array2::<i64>::zeros((1, max_len));

    for (index, token_id) in ids.iter().enumerate() {
        input_ids[[0, index]] = *token_id as i64;
    }
    for (index, mask_value) in mask.iter().enumerate() {
        attention_mask[[0, index]] = *mask_value as i64;
    }

    let inputs = ort::inputs![
        "input_ids" => ort::value::Tensor::from_array(input_ids)?,
        "attention_mask" => ort::value::Tensor::from_array(attention_mask)?,
    ];

    let outputs = session.run(inputs)?;
    let output = outputs
        .values()
        .next()
        .context("embedding model produced no outputs")?;
    let (_, data) = output.try_extract_tensor::<f32>()?;
    if data.is_empty() {
        anyhow::bail!("embedding output tensor was empty");
    }
    if !data.iter().all(|value| value.is_finite()) {
        anyhow::bail!("embedding output contained non-finite values");
    }
    Ok(())
}

impl EmbeddingProvider for LocalEmbeddingProvider {
    fn expected_dim(&self) -> Option<usize> {
        Some(self.config.embedding_dim)
    }

    fn prepare_query_text(&self, query: &str) -> String {
        self.config.query_text(query)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let provider = self.clone();
        let texts = texts.to_vec();

        task::spawn_blocking(move || {
            provider.do_embed_cancellable(&texts, &CancellationToken::new())
        })
        .await
        .map_err(api_err)?
    }

    async fn embed_batch_cancellable(
        &self,
        texts: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let provider = self.clone();
        let texts = texts.to_vec();
        let cancel = cancel.clone();

        task::spawn_blocking(move || provider.do_embed_cancellable(&texts, &cancel))
            .await
            .map_err(api_err)?
    }
}

/// Register the appropriate ONNX execution provider on a session builder.
///
/// `gpu_mem_limit_mb`: if >0, caps GPU memory arena for CUDA/ROCm.
fn register_execution_provider(
    builder: ort::session::builder::SessionBuilder,
    ep: OnnxExecutionProvider,
    gpu_mem_limit_mb: u64,
) -> ort::Result<ort::session::builder::SessionBuilder> {
    match ep {
        OnnxExecutionProvider::Cpu => {
            tracing::info!("using CPU execution provider");
            Ok(builder)
        }
        OnnxExecutionProvider::Cuda => {
            tracing::info!("registering CUDA execution provider");
            let mut cuda_ep = ort::execution_providers::CUDAExecutionProvider::default();
            if gpu_mem_limit_mb > 0 {
                let limit_bytes = gpu_mem_limit_mb as usize * 1024 * 1024;
                tracing::info!("setting CUDA memory limit: {gpu_mem_limit_mb}MB");
                cuda_ep = cuda_ep.with_memory_limit(limit_bytes);
            }
            let result = builder.with_execution_providers([cuda_ep.build()]);
            if result.is_ok() {
                tracing::info!(
                    "CUDA execution provider registered (will fall back to CPU if unavailable)"
                );
            }
            result
        }
        OnnxExecutionProvider::Rocm => {
            tracing::info!("registering ROCm execution provider");
            builder.with_execution_providers([
                ort::execution_providers::ROCmExecutionProvider::default().build(),
            ])
        }
        OnnxExecutionProvider::DirectMl => {
            tracing::info!("registering DirectML execution provider");
            builder.with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
            ])
        }
        OnnxExecutionProvider::CoreMl => {
            tracing::info!("registering CoreML execution provider");
            let result = builder.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ]);
            if result.is_ok() {
                tracing::info!(
                    "CoreML execution provider registered (will fall back to CPU if unavailable)"
                );
            }
            result
        }
        OnnxExecutionProvider::OpenVino => {
            tracing::info!("registering OpenVINO execution provider");
            builder.with_execution_providers([
                ort::execution_providers::OpenVINOExecutionProvider::default().build(),
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Right padding: the model's real final token sits before the pad run.
    #[test]
    fn last_unpadded_index_right_padded() {
        let mask = ndarray::arr2(&[[1i64, 1, 1, 0, 0]]);
        assert_eq!(last_unpadded_index(&mask, 0, 5), 2);
    }

    /// Left padding: the real final token is the last column. Counting set
    /// bits would answer 2 here, which is a pad position.
    #[test]
    fn last_unpadded_index_left_padded() {
        let mask = ndarray::arr2(&[[0i64, 0, 1, 1, 1]]);
        assert_eq!(last_unpadded_index(&mask, 0, 5), 4);
    }

    #[test]
    fn last_unpadded_index_full_row_and_single_token() {
        assert_eq!(
            last_unpadded_index(&ndarray::arr2(&[[1i64, 1, 1]]), 0, 3),
            2
        );
        assert_eq!(last_unpadded_index(&ndarray::arr2(&[[1i64]]), 0, 1), 0);
    }

    #[test]
    fn last_unpadded_index_is_per_row() {
        let mask = ndarray::arr2(&[[1i64, 1, 0, 0], [1, 1, 1, 1]]);
        assert_eq!(last_unpadded_index(&mask, 0, 4), 1);
        assert_eq!(last_unpadded_index(&mask, 1, 4), 3);
    }

    #[test]
    fn last_unpadded_index_all_padding_falls_back_to_zero() {
        let mask = ndarray::arr2(&[[0i64, 0, 0]]);
        assert_eq!(last_unpadded_index(&mask, 0, 3), 0);
    }

    #[tokio::test]
    async fn test_local_embedding_provider() {
        // Skip if ONNX Runtime is not installed (requires libonnxruntime.so)
        if crate::local_models::ensure_ort_runtime(None).is_err() {
            eprintln!("Skipping: ONNX Runtime not available");
            return;
        }
        // Since test downloads ~150MB, this could take a moment.
        let provider = LocalEmbeddingProvider::new_with_ep(OnnxExecutionProvider::Cpu)
            .await
            .unwrap();
        let texts = vec!["Hello world".to_string(), "Another test".to_string()];
        let embeddings = provider.embed_batch(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), provider.expected_dim().unwrap());

        assert!(embeddings[0].iter().all(|x| x.is_finite()));
        let sum_abs: f32 = embeddings[0].iter().map(|x| x.abs()).sum();
        assert!(sum_abs > 0.1);
    }

    #[test]
    fn adaptive_batch_scaler_shrinks_long_batches_quadratically() {
        let scaler = AdaptiveBatchScaler::new(512, 0);
        assert_eq!(scaler.recommend_batch_len(128, 64), 128);
        assert_eq!(scaler.recommend_batch_len(128, 512), 32);
    }

    #[test]
    fn adaptive_batch_scaler_dampens_cold_start_for_large_models() {
        // A 400MB model (4x reference) should get ~1/4 the cold-start batch size.
        let small = AdaptiveBatchScaler::new(512, 50);
        let large = AdaptiveBatchScaler::new(512, 400);
        let small_batch = small.recommend_batch_len(128, 512);
        let large_batch = large.recommend_batch_len(128, 512);
        assert!(
            large_batch < small_batch,
            "large model ({large_batch}) should get smaller cold-start batch than small model ({small_batch})"
        );
        // 400MB is 4x reference (100MB), so batch should be ~1/4 of 32 = 8.
        assert_eq!(large_batch, 8);
    }

    #[test]
    fn coreml_adaptive_batching_ignores_model_size_cold_start_dampening() {
        let cuda_size =
            effective_model_size_for_adaptive_batching(OnnxExecutionProvider::Cuda, 400);
        let coreml_size =
            effective_model_size_for_adaptive_batching(OnnxExecutionProvider::CoreMl, 400);

        let cuda = AdaptiveBatchScaler::new(512, cuda_size);
        let coreml = AdaptiveBatchScaler::new(512, coreml_size);

        assert_eq!(cuda.recommend_batch_len(16, 512), 1);
        assert_eq!(coreml.recommend_batch_len(16, 512), 4);
    }

    #[test]
    fn adaptive_batch_scaler_learns_safe_windows_per_length_bucket() {
        let mut scaler = AdaptiveBatchScaler::new(512, 0);
        assert_eq!(scaler.recommend_batch_len(128, 512), 32);

        scaler.note_success(512, 32);
        assert_eq!(scaler.recommend_batch_len(128, 512), 48);

        scaler.note_failure(512, 48);
        assert_eq!(scaler.recommend_batch_len(128, 512), 40);

        scaler.note_success(512, 40);
        assert_eq!(scaler.recommend_batch_len(128, 512), 44);
    }

    #[test]
    fn adaptive_batch_scaler_keeps_short_and_long_buckets_independent() {
        let mut scaler = AdaptiveBatchScaler::new(512, 0);
        scaler.note_failure(512, 48);

        assert_eq!(scaler.recommend_batch_len(128, 512), 24);
        assert_eq!(scaler.recommend_batch_len(128, 128), 128);
    }

    #[test]
    fn persisted_scaler_round_trips_by_profile_key() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("adaptive-batch-scaler.json");
        let profile = AdaptiveBatchScalerProfile {
            key: "cuda|device=rtx-4080|model=jina|max_length=512".to_string(),
            backend: OnnxExecutionProvider::Cuda,
            device_fingerprint: "rtx-4080".to_string(),
            model_identity: "jina".to_string(),
            max_length: 512,
        };
        let mut scaler = AdaptiveBatchScaler::new(512, 0);
        scaler.note_success(512, 40);
        scaler.note_failure(512, 48);

        save_persisted_batch_scaler(&path, &profile, &scaler).unwrap();
        let loaded = load_persisted_batch_scaler(&path, &profile)
            .unwrap()
            .unwrap();

        assert_eq!(loaded.recommend_batch_len(128, 512), 41);
    }

    #[test]
    fn persisted_scaler_ignores_mismatched_device_or_model() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("adaptive-batch-scaler.json");
        let profile = AdaptiveBatchScalerProfile {
            key: "cuda|device=rtx-4080|model=jina|max_length=512".to_string(),
            backend: OnnxExecutionProvider::Cuda,
            device_fingerprint: "rtx-4080".to_string(),
            model_identity: "jina".to_string(),
            max_length: 512,
        };
        let mut scaler = AdaptiveBatchScaler::new(512, 0);
        scaler.note_success(512, 40);
        save_persisted_batch_scaler(&path, &profile, &scaler).unwrap();

        let mismatched = AdaptiveBatchScalerProfile {
            key: "cuda|device=rtx-4090|model=jina|max_length=512".to_string(),
            device_fingerprint: "rtx-4090".to_string(),
            ..profile
        };

        assert!(
            load_persisted_batch_scaler(&path, &mismatched)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn persisted_scaler_applies_margin_before_first_reuse() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("adaptive-batch-scaler.json");
        let profile = AdaptiveBatchScalerProfile {
            key: "cuda|device=rtx-4080|model=jina|max_length=512".to_string(),
            backend: OnnxExecutionProvider::Cuda,
            device_fingerprint: "rtx-4080".to_string(),
            model_identity: "jina".to_string(),
            max_length: 512,
        };
        let mut scaler = AdaptiveBatchScaler::new(512, 0);
        scaler.note_success(512, 128);

        save_persisted_batch_scaler(&path, &profile, &scaler).unwrap();
        let loaded = load_persisted_batch_scaler(&path, &profile)
            .unwrap()
            .unwrap();

        assert_eq!(loaded.recommend_batch_len(128, 512), 112);
    }

    #[test]
    fn retryable_onnx_allocation_error_detection_matches_cuda_oom_messages() {
        let error = anyhow::anyhow!(
            "Non-zero status code returned while running MultiHeadAttention node. \
             Status Message: BFCArena::AllocateRawInternal Failed to allocate memory"
        );
        assert!(is_retryable_onnx_allocation_error(&error));

        let non_oom = anyhow::anyhow!("invalid graph input");
        assert!(!is_retryable_onnx_allocation_error(&non_oom));
    }
}
