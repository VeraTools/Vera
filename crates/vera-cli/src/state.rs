//! Persistent CLI state for agent-friendly setup and installs.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoredConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_mode: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<vera_core::config::InferenceBackend>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_api: Option<ApiEndpointConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reranker_api: Option<ApiEndpointConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core_config: Option<vera_core::config::VeraConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_embedding_model: Option<vera_core::local_models::LocalEmbeddingModelConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ApiEndpointConfig {
    pub base_url: String,
    pub model_id: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoredSecrets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reranker_api_key: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstallProvenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ApiSetupInput {
    pub base_url: String,
    pub model_id: String,
    pub api_key: String,
}

pub fn load_saved_config() -> Result<StoredConfig> {
    let mut config: StoredConfig = load_json_file(&config_path()?)?;
    // `setup` freezes the resolved model config here, and this copy outranks
    // the preset. Without repair, an install created before jina's pooling was
    // corrected keeps mean-pooling it indefinitely.
    config.local_embedding_model = config
        .local_embedding_model
        .map(vera_core::local_models::LocalEmbeddingModelConfig::repair_stored_defaults);
    Ok(config)
}

pub fn load_saved_secrets() -> Result<StoredSecrets> {
    load_json_file(&credentials_path()?)
}

pub fn load_install_provenance() -> Result<InstallProvenance> {
    load_json_file(&install_path()?)
}

pub fn save_backend(backend: vera_core::config::InferenceBackend) -> Result<()> {
    let mut config = load_saved_config()?;
    config.backend = Some(backend);
    config.local_mode = Some(backend.is_local());
    save_config(&config)
}

pub fn save_local_embedding_model(
    model: &vera_core::local_models::LocalEmbeddingModelConfig,
) -> Result<()> {
    let mut config = load_saved_config()?;
    config.local_embedding_model = Some(model.clone());
    save_config(&config)
}

pub fn saved_local_embedding_model()
-> Result<Option<vera_core::local_models::LocalEmbeddingModelConfig>> {
    Ok(load_saved_config()?.local_embedding_model)
}

pub fn saved_backend() -> Result<Option<vera_core::config::InferenceBackend>> {
    use vera_core::config::{InferenceBackend, OnnxExecutionProvider};

    let config = load_saved_config()?;
    Ok(config.backend.or(match config.local_mode {
        Some(true) => Some(InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu)),
        Some(false) => Some(InferenceBackend::Api),
        None => None,
    }))
}

pub fn save_install_method(install_method: Option<&str>) -> Result<()> {
    let mut config = load_saved_config()?;
    config.install_method = install_method.map(|method| method.to_string());
    save_config(&config)
}

pub fn save_runtime_config(config: &vera_core::config::VeraConfig) -> Result<()> {
    let mut stored = load_saved_config()?;
    stored.core_config = Some(config.clone());
    save_config(&stored)
}

pub fn save_api_setup(embedding: &ApiSetupInput, reranker: Option<&ApiSetupInput>) -> Result<()> {
    let mut config = load_saved_config()?;
    config.backend = Some(vera_core::config::InferenceBackend::Api);
    config.local_mode = Some(false);
    config.embedding_api = Some(ApiEndpointConfig {
        base_url: embedding.base_url.clone(),
        model_id: embedding.model_id.clone(),
    });
    config.reranker_api = reranker.map(|cfg| ApiEndpointConfig {
        base_url: cfg.base_url.clone(),
        model_id: cfg.model_id.clone(),
    });
    save_config(&config)?;

    let mut secrets = load_saved_secrets()?;
    secrets.embedding_api_key = Some(embedding.api_key.clone());
    secrets.reranker_api_key = reranker.map(|cfg| cfg.api_key.clone());
    save_secrets(&secrets)
}

/// Drop persisted API reranker settings. Called when selecting a local
/// backend so local mode cannot silently rerank through a stale saved
/// endpoint; shell-set RERANKER_MODEL_* vars still signal explicit intent.
pub fn clear_reranker_setup() -> Result<()> {
    let mut config = load_saved_config()?;
    config.reranker_api = None;
    save_config(&config)?;

    let mut secrets = load_saved_secrets()?;
    secrets.reranker_api_key = None;
    save_secrets(&secrets)
}

pub fn load_runtime_config() -> Result<vera_core::config::VeraConfig> {
    let default = vera_core::config::VeraConfig::default();
    Ok(load_saved_config()?.core_config.unwrap_or(default))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(vera_dir()?.join("config.json"))
}

pub fn credentials_path() -> Result<PathBuf> {
    Ok(vera_dir()?.join("credentials.json"))
}

pub fn install_path() -> Result<PathBuf> {
    Ok(vera_dir()?.join("install.json"))
}

pub fn vera_dir() -> Result<PathBuf> {
    vera_core::local_models::vera_home_dir()
}

pub fn user_home_dir() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("VERA_USER_HOME") {
        if !path.trim().is_empty() {
            return Ok(PathBuf::from(path));
        }
    }
    dirs::home_dir().context("Could not find home directory")
}

pub fn apply_saved_env() -> Result<()> {
    apply_saved_env_impl(false)
}

pub fn apply_saved_env_force() -> Result<()> {
    apply_saved_env_impl(true)
}

fn apply_saved_env_impl(force: bool) -> Result<()> {
    let config = load_saved_config()?;
    let secrets = load_saved_secrets()?;

    if let Some(backend) = config.backend {
        set_env_value("VERA_BACKEND", &backend.to_string(), force);
        set_env_value(
            "VERA_LOCAL",
            if backend.is_local() { "1" } else { "0" },
            force,
        );
    } else if let Some(local_mode) = config.local_mode {
        set_env_value("VERA_LOCAL", if local_mode { "1" } else { "0" }, force);
    }

    if let Some(embedding) = config.embedding_api.as_ref() {
        set_env_value("EMBEDDING_MODEL_BASE_URL", &embedding.base_url, force);
        set_env_value("EMBEDDING_MODEL_ID", &embedding.model_id, force);
    }
    if let Some(api_key) = secrets.embedding_api_key.as_deref() {
        set_env_value("EMBEDDING_MODEL_API_KEY", api_key, force);
    }

    if let Some(reranker) = config.reranker_api.as_ref() {
        set_env_value("RERANKER_MODEL_BASE_URL", &reranker.base_url, force);
        set_env_value("RERANKER_MODEL_ID", &reranker.model_id, force);
    }
    if let Some(api_key) = secrets.reranker_api_key.as_deref() {
        set_env_value("RERANKER_MODEL_API_KEY", api_key, force);
    }

    apply_local_embedding_env(config.local_embedding_model.as_ref(), force);

    Ok(())
}

fn save_config(config: &StoredConfig) -> Result<()> {
    write_json_file(&config_path()?, config)
}

fn save_secrets(secrets: &StoredSecrets) -> Result<()> {
    write_json_file(&credentials_path()?, secrets)
}

fn load_json_file<T>(path: &Path) -> Result<T>
where
    T: Default + for<'de> Deserialize<'de>,
{
    if !path.exists() {
        return Ok(T::default());
    }

    let contents = fs::read(path)
        .with_context(|| format!("failed to read persistent state: {}", path.display()))?;
    if contents.is_empty() {
        return Ok(T::default());
    }

    serde_json::from_slice(&contents)
        .with_context(|| format!("failed to parse persistent state: {}", path.display()))
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let contents = serde_json::to_vec_pretty(value)
        .with_context(|| format!("failed to serialize state for {}", path.display()))?;
    write_private_file(path, &contents)
}

fn write_private_file(path: &Path, contents: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let tmp_path = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let mut file = options
        .open(&tmp_path)
        .with_context(|| format!("failed to open {}", tmp_path.display()))?;
    file.write_all(contents)
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("failed to finalize {}", tmp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to sync {}", tmp_path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to set permissions on {}", tmp_path.display()))?;
    }

    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("failed to replace existing {}", path.display()))?;
    }
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "failed to move {} into place as {}",
            tmp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn set_env_value(key: &str, value: &str, force: bool) {
    if force || std::env::var_os(key).is_none() {
        set_process_env(key, value);
    }
}

fn set_optional_env_value(key: &str, value: Option<&str>, force: bool) {
    match value {
        Some(value) => set_env_value(key, value, force),
        None if force => clear_process_env(key),
        None => {}
    }
}

fn apply_local_embedding_env(
    model: Option<&vera_core::local_models::LocalEmbeddingModelConfig>,
    force: bool,
) {
    let env_override_present = LOCAL_EMBEDDING_SOURCE_ENV_KEYS
        .iter()
        .any(|key| std::env::var_os(key).is_some());
    if !force && env_override_present {
        return;
    }

    let repo = model.and_then(|model| match &model.source {
        vera_core::local_models::LocalEmbeddingSource::HuggingFace { repo } => Some(repo.as_str()),
        vera_core::local_models::LocalEmbeddingSource::Directory { .. } => None,
    });
    let dir = model.and_then(|model| match &model.source {
        vera_core::local_models::LocalEmbeddingSource::Directory { path } => path.to_str(),
        vera_core::local_models::LocalEmbeddingSource::HuggingFace { .. } => None,
    });

    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_REPO_ENV,
        repo,
        force,
    );
    set_optional_env_value(vera_core::local_models::LOCAL_EMBEDDING_DIR_ENV, dir, force);
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_ONNX_FILE_ENV,
        model.map(|value| value.onnx_file.as_str()),
        force,
    );
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV,
        model.and_then(|value| value.onnx_data_file.as_deref()),
        force,
    );
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_TOKENIZER_FILE_ENV,
        model.map(|value| value.tokenizer_file.as_str()),
        force,
    );
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_DIM_ENV,
        model
            .map(|value| value.embedding_dim.to_string())
            .as_deref(),
        force,
    );
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_POOLING_ENV,
        model.map(|value| value.pooling.to_string()).as_deref(),
        force,
    );
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_MAX_LENGTH_ENV,
        model.map(|value| value.max_length.to_string()).as_deref(),
        force,
    );
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_QUERY_PREFIX_ENV,
        model.and_then(|value| value.query_prefix.as_deref()),
        force,
    );
    if force {
        clear_process_env(vera_core::local_models::LEGACY_EMBEDDING_QUERY_PREFIX_ENV);
    }
}

fn set_process_env(key: &str, value: &str) {
    // Safe because Vera only mutates process environment during single-threaded
    // CLI startup, before any background work or runtime threads are created.
    unsafe {
        std::env::set_var(key, value);
    }
}

fn clear_process_env(key: &str) {
    unsafe {
        std::env::remove_var(key);
    }
}

const LOCAL_EMBEDDING_SOURCE_ENV_KEYS: &[&str] = &[
    vera_core::local_models::LOCAL_EMBEDDING_REPO_ENV,
    vera_core::local_models::LOCAL_EMBEDDING_DIR_ENV,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_config_defaults_are_empty() {
        let config = StoredConfig::default();
        assert!(config.local_mode.is_none());
        assert!(config.backend.is_none());
        assert!(config.install_method.is_none());
        assert!(config.embedding_api.is_none());
        assert!(config.reranker_api.is_none());
        assert!(config.core_config.is_none());
        assert!(config.local_embedding_model.is_none());
    }

    #[test]
    fn stored_secrets_default_empty() {
        let secrets = StoredSecrets::default();
        assert!(secrets.embedding_api_key.is_none());
        assert!(secrets.reranker_api_key.is_none());
    }

    #[test]
    fn install_provenance_defaults_are_empty() {
        let provenance = InstallProvenance::default();
        assert!(provenance.install_method.is_none());
        assert!(provenance.version.is_none());
        assert!(provenance.binary_path.is_none());
    }
}
