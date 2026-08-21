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

/// Load `config.json` verbatim. Every save helper re-reads through here and
/// writes the result back, so anything this function changes is persisted as a
/// side effect of unrelated commands. In particular a repaired `pooling` would
/// be written to disk and then refuse to parse under an older Vera on the same
/// machine, whose `FromStr` only knows `mean` and `cls`. Repairs therefore
/// belong at the points of use, not here.
pub fn load_saved_config() -> Result<StoredConfig> {
    load_json_file(&config_path()?)
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

/// The single point where a stored model config becomes a runtime one.
///
/// Every runtime reader of `local_embedding_model` goes through here, so the
/// repair cannot be forgotten by a future one and silently reinstate the
/// mean-pooled jina config. Nothing here reaches disk; see `load_saved_config`
/// for why the repair must stay out of the load path.
fn repaired_local_embedding_model(
    stored: Option<vera_core::local_models::LocalEmbeddingModelConfig>,
) -> Option<vera_core::local_models::LocalEmbeddingModelConfig> {
    stored.map(vera_core::local_models::LocalEmbeddingModelConfig::repair_stored_defaults)
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

    // Repaired in memory on the way to the process environment; see
    // `repaired_local_embedding_model`.
    let local_embedding_model = repaired_local_embedding_model(config.local_embedding_model);
    apply_local_embedding_env(local_embedding_model.as_ref(), force);

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
    set_optional_env_value(
        vera_core::local_models::LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV,
        model.and_then(|value| value.document_prefix.as_deref()),
        force,
    );
}

fn set_process_env(key: &str, value: &str) {
    // In production this runs only during single-threaded CLI startup, before
    // any background work or runtime threads exist, so no concurrent reader can
    // observe the write.
    //
    // The unit tests below break that condition: libtest runs them on several
    // threads at once. They are sound instead because every test that reads or
    // writes any of `RESTORED_ENV_KEYS` holds `VERA_HOME_LOCK` for its whole
    // body, and nothing else in the test binary touches those variables. Any
    // new test that calls this, `clear_process_env`, or a helper reaching them
    // must take the same lock.
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
    use std::sync::Mutex;

    /// `VERA_HOME` is process-global, so the tests that redirect it at the
    /// config directory must not overlap with each other.
    static VERA_HOME_LOCK: Mutex<()> = Mutex::new(());

    /// What `vera setup` wrote to `config.json` before the pooling fix.
    const LEGACY_JINA_CONFIG: &str = r#"{
      "local_embedding_model": {
        "source": {"source": "hugging-face",
                   "repo": "jinaai/jina-embeddings-v5-text-nano-retrieval"},
        "onnx_file": "onnx/model_quantized.onnx",
        "onnx_data_file": "onnx/model_quantized.onnx_data",
        "tokenizer_file": "tokenizer.json",
        "embedding_dim": 768,
        "pooling": "mean",
        "max_length": 512
      }
    }"#;

    /// What `vera setup --embedding-document-prefix 'Passage:'` writes. The
    /// prefix differs from jina's preset so the preset cannot stand in for it.
    const STORED_DOCUMENT_PREFIX_CONFIG: &str = r#"{
      "local_embedding_model": {
        "source": {"source": "hugging-face",
                   "repo": "jinaai/jina-embeddings-v5-text-nano-retrieval"},
        "onnx_file": "onnx/model_quantized.onnx",
        "onnx_data_file": "onnx/model_quantized.onnx_data",
        "tokenizer_file": "tokenizer.json",
        "embedding_dim": 768,
        "pooling": "last-token",
        "max_length": 512,
        "query_prefix": "Query:",
        "document_prefix": "Passage:"
      }
    }"#;

    /// A stored model that prefixes queries only.
    ///
    /// The repo is jina's so that `defaults_for_source` answers it with a preset
    /// that *does* carry a document prefix. That is not because the resolution
    /// consults it (the test pins the arm actually taken), but because a preset
    /// prefix is what gives the `None` assertion something to fail against:
    /// delete the explicit-model short-circuit and jina's
    /// `Document:` surfaces. An unrecognised repo would fall to
    /// `generic_defaults`, whose document prefix is `None` on both sides of that
    /// change, so nothing could distinguish them.
    const STORED_QUERY_PREFIX_ONLY_CONFIG: &str = r#"{
      "local_embedding_model": {
        "source": {"source": "hugging-face",
                   "repo": "jinaai/jina-embeddings-v5-text-nano-retrieval"},
        "onnx_file": "onnx/model_quantized.onnx",
        "tokenizer_file": "tokenizer.json",
        "embedding_dim": 384,
        "pooling": "cls",
        "max_length": 256,
        "query_prefix": "Ask:"
      }
    }"#;

    /// Everything `apply_saved_env_impl` can write, plus the redirect itself.
    const RESTORED_ENV_KEYS: &[&str] = &[
        "VERA_HOME",
        "VERA_BACKEND",
        "VERA_LOCAL",
        "EMBEDDING_MODEL_BASE_URL",
        "EMBEDDING_MODEL_ID",
        "EMBEDDING_MODEL_API_KEY",
        "RERANKER_MODEL_BASE_URL",
        "RERANKER_MODEL_ID",
        "RERANKER_MODEL_API_KEY",
        vera_core::local_models::LOCAL_EMBEDDING_REPO_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_DIR_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_ONNX_FILE_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_ONNX_DATA_FILE_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_TOKENIZER_FILE_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_DIM_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_POOLING_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_MAX_LENGTH_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_QUERY_PREFIX_ENV,
        vera_core::local_models::LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV,
        vera_core::local_models::LEGACY_EMBEDDING_QUERY_PREFIX_ENV,
    ];

    struct VeraHomeGuard {
        _dir: tempfile::TempDir,
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Vec<Option<std::ffi::OsString>>,
    }

    impl Drop for VeraHomeGuard {
        fn drop(&mut self) {
            for (key, value) in RESTORED_ENV_KEYS.iter().zip(&self.previous) {
                match value {
                    Some(value) => set_process_env(key, &value.to_string_lossy()),
                    None => clear_process_env(key),
                }
            }
        }
    }

    /// Point `VERA_HOME` at a temp dir seeded with `contents` so no test can
    /// reach the developer's real `~/.vera/config.json`.
    fn with_stored_config(contents: &str) -> VeraHomeGuard {
        let lock = VERA_HOME_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = RESTORED_ENV_KEYS.iter().map(std::env::var_os).collect();
        let dir = tempfile::tempdir().unwrap();
        set_process_env("VERA_HOME", dir.path().to_str().unwrap());
        fs::write(config_path().unwrap(), contents).unwrap();
        VeraHomeGuard {
            _dir: dir,
            _lock: lock,
            previous,
        }
    }

    fn stored_pooling_on_disk() -> String {
        let raw = fs::read(config_path().unwrap()).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        value["local_embedding_model"]["pooling"]
            .as_str()
            .expect("stored config should still carry a pooling field")
            .to_string()
    }

    #[test]
    fn unrelated_save_does_not_rewrite_stored_pooling() {
        let _guard = with_stored_config(LEGACY_JINA_CONFIG);
        assert_eq!(stored_pooling_on_disk(), "mean");

        // Every save helper is a load-mutate-save cycle over the same struct,
        // so a repair applied at load time would be persisted from here.
        save_backend(vera_core::config::InferenceBackend::OnnxJina(
            vera_core::config::OnnxExecutionProvider::Cpu,
        ))
        .unwrap();

        // A Vera older than `last-token` still has to be able to parse this
        // file; its `FromStr` accepts only `mean` and `cls`.
        assert_eq!(stored_pooling_on_disk(), "mean");
    }

    #[test]
    fn repair_command_does_not_rewrite_stored_pooling() {
        let _guard = with_stored_config(LEGACY_JINA_CONFIG);
        assert_eq!(stored_pooling_on_disk(), "mean");

        // `vera repair` resolves an embedding model and hands it to
        // `setup::configure_backend`, which persists it verbatim. Sourcing it
        // from the repaired accessor put `last-token` in the file and bricked
        // an older Vera installed alongside — and unlike `vera setup`, the user
        // never picked a pooling mode here.
        let model = crate::commands::repair::embedding_model_to_persist(
            vera_core::config::InferenceBackend::OnnxJina(
                vera_core::config::OnnxExecutionProvider::Cpu,
            ),
        )
        .unwrap()
        .expect("an ONNX backend always carries an embedding model");
        save_local_embedding_model(&model).unwrap();

        assert_eq!(stored_pooling_on_disk(), "mean");
    }

    #[test]
    fn runtime_readers_repair_stored_pooling_in_memory() {
        let _guard = with_stored_config(LEGACY_JINA_CONFIG);

        let model =
            repaired_local_embedding_model(load_saved_config().unwrap().local_embedding_model)
                .expect("stored config carries a local embedding model");
        assert_eq!(
            model.pooling,
            vera_core::local_models::LocalEmbeddingPooling::LastToken
        );

        apply_saved_env_force().unwrap();
        assert_eq!(
            std::env::var(vera_core::local_models::LOCAL_EMBEDDING_POOLING_ENV).unwrap(),
            "last-token"
        );

        assert_eq!(stored_pooling_on_disk(), "mean");
    }

    #[test]
    fn stored_document_prefix_reaches_the_env_config() {
        let _guard = with_stored_config(STORED_DOCUMENT_PREFIX_CONFIG);

        apply_saved_env_force().unwrap();

        // `from_env` is the only reader the embedding pipeline has, so a stored
        // prefix that never reaches the environment is a dropped flag.
        let model = vera_core::local_models::LocalEmbeddingModelConfig::from_env().unwrap();
        assert_eq!(model.document_prefix.as_deref(), Some("Passage:"));
    }

    #[test]
    fn a_stored_config_without_a_document_prefix_clears_a_stale_one() {
        let _guard = with_stored_config(STORED_QUERY_PREFIX_ONLY_CONFIG);
        set_process_env(
            vera_core::local_models::LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV,
            "Stale-Passage-Marker:",
        );

        apply_saved_env_force().unwrap();

        // Forcing the stored config over the environment has to remove a
        // document prefix the stored config does not have, or an inherited one
        // would keep prefixing passages the stored model never asked for.
        assert!(
            std::env::var_os(vera_core::local_models::LOCAL_EMBEDDING_DOCUMENT_PREFIX_ENV)
                .is_none()
        );
        // Clearing one side must not clear the other.
        assert_eq!(
            std::env::var(vera_core::local_models::LOCAL_EMBEDDING_QUERY_PREFIX_ENV).unwrap(),
            "Ask:"
        );

        // The force path exports a source and an onnx file together, which is
        // exactly the pair `model_source_and_onnx_file_are_set` tests, so
        // `explicit_model_env` is always on downstream of it. That makes
        // `resolve_optional_env_value` take its `None if explicit_model_env`
        // arm; the `None => default` arm is unreachable from this entry point,
        // so no fixture stored through `config.json` can exercise it.
        assert!(
            std::env::var_os(vera_core::local_models::LOCAL_EMBEDDING_REPO_ENV).is_some()
                && std::env::var_os(vera_core::local_models::LOCAL_EMBEDDING_ONNX_FILE_ENV)
                    .is_some(),
            "the force path is supposed to make the model explicit through the environment"
        );

        let model = vera_core::local_models::LocalEmbeddingModelConfig::from_env().unwrap();
        assert_eq!(model.query_prefix.as_deref(), Some("Ask:"));
        // What is left to pin is that arm returning nothing rather than the
        // preset's prefix. There is a preset prefix to return, so `None` below
        // is a declined default and not an absent one.
        assert!(
            vera_core::local_models::LocalEmbeddingModelConfig::jina()
                .document_prefix
                .is_some(),
            "the fixture's repo must have a preset document prefix, or `None` below proves nothing"
        );
        assert_eq!(model.document_prefix, None);
    }

    /// The opt-out has to survive the file and the environment, not just the
    /// struct. It used to be filtered out on the way to `config.json`, whose
    /// missing key then let jina's preset reinstate the prefix on the next run,
    /// so disabling a prefix lasted exactly one invocation.
    #[test]
    fn an_explicitly_emptied_prefix_survives_a_save_and_reload() {
        let _guard = with_stored_config("{}");

        let mut model = vera_core::local_models::LocalEmbeddingModelConfig::jina();
        model.query_prefix = Some(String::new());
        model.document_prefix = Some(String::new());
        save_local_embedding_model(&model).unwrap();

        // The empty value has to reach the file; a skipped key is what the
        // preset fills back in.
        let raw = fs::read(config_path().unwrap()).unwrap();
        let stored: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(stored["local_embedding_model"]["query_prefix"], "");
        assert_eq!(stored["local_embedding_model"]["document_prefix"], "");

        apply_saved_env_force().unwrap();

        let reloaded = vera_core::local_models::LocalEmbeddingModelConfig::from_env().unwrap();
        assert_eq!(
            reloaded.query_prefix, None,
            "jina's preset query prefix came back after an explicit opt-out"
        );
        assert_eq!(
            reloaded.document_prefix, None,
            "jina's preset document prefix came back after an explicit opt-out"
        );
        assert_eq!(reloaded.query_text("find main"), "find main");
        assert_eq!(reloaded.document_text("fn main() {}"), "fn main() {}");
    }

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
