//! `vera setup` — persist a preferred Vera mode and bootstrap first-run state.

use std::io::IsTerminal;

use anyhow::{Context, bail};
use serde::Serialize;
use vera_core::config::{InferenceBackend, OnnxExecutionProvider};
use vera_core::local_models::{
    LocalEmbeddingModelConfig, LocalEmbeddingPooling, normalize_huggingface_repo,
};

use crate::commands;
use crate::helpers::LocalEmbeddingModelFlags;
use crate::state::{self, ApiSetupInput};

#[derive(Debug, Serialize)]
pub(crate) struct SetupReport {
    mode: String,
    config_path: String,
    credentials_path: String,
    models_prefetched: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    onnx_runtime_ready: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_embedding_model: Option<String>,
    indexed_path: Option<String>,
}

/// Remedy shown when a non-interactive invocation cannot prompt.
const NON_INTERACTIVE_HINT: &str = "Hint: pass a backend flag (for example `--onnx-jina-coreml`, \
     `--onnx-jina-cuda`, or `--potion-code`), `--yes` to auto-detect one, or `--api` with \
     EMBEDDING_MODEL_BASE_URL, EMBEDDING_MODEL_ID, and EMBEDDING_MODEL_API_KEY set.";

/// `backend`: Some(local backend) for local, None + api=true for API, None + api=false defaults to auto-detect.
/// `allow_wizard`: bare interactive invocations run the full wizard only for
/// `vera setup`; `vera backend` always stays in the backend-only flow.
pub fn run(
    backend: Option<InferenceBackend>,
    api: bool,
    index_path: Option<String>,
    json_output: bool,
    yes: bool,
    embedding_flags: LocalEmbeddingModelFlags,
    allow_wizard: bool,
) -> anyhow::Result<()> {
    // Prompts need a terminal. Without one, every `cliclack` call fails with a
    // bare "not connected", so decide up front what can run unattended.
    let interactive = std::io::stdin().is_terminal();

    // If no flags at all and interactive, run the full wizard.
    let is_bare_interactive =
        !api && backend.is_none() && !json_output && !yes && !embedding_flags.any_set();
    if allow_wizard && is_bare_interactive && index_path.is_none() {
        if !interactive {
            bail!(
                "`vera setup` with no flags runs an interactive wizard, and no terminal is \
                 available for prompts.\n{NON_INTERACTIVE_HINT}"
            );
        }
        return run_wizard();
    }

    // Resolve: explicit backend flag wins, then --api, then auto-detect.
    let effective_backend = if api {
        InferenceBackend::Api
    } else if let Some(b) = backend {
        b
    } else if json_output || yes {
        let detected = detect_gpu();
        if !json_output {
            eprintln!("Auto-detected backend: {detected}. Use a backend flag to override.");
        }
        detected
    } else if !interactive {
        bail!(
            "no backend selected and no terminal is available for prompts.\n{NON_INTERACTIVE_HINT}"
        );
    } else {
        prompt_backend()?
    };
    if !effective_backend.is_onnx() && embedding_flags.any_set() {
        bail!("custom local embedding flags can only be used with local ONNX backends");
    }
    let local_embedding_model = effective_backend
        .is_onnx()
        .then(|| resolve_local_embedding_model(&embedding_flags))
        .transpose()?;

    // An explicit backend flag already answers everything the confirmation asks
    // about, so a non-interactive run has nothing left to confirm.
    if !yes
        && !json_output
        && interactive
        && !confirm(
            &effective_backend,
            local_embedding_model.as_ref(),
            index_path.as_deref(),
        )?
    {
        if !json_output {
            println!("Cancelled.");
        }
        return Ok(());
    }

    let needs_api_prompt = should_prompt_api_config(effective_backend, json_output, yes);
    if needs_api_prompt && !interactive {
        // With complete credentials in the environment there is nothing left
        // to prompt for; on a missing variable this error names it.
        read_required_api_env(
            "EMBEDDING_MODEL_BASE_URL",
            "EMBEDDING_MODEL_ID",
            "EMBEDDING_MODEL_API_KEY",
        )
        .context("API mode needs endpoint credentials and no terminal is available for prompts")?;
    }
    let api_setup = (needs_api_prompt && interactive)
        .then(prompt_api_setup)
        .transpose()?;

    configure_backend_with_api_setup(
        effective_backend,
        local_embedding_model,
        index_path,
        json_output,
        "Vera setup complete.",
        api_setup,
    )
}

/// Full interactive setup wizard: backend, agent skills, optional indexing.
fn run_wizard() -> anyhow::Result<()> {
    cliclack::intro("vera setup")?;

    // Step 1: Backend selection
    cliclack::log::step("Step 1: Backend")?;
    let effective_backend = prompt_backend_select()?;
    let local_embedding_model = effective_backend
        .is_onnx()
        .then(LocalEmbeddingModelConfig::default);

    if effective_backend == InferenceBackend::Api {
        configure_api_interactive()?;
    } else {
        configure_backend(
            effective_backend,
            local_embedding_model,
            None,
            false,
            "Backend configured.",
        )?;
    }

    // Step 2: Agent skill installation
    cliclack::log::step("Step 2: Agent skills")?;
    let install_skills: bool = cliclack::confirm("Install Vera skills for coding agents?")
        .initial_value(true)
        .interact()?;
    if install_skills {
        commands::agent::run(commands::agent::AgentCommand::Install, None, None, false)?;
    }

    // Step 3: Optional indexing
    cliclack::log::step("Step 3: Index a project")?;
    let index_now: bool = cliclack::confirm("Index a project now?")
        .initial_value(false)
        .interact()?;
    if index_now {
        let path: String = cliclack::input("Project path")
            .default_input(".")
            .interact()?;
        commands::index::execute(
            path.trim(),
            false,
            effective_backend,
            Vec::new(),
            false,
            false,
            false,
            false,
        )?;
    }

    cliclack::outro("Setup complete! Run `vera search \"query\"` to get started.")?;
    Ok(())
}

pub(crate) fn configure_backend(
    effective_backend: InferenceBackend,
    local_embedding_model: Option<LocalEmbeddingModelConfig>,
    index_path: Option<String>,
    json_output: bool,
    success_header: &str,
) -> anyhow::Result<()> {
    configure_backend_with_api_setup(
        effective_backend,
        local_embedding_model,
        index_path,
        json_output,
        success_header,
        None,
    )
}

fn configure_backend_with_api_setup(
    effective_backend: InferenceBackend,
    local_embedding_model: Option<LocalEmbeddingModelConfig>,
    index_path: Option<String>,
    json_output: bool,
    success_header: &str,
    api_setup: Option<(ApiSetupInput, Option<ApiSetupInput>)>,
) -> anyhow::Result<()> {
    let use_local = effective_backend.is_local();
    let mut models_prefetched = 0usize;
    let onnx_runtime_ready;
    let mut local_embedding_summary = None;

    match effective_backend {
        InferenceBackend::OnnxJina(ep) => {
            let local_embedding_model = local_embedding_model.unwrap_or_default();
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;
            let prefetched = rt.block_on(vera_core::local_models::prepare_local_models_for_ep(
                ep,
                &local_embedding_model,
            ))?;
            models_prefetched = prefetched.len();
            // Use the downloaded library path (first prefetched file) for the readiness check.
            onnx_runtime_ready = Some(
                vera_core::local_models::ensure_ort_runtime(
                    prefetched.first().map(|p| p.as_path()),
                )
                .is_ok(),
            );
            state::save_backend(effective_backend)?;
            state::save_local_embedding_model(&local_embedding_model)?;
            state::clear_reranker_setup()?;
            state::apply_saved_env_force()?;
            local_embedding_summary = Some(local_embedding_model.display_name());
        }
        InferenceBackend::PotionCode => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;
            rt.block_on(vera_core::local_models::ensure_potion_code_assets())?;
            models_prefetched = vera_core::local_models::inspect_potion_code_model_files()?.len();
            onnx_runtime_ready = None;
            state::save_backend(effective_backend)?;
            state::clear_reranker_setup()?;
            state::apply_saved_env_force()?;
            local_embedding_summary =
                Some(vera_core::local_models::potion_code_model_name().to_string());
        }
        InferenceBackend::Api => {
            let (embedding, reranker) = match api_setup {
                Some((embedding, reranker)) => (embedding, reranker),
                None => (
                    read_required_api_env(
                        "EMBEDDING_MODEL_BASE_URL",
                        "EMBEDDING_MODEL_ID",
                        "EMBEDDING_MODEL_API_KEY",
                    )?,
                    read_optional_api_env(
                        "RERANKER_MODEL_BASE_URL",
                        "RERANKER_MODEL_ID",
                        "RERANKER_MODEL_API_KEY",
                    )?,
                ),
            };
            state::save_api_setup(&embedding, reranker.as_ref())?;
            state::apply_saved_env_force()?;
            onnx_runtime_ready = None;
        }
    }

    if state::load_saved_config()?.install_method.is_none() {
        if let Some(install_method) = crate::update_check::resolve_install_method().install_method {
            state::save_install_method(Some(&install_method))?;
        }
    }

    if let Some(path) = index_path.as_deref() {
        commands::index::execute(
            path,
            json_output,
            effective_backend,
            Vec::new(),
            false,
            false,
            false,
            false,
        )?;
    }

    let report = SetupReport {
        mode: effective_backend.to_string(),
        config_path: state::config_path()?.display().to_string(),
        credentials_path: state::credentials_path()?.display().to_string(),
        models_prefetched,
        onnx_runtime_ready,
        local_embedding_model: local_embedding_summary,
        indexed_path: index_path,
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{success_header}");
        println!();
        println!("  Mode:                 {}", report.mode);
        println!("  Config:               {}", report.config_path);
        println!("  Credentials:          {}", report.credentials_path);
        if use_local {
            if let Some(model) = report.local_embedding_model.as_deref() {
                println!("  Embedding model:      {model}");
            }
            println!("  Prefetched model files: {}", report.models_prefetched);
            if let Some(ready) = report.onnx_runtime_ready {
                println!(
                    "  ONNX Runtime ready:   {}",
                    if ready { "yes" } else { "no" }
                );
            }
        }
        if let Some(path) = report.indexed_path.as_deref() {
            println!("  Indexed path:         {path}");
        }
        if effective_backend == InferenceBackend::Api {
            println!();
            println!("  Your API settings are saved in the config file above.");
            println!("  You can remove any EMBEDDING_MODEL_* / RERANKER_MODEL_* env vars");
            println!("  from your shell. Vera reads from the config file at runtime.");
        }
    }

    Ok(())
}

fn should_prompt_api_config(
    effective_backend: InferenceBackend,
    json_output: bool,
    yes: bool,
) -> bool {
    effective_backend == InferenceBackend::Api && !json_output && !yes
}

/// Probe the system for a usable GPU and return the best local backend.
/// Falls back to Potion Code on CPU if nothing is detected.
fn detect_gpu() -> InferenceBackend {
    // NVIDIA: check for nvidia-smi or vendor ID (0x10de) in sysfs
    let has_nvidia = std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
        || (cfg!(target_os = "linux")
            && std::process::Command::new("sh")
                .args([
                    "-c",
                    "grep -rql 0x10de /sys/class/drm/*/device/vendor 2>/dev/null",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success()));
    if has_nvidia {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda);
    }

    // Apple Silicon: macOS + aarch64
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl);
    }

    // AMD ROCm: check for rocminfo (Linux only)
    if cfg!(target_os = "linux")
        && std::process::Command::new("rocminfo")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm);
    }

    // Intel OpenVINO: check for Intel GPU via vendor ID (0x8086) in sysfs
    if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        let has_intel_gpu = std::process::Command::new("sh")
            .args([
                "-c",
                "grep -rql 0x8086 /sys/class/drm/*/device/vendor 2>/dev/null",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if has_intel_gpu {
            return InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino);
        }
    }

    // Windows: DirectML works with any DirectX 12 GPU
    if cfg!(target_os = "windows") {
        return InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl);
    }

    InferenceBackend::PotionCode
}

/// Show an interactive backend selection menu. Auto-detect is the default.
fn prompt_backend() -> anyhow::Result<InferenceBackend> {
    cliclack::intro("vera backend")?;
    let backend = prompt_backend_select()?;
    Ok(backend)
}

/// Backend selection menu items (no intro/outro, for embedding in wizards).
fn prompt_backend_select() -> anyhow::Result<InferenceBackend> {
    let detected = detect_gpu();
    let detected_hint = match detected {
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda) => "NVIDIA GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl) => "Apple Silicon detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm) => "AMD GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino) => "Intel GPU detected",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl) => "DirectX 12 GPU assumed",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu) => "Jina ONNX CPU selected",
        InferenceBackend::PotionCode => "no GPU detected, will use Potion CPU",
        InferenceBackend::Api => "API mode",
    };

    let backend: InferenceBackend = cliclack::select("Select a backend")
        .item(
            detected,
            format!("Auto-detect ({detected_hint})"),
            "recommended",
        )
        .item(
            InferenceBackend::Api,
            "API mode",
            "remote OpenAI-compatible endpoints",
        )
        .item(
            InferenceBackend::PotionCode,
            "Potion Code CPU",
            "CPU-only machines",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda),
            "CUDA",
            "NVIDIA GPU",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm),
            "ROCm",
            "AMD GPU, Linux",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl),
            "CoreML",
            "Apple Silicon, macOS",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino),
            "OpenVINO",
            "Intel GPU/iGPU, Linux",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl),
            "DirectML",
            "DirectX 12 GPU, Windows",
        )
        .item(
            InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu),
            "Jina ONNX CPU",
            "compatibility path",
        )
        .interact()?;

    Ok(backend)
}

fn resolve_local_embedding_model(
    flags: &LocalEmbeddingModelFlags,
) -> anyhow::Result<LocalEmbeddingModelConfig> {
    let mut model = if flags.code_rank_embed {
        LocalEmbeddingModelConfig::coderankembed()
    } else if let Some(repo_or_url) = flags.embedding_repo.as_deref() {
        LocalEmbeddingModelConfig::from_huggingface_repo(normalize_huggingface_repo(repo_or_url)?)
    } else if let Some(dir) = flags.embedding_dir.as_deref() {
        let path = std::path::Path::new(dir)
            .canonicalize()
            .with_context(|| format!("failed to resolve embedding directory: {dir}"))?;
        LocalEmbeddingModelConfig::from_directory(path)
    } else {
        LocalEmbeddingModelConfig::default()
    };

    if let Some(onnx_file) = flags.embedding_onnx_file.as_ref() {
        model.onnx_file = onnx_file.clone();
    }
    if flags.embedding_no_onnx_data {
        model.onnx_data_file = None;
    } else if let Some(onnx_data_file) = flags.embedding_onnx_data_file.as_ref() {
        model.onnx_data_file = Some(onnx_data_file.clone());
    }
    if let Some(tokenizer_file) = flags.embedding_tokenizer_file.as_ref() {
        model.tokenizer_file = tokenizer_file.clone();
    }
    if let Some(dim) = flags.embedding_dim {
        model.embedding_dim = dim;
    }
    if let Some(pooling) = flags.embedding_pooling.as_deref() {
        model.pooling = pooling
            .parse::<LocalEmbeddingPooling>()
            .map_err(anyhow::Error::msg)?;
    }
    if let Some(max_length) = flags.embedding_max_length {
        model.max_length = max_length;
    }
    // An empty value is kept rather than collapsed to `None`: it is the only
    // way to turn a preset's prefix off, and `None` would be restored from the
    // preset on the next run. Everything downstream treats an empty prefix as
    // no prefix.
    if let Some(document_prefix) = flags.embedding_document_prefix.as_ref() {
        model.document_prefix = Some(document_prefix.trim().to_string());
    }
    if let Some(query_prefix) = flags.embedding_query_prefix.as_ref() {
        model.query_prefix = Some(query_prefix.trim().to_string());
    }

    Ok(model)
}

fn confirm(
    backend: &InferenceBackend,
    local_embedding_model: Option<&LocalEmbeddingModelConfig>,
    index_path: Option<&str>,
) -> anyhow::Result<bool> {
    let mut msg = format!("Configure Vera for {backend} mode");
    if let Some(model) = local_embedding_model {
        msg.push_str(&format!(", embedding model: {}", model.display_name()));
    }
    if let Some(path) = index_path {
        msg.push_str(&format!(", then index: {path}"));
    }
    msg.push('?');
    let yes: bool = cliclack::confirm(msg).interact()?;
    Ok(yes)
}

/// Interactive API configuration for the setup wizard.
/// Offers common provider presets and prompts for credentials.
fn configure_api_interactive() -> anyhow::Result<()> {
    let (embedding, reranker) = prompt_api_setup()?;

    state::save_backend(InferenceBackend::Api)?;
    state::save_api_setup(&embedding, reranker.as_ref())?;
    state::apply_saved_env_force()?;

    if state::load_saved_config()?.install_method.is_none() {
        if let Some(install_method) = crate::update_check::resolve_install_method().install_method {
            state::save_install_method(Some(&install_method))?;
        }
    }

    cliclack::log::success("API backend configured.")?;
    cliclack::log::info(
        "Your credentials are saved in Vera's config directory. You can remove any \
         EMBEDDING_MODEL_* / RERANKER_MODEL_* env vars from your shell.",
    )?;
    Ok(())
}

fn prompt_api_setup() -> anyhow::Result<(ApiSetupInput, Option<ApiSetupInput>)> {
    #[derive(Clone)]
    struct ApiPreset {
        label: &'static str,
        hint: &'static str,
        embedding_base_url: &'static str,
        embedding_model: &'static str,
        reranker_base_url: &'static str,
        reranker_model: &'static str,
    }

    let presets = [
        ApiPreset {
            label: "OpenAI",
            hint: "text-embedding-3-small, no built-in reranker",
            embedding_base_url: "https://api.openai.com/v1",
            embedding_model: "text-embedding-3-small",
            reranker_base_url: "",
            reranker_model: "",
        },
        ApiPreset {
            label: "Jina AI",
            hint: "jina-embeddings-v3 + jina-reranker-v2",
            embedding_base_url: "https://api.jina.ai/v1",
            embedding_model: "jina-embeddings-v3",
            reranker_base_url: "https://api.jina.ai/v1",
            reranker_model: "jina-reranker-v2-base-multilingual",
        },
        ApiPreset {
            label: "Voyage AI",
            hint: "voyage-code-3 + rerank-2",
            embedding_base_url: "https://api.voyageai.com/v1",
            embedding_model: "voyage-code-3",
            reranker_base_url: "https://api.voyageai.com/v1",
            reranker_model: "rerank-2",
        },
        ApiPreset {
            label: "Custom",
            hint: "enter your own OpenAI-compatible endpoints",
            embedding_base_url: "",
            embedding_model: "",
            reranker_base_url: "",
            reranker_model: "",
        },
    ];

    let mut select = cliclack::select("Select an API provider");
    for (i, p) in presets.iter().enumerate() {
        select = select.item(i, p.label, p.hint);
    }
    let choice: usize = select.interact()?;
    let preset = &presets[choice];

    // Embedding base URL
    let embedding_base_url = prompt_required_input(
        "Embedding API base URL",
        (!preset.embedding_base_url.is_empty()).then_some(preset.embedding_base_url),
        "https://api.openai.com/v1",
    )?;

    // Embedding model
    let embedding_model = prompt_required_input(
        "Embedding model ID",
        (!preset.embedding_model.is_empty()).then_some(preset.embedding_model),
        "text-embedding-3-small",
    )?;

    // Embedding API key
    let embedding_api_key: String = cliclack::password("Embedding API key")
        .mask('▪')
        .interact()?;
    if embedding_api_key.is_empty() {
        bail!("embedding API key is required");
    }

    let embedding = ApiSetupInput {
        base_url: embedding_base_url,
        model_id: embedding_model,
        api_key: embedding_api_key.clone(),
    };

    // Reranker (optional)
    let setup_reranker: bool =
        cliclack::confirm("Configure a reranker? (improves search precision)")
            .initial_value(!preset.reranker_base_url.is_empty())
            .interact()?;

    let reranker = if setup_reranker {
        let reranker_base_url = prompt_required_input(
            "Reranker API base URL",
            (!preset.reranker_base_url.is_empty()).then_some(preset.reranker_base_url),
            "https://api.jina.ai/v1",
        )?;

        let reranker_model = prompt_required_input(
            "Reranker model ID",
            (!preset.reranker_model.is_empty()).then_some(preset.reranker_model),
            "jina-reranker-v2-base-multilingual",
        )?;

        let reranker_api_key: String =
            cliclack::password("Reranker API key (Enter to reuse embedding key)")
                .mask('▪')
                .interact()?;
        let reranker_api_key = if reranker_api_key.is_empty() {
            embedding_api_key
        } else {
            reranker_api_key
        };

        Some(ApiSetupInput {
            base_url: reranker_base_url,
            model_id: reranker_model,
            api_key: reranker_api_key,
        })
    } else {
        None
    };

    Ok((embedding, reranker))
}

fn prompt_required_input(
    label: &str,
    default: Option<&str>,
    placeholder: &str,
) -> anyhow::Result<String> {
    let value: String = if let Some(default) = default {
        cliclack::input(label).default_input(default).interact()?
    } else {
        cliclack::input(label).placeholder(placeholder).interact()?
    };
    let value = value.trim().to_string();
    if value.is_empty() {
        bail!("{} is required", label.to_ascii_lowercase());
    }
    Ok(value)
}

/// Read an env var, treating unset and empty/whitespace-only values the same.
fn read_api_env_var(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn read_required_api_env(
    base_key: &str,
    model_key: &str,
    api_key_key: &str,
) -> anyhow::Result<ApiSetupInput> {
    Ok(ApiSetupInput {
        base_url: read_api_env_var(base_key).with_context(|| {
            format!("{base_key} must be set and non-empty for `vera setup --api`")
        })?,
        model_id: read_api_env_var(model_key).with_context(|| {
            format!("{model_key} must be set and non-empty for `vera setup --api`")
        })?,
        api_key: read_api_env_var(api_key_key).with_context(|| {
            format!("{api_key_key} must be set and non-empty for `vera setup --api`")
        })?,
    })
}

fn read_optional_api_env(
    base_key: &str,
    model_key: &str,
    api_key_key: &str,
) -> anyhow::Result<Option<ApiSetupInput>> {
    let base = read_api_env_var(base_key);
    let model = read_api_env_var(model_key);
    let api_key = read_api_env_var(api_key_key);

    match (base, model, api_key) {
        (Some(base_url), Some(model_id), Some(api_key)) => Ok(Some(ApiSetupInput {
            base_url,
            model_id,
            api_key,
        })),
        (None, None, None) => Ok(None),
        _ => bail!(
            "reranker config is incomplete. Set all of {base_key}, {model_key}, and {api_key_key}, or leave all three unset."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_interactive_api_setup_prompts_for_config() {
        assert!(should_prompt_api_config(
            InferenceBackend::Api,
            false,
            false
        ));
    }

    #[test]
    fn noninteractive_api_setup_uses_environment_config() {
        assert!(!should_prompt_api_config(
            InferenceBackend::Api,
            true,
            false
        ));
        assert!(!should_prompt_api_config(
            InferenceBackend::Api,
            false,
            true
        ));
        assert!(!should_prompt_api_config(
            InferenceBackend::PotionCode,
            false,
            false
        ));
    }

    /// `--embedding-query-prefix ""` is the only way to turn a preset's prefix
    /// off. Collapsing it to `None` made `config.json` omit the key entirely,
    /// and the preset put jina's prefix back on the very next run.
    #[test]
    fn an_explicitly_emptied_prefix_flag_is_not_collapsed_to_absent() {
        let flags = LocalEmbeddingModelFlags {
            embedding_query_prefix: Some(String::new()),
            embedding_document_prefix: Some("   ".to_string()),
            ..LocalEmbeddingModelFlags::default()
        };

        let model = resolve_local_embedding_model(&flags).unwrap();

        assert_eq!(
            model.query_prefix.as_deref(),
            Some(""),
            "the flag defaulted back to jina's preset query prefix"
        );
        assert_eq!(
            model.document_prefix.as_deref(),
            Some(""),
            "the flag defaulted back to jina's preset document prefix"
        );
        // Kept, but still nothing: an empty prefix embeds the text unchanged.
        assert_eq!(model.query_text("find main"), "find main");
        assert_eq!(model.document_text("fn main() {}"), "fn main() {}");
    }
}
