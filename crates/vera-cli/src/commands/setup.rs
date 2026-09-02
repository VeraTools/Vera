//! `vera setup` — persist a preferred Vera mode and bootstrap first-run state.

use std::io::IsTerminal;

use anyhow::{Context, bail};
use serde::Serialize;
use vera_core::config::{InferenceBackend, OnnxExecutionProvider, RerankerProtocol};
use vera_core::local_models::{
    LocalEmbeddingModelConfig, LocalEmbeddingPooling, normalize_huggingface_repo,
};

use crate::commands;
use crate::helpers::LocalEmbeddingModelFlags;
use crate::state::{self, ApiSetupInput};

/// Stable identifier for API presets. Qwen-specific optimizations
/// (single shared key and auto Generic protocol) key on this discriminant
/// instead of display strings or model_id/base_url equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApiPresetId {
    OpenAi,
    Jina,
    Voyage,
    Qwen,
    Custom,
}

impl ApiPresetId {
    pub(crate) fn is_qwen(self) -> bool {
        self == Self::Qwen
    }
}

pub(crate) fn preset_uses_single_shared_key(id: ApiPresetId) -> bool {
    id.is_qwen()
}

pub(crate) fn preset_uses_auto_generic(id: ApiPresetId) -> bool {
    id.is_qwen()
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ApiPreset {
    pub(crate) id: ApiPresetId,
    pub(crate) label: &'static str,
    pub(crate) hint: &'static str,
    pub(crate) embedding_base_url: &'static str,
    pub(crate) embedding_model: &'static str,
    pub(crate) reranker_base_url: &'static str,
    pub(crate) reranker_model: &'static str,
}

pub(crate) const API_PRESETS: &[ApiPreset] = &[
    ApiPreset {
        id: ApiPresetId::OpenAi,
        label: "OpenAI",
        hint: "text-embedding-3-small, no built-in reranker",
        embedding_base_url: "https://api.openai.com/v1",
        embedding_model: "text-embedding-3-small",
        reranker_base_url: "",
        reranker_model: "",
    },
    ApiPreset {
        id: ApiPresetId::Jina,
        label: "Jina AI",
        hint: "jina-embeddings-v3 + jina-reranker-v2",
        embedding_base_url: "https://api.jina.ai/v1",
        embedding_model: "jina-embeddings-v3",
        reranker_base_url: "https://api.jina.ai/v1",
        reranker_model: "jina-reranker-v2-base-multilingual",
    },
    ApiPreset {
        id: ApiPresetId::Voyage,
        label: "Voyage AI",
        hint: "voyage-code-3 + rerank-2",
        embedding_base_url: "https://api.voyageai.com/v1",
        embedding_model: "voyage-code-3",
        reranker_base_url: "https://api.voyageai.com/v1",
        reranker_model: "rerank-2",
    },
    ApiPreset {
        id: ApiPresetId::Qwen,
        label: "Qwen (OpenRouter)",
        hint: "qwen3-embedding-8b + qwen3-reranker-8b via OpenRouter (paid usage)",
        embedding_base_url: "https://openrouter.ai/api/v1",
        embedding_model: "qwen/qwen3-embedding-8b",
        reranker_base_url: "https://openrouter.ai/api/v1",
        reranker_model: "qwen/qwen3-reranker-8b",
    },
    ApiPreset {
        id: ApiPresetId::Custom,
        label: "Custom",
        hint: "enter your own OpenAI-compatible endpoints",
        embedding_base_url: "",
        embedding_model: "",
        reranker_base_url: "",
        reranker_model: "",
    },
];

#[allow(dead_code)] // exercised by the preset-identity round-trip test
pub(crate) fn preset_by_id(id: ApiPresetId) -> &'static ApiPreset {
    API_PRESETS
        .iter()
        .find(|preset| preset.id == id)
        .expect("preset id must exist in API_PRESETS")
}

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
const NON_INTERACTIVE_HINT: &str = "Hint: pass `--yes` for the default Potion Code backend, a GPU flag \
     (for example `--onnx-jina-cuda` or `--onnx-jina-coreml`), or `--api` with \
     EMBEDDING_MODEL_BASE_URL, EMBEDDING_MODEL_ID, and EMBEDDING_MODEL_API_KEY set, \
     or use `vera config set` for non-interactive configuration.";

/// The backend `vera setup` and `vera backend` pick when no flag is given.
///
/// Potion Code runs on any CPU with a ~50 MB download and no ONNX runtime, so
/// it is the default on every machine. GPU ONNX backends stay available
/// through flags and the interactive menu's auto-detect entry.
pub(crate) fn default_setup_backend() -> InferenceBackend {
    InferenceBackend::PotionCode
}

/// `backend`: Some(local backend) for local, None + api=true for API, None + api=false uses the default.
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

    // Resolve: explicit backend flag wins, then --api, then the default.
    let effective_backend = if api {
        InferenceBackend::Api
    } else if let Some(b) = backend {
        b
    } else if json_output || yes {
        if !json_output {
            eprintln!(
                "Using default backend: {}. Use a backend flag (e.g. `--onnx-jina-cuda`) to override.",
                default_setup_backend()
            );
        }
        default_setup_backend()
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
    let api_setup_with_preset = (needs_api_prompt && interactive)
        .then(prompt_api_setup)
        .transpose()?;
    // Wizard cares about preset identity (Qwen single-key + auto-Generic);
    // the generic `run` path only needs the raw inputs, so strip the id here.
    let api_setup =
        api_setup_with_preset.map(|(embedding, reranker, _preset_id)| (embedding, reranker));

    configure_backend_with_api_setup(
        effective_backend,
        local_embedding_model,
        index_path,
        json_output,
        "Vera setup complete.",
        api_setup,
        true,
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
        // Friction: Step 2 (agent skills) and Step 3 (index now) asked two extra
        // confirmations before the first search could run. For API first-run the
        // preset is complete after the single key entry, so skip directly to the
        // outro and let `vera index` / `vera search` run separately.
        cliclack::outro(
            "Setup complete! Run `vera index .` and `vera search \"query\"` to get started.",
        )?;
        return Ok(());
    } else {
        configure_backend(
            effective_backend,
            local_embedding_model,
            None,
            false,
            "Backend configured.",
        )?;
    }

    // Step 2: Agent skill installation (local backends only)
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
        .initial_value(true)
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
        true,
    )
}

pub(crate) fn repair_backend(
    effective_backend: InferenceBackend,
    local_embedding_model: Option<LocalEmbeddingModelConfig>,
    json_output: bool,
    success_header: &str,
) -> anyhow::Result<()> {
    configure_backend_with_api_setup(
        effective_backend,
        local_embedding_model,
        None,
        json_output,
        success_header,
        None,
        false,
    )
}

fn configure_backend_with_api_setup(
    effective_backend: InferenceBackend,
    local_embedding_model: Option<LocalEmbeddingModelConfig>,
    index_path: Option<String>,
    json_output: bool,
    success_header: &str,
    api_setup: Option<(ApiSetupInput, Option<ApiSetupInput>)>,
    persist_state: bool,
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
            if persist_state {
                state::save_backend(effective_backend)?;
                state::save_local_embedding_model(&local_embedding_model)?;
                state::clear_reranker_setup()?;
            }
            state::apply_saved_env_force()?;
            local_embedding_summary = Some(local_embedding_model.display_name());
        }
        InferenceBackend::PotionCode => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;
            rt.block_on(vera_core::local_models::ensure_potion_code_assets())?;
            models_prefetched = vera_core::local_models::inspect_potion_code_model_files()?.len();
            onnx_runtime_ready = None;
            if persist_state {
                state::save_backend(effective_backend)?;
                state::clear_reranker_setup()?;
            }
            state::apply_saved_env_force()?;
            local_embedding_summary = Some(vera_core::local_models::potion_code_model_name());
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
            if persist_state {
                state::save_api_setup(&embedding, reranker.as_ref())?;
            }
            state::apply_saved_env_force()?;
            onnx_runtime_ready = None;
        }
    }

    if persist_state
        && state::load_saved_config()?.install_method.is_none()
        && let Some(install_method) = crate::update_check::resolve_install_method().install_method
    {
        state::save_install_method(Some(&install_method))?;
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

/// Probe the system for a usable GPU for the interactive menu's auto-detect
/// entry. Falls back to Potion Code on CPU if nothing is detected.
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

    // Windows: ask Direct3D 12 itself whether a device can be created, the way
    // every other branch asks the machine a question (#213, #248). A driver
    // that answers yes is the whole of DirectML's hardware requirement; anything
    // else falls through to the common Potion Code fallback below.
    if let Some(provider) = directml_provider_for_d3d12(has_directx12_adapter()) {
        return InferenceBackend::OnnxJina(provider);
    }

    InferenceBackend::PotionCode
}

/// Maps "this machine has a Direct3D 12 device" onto the DirectML provider.
///
/// DirectML's documented hardware requirement is a DirectX 12 capable device
/// and nothing more, so the answer to the probe is the answer for the backend.
/// Pure over its input so non-Windows tests exercise both arms.
fn directml_provider_for_d3d12(has_d3d12_device: bool) -> Option<OnnxExecutionProvider> {
    has_d3d12_device.then_some(OnnxExecutionProvider::DirectMl)
}

/// COM interface identifier, laid out as the Windows `GUID` struct.
///
/// This and the items below exist for the Windows probe; the tests drive them
/// on every platform through a stand-in `D3D12CreateDevice`, which is why the
/// gate admits `test` as well as `windows`.
#[cfg(any(windows, test))]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Iid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

#[cfg(any(windows, test))]
/// `ID3D12Device`, `189819f1-1db6-4b57-be54-1821339b85f7`. Taken from the
/// Windows metadata projection (`windows` crate, `Win32/Graphics/Direct3D12`),
/// not from a header transcribed by hand.
const IID_ID3D12_DEVICE: Iid = Iid {
    data1: 0x189819f1,
    data2: 0x1db6,
    data3: 0x4b57,
    data4: [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
};

#[cfg(any(windows, test))]
/// `D3D_FEATURE_LEVEL_11_0`, the floor for a Direct3D 12 device.
const D3D_FEATURE_LEVEL_11_0: i32 = 45056;

#[cfg(any(windows, test))]
/// The `d3d12.dll` export, as the raw ABI rather than a safe wrapper. Naming
/// the type lets the probe body be checked by the compiler and driven by a
/// stand-in function on every platform, so only the symbol lookup is
/// Windows-only.
type D3d12CreateDevice = unsafe extern "system" fn(
    adapter: *mut std::ffi::c_void,
    minimum_feature_level: i32,
    riid: *const Iid,
    device: *mut *mut std::ffi::c_void,
) -> i32;

#[cfg(any(windows, test))]
/// Whether `D3D12CreateDevice` reports that a device could be created.
///
/// The call passes a null `ppDevice`, which is the documented way to test
/// device creation without performing it, and a null adapter, which selects
/// the default adapter — the same adapter ONNX Runtime's DirectML provider
/// uses at its default `device_id` of 0. **Under a null `ppDevice` a success
/// is `S_FALSE` (1), not `S_OK` (0)**, so the verdict is the sign bit of the
/// `HRESULT` and never an equality against zero.
fn d3d12_device_can_be_created(create_device: D3d12CreateDevice) -> bool {
    let hresult = unsafe {
        create_device(
            std::ptr::null_mut(),
            D3D_FEATURE_LEVEL_11_0,
            &IID_ID3D12_DEVICE,
            std::ptr::null_mut(),
        )
    };
    let supported = hresult >= 0;
    if !supported {
        // Keeps a wrong backend choice on user hardware diagnosable from a run,
        // instead of collapsing every refusal into a bare `false`.
        tracing::debug!(
            hresult = format!("0x{hresult:08x}"),
            "D3D12CreateDevice declined"
        );
    }
    supported
}

/// Whether this machine has a Direct3D 12 capable adapter.
///
/// `d3d12.dll` is resolved at run time rather than linked, so a Windows build
/// still starts on an installation that does not ship it; its absence is a
/// "no", and it is searched for in System32 only. The module is deliberately left loaded: the probe may have caused
/// Direct3D to cache the per-adapter device singleton, and unloading the
/// library underneath that is not worth one freed handle in a setup path.
/// Non-Windows targets answer "no" so the shared `detect_gpu` body compiles
/// and runs everywhere.
fn has_directx12_adapter() -> bool {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::LibraryLoader::{
            GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExA,
        };

        // `LOAD_LIBRARY_SEARCH_SYSTEM32` rather than a bare `LoadLibraryA`: the
        // default search order reaches the current directory, and setup runs
        // inside whatever project the user is standing in. `d3d12.dll` is not
        // reliably a KnownDLL, so a copy checked into a repository would
        // otherwise load ahead of the system one and run its `DllMain` here.
        let module = unsafe {
            LoadLibraryExA(
                c"d3d12.dll".as_ptr().cast(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        if module.is_null() {
            tracing::debug!("d3d12.dll not present in System32; no DirectX 12 adapter");
            return false;
        }
        let Some(symbol) =
            (unsafe { GetProcAddress(module, c"D3D12CreateDevice".as_ptr().cast()) })
        else {
            tracing::debug!("d3d12.dll exports no D3D12CreateDevice; no DirectX 12 adapter");
            return false;
        };
        let create_device: D3d12CreateDevice = unsafe { std::mem::transmute(symbol) };
        d3d12_device_can_be_created(create_device)
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

/// Show an interactive backend selection menu. Potion Code is the default.
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
            InferenceBackend::PotionCode,
            "Potion Code CPU",
            "static embeddings, works everywhere (default)",
        )
        .item(
            detected,
            format!("Auto-detect ({detected_hint})"),
            "GPU ONNX backend",
        )
        .item(
            InferenceBackend::Api,
            "API mode",
            "remote OpenAI-compatible endpoints",
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
///
/// The Qwen (OpenRouter) preset uses `qwen/qwen3-embedding-8b` +
/// `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1` with a single
/// shared API key. The reranker for this preset relies on the generic
/// protocol (`top_n`/`results`) unless the operator overrides it in the
/// subsequent reranker-protocol step or via `vera config set`; other presets
/// default to auto-detect (voyage on `voyageai.com`, generic elsewhere).
fn configure_api_interactive() -> anyhow::Result<()> {
    let (embedding, reranker, preset_id) = prompt_api_setup()?;
    // Collect reranker protocol settings before any persistence so cancellation
    // at any prompt leaves existing config untouched.
    // Qwen (OpenRouter) uses the generic protocol; apply it without prompting
    // so the preset completes with a single key entry (friction: before, three
    // extra prompts for protocol/endpoint/task even though defaults are correct).
    let is_qwen = preset_uses_auto_generic(preset_id);
    let reranker_protocol_update = if reranker.is_some() {
        if is_qwen {
            Some(RerankerProtocolUpdate {
                protocol: Some(RerankerProtocol::Generic),
                endpoint_path: None,
                task_instruction: None,
                task_field: None,
            })
        } else {
            Some(prompt_reranker_protocol_settings_for_preset(preset_id)?)
        }
    } else {
        None
    };

    state::save_backend(InferenceBackend::Api)?;
    state::save_api_setup(&embedding, reranker.as_ref())?;
    if let Some(update) = reranker_protocol_update {
        let mut runtime = state::load_runtime_config()?;
        apply_reranker_protocol_update(&mut runtime, update);
        state::save_runtime_config(&runtime)?;
    }
    state::apply_saved_env_force()?;

    if state::load_saved_config()?.install_method.is_none()
        && let Some(install_method) = crate::update_check::resolve_install_method().install_method
    {
        state::save_install_method(Some(&install_method))?;
    }

    cliclack::log::success("API backend configured.")?;
    cliclack::log::info(
        "Your credentials are saved in Vera's config directory. You can remove any \
         EMBEDDING_MODEL_* / RERANKER_MODEL_* env vars from your shell.",
    )?;
    Ok(())
}

#[derive(Debug, Clone)]
struct RerankerProtocolUpdate {
    protocol: Option<RerankerProtocol>,
    endpoint_path: Option<String>,
    task_instruction: Option<String>,
    task_field: Option<String>,
}

fn apply_reranker_protocol_update(
    runtime: &mut vera_core::config::VeraConfig,
    update: RerankerProtocolUpdate,
) {
    runtime.retrieval.reranker_protocol = update.protocol;
    runtime.retrieval.reranker_endpoint_path = update.endpoint_path;
    runtime.retrieval.reranker_task_instruction = update.task_instruction;
    runtime.retrieval.reranker_task_field = update.task_field;
}

fn prompt_reranker_protocol_settings_for_preset(
    preset_id: ApiPresetId,
) -> anyhow::Result<RerankerProtocolUpdate> {
    let existing = state::load_runtime_config().unwrap_or_default();
    let existing_protocol = existing.retrieval.reranker_protocol;
    let existing_endpoint = existing.retrieval.reranker_endpoint_path.clone();
    let existing_instruction = existing.retrieval.reranker_task_instruction.clone();
    let existing_field = existing.retrieval.reranker_task_field.clone();

    // Qwen (OpenRouter) uses the generic wire protocol; suggest explicit generic
    // so the preset round-trips with a persisted protocol setting. Other presets
    // keep auto-detect as the default, which resolves to generic for non-Voyage
    // hosts and to voyage on voyageai.com.
    let suggested_protocol = if preset_uses_auto_generic(preset_id) {
        Some(RerankerProtocol::Generic)
    } else {
        None
    };
    let default_protocol = existing_protocol.or(suggested_protocol);

    // Map protocol to cliclack choice index: 0=auto, 1=generic, 2=voyage.
    let protocol_choice: usize = {
        let mut select = cliclack::select("Reranker protocol (wire format for rerank requests)");
        select = select.item(
            0usize,
            "Auto-detect",
            "voyage on voyageai.com, generic elsewhere",
        );
        select = select.item(
            1usize,
            "Generic (top_n/results)",
            "for OpenRouter, Jina, Cohere",
        );
        select = select.item(2usize, "Voyage (top_k/data)", "for api.voyageai.com");
        let default_index = match default_protocol {
            None => 0,
            Some(RerankerProtocol::Generic) => 1,
            Some(RerankerProtocol::Voyage) => 2,
        };
        // cliclack's `initial_value` selects the highlighted item before interaction.
        select = select.initial_value(default_index);
        select.interact()?
    };
    let protocol = match protocol_choice {
        1 => Some(RerankerProtocol::Generic),
        2 => Some(RerankerProtocol::Voyage),
        _ => None,
    };

    // Endpoint path: empty means preset default (`{base}/rerank`). Validation
    // enforces leading slash when set; keep raw None for default.
    let endpoint_path: Option<String> = {
        let default_display = existing_endpoint.as_deref().unwrap_or("");
        let input: String = if default_display.is_empty() {
            cliclack::input("Reranker endpoint path (Enter for default {base}/rerank)")
                .placeholder("/rerank")
                .required(false)
                .interact()?
        } else {
            cliclack::input("Reranker endpoint path (Enter for default {base}/rerank)")
                .default_input(default_display)
                .interact()?
        };
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            if !trimmed.starts_with('/') {
                bail!("reranker endpoint path must start with '/'");
            }
            Some(trimmed)
        }
    };

    // Task instruction / field: optional, empty means not set.
    let task_instruction: Option<String> = {
        let default_display = existing_instruction.as_deref().unwrap_or("");
        let input: String = if default_display.is_empty() {
            cliclack::input("Reranker task instruction (optional, Enter to skip)")
                .placeholder("e.g. Given a query, retrieve relevant code")
                .required(false)
                .interact()?
        } else {
            cliclack::input("Reranker task instruction (optional, Enter to skip)")
                .default_input(default_display)
                .interact()?
        };
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    };

    let task_field: Option<String> = if task_instruction.is_some() {
        let default_display = existing_field.as_deref().unwrap_or("");
        let prompt_label = "Reranker task field (optional, Enter to skip)";
        let input: String = if default_display.is_empty() {
            cliclack::input(prompt_label)
                .placeholder("e.g. instruction")
                .required(false)
                .interact()?
        } else {
            cliclack::input(prompt_label)
                .default_input(default_display)
                .interact()?
        };
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    } else {
        // If no instruction, keep existing field only if previously set and instruction was previously set; otherwise clear.
        // To keep idempotency, preserve existing field when instruction is None but field was set? For simplicity, clear field when instruction is None and prompt not shown.
        // However for idempotency we should preserve existing field only if instruction also preserved. Since instruction is None, field should be None.
        // But to respect "changing exactly one field preserves others", we need to allow preserving field even when instruction cleared? For now, preserve existing if instruction was previously set? Simpler: if task_instruction is None and existing field is Some, keep it only if user explicitly wants? We'll clear to avoid orphan.
        None
    };

    Ok(RerankerProtocolUpdate {
        protocol,
        endpoint_path,
        task_instruction,
        task_field,
    })
}

fn prompt_api_setup() -> anyhow::Result<(ApiSetupInput, Option<ApiSetupInput>, ApiPresetId)> {
    let presets = API_PRESETS;

    let mut select = cliclack::select("Select an API provider");
    for (i, p) in presets.iter().enumerate() {
        select = select.item(i, p.label, p.hint);
    }
    let choice: usize = select.interact()?;
    let preset = &presets[choice];

    // Friction: selecting Qwen previously required Enter to accept the
    // embedding base URL, model ID, reranker base URL and model ID even
    // though the preset already defines them exactly (transcript showed four
    // redundant Enter presses before the key). Skip those prompts for Qwen
    // and go straight to the single credential entry, reusing it for both
    // endpoints without a second prompt.
    if preset_uses_single_shared_key(preset.id) {
        let api_key: String = cliclack::password("OpenRouter API key")
            .mask('▪')
            .interact()?;
        if api_key.trim().is_empty() {
            bail!("OpenRouter API key is required");
        }
        let embedding = ApiSetupInput {
            base_url: preset.embedding_base_url.to_string(),
            model_id: preset.embedding_model.to_string(),
            api_key: api_key.clone(),
        };
        let reranker = Some(ApiSetupInput {
            base_url: preset.reranker_base_url.to_string(),
            model_id: preset.reranker_model.to_string(),
            api_key,
        });
        return Ok((embedding, reranker, preset.id));
    }

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
                .allow_empty()
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

    Ok((embedding, reranker, preset.id))
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
    fn setup_with_no_flags_defaults_to_potion_code() {
        // Potion Code runs on any CPU with no ONNX runtime, so it is the
        // default on every machine; GPU ONNX backends stay opt-in via flags.
        assert_eq!(default_setup_backend(), InferenceBackend::PotionCode);
    }

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

    /// Pins only the mapping from the probe's answer to the provider. The
    /// probe itself is covered by the tests below; the Windows-only symbol
    /// lookup in `has_directx12_adapter` is covered by neither, and cannot be
    /// from a non-Windows host.
    #[test]
    fn a_d3d12_answer_maps_to_the_directml_provider() {
        assert_eq!(
            directml_provider_for_d3d12(true),
            Some(OnnxExecutionProvider::DirectMl)
        );
        assert_eq!(directml_provider_for_d3d12(false), None);
    }

    /// `S_FALSE`, not `S_OK`, is what `D3D12CreateDevice` returns when it
    /// succeeds under a null `ppDevice` (#248). A probe that tested for zero
    /// would call every Direct3D 12 machine in existence unsupported, so this
    /// pins the success arm on the value the API actually returns.
    #[test]
    fn a_null_ppdevice_success_is_s_false_and_still_counts_as_supported() {
        unsafe extern "system" fn returns_s_false(
            _adapter: *mut std::ffi::c_void,
            _feature_level: i32,
            _riid: *const Iid,
            _device: *mut *mut std::ffi::c_void,
        ) -> i32 {
            1 // S_FALSE
        }
        assert!(d3d12_device_can_be_created(returns_s_false));
    }

    /// The other side: real failure HRESULTs are negative, and each must read
    /// as "no Direct3D 12 here" rather than being mistaken for a status code.
    #[test]
    fn failing_hresults_report_no_d3d12_device() {
        unsafe extern "system" fn returns_e_fail(
            _adapter: *mut std::ffi::c_void,
            _feature_level: i32,
            _riid: *const Iid,
            _device: *mut *mut std::ffi::c_void,
        ) -> i32 {
            0x8000_4005_u32 as i32 // E_FAIL
        }
        unsafe extern "system" fn returns_dxgi_not_found(
            _adapter: *mut std::ffi::c_void,
            _feature_level: i32,
            _riid: *const Iid,
            _device: *mut *mut std::ffi::c_void,
        ) -> i32 {
            0x887A_0002_u32 as i32 // DXGI_ERROR_NOT_FOUND
        }
        assert!(!d3d12_device_can_be_created(returns_e_fail));
        assert!(!d3d12_device_can_be_created(returns_dxgi_not_found));
    }

    /// The call must ask for a device it would actually be able to use: the
    /// Direct3D 12 floor feature level, `ID3D12Device`, and a null `ppDevice`
    /// so nothing is created. Recorded from inside the call rather than
    /// asserted about the source.
    #[test]
    fn the_probe_asks_for_a_feature_level_11_id3d12device_without_creating_one() {
        const UNSET: Iid = Iid {
            data1: 0,
            data2: 0,
            data3: 0,
            data4: [0; 8],
        };
        thread_local! {
            static SEEN: std::cell::Cell<(i32, Iid, bool)> = const {
                std::cell::Cell::new((0, UNSET, false))
            };
        }
        unsafe extern "system" fn record(
            _adapter: *mut std::ffi::c_void,
            feature_level: i32,
            riid: *const Iid,
            device: *mut *mut std::ffi::c_void,
        ) -> i32 {
            let riid = unsafe { *riid };
            SEEN.with(|seen| seen.set((feature_level, riid, device.is_null())));
            1
        }
        assert!(d3d12_device_can_be_created(record));
        let (feature_level, riid, ppdevice_was_null) = SEEN.with(std::cell::Cell::get);
        assert_eq!(feature_level, 45056, "D3D_FEATURE_LEVEL_11_0");
        // The whole GUID: a typo in any field would still name a real
        // interface, and the probe would report "no Direct3D 12" on capable
        // hardware while every partial assertion kept passing.
        assert_eq!(
            riid,
            Iid {
                data1: 0x189819f1,
                data2: 0x1db6,
                data3: 0x4b57,
                data4: [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
            },
            "IID_ID3D12Device"
        );
        assert!(ppdevice_was_null, "the probe must not create a device");
    }

    /// Every non-Windows target answers "no" without a probe, so auto-detect
    /// reaches the shared Potion Code fallback rather than a backend that
    /// cannot load.
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn non_windows_targets_never_report_a_directx12_adapter() {
        assert!(!has_directx12_adapter());
        assert_eq!(directml_provider_for_d3d12(has_directx12_adapter()), None);
    }

    /// Qwen optimizations must trigger on preset identity (ApiPresetId::Qwen),
    /// not on display strings or model_id/base_url equality. Changing the
    /// label text must not change behavior, and spoofing the label on a
    /// different preset must not trigger Qwen-only paths. This pins the
    /// single-shared-key prompt skip and the auto Generic protocol.
    #[test]
    fn qwen_optimizations_trigger_on_preset_identity_not_label_text() {
        // Single-shared-key and auto-Generic helpers are pure over the id.
        for (id, should_be_qwen) in [
            (ApiPresetId::OpenAi, false),
            (ApiPresetId::Jina, false),
            (ApiPresetId::Voyage, false),
            (ApiPresetId::Qwen, true),
            (ApiPresetId::Custom, false),
        ] {
            assert_eq!(
                preset_uses_single_shared_key(id),
                should_be_qwen,
                "single-shared-key for {id:?}"
            );
            assert_eq!(
                preset_uses_auto_generic(id),
                should_be_qwen,
                "auto-Generic for {id:?}"
            );
            assert_eq!(id.is_qwen(), should_be_qwen, "is_qwen for {id:?}");
        }

        // Round-trip: the canonical Qwen entry still round-trips through id.
        let qwen = preset_by_id(ApiPresetId::Qwen);
        assert_eq!(qwen.id, ApiPresetId::Qwen);
        assert_eq!(qwen.label, "Qwen (OpenRouter)");
        assert_eq!(qwen.embedding_model, "qwen/qwen3-embedding-8b");
        assert_eq!(qwen.reranker_model, "qwen/qwen3-reranker-8b");
        assert!(preset_uses_single_shared_key(qwen.id));
        assert!(preset_uses_auto_generic(qwen.id));

        // Label spoof: Qwen id with a fake label must still be treated as Qwen
        // (proves we don't key on `preset.label == "Qwen (OpenRouter)"`).
        let spoofed = ApiPreset {
            id: ApiPresetId::Qwen,
            label: "Totally Not Qwen",
            hint: "spoofed",
            embedding_base_url: "https://example.com/v1",
            embedding_model: "fake",
            reranker_base_url: "https://example.com/v1",
            reranker_model: "fake",
        };
        assert!(preset_uses_single_shared_key(spoofed.id));
        assert!(preset_uses_auto_generic(spoofed.id));

        // Opposite spoof: Custom id whose label pretends to be Qwen must NOT
        // be treated as Qwen (proves label text is ignored).
        let fake_qwen = ApiPreset {
            id: ApiPresetId::Custom,
            label: "Qwen (OpenRouter)",
            hint: "spoofed qwen label on wrong id",
            embedding_base_url: "https://openrouter.ai/api/v1",
            embedding_model: "qwen/qwen3-embedding-8b",
            reranker_base_url: "https://openrouter.ai/api/v1",
            reranker_model: "qwen/qwen3-reranker-8b",
        };
        assert!(!preset_uses_single_shared_key(fake_qwen.id));
        assert!(!preset_uses_auto_generic(fake_qwen.id));
    }
}
