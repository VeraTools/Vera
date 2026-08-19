//! `vera doctor` — inspect the current Vera setup for common failures.

use serde::Serialize;

use crate::state;
use crate::update_check::{self, VersionCheckSource};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CheckStatus {
    Ok,
    Skip,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize)]
struct DoctorCheck {
    name: &'static str,
    status: CheckStatus,
    detail: String,
}

/// Execution providers the local ONNX sessions actually run on.
///
/// A single requested backend does not decide this: the embedding session and
/// the reranker session each resolve it independently, and either can land on
/// CPU while the request was a GPU provider. Preflighting the request instead
/// of the resolution makes doctor report failures for a configuration
/// production runs fine, and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EffectiveProviders {
    embedding: vera_core::config::OnnxExecutionProvider,
    reranker: vera_core::config::OnnxExecutionProvider,
}

impl EffectiveProviders {
    fn resolve(
        ep: vera_core::config::OnnxExecutionProvider,
        embedding_model: &vera_core::local_models::LocalEmbeddingModelConfig,
    ) -> Self {
        Self {
            embedding: vera_core::local_models::embedding_execution_provider(ep, embedding_model),
            reranker: vera_core::local_models::reranker_execution_provider(ep),
        }
    }

    /// Distinct non-CPU providers needing a shared-library dependency check,
    /// in embedding-then-reranker order. Empty when both sessions run on CPU,
    /// where provider-specific checks are not needed.
    fn dependency_check_providers(self) -> Vec<vera_core::config::OnnxExecutionProvider> {
        let mut providers = Vec::new();
        for candidate in [self.embedding, self.reranker] {
            if candidate != vera_core::config::OnnxExecutionProvider::Cpu
                && !providers.contains(&candidate)
            {
                providers.push(candidate);
            }
        }
        providers
    }
}

#[derive(Debug, Clone, Serialize)]
struct DoctorReport {
    version: String,
    overall_ok: bool,
    checks: Vec<DoctorCheck>,
}

pub fn run(json_output: bool, probe: bool) -> anyhow::Result<()> {
    let mut checks = Vec::new();
    let version = update_check::current_version().to_string();

    let version_status = update_check::binary_version_status(true);
    checks.push(version_check(&version_status));

    let config_path = state::config_path()?;
    checks.push(DoctorCheck {
        name: "config-file",
        status: if config_path.exists() {
            CheckStatus::Ok
        } else {
            CheckStatus::Warn
        },
        detail: config_path.display().to_string(),
    });

    let saved_config = state::load_saved_config()?;
    checks.push(saved_backend_check(&saved_config));

    let backend = vera_core::config::resolve_backend(None);
    let local_mode = backend.is_local();
    checks.push(DoctorCheck {
        name: "effective-mode",
        status: CheckStatus::Ok,
        detail: if local_mode {
            "local".to_string()
        } else {
            "api".to_string()
        },
    });
    checks.push(DoctorCheck {
        name: "effective-backend",
        status: CheckStatus::Ok,
        detail: backend.to_string(),
    });

    match backend {
        vera_core::config::InferenceBackend::OnnxJina(ep) => {
            let embedding_model = vera_core::local_models::LocalEmbeddingModelConfig::from_env()?;
            checks.push(DoctorCheck {
                name: "local-embedding-model",
                status: CheckStatus::Ok,
                detail: embedding_model.display_name(),
            });
            // The embedding and reranker sessions each resolve the requested
            // backend to their own provider, so preflight the resolved pair
            // rather than the request: otherwise doctor checks a library
            // production never loads and skips every probe when it is absent.
            let providers = EffectiveProviders::resolve(ep, &embedding_model);
            let runtime_path =
                vera_core::local_models::ort_library_path_for_ep(providers.embedding)?;
            let runtime_check = vera_core::local_models::ensure_ort_runtime(Some(&runtime_path));
            let runtime_detail = match &runtime_check {
                Ok(()) => runtime_path.display().to_string(),
                Err(err) => format!("{} ({})", runtime_path.display(), one_line_error(err)),
            };
            checks.push(DoctorCheck {
                name: "onnx-runtime",
                status: if runtime_check.is_ok() {
                    CheckStatus::Ok
                } else {
                    CheckStatus::Fail
                },
                detail: runtime_detail,
            });

            let model_assets = vera_core::local_models::inspect_local_model_files_for_ep(
                providers.embedding,
                &embedding_model,
            )?;
            // The hint stays on the requested backend: `vera repair
            // --onnx-jina-<ep>` is what installs that backend's files.
            let repair_hint = format!("run `vera repair --onnx-jina-{ep}`");
            checks.push(local_model_assets_check(&model_assets, &repair_hint));
            if probe {
                checks.extend(probe_local_backend(
                    ep,
                    providers,
                    &runtime_path,
                    &model_assets,
                )?);
            }
        }
        vera_core::config::InferenceBackend::PotionCode => {
            checks.push(DoctorCheck {
                name: "local-embedding-model",
                status: CheckStatus::Ok,
                detail: vera_core::local_models::potion_code_model_name().to_string(),
            });
            let model_assets = vera_core::local_models::inspect_potion_code_model_files()?;
            checks.push(local_model_assets_check(
                &model_assets,
                "run `vera repair --potion-code`",
            ));
            if probe {
                checks.extend(probe_potion_backend(&model_assets));
            }
        }
        vera_core::config::InferenceBackend::Api => {
            checks.push(check_env_group(
                "embedding-api",
                &[
                    "EMBEDDING_MODEL_BASE_URL",
                    "EMBEDDING_MODEL_ID",
                    "EMBEDDING_MODEL_API_KEY",
                ],
            ));
            checks.push(check_env_group(
                "reranker-api",
                &[
                    "RERANKER_MODEL_BASE_URL",
                    "RERANKER_MODEL_ID",
                    "RERANKER_MODEL_API_KEY",
                ],
            ));
            if probe {
                checks.push(skipped_check(
                    "probe",
                    "probe is only available for local backends",
                ));
            }
        }
    }

    let cwd = std::env::current_dir()?;
    let index_dir = vera_core::indexing::index_dir(&cwd);
    checks.push(DoctorCheck {
        name: "current-index",
        status: if index_dir.exists() {
            CheckStatus::Ok
        } else {
            CheckStatus::Warn
        },
        detail: index_dir.display().to_string(),
    });

    let overall_ok = checks
        .iter()
        .all(|check| !matches!(check.status, CheckStatus::Fail));
    let report = DoctorReport {
        version,
        overall_ok,
        checks,
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Vera doctor v{}", report.version);
        println!();
        for check in &report.checks {
            let icon = match check.status {
                CheckStatus::Ok => "ok",
                CheckStatus::Skip => "skip",
                CheckStatus::Warn => "warn",
                CheckStatus::Fail => "fail",
            };
            println!("  {:<5} {:<14} {}", icon, check.name, check.detail);
        }
    }

    Ok(())
}

fn check_env_group(name: &'static str, keys: &[&'static str]) -> DoctorCheck {
    let present = keys
        .iter()
        .filter(|key| std::env::var_os(key).is_some())
        .count();

    let status = match present {
        0 => CheckStatus::Warn,
        n if n == keys.len() => CheckStatus::Ok,
        _ => CheckStatus::Fail,
    };

    DoctorCheck {
        name,
        status,
        detail: format!("{present}/{} variables present", keys.len()),
    }
}

fn saved_backend_check(config: &state::StoredConfig) -> DoctorCheck {
    match config.backend {
        Some(backend) => DoctorCheck {
            name: "saved-backend",
            status: CheckStatus::Ok,
            detail: backend.to_string(),
        },
        None => match config.local_mode {
            Some(true) => DoctorCheck {
                name: "saved-backend",
                status: CheckStatus::Warn,
                detail: "legacy local mode config (defaults to onnx-jina-cpu)".to_string(),
            },
            Some(false) => DoctorCheck {
                name: "saved-backend",
                status: CheckStatus::Warn,
                detail: "legacy api mode config".to_string(),
            },
            None => DoctorCheck {
                name: "saved-backend",
                status: CheckStatus::Warn,
                detail: "not configured".to_string(),
            },
        },
    }
}

fn local_model_assets_check(
    model_assets: &[vera_core::local_models::LocalModelAssetStatus],
    repair_hint: &str,
) -> DoctorCheck {
    let valid = model_assets.iter().filter(|asset| asset.is_valid()).count();
    let missing = model_assets
        .iter()
        .filter(|asset| asset.is_missing())
        .map(|asset| asset.name)
        .collect::<Vec<_>>();
    let invalid = model_assets
        .iter()
        .filter(|asset| asset.is_invalid())
        .map(|asset| {
            asset.detail.as_deref().map_or_else(
                || asset.name.to_string(),
                |detail| format!("{} ({detail})", asset.name),
            )
        })
        .collect::<Vec<_>>();
    let status = if invalid.is_empty() && missing.is_empty() {
        CheckStatus::Ok
    } else if invalid.is_empty() {
        CheckStatus::Warn
    } else {
        CheckStatus::Fail
    };
    let detail = if !invalid.is_empty() {
        format!(
            "{valid}/{} local assets valid; invalid: {}; {repair_hint}",
            model_assets.len(),
            invalid.join(", ")
        )
    } else if !missing.is_empty() {
        format!(
            "{valid}/{} local assets valid; missing: {}; {repair_hint}",
            model_assets.len(),
            missing.join(", ")
        )
    } else {
        format!("{valid}/{} local assets valid", model_assets.len())
    };
    DoctorCheck {
        name: "local-models",
        status,
        detail,
    }
}

fn probe_potion_backend(
    model_assets: &[vera_core::local_models::LocalModelAssetStatus],
) -> Vec<DoctorCheck> {
    let missing = missing_assets(
        model_assets,
        &["potion-tokenizer", "potion-model", "potion-config"],
    );
    if !missing.is_empty() {
        return vec![skipped_check(
            "probe-potion-code",
            format!("skipped because assets are missing: {}", missing.join(", ")),
        )];
    }

    vec![result_check(
        "probe-potion-code",
        "potion-code returned a finite embedding".to_string(),
        vera_core::embedding::Model2VecProvider::probe_inference(),
    )]
}

fn probe_local_backend(
    ep: vera_core::config::OnnxExecutionProvider,
    providers: EffectiveProviders,
    runtime_path: &std::path::Path,
    model_assets: &[vera_core::local_models::LocalModelAssetStatus],
) -> anyhow::Result<Vec<DoctorCheck>> {
    let embedding_ep = providers.embedding;
    let reranker_ep = providers.reranker;
    let mut checks = Vec::new();

    // `ensure_ort_runtime` is a process-wide `OnceLock`: only the FIRST path
    // handed to it is dlopened, and every later call replays that first result
    // whatever path it is given. `runtime_path` is the embedding provider's
    // library, so when the two components resolve to different providers the
    // reranker's library can only be existence-checked (which is what
    // `dependency_probe_status` does), never load-checked. Calling
    // `ensure_ort_runtime` a second time would report the first library's
    // outcome under the second library's name.
    let ort_stage = result_check(
        "probe-ort-library",
        runtime_path.display().to_string(),
        vera_core::local_models::ensure_ort_runtime(Some(runtime_path)),
    );
    let ort_ok = matches!(ort_stage.status, CheckStatus::Ok);
    checks.push(ort_stage);

    let mut embedding_provider_ok = false;
    let mut reranker_provider_ok = false;
    let provider_stage = if ort_ok {
        let embedding_result =
            vera_core::embedding::local_provider::LocalEmbeddingProvider::probe_provider_registration(
                embedding_ep,
            );
        embedding_provider_ok = embedding_result.is_ok();
        if reranker_ep == embedding_ep {
            reranker_provider_ok = embedding_provider_ok;
            result_check(
                "probe-provider-registration",
                format!("registered {embedding_ep} for embedding and reranking"),
                embedding_result,
            )
        } else {
            let reranker_result =
                vera_core::embedding::local_provider::LocalEmbeddingProvider::probe_provider_registration(
                    reranker_ep,
                );
            reranker_provider_ok = reranker_result.is_ok();
            let failures = [
                embedding_result
                    .err()
                    .map(|err| format!("embedding {embedding_ep}: {}", one_line_error(&err))),
                reranker_result
                    .err()
                    .map(|err| format!("reranker {reranker_ep}: {}", one_line_error(&err))),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            if failures.is_empty() {
                DoctorCheck {
                    name: "probe-provider-registration",
                    status: CheckStatus::Ok,
                    detail: format!(
                        "registered {embedding_ep} for embedding and {reranker_ep} for reranking"
                    ),
                }
            } else {
                DoctorCheck {
                    name: "probe-provider-registration",
                    status: CheckStatus::Fail,
                    detail: failures.join("; "),
                }
            }
        }
    } else {
        skipped_check(
            "probe-provider-registration",
            "skipped because ONNX Runtime could not be initialized",
        )
    };
    checks.push(provider_stage);

    let dependency_providers = providers.dependency_check_providers();
    let mut failed_dependency_providers = Vec::new();
    let dependencies_stage = if dependency_providers.is_empty() {
        skipped_check(
            "probe-dependencies",
            "skipped because the embedding and reranker sessions both run on CPU, where provider-specific shared-library checks are not needed",
        )
    } else if ort_ok {
        let mut status = CheckStatus::Ok;
        let mut details = Vec::new();
        for provider in &dependency_providers {
            let library_path = if *provider == embedding_ep {
                runtime_path.to_path_buf()
            } else {
                vera_core::local_models::ort_library_path_for_ep(*provider)?
            };
            let (provider_status, detail) = dependency_probe_status(*provider, &library_path);
            if matches!(provider_status, CheckStatus::Fail) {
                failed_dependency_providers.push(*provider);
            }
            status = worse_status(status, provider_status);
            details.push(if dependency_providers.len() > 1 {
                format!("{provider}: {detail}")
            } else {
                detail
            });
        }
        DoctorCheck {
            name: "probe-dependencies",
            status,
            detail: details.join("; "),
        }
    } else {
        skipped_check(
            "probe-dependencies",
            "skipped because ONNX Runtime could not be initialized",
        )
    };
    let embedding_dependencies_ok = !failed_dependency_providers.contains(&embedding_ep);
    let reranker_dependencies_ok = !failed_dependency_providers.contains(&reranker_ep);
    checks.push(dependencies_stage);

    let embedding_ok = ort_ok && embedding_provider_ok && embedding_dependencies_ok;
    let reranker_ok = ort_ok && reranker_provider_ok && reranker_dependencies_ok;

    let embedding_session_stage = if embedding_ok {
        let missing = missing_assets(model_assets, &["embedding-onnx", "embedding-onnx-data"]);
        if missing.is_empty() {
            result_check(
                "probe-embedding-session",
                "embedding session created".to_string(),
                wrap_onnx_session_probe(
                    vera_core::embedding::local_provider::LocalEmbeddingProvider::probe_session(ep),
                ),
            )
        } else {
            skipped_check(
                "probe-embedding-session",
                format!("skipped because assets are missing: {}", missing.join(", ")),
            )
        }
    } else {
        skipped_check(
            "probe-embedding-session",
            "skipped because provider registration or dependencies failed",
        )
    };
    checks.push(embedding_session_stage);

    let reranker_session_stage = if reranker_ok {
        let missing = missing_assets(model_assets, &["reranker-onnx"]);
        if missing.is_empty() {
            result_check(
                "probe-reranker-session",
                "reranker session created".to_string(),
                wrap_onnx_session_probe(
                    vera_core::retrieval::local_reranker::LocalReranker::probe_session(ep),
                ),
            )
        } else {
            skipped_check(
                "probe-reranker-session",
                format!("skipped because assets are missing: {}", missing.join(", ")),
            )
        }
    } else {
        skipped_check(
            "probe-reranker-session",
            "skipped because provider registration or dependencies failed",
        )
    };
    checks.push(reranker_session_stage);

    let tiny_inference_stage = if embedding_ok && reranker_ok {
        let missing = missing_assets(
            model_assets,
            &[
                "embedding-onnx",
                "embedding-onnx-data",
                "embedding-tokenizer",
                "reranker-onnx",
                "reranker-tokenizer",
            ],
        );
        if missing.is_empty() {
            let result =
                vera_core::embedding::local_provider::LocalEmbeddingProvider::probe_inference(ep)
                    .map_err(|err| anyhow::anyhow!("embedding probe failed: {err}"))
                    .and_then(|_| {
                        vera_core::retrieval::local_reranker::LocalReranker::probe_inference(ep)
                            .map_err(|err| anyhow::anyhow!("reranker probe failed: {err}"))
                    });
            result_check(
                "probe-tiny-inference",
                "embedding and reranker returned finite outputs".to_string(),
                result,
            )
        } else {
            skipped_check(
                "probe-tiny-inference",
                format!("skipped because assets are missing: {}", missing.join(", ")),
            )
        }
    } else {
        skipped_check(
            "probe-tiny-inference",
            "skipped because provider registration or dependencies failed",
        )
    };
    let tiny_inference_ok = matches!(tiny_inference_stage.status, CheckStatus::Ok);
    checks.push(tiny_inference_stage);

    if !dependency_providers.is_empty() {
        checks.push(if tiny_inference_ok {
            DoctorCheck {
                name: "probe-provider-confirmation",
                status: CheckStatus::Ok,
                detail: "session init and tiny inference succeeded; active GPU execution still cannot be confirmed via the ONNX Runtime Rust API, so use trace logs if you need explicit provider confirmation".to_string(),
            }
        } else {
            skipped_check(
                "probe-provider-confirmation",
                "skipped because the tiny inference probe did not succeed",
            )
        });
    }

    // No prebuilt reranker ONNX export runs on the CoreML GPU (the quantized
    // export has ops the CoreML EP cannot execute, the fp16 export has an
    // unsupported input dtype), so the reranker is explicitly pinned to CPU
    // via `reranker_execution_provider`: CoreML can accept a fused subgraph
    // and then fail at inference. The probes above cannot detect this, so
    // surface it deterministically here.
    if ep == vera_core::config::OnnxExecutionProvider::CoreMl {
        // The embedding model is not unconditionally GPU-accelerated either:
        // some models are pinned to CPU under CoreML (see
        // `embedding_execution_provider`). Report what the configured model
        // actually does rather than asserting embeddings run on the GPU. This
        // is the same resolution the probes above ran on, so the two cannot
        // disagree.
        let embedding_on_cpu = embedding_ep == vera_core::config::OnnxExecutionProvider::Cpu;
        let detail = if embedding_on_cpu {
            "the reranker and the configured embedding model both run on CPU under CoreML (no CoreML-compatible reranker ONNX export exists, and this embedding model fails at inference on the CoreML provider), so this backend is not currently GPU-accelerated. If search latency is unacceptable, disable reranking with `vera config set retrieval.reranking_enabled false`"
        } else {
            "the reranker runs on CPU under CoreML (no CoreML-compatible reranker ONNX export exists). The embedding model is configured for CoreML, but ONNX Runtime can still place nodes on CPU, so active GPU execution cannot be confirmed here. If reranking latency is unacceptable, disable it with `vera config set retrieval.reranking_enabled false`"
        };
        checks.push(DoctorCheck {
            name: "probe-reranker-coreml-cpu",
            status: CheckStatus::Warn,
            detail: detail.to_string(),
        });
    }

    Ok(checks)
}

fn missing_assets(
    model_assets: &[vera_core::local_models::LocalModelAssetStatus],
    required: &[&str],
) -> Vec<&'static str> {
    required
        .iter()
        .filter_map(|required_name| {
            model_assets
                .iter()
                .find(|asset| asset.name == *required_name)
                .filter(|asset| !asset.is_valid())
                .map(|asset| asset.name)
        })
        .collect()
}

fn result_check(
    name: &'static str,
    success_detail: String,
    result: anyhow::Result<()>,
) -> DoctorCheck {
    match result {
        Ok(()) => DoctorCheck {
            name,
            status: CheckStatus::Ok,
            detail: success_detail,
        },
        Err(err) => DoctorCheck {
            name,
            status: CheckStatus::Fail,
            detail: one_line_error(&err),
        },
    }
}

fn skipped_check(name: &'static str, detail: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name,
        status: CheckStatus::Skip,
        detail: detail.into(),
    }
}

fn dependency_probe_status(
    ep: vera_core::config::OnnxExecutionProvider,
    runtime_path: &std::path::Path,
) -> (CheckStatus, String) {
    if !runtime_path.exists() {
        return (
            CheckStatus::Skip,
            format!("skipped because {} is missing", runtime_path.display()),
        );
    }

    match vera_core::local_models::inspect_provider_dependencies(ep, runtime_path) {
        Ok(Some(status)) => {
            if status.missing_details.is_empty() {
                (
                    CheckStatus::Ok,
                    "found no unresolved ONNX Runtime dependencies".to_string(),
                )
            } else {
                (
                    CheckStatus::Fail,
                    format!(
                        "missing shared libraries: {}",
                        status.missing_details.join("; ")
                    ),
                )
            }
        }
        Ok(None) => (
            CheckStatus::Skip,
            "dependency inspection is currently available on Linux with `ldd` and macOS with `otool`".to_string(),
        ),
        Err(err) => (CheckStatus::Warn, one_line_error(&err)),
    }
}

/// Keep the most severe of two statuses when one check covers several
/// providers.
fn worse_status(current: CheckStatus, next: CheckStatus) -> CheckStatus {
    fn severity(status: &CheckStatus) -> u8 {
        match status {
            CheckStatus::Ok => 0,
            CheckStatus::Skip => 1,
            CheckStatus::Warn => 2,
            CheckStatus::Fail => 3,
        }
    }

    if severity(&next) > severity(&current) {
        next
    } else {
        current
    }
}

fn wrap_onnx_session_probe(result: anyhow::Result<()>) -> anyhow::Result<()> {
    result.map_err(|err| anyhow::anyhow!(vera_core::local_models::wrap_ort_error(err)))
}

fn version_check(status: &update_check::BinaryVersionStatus) -> DoctorCheck {
    let detail = match status.latest_version.as_deref() {
        Some(latest) if status.update_available() => match status.source {
            VersionCheckSource::Live => format!(
                "v{latest} available (current: v{}; update: `{}`)",
                status.current_version,
                status.update_command()
            ),
            VersionCheckSource::Cache => format!(
                "v{latest} available (cached; current: v{}; update: `{}`)",
                status.current_version,
                status.update_command()
            ),
            VersionCheckSource::Unavailable => unreachable!(),
        },
        Some(latest) => match status.source {
            VersionCheckSource::Live => format!("up to date (latest: v{latest})"),
            VersionCheckSource::Cache => format!("up to date (cached latest: v{latest})"),
            VersionCheckSource::Unavailable => unreachable!(),
        },
        None => "could not check GitHub releases".to_string(),
    };

    DoctorCheck {
        name: "version-check",
        status: if status.update_available() || status.latest_version.is_none() {
            CheckStatus::Warn
        } else {
            CheckStatus::Ok
        },
        detail,
    }
}

fn one_line_error(err: &anyhow::Error) -> String {
    err.to_string()
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::EffectiveProviders;
    use vera_core::config::OnnxExecutionProvider;
    use vera_core::local_models::LocalEmbeddingModelConfig;

    #[test]
    fn coreml_pins_both_sessions_to_cpu_for_coderankembed() {
        let providers = EffectiveProviders::resolve(
            OnnxExecutionProvider::CoreMl,
            &LocalEmbeddingModelConfig::coderankembed(),
        );

        assert_eq!(providers.embedding, OnnxExecutionProvider::Cpu);
        assert_eq!(providers.reranker, OnnxExecutionProvider::Cpu);
        assert!(providers.dependency_check_providers().is_empty());
    }

    #[test]
    fn coreml_keeps_jina_embeddings_on_gpu_and_reranks_on_cpu() {
        let providers = EffectiveProviders::resolve(
            OnnxExecutionProvider::CoreMl,
            &LocalEmbeddingModelConfig::jina(),
        );

        assert_eq!(providers.embedding, OnnxExecutionProvider::CoreMl);
        assert_eq!(providers.reranker, OnnxExecutionProvider::Cpu);
        assert_eq!(
            providers.dependency_check_providers(),
            vec![OnnxExecutionProvider::CoreMl]
        );
    }

    #[test]
    fn cpu_needs_no_dependency_check_for_any_model() {
        for config in [
            LocalEmbeddingModelConfig::jina(),
            LocalEmbeddingModelConfig::coderankembed(),
        ] {
            let providers = EffectiveProviders::resolve(OnnxExecutionProvider::Cpu, &config);

            assert_eq!(providers.embedding, OnnxExecutionProvider::Cpu);
            assert_eq!(providers.reranker, OnnxExecutionProvider::Cpu);
            assert!(providers.dependency_check_providers().is_empty());
        }
    }

    #[test]
    fn non_coreml_gpu_backends_keep_both_sessions_on_the_requested_provider() {
        let providers = EffectiveProviders::resolve(
            OnnxExecutionProvider::Cuda,
            &LocalEmbeddingModelConfig::coderankembed(),
        );

        assert_eq!(providers.embedding, OnnxExecutionProvider::Cuda);
        assert_eq!(providers.reranker, OnnxExecutionProvider::Cuda);
        assert_eq!(
            providers.dependency_check_providers(),
            vec![OnnxExecutionProvider::Cuda]
        );
    }
}
