//! `vera doctor` — inspect the current Vera setup for common failures.

use std::path::Path;

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
    // A config we cannot parse is the situation doctor exists to report, so it
    // becomes a failing check rather than an early return. Continuing with the
    // defaults keeps the remaining checks available; they describe what the
    // process would actually do, since a broken stored config is ignored at
    // load time everywhere else too.
    let loaded = state::load_saved_config();
    checks.push(config_file_check(&config_path, loaded.as_ref().err()));
    let saved_config = loaded.unwrap_or_default();
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
            // Same reasoning as the stored config above: a bad
            // VERA_LOCAL_EMBEDDING_* override is a thing to report, not a
            // reason to stop. The runtime check below needs only the execution
            // provider, so it still runs and is still worth having.
            let embedding_model =
                match vera_core::local_models::LocalEmbeddingModelConfig::from_env() {
                    Ok(embedding_model) => {
                        checks.push(DoctorCheck {
                            name: "local-embedding-model",
                            status: CheckStatus::Ok,
                            detail: embedding_model.display_name(),
                        });
                        Some(embedding_model)
                    }
                    Err(err) => {
                        checks.push(DoctorCheck {
                            name: "local-embedding-model",
                            status: CheckStatus::Fail,
                            detail: one_line_error(&err),
                        });
                        None
                    }
                };
            let runtime_path = vera_core::local_models::ort_library_path_for_ep(ep)?;
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

            match embedding_model {
                Some(embedding_model) => {
                    // Inspecting the assets can fail on its own — an unreadable
                    // model directory, a revision that cannot be resolved — and
                    // that is another thing to report rather than a reason to
                    // abandon the checks already gathered.
                    match vera_core::local_models::inspect_local_model_files_for_ep(
                        ep,
                        &embedding_model,
                    ) {
                        Ok(model_assets) => {
                            let repair_hint = format!("run `vera repair --onnx-jina-{ep}`");
                            checks.push(local_model_assets_check(&model_assets, &repair_hint));
                            if probe {
                                checks.extend(probe_local_backend(
                                    ep,
                                    &runtime_path,
                                    &model_assets,
                                )?);
                            }
                        }
                        Err(err) => {
                            checks.push(DoctorCheck {
                                name: "local-model-assets",
                                status: CheckStatus::Fail,
                                detail: one_line_error(&err),
                            });
                            if probe {
                                checks.push(skipped_check(
                                    "probe",
                                    "skipped: the local model assets could not be inspected",
                                ));
                            }
                        }
                    }
                }
                None => {
                    // Which files to look for is derived from the model
                    // configuration that just failed to load, so there is
                    // nothing to inspect rather than nothing wrong.
                    checks.push(skipped_check(
                        "local-model-assets",
                        "skipped: the embedding model configuration could not be read",
                    ));
                    if probe {
                        checks.push(skipped_check(
                            "probe",
                            "skipped: the embedding model configuration could not be read",
                        ));
                    }
                }
            }
        }
        vera_core::config::InferenceBackend::PotionCode => {
            checks.push(DoctorCheck {
                name: "local-embedding-model",
                status: CheckStatus::Ok,
                detail: vera_core::local_models::potion_code_model_name(),
            });
            match vera_core::local_models::inspect_potion_code_model_files() {
                Ok(model_assets) => {
                    checks.push(local_model_assets_check(
                        &model_assets,
                        "run `vera repair --potion-code`",
                    ));
                    if probe {
                        checks.extend(probe_potion_backend(&model_assets));
                    }
                }
                Err(err) => {
                    checks.push(DoctorCheck {
                        name: "local-model-assets",
                        status: CheckStatus::Fail,
                        detail: one_line_error(&err),
                    });
                    if probe {
                        checks.push(skipped_check(
                            "probe",
                            "skipped: the local model assets could not be inspected",
                        ));
                    }
                }
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

    let verdict = check_verdict(&checks);
    let report = DoctorReport {
        version,
        overall_ok: verdict.is_ok(),
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

    verdict
}

/// The single source of the `overall_ok` field and the process exit code, so the
/// two cannot disagree.
///
/// Only `fail` checks count. Warnings and skips are excluded deliberately:
/// `version-check` warns whenever GitHub cannot be reached, and `config-file`,
/// `saved-backend` and `current-index` warn on a working install that has no
/// stored config or no index in the current directory. Exiting non-zero on those
/// would make `vera doctor` report failure on a healthy machine, which is worse
/// for the `vera doctor && vera index .` case than always exiting 0.
fn check_verdict(checks: &[DoctorCheck]) -> anyhow::Result<()> {
    let failed = failed_check_names(checks);
    if failed.is_empty() {
        return Ok(());
    }
    anyhow::bail!(
        "vera doctor found {} failing check{}: {}",
        failed.len(),
        if failed.len() == 1 { "" } else { "s" },
        failed.join(", ")
    )
}

fn failed_check_names(checks: &[DoctorCheck]) -> Vec<&'static str> {
    checks
        .iter()
        .filter(|check| matches!(check.status, CheckStatus::Fail))
        .map(|check| check.name)
        .collect()
}

fn check_env_group(name: &'static str, keys: &[&'static str]) -> DoctorCheck {
    // Non-UTF-8 values fail at request time just like unset ones.
    let values = keys
        .iter()
        .map(|key| std::env::var_os(key).and_then(|value| value.into_string().ok()))
        .collect::<Vec<_>>();
    check_env_values(name, keys.len(), &values)
}

/// A value counts as present only when it is usable: unset, non-UTF-8, and
/// empty/whitespace-only values (a common shellrc leftover after revoking a
/// key) all fail the same way at request time (#147).
fn check_env_values(name: &'static str, total: usize, values: &[Option<String>]) -> DoctorCheck {
    let present = values
        .iter()
        .flatten()
        .filter(|value| !value.trim().is_empty())
        .count();

    let status = match present {
        0 => CheckStatus::Warn,
        n if n == total => CheckStatus::Ok,
        _ => CheckStatus::Fail,
    };

    DoctorCheck {
        name,
        status,
        detail: format!("{present}/{total} variables present"),
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
    runtime_path: &std::path::Path,
    model_assets: &[vera_core::local_models::LocalModelAssetStatus],
) -> anyhow::Result<Vec<DoctorCheck>> {
    let mut checks = Vec::new();

    let ort_stage = result_check(
        "probe-ort-library",
        runtime_path.display().to_string(),
        vera_core::local_models::ensure_ort_runtime(Some(runtime_path)),
    );
    let ort_ok = matches!(ort_stage.status, CheckStatus::Ok);
    checks.push(ort_stage);

    let provider_stage = if ort_ok {
        result_check(
            "probe-provider-registration",
            format!("registered {}", ep),
            vera_core::embedding::local_provider::LocalEmbeddingProvider::probe_provider_registration(
                ep,
            ),
        )
    } else {
        skipped_check(
            "probe-provider-registration",
            "skipped because ONNX Runtime could not be initialized",
        )
    };
    let provider_ok = matches!(provider_stage.status, CheckStatus::Ok);
    checks.push(provider_stage);

    let dependencies_stage = if ep == vera_core::config::OnnxExecutionProvider::Cpu {
        skipped_check(
            "probe-dependencies",
            "skipped for the CPU backend because provider-specific shared-library checks are not needed",
        )
    } else if ort_ok {
        dependency_probe_check(ep, runtime_path)
    } else {
        skipped_check(
            "probe-dependencies",
            "skipped because ONNX Runtime could not be initialized",
        )
    };
    let dependencies_ok = !matches!(dependencies_stage.status, CheckStatus::Fail);
    checks.push(dependencies_stage);

    let embedding_session_stage = if ort_ok && provider_ok && dependencies_ok {
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

    let reranker_session_stage = if ort_ok && provider_ok && dependencies_ok {
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

    let tiny_inference_stage = if ort_ok && provider_ok && dependencies_ok {
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

    if ep != vera_core::config::OnnxExecutionProvider::Cpu {
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
        checks.push(DoctorCheck {
            name: "probe-reranker-coreml-cpu",
            status: CheckStatus::Warn,
            detail: "the reranker runs on CPU under CoreML (no CoreML-compatible reranker ONNX export exists); only the embedding model is GPU-accelerated. Reranking will be slower than embedding. If reranking latency is unacceptable, disable it with `vera config set retrieval.reranking_enabled false`".to_string(),
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

fn dependency_probe_check(
    ep: vera_core::config::OnnxExecutionProvider,
    runtime_path: &std::path::Path,
) -> DoctorCheck {
    if !runtime_path.exists() {
        return skipped_check(
            "probe-dependencies",
            format!("skipped because {} is missing", runtime_path.display()),
        );
    }

    match vera_core::local_models::inspect_provider_dependencies(ep, runtime_path) {
        Ok(Some(status)) => {
            if status.missing_details.is_empty() {
                DoctorCheck {
                    name: "probe-dependencies",
                    status: CheckStatus::Ok,
                    detail: "found no unresolved ONNX Runtime dependencies".to_string(),
                }
            } else {
                DoctorCheck {
                    name: "probe-dependencies",
                    status: CheckStatus::Fail,
                    detail: format!(
                        "missing shared libraries: {}",
                        status.missing_details.join("; ")
                    ),
                }
            }
        }
        Ok(None) => skipped_check(
            "probe-dependencies",
            "dependency inspection is currently available on Linux with `ldd` and macOS with `otool`",
        ),
        Err(err) => DoctorCheck {
            name: "probe-dependencies",
            status: CheckStatus::Warn,
            detail: one_line_error(&err),
        },
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

/// Status for the stored config, including the case where it cannot be parsed.
///
/// A config the CLI cannot read is the situation `vera doctor` exists to
/// report, so it is a failing check carrying the parse error rather than an
/// early return that discards every other check. Pure so the three outcomes can
/// be tested without a home directory.
fn config_file_check(config_path: &Path, load_error: Option<&anyhow::Error>) -> DoctorCheck {
    match load_error {
        Some(err) => DoctorCheck {
            name: "config-file",
            status: CheckStatus::Fail,
            detail: format!("{}: {}", config_path.display(), one_line_error(err)),
        },
        None => DoctorCheck {
            name: "config-file",
            status: if config_path.exists() {
                CheckStatus::Ok
            } else {
                CheckStatus::Warn
            },
            detail: config_path.display().to_string(),
        },
    }
}

fn one_line_error(err: &anyhow::Error) -> String {
    // `{:#}` rather than `to_string()`: the plain Display of an anyhow error is
    // the outermost context alone, so a wrapped failure renders as "failed to
    // parse persistent state" with the parse error — the part the user needs —
    // discarded. Every caller here is building a check detail for a human.
    format!("{err:#}")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(name: &'static str, status: CheckStatus) -> DoctorCheck {
        DoctorCheck {
            name,
            status,
            detail: String::new(),
        }
    }

    /// The four statuses a real run mixes, with two failures among them.
    fn mixed_checks() -> Vec<DoctorCheck> {
        vec![
            check("version-check", CheckStatus::Warn),
            check("config-file", CheckStatus::Ok),
            check("probe-dependencies", CheckStatus::Skip),
            check("onnx-runtime", CheckStatus::Fail),
            check("local-models", CheckStatus::Fail),
        ]
    }

    #[test]
    fn an_unparseable_config_fails_the_check_and_carries_the_parse_error() {
        let path = Path::new("/nonexistent/vera/config.json");
        let err = anyhow::anyhow!("EOF while parsing a value at line 2 column 0")
            .context("failed to parse persistent state");

        let check = config_file_check(path, Some(&err));

        assert!(
            matches!(check.status, CheckStatus::Fail),
            "an unparseable config must fail the check, not warn: {:?}",
            check.status
        );
        // The point of the check is that the user learns what is wrong with the
        // file, so both the path and the underlying parse error have to survive
        // into the detail. Asserting only the status would pass on a check that
        // reported "something is wrong" and nothing else.
        assert!(
            check.detail.contains("/nonexistent/vera/config.json"),
            "detail lost the path: {}",
            check.detail
        );
        assert!(
            check.detail.contains("EOF while parsing a value"),
            "detail lost the parse error: {}",
            check.detail
        );
        assert_eq!(
            failed_check_names(&[check]),
            vec!["config-file"],
            "the failure must reach the verdict, so the exit code is non-zero"
        );
    }

    #[test]
    fn a_readable_config_keeps_the_existing_ok_and_warn_split() {
        // The negative half: introducing the failure case must not disturb the
        // two outcomes that were already there.
        let temp = tempfile::tempdir().unwrap();
        let present = temp.path().join("config.json");
        std::fs::write(&present, "{}").unwrap();
        let absent = temp.path().join("missing.json");

        let found = config_file_check(&present, None);
        assert!(matches!(found.status, CheckStatus::Ok), "{found:?}");
        assert_eq!(found.detail, present.display().to_string());

        let not_found = config_file_check(&absent, None);
        assert!(
            matches!(not_found.status, CheckStatus::Warn),
            "{not_found:?}"
        );
        assert_eq!(not_found.detail, absent.display().to_string());

        assert!(
            failed_check_names(&[found, not_found]).is_empty(),
            "a missing config is not a failure; exit code must stay 0"
        );
    }

    #[test]
    fn failing_checks_produce_an_error_naming_every_one_of_them() {
        let checks = mixed_checks();
        assert_eq!(
            failed_check_names(&checks),
            vec!["onnx-runtime", "local-models"],
            "fixture must carry exactly the two failures the assertions below expect"
        );

        let err = match check_verdict(&checks) {
            Ok(()) => panic!(
                "vera doctor reported success while these checks failed: {:?}",
                failed_check_names(&checks)
            ),
            Err(err) => err.to_string(),
        };
        assert_eq!(
            err, "vera doctor found 2 failing checks: onnx-runtime, local-models",
            "the error main turns into exit 1 must name both failures"
        );
    }

    #[test]
    fn a_single_failure_is_named_in_the_singular() {
        let checks = vec![
            check("version-check", CheckStatus::Warn),
            check("onnx-runtime", CheckStatus::Fail),
        ];
        let err = match check_verdict(&checks) {
            Ok(()) => panic!("vera doctor reported success while onnx-runtime failed"),
            Err(err) => err.to_string(),
        };
        assert_eq!(err, "vera doctor found 1 failing check: onnx-runtime");
    }

    #[test]
    fn warnings_and_skips_alone_keep_the_exit_code_at_zero() {
        let mut checks = mixed_checks();
        checks.retain(|check| !matches!(check.status, CheckStatus::Fail));

        // Presence first: without a warn and a skip in the fixture this test
        // would pass against an implementation that failed on either.
        assert!(
            checks
                .iter()
                .any(|check| matches!(check.status, CheckStatus::Warn)),
            "fixture must contain a warn check"
        );
        assert!(
            checks
                .iter()
                .any(|check| matches!(check.status, CheckStatus::Skip)),
            "fixture must contain a skip check"
        );
        assert!(failed_check_names(&checks).is_empty());

        assert!(
            check_verdict(&checks).is_ok(),
            "a warning must not fail the run: version-check, config-file, \
             saved-backend and current-index all warn on a healthy machine"
        );
    }

    /// #147 regression: empty and whitespace-only API env values are shellrc
    /// leftovers that no consumer accepts, so they must score as absent.
    #[test]
    fn blank_env_values_score_as_absent() {
        let values = vec![
            Some("https://api.example.com".to_string()),
            Some("  \t ".to_string()),
            Some(String::new()),
        ];
        let check = check_env_values("embedding-api", 3, &values);
        assert!(matches!(check.status, CheckStatus::Fail));
        assert_eq!(check.detail, "1/3 variables present");
    }

    #[test]
    fn complete_env_group_scores_ok_and_missing_group_warns() {
        let complete = vec![Some("a".to_string()), Some("b".to_string())];
        assert!(matches!(
            check_env_values("embedding-api", 2, &complete).status,
            CheckStatus::Ok
        ));

        let absent = vec![None, None];
        let check = check_env_values("reranker-api", 2, &absent);
        assert!(matches!(check.status, CheckStatus::Warn));
        assert_eq!(check.detail, "0/2 variables present");
    }
}
