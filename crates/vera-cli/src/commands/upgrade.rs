//! `vera upgrade` — inspect or apply the binary update plan.

use anyhow::{Result, bail};
use serde::Serialize;

use crate::state;
use crate::update_check::{self, InstallMethodSource};

#[derive(Debug, Serialize)]
struct UpgradeReport {
    current_version: String,
    latest_version: Option<String>,
    update_available: bool,
    install_method: Option<String>,
    install_method_source: String,
    detected_install_methods: Vec<String>,
    update_command: String,
    apply_supported: bool,
    applied: bool,
    /// Version recorded by the installer after `--apply`, when it could be read.
    #[serde(skip_serializing_if = "Option::is_none")]
    installed_version: Option<String>,
}

pub fn run(apply: bool, json_output: bool) -> Result<()> {
    let status = update_check::binary_version_status(true);
    let mut report = UpgradeReport {
        current_version: status.current_version.to_string(),
        latest_version: status.latest_version.clone(),
        update_available: status.update_available(),
        install_method: status.install_method.clone(),
        install_method_source: install_method_source_name(status.install_method_source).to_string(),
        detected_install_methods: status.detected_install_methods.clone(),
        update_command: status.update_command(),
        apply_supported: status.can_apply_update(),
        applied: false,
        installed_version: None,
    };

    if !apply {
        return print_report(&report, json_output);
    }

    if !status.update_available() {
        if report.latest_version.is_none() {
            bail!("could not determine the latest Vera version; rerun `vera upgrade` later");
        }
        return print_report(&report, json_output);
    }

    if !status.can_apply_update() {
        bail!(apply_error(&status));
    }

    let method = status
        .install_method
        .as_deref()
        .expect("apply requires a resolved install method");
    update_check::apply_update(method)?;

    // The installer command exiting 0 does not mean the new version landed. A
    // package registry can lag behind a GitHub release, in which case the
    // installer resolves the version already installed and the upgrade
    // silently no-ops. Check whether the installed version actually changed
    // rather than reporting success on the exit code alone.
    report.installed_version = state::load_install_provenance()
        .ok()
        .and_then(|provenance| provenance.version);

    match verification_outcome(
        method,
        &report.current_version,
        report.latest_version.as_deref(),
        report.installed_version.as_deref(),
    ) {
        VerificationOutcome::Confirmed => report.applied = true,
        VerificationOutcome::Mismatch(message) => {
            print_report(&report, json_output)?;
            bail!(message);
        }
        VerificationOutcome::Unknown => {
            let message = "could not confirm the upgrade applied because the installed version is unavailable";
            if !json_output {
                eprintln!("Warning: {message}.");
            }
            print_report(&report, json_output)?;
            bail!(message);
        }
    }

    print_report(&report, json_output)
}

/// Describe why an applied upgrade did not take effect, or `None` when it did.
///
/// A package registry can lag behind a GitHub release, in which case the
/// installer keeps resolving the version already installed, every installer
/// command exits 0, and the upgrade silently no-ops.
///
/// The test is whether the installed version *changed*, not whether it equals
/// the advertised one. The installer resolves from the package registry while
/// `latest` comes from the GitHub release, and the two can legitimately differ:
/// a registry that is ahead of the release, or one that resolves an
/// intermediate version, still performed a real upgrade. Comparing against
/// `latest` reports those as failures and points the user at an older build.
fn applied_version_mismatch(
    method: &str,
    current: &str,
    latest: Option<&str>,
    installed: &str,
) -> Option<String> {
    if installed != current {
        return None;
    }
    let target = latest.unwrap_or("the new version");
    Some(format!(
        "upgrade did not take effect: still on {installed}.\n\
         This usually means the {method} package for {target} has not been published yet, so the \
         installer resolved {installed} again.\n\
         Hint: retry later, or install the {target} binary from \
         https://github.com/VeraTools/Vera/releases"
    ))
}

#[derive(Debug, PartialEq, Eq)]
enum VerificationOutcome {
    Confirmed,
    Mismatch(String),
    Unknown,
}

fn verification_outcome(
    method: &str,
    current: &str,
    latest: Option<&str>,
    installed: Option<&str>,
) -> VerificationOutcome {
    let Some(installed) = installed else {
        return VerificationOutcome::Unknown;
    };
    match applied_version_mismatch(method, current, latest, installed) {
        Some(message) => VerificationOutcome::Mismatch(message),
        None => VerificationOutcome::Confirmed,
    }
}

#[cfg(test)]
mod tests {
    use super::{VerificationOutcome, applied_version_mismatch, verification_outcome};

    #[test]
    fn reports_nothing_when_the_version_changed() {
        assert!(applied_version_mismatch("bun", "0.12.13", Some("1.0.0"), "1.0.0").is_none());
    }

    #[test]
    fn reports_the_no_op_with_the_version_and_the_install_method() {
        let message = applied_version_mismatch("bun", "0.12.13", Some("1.0.0"), "0.12.13")
            .expect("a stale registry must be reported, not treated as success");
        assert!(message.contains("0.12.13"), "{message}");
        assert!(message.contains("1.0.0"), "{message}");
        assert!(message.contains("bun"), "{message}");
    }

    /// The installer resolves from the package registry, `latest` comes from
    /// the GitHub release, and the two can legitimately disagree. A registry
    /// that is ahead of the release still performed a real upgrade, so it must
    /// not be reported as a failure telling the user to install the older
    /// GitHub build.
    #[test]
    fn accepts_a_version_the_registry_resolved_ahead_of_the_release() {
        assert!(
            applied_version_mismatch("bun", "1.0.0", Some("1.0.1"), "1.0.2").is_none(),
            "a registry ahead of the GitHub release is still a real upgrade"
        );
        assert_eq!(
            verification_outcome("bun", "1.0.0", Some("1.0.1"), Some("1.0.2")),
            VerificationOutcome::Confirmed
        );
    }

    /// An intermediate version is also a real upgrade.
    #[test]
    fn accepts_an_intermediate_version_between_current_and_latest() {
        assert_eq!(
            verification_outcome("pip", "1.0.0", Some("1.2.0"), Some("1.1.0")),
            VerificationOutcome::Confirmed
        );
    }

    #[test]
    fn detects_the_no_op_even_when_the_release_version_is_unknown() {
        assert!(
            applied_version_mismatch("bun", "0.12.13", None, "0.12.13").is_some(),
            "the no-op is detectable without knowing the advertised version"
        );
    }

    #[test]
    fn confirms_when_the_installed_version_moved_off_the_current_one() {
        assert_eq!(
            verification_outcome("bun", "0.12.13", Some("1.0.0"), Some("1.0.0")),
            VerificationOutcome::Confirmed
        );
    }

    #[test]
    fn keeps_a_no_op_upgrade_unapplied() {
        assert!(matches!(
            verification_outcome("bun", "0.12.13", Some("1.0.0"), Some("0.12.13")),
            VerificationOutcome::Mismatch(_)
        ));
    }

    #[test]
    fn keeps_unknown_versions_unapplied() {
        assert_eq!(
            verification_outcome("bun", "0.12.13", Some("1.0.0"), None),
            VerificationOutcome::Unknown
        );
    }
}

fn print_report(report: &UpgradeReport, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    println!("Current version: {}", report.current_version);
    if let Some(latest) = report.latest_version.as_deref() {
        println!("Latest version:  {latest}");
    } else {
        println!("Latest version:  unavailable");
    }
    println!(
        "Update status:    {}",
        if report.update_available {
            "update available"
        } else {
            "already up to date"
        }
    );
    println!(
        "Install method:   {} ({})",
        report.install_method.as_deref().unwrap_or("unknown"),
        report.install_method_source
    );
    if !report.detected_install_methods.is_empty() {
        println!(
            "Detected methods: {}",
            report.detected_install_methods.join(", ")
        );
    }
    println!("Update command:   {}", report.update_command);

    if report.applied {
        println!("Applied:          yes");
    } else if report.apply_supported {
        println!("Apply support:    yes (`vera upgrade --apply`)");
    } else {
        println!("Apply support:    no (manual update required)");
        print_manual_commands();
    }

    Ok(())
}

fn apply_error(status: &update_check::BinaryVersionStatus) -> String {
    match status.install_method_source {
        InstallMethodSource::Ambiguous => format!(
            "multiple install methods were detected ({}); refusing to guess.\nRun one of these manually:\n{}",
            status.detected_install_methods.join(", "),
            manual_command_lines()
        ),
        InstallMethodSource::Unknown => format!(
            "could not determine how Vera was installed.\nRun one of these manually:\n{}",
            manual_command_lines()
        ),
        _ => "could not determine a supported install method".to_string(),
    }
}

fn print_manual_commands() {
    println!("Manual options:");
    for method in update_check::supported_update_methods() {
        println!(
            "  {:<5} {}",
            format!("{method}:"),
            update_check::suggested_update_command(Some(method))
        );
    }
}

fn manual_command_lines() -> String {
    update_check::supported_update_methods()
        .iter()
        .map(|method| {
            format!(
                "  {}: {}",
                method,
                update_check::suggested_update_command(Some(method))
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn install_method_source_name(source: InstallMethodSource) -> &'static str {
    match source {
        InstallMethodSource::Provenance => "provenance",
        InstallMethodSource::Heuristic => "heuristic",
        InstallMethodSource::Ambiguous => "ambiguous",
        InstallMethodSource::Unknown => "unknown",
    }
}
