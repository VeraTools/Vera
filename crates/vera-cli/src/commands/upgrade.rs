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

    // Read what the installer recorded *before* running it. The no-op is
    // "the installer resolved the same version again", so the comparison has to
    // be against the installer's own previous record. Comparing against the
    // running binary's version instead would misread a stale or foreign
    // provenance record as a successful upgrade.
    let baseline_version = state::load_install_provenance()
        .ok()
        .and_then(|provenance| provenance.version);

    update_check::apply_update(method)?;

    // The installer command exiting 0 does not mean the new version landed. A
    // package registry can lag behind a GitHub release, in which case the
    // installer resolves the version already installed and the upgrade
    // silently no-ops. Check whether the recorded version actually changed
    // rather than reporting success on the exit code alone.
    report.installed_version = state::load_install_provenance()
        .ok()
        .and_then(|provenance| provenance.version);

    match verification_outcome(
        method,
        baseline_version.as_deref(),
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
/// The test is whether the installer's own record *changed*, comparing the
/// provenance written before the install against the one written after.
///
/// Not against `latest`: that comes from the GitHub release while the installer
/// resolves from a package registry, and the two can legitimately differ. A
/// registry ahead of the release, or one resolving an intermediate version,
/// still performed a real upgrade, and comparing against `latest` would report
/// those as failures and point the user at an older build.
///
/// Not against the running binary's version either: provenance can be stale or
/// belong to another installation, so a no-op could leave it differing from the
/// running version and read as success.
///
/// `latest` is used only to word the hint.
fn applied_version_mismatch(
    method: &str,
    baseline: &str,
    latest: Option<&str>,
    installed: &str,
) -> Option<String> {
    if installed != baseline {
        return None;
    }
    let target = latest.unwrap_or("new version");
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

/// Without a provenance record from either side of the install there is nothing
/// to compare, so the outcome is [`VerificationOutcome::Unknown`] rather than an
/// assumed success.
fn verification_outcome(
    method: &str,
    baseline: Option<&str>,
    latest: Option<&str>,
    installed: Option<&str>,
) -> VerificationOutcome {
    let (Some(baseline), Some(installed)) = (baseline, installed) else {
        return VerificationOutcome::Unknown;
    };
    match applied_version_mismatch(method, baseline, latest, installed) {
        Some(message) => VerificationOutcome::Mismatch(message),
        None => VerificationOutcome::Confirmed,
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

#[cfg(test)]
mod tests {
    use super::{VerificationOutcome, applied_version_mismatch, verification_outcome};

    #[test]
    fn reports_nothing_when_the_recorded_version_changed() {
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
    /// ahead of the release still performed a real upgrade, so it must not be
    /// reported as a failure telling the user to install the older build.
    #[test]
    fn accepts_a_version_the_registry_resolved_ahead_of_the_release() {
        assert!(
            applied_version_mismatch("bun", "1.0.0", Some("1.0.1"), "1.0.2").is_none(),
            "a registry ahead of the GitHub release is still a real upgrade"
        );
        assert_eq!(
            verification_outcome("bun", Some("1.0.0"), Some("1.0.1"), Some("1.0.2")),
            VerificationOutcome::Confirmed
        );
    }

    /// An intermediate version is also a real upgrade.
    #[test]
    fn accepts_an_intermediate_version_between_baseline_and_latest() {
        assert_eq!(
            verification_outcome("pip", Some("1.0.0"), Some("1.2.0"), Some("1.1.0")),
            VerificationOutcome::Confirmed
        );
    }

    /// The baseline is the installer's own previous record, not the running
    /// binary. A stale or foreign provenance record that happens to differ from
    /// the running version must not be read as a successful upgrade.
    #[test]
    fn detects_the_no_op_when_provenance_disagrees_with_the_running_binary() {
        assert!(matches!(
            verification_outcome("bun", Some("0.11.0"), Some("1.0.0"), Some("0.11.0")),
            VerificationOutcome::Mismatch(_)
        ));
    }

    /// `latest` only words the hint, so an unknown release must still produce
    /// readable prose rather than "the the new version".
    #[test]
    fn renders_a_readable_hint_when_the_release_version_is_unknown() {
        let message = applied_version_mismatch("bun", "0.12.13", None, "0.12.13")
            .expect("the no-op is detectable without knowing the advertised version");
        assert!(!message.contains("the the"), "{message}");
        assert!(
            message.contains("install the new version binary"),
            "{message}"
        );
    }

    #[test]
    fn keeps_a_no_op_upgrade_unapplied() {
        assert!(matches!(
            verification_outcome("bun", Some("0.12.13"), Some("1.0.0"), Some("0.12.13")),
            VerificationOutcome::Mismatch(_)
        ));
    }

    /// Without a record from either side of the install there is nothing to
    /// compare, so neither success nor failure may be assumed.
    #[test]
    fn keeps_unverifiable_upgrades_unapplied() {
        assert_eq!(
            verification_outcome("bun", Some("0.12.13"), Some("1.0.0"), None),
            VerificationOutcome::Unknown
        );
        assert_eq!(
            verification_outcome("bun", None, Some("1.0.0"), Some("1.0.0")),
            VerificationOutcome::Unknown
        );
    }
}
