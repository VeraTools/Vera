//! `vera upgrade` — inspect or apply the binary update plan.

use std::cmp::Ordering;
use std::path::Path;

use anyhow::{Result, bail};
use semver::Version;
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
    // be against the installer's own previous record. Only trust that record as
    // a baseline when it identifies the binary that is currently running;
    // otherwise a stale or foreign record could make an unchanged binary look
    // upgraded when the wrapper rewrites install.json.
    let baseline_provenance = state::load_install_provenance().ok();
    let current_executable = std::env::current_exe().ok();
    let baseline_version = verified_baseline_version(
        baseline_provenance.as_ref(),
        &report.current_version,
        current_executable.as_deref(),
    );

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
        baseline_version,
        report.latest_version.as_deref(),
        report.installed_version.as_deref(),
    ) {
        VerificationOutcome::Confirmed => report.applied = true,
        VerificationOutcome::Mismatch(message) => {
            print_report(&report, json_output)?;
            bail!(message);
        }
        VerificationOutcome::Unknown => {
            let message = "could not confirm the upgrade because install provenance is missing or does not identify the running binary";
            if !json_output {
                eprintln!("Warning: {message}.");
            }
            print_report(&report, json_output)?;
            bail!(message);
        }
    }

    print_report(&report, json_output)
}

fn verified_baseline_version<'a>(
    provenance: Option<&'a state::InstallProvenance>,
    current_version: &str,
    current_executable: Option<&Path>,
) -> Option<&'a str> {
    let provenance = provenance?;
    let version = provenance.version.as_deref()?;
    if !same_version(version, current_version) {
        return None;
    }

    let current_executable = current_executable?.canonicalize().ok()?;
    let recorded_executable = Path::new(provenance.binary_path.as_deref()?)
        .canonicalize()
        .ok()?;
    (recorded_executable == current_executable).then_some(version)
}

fn normalize_version(version: &str) -> &str {
    version.strip_prefix('v').unwrap_or(version)
}

fn same_version(left: &str, right: &str) -> bool {
    normalize_version(left) == normalize_version(right)
}

fn version_order(baseline: &str, installed: &str) -> Option<Ordering> {
    let baseline = Version::parse(normalize_version(baseline)).ok()?;
    let installed = Version::parse(normalize_version(installed)).ok()?;
    Some(installed.cmp(&baseline))
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
    if !same_version(installed, baseline) {
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

fn applied_version_downgrade(
    method: &str,
    baseline: &str,
    latest: Option<&str>,
    installed: &str,
) -> String {
    let target = latest.unwrap_or("new version");
    format!(
        "upgrade resolved older version {installed} than the current {baseline}.\n\
         This usually means the {method} package for {target} resolved to an older release.\n\
         Hint: retry later, or install the {target} binary from \
         https://github.com/VeraTools/Vera/releases"
    )
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
    if let Some(message) = applied_version_mismatch(method, baseline, latest, installed) {
        return VerificationOutcome::Mismatch(message);
    }
    match version_order(baseline, installed) {
        Some(Ordering::Less) => VerificationOutcome::Mismatch(applied_version_downgrade(
            method, baseline, latest, installed,
        )),
        Some(Ordering::Equal | Ordering::Greater) => VerificationOutcome::Confirmed,
        None => VerificationOutcome::Unknown,
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
    use super::{
        VerificationOutcome, applied_version_mismatch, same_version, verification_outcome,
        verified_baseline_version,
    };
    use crate::state::InstallProvenance;

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

    /// A provenance record only qualifies as the baseline when it identifies
    /// the running binary: same version (v-prefix tolerant) and same binary
    /// path. Stale, foreign, or missing records make the upgrade unverifiable
    /// instead of letting a rewritten record pose as a successful upgrade.
    #[test]
    fn baseline_requires_provenance_identifying_the_running_binary() {
        let exe = std::env::current_exe().unwrap();
        let matching = InstallProvenance {
            version: Some("1.0.0".to_string()),
            binary_path: Some(exe.to_string_lossy().to_string()),
            ..Default::default()
        };
        assert_eq!(
            verified_baseline_version(Some(&matching), "1.0.0", Some(exe.as_path())),
            Some("1.0.0")
        );

        // Stale record: the recorded version is not the running version.
        assert_eq!(
            verified_baseline_version(Some(&matching), "1.0.1", Some(exe.as_path())),
            None
        );

        // Foreign record: same version, but recorded for another binary.
        let foreign = InstallProvenance {
            binary_path: Some("/dev/null".to_string()),
            ..matching.clone()
        };
        assert_eq!(
            verified_baseline_version(Some(&foreign), "1.0.0", Some(exe.as_path())),
            None
        );

        // No record, or a record without the fields needed to verify.
        assert_eq!(
            verified_baseline_version(None, "1.0.0", Some(exe.as_path())),
            None
        );
        let incomplete = InstallProvenance {
            version: Some("1.0.0".to_string()),
            ..Default::default()
        };
        assert_eq!(
            verified_baseline_version(Some(&incomplete), "1.0.0", Some(exe.as_path())),
            None
        );
    }

    #[test]
    fn version_comparison_ignores_a_v_prefix() {
        assert!(same_version("v1.0.0", "1.0.0"));
        assert!(!same_version("v1.0.0", "1.0.1"));
    }

    /// A registry resolving an older version than the baseline (yank,
    /// rollback, channel mismatch) is a failed upgrade, not a success.
    #[test]
    fn flags_a_downgrade_as_a_failed_upgrade() {
        let outcome = verification_outcome("bun", Some("1.2.0"), Some("1.3.0"), Some("1.1.0"));
        match outcome {
            VerificationOutcome::Mismatch(message) => {
                assert!(message.contains("older version"), "{message}");
                assert!(message.contains("1.1.0"), "{message}");
                assert!(message.contains("bun"), "{message}");
            }
            other => panic!("a downgrade must not be reported as success: {other:?}"),
        }
    }

    /// Versions that cannot be parsed change the record but cannot be ordered,
    /// so the outcome is unverifiable rather than assumed success.
    #[test]
    fn unparseable_version_change_is_unverifiable() {
        assert_eq!(
            verification_outcome(
                "bun",
                Some("nightly-abc"),
                Some("1.0.0"),
                Some("nightly-def")
            ),
            VerificationOutcome::Unknown
        );
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
