//! `vera uninstall` — remove Vera binary, models, config, and agent skills.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::agent;
use crate::state;

/// Candidate directories where the shim may have been placed.
fn shim_candidates(home: &Path, user_bin_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = user_bin_dir {
        dirs.push(dir.to_path_buf());
    }
    #[cfg(windows)]
    {
        dirs.push(home.join("AppData").join("Roaming").join("npm"));
        dirs.push(
            home.join("AppData")
                .join("Local")
                .join("Programs")
                .join("Vera")
                .join("bin"),
        );
    }
    #[cfg(not(windows))]
    {
        dirs.push(home.join(".local").join("bin"));
        dirs.push(home.join(".cargo").join("bin"));
        dirs.push(home.join("bin"));
    }
    dirs
}

/// File names Vera may occupy in the candidate directories: the script shim
/// the npm/pip/bun installers write, plus the binary `cargo install` places
/// next to them (#212). Only exact matches are treated as Vera's.
fn entry_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["vera.cmd", "vera.exe"]
    } else {
        &["vera"]
    }
}

/// The two shapes of PATH entry that belong to Vera.
enum LaunchEntry {
    /// Script shim written by the installers; remove directly.
    Shim,
    /// Real ELF/Mach-O/PE binary left by `cargo install` (#212); also ours.
    CargoBinary,
}

impl LaunchEntry {
    fn removed_label(&self) -> &'static str {
        match self {
            Self::Shim => "PATH shim",
            Self::CargoBinary => "cargo-installed binary",
        }
    }
}

/// Recognizes a candidate path as a removable Vera launcher, or leaves it
/// unclassified so unrelated files stay untouched.
///
/// A shim is UTF-8 text mentioning vera or a symlink pointing at one. The
/// cargo-installed binary is neither: it fails UTF-8 decoding by construction
/// (#212), so only "unreadable as text plus a regular executable file with an
/// exact Vera entry name" attributes it to cargo without grabbing anything
/// else named `vera`.
fn classify_launch_entry(entry: &Path) -> Option<LaunchEntry> {
    let read_as_text = fs::read_to_string(entry);
    if read_as_text
        .as_ref()
        .is_ok_and(|text| text.contains("vera"))
        || fs::read_link(entry).is_ok_and(|target| target.to_string_lossy().contains("vera"))
    {
        return Some(LaunchEntry::Shim);
    }
    // Decodable text that never mentions vera belongs to someone else.
    let is_cargo_binary = read_as_text.is_err()
        && fs::symlink_metadata(entry).is_ok_and(|meta| meta.is_file())
        && is_executable(entry);
    is_cargo_binary.then_some(LaunchEntry::CargoBinary)
}

/// Executability where the platform tracks it. Windows has no file mode bit;
/// membership among the exact entry names above is what restricts candidates.
fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path)
            .map(|meta| meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn configured_user_bin_dir() -> Option<PathBuf> {
    std::env::var_os("VERA_USER_BIN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub fn run(json_output: bool) -> Result<()> {
    let home = state::user_home_dir()?;
    let vera_home = state::vera_dir()?;
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let user_bin_dir = configured_user_bin_dir();

    run_at(
        &home,
        &vera_home,
        &cwd,
        user_bin_dir.as_deref(),
        json_output,
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    )
}

fn run_at(
    home: &Path,
    vera_home: &Path,
    cwd: &Path,
    user_bin_dir: Option<&Path>,
    json_output: bool,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<()> {
    let mut removed = Vec::new();

    // 1. Remove agent skill files (all clients, all scopes).
    let skill_removal = fold_skill_removal(agent::remove_all_skills(cwd, home));
    // Uninstall continues past a failure so the rest of the cleanup still runs,
    // but it must not end in success: the failures are reported on stderr here,
    // echoed into the exit code at the bottom of this function, and reflected
    // as `"complete": false` in the JSON document.
    for error in &skill_removal.failures {
        writeln!(stderr, "  {error:#}")?;
    }
    let removed_skills: Vec<&str> = skill_removal
        .reports
        .iter()
        .filter(|report| report.was_removed())
        .map(|report| report.path())
        .collect();
    if !removed_skills.is_empty() {
        removed.push("agent skills");
    }
    if !json_output {
        agent::write_removed_skill_locations(&skill_removal, stdout)?;
    }

    // 2. Remove Vera data directory (binary cache, models, libs, config, credentials).
    if vera_home.exists() {
        fs::remove_dir_all(vera_home)?;
        removed.push("vera data dir");
        if !json_output {
            writeln!(stderr, "  Removed {}", vera_home.display())?;
        }
    }

    // 3. Remove the PATH shim or cargo-installed launcher binary (#212).
    let mut removed_any_entry = false;
    // Removals that were skipped: the trailing report must name what stayed
    // instead of claiming a complete uninstall.
    let mut leftover_failures: Vec<(PathBuf, anyhow::Error)> = Vec::new();
    for dir in shim_candidates(home, user_bin_dir) {
        for name in entry_names() {
            let entry = dir.join(name);
            if !entry.exists() {
                continue;
            }
            let Some(kind) = classify_launch_entry(&entry) else {
                // Not ours: leave it alone silently, as before.
                continue;
            };
            match fs::remove_file(&entry) {
                Ok(()) => {
                    if !removed_any_entry {
                        removed.push(kind.removed_label());
                    }
                    removed_any_entry = true;
                    if !json_output {
                        writeln!(
                            stderr,
                            "  Removed {} {}",
                            kind.removed_label(),
                            entry.display()
                        )?;
                    }
                }
                Err(error) => {
                    writeln!(stderr, "  Left in place: {}: {error}", entry.display())?;
                    leftover_failures.push((entry.clone(), error.into()));
                }
            }
        }
    }

    // Completion covers every phase that can strand files on disk: agent
    // skills and PATH entries.
    let complete = skill_removal.failures.is_empty() && leftover_failures.is_empty();

    if json_output {
        let mut document = serde_json::json!({
            "uninstalled": true,
            "complete": complete,
            "removed": removed,
            "skills": removed_skills,
        });
        if !leftover_failures.is_empty() {
            // #212: a partial removal must not masquerade as a clean one, and
            // the report has to say what was left behind.
            document["left_behind"] = serde_json::json!(
                leftover_failures
                    .iter()
                    .map(|(path, _)| path.display().to_string())
                    .collect::<Vec<_>>()
            );
        }
        writeln!(stdout, "{}", document)?;
    } else {
        writeln!(stderr)?;
        if complete {
            writeln!(stderr, "Vera has been uninstalled.")?;
        } else {
            // The per-item stderr lines above carry the specifics.
            writeln!(stderr, "Vera was partially uninstalled.")?;
        }
        writeln!(
            stderr,
            "Per-project indexes (.vera/ in each project) were not removed."
        )?;
    }

    // Mirror `agent::do_remove`: report everything first, then let the first
    // failure fail the command so automation cannot read exit 0 while skill
    // directories or PATH launchers survive on disk.
    if let Some(first) = skill_removal.failures.into_iter().next() {
        return Err(first.context("uninstall did not complete"));
    }
    if let Some((left_behind, error)) = leftover_failures.into_iter().next() {
        return Err(error.context(format!(
            "uninstall did not complete; {} remains on PATH",
            left_behind.display()
        )));
    }
    Ok(())
}

fn fold_skill_removal(removal: Result<agent::SkillRemoval>) -> agent::SkillRemoval {
    match removal {
        Ok(removal) => removal,
        Err(e) => {
            tracing::warn!("failed to resolve agent skill locations: {e:#}");
            // A location that cannot be resolved is still an unfinished
            // uninstall: keep it as a failure instead of letting the run
            // report a complete removal.
            let mut removal = agent::SkillRemoval::default();
            removal.failures.push(e);
            removal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    struct Roots {
        _temp: tempfile::TempDir,
        home: PathBuf,
        cwd: PathBuf,
        vera_home: PathBuf,
        user_bin_dir: PathBuf,
    }

    /// A home/project tree that exists only under a temp directory, so no test
    /// can reach a real skill install.
    fn roots() -> Roots {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let cwd = temp.path().join("project");
        let vera_home = home.join(".vera");
        let user_bin_dir = temp.path().join("bin");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&cwd).unwrap();
        fs::create_dir_all(&user_bin_dir).unwrap();
        Roots {
            _temp: temp,
            home,
            cwd,
            vera_home,
            user_bin_dir,
        }
    }

    /// Install a fake Claude global skill: `<home>/.claude/skills/vera/SKILL.md`.
    fn install_claude_global_skill(home: &Path) -> PathBuf {
        let path = home.join(".claude").join("skills").join("vera");
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("SKILL.md"), "test").unwrap();
        path
    }

    fn uninstall(roots: &Roots, json_output: bool) -> (String, String) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        run_at(
            &roots.home,
            &roots.vera_home,
            &roots.cwd,
            Some(roots.user_bin_dir.as_path()),
            json_output,
            &mut stdout,
            &mut stderr,
        )
        .unwrap_or_else(|e| panic!("a clean uninstall must succeed, got: {e:#}"));
        (
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        )
    }

    /// A run whose cleanup fails partway: captures what was printed even
    /// though the command reports failure.
    #[cfg(unix)]
    fn capture_failing_run(roots: &Roots, json_output: bool) -> (Option<String>, String, String) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let error = run_at(
            &roots.home,
            &roots.vera_home,
            &roots.cwd,
            Some(roots.user_bin_dir.as_path()),
            json_output,
            &mut stdout,
            &mut stderr,
        )
        .err()
        .map(|e| e.to_string());
        (
            error,
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        )
    }

    #[test]
    fn uninstall_json_emits_exactly_one_document() {
        let roots = roots();
        let skill = install_claude_global_skill(&roots.home);

        let (stdout, _) = uninstall(&roots, true);

        // Strict parse: this is what `json.load` and `serde_json::from_str` do,
        // and it fails with trailing input if a second document is printed.
        let document: serde_json::Value = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("stdout is not a single JSON document ({e}): {stdout}"));

        assert_eq!(document["uninstalled"], serde_json::json!(true));
        assert_eq!(
            document["skills"],
            serde_json::json!([skill.display().to_string()])
        );
        assert!(!skill.exists());
    }

    #[test]
    fn uninstall_json_claims_only_categories_that_were_removed() {
        let roots = roots();

        let (stdout, _) = uninstall(&roots, true);

        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(document["removed"], serde_json::json!([]));
        assert_eq!(document["skills"], serde_json::json!([]));
    }

    #[test]
    fn uninstall_human_output_lists_only_removed_locations() {
        let roots = roots();
        let skill = install_claude_global_skill(&roots.home);
        let not_installed = roots.home.join(".gemini").join("skills").join("vera");

        let (stdout, _) = uninstall(&roots, false);

        assert!(stdout.contains("Removed Vera skill from:"), "{stdout}");
        assert!(stdout.contains(&skill.display().to_string()), "{stdout}");
        assert!(
            !stdout.contains(&not_installed.display().to_string()),
            "{stdout}"
        );
        // Heading, blank line, and exactly one row.
        assert_eq!(stdout.lines().count(), 3, "{stdout}");
    }

    /// A skill directory that cannot be deleted, ordered after Claude's so an
    /// earlier location is already gone by the time it fails.
    #[cfg(unix)]
    fn install_unremovable_gemini_global_skill(home: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = home.join(".gemini").join("skills").join("vera");
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("SKILL.md"), "test").unwrap();
        let parent = home.join(".gemini").join("skills");
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o555)).unwrap();
        parent
    }

    #[cfg(unix)]
    fn allow_cleanup(parent: &Path) {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn uninstall_json_reports_skills_removed_before_a_later_removal_failed() {
        let roots = roots();
        let claude = install_claude_global_skill(&roots.home);
        let locked = install_unremovable_gemini_global_skill(&roots.home);

        // #149 regression: a partial removal is a failed uninstall. The JSON
        // document still reaches stdout, but `complete` must be false and the
        // process error must be set.
        let (error, stdout, _) = capture_failing_run(&roots, true);
        let claude_was_deleted = !claude.exists();
        allow_cleanup(&locked);

        assert!(
            claude_was_deleted,
            "fixture does not discriminate: the earlier skill was never deleted"
        );
        assert!(
            error
                .as_deref()
                .is_some_and(|e| e.contains("uninstall did not complete")),
            "partial removal must fail the command: {error:?}"
        );
        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(document["complete"], serde_json::json!(false), "{stdout}");
        assert_eq!(
            document["skills"],
            serde_json::json!([claude.display().to_string()]),
            "{stdout}"
        );
        assert_eq!(
            document["removed"],
            serde_json::json!(["agent skills"]),
            "{stdout}"
        );
    }

    #[test]
    fn uninstall_json_marks_a_clean_removal_complete() {
        let roots = roots();
        install_claude_global_skill(&roots.home);

        let (stdout, _) = uninstall(&roots, true);

        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(document["complete"], serde_json::json!(true));
    }

    #[cfg(unix)]
    #[test]
    fn uninstall_human_output_names_skills_removed_before_a_later_removal_failed() {
        let roots = roots();
        let claude = install_claude_global_skill(&roots.home);
        let locked = install_unremovable_gemini_global_skill(&roots.home);

        let (error, stdout, stderr) = capture_failing_run(&roots, false);
        let claude_was_deleted = !claude.exists();
        allow_cleanup(&locked);

        assert!(
            claude_was_deleted,
            "fixture does not discriminate: the earlier skill was never deleted"
        );
        assert!(error.is_some(), "partial removal must fail the command");
        assert!(stdout.contains(&claude.display().to_string()), "{stdout}");
        assert!(
            !stdout.contains("No Vera skill installations found."),
            "claimed nothing was installed after deleting {}: {stdout}",
            claude.display()
        );
        assert!(
            stderr.contains("failed to remove installed skill at"),
            "the failure was never reported: {stderr}"
        );
        // #149: the success line must not contradict the reported failures.
        assert!(
            !stderr.contains("Vera has been uninstalled."),
            "claimed a complete uninstall despite the failure: {stderr}"
        );
        assert!(
            stderr.contains("Vera was partially uninstalled"),
            "the partial outcome was never stated: {stderr}"
        );
    }

    /// The only installed skill fails to delete: nothing was removed, but
    /// claiming nothing was *installed* would be a different lie.
    #[cfg(unix)]
    #[test]
    fn uninstall_human_output_does_not_claim_nothing_was_installed_when_removal_failed() {
        let roots = roots();
        let locked = install_unremovable_gemini_global_skill(&roots.home);

        let (error, stdout, stderr) = capture_failing_run(&roots, false);
        allow_cleanup(&locked);

        assert!(error.is_some(), "partial removal must fail the command");
        assert!(
            !stdout.contains("No Vera skill installations found."),
            "{stdout}"
        );
        assert!(
            stderr.contains("failed to remove installed skill at"),
            "{stderr}"
        );
    }

    /// A skill whose `SKILL.md` cannot be stat'd at all, because its own
    /// directory denies traversal. `Path::exists` reports that as absent.
    #[cfg(unix)]
    fn install_uninspectable_claude_global_skill(home: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = home.join(".claude").join("skills").join("vera");
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("SKILL.md"), "test").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
        path
    }

    /// An installed skill that cannot be inspected is not an absent one. Reporting
    /// it as absent leaves it on disk while claiming it was never there.
    #[cfg(unix)]
    #[test]
    fn uninstall_does_not_report_an_uninspectable_skill_as_absent() {
        let roots = roots();
        let locked = install_uninspectable_claude_global_skill(&roots.home);

        let (error, stdout, stderr) = capture_failing_run(&roots, false);
        allow_cleanup(&locked);
        let skill_survived = locked.join("SKILL.md").exists();

        assert!(error.is_some(), "partial removal must fail the command");
        assert!(
            skill_survived,
            "fixture does not discriminate: the skill was deleted after all"
        );
        assert!(
            !stdout.contains("No Vera skill installations found."),
            "claimed nothing was installed while {} was still on disk: {stdout}",
            locked.display()
        );
        assert!(
            stderr.contains("failed to check for an installed skill at"),
            "the inspection failure was never reported: {stderr}"
        );
    }

    /// A location that cannot even be resolved (for example an unreadable
    /// project directory) must count as an unfinished uninstall, not vanish
    /// into a default that reports `complete: true`.
    #[test]
    fn a_failed_location_resolution_is_kept_as_a_failure() {
        let removal = fold_skill_removal(Err(anyhow::anyhow!("cannot list agent locations")));

        assert_eq!(removal.failures.len(), 1);
        assert!(
            removal.failures[0]
                .to_string()
                .contains("cannot list agent locations"),
            "the resolution error lost its message: {:#}",
            removal.failures[0]
        );
    }

    #[test]
    fn uninstall_human_output_reports_nothing_when_no_skills_are_installed() {
        let roots = roots();

        let (stdout, stderr) = uninstall(&roots, false);

        assert_eq!(stdout.trim(), "No Vera skill installations found.");
        assert!(stderr.contains("Vera has been uninstalled."));
    }

    /// Stands in for the payload `cargo install vera` writes (#212): raw
    /// bytes starting with ELF magic, which fails UTF-8 decoding exactly like
    /// the real ELF/Mach-O binary does.
    #[cfg(unix)]
    fn install_cargo_binary(home: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let bin_dir = home.join(".cargo").join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let path = bin_dir.join("vera");
        fs::write(&path, [0x7f, b'E', b'L', b'F', 0xcf]).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// #212 regression: a cargo-installed launcher used to fail the text read,
    /// get skipped, and stay on PATH while the uninstall claimed success.
    #[cfg(unix)]
    #[test]
    fn uninstall_removes_the_cargo_installed_binary_from_the_path() {
        let roots = roots();
        let binary = install_cargo_binary(&roots.home);

        let (stdout, _) = uninstall(&roots, true);

        assert!(
            !binary.exists(),
            "the cargo-installed {} stayed on PATH",
            binary.display()
        );
        let document: serde_json::Value = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("stdout is not a single JSON document ({e}): {stdout}"));
        assert_eq!(document["complete"], serde_json::json!(true), "{stdout}");
        assert!(
            document["removed"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tag| tag == "cargo-installed binary"),
            "{stdout}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn uninstall_human_output_reports_the_removed_cargo_binary() {
        let roots = roots();
        install_cargo_binary(&roots.home);

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            stderr.contains("Removed cargo-installed binary"),
            "{stderr}"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// Two lookalikes that are not ours: a readable script that never mentions
    /// vera, and unreadable data without an executable bit. Neither may be
    /// deleted, and skipping them silently keeps the run complete (#212).
    #[cfg(unix)]
    #[test]
    fn uninstall_leaves_foreign_lookalikes_in_place_without_failing() {
        use std::os::unix::fs::PermissionsExt;
        let roots = roots();
        let script_dir = roots.home.join(".local").join("bin");
        fs::create_dir_all(&script_dir).unwrap();
        let foreign_script = script_dir.join("vera");
        // The content must not contain the string "vera", or it would look
        // like our shim by today's matching rule.
        fs::write(&foreign_script, "#!/bin/sh\necho hello\n").unwrap();
        fs::set_permissions(&foreign_script, fs::Permissions::from_mode(0o755)).unwrap();
        let data_dir = roots.home.join("bin");
        fs::create_dir_all(&data_dir).unwrap();
        let foreign_data = data_dir.join("vera");
        fs::write(&foreign_data, [0xcf]).unwrap();

        let (_, stderr) = uninstall(&roots, false);

        assert!(foreign_script.exists(), "deleted someone else's script");
        assert!(foreign_data.exists(), "deleted someone else's data file");
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// #212: when a recognized launcher cannot be removed, the JSON document
    /// must stop claiming `"complete": true`, name what stayed behind, and the
    /// command must not end in success. The human output carries the same
    /// honesty in both directions.
    #[cfg(unix)]
    #[test]
    fn uninstall_reports_a_leftover_instead_of_claiming_complete_removal() {
        use std::os::unix::fs::PermissionsExt;

        // The temp tree has to stay alive through every assertion, so this is
        // a loop with inline fixtures rather than a capturing closure.
        for json_output in [true, false] {
            let roots = roots();
            let binary = install_cargo_binary(&roots.home);
            let bin_dir = roots.home.join(".cargo").join("bin");
            fs::set_permissions(&bin_dir, fs::Permissions::from_mode(0o555)).unwrap();

            let (error, stdout, stderr) = capture_failing_run(&roots, json_output);

            assert!(
                binary.exists(),
                "fixture does not discriminate: the removal succeeded"
            );
            assert!(
                error
                    .as_deref()
                    .is_some_and(|e| e.contains("uninstall did not complete")),
                "a skipped removal must fail the command: {error:?}"
            );
            if json_output {
                let document: serde_json::Value =
                    serde_json::from_str(&stdout).unwrap_or_else(|e| {
                        panic!("stdout is not a single JSON document ({e}): {stdout}")
                    });
                assert_eq!(document["complete"], serde_json::json!(false), "{stdout}");
                assert_eq!(document["removed"], serde_json::json!([]), "{stdout}");
                assert_eq!(
                    document["left_behind"],
                    serde_json::json!([binary.display().to_string()]),
                    "{stdout}"
                );
            } else {
                assert!(stderr.contains("Left in place:"), "{stderr}");
                assert!(
                    stderr.contains("Vera was partially uninstalled."),
                    "{stderr}"
                );
                assert!(
                    !stderr.contains("Vera has been uninstalled."),
                    "claimed complete removal while {} remained: {stderr}",
                    binary.display()
                );
            }

            // Restore before the next iteration's temp tree replaces this one.
            fs::set_permissions(&bin_dir, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
}
