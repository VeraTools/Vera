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

fn shim_name() -> &'static str {
    if cfg!(windows) { "vera.cmd" } else { "vera" }
}

/// What the file sitting at a shim candidate path actually is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShimKind {
    /// A launcher we installed: a text script naming vera, or a symlink to one.
    Ours,
    /// A file we cannot read as text, i.e. a compiled binary. `cargo install`
    /// puts the real executable on `PATH` rather than a shim, so this is a Vera
    /// the uninstaller did not install and does not own.
    ForeignBinary,
    /// Readable text that says nothing about Vera. Someone else's `vera`.
    Unrelated,
}

/// Classify a shim candidate.
///
/// Split out from the removal loop because the previous inline check collapsed
/// "this is not ours" and "I could not read this" into one `unwrap_or(false)`,
/// which is exactly the distinction that decides whether the uninstall was
/// complete. Pure, so all three outcomes can be tested.
fn classify_shim(shim: &Path) -> ShimKind {
    if fs::read_link(shim).is_ok_and(|target| target.to_string_lossy().contains("vera")) {
        return ShimKind::Ours;
    }
    match fs::read_to_string(shim) {
        Ok(contents) if contents.contains("vera") => ShimKind::Ours,
        Ok(_) => ShimKind::Unrelated,
        // Not valid UTF-8, or otherwise unreadable as text. The file is named
        // exactly `vera`/`vera.cmd` and sits in a bin directory, so treat it as
        // a Vera binary to report rather than as an unrelated file to ignore.
        Err(_) => ShimKind::ForeignBinary,
    }
}

fn is_cargo_bin(path: &Path) -> bool {
    path.parent()
        .is_some_and(|dir| dir.ends_with(Path::new(".cargo").join("bin")))
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
    let complete = skill_removal.failures.is_empty();
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

    // 3. Remove the PATH shim.
    let name = shim_name();
    let mut removed_any_shim = false;
    let mut left_in_place: Vec<PathBuf> = Vec::new();
    for dir in shim_candidates(home, user_bin_dir) {
        let shim = dir.join(name);
        if !shim.exists() {
            continue;
        }
        match classify_shim(&shim) {
            ShimKind::Ours => {
                fs::remove_file(&shim)?;
                removed_any_shim = true;
                if !json_output {
                    writeln!(stderr, "  Removed shim {}", shim.display())?;
                }
            }
            ShimKind::ForeignBinary => left_in_place.push(shim),
            ShimKind::Unrelated => {}
        }
    }
    if removed_any_shim {
        removed.push("PATH shim");
    }
    for path in &left_in_place {
        writeln!(
            stderr,
            "  Left in place: {} is a binary, not a shim we installed{}",
            path.display(),
            if is_cargo_bin(path) {
                " — run `cargo uninstall vera` to remove it"
            } else {
                " — remove it yourself if you no longer want it"
            }
        )?;
    }

    // A binary we deliberately did not remove is still a Vera left on `PATH`,
    // so it cannot be reported as a complete uninstall.
    let complete = complete && left_in_place.is_empty();

    if json_output {
        writeln!(
            stdout,
            "{}",
            serde_json::json!({
                "uninstalled": true,
                "complete": complete,
                "removed": removed,
                "skills": removed_skills,
                "left_in_place": left_in_place
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>(),
            })
        )?;
    } else {
        writeln!(stderr)?;
        if complete {
            writeln!(stderr, "Vera has been uninstalled.")?;
        } else {
            // Both reasons can hold at once, so neither branch may hide the
            // other: the user needs to know about every part that survived.
            writeln!(stderr, "Vera was partially uninstalled.")?;
            if !skill_removal.failures.is_empty() {
                writeln!(stderr, "  Some skills could not be removed.")?;
            }
            if !left_in_place.is_empty() {
                writeln!(stderr, "  A Vera binary is still on your PATH.")?;
            }
        }
        writeln!(
            stderr,
            "Per-project indexes (.vera/ in each project) were not removed."
        )?;
    }

    // Mirror `agent::do_remove`: report everything first, then let the first
    // failure fail the command so automation cannot read exit 0 while skill
    // directories survive on disk.
    if let Some(first) = skill_removal.failures.into_iter().next() {
        return Err(first.context("uninstall did not complete"));
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

    /// A run whose skill removal fails partway: captures what was printed even
    /// though the command reports failure.
    #[cfg(unix)]
    fn uninstall_with_failing_skill(
        roots: &Roots,
        json_output: bool,
    ) -> (Option<String>, String, String) {
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
        let (error, stdout, _) = uninstall_with_failing_skill(&roots, true);
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

    /// The bytes that make this defect reproduce: a Mach-O magic number, which
    /// `read_to_string` rejects as invalid UTF-8 exactly as a real
    /// `cargo install`ed binary does. A text fixture would be readable and
    /// would never reach the branch under test.
    const BINARY_MAGIC: &[u8] = &[0xcf, 0xfa, 0xed, 0xfe, 0x0c, 0x00, 0x00, 0x01];

    #[test]
    fn a_binary_on_path_is_reported_and_left_rather_than_silently_kept() {
        let roots = roots();
        let binary = roots.user_bin_dir.join(shim_name());
        fs::write(&binary, BINARY_MAGIC).unwrap();
        assert!(
            fs::read_to_string(&binary).is_err(),
            "fixture must be unreadable as text, or it exercises the shim branch instead"
        );

        let (stdout, stderr) = uninstall(&roots, true);

        assert!(
            binary.exists(),
            "a binary we did not install must not be deleted"
        );
        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(
            document["complete"],
            serde_json::json!(false),
            "a Vera still on PATH is not a complete uninstall: {stdout}"
        );
        assert_eq!(
            document["left_in_place"],
            serde_json::json!([binary.display().to_string()]),
            "the surviving binary must be named in the document: {stdout}"
        );
        assert!(
            stderr.contains("Left in place") && stderr.contains(&binary.display().to_string()),
            "stderr must name the file that survived: {stderr}"
        );
    }

    #[test]
    fn a_text_shim_is_still_removed_and_still_reports_complete() {
        // The negative half: adding the binary case must not stop the case that
        // already worked, which is what every non-cargo install writes.
        let roots = roots();
        let shim = roots.user_bin_dir.join(shim_name());
        fs::write(
            &shim,
            "#!/bin/sh\nexec \"$HOME/.vera/bin/1.0.0/vera\" \"$@\"\n",
        )
        .unwrap();

        let (stdout, stderr) = uninstall(&roots, true);

        assert!(!shim.exists(), "our own shim must still be removed");
        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(document["complete"], serde_json::json!(true), "{stdout}");
        assert_eq!(
            document["left_in_place"],
            serde_json::json!([]),
            "nothing survived, so the list must be empty: {stdout}"
        );
        assert!(
            !stderr.contains("Left in place"),
            "a removed shim must not be reported as left behind: {stderr}"
        );
    }

    #[test]
    fn an_unrelated_binary_named_vera_is_left_alone_but_still_reported() {
        // Someone else's `vera` we cannot read is indistinguishable from ours by
        // content, so it is reported rather than deleted. This pins that the
        // choice is "report, never delete", not "delete anything unreadable".
        let roots = roots();
        let other = roots.user_bin_dir.join(shim_name());
        fs::write(&other, BINARY_MAGIC).unwrap();

        let (_, _) = uninstall(&roots, true);

        assert!(other.exists(), "an unreadable file must never be deleted");
    }

    #[test]
    fn classify_shim_separates_unreadable_from_unrelated() {
        // The defect was that `unwrap_or(false)` collapsed these two, which is
        // the distinction that decides whether the uninstall was complete.
        let temp = tempdir().unwrap();

        let ours = temp.path().join("ours");
        fs::write(&ours, "#!/bin/sh\nexec .../vera \"$@\"\n").unwrap();
        assert_eq!(classify_shim(&ours), ShimKind::Ours);

        let binary = temp.path().join("binary");
        fs::write(&binary, BINARY_MAGIC).unwrap();
        assert_eq!(classify_shim(&binary), ShimKind::ForeignBinary);

        let unrelated = temp.path().join("unrelated");
        fs::write(&unrelated, "#!/bin/sh\necho not this tool\n").unwrap();
        assert_eq!(classify_shim(&unrelated), ShimKind::Unrelated);
    }

    #[cfg(unix)]
    #[test]
    fn classify_shim_recognizes_a_symlink_to_vera() {
        let temp = tempdir().unwrap();
        let target = temp.path().join("vera-real");
        fs::write(&target, BINARY_MAGIC).unwrap();
        let link = temp.path().join("vera");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        // Reached through the symlink arm, not the text arm: the target is
        // unreadable as text, so without that arm this would be ForeignBinary.
        assert_eq!(classify_shim(&link), ShimKind::Ours);
    }

    #[cfg(unix)]
    #[test]
    fn uninstall_human_output_names_skills_removed_before_a_later_removal_failed() {
        let roots = roots();
        let claude = install_claude_global_skill(&roots.home);
        let locked = install_unremovable_gemini_global_skill(&roots.home);

        let (error, stdout, stderr) = uninstall_with_failing_skill(&roots, false);
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

        let (error, stdout, stderr) = uninstall_with_failing_skill(&roots, false);
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

        let (error, stdout, stderr) = uninstall_with_failing_skill(&roots, false);
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
}
