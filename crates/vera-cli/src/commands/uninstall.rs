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
    let skill_removal = match agent::remove_all_skills(cwd, home) {
        Ok(removal) => removal,
        Err(e) => {
            tracing::warn!("failed to resolve agent skill locations: {e:#}");
            agent::SkillRemoval::default()
        }
    };
    // Uninstall continues past a skill that cannot be deleted, so the failure is
    // only visible if it is reported here. stderr keeps it out of the JSON
    // document on stdout.
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
    for dir in shim_candidates(home, user_bin_dir) {
        let shim = dir.join(name);
        if shim.exists() {
            // Only remove if it's a Vera shim (contains "vera" in content or is a symlink to vera).
            let is_vera_shim = fs::read_to_string(&shim)
                .map(|c| c.contains("vera"))
                .unwrap_or(false)
                || fs::read_link(&shim)
                    .map(|t| t.to_string_lossy().contains("vera"))
                    .unwrap_or(false);
            if is_vera_shim {
                fs::remove_file(&shim)?;
                removed_any_shim = true;
                if !json_output {
                    writeln!(stderr, "  Removed shim {}", shim.display())?;
                }
            }
        }
    }
    if removed_any_shim {
        removed.push("PATH shim");
    }

    if json_output {
        writeln!(
            stdout,
            "{}",
            serde_json::json!({
                "uninstalled": true,
                "removed": removed,
                "skills": removed_skills,
            })
        )?;
    } else {
        writeln!(stderr)?;
        writeln!(stderr, "Vera has been uninstalled.")?;
        writeln!(
            stderr,
            "Per-project indexes (.vera/ in each project) were not removed."
        )?;
    }

    Ok(())
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
        .unwrap();
        (
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

        let (stdout, _) = uninstall(&roots, true);
        let claude_was_deleted = !claude.exists();
        allow_cleanup(&locked);

        assert!(
            claude_was_deleted,
            "fixture does not discriminate: the earlier skill was never deleted"
        );
        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
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

    #[cfg(unix)]
    #[test]
    fn uninstall_human_output_names_skills_removed_before_a_later_removal_failed() {
        let roots = roots();
        let claude = install_claude_global_skill(&roots.home);
        let locked = install_unremovable_gemini_global_skill(&roots.home);

        let (stdout, stderr) = uninstall(&roots, false);
        let claude_was_deleted = !claude.exists();
        allow_cleanup(&locked);

        assert!(
            claude_was_deleted,
            "fixture does not discriminate: the earlier skill was never deleted"
        );
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
    }

    /// The only installed skill fails to delete: nothing was removed, but
    /// claiming nothing was *installed* would be a different lie.
    #[cfg(unix)]
    #[test]
    fn uninstall_human_output_does_not_claim_nothing_was_installed_when_removal_failed() {
        let roots = roots();
        let locked = install_unremovable_gemini_global_skill(&roots.home);

        let (stdout, stderr) = uninstall(&roots, false);
        allow_cleanup(&locked);

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

        let (stdout, stderr) = uninstall(&roots, false);
        allow_cleanup(&locked);
        let skill_survived = locked.join("SKILL.md").exists();

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

    #[test]
    fn uninstall_human_output_reports_nothing_when_no_skills_are_installed() {
        let roots = roots();

        let (stdout, stderr) = uninstall(&roots, false);

        assert_eq!(stdout.trim(), "No Vera skill installations found.");
        assert!(stderr.contains("Vera has been uninstalled."));
    }
}
