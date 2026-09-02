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

/// The launcher the installers write, with the binary path left as a hole.
///
/// `packages/npm-cli/bin/vera.js` and `packages/python-cli/.../__main__.py`
/// emit these two forms and nothing else, character for character. Matching
/// them is the whole of shim recognition: uninstall never has to understand
/// shell or batch, only to recognize what it wrote.
const UNIX_SHIM: (&str, &str) = ("#!/bin/sh\nexec \"", "\" \"$@\"\n");
const WINDOWS_SHIM: (&str, &str) = ("@echo off\r\n\"", "\" %*\r\n");

/// The binary path a shim launches, if the file is one of the two templates.
///
/// A byte comparison with one hole, rather than a parse. Nothing about
/// quoting, escaping, comments, separators or dialects can change the answer,
/// because a file that is not exactly one of these shapes is not ours.
fn shim_target(text: &str) -> Option<&str> {
    [UNIX_SHIM, WINDOWS_SHIM]
        .into_iter()
        .find_map(|(head, tail)| {
            text.strip_prefix(head)
                .and_then(|rest| rest.strip_suffix(tail))
                // Both ends are anchored, so whatever lies between them is the
                // path, quote characters included: a unix path may contain one and
                // the installer writes it through verbatim.
                .filter(|target| !target.is_empty())
        })
}

/// Whether a launcher path is one this installation put there.
///
/// Two ways to qualify, and either is enough. The recorded path is
/// authoritative: `install.json` carries `binary_path`, which is what `upgrade`
/// already uses to find the installed executable. Containment in the Vera home
/// qualifies as well, and not only as a fallback when nothing was recorded.
///
/// That second arm carries weight: step 2 removes the Vera home before this
/// runs, so a chain through an intermediate link inside it, such as
/// `PATH/vera -> ~/.vera/current -> <recorded binary>`, resolves only as far as
/// the link that has just been deleted. Requiring an exact match against the
/// recorded path would leave that alias on PATH. A path inside our own
/// directory is ours whether or not it is the one we wrote down.
fn is_our_binary(target: &Path, recorded: Option<&Path>, vera_home: &Path) -> bool {
    recorded.is_some_and(|recorded| lexically_normalize(target) == lexically_normalize(recorded))
        || is_inside(target, vera_home)
}

/// Recognizes a candidate path as a removable Vera launcher, or leaves it
/// unclassified so unrelated files stay untouched.
///
/// A shim is one of the two files the installers write, naming this
/// installation's binary, or a symlink resolving to that binary. The
/// cargo-installed binary is neither: it fails UTF-8 decoding by construction
/// (#212), so only "unreadable as text plus a regular executable file with an
/// exact Vera entry name" attributes it to cargo without grabbing anything
/// else named `vera`.
fn classify_launch_entry(
    entry: &Path,
    home: &Path,
    vera_home: &Path,
    recorded: Option<&Path>,
) -> Option<LaunchEntry> {
    let read_as_text = fs::read_to_string(entry);
    let launches_vera = read_as_text
        .as_deref()
        .ok()
        .and_then(shim_target)
        .is_some_and(|target| is_our_binary(Path::new(target), recorded, vera_home));
    if launches_vera || symlink_points_at_vera(entry, vera_home, recorded) {
        return Some(LaunchEntry::Shim);
    }
    // Decodable text that is not one of our templates belongs to someone else.
    //
    // The cargo arm needs evidence of its own, and the only evidence available
    // is where the file sits: `cargo install` writes to `~/.cargo/bin`, so an
    // unreadable executable named `vera` there is a cargo artifact by
    // construction. In the other candidate directories it is just somebody
    // else's program with the same name. Checking the executable *format*
    // would not help, because any binary named `vera` passes that too.
    let is_cargo_binary = entry
        .parent()
        .is_some_and(|parent| is_cargo_bin_dir(parent, &cargo_bin_dir(home)))
        && read_as_text.is_err()
        && fs::symlink_metadata(entry).is_ok_and(|meta| meta.is_file())
        && is_executable(entry);
    is_cargo_binary.then_some(LaunchEntry::CargoBinary)
}

/// Resolves `.` and `..` textually. Used instead of `canonicalize` because the
/// directories being compared may already have been deleted.
fn lexically_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

/// Whether `path` sits inside `root`, compared component-wise so that
/// `/opt/vera-extra` is not read as being inside `/opt/vera`.
fn is_inside(path: &Path, root: &Path) -> bool {
    if root.as_os_str().is_empty() {
        return false;
    }
    lexically_normalize(path).starts_with(lexically_normalize(root))
}

/// How many links to follow before giving up, so a cycle cannot hang the run.
const MAX_SYMLINK_HOPS: usize = 40;

/// Where a symlink chain lands, resolved one hop at a time.
///
/// Each relative target is resolved against its own link's directory, so
/// `vera -> ../lib/vera/bin/vera` is judged on where it actually lands rather
/// than on how it is spelled. Resolution stops at the first path that is not a
/// link, which includes a dangling one: that path is still the answer, and
/// comparing it lexically is what lets a broken alias into Vera's own files be
/// recognized and removed.
fn resolve_symlink_chain(entry: &Path) -> Option<PathBuf> {
    let mut current = fs::read_link(entry).ok()?;
    if current.is_relative() {
        current = entry.parent().unwrap_or(Path::new("")).join(current);
    }
    for _ in 0..MAX_SYMLINK_HOPS {
        let Ok(next) = fs::read_link(&current) else {
            return Some(current);
        };
        current = if next.is_absolute() {
            next
        } else {
            current.parent().unwrap_or(Path::new("")).join(next)
        };
    }
    Some(current)
}

/// Whether a symlink at `entry` resolves to this installation's binary.
///
/// The whole chain is followed, so an alias that reaches Vera through another
/// link is still ours. Stopping at the first hop left such an alias on PATH
/// while the run reported a complete uninstall.
fn symlink_points_at_vera(entry: &Path, vera_home: &Path, recorded: Option<&Path>) -> bool {
    resolve_symlink_chain(entry)
        .is_some_and(|resolved| is_our_binary(&resolved, recorded, vera_home))
}

/// Where `cargo install` places binaries: `$CARGO_HOME/bin`, falling back to
/// `~/.cargo/bin`.
///
/// Derived rather than pattern-matched. A configured `VERA_USER_BIN_DIR` that
/// merely *ends* in `.cargo/bin` is not cargo's directory, and treating it as
/// one would hand every unreadable executable there to the cargo arm.
fn cargo_bin_dir(home: &Path) -> PathBuf {
    std::env::var_os("CARGO_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".cargo"))
        .join("bin")
}

/// Whether a directory is cargo's own bin directory.
fn is_cargo_bin_dir(dir: &Path, cargo_bin: &Path) -> bool {
    lexically_normalize(dir) == lexically_normalize(cargo_bin)
}

/// Resolves a configured bin-directory override against the working directory.
///
/// A relative override would otherwise be compared against absolute paths when
/// a symlink chain is resolved, so an owned relative link survives the run.
fn resolve_user_bin_dir(cwd: &Path, dir: PathBuf) -> PathBuf {
    if dir.is_absolute() {
        dir
    } else {
        cwd.join(dir)
    }
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
    // Read before step 2 removes the Vera home: this is the record of which
    // binary the installer placed, and it lives inside the directory that is
    // about to go.
    let recorded_binary = state::load_install_provenance()
        .ok()
        .and_then(|provenance| provenance.binary_path)
        .map(PathBuf::from);
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    // A relative override would otherwise be compared against absolute paths
    // when a symlink chain is resolved, so an owned relative link survives.
    let user_bin_dir = configured_user_bin_dir().map(|dir| resolve_user_bin_dir(&cwd, dir));

    run_at(
        InstallLayout {
            home: &home,
            vera_home: &vera_home,
            cwd: &cwd,
            user_bin_dir: user_bin_dir.as_deref(),
            recorded_binary: recorded_binary.as_deref(),
        },
        json_output,
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    )
}

/// Where this installation put its files, resolved once by the caller so the
/// body never consults the environment.
struct InstallLayout<'a> {
    home: &'a Path,
    vera_home: &'a Path,
    cwd: &'a Path,
    user_bin_dir: Option<&'a Path>,
    /// The binary the installer recorded in `install.json`, read before the
    /// Vera home is removed.
    recorded_binary: Option<&'a Path>,
}

fn run_at(
    layout: InstallLayout<'_>,
    json_output: bool,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<()> {
    let InstallLayout {
        home,
        vera_home,
        cwd,
        user_bin_dir,
        recorded_binary,
    } = layout;
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
            // `exists` follows the link and so reports `false` for a broken
            // one, which left a dangling Vera symlink on PATH while the run
            // still claimed a complete uninstall. Ask about the link itself.
            if entry.symlink_metadata().is_err() {
                continue;
            }
            let Some(kind) = classify_launch_entry(&entry, home, vera_home, recorded_binary) else {
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
            InstallLayout {
                home: &roots.home,
                vera_home: &roots.vera_home,
                cwd: &roots.cwd,
                user_bin_dir: Some(roots.user_bin_dir.as_path()),
                recorded_binary: None,
            },
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
            InstallLayout {
                home: &roots.home,
                vera_home: &roots.vera_home,
                cwd: &roots.cwd,
                user_bin_dir: Some(roots.user_bin_dir.as_path()),
                recorded_binary: None,
            },
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
    fn install_shim(dir: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        fs::create_dir_all(dir).unwrap();
        let path = dir.join("vera");
        fs::write(&path, body).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// The exact file `packages/npm-cli/bin/vera.js` and the Python wrapper
    /// write, built from the same Vera home the uninstall resolves.
    #[cfg(unix)]
    #[test]
    fn uninstall_removes_the_shim_the_installers_write() {
        for template in [
            "#!/bin/sh\nexec \"{bin}\" \"$@\"\n",
            "@echo off\r\n\"{bin}\" %*\r\n",
        ] {
            let roots = roots();
            let binary = roots
                .vera_home
                .join("bin")
                .join("1.3.0")
                .join("aarch64-apple-darwin")
                .join("vera");
            let body = template.replace("{bin}", &binary.display().to_string());
            let shim = install_shim(&roots.home.join(".local").join("bin"), &body);

            let (_, stderr) = uninstall(&roots, false);

            assert!(!shim.exists(), "our own shim survived: {body:?} / {stderr}");
            assert!(stderr.contains("Removed PATH shim"), "{stderr}");
        }
    }

    /// A symlink resolving to the installed binary is ours; one resolving
    /// anywhere else is not, however it is spelled.
    #[cfg(unix)]
    #[test]
    fn a_symlink_is_judged_by_where_it_resolves() {
        let roots = roots();
        let bin = roots.home.join(".local").join("bin");
        fs::create_dir_all(&bin).unwrap();
        let ours = bin.join("vera");
        std::os::unix::fs::symlink(roots.vera_home.join("bin").join("vera"), &ours).unwrap();
        // Dangling by construction: the target does not exist.
        assert!(!ours.exists());

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            ours.symlink_metadata().is_err(),
            "a dangling Vera symlink stayed on PATH: {stderr}"
        );
    }

    /// A path may contain a quote character, and the installer writes it into
    /// the shim verbatim. Rejecting such a target left the shim on PATH.
    #[cfg(unix)]
    #[test]
    fn a_quote_in_the_home_path_does_not_hide_the_shim() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("ho\"me");
        let bin = home.join(".local").join("bin");
        let vera_home = home.join(".vera");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(temp.path().join("project")).unwrap();
        let binary = vera_home.join("bin").join("1.3.0").join("x").join("vera");
        let shim = install_shim(
            &bin,
            &format!("#!/bin/sh\nexec \"{}\" \"$@\"\n", binary.display()),
        );

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        run_at(
            InstallLayout {
                home: &home,
                vera_home: &vera_home,
                cwd: &temp.path().join("project"),
                user_bin_dir: Some(bin.as_path()),
                recorded_binary: None,
            },
            false,
            &mut stdout,
            &mut stderr,
        )
        .unwrap();

        assert!(!shim.exists(), "a quote in the path hid our own shim");
    }

    /// An alias that reaches Vera through another link is still ours. Comparing
    /// only the first hop left it on PATH while the run reported success.
    #[cfg(unix)]
    #[test]
    fn a_symlink_chain_reaching_vera_is_followed() {
        let roots = roots();
        let bin = roots.home.join(".local").join("bin");
        fs::create_dir_all(&bin).unwrap();
        let binary = roots.vera_home.join("bin").join("vera");
        let middle = roots.home.join("alias-vera");
        std::os::unix::fs::symlink(&binary, &middle).unwrap();
        let entry = bin.join("vera");
        std::os::unix::fs::symlink(&middle, &entry).unwrap();

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            entry.symlink_metadata().is_err(),
            "an aliased Vera symlink stayed on PATH: {stderr}"
        );
    }

    /// The Vera home is removed before PATH entries are classified, so a chain
    /// through an intermediate link inside it resolves only as far as a link
    /// that no longer exists. That terminal path is still inside our own
    /// directory, and the alias must still be removed.
    #[cfg(unix)]
    #[test]
    fn a_chain_through_a_deleted_intermediate_is_still_ours() {
        let roots = roots();
        let bin = roots.home.join(".local").join("bin");
        let recorded = roots
            .vera_home
            .join("bin")
            .join("1.3.0")
            .join("x")
            .join("vera");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(recorded.parent().unwrap()).unwrap();
        fs::write(&recorded, "binary").unwrap();
        // PATH/vera -> <vera_home>/current -> <recorded binary>
        let middle = roots.vera_home.join("current");
        std::os::unix::fs::symlink(&recorded, &middle).unwrap();
        let entry = bin.join("vera");
        std::os::unix::fs::symlink(&middle, &entry).unwrap();

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        run_at(
            InstallLayout {
                home: &roots.home,
                vera_home: &roots.vera_home,
                cwd: &roots.cwd,
                user_bin_dir: Some(bin.as_path()),
                recorded_binary: Some(recorded.as_path()),
            },
            false,
            &mut stdout,
            &mut stderr,
        )
        .unwrap();

        let stderr = String::from_utf8(stderr).unwrap();
        assert!(
            entry.symlink_metadata().is_err(),
            "an alias through a deleted intermediate stayed on PATH: {stderr}"
        );
    }

    /// An unreadable executable named `vera` is only evidence of a cargo
    /// install where cargo puts one. Elsewhere it is somebody else's program
    /// with the same name, and deleting it is unrecoverable.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_executable_outside_the_cargo_bin_dir_is_left_alone() {
        use std::os::unix::fs::PermissionsExt;
        let roots = roots();
        // Same bytes, same permissions, two locations.
        let cargo_bin = roots.home.join(".cargo").join("bin");
        let other_bin = roots.home.join(".local").join("bin");
        fs::create_dir_all(&cargo_bin).unwrap();
        fs::create_dir_all(&other_bin).unwrap();
        let ours = cargo_bin.join("vera");
        let theirs = other_bin.join("vera");
        for path in [&ours, &theirs] {
            fs::write(path, [0x7f, b'E', b'L', b'F', 0xcf]).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
        }

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            !ours.exists(),
            "the cargo-installed binary stayed: {stderr}"
        );
        assert!(
            theirs.exists(),
            "deleted an unreadable executable that cargo never wrote: {stderr}"
        );
    }

    /// A relative `VERA_USER_BIN_DIR` must be resolved against the working
    /// directory, or a symlink chain resolved from it stays relative and never
    /// matches the absolute Vera home.
    ///
    /// Asserts the resolution itself. The previous version of this test built
    /// the absolute path in the fixture and handed that to `run_at`, so it
    /// never touched the resolution and passed with the fix removed.
    #[test]
    fn a_relative_user_bin_dir_is_resolved_against_cwd() {
        let cwd = Path::new("/work/project");
        assert_eq!(
            resolve_user_bin_dir(cwd, PathBuf::from("vendor/bin")),
            Path::new("/work/project/vendor/bin")
        );
        assert_eq!(
            resolve_user_bin_dir(cwd, PathBuf::from("../shared/bin")),
            Path::new("/work/project/../shared/bin")
        );
        // An absolute override is already an answer and must not be rebased.
        assert_eq!(
            resolve_user_bin_dir(cwd, PathBuf::from("/opt/bin")),
            Path::new("/opt/bin")
        );
    }

    /// Cargo's directory is derived from the cargo home, not matched by shape:
    /// an override that merely ends in `.cargo/bin` is somebody else's.
    #[test]
    fn only_cargos_derived_bin_directory_counts_as_cargo() {
        let home = Path::new("/home/u");
        let cargo_bin = home.join(".cargo").join("bin");
        assert!(is_cargo_bin_dir(&cargo_bin, &cargo_bin));
        assert!(is_cargo_bin_dir(
            &home.join(".cargo").join(".").join("bin"),
            &cargo_bin
        ));
        for foreign in [
            "/home/u/.local/bin",
            "/home/u/cargo/bin",
            // A configured override that ends in the same two segments.
            "/opt/sandbox/.cargo/bin",
        ] {
            assert!(
                !is_cargo_bin_dir(Path::new(foreign), &cargo_bin),
                "{foreign} is not cargo's own bin directory"
            );
        }
    }

    /// A chain that never terminates must not hang the uninstall.
    #[cfg(unix)]
    #[test]
    fn a_symlink_cycle_terminates() {
        let roots = roots();
        let bin = roots.home.join(".local").join("bin");
        fs::create_dir_all(&bin).unwrap();
        let a = bin.join("vera");
        let b = roots.home.join("loop-b");
        std::os::unix::fs::symlink(&b, &a).unwrap();
        std::os::unix::fs::symlink(&a, &b).unwrap();

        let (_, stderr) = uninstall(&roots, false);

        assert!(stderr.contains("uninstalled"), "{stderr}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_into_someone_elses_install_is_left_alone() {
        let roots = roots();
        let bin = roots.home.join(".local").join("bin");
        let other = roots.home.join("veracrypt-bin");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(&other).unwrap();
        let target = other.join("veracrypt");
        fs::write(&target, "binary").unwrap();
        let link = bin.join("vera");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let (_, stderr) = uninstall(&roots, false);

        assert!(link.symlink_metadata().is_ok(), "deleted a foreign symlink");
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// The whole corpus of foreign launchers accumulated over ten review
    /// rounds against the parser this replaces. Every one survives here by
    /// construction rather than by a rule: none of these files is one of the
    /// two templates, so none of them is ever a candidate.
    #[cfg(unix)]
    #[test]
    fn no_foreign_launcher_is_claimed() {
        for body in [
            // Mentions Vera only in a comment.
            "#!/bin/sh\n# drop-in replacement for vera\nexec /usr/bin/rg \"$@\"\n",
            "@echo off\r\nREM wrapper around vera\r\n\"%~dp0\\rg.exe\" %*\r\n",
            // Shares the letters but not the name.
            "#!/bin/sh\nexec /opt/veracrypt/bin/veracrypt \"$@\"\n",
            "#!/bin/sh\nexec /opt/vera-extra/bin/tool \"$@\"\n",
            // A directory named vera holding someone else's program.
            "#!/bin/sh\nexec /opt/vera/bin/rg \"$@\"\n",
            // Another program that merely shares the launcher name.
            "#!/bin/sh\nexec /opt/other/bin/vera \"$@\"\n",
            // A Vera path in argument position rather than program position.
            "#!/bin/sh\necho \"{home}/bin/vera\"\nexec /usr/bin/rg \"$@\"\n",
            // Separators and quoting that each cost a review round.
            "#!/bin/sh\necho \"see; {home}/bin/vera\"\nexec /usr/bin/rg \"$@\"\n",
            "#!/bin/sh\necho a\\; {home}/bin/vera\nexec /usr/bin/rg \"$@\"\n",
            "#!/bin/sh\necho safe # ; exec \"{home}/bin/vera\"\nexec /usr/bin/rg \"$@\"\n",
            "#!/bin/sh\n\"if\" \"{home}/bin/vera\"\n",
            "#!/bin/sh\necho \"case esac\"\necho a\\) {home}/bin/vera\nexec /usr/bin/rg \"$@\"\n",
            "@echo off\r\necho ({home}/bin/vera)\r\n\"%~dp0\\rg.exe\" %*\r\n",
            "@echo off\r\necho safe; \"{home}/bin/vera\"\r\n\"%~dp0\\rg.exe\" %*\r\n",
            // The template with the wrong binary: right shape, not our install.
            "#!/bin/sh\nexec \"/opt/elsewhere/bin/vera\" \"$@\"\n",
        ] {
            let roots = roots();
            let body = body.replace("{home}", &roots.vera_home.display().to_string());
            let foreign = install_shim(&roots.home.join(".local").join("bin"), &body);

            let (_, stderr) = uninstall(&roots, false);

            assert!(foreign.exists(), "claimed a foreign launcher: {body:?}");
            assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
        }
    }

    /// The recorded path is authoritative when present: a template naming a
    /// binary outside the Vera home is still ours if that is what was
    /// installed, and one naming a different binary is not.
    #[cfg(unix)]
    #[test]
    fn the_recorded_binary_path_decides_when_it_exists() {
        let recorded = Path::new("/opt/custom/vera");
        let vera_home = Path::new("/home/u/.vera");
        assert!(is_our_binary(recorded, Some(recorded), vera_home));
        assert!(!is_our_binary(
            Path::new("/opt/other/vera"),
            Some(recorded),
            vera_home
        ));
        // Containment qualifies on its own, with or without a record: an
        // intermediate link inside the Vera home has already been deleted by
        // the time a chain is resolved, so it can never match the record.
        assert!(is_our_binary(
            &vera_home.join("bin/1.3.0/x/vera"),
            None,
            vera_home
        ));
        assert!(is_our_binary(
            &vera_home.join("current"),
            Some(recorded),
            vera_home
        ));
        assert!(!is_our_binary(
            Path::new("/opt/custom/vera"),
            None,
            vera_home
        ));
    }

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
