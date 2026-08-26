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
    /// A launcher we installed: it points into the Vera data directory.
    Ours,
    /// Readable text that mentions Vera but does not point into the data
    /// directory, so it cannot be proven ours. Reported, never deleted.
    Ambiguous,
    /// A file we cannot read as text, i.e. a compiled binary. `cargo install`
    /// puts the real executable on `PATH` rather than a shim, so this is a Vera
    /// the uninstaller did not install and does not own.
    ForeignBinary,
    /// Nothing to do with Vera. Someone else's `vera`.
    Unrelated,
}

/// Classify a shim candidate.
///
/// Split out from the removal loop because the previous inline check collapsed
/// "this is not ours" and "I could not read this" into one `unwrap_or(false)`,
/// which is exactly the distinction that decides whether the uninstall was
/// complete. Pure, so every outcome can be tested.
///
/// Ownership is proven by pointing into `vera_home`, not by containing the
/// substring "vera". A launcher naming `/opt/veracrypt`, or a symlink to
/// `vera-tool`, contains it — and deleting a stranger's file is the one outcome
/// here that cannot be undone. Anything that mentions Vera without proving
/// ownership is `Ambiguous`: reported and left alone, so a real shim written by
/// an install layout we do not recognise still surfaces instead of vanishing
/// from the report.
fn classify_shim(shim: &Path, vera_home: &Path) -> ShimKind {
    // A prefix is not a path boundary: `${vera_home}-backup/bin/vera` starts
    // with the same characters and belongs to somebody else. The symlink arm
    // compares components, and the text arm requires a separator after the
    // directory, so neither accepts a sibling whose name merely begins the same
    // way.
    let shim_dir = shim.parent().unwrap_or(Path::new(""));
    let mentions_vera = |text: &str| text.to_ascii_lowercase().contains("vera");
    // Match on the stem rather than the whole file name: a `vera.cmd` launcher
    // execs `vera.exe`, so requiring the launcher's own name would stop
    // recognising our own Windows shim.
    let is_vera_executable = |token: &Path| token.file_stem().is_some_and(|stem| stem == "vera");

    // Ownership needs a line that *launches* something inside `vera_home`, not
    // a file that mentions the path anywhere. A foreign launcher naming
    // `~/.vera/config.json` in a comment would satisfy a whole-file substring
    // and be deleted for it. Requiring a non-comment line that carries a
    // `vera_home`-rooted path ending in this shim's own executable name is what
    // separates "runs our binary" from "talks about our directory".
    //
    // This recognises the launcher shapes shipped today, which point at
    // `<vera_home>/bin/<version>/<target>/vera`. A layout that stores the path
    // some other way falls through to `Ambiguous` and is reported rather than
    // deleted, which is the safe direction for a check that cannot be proven
    // exhaustive.
    let launches_from_vera_home = |text: &str| {
        text.lines()
            .map(str::trim)
            .filter(|line| !is_comment_line(line))
            .filter_map(launched_program)
            .any(|program| {
                let program = Path::new(&program);
                is_vera_executable(program) && is_inside(program, shim_dir, vera_home)
            })
    };

    // `read_link` rather than anything that resolves the target: `vera_home` is
    // already deleted by the time this runs, so an owned shim pointing into it
    // is a dangling symlink and every existence check on the target says no.
    // A relative target is resolved against the shim's own directory, which is
    // what the filesystem would do, but lexically.
    if let Ok(target) = fs::read_link(shim) {
        return if is_inside(&target, shim_dir, vera_home) {
            ShimKind::Ours
        } else if mentions_vera(&target.to_string_lossy()) {
            ShimKind::Ambiguous
        } else {
            ShimKind::Unrelated
        };
    }

    match fs::read_to_string(shim) {
        Ok(contents) if launches_from_vera_home(&contents) => ShimKind::Ours,
        Ok(contents) if mentions_vera(&contents) => ShimKind::Ambiguous,
        Ok(_) => ShimKind::Unrelated,
        // Not valid UTF-8, or otherwise unreadable as text. The file is named
        // exactly `vera`/`vera.cmd` and sits in a bin directory, so treat it as
        // a Vera binary to report rather than as an unrelated file to ignore.
        Err(_) => ShimKind::ForeignBinary,
    }
}

/// Whether a launcher line is a comment in any of the shells that write shims.
///
/// `sh` uses `#`; batch uses `::`, `rem`, and `@rem`, none of which are
/// case-sensitive. A comment that happens to name the data directory must not
/// count as evidence that this launcher runs it.
fn is_comment_line(line: &str) -> bool {
    let line = line.trim_start();
    if line.starts_with('#') || line.starts_with("::") {
        return true;
    }
    let lowered = line.to_ascii_lowercase();
    let lowered = lowered.strip_prefix('@').unwrap_or(&lowered);
    lowered == "rem" || lowered.starts_with("rem ") || lowered.starts_with("rem\t")
}

/// Lexically normalize a path, resolving `.` and `..` without touching disk.
///
/// `<vera_home>/bin/../../other/vera` passes a prefix test while resolving
/// outside the directory entirely. Nothing here may resolve against the real
/// filesystem: `vera_home` has already been deleted by the time these checks
/// run, so `canonicalize` would fail on precisely the paths that matter.
fn lexically_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Whether `candidate`, resolved against `base` if relative, sits inside `root`.
fn is_inside(candidate: &Path, base: &Path, root: &Path) -> bool {
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base.join(candidate)
    };
    lexically_normalize(&absolute).starts_with(lexically_normalize(root))
}

/// The program a launcher line executes, if the line executes one.
///
/// Only the program position counts. A foreign launcher can pass our binary
/// path as an *argument* — `exec /usr/bin/backup --runner "<vera_home>/.../vera"`
/// — and that is not our shim. Leading `exec`, `call`, `start`, `cmd /c` and
/// environment assignments are skipped because every shim shape shipped today
/// puts one of them before the program.
///
/// Deliberately not a shell parser. A line this does not understand yields
/// `None`, which lands the file in `Ambiguous`: reported, never deleted. That
/// is the safe direction for a rule that cannot be proven exhaustive, and it is
/// why `Ambiguous` exists rather than a stricter rule alone.
fn launched_program(line: &str) -> Option<String> {
    // Strip a trailing sh comment. Batch comment *lines* are handled by
    // `is_comment_line`; batch has no inline comment form worth modelling.
    let line = match line.find(" #") {
        Some(at) => &line[..at],
        None => line,
    };

    for token in launcher_tokens(line) {
        let bare = token.trim_start_matches('@');
        let lowered = bare.to_ascii_lowercase();
        let is_prelude = matches!(
            lowered.as_str(),
            "" | "exec" | "call" | "start" | "cmd" | "/c" | "/d" | "sh" | "-c"
        );
        // `VAR=value` prefixes, but not a path that happens to contain `=`.
        let is_assignment = !bare.contains(std::path::MAIN_SEPARATOR) && bare.contains('=');
        if is_prelude || is_assignment {
            continue;
        }
        return Some(token);
    }
    None
}

/// Split a launcher line into candidate path tokens, keeping quoted runs whole.
///
/// `split_whitespace` alone would tear `"C:\Users\First Last\.vera\...\vera.exe"`
/// into three tokens and lose the path, so a real shim under a home directory
/// with a space in it would stop being recognised as ours.
fn launcher_tokens(line: &str) -> impl Iterator<Item = String> + '_ {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in line.chars() {
        match quote {
            Some(open) if ch == open => {
                quote = None;
                tokens.push(std::mem::take(&mut current));
            }
            Some(_) => current.push(ch),
            None if ch == '"' || ch == '\'' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                quote = Some(ch);
            }
            None if ch.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            None => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens.into_iter()
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
    let mut left_in_place: Vec<(PathBuf, &str)> = Vec::new();
    for dir in shim_candidates(home, user_bin_dir) {
        let shim = dir.join(name);
        // `symlink_metadata`, not `exists`: step 2 above has already removed
        // `vera_home`, so a shim symlinked into it is now dangling and
        // `exists()` — which follows the link — reports it as absent. That is
        // how an owned shim could be left on PATH while the run still called
        // itself complete.
        if fs::symlink_metadata(&shim).is_err() {
            continue;
        }
        match classify_shim(&shim, vera_home) {
            ShimKind::Ours => {
                fs::remove_file(&shim)?;
                removed_any_shim = true;
                if !json_output {
                    writeln!(stderr, "  Removed shim {}", shim.display())?;
                }
            }
            ShimKind::ForeignBinary => left_in_place.push((shim, "a binary, not a shim")),
            ShimKind::Ambiguous => {
                left_in_place.push((shim, "mentions Vera but does not point into its data dir"))
            }
            ShimKind::Unrelated => {}
        }
    }
    if removed_any_shim {
        removed.push("PATH shim");
    }
    for (path, reason) in &left_in_place {
        writeln!(
            stderr,
            "  Left in place: {} — {reason}{}",
            path.display(),
            if is_cargo_bin(path) {
                ". Run `cargo uninstall vera` to remove it"
            } else {
                ". Remove it yourself if you no longer want it"
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
                    .map(|(path, _)| path.display().to_string())
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
    use std::path::MAIN_SEPARATOR;
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
            format!(
                "#!/bin/sh\nexec \"{}/bin/1.0.0/vera\" \"$@\"\n",
                roots.vera_home.display()
            ),
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

        let vera_home = temp.path().join("home").join(".vera");

        let ours = temp.path().join("ours");
        fs::write(
            &ours,
            format!(
                "#!/bin/sh\nexec \"{}/bin/1.0.0/vera\" \"$@\"\n",
                vera_home.display()
            ),
        )
        .unwrap();
        assert_eq!(classify_shim(&ours, &vera_home), ShimKind::Ours);

        let binary = temp.path().join("binary");
        fs::write(&binary, BINARY_MAGIC).unwrap();
        assert_eq!(classify_shim(&binary, &vera_home), ShimKind::ForeignBinary);

        let unrelated = temp.path().join("unrelated");
        fs::write(&unrelated, "#!/bin/sh\necho not this tool\n").unwrap();
        assert_eq!(classify_shim(&unrelated, &vera_home), ShimKind::Unrelated);
    }

    #[test]
    fn a_near_miss_name_is_never_deleted() {
        // `contains("vera")` matches /opt/veracrypt and vera-tool. Deleting a
        // stranger's launcher is the one outcome here that cannot be undone, so
        // ownership must be proven by pointing into the data directory. These
        // are reported rather than ignored: a real shim from an install layout
        // we do not recognise lands here too, and must not vanish silently.
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");

        let veracrypt = temp.path().join("veracrypt-launcher");
        fs::write(
            &veracrypt,
            "#!/bin/sh\nexec /opt/veracrypt/bin/veracrypt \"$@\"\n",
        )
        .unwrap();
        assert_eq!(
            classify_shim(&veracrypt, &vera_home),
            ShimKind::Ambiguous,
            "a veracrypt launcher must never be classified as ours"
        );

        let capitalized = temp.path().join("capitalized");
        fs::write(
            &capitalized,
            "#!/bin/sh\nexec /opt/VeraCrypt/bin/x \"$@\"\n",
        )
        .unwrap();
        assert_eq!(
            classify_shim(&capitalized, &vera_home),
            ShimKind::Ambiguous,
            "case must not decide whether a stranger's file gets deleted"
        );
    }

    #[test]
    fn merely_naming_the_data_directory_is_not_ownership() {
        // A foreign launcher that mentions the data path — in a comment, or as
        // an argument to something else entirely — must not be deleted for it.
        // Ownership means launching a binary from inside vera_home.
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");

        let commented = temp.path().join("commented");
        fs::write(
            &commented,
            format!(
                "#!/bin/sh\n# reads config from {}/config.json\nexec /opt/other/bin/other \"$@\"\n",
                vera_home.display()
            ),
        )
        .unwrap();
        assert_eq!(
            classify_shim(&commented, &vera_home),
            ShimKind::Ambiguous,
            "a comment naming the data dir is not proof of ownership"
        );

        let argument = temp.path().join("argument");
        fs::write(
            &argument,
            format!(
                "#!/bin/sh\nexec /usr/bin/backup-tool --source \"{}/models\" \"$@\"\n",
                vera_home.display()
            ),
        )
        .unwrap();
        assert_eq!(
            classify_shim(&argument, &vera_home),
            ShimKind::Ambiguous,
            "a tool that reads our directory is not our launcher"
        );

        // And the positive control, so this test cannot pass by classifying
        // everything as Ambiguous.
        let ours = temp.path().join("ours");
        fs::write(
            &ours,
            format!(
                "#!/bin/sh\nexec \"{}/bin/1.0.0/aarch64-apple-darwin/vera\" \"$@\"\n",
                vera_home.display()
            ),
        )
        .unwrap();
        assert_eq!(classify_shim(&ours, &vera_home), ShimKind::Ours);
    }

    #[test]
    fn our_path_as_an_argument_is_not_our_launcher() {
        // Only the program position counts. A tool that takes our binary as an
        // argument runs something else, and deleting it would be deleting a
        // stranger's file.
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");
        let vera_bin = format!(
            "{}{}bin{}1.0.0{}vera",
            vera_home.display(),
            MAIN_SEPARATOR,
            MAIN_SEPARATOR,
            MAIN_SEPARATOR
        );

        let runner = temp.path().join("runner");
        fs::write(
            &runner,
            format!("#!/bin/sh\nexec /usr/bin/backup --runner \"{vera_bin}\" \"$@\"\n"),
        )
        .unwrap();
        assert_eq!(
            classify_shim(&runner, &vera_home),
            ShimKind::Ambiguous,
            "our binary passed as an argument does not make this our launcher"
        );

        let inline_comment = temp.path().join("inline");
        fs::write(
            &inline_comment,
            format!("#!/bin/sh\nexec /usr/bin/other \"$@\" # was {vera_bin}\n"),
        )
        .unwrap();
        assert_eq!(
            classify_shim(&inline_comment, &vera_home),
            ShimKind::Ambiguous,
            "a trailing comment is not a launch"
        );

        // Positive control, including the `exec` prelude and an env assignment.
        let ours = temp.path().join("ours");
        fs::write(
            &ours,
            format!("#!/bin/sh\nVERA_LOG=warn exec \"{vera_bin}\" \"$@\"\n"),
        )
        .unwrap();
        assert_eq!(classify_shim(&ours, &vera_home), ShimKind::Ours);
    }

    #[test]
    fn a_parent_traversal_escaping_the_data_directory_is_not_ours() {
        // `<vera_home>/bin/../../other/vera` passes a prefix test and resolves
        // outside. Normalization is lexical on purpose: vera_home is already
        // deleted by the time this runs, so canonicalize would fail on exactly
        // the paths that matter.
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");
        let escaping = format!(
            "{}{}bin{}..{}..{}other{}vera",
            vera_home.display(),
            MAIN_SEPARATOR,
            MAIN_SEPARATOR,
            MAIN_SEPARATOR,
            MAIN_SEPARATOR,
            MAIN_SEPARATOR
        );

        let shim = temp.path().join("vera");
        fs::write(&shim, format!("#!/bin/sh\nexec \"{escaping}\" \"$@\"\n")).unwrap();

        assert_eq!(
            classify_shim(&shim, &vera_home),
            ShimKind::Ambiguous,
            "a target that traverses out of the data directory is not inside it"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_relative_owned_symlink_is_still_ours() {
        // `ln -s` commonly writes a relative target. Resolving it against the
        // shim's own directory is what the filesystem would do; without that a
        // real shim is reported instead of removed.
        let temp = tempdir().unwrap();
        let bin = temp.path().join("bin");
        fs::create_dir_all(&bin).unwrap();
        let vera_home = temp.path().join(".vera");
        let target = vera_home.join("bin").join("1.0.0").join("vera");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, BINARY_MAGIC).unwrap();

        let link = bin.join("vera");
        std::os::unix::fs::symlink(Path::new("..").join(".vera/bin/1.0.0/vera"), &link).unwrap();
        assert!(
            fs::read_link(&link).unwrap().is_relative(),
            "fixture must be a relative symlink or it tests the absolute path again"
        );

        assert_eq!(classify_shim(&link, &vera_home), ShimKind::Ours);
    }

    #[test]
    fn batch_comments_are_comments_too() {
        // `.cmd` shims comment with `rem`, `@rem` and `::`, in any case. A
        // comment naming the data directory is not evidence that the launcher
        // runs it — the same rule as `#`, which was the only one handled.
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");
        let launch = format!(
            "{}{}bin{}1.0.0{}vera.exe",
            vera_home.display(),
            MAIN_SEPARATOR,
            MAIN_SEPARATOR,
            MAIN_SEPARATOR
        );

        for prefix in ["rem", "REM", "@rem", "@REM", "::"] {
            let shim = temp
                .path()
                .join(format!("cmd-{}", prefix.replace('@', "at")));
            fs::write(
                &shim,
                format!("@echo off\n{prefix} launcher for {launch}\nother.exe %*\n"),
            )
            .unwrap();
            assert_eq!(
                classify_shim(&shim, &vera_home),
                ShimKind::Ambiguous,
                "`{prefix}` must be treated as a comment"
            );
        }

        // Positive control: the same path on a real command line is ownership.
        let real = temp.path().join("cmd-real");
        fs::write(&real, format!("@echo off\n\"{launch}\" %*\n")).unwrap();
        assert_eq!(classify_shim(&real, &vera_home), ShimKind::Ours);
    }

    #[test]
    fn a_quoted_launcher_path_containing_spaces_is_still_ours() {
        // Whitespace splitting alone tears a quoted path apart, so a home
        // directory with a space in it would stop being recognised — common on
        // Windows, where the shim lives under `C:\Users\First Last\`.
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join("First Last").join(".vera");
        let shim = temp.path().join("vera");
        fs::write(
            &shim,
            format!(
                "#!/bin/sh\nexec \"{}{}bin{}1.0.0{}vera\" \"$@\"\n",
                vera_home.display(),
                MAIN_SEPARATOR,
                MAIN_SEPARATOR,
                MAIN_SEPARATOR
            ),
        )
        .unwrap();

        assert_eq!(
            classify_shim(&shim, &vera_home),
            ShimKind::Ours,
            "a quoted path with a space in it is still our launcher"
        );
    }

    #[test]
    fn a_sibling_directory_sharing_the_prefix_is_not_ours() {
        // `${vera_home}-backup/bin/vera` starts with the same characters as
        // vera_home and belongs to somebody else. A prefix is not a boundary.
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");
        let backup = temp.path().join(".vera-backup");

        let shim = temp.path().join("vera");
        fs::write(
            &shim,
            format!("#!/bin/sh\nexec \"{}/bin/vera\" \"$@\"\n", backup.display()),
        )
        .unwrap();

        assert_eq!(
            classify_shim(&shim, &vera_home),
            ShimKind::Ambiguous,
            "a sibling directory that merely shares the prefix must not be deleted"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_into_a_sibling_directory_sharing_the_prefix_is_not_ours() {
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");
        let target = temp.path().join(".vera-backup").join("bin").join("vera");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, BINARY_MAGIC).unwrap();
        let link = temp.path().join("vera");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        assert_eq!(
            classify_shim(&link, &vera_home),
            ShimKind::Ambiguous,
            "path comparison must be component-wise, not a string prefix"
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_owned_symlink_is_removed_even_though_its_target_is_already_gone() {
        // Uninstall deletes vera_home before it reaches the shim loop, so an
        // owned shim symlinked into it is dangling by then. Anything that
        // follows the link — `Path::exists`, `read_to_string` — reports it as
        // absent, and the shim survives on PATH while the run reports success.
        let roots = roots();
        let target = roots.vera_home.join("bin").join("1.0.0").join("vera");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, BINARY_MAGIC).unwrap();
        let link = roots.user_bin_dir.join(shim_name());
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let (stdout, _) = uninstall(&roots, true);

        assert!(
            fs::symlink_metadata(&link).is_err(),
            "the owned shim must be removed even though its target was deleted first"
        );
        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert!(
            document["removed"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("PATH shim")),
            "the removal must be reported, not silent: {stdout}"
        );
        assert_eq!(document["complete"], serde_json::json!(true), "{stdout}");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_to_a_near_miss_target_is_never_deleted() {
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");
        let target = temp.path().join("vera-tool");
        fs::write(&target, "#!/bin/sh\necho other tool\n").unwrap();
        let link = temp.path().join("vera");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        assert_eq!(
            classify_shim(&link, &vera_home),
            ShimKind::Ambiguous,
            "a symlink to vera-tool is not proof of ownership"
        );
    }

    #[test]
    fn an_ambiguous_shim_is_reported_and_blocks_the_complete_claim() {
        let roots = roots();
        let shim = roots.user_bin_dir.join(shim_name());
        fs::write(
            &shim,
            "#!/bin/sh\nexec /opt/veracrypt/bin/veracrypt \"$@\"\n",
        )
        .unwrap();

        let (stdout, stderr) = uninstall(&roots, true);

        assert!(shim.exists(), "an unproven file must never be deleted");
        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(document["complete"], serde_json::json!(false), "{stdout}");
        assert_eq!(
            document["left_in_place"],
            serde_json::json!([shim.display().to_string()]),
            "{stdout}"
        );
        assert!(stderr.contains("Left in place"), "{stderr}");
    }

    #[cfg(unix)]
    #[test]
    fn classify_shim_recognizes_a_symlink_to_vera() {
        let temp = tempdir().unwrap();
        let vera_home = temp.path().join(".vera");
        let target = vera_home.join("bin").join("1.0.0").join("vera");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, BINARY_MAGIC).unwrap();
        let link = temp.path().join("vera");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        // Reached through the symlink arm, not the text arm: the target is
        // unreadable as text, so without that arm this would be ForeignBinary.
        assert_eq!(classify_shim(&link, &vera_home), ShimKind::Ours);
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
