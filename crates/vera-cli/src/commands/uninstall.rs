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

/// The launcher file names the installers write. `packages/npm-cli/bin/vera.js`
/// emits `exec "<vera_home>/bin/<version>/<target>/vera" "$@"` on unix and the
/// `.exe` equivalent on Windows, so a genuine shim always launches a program
/// with one of these names, from inside the Vera home.
const VERA_LAUNCHER_NAMES: &[&str] = &["vera", "vera.exe", "vera.cmd"];

/// Words that stand in front of the program on a launch line without being it.
const LAUNCH_PREFIXES: &[&str] = &["exec", "command", "builtin", "nohup", "env"];

/// Whether a line of a shim is a comment rather than something that runs.
///
/// A comment naming Vera is not a launch of Vera, so ownership must not be
/// read out of one (#249). Covers the shell family (`#`, including the
/// shebang, which names the interpreter and not the program) and the batch
/// family (`rem`, `@rem`, `::`) that Windows `.cmd` shims are written in.
fn is_comment_line(line: &str) -> bool {
    let line = line.trim_start();
    let unprefixed = line.strip_prefix('@').unwrap_or(line);
    line.starts_with('#')
        || unprefixed.starts_with("::")
        || unprefixed
            .split_whitespace()
            .next()
            .is_some_and(|word| word.eq_ignore_ascii_case("rem"))
}

/// Whether a token on a launch line names Vera's own launcher.
///
/// Both halves are required. The file name alone is not ownership: somebody
/// else's `/opt/other/bin/vera` is named the same and is not ours. Containment
/// alone is not either. The installers only ever write a launcher that
/// satisfies both, so requiring both costs no real coverage.
///
/// Containment is lexical because the data directory has already been removed
/// by the time the launcher is classified, which makes `canonicalize`
/// unavailable, and it is component-wise so `/opt/vera-extra` is not read as
/// being inside `/opt/vera`.
fn token_belongs_to_vera(token: &str, vera_home: &Path) -> bool {
    let path = Path::new(token);
    if path.as_os_str().is_empty() {
        return false;
    }
    let is_launcher = path.file_name().is_some_and(|name| {
        let text = name.to_string_lossy();
        VERA_LAUNCHER_NAMES
            .iter()
            .any(|known| text.eq_ignore_ascii_case(known))
    });
    is_launcher && is_inside(path, vera_home)
}

/// Whether a launcher is written in the batch family rather than the shell
/// family. Decides which character escapes, and whether `#` comments.
///
/// `packages/npm-cli/bin/vera.js` writes `@echo off` and `%*` on Windows, so
/// both markers identify our own shim as well as anyone else's.
fn looks_like_batch(text: &str) -> bool {
    let first = text.lines().next().unwrap_or("").trim_start();
    // A shebang settles it: the kernel will run this through a shell, whatever
    // else the file happens to contain. Without this, a unix script carrying a
    // literal `%*` anywhere read as batch, which swaps the escape character
    // and so stops honouring `\;` — turning an escaped separator back into a
    // command boundary.
    if first.starts_with("#!") {
        return false;
    }
    first.to_ascii_lowercase().starts_with("@echo") || text.contains("%*") || text.contains("%~dp0")
}

/// Characters whose special meaning an escape suppresses. Escaping anything
/// else leaves the escape character in place, so a Windows path keeps its
/// backslashes and a unix path keeps a literal caret.
fn is_special(ch: char) -> bool {
    matches!(ch, '|' | '&' | ';' | '"' | '\'' | '`' | '#' | '\\' | '^') || ch.is_whitespace()
}

/// The program each command on a line runs.
///
/// Quoting, escaping and inline comments are all honoured, because each of
/// them decides whether a Vera-looking path is something the shell would
/// actually run. A separator or a space inside a quoted path must not invent a
/// command or cut a path in half — a `VERA_HOME` under a directory with a
/// space produces exactly that shim, and failing to recognize it leaves Vera's
/// own launcher on PATH.
fn launched_programs(line: &str, batch: bool) -> Vec<String> {
    let escape = if batch { '^' } else { '\\' };
    let mut programs = Vec::new();
    let mut tokens: Vec<String> = Vec::new();
    let mut token = String::new();
    // Tracked separately from `token`, which stays empty for a quoted empty
    // word: `""#` is a `#` inside a word, not the start of a comment.
    let mut word_started = false;
    let mut quote: Option<char> = None;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        // Single quotes suppress escaping in the shell.
        if quote != Some('\'') && ch == escape && chars.peek().copied().is_some_and(is_special) {
            token.push(chars.next().expect("peeked"));
            word_started = true;
            continue;
        }
        match quote {
            Some(open) if ch == open => quote = None,
            Some(_) => token.push(ch),
            None => match ch {
                // An unquoted `#` at a word boundary comments out the rest of
                // the line. Batch has no such rule, and truncating there would
                // lose our own launcher.
                '#' if !batch && !word_started => break,
                // `cmd` quotes with `"` alone; a bare apostrophe is ordinary
                // text there, and treating it as a quote swallowed the rest of
                // the line, hiding a launch after the next separator.
                '"' => {
                    quote = Some(ch);
                    word_started = true;
                }
                '\'' | '`' if !batch => {
                    quote = Some(ch);
                    word_started = true;
                }
                // Command separators outside quotes. `;` separates commands in
                // the shell only; `cmd` passes it through as argument text, so
                // splitting on it there invents a command that never runs.
                '|' | '&' => {
                    if word_started {
                        tokens.push(std::mem::take(&mut token));
                    }
                    word_started = false;
                    programs.extend(program_of(&tokens, batch));
                    tokens.clear();
                }
                ';' if !batch => {
                    if word_started {
                        tokens.push(std::mem::take(&mut token));
                    }
                    word_started = false;
                    programs.extend(program_of(&tokens, batch));
                    tokens.clear();
                }
                _ if ch.is_whitespace() => {
                    if word_started {
                        tokens.push(std::mem::take(&mut token));
                    }
                    word_started = false;
                }
                _ => {
                    token.push(ch);
                    word_started = true;
                }
            },
        }
    }
    if word_started {
        tokens.push(token);
    }
    programs.extend(program_of(&tokens, batch));
    programs
}

/// The program among one command's tokens, stepping over the words that stand
/// in front of it without being it.
///
/// Only this position decides ownership. An argument is not a launch: a script
/// that prints a Vera path and then runs something else has not run Vera.
fn program_of(tokens: &[String], batch: bool) -> Option<String> {
    tokens
        .iter()
        .map(|token| token.strip_prefix('@').unwrap_or(token))
        .find(|token| {
            !token.is_empty()
                && !LAUNCH_PREFIXES
                    .iter()
                    .any(|prefix| token.eq_ignore_ascii_case(prefix))
                // `NAME=value` prefixes set the environment for what follows.
                // Decided on the name, not on punctuation in the value: an
                // assignment to a path is still an assignment, and the program
                // is whatever comes after it. Shell only — `cmd` has no such
                // prefix form, so there a leading `NAME=...` is the program.
                && !(!batch
                    && token
                        .split_once('=')
                        .is_some_and(|(name, _)| is_variable_name(name)))
        })
        .map(str::to_string)
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

/// Whether a symlink at `entry` points into Vera's own files.
///
/// A relative target is resolved against the link's own directory first, so
/// `vera -> ../lib/vera/bin/vera` is judged on where it actually lands.
fn symlink_points_at_vera(entry: &Path, vera_home: &Path) -> bool {
    let Ok(target) = fs::read_link(entry) else {
        return false;
    };
    let resolved = if target.is_absolute() {
        target
    } else {
        entry.parent().unwrap_or(Path::new("")).join(target)
    };
    token_belongs_to_vera(&resolved.to_string_lossy(), vera_home)
}

/// Whether the text of a launcher script actually launches Vera.
///
/// Ownership is decided from the program each command runs, judged by file
/// name and by containment in the Vera home — not from the file containing the
/// four letters `vera` anywhere, which a comment, an argument, or an unrelated
/// word such as `coverage` satisfies (#249).
fn script_launches_vera(text: &str, vera_home: &Path) -> bool {
    let batch = looks_like_batch(text);
    text.lines()
        .filter(|line| !is_comment_line(line))
        .flat_map(|line| launched_programs(line, batch))
        .any(|program| token_belongs_to_vera(&program, vera_home))
}

/// Whether a word is a shell variable name, and so the left side of an
/// environment assignment rather than a program.
fn is_variable_name(word: &str) -> bool {
    let mut chars = word.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

/// Recognizes a candidate path as a removable Vera launcher, or leaves it
/// unclassified so unrelated files stay untouched.
///
/// A shim is a script that launches Vera or a symlink that points at one. The
/// cargo-installed binary is neither: it fails UTF-8 decoding by construction
/// (#212), so only "unreadable as text plus a regular executable file with an
/// exact Vera entry name" attributes it to cargo without grabbing anything
/// else named `vera`.
fn classify_launch_entry(entry: &Path, vera_home: &Path) -> Option<LaunchEntry> {
    let read_as_text = fs::read_to_string(entry);
    if read_as_text
        .as_ref()
        .is_ok_and(|text| script_launches_vera(text, vera_home))
        || symlink_points_at_vera(entry, vera_home)
    {
        return Some(LaunchEntry::Shim);
    }
    // Decodable text that never launches vera belongs to someone else.
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
            // `exists` follows the link and so reports `false` for a broken
            // one, which left a dangling Vera symlink on PATH while the run
            // still claimed a complete uninstall (#249). Ask about the link
            // itself instead.
            if entry.symlink_metadata().is_err() {
                continue;
            }
            let Some(kind) = classify_launch_entry(&entry, vera_home) else {
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

    /// The shim the installers write. Kept next to the foreign fixtures so the
    /// suite proves the classifier separates them rather than only proving it
    /// declines things.
    #[cfg(unix)]
    fn install_vera_shim(dir: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        fs::create_dir_all(dir).unwrap();
        let path = dir.join("vera");
        fs::write(&path, body).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// The exact shape `packages/npm-cli/bin/vera.js` writes, built from the
    /// same `vera_home` the uninstall resolves, so the positive case is the
    /// real installer output rather than an invented one.
    #[cfg(unix)]
    #[test]
    fn uninstall_removes_the_shim_the_installer_writes() {
        let roots = roots();
        let binary = roots
            .vera_home
            .join("bin")
            .join("1.3.0")
            .join("aarch64-apple-darwin")
            .join("vera");
        let shim = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!("#!/bin/sh\nexec \"{}\" \"$@\"\n", binary.display()),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(!shim.exists(), "our own shim survived");
        assert!(stderr.contains("Removed PATH shim"), "{stderr}");
    }

    /// Ownership survives a chained command: the program still has to be Vera,
    /// but it does not have to be the first thing on the line.
    #[cfg(unix)]
    #[test]
    fn a_chained_command_that_ends_in_vera_is_still_ours() {
        let roots = roots();
        let shim = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\ncd /tmp && exec \"{}/bin/1.3.0/aarch64-apple-darwin/vera\" \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            !shim.exists(),
            "our own shim survived a chained launch line"
        );
        assert!(stderr.contains("Removed PATH shim"), "{stderr}");
    }

    /// #249: ownership is read from what the script runs, not from the four
    /// letters `vera` appearing anywhere in it. A foreign launcher that only
    /// mentions Vera in a comment, and one whose path merely starts with the
    /// same letters, must both survive.
    #[cfg(unix)]
    #[test]
    fn a_mention_of_vera_outside_the_launch_line_does_not_make_a_script_ours() {
        for body in [
            // Named in a shell comment.
            "#!/bin/sh\n# drop-in replacement for vera\nexec /usr/bin/rg \"$@\"\n",
            // Named in a batch comment, which `.cmd` shims are written in.
            "@echo off\nREM wrapper around vera\n\"%~dp0\\rg.exe\" %*\n",
            // Not named at all: a longer word that contains the letters.
            "#!/bin/sh\nexec /opt/veracrypt/bin/veracrypt \"$@\"\n",
            // A directory that starts with the same letters but is not ours.
            "#!/bin/sh\nexec /opt/vera-extra/bin/tool \"$@\"\n",
            // A directory literally named `vera` that holds someone else's
            // program: the launcher is `rg`, so the entry is not ours.
            "#!/bin/sh\nexec /opt/vera/bin/rg \"$@\"\n",
            // A foreign executable that merely shares the launcher name.
            "#!/bin/sh\nexec /opt/other/bin/vera \"$@\"\n",
            // A real Vera path, but in an argument rather than in program
            // position. Printing a path is not launching it.
            "#!/bin/sh\necho \"{home}/bin/vera\"\nexec /usr/bin/rg \"$@\"\n",
        ] {
            let roots = roots();
            // `{home}` stands for the resolved Vera home: the classifier does
            // not expand shell variables, so a `$HOME` fixture would be
            // rejected for the wrong reason.
            let body = &body.replace("{home}", &roots.vera_home.display().to_string());
            let foreign = install_vera_shim(&roots.home.join(".local").join("bin"), body);

            let (_, stderr) = uninstall(&roots, false);

            assert!(
                foreign.exists(),
                "deleted a script that never launches vera: {body:?}"
            );
            assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
        }
    }

    /// #249: the symlink arm gets the same component-wise treatment. A link
    /// named `vera` that points at somebody else's binary is not ours.
    #[cfg(unix)]
    #[test]
    fn a_symlink_is_judged_by_where_it_lands_not_by_its_target_spelling() {
        let roots = roots();
        let bin = roots.home.join(".local").join("bin");
        fs::create_dir_all(&bin).unwrap();
        let elsewhere = roots.home.join("veracrypt-bin");
        fs::create_dir_all(&elsewhere).unwrap();
        let foreign_target = elsewhere.join("veracrypt");
        fs::write(&foreign_target, "binary").unwrap();
        let link = bin.join("vera");
        std::os::unix::fs::symlink(&foreign_target, &link).unwrap();

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            link.symlink_metadata().is_ok(),
            "deleted a symlink into someone else's install"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// #249: a Vera symlink whose target is already gone still occupies the
    /// name on PATH. `exists()` follows the link and reported `false`, so the
    /// entry was skipped and the run claimed a clean uninstall over it.
    #[cfg(unix)]
    #[test]
    fn a_dangling_vera_symlink_is_removed_rather_than_skipped() {
        let roots = roots();
        let bin = roots.home.join(".local").join("bin");
        fs::create_dir_all(&bin).unwrap();
        let link = bin.join("vera");
        std::os::unix::fs::symlink(roots.vera_home.join("bin").join("vera"), &link).unwrap();
        assert!(
            !link.exists(),
            "fixture does not discriminate: the target must be missing"
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            link.symlink_metadata().is_err(),
            "the dangling shim stayed on PATH"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// A Vera home under a directory with a space produces a shim whose binary
    /// path is one quoted token. Splitting on whitespace cut it in half, so the
    /// launcher went unrecognized and stayed on PATH while the run reported a
    /// complete uninstall — the same dishonesty as #212, from a different
    /// direction.
    #[cfg(unix)]
    #[test]
    fn a_vera_home_containing_a_space_is_still_recognized() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("My Home");
        let bin = home.join(".local").join("bin");
        let vera_home = home.join(".vera");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(temp.path().join("project")).unwrap();
        let shim = install_vera_shim(
            &bin,
            &format!(
                "#!/bin/sh\nexec \"{}/bin/1.3.0/x/vera\" \"$@\"\n",
                vera_home.display()
            ),
        );

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        run_at(
            &home,
            &vera_home,
            &temp.path().join("project"),
            Some(bin.as_path()),
            false,
            &mut stdout,
            &mut stderr,
        )
        .unwrap();

        assert!(
            !shim.exists(),
            "a shim under a path with a space was left on PATH"
        );
    }

    /// An inline `#` comments out the rest of the line, separators included.
    /// Parsing through it found a Vera path in what looked like program
    /// position and deleted a foreign launcher the shell never ran.
    #[cfg(unix)]
    #[test]
    fn an_inline_comment_ends_the_line() {
        let roots = roots();
        let foreign = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\necho safe # ; exec \"{}/bin/1.3.0/x/vera\"\nexec /usr/bin/rg \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            foreign.exists(),
            "a commented-out launch line was read as a real one"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// A `#` inside a word is not a comment, so a launch is still a launch.
    #[cfg(unix)]
    #[test]
    fn a_hash_inside_a_word_does_not_start_a_comment() {
        let roots = roots();
        let shim = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\nexec \"{}/bin/1.0#rc1/x/vera\" \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(!shim.exists(), "our own shim survived: {stderr}");
    }

    /// An escaped separator is literal text, not a command boundary. Splitting
    /// on it invented a command whose first token was a Vera path.
    #[cfg(unix)]
    #[test]
    fn an_escaped_separator_does_not_invent_a_command() {
        let roots = roots();
        let foreign = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\necho a\\; {}/bin/vera\nexec /usr/bin/rg \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            foreign.exists(),
            "an escaped separator was read as a command boundary"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// `NAME=value` in front of the program is an assignment, whatever the
    /// value looks like. Treating an assignment to a path as the program meant
    /// the real program was never examined, so our own shim survived.
    #[cfg(unix)]
    #[test]
    fn an_environment_assignment_is_not_the_program() {
        let roots = roots();
        let shim = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\nVERA_LOG=/var/log/vera.log exec \"{}/bin/1.3.0/x/vera\" \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(!shim.exists(), "our own shim survived: {stderr}");
    }

    /// A `#` after an empty quoted word is inside that word, so the line keeps
    /// running. Reading it as a comment stopped the scan before the launch and
    /// left Vera's own shim on PATH.
    #[cfg(unix)]
    #[test]
    fn a_hash_after_an_empty_quoted_word_is_not_a_comment() {
        let roots = roots();
        let shim = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\necho \"\"# ; exec \"{}/bin/1.3.0/x/vera\" \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(!shim.exists(), "our own shim survived: {stderr}");
    }

    /// `#` is ordinary text in the batch family, so a `.cmd` shim that launches
    /// Vera after one is still ours. The shell rule must not reach it.
    #[cfg(unix)]
    #[test]
    fn a_batch_shim_is_not_truncated_at_a_hash() {
        let roots = roots();
        let shim = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "@echo off\r\necho # && \"{}/bin/1.3.0/x/vera\" %*\r\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            !shim.exists(),
            "a batch shim was truncated at a hash: {stderr}"
        );
    }

    /// A shebang decides the family whatever else the file contains. A unix
    /// script that merely mentions `%*` used to read as batch, which swaps the
    /// escape character and so stopped honouring `\;`, turning an escaped
    /// separator back into a command boundary.
    #[cfg(unix)]
    #[test]
    fn a_shebang_settles_the_family_against_a_stray_batch_marker() {
        let roots = roots();
        let foreign = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\n# prints %* like a batch file would\necho a\\; {}/bin/vera\nexec /usr/bin/rg \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            foreign.exists(),
            "a shell script was parsed as batch and its escape ignored"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// `cmd` has no `NAME=value` prefix form, so a batch line starting with one
    /// is running that token, not setting a variable and running the next.
    #[cfg(unix)]
    #[test]
    fn a_batch_line_has_no_environment_assignment_prefix() {
        let roots = roots();
        let foreign = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "@echo off\r\nNAME=C:\\tmp \"{}/bin/vera\" %*\r\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            foreign.exists(),
            "a batch line was read as a shell assignment and the next token taken as the program"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// `;` separates commands in the shell but is ordinary argument text in
    /// `cmd`. Splitting on it in a batch file invents a command that never
    /// runs, and its first token was a Vera path.
    #[cfg(unix)]
    #[test]
    fn a_semicolon_is_not_a_batch_command_separator() {
        let roots = roots();
        let foreign = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "@echo off\r\necho safe; \"{}/bin/vera\"\r\n\"%~dp0\\rg.exe\" %*\r\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            foreign.exists(),
            "a batch line was split on `;` and a Vera path taken as a program"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// `cmd` quotes with `"` alone. Treating an apostrophe as a quote swallowed
    /// the rest of the line, so a real launch after the next separator was
    /// never seen and our own shim stayed on PATH.
    #[cfg(unix)]
    #[test]
    fn an_apostrophe_does_not_quote_in_a_batch_shim() {
        let roots = roots();
        let shim = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "@echo off\r\necho don't & \"{}/bin/1.3.0/x/vera\" %*\r\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(!shim.exists(), "our own batch shim survived: {stderr}");
    }

    /// A separator inside a quoted argument is data, not a command boundary.
    /// Splitting on it invented a second command whose first token was a Vera
    /// path, and the foreign launcher was deleted.
    #[cfg(unix)]
    #[test]
    fn a_separator_inside_quotes_does_not_invent_a_command() {
        let roots = roots();
        let foreign = install_vera_shim(
            &roots.home.join(".local").join("bin"),
            &format!(
                "#!/bin/sh\necho \"see; {}/bin/vera\"\nexec /usr/bin/rg \"$@\"\n",
                roots.vera_home.display()
            ),
        );

        let (_, stderr) = uninstall(&roots, false);

        assert!(
            foreign.exists(),
            "a quoted argument was read as a second command and the entry deleted"
        );
        assert!(stderr.contains("Vera has been uninstalled."), "{stderr}");
    }

    /// `..` has to be resolved before the containment test, or a path that
    /// climbs back out of Vera's home would still read as inside it.
    #[test]
    fn containment_resolves_parent_segments_and_respects_component_boundaries() {
        let home = Path::new("/opt/vera");
        assert!(is_inside(Path::new("/opt/vera/bin/vera"), home));
        assert!(is_inside(Path::new("/opt/vera/bin/../bin/vera"), home));
        assert!(!is_inside(Path::new("/opt/vera/../other/vera"), home));
        assert!(!is_inside(Path::new("/opt/vera-extra/bin"), home));
        // An empty root would otherwise make every path "inside" it.
        assert!(!is_inside(Path::new("/opt/vera"), Path::new("")));
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
