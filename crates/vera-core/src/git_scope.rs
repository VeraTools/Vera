//! Git-aware path scoping for changed-file workflows.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Restrict a command to files selected from git state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitScope {
    /// Modified, staged, and untracked files in the current working tree.
    Changed,
    /// Files changed since a specific revision.
    Since(String),
    /// Files changed since `merge-base(HEAD, rev)`.
    Base(String),
}

/// Resolve a git scope to an exact set of paths relative to `index_root`.
///
/// The index stores paths relative to the directory that was indexed, which is
/// not necessarily the git repository root. Every git command therefore runs at
/// the repository root, so all output shares one base, and the result is
/// re-rooted onto the index root before it is returned.
pub fn resolve_scope(index_root: &Path, scope: &GitScope) -> Result<HashSet<String>> {
    let layout = RepoLayout::resolve(index_root)?;
    let toplevel = layout.toplevel.as_path();

    let files = match scope {
        GitScope::Changed => resolve_changed_files(toplevel)?,
        GitScope::Since(rev) => {
            validate_revision(rev)?;
            resolve_diff_files(toplevel, rev)?
        }
        GitScope::Base(rev) => {
            validate_revision(rev)?;
            let merge_base = run_git(toplevel, &["merge-base", "HEAD", rev])?;
            let merge_base = merge_base.trim();
            if merge_base.is_empty() {
                bail!("git merge-base returned an empty revision for {rev}");
            }
            resolve_diff_files(toplevel, merge_base)?
        }
    };

    let mut files = layout.reroot(files);
    add_platform_path_variants(&mut files);
    Ok(files)
}

/// Where the indexed directory sits inside the git repository containing it.
struct RepoLayout {
    /// Absolute path to the repository root. Every git command runs here so
    /// that no output depends on `diff.relative` or on git's per-command choice
    /// of base.
    toplevel: PathBuf,
    /// The index root relative to the repository root, `/`-separated and ending
    /// in `/`. Empty when the index root *is* the repository root.
    prefix: String,
}

impl RepoLayout {
    fn resolve(index_root: &Path) -> Result<Self> {
        // Two calls rather than one multi-flag call: `git rev-parse` separates
        // its answers by newline, and a repository path may contain one.
        let toplevel = run_git(index_root, &["rev-parse", "--show-toplevel"])
            .context("failed to verify git repository for changed-file scope")?;
        let toplevel = strip_trailing_newline(&toplevel);
        if toplevel.is_empty() {
            bail!(
                "git found no work tree for {} (a bare repository has no files to scope)",
                index_root.display()
            );
        }

        let prefix = run_git(index_root, &["rev-parse", "--show-prefix"])
            .context("failed to locate the indexed directory inside its git repository")?;

        Ok(Self {
            toplevel: PathBuf::from(toplevel),
            prefix: strip_trailing_newline(&prefix).to_string(),
        })
    }

    /// Translate repository-root-relative paths to index-root-relative ones,
    /// dropping the files that live outside the indexed directory.
    fn reroot(&self, paths: HashSet<String>) -> HashSet<String> {
        if self.prefix.is_empty() {
            return paths;
        }
        paths
            .into_iter()
            .filter_map(|path| {
                path.strip_prefix(&self.prefix)
                    .filter(|rest| !rest.is_empty())
                    .map(ToOwned::to_owned)
            })
            .collect()
    }
}

/// `git rev-parse` terminates its answer with a line ending.
fn strip_trailing_newline(output: &str) -> &str {
    output.trim_end_matches(['\r', '\n'])
}

/// Reject revisions that git would parse as options (e.g. `--output=...`).
/// Callers can come from MCP requests, so never pass raw input to git argv.
fn validate_revision(rev: &str) -> Result<()> {
    anyhow::ensure!(
        !rev.trim_start().starts_with('-'),
        "invalid git revision: {rev}"
    );
    Ok(())
}

fn resolve_changed_files(toplevel: &Path) -> Result<HashSet<String>> {
    let mut files = HashSet::new();
    for args in [
        vec![
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMRTUXB",
            "--cached",
            "--",
        ],
        vec!["diff", "--name-only", "-z", "--diff-filter=ACMRTUXB", "--"],
        vec!["ls-files", "--others", "--exclude-standard", "-z", "--"],
    ] {
        files.extend(parse_git_paths(&run_git(toplevel, &args)?));
    }
    Ok(files)
}

fn resolve_diff_files(toplevel: &Path, revision: &str) -> Result<HashSet<String>> {
    let output = run_git(
        toplevel,
        &[
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMRTUXB",
            revision,
            "--",
        ],
    )?;
    Ok(parse_git_paths(&output))
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let details = if stderr.is_empty() {
            format!(
                "git {} exited with status {}",
                args.join(" "),
                output.status
            )
        } else {
            stderr
        };
        bail!(details);
    }

    String::from_utf8(output.stdout)
        .with_context(|| format!("git {} produced non-UTF-8 output", args.join(" ")))
}

fn parse_git_paths(output: &str) -> HashSet<String> {
    output
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

// Git emits repository paths with `/`, while the index stores paths using the
// platform separator. Keep both spellings at the scope boundary because some
// callers compare paths directly instead of going through SearchFilters.
#[cfg(windows)]
fn add_platform_path_variants(paths: &mut HashSet<String>) {
    let windows_paths: Vec<String> = paths.iter().map(|path| path.replace('/', "\\")).collect();
    paths.extend(windows_paths);
}

#[cfg(not(windows))]
fn add_platform_path_variants(_paths: &mut HashSet<String>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use tempfile::TempDir;

    #[test]
    fn parse_git_paths_preserves_nul_delimited_pathnames() {
        let paths = parse_git_paths(" leading README.md \0\"quoted\".rs\0line\nbreak.py\0");
        assert!(paths.contains(" leading README.md "));
        assert!(paths.contains("\"quoted\".rs"));
        assert!(paths.contains("line\nbreak.py"));
        assert_eq!(paths.len(), 3);
    }

    #[test]
    fn parse_git_paths_skips_empty_records() {
        let paths = parse_git_paths("src/main.rs\0\0README.md\0");
        assert!(paths.contains("src/main.rs"));
        assert!(paths.contains("README.md"));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn resolve_changed_scope_includes_modified_and_untracked_files() {
        let repo = init_repo();
        std::fs::write(
            repo.path().join("tracked.rs"),
            "fn tracked() { println!(\"changed\"); }\n",
        )
        .unwrap();
        std::fs::write(repo.path().join("new.py"), "def created():\n    pass\n").unwrap();

        let paths = resolve_scope(repo.path(), &GitScope::Changed).unwrap();
        assert!(paths.contains("tracked.rs"));
        assert!(paths.contains("new.py"));
    }

    #[test]
    fn resolve_changed_scope_preserves_whitespace_and_quotes() {
        let repo = init_repo();
        let path = " leading \"quoted\" file.rs ";
        std::fs::write(repo.path().join(path), "fn changed() {}\n").unwrap();

        let paths = resolve_scope(repo.path(), &GitScope::Changed).unwrap();

        assert!(paths.contains(path));
    }

    #[test]
    fn resolve_since_scope_uses_revision_diff() {
        let repo = init_repo();
        let head = run_git(repo.path(), &["rev-parse", "HEAD"]).unwrap();
        std::fs::write(
            repo.path().join("tracked.rs"),
            "fn tracked() { println!(\"changed\"); }\n",
        )
        .unwrap();

        let paths = resolve_scope(repo.path(), &GitScope::Since(head.trim().to_string())).unwrap();
        assert!(paths.contains("tracked.rs"));
    }

    #[test]
    fn resolve_base_scope_uses_merge_base() {
        let repo = init_repo();
        run_git(repo.path(), &["checkout", "-b", "feature"]).unwrap();
        std::fs::write(
            repo.path().join("tracked.rs"),
            "fn tracked() { println!(\"feature\"); }\n",
        )
        .unwrap();
        commit_all(repo.path(), "feature commit");

        let paths = resolve_scope(repo.path(), &GitScope::Base("HEAD~1".to_string())).unwrap();
        assert!(paths.contains("tracked.rs"));
    }

    #[test]
    fn resolve_scope_rejects_option_like_revisions() {
        let repo = init_repo();
        let since = resolve_scope(
            repo.path(),
            &GitScope::Since("--output=/tmp/vera-pwn".to_string()),
        );
        assert!(since.is_err());
        let base = resolve_scope(repo.path(), &GitScope::Base("--all".to_string()));
        assert!(base.is_err());
    }

    #[test]
    fn reroot_drops_paths_outside_the_indexed_directory() {
        let layout = RepoLayout {
            toplevel: PathBuf::from("/repo"),
            prefix: "pkg/".to_string(),
        };

        let rerooted = layout.reroot(HashSet::from([
            "pkg/src/main.rs".to_string(),
            "pkg".to_string(),
            "pkgsuffix/sibling.rs".to_string(),
            "other/elsewhere.rs".to_string(),
            "README.md".to_string(),
        ]));

        assert_eq!(sorted_slash_paths(&rerooted), ["src/main.rs"]);
    }

    #[test]
    fn resolve_changed_scope_from_a_subdirectory_index_root_reroots_every_half() {
        let repo = init_monorepo();
        let index_root = repo.path().join("pkg");

        // One file per half of `resolve_changed_files`, so a half left on the
        // repository-root base is a missing member rather than an empty set.
        std::fs::write(
            index_root.join("src/tracked.rs"),
            "fn tracked() { println!(\"modified\"); }\n",
        )
        .unwrap();
        std::fs::write(index_root.join("src/staged.rs"), "fn staged() {}\n").unwrap();
        git(repo.path(), &["add", "pkg/src/staged.rs"]).unwrap();
        std::fs::write(index_root.join("src/untracked.rs"), "fn untracked() {}\n").unwrap();

        // Changed, but outside the indexed directory.
        std::fs::write(
            repo.path().join("other/elsewhere.rs"),
            "fn elsewhere() { println!(\"modified\"); }\n",
        )
        .unwrap();
        std::fs::write(repo.path().join("README.md"), "changed\n").unwrap();

        let paths = resolve_scope(&index_root, &GitScope::Changed).unwrap();

        assert_eq!(
            sorted_slash_paths(&paths),
            ["src/staged.rs", "src/tracked.rs", "src/untracked.rs"]
        );
    }

    #[test]
    fn resolve_since_scope_from_a_subdirectory_index_root_reroots_paths() {
        let repo = init_monorepo();
        let index_root = repo.path().join("pkg");
        let head = run_git(repo.path(), &["rev-parse", "HEAD"]).unwrap();
        std::fs::write(
            index_root.join("src/tracked.rs"),
            "fn tracked() { println!(\"changed\"); }\n",
        )
        .unwrap();
        std::fs::write(
            repo.path().join("other/elsewhere.rs"),
            "fn elsewhere() { println!(\"changed\"); }\n",
        )
        .unwrap();
        commit_all(repo.path(), "second commit");

        let paths = resolve_scope(&index_root, &GitScope::Since(head.trim().to_string())).unwrap();

        assert_eq!(sorted_slash_paths(&paths), ["src/tracked.rs"]);
    }

    #[test]
    fn resolve_base_scope_from_a_subdirectory_index_root_reroots_paths() {
        let repo = init_monorepo();
        let index_root = repo.path().join("pkg");
        run_git(repo.path(), &["checkout", "-b", "feature"]).unwrap();
        std::fs::write(
            index_root.join("src/tracked.rs"),
            "fn tracked() { println!(\"feature\"); }\n",
        )
        .unwrap();
        std::fs::write(
            repo.path().join("other/elsewhere.rs"),
            "fn elsewhere() { println!(\"feature\"); }\n",
        )
        .unwrap();
        commit_all(repo.path(), "feature commit");

        let paths = resolve_scope(&index_root, &GitScope::Base("HEAD~1".to_string())).unwrap();

        assert_eq!(sorted_slash_paths(&paths), ["src/tracked.rs"]);
    }

    #[test]
    fn resolve_changed_scope_at_the_repository_root_keeps_subdirectory_prefixes() {
        let repo = init_monorepo();
        std::fs::write(
            repo.path().join("pkg/src/tracked.rs"),
            "fn tracked() { println!(\"modified\"); }\n",
        )
        .unwrap();

        let paths = resolve_scope(repo.path(), &GitScope::Changed).unwrap();

        assert_eq!(sorted_slash_paths(&paths), ["pkg/src/tracked.rs"]);
    }

    #[test]
    fn resolve_scope_outside_a_git_repository_reports_the_missing_repository() {
        let dir = TempDir::new().unwrap();

        let err = resolve_scope(dir.path(), &GitScope::Changed).unwrap_err();

        let message = format!("{err:#}");
        assert!(
            message.contains("failed to verify git repository for changed-file scope"),
            "{message}"
        );
    }

    /// Windows keeps a `\`-spelled variant of every path alongside the `/` one;
    /// compare only the canonical spelling.
    fn sorted_slash_paths(paths: &HashSet<String>) -> Vec<String> {
        let mut sorted: Vec<String> = paths
            .iter()
            .filter(|path| !path.contains('\\'))
            .cloned()
            .collect();
        sorted.sort();
        sorted
    }

    fn init_repo() -> TempDir {
        let repo = TempDir::new().unwrap();
        git(repo.path(), &["init"]).unwrap();
        std::fs::write(repo.path().join("tracked.rs"), "fn tracked() {}\n").unwrap();
        commit_all(repo.path(), "initial commit");
        repo
    }

    /// A repository whose root sits one level above the directory that gets
    /// indexed, plus a sibling directory that must never enter the scope.
    fn init_monorepo() -> TempDir {
        let repo = TempDir::new().unwrap();
        git(repo.path(), &["init"]).unwrap();
        std::fs::create_dir_all(repo.path().join("pkg/src")).unwrap();
        std::fs::create_dir_all(repo.path().join("other")).unwrap();
        std::fs::write(repo.path().join("pkg/src/tracked.rs"), "fn tracked() {}\n").unwrap();
        std::fs::write(
            repo.path().join("other/elsewhere.rs"),
            "fn elsewhere() {}\n",
        )
        .unwrap();
        std::fs::write(repo.path().join("README.md"), "initial\n").unwrap();
        commit_all(repo.path(), "initial commit");
        repo
    }

    fn commit_all(repo_root: &Path, message: &str) {
        git(repo_root, &["add", "."]).unwrap();
        git(
            repo_root,
            &[
                "-c",
                "user.name=Vera Test",
                "-c",
                "user.email=vera@example.com",
                "commit",
                "-m",
                message,
            ],
        )
        .unwrap();
    }

    fn git(repo_root: &Path, args: &[&str]) -> Result<()> {
        let status = Command::new("git")
            .args(args)
            .current_dir(repo_root)
            .status()
            .with_context(|| format!("failed to run git {}", args.join(" ")))?;
        if !status.success() {
            bail!("git {} failed with status {}", args.join(" "), status);
        }
        Ok(())
    }
}
