//! Containment for paths that come back out of an index.
//!
//! `.vera/` is an ordinary directory: it can be committed, cloned, and edited,
//! so every `file_path` stored in `metadata.db` is untrusted input. Joining one
//! onto the project root is only safe after it has been checked to stay inside
//! that root, because `Path::join` replaces the base when the right-hand side
//! is absolute, and `..` components traverse out of it.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use tracing::warn;

/// Where an untrusted path resolves relative to a project root.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Containment {
    /// Resolves inside the root; carries the canonicalized absolute path.
    Inside(PathBuf),
    /// Resolves outside the root, or has a shape that cannot resolve inside it.
    Escaped,
    /// Does not resolve at all: missing, unreadable, or a broken symlink.
    Unresolved,
}

/// Canonicalize the project root that `index_dir` lives in.
///
/// Resolve this once per query and reuse it: containment compares against a
/// canonicalized root, so the root itself has to be canonical for files under a
/// symlinked root to match.
pub(crate) fn canonical_project_root(index_dir: &Path) -> Result<PathBuf> {
    let parent = index_dir
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine project root from index dir"))?;
    // A relative index dir such as `.vera` has an empty parent, which names the
    // current directory rather than nothing.
    let root = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    root.canonicalize()
        .with_context(|| format!("failed to canonicalize project root: {}", root.display()))
}

/// Resolve `candidate` and report where it lands relative to `canonical_root`.
pub(crate) fn resolve_within(canonical_root: &Path, candidate: &Path) -> Containment {
    let Ok(canonical) = candidate.canonicalize() else {
        return Containment::Unresolved;
    };
    if canonical.starts_with(canonical_root) {
        Containment::Inside(canonical)
    } else {
        Containment::Escaped
    }
}

/// Resolve a path stored in the index against the canonicalized project root.
///
/// Returns `None` for anything that must not be read. An escape is logged:
/// a stored path that leaves the project root means the index disagrees with
/// the repository it sits in, and a silent skip would hide that.
pub(crate) fn resolve_indexed_path(canonical_root: &Path, relative: &str) -> Option<PathBuf> {
    let path = Path::new(relative);
    let shape_ok = !relative.is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir));

    let containment = if shape_ok {
        resolve_within(canonical_root, &canonical_root.join(path))
    } else {
        Containment::Escaped
    };

    match containment {
        Containment::Inside(path) => Some(path),
        Containment::Escaped => {
            warn!(
                file = %relative,
                root = %canonical_root.display(),
                "skipping indexed path that escapes the project root"
            );
            None
        }
        Containment::Unresolved => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        _temp: tempfile::TempDir,
        root: PathBuf,
        canary: PathBuf,
    }

    /// A project root with a canary file next to it, outside the root.
    fn fixture() -> Fixture {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "fn inside() {}\n").unwrap();
        let canary = temp.path().join("canary.txt");
        std::fs::write(&canary, "CANARY-SECRET\n").unwrap();
        let root = root.canonicalize().unwrap();
        Fixture {
            _temp: temp,
            root,
            canary,
        }
    }

    #[test]
    fn relative_path_inside_the_root_resolves() {
        let f = fixture();
        let resolved = resolve_indexed_path(&f.root, "src/lib.rs").unwrap();
        assert_eq!(resolved, f.root.join("src/lib.rs"));
    }

    #[test]
    fn cur_dir_components_in_stored_path_resolve_inside_the_root() {
        let f = fixture();
        for relative in ["./src/lib.rs", "src/./lib.rs"] {
            let resolved = resolve_indexed_path(&f.root, relative).unwrap();
            assert_eq!(resolved, f.root.join("src/lib.rs"));
        }
    }

    #[test]
    fn absolute_stored_path_is_rejected() {
        let f = fixture();
        let absolute = f.canary.to_str().unwrap();
        assert!(Path::new(absolute).is_absolute());
        assert_eq!(resolve_indexed_path(&f.root, absolute), None);
    }

    #[test]
    fn parent_traversal_stored_path_is_rejected() {
        let f = fixture();
        assert_eq!(resolve_indexed_path(&f.root, "../canary.txt"), None);
        assert_eq!(resolve_indexed_path(&f.root, "src/../../canary.txt"), None);
    }

    #[test]
    fn missing_file_is_unresolved_not_escaped() {
        let f = fixture();
        assert_eq!(
            resolve_within(&f.root, &f.root.join("src/gone.rs")),
            Containment::Unresolved
        );
    }

    #[test]
    fn resolve_within_reports_a_path_outside_the_root() {
        let f = fixture();
        assert_eq!(resolve_within(&f.root, &f.canary), Containment::Escaped);
    }

    #[test]
    fn canonical_project_root_is_the_index_dir_parent() {
        let f = fixture();
        let root = canonical_project_root(&f.root.join(".vera")).unwrap();
        assert_eq!(root, f.root);
    }

    #[test]
    fn relative_index_dir_takes_the_current_directory_as_the_root() {
        // `Path::new(".vera").parent()` is `""`, not `None`, and a trailing
        // slash is normalized away before the parent is taken, so both spellings
        // have to land on the current directory.
        let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
        assert_eq!(canonical_project_root(Path::new(".vera")).unwrap(), cwd);
        assert_eq!(canonical_project_root(Path::new(".vera/")).unwrap(), cwd);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_out_of_the_root_is_rejected_before_it_is_read() {
        let f = fixture();
        std::os::unix::fs::symlink(&f.canary, f.root.join("src/leak.rs")).unwrap();
        assert_eq!(resolve_indexed_path(&f.root, "src/leak.rs"), None);
    }
}
