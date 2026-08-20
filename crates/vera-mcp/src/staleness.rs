//! Stale-index detection for MCP tool results.
//!
//! MCP has no out-of-band channel the agent reads, so a `tracing::warn!` never
//! reaches it. The notice built here is attached to the tool result itself, in
//! the same wording `vera search` prints on stderr, so both surfaces agree.
//!
//! A freshness scan re-hashes every tracked file, which is too expensive to run
//! on every tool call, so the outcome is cached for [`STALENESS_TTL`].

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use vera_core::indexing::IndexFreshness;

/// How long one freshness scan result is reused before rescanning.
const STALENESS_TTL: Duration = Duration::from_secs(30);

static CACHE: Mutex<Option<CachedStaleness>> = Mutex::new(None);

struct CachedStaleness {
    repo: PathBuf,
    checked_at: Instant,
    notice: Option<String>,
}

/// Render the stale-index notice, or `None` when the index covers the tree.
///
/// The wording matches `warn_if_index_stale` in `vera-cli`.
pub fn notice_for(freshness: &IndexFreshness) -> Option<String> {
    if !freshness.is_stale() {
        return None;
    }
    Some(format!(
        "warning: index may be stale: {}. Search and grep only cover indexed files. Run `vera update .` or `vera watch .`.",
        freshness.summary()
    ))
}

/// Stale-index notice for `repo`, rescanning at most once per [`STALENESS_TTL`].
///
/// A scan failure is not a tool failure: it is logged and treated as fresh.
pub fn notice_for_repo(repo: &Path) -> Option<String> {
    let mut guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
    if let Some(cached) = guard
        .as_ref()
        .filter(|cached| cached.repo == repo && cached.checked_at.elapsed() < STALENESS_TTL)
    {
        return cached.notice.clone();
    }

    let config = crate::saved_config::load_saved_runtime_config();
    let notice = match vera_core::indexing::detect_staleness(repo, &config.indexing) {
        Ok(freshness) => notice_for(&freshness),
        Err(err) => {
            tracing::debug!(error = %err, "failed to check index freshness");
            None
        }
    };
    *guard = Some(CachedStaleness {
        repo: repo.to_path_buf(),
        checked_at: Instant::now(),
        notice: notice.clone(),
    });
    notice
}

#[cfg(test)]
mod tests {
    use super::*;
    use vera_core::indexing::content_hash;
    use vera_core::storage::metadata::MetadataStore;

    fn write_file(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn open_metadata(root: &Path) -> MetadataStore {
        let index_dir = root.join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        MetadataStore::open(&index_dir.join("metadata.db")).unwrap()
    }

    #[test]
    fn stale_index_produces_a_notice_naming_the_drift() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", "pub fn current() {}\n");
        write_file(dir.path(), "src/new.rs", "pub fn added() {}\n");

        let metadata = open_metadata(dir.path());
        metadata
            .set_file_hash("src/lib.rs", &content_hash("pub fn previous() {}\n"))
            .unwrap();
        drop(metadata);

        let notice = notice_for_repo(dir.path()).expect("stale index must produce a notice");
        assert_eq!(
            notice,
            "warning: index may be stale: 1 added, 1 modified. \
             Search and grep only cover indexed files. \
             Run `vera update .` or `vera watch .`."
        );
    }

    #[test]
    fn fresh_index_produces_no_notice() {
        let dir = tempfile::tempdir().unwrap();
        let source = "pub fn current() {}\n";
        write_file(dir.path(), "src/lib.rs", source);

        let metadata = open_metadata(dir.path());
        metadata
            .set_file_hash("src/lib.rs", &content_hash(source))
            .unwrap();
        drop(metadata);

        assert_eq!(notice_for_repo(dir.path()), None);
    }

    #[test]
    fn notice_reports_deletions() {
        let freshness = IndexFreshness {
            files_added: 0,
            files_modified: 0,
            files_deleted: 3,
        };
        assert_eq!(
            notice_for(&freshness).unwrap(),
            "warning: index may be stale: 3 deleted. \
             Search and grep only cover indexed files. \
             Run `vera update .` or `vera watch .`."
        );
    }
}
