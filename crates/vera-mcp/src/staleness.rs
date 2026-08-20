//! Stale-index detection for MCP tool results.
//!
//! MCP has no out-of-band channel the agent reads, so a `tracing::warn!` never
//! reaches it. The notice is `IndexFreshness::stale_warning`, the same string
//! `vera search` prints on stderr, attached to the tool result itself.
//!
//! A freshness scan re-hashes every tracked file, which is too expensive to run
//! on every tool call, so the outcome is cached for [`STALENESS_TTL`].

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long one freshness scan result is reused before rescanning.
const STALENESS_TTL: Duration = Duration::from_secs(30);

/// How many repositories keep a scan result at once. `get_overview` takes a
/// `path` argument, so one server answers about more than the tree it was
/// started in, and a single-entry cache would rescan on every alternation.
const CACHE_CAPACITY: usize = 8;

static CACHE: Mutex<Vec<CachedStaleness>> = Mutex::new(Vec::new());

struct CachedStaleness {
    repo: PathBuf,
    checked_at: Instant,
    notice: Option<String>,
}

/// Cache key for `repo`: the resolved path, so that `.` and the absolute path
/// to the same tree share one entry. An unresolvable path keys on itself.
fn cache_key(repo: &Path) -> PathBuf {
    std::fs::canonicalize(repo).unwrap_or_else(|_| repo.to_path_buf())
}

/// Cached notice for `key`, or `None` when there is no live entry.
fn cached_notice(key: &Path) -> Option<Option<String>> {
    let guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
    guard
        .iter()
        .find(|cached| cached.repo == key && cached.checked_at.elapsed() < STALENESS_TTL)
        .map(|cached| cached.notice.clone())
}

/// Record `notice` for `key`, replacing any entry for the same repository and
/// dropping the least recently scanned one once [`CACHE_CAPACITY`] is reached.
fn store_notice(key: PathBuf, notice: Option<String>) {
    let mut guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
    guard.retain(|cached| cached.repo != key);
    if guard.len() >= CACHE_CAPACITY {
        if let Some(oldest) = guard
            .iter()
            .enumerate()
            .min_by_key(|(_, cached)| cached.checked_at)
            .map(|(index, _)| index)
        {
            guard.remove(oldest);
        }
    }
    guard.push(CachedStaleness {
        repo: key,
        checked_at: Instant::now(),
        notice,
    });
}

/// Stale-index notice for `repo`, rescanning at most once per [`STALENESS_TTL`].
///
/// A scan failure is not a tool failure: it is logged and treated as fresh.
pub fn notice_for_repo(repo: &Path) -> Option<String> {
    let key = cache_key(repo);
    if let Some(notice) = cached_notice(&key) {
        return notice;
    }

    // Scanned without the lock: it re-hashes every tracked file, and holding
    // the lock across that would serialize unrelated repositories. A duplicate
    // concurrent scan of the same repository is wasteful, never wrong.
    let config = crate::saved_config::load_saved_runtime_config();
    let notice = match vera_core::indexing::detect_staleness(repo, &config.indexing) {
        Ok(freshness) => freshness.stale_warning(),
        Err(err) => {
            tracing::debug!(error = %err, "failed to check index freshness");
            None
        }
    };
    store_notice(key, notice.clone());
    notice
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{repo_indexed_as, write_file};

    #[test]
    fn stale_index_produces_a_notice_naming_the_drift() {
        let dir = repo_indexed_as("pub fn current() {}\n", "pub fn previous() {}\n");
        write_file(dir.path(), "src/new.rs", "pub fn added() {}\n");

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
        let source = "pub fn current() {}\n";
        let dir = repo_indexed_as(source, source);

        assert_eq!(notice_for_repo(dir.path()), None);
    }

    #[test]
    fn checking_one_repo_does_not_evict_another() {
        let first = repo_indexed_as("pub fn current() {}\n", "pub fn previous() {}\n");
        let second = repo_indexed_as("pub fn other() {}\n", "pub fn other() {}\n");

        let cached = notice_for_repo(first.path()).expect("stale index must produce a notice");
        assert_eq!(notice_for_repo(second.path()), None);

        // The first tree is now genuinely fresh, so a rescan would answer
        // `None`. Within the TTL the cached answer must survive the second
        // repository's check.
        write_file(first.path(), "src/lib.rs", "pub fn previous() {}\n");
        assert_eq!(
            notice_for_repo(first.path()).as_deref(),
            Some(cached.as_str()),
            "a second repository's check must not evict the first repository's cached scan"
        );
    }
}
