//! Index freshness metadata and stale-index detection.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rayon::prelude::*;
use tracing::warn;

use crate::config::IndexingConfig;
use crate::discovery;
use crate::storage::metadata::MetadataStore;

use super::index_dir;
use super::update::{detect_language_for_path, hash_for_indexing_source};

const INDEXING_CONFIG_KEY: &str = "indexing_config";
const INDEX_REFRESHED_AT_KEY: &str = "index_refreshed_at_unix_ms";

/// Summary of drift between the working tree and the current index.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct IndexFreshness {
    pub files_added: usize,
    pub files_modified: usize,
    pub files_deleted: usize,
}

impl IndexFreshness {
    pub fn is_stale(&self) -> bool {
        self.total_changes() > 0
    }

    pub fn total_changes(&self) -> usize {
        self.files_added + self.files_modified + self.files_deleted
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.files_added > 0 {
            parts.push(format!("{} added", self.files_added));
        }
        if self.files_modified > 0 {
            parts.push(format!("{} modified", self.files_modified));
        }
        if self.files_deleted > 0 {
            parts.push(format!("{} deleted", self.files_deleted));
        }
        parts.join(", ")
    }
}

pub(crate) fn record_index_snapshot(
    metadata_store: &MetadataStore,
    indexing_config: &IndexingConfig,
) -> Result<()> {
    metadata_store
        .set_index_meta(
            INDEXING_CONFIG_KEY,
            &serde_json::to_string(indexing_config).context("failed to encode indexing config")?,
        )
        .context("failed to store indexing config metadata")?;
    metadata_store
        .set_index_meta(
            INDEX_REFRESHED_AT_KEY,
            &current_time_millis()
                .context("failed to compute index refresh timestamp")?
                .to_string(),
        )
        .context("failed to store index refresh timestamp")?;
    Ok(())
}

/// Compare the current repo contents against the index metadata.
///
/// New and deleted files are detected via discovery vs tracked files. Modified
/// files are verified with the stored content hashes for every tracked file.
pub fn detect_staleness(
    repo_path: &Path,
    fallback_config: &IndexingConfig,
) -> Result<IndexFreshness> {
    let repo_root = repo_path
        .canonicalize()
        .with_context(|| format!("failed to resolve repo path: {}", repo_path.display()))?;
    let metadata_path = index_dir(&repo_root).join("metadata.db");
    let metadata_store =
        MetadataStore::open(&metadata_path).context("failed to open metadata store")?;

    let indexing_config = load_indexing_config(&metadata_store, fallback_config);
    let discovery = discovery::discover_files(&repo_root, &indexing_config)
        .context("failed to discover files for freshness scan")?;

    let current_files: HashMap<String, PathBuf> = discovery
        .files
        .into_iter()
        .map(|file| (file.relative_path, file.absolute_path))
        .collect();
    let tracked_files: HashSet<String> = metadata_store
        .tracked_files()
        .context("failed to read tracked files")?
        .into_iter()
        .collect();

    let files_added = current_files
        .keys()
        .filter(|path| !tracked_files.contains(path.as_str()))
        .count();
    let files_deleted = tracked_files
        .iter()
        .filter(|path| !current_files.contains_key(path.as_str()))
        .count();

    let files_modified =
        count_modified_files(&current_files, &tracked_files, &metadata_store, &repo_root)?;

    Ok(IndexFreshness {
        files_added,
        files_modified,
        files_deleted,
    })
}

fn count_modified_files(
    current_files: &HashMap<String, PathBuf>,
    tracked_files: &HashSet<String>,
    metadata_store: &MetadataStore,
    repo_root: &Path,
) -> Result<usize> {
    let tracked_current: Vec<(&String, &PathBuf)> = current_files
        .iter()
        .filter(|(path, _)| tracked_files.contains(path.as_str()))
        .collect();

    // Reading and hashing is I/O and CPU bound, so it runs under rayon. The
    // metadata lookup that follows stays sequential: `MetadataStore` wraps a
    // single SQLite connection and is not `Sync`, so it cannot be called from
    // several rayon threads at once.
    //
    // `None` means the file could not be read. It is counted as modified
    // rather than skipped, so an unreadable tracked file cannot make a stale
    // index look current (#74).
    let hashed: Vec<(&String, Option<String>)> = tracked_current
        .par_iter()
        .map(|(rel_path, absolute_path)| {
            let content = match crate::discovery::read_source_lossy(absolute_path) {
                Ok(content) => content,
                Err(err) => {
                    warn!(
                        file = %rel_path,
                        error = %err,
                        "failed to read file during freshness scan"
                    );
                    return (*rel_path, None);
                }
            };
            let language = detect_language_for_path(rel_path);
            let current_hash = hash_for_indexing_source(&content, rel_path, language, repo_root);
            (*rel_path, Some(current_hash))
        })
        .collect();

    let mut files_modified = 0usize;
    for (rel_path, current_hash) in hashed {
        let Some(current_hash) = current_hash else {
            files_modified += 1;
            continue;
        };
        let stored_hash = metadata_store
            .get_file_hash(rel_path)
            .with_context(|| format!("failed to read stored hash for {rel_path}"))?;
        if stored_hash.as_deref() != Some(current_hash.as_str()) {
            files_modified += 1;
        }
    }

    Ok(files_modified)
}

fn load_indexing_config(
    metadata_store: &MetadataStore,
    fallback_config: &IndexingConfig,
) -> IndexingConfig {
    match metadata_store.get_index_meta(INDEXING_CONFIG_KEY) {
        Ok(Some(encoded)) => match serde_json::from_str(&encoded) {
            Ok(config) => config,
            Err(err) => {
                warn!(error = %err, "failed to decode saved indexing config");
                fallback_config.clone()
            }
        },
        Ok(None) => fallback_config.clone(),
        Err(err) => {
            warn!(error = %err, "failed to read saved indexing config");
            fallback_config.clone()
        }
    }
}

fn current_time_millis() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IndexingConfig;
    use crate::indexing::content_hash;
    use tempfile::tempdir;

    fn write_file(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn detects_added_modified_and_deleted_files() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", "pub fn current() {}\n");
        write_file(dir.path(), "src/new.rs", "pub fn added() {}\n");

        let index_dir = dir.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let metadata = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        metadata
            .set_file_hash("src/lib.rs", &content_hash("pub fn previous() {}\n"))
            .unwrap();
        metadata
            .set_file_hash("src/deleted.rs", &content_hash("pub fn deleted() {}\n"))
            .unwrap();

        let freshness = detect_staleness(dir.path(), &IndexingConfig::default()).unwrap();
        assert_eq!(
            freshness,
            IndexFreshness {
                files_added: 1,
                files_modified: 1,
                files_deleted: 1,
            }
        );
    }

    #[test]
    fn freshness_scan_checks_tracked_files_even_when_snapshot_is_newer() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", "pub fn current() {}\n");

        let index_dir = dir.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let metadata = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        metadata
            .set_file_hash("src/lib.rs", &content_hash("pub fn previous() {}\n"))
            .unwrap();
        metadata
            .set_index_meta(INDEX_REFRESHED_AT_KEY, &u64::MAX.to_string())
            .unwrap();

        let freshness = detect_staleness(dir.path(), &IndexingConfig::default()).unwrap();
        assert_eq!(freshness.files_modified, 1);
    }

    #[test]
    fn freshness_scan_marks_tracked_read_failures_as_modified() {
        let dir = tempdir().unwrap();
        let read_failure_path = dir.path().join("src/unreadable.rs");
        // Model a tracked file being replaced by a directory after discovery.
        std::fs::create_dir_all(&read_failure_path).unwrap();

        let index_dir = dir.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let metadata = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        metadata
            .set_file_hash("src/unreadable.rs", &content_hash("fn indexed() {}\n"))
            .unwrap();

        let current_files = HashMap::from([("src/unreadable.rs".to_string(), read_failure_path)]);
        let tracked_files = HashSet::from(["src/unreadable.rs".to_string()]);
        let files_modified =
            count_modified_files(&current_files, &tracked_files, &metadata, dir.path()).unwrap();

        assert_eq!(files_modified, 1);
    }

    #[test]
    fn freshness_scan_uses_saved_indexing_config() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "generated/out.rs", "pub fn generated() {}\n");

        let index_dir = dir.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let metadata = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();

        let saved_config = IndexingConfig {
            extra_excludes: vec!["generated/**".to_string()],
            ..Default::default()
        };
        record_index_snapshot(&metadata, &saved_config).unwrap();

        let freshness = detect_staleness(dir.path(), &IndexingConfig::default()).unwrap();
        assert_eq!(freshness.files_added, 0);
    }
}
