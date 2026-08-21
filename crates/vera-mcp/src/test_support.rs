//! Fixtures shared by the staleness and tool-result tests.
//!
//! Both build a repository whose working tree and recorded index hashes are
//! set independently, which is what makes an index look stale.

use std::path::Path;

use tempfile::TempDir;
use vera_core::indexing::content_hash;
use vera_core::storage::metadata::MetadataStore;

/// Write `content` to `relative` under `root`, creating parent directories.
pub fn write_file(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

/// Open (creating if needed) the metadata store of the index under `root`.
pub fn open_metadata(root: &Path) -> MetadataStore {
    let index_dir = root.join(".vera");
    std::fs::create_dir_all(&index_dir).unwrap();
    MetadataStore::open(&index_dir.join("metadata.db")).unwrap()
}

/// Build a repository holding `source` in `src/lib.rs`, indexed under the hash
/// of `indexed_source`. Passing the same text twice yields a fresh index.
pub fn repo_indexed_as(source: &str, indexed_source: &str) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_file(dir.path(), "src/lib.rs", source);
    let metadata = open_metadata(dir.path());
    metadata
        .set_file_hash("src/lib.rs", &content_hash(indexed_source))
        .unwrap();
    dir
}
