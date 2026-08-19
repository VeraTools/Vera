//! Tests for incremental update logic.

use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tempfile::TempDir;

use crate::config::VeraConfig;
use crate::embedding::test_helpers::MockProvider;
use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::indexing::{
    UpdateOptions, UpdateProgress, index_dir, index_repository, update_repository,
    update_repository_with_options_and_progress, update_repository_with_progress,
};
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;

use super::content_hash;

fn default_config() -> VeraConfig {
    VeraConfig::default()
}

/// Create a temp repo with `files`, run the initial index, and return the
/// repo dir, provider, config, and initial index summary.
async fn indexed_repo(
    files: &[(&str, &str)],
) -> (
    TempDir,
    MockProvider,
    VeraConfig,
    crate::indexing::IndexSummary,
) {
    let dir = TempDir::new().unwrap();
    for (name, content) in files {
        let path = dir.path().join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }
    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    (dir, provider, config, summary)
}

struct BatchBoundProvider {
    max_batch_size: usize,
    active_inputs: AtomicUsize,
    peak_inputs: AtomicUsize,
}

struct FailingProvider;

impl EmbeddingProvider for FailingProvider {
    async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Err(EmbeddingError::ApiError {
            status: 503,
            message: "provider unavailable".to_string(),
        })
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(8)
    }
}

impl BatchBoundProvider {
    fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            active_inputs: AtomicUsize::new(0),
            peak_inputs: AtomicUsize::new(0),
        }
    }
}

impl EmbeddingProvider for BatchBoundProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.len() > self.max_batch_size {
            return Err(EmbeddingError::ApiError {
                status: 400,
                message: format!(
                    "batch contained {} inputs, limit is {}",
                    texts.len(),
                    self.max_batch_size
                ),
            });
        }

        let active = self.active_inputs.fetch_add(texts.len(), Ordering::SeqCst) + texts.len();
        self.peak_inputs.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(25)).await;
        self.active_inputs.fetch_sub(texts.len(), Ordering::SeqCst);

        Ok(vec![vec![0.0; 8]; texts.len()])
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(8)
    }

    fn max_batch_size(&self) -> Option<usize> {
        Some(self.max_batch_size)
    }
}

// ── Content hash tests ──────────────────────────────────────────────

#[test]
fn content_hash_deterministic() {
    let h1 = content_hash("fn main() {}");
    let h2 = content_hash("fn main() {}");
    assert_eq!(h1, h2);
}

#[test]
fn content_hash_different_content() {
    let h1 = content_hash("fn main() {}");
    let h2 = content_hash("fn main() { println!(\"hi\"); }");
    assert_ne!(h1, h2);
}

#[test]
fn content_hash_is_hex_sha256() {
    let h = content_hash("hello");
    // SHA-256 hex is 64 characters.
    assert_eq!(h.len(), 64);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

// ── Update: no changes ──────────────────────────────────────────────

#[tokio::test]
async fn update_no_changes() {
    let (dir, provider, config, idx_summary) =
        indexed_repo(&[("main.rs", "fn main() {\n    println!(\"hello\");\n}\n")]).await;
    assert!(idx_summary.chunks_created > 0);

    // Update with no changes.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 0);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 0);
    assert_eq!(update_summary.files_deferred, 0);
    assert!(update_summary.files_unchanged > 0);
    assert_eq!(
        update_summary.total_chunks,
        idx_summary.chunks_created as u64
    );
}

#[tokio::test]
async fn update_reports_progress_and_respects_max_in_flight_input_bound() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();

    let initial_provider = MockProvider::new(8);
    let initial_config = default_config();
    index_repository(dir.path(), &initial_provider, &initial_config, "mock-model")
        .await
        .unwrap();

    for name in [
        "one.rs", "two.rs", "three.rs", "four.rs", "five.rs", "six.rs",
    ] {
        fs::write(dir.path().join(name), format!("fn {}() {{}}\n", &name[..3])).unwrap();
    }

    let provider = BatchBoundProvider::new(2);
    let mut config = default_config();
    config.embedding.batch_size = 2;
    config.embedding.max_concurrent_requests = 8;
    config.embedding.max_in_flight_inputs = 4;

    let events = std::sync::Mutex::new(Vec::new());
    let summary =
        update_repository_with_progress(dir.path(), &provider, &config, "mock-model", |event| {
            events.lock().unwrap().push(event)
        })
        .await
        .unwrap();

    assert_eq!(summary.files_added, 6);
    assert_eq!(summary.files_deferred, 0);
    let peak_inputs = provider.peak_inputs.load(Ordering::SeqCst);
    assert!(peak_inputs > 2);
    assert!(peak_inputs <= 4);

    let events = events.into_inner().unwrap();
    assert!(matches!(
        events.first(),
        Some(UpdateProgress::DiscoveryDone { .. })
    ));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, UpdateProgress::ClassificationDone { added: 6, .. }))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        UpdateProgress::ParsingDone {
            file_count: 6,
            chunk_count
        } if *chunk_count >= 6
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        UpdateProgress::EmbeddingProgress { done, total } if done == total && *total >= 6
    )));
    assert!(matches!(events.last(), Some(UpdateProgress::StorageDone)));
}

#[tokio::test]
async fn update_keeps_indexed_data_when_a_discovered_file_cannot_be_read() {
    let (dir, provider, config, _) =
        indexed_repo(&[("main.rs", "fn preserved_symbol() -> u32 { 42 }\n")]).await;
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let hash_before = metadata.get_file_hash("main.rs").unwrap();
    let chunks_before: Vec<_> = metadata
        .get_chunks_by_file("main.rs")
        .unwrap()
        .into_iter()
        .map(|chunk| (chunk.id, chunk.content))
        .collect();
    let vectors_before = VectorStore::open(&idx.join("vectors.db"), 8)
        .unwrap()
        .count()
        .unwrap();
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    assert!(!bm25.search("preserved_symbol", 10).unwrap().is_empty());

    let source = dir.path().join("main.rs");
    let replaced = std::sync::atomic::AtomicBool::new(false);
    update_repository_with_progress(dir.path(), &provider, &config, "mock-model", |event| {
        if matches!(event, UpdateProgress::DiscoveryDone { .. })
            && !replaced.swap(true, Ordering::SeqCst)
        {
            fs::remove_file(&source).unwrap();
            fs::create_dir(&source).unwrap();
        }
    })
    .await
    .unwrap();

    assert_eq!(metadata.get_file_hash("main.rs").unwrap(), hash_before);
    let chunks_after: Vec<_> = metadata
        .get_chunks_by_file("main.rs")
        .unwrap()
        .into_iter()
        .map(|chunk| (chunk.id, chunk.content))
        .collect();
    assert_eq!(chunks_after, chunks_before);
    assert_eq!(
        VectorStore::open(&idx.join("vectors.db"), 8)
            .unwrap()
            .count()
            .unwrap(),
        vectors_before
    );
    assert!(!bm25.search("preserved_symbol", 10).unwrap().is_empty());
}

#[tokio::test]
async fn update_embedding_failure_preserves_existing_parse_data() {
    let (dir, _provider, config, _) = indexed_repo(&[(
        "types.ts",
        "class Loader {}\nclass CachedLoader extends Loader {}\n",
    )])
    .await;
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert_eq!(metadata.find_type_relations("Loader").unwrap().len(), 1);

    fs::write(
        dir.path().join("types.ts"),
        "class Saver {}\nclass CachedSaver extends Saver {}\n",
    )
    .unwrap();

    let error = update_repository(dir.path(), &FailingProvider, &config, "mock-model")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("embedding generation failed"));
    assert_eq!(metadata.find_type_relations("Loader").unwrap().len(), 1);
    assert!(metadata.find_type_relations("Saver").unwrap().is_empty());
}

#[tokio::test]
async fn update_max_files_defers_and_resumes_in_path_order() {
    let (dir, provider, config, _) = indexed_repo(&[("main.rs", "fn main() {}\n")]).await;

    for name in ["gamma.rs", "alpha.rs", "beta.rs"] {
        fs::write(dir.path().join(name), format!("fn {}() {{}}\n", &name[..4])).unwrap();
    }

    let options = UpdateOptions { max_files: Some(2) };
    let first = update_repository_with_options_and_progress(
        dir.path(),
        &provider,
        &config,
        "mock-model",
        &options,
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(first.files_added, 2);
    assert_eq!(first.files_deferred, 1);
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert!(metadata.get_file_hash("alpha.rs").unwrap().is_some());
    assert!(metadata.get_file_hash("beta.rs").unwrap().is_some());
    assert!(metadata.get_file_hash("gamma.rs").unwrap().is_none());

    let second = update_repository_with_options_and_progress(
        dir.path(),
        &provider,
        &config,
        "mock-model",
        &options,
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(second.files_added, 1);
    assert_eq!(second.files_deferred, 0);
    assert!(metadata.get_file_hash("gamma.rs").unwrap().is_some());
}

#[tokio::test]
async fn update_max_files_prioritizes_modifications_and_always_deletes() {
    let (dir, provider, config, _) = indexed_repo(&[
        ("modified.rs", "fn value() -> u8 { 1 }\n"),
        ("deleted.rs", "fn obsolete() {}\n"),
    ])
    .await;

    let modified_content = "fn value() -> u8 { 2 }\n";
    fs::write(dir.path().join("modified.rs"), modified_content).unwrap();
    fs::remove_file(dir.path().join("deleted.rs")).unwrap();
    fs::write(dir.path().join("added.rs"), "fn added() {}\n").unwrap();

    let options = UpdateOptions { max_files: Some(1) };
    let summary = update_repository_with_options_and_progress(
        dir.path(),
        &provider,
        &config,
        "mock-model",
        &options,
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(summary.files_modified, 1);
    assert_eq!(summary.files_added, 0);
    assert_eq!(summary.files_deleted, 1);
    assert_eq!(summary.files_deferred, 1);

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert_eq!(
        metadata.get_file_hash("modified.rs").unwrap().as_deref(),
        Some(content_hash(modified_content).as_str())
    );
    assert!(metadata.get_file_hash("deleted.rs").unwrap().is_none());
    assert!(metadata.get_file_hash("added.rs").unwrap().is_none());
}

// ── Update: file modified ───────────────────────────────────────────

#[tokio::test]
async fn update_modified_file() {
    let (dir, provider, config, idx_summary) = indexed_repo(&[
        ("main.rs", "fn main() {\n    println!(\"hello\");\n}\n"),
        ("lib.py", "def greet():\n    print('hi')\n"),
    ])
    .await;
    let _initial_chunks = idx_summary.chunks_created as u64;

    // Modify one file.
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"updated content\");\n}\n\nfn helper() {\n    // new function\n}\n",
    )
    .unwrap();

    // Update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 0);
    assert_eq!(update_summary.files_unchanged, 1); // lib.py unchanged

    // Verify updated content is in the index.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("updated content", 10).unwrap();
    assert!(!results.is_empty(), "BM25 should find updated content");

    // Verify helper function is searchable.
    let results = bm25.search("helper", 10).unwrap();
    assert!(!results.is_empty(), "BM25 should find new function");
}

#[tokio::test]
async fn update_detects_rst_include_dependency_changes() {
    let (dir, provider, config, _) = indexed_repo(&[
        (
            "docs/index.rst",
            "Guide\n=====\n\n.. include:: includes/common.rst.inc\n",
        ),
        (
            "docs/includes/common.rst.inc",
            "Original include fragment text.\n",
        ),
    ])
    .await;

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let indexed_files = metadata.indexed_files().unwrap();
    assert!(indexed_files.contains(&"docs/index.rst".to_string()));
    assert!(
        !indexed_files.contains(&"docs/includes/common.rst.inc".to_string()),
        "rst include fragments should not be indexed as standalone files"
    );

    fs::write(
        dir.path().join("docs/includes/common.rst.inc"),
        "Updated include fragment text from dependency.\n",
    )
    .unwrap();

    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 0);

    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25
        .search("Updated include fragment text from dependency", 10)
        .unwrap();
    assert!(
        !results.is_empty(),
        "BM25 should reflect updated included RST content"
    );
}

// ── Update: file added ──────────────────────────────────────────────

#[tokio::test]
async fn update_added_file() {
    let (dir, provider, config, idx_summary) =
        indexed_repo(&[("main.rs", "fn main() {\n    println!(\"hello\");\n}\n")]).await;
    let initial_chunks = idx_summary.chunks_created as u64;

    // Add a new file.
    fs::write(
        dir.path().join("utils.py"),
        "def utility_function():\n    return 42\n",
    )
    .unwrap();

    // Update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 0);
    assert_eq!(update_summary.files_added, 1);
    assert_eq!(update_summary.files_deleted, 0);
    assert!(
        update_summary.total_chunks > initial_chunks,
        "total chunks should increase after adding a file"
    );

    // Verify new file content is searchable.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("utility_function", 10).unwrap();
    assert!(
        !results.is_empty(),
        "BM25 should find content from new file"
    );

    // Verify stats: file count should have increased.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let files = metadata.indexed_files().unwrap();
    assert!(files.contains(&"utils.py".to_string()));
}

#[tokio::test]
async fn update_replaces_type_relations_for_modified_file() {
    let (dir, provider, config, _) = indexed_repo(&[(
        "types.ts",
        "interface Loader {}\nclass Repo implements Loader {\n}\n",
    )])
    .await;

    fs::write(
        dir.path().join("types.ts"),
        "interface Saver {}\nclass Repo implements Saver {\n}\n",
    )
    .unwrap();

    update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let store = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert!(store.find_type_relations("Loader").unwrap().is_empty());
    let saver = store.find_type_relations("Saver").unwrap();
    assert_eq!(saver.len(), 1);
    assert_eq!(saver[0].owner, "Repo");
}

// ── Update: file deleted ────────────────────────────────────────────

#[tokio::test]
async fn update_deleted_file() {
    let (dir, provider, config, idx_summary) = indexed_repo(&[
        ("main.rs", "fn main() {\n    println!(\"hello\");\n}\n"),
        ("lib.py", "def greet():\n    print('hi')\n"),
    ])
    .await;
    let initial_chunks = idx_summary.chunks_created as u64;

    // Delete a file.
    fs::remove_file(dir.path().join("lib.py")).unwrap();

    // Update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 0);
    assert_eq!(update_summary.files_added, 0);
    assert_eq!(update_summary.files_deleted, 1);
    assert!(
        update_summary.total_chunks < initial_chunks,
        "total chunks should decrease after deleting a file"
    );

    // Verify deleted content is no longer searchable.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("greet", 10).unwrap();
    assert!(
        results.is_empty(),
        "BM25 should not find content from deleted file"
    );

    // Verify stats: file count should have decreased.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let files = metadata.indexed_files().unwrap();
    assert!(!files.contains(&"lib.py".to_string()));
}

// ── Update: no index exists ─────────────────────────────────────────

#[tokio::test]
async fn update_without_index_fails() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();

    let result = update_repository(dir.path(), &provider, &config, "mock-model").await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("no index found"),
        "should report no index: {err}"
    );
}

// ── Update: consistency with fresh index ────────────────────────────

#[tokio::test]
async fn update_matches_fresh_index() {
    let (dir, provider, config, _) = indexed_repo(&[
        ("main.rs", "fn main() {\n    println!(\"hello\");\n}\n"),
        ("lib.py", "def greet():\n    print('hi')\n"),
        ("config.toml", "[package]\nname = \"test\"\n"),
    ])
    .await;

    // Apply a sequence of changes.
    // 1. Modify main.rs
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"updated\");\n}\nfn helper() { 42 }\n",
    )
    .unwrap();
    // 2. Delete lib.py
    fs::remove_file(dir.path().join("lib.py")).unwrap();
    // 3. Add a new file
    fs::write(
        dir.path().join("utils.rs"),
        "pub fn utility() -> i32 {\n    42\n}\n",
    )
    .unwrap();

    // Run update.
    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 1);
    assert_eq!(update_summary.files_deleted, 1);

    // Get stats after update.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let updated_metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let updated_chunk_count = updated_metadata.chunk_count().unwrap();
    let updated_files = updated_metadata.indexed_files().unwrap();
    let updated_languages = updated_metadata.language_stats().unwrap();

    // Now do a fresh index of the same directory.
    let _fresh_summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let fresh_metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let fresh_chunk_count = fresh_metadata.chunk_count().unwrap();
    let fresh_files = fresh_metadata.indexed_files().unwrap();
    let fresh_languages = fresh_metadata.language_stats().unwrap();

    // Verify consistency.
    assert_eq!(
        updated_chunk_count, fresh_chunk_count,
        "chunk count should match fresh index"
    );
    assert_eq!(
        updated_files, fresh_files,
        "indexed files should match fresh index"
    );
    assert_eq!(
        updated_languages, fresh_languages,
        "language stats should match fresh index"
    );
}

// ── Update: mixed operations ────────────────────────────────────────

#[tokio::test]
async fn update_mixed_add_modify_delete() {
    let (dir, provider, config, _) = indexed_repo(&[
        ("a.rs", "fn a() {}"),
        ("b.py", "def b(): pass"),
        ("c.go", "func c() {}"),
    ])
    .await;

    // Modify a.rs, delete b.py, add d.ts.
    fs::write(dir.path().join("a.rs"), "fn a_updated() { 42 }").unwrap();
    fs::remove_file(dir.path().join("b.py")).unwrap();
    fs::write(dir.path().join("d.ts"), "function d(): void {}").unwrap();

    let update_summary = update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(update_summary.files_modified, 1);
    assert_eq!(update_summary.files_added, 1);
    assert_eq!(update_summary.files_deleted, 1);
    assert_eq!(update_summary.files_unchanged, 1); // c.go

    // Verify the modified content is searchable.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("a_updated", 10).unwrap();
    assert!(!results.is_empty(), "should find updated function name");

    // Verify deleted content is gone.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let files = metadata.indexed_files().unwrap();
    assert!(!files.contains(&"b.py".to_string()));
    assert!(files.contains(&"d.ts".to_string()));
}

// ── Update: vector store consistency ────────────────────────────────

#[tokio::test]
async fn update_vector_store_consistent() {
    let (dir, provider, config, _) =
        indexed_repo(&[("main.rs", "fn main() { println!(\"hello\"); }")]).await;

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let initial_vec_count = VectorStore::open(&idx.join("vectors.db"), 8)
        .unwrap()
        .count()
        .unwrap();

    // Add a file.
    fs::write(dir.path().join("lib.py"), "def lib(): return 1").unwrap();
    update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let after_add_vec_count = VectorStore::open(&idx.join("vectors.db"), 8)
        .unwrap()
        .count()
        .unwrap();
    assert!(
        after_add_vec_count > initial_vec_count,
        "vector count should increase after adding a file"
    );

    // Delete the added file.
    fs::remove_file(dir.path().join("lib.py")).unwrap();
    update_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let after_del_vec_count = VectorStore::open(&idx.join("vectors.db"), 8)
        .unwrap()
        .count()
        .unwrap();
    assert_eq!(
        after_del_vec_count, initial_vec_count,
        "vector count should return to initial after delete"
    );

    // Verify metadata and vectors are in sync.
    let metadata = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert_eq!(
        metadata.chunk_count().unwrap(),
        after_del_vec_count,
        "metadata chunk count should match vector count"
    );
}
