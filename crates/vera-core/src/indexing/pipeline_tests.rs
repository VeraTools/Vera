//! Tests for the indexing pipeline orchestrator.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tempfile::TempDir;

use super::*;
use crate::CancellationToken;
use crate::config::VeraConfig;
use crate::embedding::test_helpers::MockProvider;
use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;

fn default_config() -> VeraConfig {
    VeraConfig::default()
}

#[derive(Clone)]
struct BlockingProvider {
    started: Arc<tokio::sync::Notify>,
}

impl EmbeddingProvider for BlockingProvider {
    async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.started.notify_one();
        std::future::pending().await
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(8)
    }
}

/// Cancels the token mid-request and then fails, modelling a provider error
/// that completes at the same moment cancellation fires.
struct CancelThenFailProvider;

impl EmbeddingProvider for CancelThenFailProvider {
    async fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Err(EmbeddingError::ApiError {
            status: 500,
            message: "provider exploded".to_string(),
        })
    }

    async fn embed_batch_cancellable(
        &self,
        texts: &[String],
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        cancel.cancel();
        self.embed_batch(texts).await
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(8)
    }
}

#[tokio::test]
async fn index_simple_repo() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    fs::write(dir.path().join("lib.py"), "def greet():\n    print('hi')\n").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary =
        index_repository_with_progress(dir.path(), &provider, &config, "mock-model", |_| {})
            .await
            .unwrap();

    assert_eq!(summary.files_parsed, 2);
    assert!(summary.chunks_created > 0);
    assert_eq!(summary.chunks_created, summary.embeddings_generated);
    assert_eq!(summary.binary_skipped, 0);
    assert_eq!(summary.error_skipped, 0);
    assert!(summary.elapsed_secs >= 0.0);

    // Verify index artifacts exist on disk.
    let idx = index_dir(dir.path());
    assert!(idx.join("metadata.db").exists());
    assert!(idx.join("vectors.db").exists());
    assert!(idx.join("bm25").exists());
}

#[tokio::test]
async fn pre_cancelled_index_stops_before_creating_artifacts() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let error = super::index_repository_with_progress_and_cancellation(
        dir.path(),
        &provider,
        &config,
        "mock-model",
        |_| {},
        &cancellation,
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("operation cancelled"));
    assert!(!index_dir(dir.path()).exists());
}

#[tokio::test]
async fn cancellation_stops_an_in_flight_embedding_request() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();
    let started = Arc::new(tokio::sync::Notify::new());
    let provider = BlockingProvider {
        started: started.clone(),
    };
    let config = default_config();
    let cancellation = CancellationToken::new();
    let task_cancellation = cancellation.clone();
    let path = dir.path().to_path_buf();

    let task = tokio::spawn(async move {
        super::index_repository_with_progress_and_cancellation(
            &path,
            &provider,
            &config,
            "mock-model",
            |_| {},
            &task_cancellation,
        )
        .await
    });
    started.notified().await;
    cancellation.cancel();

    let error = tokio::time::timeout(Duration::from_millis(250), task)
        .await
        .expect("cancellation must stop the active embedding request")
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("cancel"));
    assert!(!dir.path().join(".vera").exists());
}

#[tokio::test]
async fn cancellation_after_embedding_does_not_publish_index_artifacts() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();
    let provider = MockProvider::new(8);
    let config = default_config();
    let cancellation = CancellationToken::new();
    let progress_cancellation = cancellation.clone();

    let error = super::index_repository_with_progress_and_cancellation(
        dir.path(),
        &provider,
        &config,
        "mock-model",
        move |event| {
            if matches!(event, IndexProgress::EmbeddingDone { .. }) {
                progress_cancellation.cancel();
            }
        },
        &cancellation,
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("operation cancelled"));
    assert!(!dir.path().join(".vera").exists());
}

#[tokio::test]
async fn provider_error_wins_over_simultaneous_cancellation() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();
    let config = default_config();

    let error = super::index_repository_with_progress_and_cancellation(
        dir.path(),
        &CancelThenFailProvider,
        &config,
        "mock-model",
        |_| {},
        &CancellationToken::new(),
    )
    .await
    .unwrap_err();

    assert!(
        error.to_string().contains("embedding generation failed"),
        "provider error should win over the simultaneous cancellation: {error}"
    );
}

#[tokio::test]
async fn index_invalid_path() {
    let provider = MockProvider::new(8);
    let config = default_config();
    let result = index_repository(
        Path::new("/nonexistent/path/xyz"),
        &provider,
        &config,
        "mock-model",
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("does not exist"),
        "error should mention path does not exist: {err}"
    );
}

#[tokio::test]
async fn index_file_not_directory() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("file.txt");
    fs::write(&file, "content").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let result = index_repository(&file, &provider, &config, "mock-model").await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a directory"),
        "error should mention not a directory: {err}"
    );
}

#[tokio::test]
async fn index_empty_repo() {
    let dir = TempDir::new().unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(summary.files_parsed, 0);
    assert_eq!(summary.chunks_created, 0);
    assert_eq!(summary.embeddings_generated, 0);
}

#[tokio::test]
async fn index_skips_binary_files() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
    fs::write(dir.path().join("image.png"), "not real png data").unwrap();
    // File with null bytes (binary content).
    fs::write(dir.path().join("data.dat"), b"some\x00binary").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(summary.files_parsed, 1);
    assert!(summary.binary_skipped >= 1);
}

#[tokio::test]
async fn index_stores_correct_metadata() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("hello.rs"),
        "fn hello() {\n    println!(\"world\");\n}\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert!(summary.chunks_created > 0);

    // Verify metadata store contents.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let store = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    assert_eq!(store.chunk_count().unwrap(), summary.chunks_created as u64);

    // Verify vector store contents.
    let vstore = VectorStore::open(&idx.join("vectors.db"), 8).unwrap();
    assert_eq!(vstore.count().unwrap(), summary.embeddings_generated as u64);
}

#[tokio::test]
async fn index_stores_type_relations() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("types.ts"),
        "interface Loader {}\nclass Repo implements Loader {\n  run() {}\n}\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let store = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let relations = store.find_type_relations("Loader").unwrap();
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0].owner, "Repo");
    assert_eq!(relations[0].file_path, "types.ts");
}

#[tokio::test]
async fn reindex_with_different_embedding_dim_recreates_vector_store() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

    let config = default_config();

    let first_provider = MockProvider::new(8);
    index_repository(dir.path(), &first_provider, &config, "mock-model-8")
        .await
        .unwrap();

    let second_provider = MockProvider::new(4);
    let summary = index_repository(dir.path(), &second_provider, &config, "mock-model-4")
        .await
        .unwrap();

    assert!(summary.embeddings_generated > 0);

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let vstore = VectorStore::open(&idx.join("vectors.db"), 4).unwrap();
    assert_eq!(vstore.count().unwrap(), summary.embeddings_generated as u64);
}

#[tokio::test]
async fn index_stores_bm25_index() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("app.py"),
        "def authenticate_user(username, password):\n    return True\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let _summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    // Verify BM25 index can be searched.
    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let bm25 = Bm25Index::open(&idx.join("bm25")).unwrap();
    let results = bm25.search("authenticate", 10).unwrap();
    assert!(
        !results.is_empty(),
        "BM25 should find 'authenticate' keyword"
    );
}

#[tokio::test]
async fn index_summary_reports_parse_errors() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("good.rs"), "fn good() {}").unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    // No errors expected for a simple valid file.
    assert!(summary.parse_errors.is_empty());
    assert_eq!(summary.files_parsed, 1);
}

#[tokio::test]
async fn index_persists_tree_sitter_health() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("broken.rs"),
        "fn broken( {\n    let x = ;\n}\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(summary.files_with_tree_sitter_errors, 1);

    let idx = index_dir(&dir.path().canonicalize().unwrap());
    let store = MetadataStore::open(&idx.join("metadata.db")).unwrap();
    let states = store.file_states().unwrap();
    assert_eq!(states.len(), 1);
    assert!(states[0].tree_has_error);

    let stats = crate::stats::collect_stats(dir.path()).unwrap();
    assert_eq!(stats.index_health.files_with_tree_sitter_errors, 1);
}

#[tokio::test]
async fn index_handles_mixed_languages() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
    fs::write(dir.path().join("app.py"), "def run(): pass").unwrap();
    fs::write(dir.path().join("index.ts"), "function hello() {}").unwrap();
    fs::write(
        dir.path().join("config.toml"),
        "[package]\nname = \"test\"\n",
    )
    .unwrap();

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    assert_eq!(summary.files_parsed, 4);
    assert!(summary.chunks_created >= 4);
    assert_eq!(summary.chunks_created, summary.embeddings_generated);
}

#[tokio::test]
async fn index_permission_error_continues() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("good.rs"), "fn good() {}").unwrap();
    let unreadable = dir.path().join("secret.py");
    fs::write(&unreadable, "def secret(): pass").unwrap();

    // Make file unreadable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000)).unwrap();
    }

    let provider = MockProvider::new(8);
    let config = default_config();
    let summary = index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();

    // Should still complete successfully (exit 0).
    assert!(summary.files_parsed >= 1);

    // Restore permissions for cleanup.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o644));
    }
}

/// Freshness keys, duplicated here because `freshness` keeps them private.
const INDEXING_CONFIG_META_KEY: &str = "indexing_config";
const INDEX_REFRESHED_AT_META_KEY: &str = "index_refreshed_at_unix_ms";

fn store_index_into(idx_dir: &Path) -> Result<()> {
    let indexing_config = crate::config::IndexingConfig::default();
    store_index(
        idx_dir,
        &[],
        &[],
        &[("src/lib.rs".to_string(), "hash-a".to_string())],
        &[],
        &[],
        IndexBuildMetadata {
            file_states: &[],
            indexing_config: &indexing_config,
            model_name: "mock-model",
        },
    )
}

/// A failed store rebuild must not leave the index certified current.
///
/// Both stores are deleted and repopulated by `store_index`. The failure is
/// injected by leaving a plain file where the store expects a directory (and a
/// directory where it expects a file), which makes the reset step fail after
/// the preceding stores have already been written. This proves the hashes and
/// the freshness stamp land after both stores; it does not simulate a kill
/// signal, and `store_index` has no seam for one.
#[test]
fn store_index_defers_hashes_and_freshness_until_stores_are_rebuilt() {
    for poison in ["vector", "bm25"] {
        let dir = TempDir::new().unwrap();
        let idx_dir = dir.path().join(".vera");
        fs::create_dir_all(&idx_dir).unwrap();
        match poison {
            // `remove_file` fails on a directory.
            "vector" => fs::create_dir(idx_dir.join(VECTOR_DB)).unwrap(),
            // `remove_dir_all` fails on a plain file.
            _ => fs::write(idx_dir.join(BM25_SUBDIR), "not a directory").unwrap(),
        }

        let err = store_index_into(&idx_dir).unwrap_err();
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("failed to reset"),
            "{poison}: unexpected failure: {rendered}"
        );

        let store = MetadataStore::open(&idx_dir.join(METADATA_DB)).unwrap();
        assert_eq!(
            store.get_file_hash("src/lib.rs").unwrap(),
            None,
            "{poison}: hashes were committed before the stores were rebuilt"
        );
        assert_eq!(
            store.get_index_meta(INDEX_REFRESHED_AT_META_KEY).unwrap(),
            None,
            "{poison}: the index was stamped fresh before the stores were rebuilt"
        );
        assert_eq!(
            store.get_index_meta(INDEXING_CONFIG_META_KEY).unwrap(),
            None,
            "{poison}: the freshness snapshot was recorded before the stores were rebuilt"
        );
    }
}

/// The deferred writes still happen when the rebuild succeeds.
#[test]
fn store_index_records_hashes_and_freshness_on_success() {
    let dir = TempDir::new().unwrap();
    let idx_dir = dir.path().join(".vera");

    store_index_into(&idx_dir).unwrap();

    let store = MetadataStore::open(&idx_dir.join(METADATA_DB)).unwrap();
    assert_eq!(
        store.get_file_hash("src/lib.rs").unwrap().as_deref(),
        Some("hash-a")
    );
    assert!(
        store
            .get_index_meta(INDEX_REFRESHED_AT_META_KEY)
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .get_index_meta(INDEXING_CONFIG_META_KEY)
            .unwrap()
            .is_some()
    );
}
