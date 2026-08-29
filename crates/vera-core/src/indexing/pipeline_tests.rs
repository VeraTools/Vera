//! Tests for the indexing pipeline orchestrator.

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
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

#[derive(Clone)]
struct RecordingProvider {
    calls: Arc<Mutex<Vec<usize>>>,
    fail_on_call: Option<usize>,
    dim: usize,
}

impl RecordingProvider {
    fn new(dim: usize) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            fail_on_call: None,
            dim,
        }
    }

    fn failing_on_call(dim: usize, call: usize) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            fail_on_call: Some(call),
            dim,
        }
    }

    fn call_lengths(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }
}

impl EmbeddingProvider for RecordingProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(texts.len());
            calls.len()
        };
        if self.fail_on_call == Some(call) {
            return Err(EmbeddingError::ApiError {
                status: 503,
                message: "recording provider failed".to_string(),
            });
        }
        Ok(texts.iter().map(|_| vec![0.5; self.dim]).collect())
    }

    fn expected_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
}

fn write_window_corpus(dir: &Path, file_count: usize) {
    for index in 0..file_count {
        fs::write(
            dir.join(format!("file_{index:02}.rs")),
            format!("fn item_{index}() {{}}\n"),
        )
        .unwrap();
    }
}

fn index_fingerprint(idx_dir: &Path) -> Vec<(String, String)> {
    let store = MetadataStore::open(&idx_dir.join(METADATA_DB)).unwrap();
    let mut fingerprint = Vec::new();
    for file_path in store.indexed_files().unwrap() {
        let hash = store.get_file_hash(&file_path).unwrap().unwrap();
        for chunk in store.get_chunks_by_file(&file_path).unwrap() {
            fingerprint.push((chunk.id, hash.clone()));
        }
    }
    fingerprint
}

#[tokio::test]
async fn full_index_embeds_bounded_windows_and_preserves_order() {
    let windowed_dir = TempDir::new().unwrap();
    let single_window_dir = TempDir::new().unwrap();
    write_window_corpus(windowed_dir.path(), 9);
    write_window_corpus(single_window_dir.path(), 9);

    let config = default_config();
    let windowed_provider = RecordingProvider::new(8);
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_ref = events.clone();
    let windowed_summary = index_repository_with_progress_and_cancellation_with_window_target(
        windowed_dir.path(),
        &windowed_provider,
        &config,
        "mock-model",
        move |event| events_ref.lock().unwrap().push(event),
        &CancellationToken::new(),
        2,
    )
    .await
    .unwrap();

    let single_window_provider = RecordingProvider::new(8);
    index_repository_with_progress_and_cancellation_with_window_target(
        single_window_dir.path(),
        &single_window_provider,
        &config,
        "mock-model",
        |_| {},
        &CancellationToken::new(),
        1000,
    )
    .await
    .unwrap();

    let windowed_calls = windowed_provider.call_lengths();
    assert!(windowed_calls.len() >= 3, "expected at least three windows");
    assert!(windowed_calls.iter().all(|size| *size <= 2));
    assert_eq!(
        index_fingerprint(&index_dir(windowed_dir.path())),
        index_fingerprint(&index_dir(single_window_dir.path()))
    );

    let events = events.lock().unwrap().clone();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, IndexProgress::ParsingDone { .. }))
            .count(),
        1
    );
    let parsing_done = events
        .iter()
        .find_map(|event| match event {
            IndexProgress::ParsingDone { chunk_count } => Some(*chunk_count),
            _ => None,
        })
        .unwrap();
    assert_eq!(parsing_done, windowed_summary.chunks_created);

    // Honest-denominator contract (strengthened from the legacy lenient
    // `previous_total >=` assertion): growing per-window totals must never be
    // presented as a fixed denominator. The renderer interprets events via
    // `HonestProgressTracker`: before `ParsingDone` the embed indicator is
    // indeterminate (no percentage/total), after it switches to a fixed total.
    use crate::indexing::progress::{EmbedDisplay, HonestProgressTracker};

    let mut tracker = HonestProgressTracker::new();
    let mut parsing_done_index: Option<usize> = None;
    let mut fixed_total: Option<usize> = None;
    let mut seen_indeterminate = false;
    let mut seen_determinate = false;
    let mut last_done = 0usize;
    let mut last_fixed: Option<usize> = None;

    for (idx, event) in events.iter().enumerate() {
        if let IndexProgress::ParsingDone { chunk_count } = event {
            parsing_done_index = Some(idx);
            fixed_total = Some(*chunk_count);
        }
        if let Some(display) = tracker.handle(event) {
            match display {
                EmbedDisplay::Indeterminate { done } => {
                    assert!(
                        parsing_done_index.is_none(),
                        "indeterminate display must occur before ParsingDone (event {idx})"
                    );
                    assert!(
                        done >= last_done,
                        "done must be monotonic: {done} < {last_done}"
                    );
                    assert!(
                        !display.shows_percentage(),
                        "indeterminate must not show percentage"
                    );
                    seen_indeterminate = true;
                    last_done = done;
                }
                EmbedDisplay::Determinate { done, total } => {
                    assert!(
                        parsing_done_index.is_some(),
                        "determinate display only after ParsingDone (event {idx})"
                    );
                    assert_eq!(
                        total,
                        fixed_total.unwrap(),
                        "fixed total must equal ParsingDone chunk_count and never be restated"
                    );
                    if let Some(prev) = last_fixed {
                        assert_eq!(
                            prev, total,
                            "fixed total must never be restated at a different value"
                        );
                    }
                    last_fixed = Some(total);
                    assert!(
                        done >= last_done,
                        "done must be monotonic after ParsingDone"
                    );
                    assert!(
                        display.shows_percentage(),
                        "determinate must show percentage"
                    );
                    assert!(
                        display.message().contains(&format!("{done}/{total}")),
                        "determinate message must contain done/total"
                    );
                    seen_determinate = true;
                    last_done = done;
                }
                EmbedDisplay::Done { count } => {
                    assert_eq!(count, windowed_summary.chunks_created);
                    assert_eq!(count, windowed_summary.embeddings_generated);
                }
            }
        }
    }

    // Structural overlap: at least one embedding must start while parsing is
    // still in progress (windowed pipeline parses one window ahead).
    assert!(
        seen_indeterminate,
        "windowed pipeline must overlap parse/embed/store: at least one EmbeddingProgress while parsing still in progress"
    );
    assert!(
        seen_determinate,
        "must switch to fixed total after ParsingDone"
    );
    assert_eq!(fixed_total.unwrap(), windowed_summary.chunks_created);
    assert_eq!(last_done, windowed_summary.embeddings_generated);
}

#[tokio::test]
async fn failed_window_keeps_live_index_and_cleans_staging() {
    let dir = TempDir::new().unwrap();
    write_window_corpus(dir.path(), 6);
    let idx_dir = index_dir(dir.path());
    fs::create_dir_all(&idx_dir).unwrap();
    fs::write(idx_dir.join("sentinel"), "previous index").unwrap();

    let provider = RecordingProvider::failing_on_call(8, 2);
    let error = index_repository_with_progress_and_cancellation_with_window_target(
        dir.path(),
        &provider,
        &default_config(),
        "mock-model",
        |_| {},
        &CancellationToken::new(),
        2,
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("embedding generation failed"));
    assert_eq!(
        fs::read_to_string(idx_dir.join("sentinel")).unwrap(),
        "previous index"
    );
    assert!(!sibling_index_dir(&idx_dir, "build").exists());
    assert!(!sibling_index_dir(&idx_dir, "old").exists());
}

#[test]
fn stale_staging_and_old_index_are_recovered_on_startup() {
    let dir = TempDir::new().unwrap();
    let idx_dir = index_dir(dir.path());
    let old_dir = sibling_index_dir(&idx_dir, "old");
    let build_dir = sibling_index_dir(&idx_dir, "build");
    fs::create_dir_all(&old_dir).unwrap();
    fs::create_dir_all(&build_dir).unwrap();
    fs::write(old_dir.join("sentinel"), "previous index").unwrap();
    fs::write(build_dir.join("stale"), "discard me").unwrap();

    recover_index_directories(&idx_dir).unwrap();

    assert_eq!(
        fs::read_to_string(idx_dir.join("sentinel")).unwrap(),
        "previous index"
    );
    assert!(!old_dir.exists());
    assert!(!build_dir.exists());
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

/// A completed windowed build publishes hashes and the freshness stamp, the
/// markers `detect_staleness` relies on to treat the index as current.
#[tokio::test]
async fn successful_index_publishes_hashes_and_freshness() {
    let dir = TempDir::new().unwrap();
    write_window_corpus(dir.path(), 3);

    let provider = MockProvider::new(8);
    index_repository(dir.path(), &provider, &default_config(), "mock-model")
        .await
        .unwrap();

    let store = MetadataStore::open(&index_dir(dir.path()).join(METADATA_DB)).unwrap();
    for file_path in store.indexed_files().unwrap() {
        assert!(
            store.get_file_hash(&file_path).unwrap().is_some(),
            "{file_path}: hash missing after a successful build"
        );
    }
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
