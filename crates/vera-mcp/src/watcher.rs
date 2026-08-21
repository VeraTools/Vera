//! Background file watcher for automatic index updates in MCP mode.
//!
//! Watches a project directory for file changes and triggers incremental
//! index updates after a debounce period. This keeps the index fresh
//! without requiring manual update calls.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify_debouncer_mini::{DebouncedEventKind, new_debouncer};
use tracing::{debug, info, warn};

use vera_core::config::IndexingConfig;
use vera_core::discovery::ExclusionMatcher;
use vera_core::indexing::UpdateSummary;
use vera_core::storage::metadata::MetadataStore;

/// Debounce interval: wait this long after the last file change before updating.
const DEBOUNCE_SECS: u64 = 2;

/// Runs one incremental update. Built once per watcher and reused across cycles,
/// so the tokio runtime and the embedding model are paid for once rather than
/// per debounce window.
trait IncrementalUpdate: Send + Sync {
    fn update(&self, repo_path: &Path) -> Result<UpdateSummary, anyhow::Error>;
}

type Engine = Arc<dyn IncrementalUpdate>;
type EngineBuilder = Arc<dyn Fn() -> Result<Engine, anyhow::Error> + Send + Sync>;

/// Holds the update engine for the lifetime of the watcher.
///
/// Construction is deferred to the first cycle so starting a watcher never
/// loads a model, and a construction failure stays a per-cycle error that the
/// next cycle can retry instead of killing the watcher.
struct SharedEngine {
    build: EngineBuilder,
    cached: Mutex<Option<Engine>>,
}

impl SharedEngine {
    fn get(&self) -> Result<Engine, anyhow::Error> {
        let mut guard = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(engine) = guard.as_ref() {
            return Ok(Arc::clone(engine));
        }
        let engine = (self.build)()?;
        *guard = Some(Arc::clone(&engine));
        Ok(engine)
    }
}

/// The production engine: one tokio runtime and one embedding provider.
struct EmbeddingUpdateEngine {
    runtime: tokio::runtime::Runtime,
    provider: vera_core::embedding::DynamicProvider,
    config: vera_core::config::VeraConfig,
    model_name: String,
}

impl EmbeddingUpdateEngine {
    fn build(
        config: vera_core::config::VeraConfig,
        backend: vera_core::config::InferenceBackend,
    ) -> Result<Self, anyhow::Error> {
        let runtime = tokio::runtime::Runtime::new()?;
        let (provider, model_name) = runtime.block_on(
            vera_core::embedding::create_dynamic_provider(&config, backend),
        )?;
        Ok(Self {
            runtime,
            provider,
            config,
            model_name,
        })
    }
}

impl IncrementalUpdate for EmbeddingUpdateEngine {
    fn update(&self, repo_path: &Path) -> Result<UpdateSummary, anyhow::Error> {
        self.runtime
            .block_on(vera_core::indexing::update_repository(
                repo_path,
                &self.provider,
                &self.config,
                &self.model_name,
            ))
    }
}

/// Handle to a running file watcher. Dropping it stops the watcher.
pub struct WatchHandle {
    _watcher: notify_debouncer_mini::Debouncer<notify::RecommendedWatcher>,
    /// Set to true when an update is in progress.
    updating: Arc<AtomicBool>,
}

impl WatchHandle {
    /// True if an incremental update is currently running.
    pub fn is_updating(&self) -> bool {
        self.updating.load(Ordering::Relaxed)
    }
}

/// Start watching a project directory for file changes.
///
/// When changes are detected (after debouncing), triggers an incremental
/// index update in a background thread. Returns a handle that keeps the
/// watcher alive; drop it to stop watching.
pub fn start_watching(repo_path: &Path) -> Result<WatchHandle, String> {
    start_watching_internal(repo_path, false)
}

/// Start watching with progress logs printed to stderr.
///
/// Intended for `vera watch` CLI mode where users expect visible activity.
pub fn start_watching_with_progress(repo_path: &Path) -> Result<WatchHandle, String> {
    start_watching_internal(repo_path, true)
}

fn start_watching_internal(repo_path: &Path, progress_logs: bool) -> Result<WatchHandle, String> {
    let backend = vera_core::config::resolve_backend(None);
    let mut config = crate::saved_config::load_saved_runtime_config();
    config.adjust_for_backend(backend);

    let indexing = config.indexing.clone();
    let build: EngineBuilder = Arc::new(move || {
        let engine = EmbeddingUpdateEngine::build(config.clone(), backend)?;
        Ok(Arc::new(engine) as Engine)
    });

    start_watching_with(
        repo_path,
        progress_logs,
        Duration::from_secs(DEBOUNCE_SECS),
        &indexing,
        build,
    )
}

fn start_watching_with(
    repo_path: &Path,
    progress_logs: bool,
    debounce: Duration,
    indexing: &IndexingConfig,
    build: EngineBuilder,
) -> Result<WatchHandle, String> {
    let repo_path = repo_path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve path: {e}"))?;

    let idx_dir = vera_core::indexing::index_dir(&repo_path);
    if !idx_dir.exists() {
        return Err("No index found. Run search_code first to auto-index.".to_string());
    }

    let exclusions = ExclusionMatcher::new(&repo_path, indexing)
        .map_err(|e| format!("Failed to build watcher exclusions: {e:#}"))?;

    // The index directory is filtered on its own, never through `exclusions`:
    // an update cycle writes into it, so a watcher that reacts to those writes
    // re-triggers itself. `no_default_excludes`, or a stored `default_excludes`
    // without `.vera`, would switch that guard off.
    let index_dir = idx_dir.clone();
    let tracked_paths = load_tracked_paths(&idx_dir, &repo_path);

    let updating = Arc::new(AtomicBool::new(false));
    let updating_clone = updating.clone();
    let repo_clone = repo_path.clone();
    let engine = Arc::new(SharedEngine {
        build,
        cached: Mutex::new(None),
    });

    let mut debouncer = new_debouncer(
        debounce,
        move |events: Result<Vec<notify_debouncer_mini::DebouncedEvent>, notify::Error>| {
            let events = match events {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = %e, "File watcher error");
                    return;
                }
            };

            // Ignore the index directory always, plus the directories indexing
            // would not walk anyway (target, node_modules, ...). A build
            // churning target/ used to start a full update cycle per debounce
            // window.
            let has_relevant_changes = events.iter().any(|e| {
                e.kind == DebouncedEventKind::Any
                    && !e.path.starts_with(&index_dir)
                    && is_relevant_change(&e.path, &exclusions, &tracked_paths)
            });

            if !has_relevant_changes {
                return;
            }

            // Skip if already updating.
            if updating_clone
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                .is_err()
            {
                debug!("Skipping auto-update: previous update still running");
                if progress_logs {
                    eprintln!(
                        "[watch] update already running, changes will be picked up next cycle"
                    );
                }
                return;
            }

            if progress_logs {
                eprintln!("[watch] file changes detected, starting incremental update");
            }

            let repo = repo_clone.clone();
            let flag = updating_clone.clone();
            let engine = Arc::clone(&engine);

            std::thread::spawn(move || {
                run_incremental_update(&engine, &repo, &flag, progress_logs);
            });
        },
    )
    .map_err(|e| format!("Failed to create file watcher: {e}"))?;

    debouncer
        .watcher()
        .watch(&repo_path, notify::RecursiveMode::Recursive)
        .map_err(|e| watch_failure_message(&repo_path, &e))?;

    info!(path = %repo_path.display(), "Started file watcher for auto-indexing");

    Ok(WatchHandle {
        _watcher: debouncer,
        updating,
    })
}

fn load_tracked_paths(idx_dir: &Path, repo_path: &Path) -> HashSet<std::path::PathBuf> {
    let metadata_path = idx_dir.join("metadata.db");
    let Ok(metadata) = MetadataStore::open(&metadata_path) else {
        return HashSet::new();
    };
    metadata
        .tracked_files()
        .unwrap_or_default()
        .into_iter()
        .map(|path| repo_path.join(path))
        .collect()
}

/// Keep excluded descendants quiet, but let a path that was indexed before it
/// became excluded reach the incremental scan so its old rows can be purged.
fn is_relevant_change(
    path: &Path,
    exclusions: &ExclusionMatcher,
    tracked_paths: &HashSet<std::path::PathBuf>,
) -> bool {
    !exclusions.is_excluded(path) || tracked_paths.contains(path)
}

/// Describe a failed `watch()` call, with the remedy for the one failure mode
/// that is both common and fixable: the per-user OS watch limit, which a
/// recursive watch over a repository with build output can exhaust.
fn watch_failure_message(repo_path: &Path, error: &notify::Error) -> String {
    if matches!(error.kind, notify::ErrorKind::MaxFilesWatch) {
        return format!(
            "Failed to watch {}: the OS file watch limit was reached, so the index will not \
             auto-update. Raise it (Linux: sysctl fs.inotify.max_user_watches) or exclude build \
             output directories from the repository.",
            repo_path.display()
        );
    }
    format!("Failed to watch directory: {error}")
}

/// Run an incremental update, resetting the flag when done.
fn run_incremental_update(
    engine: &SharedEngine,
    repo_path: &Path,
    updating: &AtomicBool,
    progress_logs: bool,
) {
    debug!(path = %repo_path.display(), "Auto-update triggered by file changes");

    let result = engine.get().and_then(|engine| engine.update(repo_path));

    match result {
        Ok(summary) => {
            let changed = summary.files_modified + summary.files_added + summary.files_deleted;
            if changed > 0 {
                info!(
                    modified = summary.files_modified,
                    added = summary.files_added,
                    deleted = summary.files_deleted,
                    "Auto-update complete"
                );
                if progress_logs {
                    eprintln!(
                        "[watch] update complete: {} modified, {} added, {} deleted",
                        summary.files_modified, summary.files_added, summary.files_deleted
                    );
                }
            } else {
                debug!("Auto-update: no changes detected");
                if progress_logs {
                    eprintln!("[watch] no indexable changes detected");
                }
            }
        }
        Err(e) => {
            warn!(error = %e, "Auto-update failed");
            if progress_logs {
                eprintln!("[watch] update failed: {e}");
            }
        }
    }

    updating.store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::atomic::AtomicUsize;

    use vera_core::config::VeraConfig;
    use vera_core::embedding::{EmbeddingError, EmbeddingProvider};
    use vera_core::indexing::{index_repository, update_repository};
    use vera_core::storage::metadata::MetadataStore;

    const TEST_DEBOUNCE: Duration = Duration::from_millis(300);

    /// Stands in for the real engine: records that it was built, and how many
    /// update cycles ran through it.
    struct CountingEngine {
        updates: Arc<AtomicUsize>,
    }

    impl IncrementalUpdate for CountingEngine {
        fn update(&self, _repo_path: &Path) -> Result<UpdateSummary, anyhow::Error> {
            self.updates.fetch_add(1, Ordering::SeqCst);
            Ok(UpdateSummary {
                files_modified: 0,
                files_added: 0,
                files_deleted: 0,
                files_unchanged: 0,
                files_with_tree_sitter_errors: 0,
                files_using_tier0_fallback: 0,
                parse_errors: Vec::new(),
                files_deferred: 0,
                total_chunks: 0,
                elapsed_secs: 0.0,
            })
        }
    }

    struct TestProvider;

    impl EmbeddingProvider for TestProvider {
        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }

        fn expected_dim(&self) -> Option<usize> {
            Some(8)
        }
    }

    struct IndexingEngine {
        runtime: tokio::runtime::Runtime,
        config: VeraConfig,
    }

    impl IncrementalUpdate for IndexingEngine {
        fn update(&self, repo_path: &Path) -> Result<UpdateSummary, anyhow::Error> {
            self.runtime.block_on(update_repository(
                repo_path,
                &TestProvider,
                &self.config,
                "test-model",
            ))
        }
    }

    fn indexing_builder(config: VeraConfig) -> EngineBuilder {
        Arc::new(move || {
            Ok(Arc::new(IndexingEngine {
                runtime: tokio::runtime::Runtime::new()?,
                config: config.clone(),
            }) as Engine)
        })
    }

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        tokio::runtime::Runtime::new()
            .expect("test runtime")
            .block_on(future)
    }

    fn counting_builder(builds: Arc<AtomicUsize>, updates: Arc<AtomicUsize>) -> EngineBuilder {
        Arc::new(move || {
            builds.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(CountingEngine {
                updates: Arc::clone(&updates),
            }) as Engine)
        })
    }

    /// A repository laid out with the directories the watcher has to tell apart.
    fn repo_fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join(".vera")).expect("index dir");
        std::fs::create_dir_all(dir.path().join("src")).expect("src dir");
        std::fs::create_dir_all(dir.path().join("target/debug")).expect("target dir");
        dir
    }

    fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if predicate() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        predicate()
    }

    #[test]
    fn engine_is_built_once_across_debounce_cycles() {
        let repo = repo_fixture();
        let builds = Arc::new(AtomicUsize::new(0));
        let updates = Arc::new(AtomicUsize::new(0));

        let _handle = start_watching_with(
            repo.path(),
            false,
            TEST_DEBOUNCE,
            &IndexingConfig::default(),
            counting_builder(Arc::clone(&builds), Arc::clone(&updates)),
        )
        .expect("watcher starts");

        std::fs::write(repo.path().join("src/first.rs"), "fn first() {}").expect("write");
        assert!(
            wait_until(Duration::from_secs(10), || updates.load(Ordering::SeqCst)
                >= 1),
            "first change should trigger an update cycle"
        );

        std::fs::write(repo.path().join("src/second.rs"), "fn second() {}").expect("write");
        assert!(
            wait_until(Duration::from_secs(10), || updates.load(Ordering::SeqCst)
                >= 2),
            "second change should trigger another update cycle"
        );

        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "the engine must be built once and reused, not rebuilt per cycle"
        );
    }

    #[test]
    fn changes_under_excluded_directories_do_not_trigger_updates() {
        let repo = repo_fixture();
        let builds = Arc::new(AtomicUsize::new(0));
        let updates = Arc::new(AtomicUsize::new(0));

        let _handle = start_watching_with(
            repo.path(),
            false,
            TEST_DEBOUNCE,
            &IndexingConfig::default(),
            counting_builder(Arc::clone(&builds), Arc::clone(&updates)),
        )
        .expect("watcher starts");

        // Presence first: a watcher that never fires would pass the absence
        // assertion below on its own.
        std::fs::write(repo.path().join("src/real.rs"), "fn real() {}").expect("write");
        assert!(
            wait_until(Duration::from_secs(10), || updates.load(Ordering::SeqCst)
                == 1),
            "an indexable change must trigger exactly one update cycle"
        );

        for i in 0..5 {
            std::fs::write(
                repo.path().join(format!("target/debug/artifact{i}.o")),
                format!("build output {i}"),
            )
            .expect("write");
        }
        std::thread::sleep(TEST_DEBOUNCE * 6);

        assert_eq!(
            updates.load(Ordering::SeqCst),
            1,
            "build output under target/ must not trigger an update cycle"
        );
    }

    /// An update cycle writes into `.vera`, so a watcher that reacts to those
    /// writes re-triggers itself forever. The index directory therefore cannot
    /// be filtered through the configurable exclusions: `no_default_excludes`
    /// (and a stored `default_excludes` that omits `.vera`) would switch the
    /// guard off.
    #[test]
    fn index_writes_never_trigger_updates_even_without_default_excludes() {
        let repo = repo_fixture();
        let builds = Arc::new(AtomicUsize::new(0));
        let updates = Arc::new(AtomicUsize::new(0));
        let indexing = IndexingConfig {
            no_default_excludes: true,
            ..IndexingConfig::default()
        };

        let _handle = start_watching_with(
            repo.path(),
            false,
            TEST_DEBOUNCE,
            &indexing,
            counting_builder(Arc::clone(&builds), Arc::clone(&updates)),
        )
        .expect("watcher starts");

        // Presence first: with the default exclusions off, an ordinary source
        // change must still be seen, or the absence assertion below is vacuous.
        std::fs::write(repo.path().join("src/real.rs"), "fn real() {}").expect("write");
        assert!(
            wait_until(Duration::from_secs(10), || updates.load(Ordering::SeqCst)
                == 1),
            "an indexable change must trigger exactly one update cycle"
        );

        for i in 0..5 {
            std::fs::write(
                repo.path().join(format!(".vera/chunk{i}.db")),
                format!("index write {i}"),
            )
            .expect("write");
        }
        std::thread::sleep(TEST_DEBOUNCE * 6);

        assert_eq!(
            updates.load(Ordering::SeqCst),
            1,
            "writes into the index directory must never trigger an update cycle"
        );
    }

    #[test]
    fn replacing_indexed_file_with_excluded_directory_purges_chunks() {
        let repo = tempfile::tempdir().expect("tempdir");
        let target = repo.path().join("target");
        std::fs::write(&target, "fn stale_target_symbol() {}\n").expect("write source");
        let config = VeraConfig::default();

        block_on(index_repository(
            repo.path(),
            &TestProvider,
            &config,
            "test-model",
        ))
        .expect("initial index");

        let index_dir = vera_core::indexing::index_dir(&repo.path().canonicalize().unwrap());
        let metadata = MetadataStore::open(&index_dir.join("metadata.db")).expect("metadata");
        assert!(
            !metadata
                .get_chunks_by_file("target")
                .expect("initial chunks")
                .is_empty(),
            "the regular file must be indexed before the transition"
        );

        let indexing = config.indexing.clone();
        let _handle = start_watching_with(
            repo.path(),
            false,
            TEST_DEBOUNCE,
            &indexing,
            indexing_builder(config),
        )
        .expect("watcher starts");

        std::fs::remove_file(&target).expect("remove source");
        std::fs::create_dir(&target).expect("replace with excluded directory");

        assert!(
            wait_until(Duration::from_secs(10), || metadata
                .get_chunks_by_file("target")
                .expect("chunks after update")
                .is_empty()),
            "the directory transition must trigger an update that purges stale chunks"
        );
    }
}
