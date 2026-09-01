//! Hybrid retrieval engine combining BM25 and vector search via RRF fusion.
//!
//! Runs both BM25 keyword search and vector similarity search in parallel,
//! then merges the results using Reciprocal Rank Fusion (RRF). Items appearing
//! in both result sets rank higher than single-source results.
//!
//! RRF score: `score = sum(1 / (k + rank_i))` where `k` is a constant
//! (typically 60) and `rank_i` is the 1-based rank of the item in each
//! source result list.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::embedding::EmbeddingProvider;
use crate::retrieval::bm25::search_bm25_with_stores_and_filters;
use crate::retrieval::graph_augmentation::augment_pool;
use crate::retrieval::query_classifier::{QueryType, classify_query, params_for_query_type};
use crate::retrieval::query_utils::result_key;
use crate::retrieval::ranking::is_path_weighted_query;
use crate::retrieval::reranker::{Reranker, rerank_results};
use crate::retrieval::vector::{
    VectorSearchError, search_vector_with_cached_stores_timed, search_vector_with_stores_timed,
};
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::{MAX_KNN_K, VectorStore};
use crate::types::{SearchFilters, SearchResult};

fn emit_vec0_truncation_warning(
    is_flat: bool,
    index_count: usize,
    clamped_requested: usize,
    fetched: usize,
    filters_empty: bool,
    filter_has_matches: bool,
) {
    if !should_emit_vec0_truncation_warning(
        is_flat,
        index_count,
        clamped_requested,
        fetched,
        filters_empty,
        filter_has_matches,
    ) {
        return;
    }
    if filters_empty {
        warn!(
            backend = "vec0",
            index_count = index_count,
            requested = clamped_requested,
            cap = MAX_KNN_K,
            "vec0 vector search truncated at sqlite-vec KNN cap (4096); results may be incomplete. Use the default flat vector scan (unset VERA_VECTOR_SCAN) for complete results"
        );
    } else {
        warn!(
            backend = "vec0",
            index_count = index_count,
            requested = clamped_requested,
            cap = MAX_KNN_K,
            "vec0 vector search truncated at sqlite-vec KNN cap (4096) with active filters; results may be incomplete. Use the default flat vector scan (unset VERA_VECTOR_SCAN) for complete results"
        );
    }
}

/// Whether a vec0 truncation warning should be emitted.
///
/// Pinned decision matrix (VAL-FILTER-008):
/// - Only vec0 (flat never warns)
/// - Only when index > cap and the request was actually clamped at the cap
/// - For filtered queries, only when the filter matches at least one chunk index-wide
///   (true negatives stay quiet, including scope/include_generated filtered negatives)
/// - For unfiltered queries, the truncated case indicates plausible loss (deep limit)
/// - Below-cap indexes never warn (pool clamped to index size first)
pub(crate) fn should_emit_vec0_truncation_warning(
    is_flat: bool,
    index_count: usize,
    clamped_requested: usize,
    fetched: usize,
    filters_empty: bool,
    filter_has_matches: bool,
) -> bool {
    if is_flat {
        return false;
    }
    if index_count <= MAX_KNN_K {
        return false;
    }
    let truncated = clamped_requested > MAX_KNN_K && fetched == MAX_KNN_K;
    if !truncated {
        return false;
    }
    if filters_empty {
        return true;
    }
    filter_has_matches
}

/// Open search stores reused for the active index directory.
pub(crate) struct SearchStores {
    pub(crate) bm25: Bm25Index,
    pub(crate) bm25_metadata: Mutex<MetadataStore>,
    pub(crate) vector_metadata: Mutex<MetadataStore>,
    vector_path: PathBuf,
    vector: Mutex<Option<CachedVectorStore>>,
    metadata_path: PathBuf,
    indexed_files: Mutex<Option<CachedIndexedFiles>>,
    cached_index_meta: Mutex<Option<CachedIndexMeta>>,
    open_stamp: MetadataDbStamp,
}

struct CachedIndexedFiles {
    stamp: MetadataDbStamp,
    files: Arc<Vec<String>>,
}

struct CachedIndexMeta {
    stamp: MetadataDbStamp,
    model_name: Option<String>,
    embedding_dim: Option<String>,
    document_prefix: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MetadataDbStamp {
    db_len: Option<u64>,
    db_mtime: Option<SystemTime>,
    wal_len: Option<u64>,
    wal_mtime: Option<SystemTime>,
}

fn metadata_db_stamp(db_path: &Path) -> MetadataDbStamp {
    // WAL-mode commits append to the -wal file without touching the main
    // database file until a checkpoint, so both files must contribute.
    let wal_path = db_path.with_file_name(format!(
        "{}-wal",
        db_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    let stats = |path: &Path| {
        std::fs::metadata(path)
            .ok()
            .map(|metadata| (metadata.len(), metadata.modified().ok()))
    };
    let (db_len, db_mtime) = stats(db_path).map_or((None, None), |(len, mtime)| (Some(len), mtime));
    let (wal_len, wal_mtime) =
        stats(&wal_path).map_or((None, None), |(len, mtime)| (Some(len), mtime));
    MetadataDbStamp {
        db_len,
        db_mtime,
        wal_len,
        wal_mtime,
    }
}

struct CachedVectorStore {
    dim: usize,
    stamp: VectorStoreStamp,
    store: Arc<Mutex<VectorStore>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct VectorStoreStamp {
    db_len: Option<u64>,
    db_mtime: Option<SystemTime>,
    manifest_mtime: Option<SystemTime>,
    manifest_generation: Option<u64>,
}

fn vector_store_stamp(vector_path: &Path) -> VectorStoreStamp {
    let db_metadata = std::fs::metadata(vector_path).ok();
    let manifest_path = vector_path.with_file_name("vectors.manifest");
    let manifest_mtime = std::fs::metadata(&manifest_path)
        .and_then(|metadata| metadata.modified())
        .ok();
    let manifest_generation = std::fs::read(&manifest_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|manifest| {
            manifest
                .get("generation")
                .and_then(serde_json::Value::as_u64)
        });
    VectorStoreStamp {
        db_len: db_metadata.as_ref().map(std::fs::Metadata::len),
        db_mtime: db_metadata.and_then(|metadata| metadata.modified().ok()),
        manifest_mtime,
        manifest_generation,
    }
}

impl SearchStores {
    pub(crate) fn open(index_dir: &Path) -> Result<Self> {
        let bm25 = Bm25Index::open(&index_dir.join("bm25"))
            .context("failed to open BM25 index for search")?;
        let metadata_path = index_dir.join("metadata.db");
        let bm25_metadata = MetadataStore::open(&metadata_path)
            .context("failed to open metadata store for search")?;
        let vector_metadata = MetadataStore::open(&metadata_path)
            .context("failed to open vector metadata store for search")?;
        let open_stamp = metadata_db_stamp(&metadata_path);
        Ok(Self {
            bm25,
            bm25_metadata: Mutex::new(bm25_metadata),
            vector_metadata: Mutex::new(vector_metadata),
            vector_path: index_dir.join("vectors.db"),
            vector: Mutex::new(None),
            metadata_path,
            indexed_files: Mutex::new(None),
            cached_index_meta: Mutex::new(None),
            open_stamp,
        })
    }

    /// Indexed file list, cached against the metadata database stamp.
    ///
    /// `MetadataStore::indexed_files()` runs a DISTINCT scan over every chunk
    /// row, which costs tens of milliseconds on large indexes. Searches repeat
    /// it per query through exact-match augmentation, so the list is cached
    /// and refreshed only when metadata.db (or its WAL) changes on disk.
    ///
    /// Reads the stamp before querying so a concurrent commit invalidates the
    /// entry instead of certifying a stale list. A fresh connection is used on
    /// misses so this never interacts with the `bm25_metadata` lock.
    pub(crate) fn indexed_files(&self) -> Result<Arc<Vec<String>>> {
        let stamp = metadata_db_stamp(&self.metadata_path);
        {
            let cached = self
                .indexed_files
                .lock()
                .map_err(|_| anyhow::anyhow!("indexed files cache lock poisoned"))?;
            if let Some(cached) = cached.as_ref()
                && cached.stamp == stamp
            {
                return Ok(Arc::clone(&cached.files));
            }
        }
        let files = Arc::new(
            MetadataStore::open(&self.metadata_path)
                .context("failed to open metadata store for file list")?
                .indexed_files()?,
        );
        let mut cached = self
            .indexed_files
            .lock()
            .map_err(|_| anyhow::anyhow!("indexed files cache lock poisoned"))?;
        *cached = Some(CachedIndexedFiles {
            stamp,
            files: Arc::clone(&files),
        });
        Ok(files)
    }

    /// Index metadata (model_name, embedding_dim, document_prefix) cached
    /// against the metadata database stamp.
    ///
    /// Profiling (`docs/197-profiling.md`): `SearchContext::search` performs
    /// three `get_index_meta` SQLite reads per query (~0.45 ms per query,
    /// 4-5% of warm p50). This cache serves warm queries from memory and
    /// refreshes only when `metadata.db` or its WAL changes, checked via the
    /// same `MetadataDbStamp` used for `indexed_files` to guarantee staleness
    /// safety.
    ///
    /// Stamp is read before checking the cache so a concurrent commit
    /// invalidates the entry rather than certifying stale metadata. On miss a
    /// fresh `MetadataStore` connection is used to avoid coupling to the
    /// `bm25_metadata` lock.
    pub(crate) fn cached_index_meta(
        &self,
    ) -> Result<(Option<String>, Option<String>, Option<String>)> {
        let stamp = metadata_db_stamp(&self.metadata_path);
        {
            let cached = self
                .cached_index_meta
                .lock()
                .map_err(|_| anyhow::anyhow!("index meta cache lock poisoned"))?;
            if let Some(cached) = cached.as_ref()
                && cached.stamp == stamp
            {
                return Ok((
                    cached.model_name.clone(),
                    cached.embedding_dim.clone(),
                    cached.document_prefix.clone(),
                ));
            }
        }
        let store = MetadataStore::open(&self.metadata_path)
            .context("failed to open metadata store for index meta")?;
        let model_name = store.get_index_meta("model_name")?;
        let embedding_dim = store.get_index_meta("embedding_dim")?;
        let document_prefix = store.get_index_meta("document_prefix")?;
        let mut cached = self
            .cached_index_meta
            .lock()
            .map_err(|_| anyhow::anyhow!("index meta cache lock poisoned"))?;
        *cached = Some(CachedIndexMeta {
            stamp,
            model_name: model_name.clone(),
            embedding_dim: embedding_dim.clone(),
            document_prefix: document_prefix.clone(),
        });
        Ok((model_name, embedding_dim, document_prefix))
    }

    pub(crate) fn is_open_stamp_current(&self) -> bool {
        metadata_db_stamp(&self.metadata_path) == self.open_stamp
    }

    pub(crate) fn vector_store(&self, dim: usize) -> Result<Arc<Mutex<VectorStore>>> {
        let mut cached = self
            .vector
            .lock()
            .map_err(|_| anyhow::anyhow!("vector store cache lock poisoned"))?;
        let stamp = vector_store_stamp(&self.vector_path);
        if let Some(cached_store) = cached.as_ref()
            && cached_store.dim == dim
            && cached_store.stamp == stamp
        {
            return Ok(Arc::clone(&cached_store.store));
        }

        let store = Arc::new(Mutex::new(
            VectorStore::open(&self.vector_path, dim).context("failed to open vector store")?,
        ));
        let stamp = vector_store_stamp(&self.vector_path);
        *cached = Some(CachedVectorStore {
            dim,
            stamp,
            store: Arc::clone(&store),
        });
        Ok(store)
    }
}

/// Errors specific to hybrid search.
#[derive(Debug, thiserror::Error)]
pub enum HybridSearchError {
    /// Both BM25 and vector search failed.
    #[error("both BM25 and vector search failed: bm25={bm25_error}, vector={vector_error}")]
    BothFailed {
        bm25_error: String,
        vector_error: String,
    },

    /// The rerank was cancelled via the cancellation token.
    #[error("rerank cancelled")]
    Cancelled,

    /// Storage or pipeline error.
    #[error("{0}")]
    PipelineError(#[from] anyhow::Error),
}

/// Compute the number of vector candidates to fetch for a given limit and
/// query type multiplier. Ensures at least 50 candidates for any limit.
pub fn compute_vector_candidates(limit: usize, multiplier: usize) -> usize {
    limit.saturating_mul(multiplier).max(50)
}

fn compute_bm25_candidates(query: &str, limit: usize) -> usize {
    let query_type = classify_query(query);
    let token_count = query.split_whitespace().count();

    if is_path_weighted_query(query) {
        return limit.saturating_mul(5).max(100);
    }
    if query_type == QueryType::NaturalLanguage {
        return limit.saturating_mul(4).max(limit + 40);
    }
    if token_count <= 2 {
        return limit.saturating_mul(4).max(80);
    }

    limit.saturating_mul(3).max(limit + 20)
}

/// Per-stage timing data from hybrid search.
#[derive(Debug, Default)]
pub struct HybridTimings {
    pub embedding: Option<Duration>,
    pub bm25: Option<Duration>,
    pub vector: Option<Duration>,
    pub fusion: Option<Duration>,
    pub reranking: Option<Duration>,
}

/// Perform hybrid search combining BM25 and vector retrieval via RRF fusion.
///
/// Runs BM25 and vector search concurrently, then merges the results using
/// Reciprocal Rank Fusion (RRF). BM25 runs in a blocking task with its own
/// database connections while vector search (embedding + nearest-neighbor)
/// runs on the async runtime. If vector search fails (e.g., embedding API
/// unavailable), falls back to BM25-only results with a warning.
#[allow(clippy::too_many_arguments)]
pub async fn search_hybrid(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    vector_candidates: usize,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    search_hybrid_inner(
        index_dir,
        provider,
        bm25_query,
        vector_query,
        filters,
        limit,
        rrf_k,
        stored_dim,
        vector_candidates,
        None,
    )
    .await
}

/// Perform hybrid search using stores retained by a reusable search context.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn search_hybrid_with_stores(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    vector_candidates: usize,
    stores: Arc<SearchStores>,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    search_hybrid_inner(
        index_dir,
        provider,
        bm25_query,
        vector_query,
        filters,
        limit,
        rrf_k,
        stored_dim,
        vector_candidates,
        Some(stores),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn search_hybrid_inner(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    vector_candidates: usize,
    stores: Option<Arc<SearchStores>>,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    let query_type = classify_query(bm25_query);
    let query_params = params_for_query_type(query_type);
    let bm25_candidates = compute_bm25_candidates(bm25_query, limit);
    let mut timings = HybridTimings::default();

    // Spawn BM25 search in a blocking task. Reusable search contexts retain
    // the Tantivy reader and SQLite connection; direct callers keep the
    // original open-per-call behavior.
    let bm25_dir = index_dir.join("bm25");
    let metadata_path = index_dir.join("metadata.db");
    let bm25_query = bm25_query.to_string();
    let bm25_filters = filters.clone();
    let stores_for_bm25 = stores.clone();
    let bm25_handle = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let result = match stores_for_bm25 {
            Some(stores) => stores
                .bm25_metadata
                .lock()
                .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))
                .and_then(|store| {
                    search_bm25_with_stores_and_filters(
                        &stores.bm25,
                        &store,
                        &bm25_query,
                        &bm25_filters,
                        bm25_candidates,
                    )
                }),
            None => Bm25Index::open(&bm25_dir)
                .context("failed to open BM25 index for search")
                .and_then(|index| {
                    let store = MetadataStore::open(&metadata_path)
                        .context("failed to open metadata store for BM25 search")?;
                    search_bm25_with_stores_and_filters(
                        &index,
                        &store,
                        &bm25_query,
                        &bm25_filters,
                        bm25_candidates,
                    )
                }),
        };
        (result, start.elapsed())
    });

    // Run vector search concurrently on the async runtime.
    // Flat vs vec0 handling: flat honors the requested depth up to the real index size,
    // vec0 is bounded by MAX_KNN_K. For filtered queries on the flat path we fetch the
    // entire index so a low-ranking island is reachable independent of rank position.
    // The pool sizing never requests more candidates than the index holds, keeping
    // below-cap indexes quiet and preventing spurious truncation warnings.
    // Regression bisect (issue #197 correction comment 5485896644): re-measured
    // 072c725 mean p50 7.879 ms / p95 60.880 ms (same-host, 3 runs) vs fc20352
    // 11.694 / 124.825 (+48%/+105%, nDCG parity). LRU thrashing hypothesis
    // rejected by deterministic task-order simulation (1251 tasks, 63 repos,
    // 62 switches, hit rate 95% at capacities 4/63 — grouped order, not
    // round-robin). Dominant warm regression is per-query vector-store
    // reopen: this function previously opened the vector store twice per
    // query (once for is_flat/index_count sizing, once for the search), each
    // doing a manifest JSON read + stamp check. Deduplicate to a single
    // open per query to save the ~0.02–0.05 ms manifest stall and halve lock
    // contention on large indexes.
    let vector_start = Instant::now();

    // Deduplicate vector-store open: single successful open supplies both
    // backend detection (is_flat, index_count) and the later vector search.
    // This halves manifest reads + Mutex contention per warm query while
    // preserving stamp-guarded staleness (the single open already checked).
    // Retain the original error as context so failures remain debuggable
    // (fallback to BM25 is unchanged; the error is surfaced via the warn! log
    // and the BothFailed variant).
    let mut vector_store_error: Option<anyhow::Error> = None;
    let vector_store_cached: Option<Arc<Mutex<VectorStore>>> = match &stores {
        Some(s) => match s.vector_store(stored_dim) {
            Ok(v) => Some(v),
            Err(e) => {
                vector_store_error = Some(e);
                None
            }
        },
        None => None,
    };
    let vector_store_direct: Option<VectorStore> = if stores.is_none() {
        match VectorStore::open(&index_dir.join("vectors.db"), stored_dim) {
            Ok(v) => Some(v),
            Err(e) => {
                vector_store_error = Some(e);
                None
            }
        }
    } else {
        None
    };
    let (is_flat, index_count) = match (&vector_store_cached, &vector_store_direct) {
        (Some(vs), _) => {
            let guard = vs.lock().ok();
            guard
                .map(|g| (g.is_flat(), g.count().unwrap_or(0) as usize))
                .unwrap_or((true, usize::MAX))
        }
        (None, Some(vs)) => (vs.is_flat(), vs.count().unwrap_or(0) as usize),
        _ => (true, usize::MAX),
    };

    let vector_fetch = if filters.is_empty() {
        // Unfiltered: use the query-type-aware candidate count, clamped to the index.
        let base = vector_candidates.min(index_count);
        if is_flat { base } else { base.min(MAX_KNN_K) }
    } else if is_flat {
        // Flat filtered: fetch the whole index so post-filtering can reach any chunk.
        index_count
    } else {
        // vec0 filtered: fetch as many as the backend allows (cap) to maximize recall.
        index_count.min(MAX_KNN_K)
    };

    // Use the shared candidate_pool helper so the diagnostic cap arithmetic cannot drift from the
    // vector layer's actual fetching logic.
    let (clamped_requested, fetched_for_diag) =
        crate::retrieval::vector::candidate_pool(vector_fetch, index_count, is_flat);

    let vector_outcome = match (stores, vector_store_cached, vector_store_direct) {
        (Some(stores), Some(vector_store), _) => {
            match search_vector_with_cached_stores_timed(
                vector_store.as_ref(),
                &stores.vector_metadata,
                provider,
                vector_query,
                vector_fetch,
            )
            .await
            {
                Ok((mut results, embed_elapsed)) => {
                    if !filters.is_empty() {
                        results.retain(|result| filters.matches(result));
                    }
                    let has_matches = if filters.is_empty() {
                        false
                    } else {
                        stores
                            .vector_metadata
                            .lock()
                            .ok()
                            .and_then(|m| m.has_filter_matches(filters).ok())
                            .unwrap_or(false)
                    };
                    emit_vec0_truncation_warning(
                        is_flat,
                        index_count,
                        clamped_requested,
                        fetched_for_diag,
                        filters.is_empty(),
                        has_matches,
                    );
                    Ok((results, embed_elapsed))
                }
                Err(err) => Err(err),
            }
        }
        (Some(_), None, _) => Err(VectorSearchError::StorageError(
            vector_store_error
                .map(|e| e.context("failed to open vector store"))
                .unwrap_or_else(|| anyhow::anyhow!("failed to open vector store")),
        )),
        (None, _, Some(vector_store)) => {
            let vector_metadata_result = MetadataStore::open(&index_dir.join("metadata.db"))
                .context("failed to open metadata store");
            match vector_metadata_result {
                Ok(vector_metadata) => {
                    match search_vector_with_stores_timed(
                        &vector_store,
                        &vector_metadata,
                        provider,
                        vector_query,
                        vector_fetch,
                    )
                    .await
                    {
                        Ok((mut results, embed_elapsed)) => {
                            if !filters.is_empty() {
                                results.retain(|result| filters.matches(result));
                            }
                            let has_matches = if filters.is_empty() {
                                false
                            } else {
                                vector_metadata.has_filter_matches(filters).unwrap_or(false)
                            };
                            emit_vec0_truncation_warning(
                                is_flat,
                                index_count,
                                clamped_requested,
                                fetched_for_diag,
                                filters.is_empty(),
                                has_matches,
                            );
                            Ok((results, embed_elapsed))
                        }
                        Err(err) => Err(err),
                    }
                }
                Err(err) => Err(VectorSearchError::StorageError(err)),
            }
        }
        (None, _, None) => Err(VectorSearchError::StorageError(
            vector_store_error
                .map(|e| e.context("failed to open vector store"))
                .unwrap_or_else(|| anyhow::anyhow!("failed to open vector store")),
        )),
    };
    let embed_elapsed = vector_outcome
        .as_ref()
        .ok()
        .map(|(_, embed_elapsed)| *embed_elapsed);
    charge_vector_span(&mut timings, vector_start.elapsed(), embed_elapsed);
    let vector_results = vector_outcome.map(|(results, _)| results);

    // Await the BM25 result (should already be done or nearly done).
    let (bm25_results, bm25_elapsed) = bm25_handle.await.map_err(|e| {
        HybridSearchError::PipelineError(anyhow::anyhow!("BM25 task panicked: {e}"))
    })?;
    timings.bm25 = Some(bm25_elapsed);

    match (bm25_results, vector_results) {
        (Ok(bm25), Ok(vector)) => {
            debug!(
                bm25_count = bm25.len(),
                vector_count = vector.len(),
                query_type = ?query_type,
                bm25_weight = query_params.bm25_weight,
                vector_weight = query_params.vector_weight,
                "merging BM25 and vector results via weighted RRF"
            );
            let fusion_start = Instant::now();
            let fused = fuse_rrf_weighted(
                &bm25,
                &vector,
                rrf_k,
                limit,
                query_params.bm25_weight,
                query_params.vector_weight,
            );
            timings.fusion = Some(fusion_start.elapsed());
            Ok((fused, timings))
        }
        (Ok(bm25), Err(vec_err)) => {
            warn!(
                error = %vec_err,
                "vector search failed, falling back to BM25-only results"
            );
            let mut results = bm25;
            results.truncate(limit);
            Ok((results, timings))
        }
        (Err(bm25_err), Ok(vector)) => {
            warn!(
                error = %bm25_err,
                "BM25 search failed, falling back to vector-only results"
            );
            let mut results = vector;
            results.truncate(limit);
            Ok((results, timings))
        }
        (Err(bm25_err), Err(vec_err)) => Err(HybridSearchError::BothFailed {
            bm25_error: format!("{bm25_err:#}"),
            vector_error: format!("{vec_err:#}"),
        }),
    }
}

/// Perform hybrid search with cross-encoder reranking.
///
/// Runs the full hybrid pipeline (BM25 + vector → RRF fusion), then
/// sends the top candidates to a cross-encoder reranker for more accurate
/// relevance scoring.
///
/// **Graceful degradation:**
/// - If the reranker API is unavailable (timeout, 5xx, connection error),
///   returns unreranked results with a warning logged to stderr.
/// - If the embedding API is unavailable, falls back to BM25-only results
///   (handled by the inner `search_hybrid` call).
///
/// # Arguments
/// - `index_dir` — Path to the `.vera` index directory
/// - `provider` — Embedding provider for vector search
/// - `reranker` — Reranker for result refinement
/// - `bm25_query` — Raw query text for the BM25 side (never intent-prefixed)
/// - `vector_query` — Query text for embedding and reranking (may carry intent)
/// - `filters` — Post-retrieval filters applied before fusion and reranking
/// - `fetch_limit` — Maximum number of candidates to retain through fusion
/// - `result_limit` — Maximum number of results to return to the user
/// - `rrf_k` — RRF constant (typically 60.0)
/// - `stored_dim` — Dimensionality of stored vectors
/// - `rerank_candidates` — Number of candidates to send to the reranker
/// - `vector_candidates` — Number of vector candidates to fetch (query-type-aware)
#[allow(clippy::too_many_arguments)]
pub async fn search_hybrid_reranked(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    reranker: &impl Reranker,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    rerank_candidates: usize,
    vector_candidates: usize,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    search_hybrid_reranked_with_augmentation(
        index_dir,
        provider,
        reranker,
        bm25_query,
        vector_query,
        filters,
        fetch_limit,
        result_limit,
        rrf_k,
        stored_dim,
        rerank_candidates,
        vector_candidates,
        false,
    )
    .await
}

/// Perform reranked hybrid search using stores retained by a reusable context.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn search_hybrid_reranked_with_stores(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    reranker: &impl Reranker,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    rerank_candidates: usize,
    vector_candidates: usize,
    graph_augmentation_enabled: bool,
    stores: Arc<SearchStores>,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    search_hybrid_reranked_inner(
        index_dir,
        provider,
        reranker,
        bm25_query,
        vector_query,
        filters,
        fetch_limit,
        result_limit,
        rrf_k,
        stored_dim,
        rerank_candidates,
        vector_candidates,
        graph_augmentation_enabled,
        Some(stores),
    )
    .await
}

/// Perform hybrid search with optional experimental graph augmentation.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn search_hybrid_reranked_with_augmentation(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    reranker: &impl Reranker,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    rerank_candidates: usize,
    vector_candidates: usize,
    graph_augmentation_enabled: bool,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    search_hybrid_reranked_inner(
        index_dir,
        provider,
        reranker,
        bm25_query,
        vector_query,
        filters,
        fetch_limit,
        result_limit,
        rrf_k,
        stored_dim,
        rerank_candidates,
        vector_candidates,
        graph_augmentation_enabled,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn search_hybrid_reranked_inner(
    index_dir: &Path,
    provider: &impl EmbeddingProvider,
    reranker: &impl Reranker,
    bm25_query: &str,
    vector_query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    rrf_k: f64,
    stored_dim: usize,
    rerank_candidates: usize,
    vector_candidates: usize,
    graph_augmentation_enabled: bool,
    stores: Option<Arc<SearchStores>>,
) -> Result<(Vec<SearchResult>, HybridTimings), HybridSearchError> {
    let fusion_limit = rerank_candidates.max(fetch_limit);

    let (mut hybrid_results, mut timings) = match stores {
        Some(stores) => {
            search_hybrid_with_stores(
                index_dir,
                provider,
                bm25_query,
                vector_query,
                filters,
                fusion_limit,
                rrf_k,
                stored_dim,
                vector_candidates,
                stores,
            )
            .await?
        }
        None => {
            search_hybrid(
                index_dir,
                provider,
                bm25_query,
                vector_query,
                filters,
                fusion_limit,
                rrf_k,
                stored_dim,
                vector_candidates,
            )
            .await?
        }
    };

    let augmented_count = if graph_augmentation_enabled {
        augment_pool(index_dir, &mut hybrid_results, filters)
    } else {
        0
    };

    if hybrid_results.is_empty() {
        return Ok((hybrid_results, timings));
    }

    if hybrid_results.len() <= result_limit && augmented_count == 0 {
        return Ok((hybrid_results, timings));
    }

    let rerank_start = Instant::now();
    // When graph candidates were appended, score the whole expanded pool so
    // candidates beyond the normal rerank prefix still compete semantically.
    let rerank_limit = if augmented_count > 0 {
        hybrid_results.len()
    } else {
        rerank_candidates
    };
    match rerank_results(reranker, vector_query, &hybrid_results, rerank_limit).await {
        Ok(mut reranked) => {
            timings.reranking = Some(rerank_start.elapsed());
            info!(
                query = vector_query,
                candidates = hybrid_results.len(),
                reranked = reranked.len(),
                "reranking complete"
            );
            reranked = merge_reranked_results(reranked, hybrid_results, rerank_limit, fetch_limit);
            Ok((reranked, timings))
        }
        Err(rerank_err) => {
            if matches!(
                rerank_err,
                crate::retrieval::reranker::RerankerError::Cancelled
            ) {
                return Err(HybridSearchError::Cancelled);
            }
            timings.reranking = Some(rerank_start.elapsed());
            warn!(
                error = %rerank_err,
                "reranker unavailable, returning unreranked results"
            );
            eprintln!(
                "Warning: reranker unavailable ({rerank_err}), returning unreranked results."
            );
            let mut results = hybrid_results;
            results.truncate(fetch_limit);
            Ok((results, timings))
        }
    }
}

fn merge_reranked_results(
    mut reranked: Vec<SearchResult>,
    hybrid_results: Vec<SearchResult>,
    rerank_limit: usize,
    fetch_limit: usize,
) -> Vec<SearchResult> {
    let rerank_prefix_len = rerank_limit.min(hybrid_results.len());
    let mut present = reranked.iter().map(result_key).collect::<HashSet<_>>();
    let mut score_ceiling = reranked.last().map(|result| result.score);

    let mut append_unscored = |mut result: SearchResult| {
        if !present.insert(result_key(&result)) {
            return;
        }
        // RRF and reranker scores have different scales. Keep unscored candidates below the
        // reranked prefix so the public score order still matches the result order.
        if let Some(ceiling) = score_ceiling {
            result.score = result.score.min(ceiling);
        }
        score_ceiling = Some(result.score);
        reranked.push(result);
    };

    // A short reranker response can omit candidates from the prefix. Append those first,
    // preserving their original hybrid order, before the normal untouched tail.
    for result in hybrid_results.iter().take(rerank_prefix_len) {
        append_unscored(result.clone());
    }
    for result in hybrid_results.into_iter().skip(rerank_prefix_len) {
        append_unscored(result);
    }

    reranked.truncate(fetch_limit);
    reranked
}

/// Perform hybrid search using pre-opened stores (useful for testing).
///
/// Takes pre-computed BM25 and vector results and fuses them via RRF.
/// This is the core fusion logic, separated from I/O for testability.
pub fn fuse_rrf(
    bm25_results: &[SearchResult],
    vector_results: &[SearchResult],
    rrf_k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    fuse_rrf_multi_weighted(&[bm25_results, vector_results], &[1.0, 1.0], rrf_k, limit)
}

/// Fuse BM25 and vector results with explicit per-source weights.
///
/// Identifier queries pass a higher BM25 weight (2.5) so lexical matches
/// dominate; NL queries use equal weights (1.0, 1.0).
pub fn fuse_rrf_weighted(
    bm25_results: &[SearchResult],
    vector_results: &[SearchResult],
    rrf_k: f64,
    limit: usize,
    bm25_weight: f64,
    vector_weight: f64,
) -> Vec<SearchResult> {
    fuse_rrf_multi_weighted(
        &[bm25_results, vector_results],
        &[bm25_weight, vector_weight],
        rrf_k,
        limit,
    )
}

/// Fuse multiple ranked result lists with weighted reciprocal rank fusion.
///
/// Each result set has an associated weight that scales its RRF contribution.
/// A weight of 2.0 means that set's scores count double in the final ranking.
///
/// Equal scores are broken by `(file_path, line_start, line_end)` ascending, so the
/// output is a function of the inputs alone and does not vary between processes.
pub fn fuse_rrf_multi_weighted(
    result_sets: &[&[SearchResult]],
    weights: &[f64],
    rrf_k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    let mut fused: HashMap<String, (f64, SearchResult)> = HashMap::new();

    for (set_idx, result_set) in result_sets.iter().enumerate() {
        let weight = weights.get(set_idx).copied().unwrap_or(1.0);
        for (rank_0, result) in result_set.iter().enumerate() {
            let key = result_key(result);
            let rrf_score = weight / (rrf_k + (rank_0 + 1) as f64);

            fused
                .entry(key)
                .and_modify(|(score, _)| *score += rrf_score)
                .or_insert_with(|| (rrf_score, result.clone()));
        }
    }

    let mut ranked: Vec<(f64, SearchResult)> = fused.into_values().collect();
    // Ties are structural, not incidental: with equal source weights a result found only
    // in set A at rank r and one found only in set B at rank r score bit-identically.
    // `into_values` yields a per-process-random order, so a score-only sort would make the
    // output depend on the HashMap seed rather than on the index. Break ties on the same
    // (file_path, line_start, line_end) triple that keys the map, which is distinct for
    // every surviving entry and therefore a total order.
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                (&a.1.file_path, a.1.line_start, a.1.line_end).cmp(&(
                    &b.1.file_path,
                    b.1.line_start,
                    b.1.line_end,
                ))
            })
    });

    ranked
        .into_iter()
        .take(limit)
        .map(|(rrf_score, mut result)| {
            result.score = rrf_score;
            result
        })
        .collect()
}

/// Charge the one wall-clock span that covers the vector arm to the two stages
/// `--timing` reports.
///
/// The query embedding runs inside the vector search, so both stages come out
/// of `vector_span`. `embedding` gets the model cost and `vector` gets the
/// remainder, which is the storage cost: opening both stores, the KNN query and
/// metadata hydration. A vector search that failed has no embedding measurement
/// to report, so the whole span stays on `vector`.
///
/// The two must partition the span rather than each be given all of it, which
/// is what issue #105 was: the same value reported twice made either stage
/// unattributable. Both fields are written here rather than at the call site so
/// that partition is decided in one place a test can reach: a caller that only
/// received the two values could still charge the span twice.
fn charge_vector_span(
    timings: &mut HybridTimings,
    vector_span: Duration,
    embed_elapsed: Option<Duration>,
) {
    timings.embedding = embed_elapsed;
    timings.vector = Some(vector_span.saturating_sub(embed_elapsed.unwrap_or_default()));
}

#[cfg(test)]
#[path = "hybrid_tests.rs"]
mod tests;

#[cfg(test)]
mod shortfall_tests {
    use super::*;
    use crate::types::Language;

    fn result(index: usize) -> SearchResult {
        SearchResult {
            file_path: format!("candidate-{index}.rs"),
            line_start: 1,
            line_end: 1,
            content: format!("candidate {index}"),
            language: Language::Rust,
            score: 1.0 - index as f64 / 100.0,
            symbol_name: None,
            symbol_type: None,
            part_index: None,
        }
    }

    #[test]
    fn reranker_shortfall_appends_missing_candidates_in_hybrid_order() {
        let hybrid_results: Vec<_> = (0..20).map(result).collect();
        let scored_indices = (0..7).chain(13..20);
        let reranked = scored_indices
            .enumerate()
            .map(|(rank, index)| {
                let mut result = hybrid_results[index].clone();
                result.score = 100.0 - rank as f64;
                result
            })
            .collect();

        let merged = merge_reranked_results(reranked, hybrid_results, 20, 20);

        assert_eq!(merged.len(), 20);
        let missing_tail: Vec<_> = merged[14..]
            .iter()
            .map(|result| result.content.as_str())
            .collect();
        assert_eq!(
            missing_tail,
            vec![
                "candidate 7",
                "candidate 8",
                "candidate 9",
                "candidate 10",
                "candidate 11",
                "candidate 12",
            ]
        );
    }
}
