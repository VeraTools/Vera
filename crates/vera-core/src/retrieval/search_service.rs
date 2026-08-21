//! Shared search service used by both CLI and MCP.
//!
//! Encapsulates the common hybrid search flow: create embedding provider,
//! build reranker, compute fetch limits, execute search, apply filters.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::warn;

use crate::config::{InferenceBackend, VeraConfig};
use crate::embedding::{CachedEmbeddingProvider, DynamicProvider, EmbeddingProvider};
use crate::retrieval::dynamic_reranker::DynamicReranker;
pub use crate::retrieval::exact_matches::augment_multi_query_exact_matches;
use crate::retrieval::exact_matches::{
    augment_exact_match_candidates, augment_exact_match_candidates_with_store,
};
use crate::retrieval::hybrid::{
    compute_vector_candidates, search_hybrid_reranked_with_augmentation,
};
use crate::retrieval::query_classifier::{QueryType, classify_query, params_for_query_type};
use crate::retrieval::ranking::{RankingStage, is_path_weighted_query};
use crate::retrieval::{apply_filters, search_bm25_with_stores_and_filters, search_hybrid};
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::types::{SearchFilters, SearchResult};

/// Timing data for each stage of the search pipeline.
#[derive(Debug, Default)]
pub struct SearchTimings {
    pub embedding: Option<Duration>,
    pub bm25: Option<Duration>,
    pub vector: Option<Duration>,
    pub fusion: Option<Duration>,
    pub reranking: Option<Duration>,
    pub augmentation: Option<Duration>,
    pub total: Option<Duration>,
}

impl From<crate::retrieval::hybrid::HybridTimings> for SearchTimings {
    fn from(t: crate::retrieval::hybrid::HybridTimings) -> Self {
        SearchTimings {
            embedding: t.embedding,
            bm25: t.bm25,
            vector: t.vector,
            fusion: t.fusion,
            reranking: t.reranking,
            augmentation: None,
            total: None,
        }
    }
}

/// Reusable search dependencies for a process or command invocation.
///
/// Local backends can take hundreds of milliseconds or seconds to initialize.
/// Keeping the provider and reranker here lets CLI multi-query search, deep
/// search, MCP, and eval reuse loaded models across repeated queries.
pub struct SearchContext {
    provider: Option<CachedEmbeddingProvider<DynamicProvider>>,
    model_name: Option<String>,
    provider_error: Option<String>,
    reranker: Option<DynamicReranker>,
}

impl SearchContext {
    pub async fn new(config: &VeraConfig, backend: InferenceBackend) -> Self {
        let (provider, model_name, provider_error) =
            match crate::embedding::create_dynamic_provider(config, backend).await {
                Ok((provider, model_name)) => (
                    Some(CachedEmbeddingProvider::with_namespace(
                        provider,
                        512,
                        &model_name,
                    )),
                    Some(model_name),
                    None,
                ),
                Err(err) => {
                    warn!(
                        "Failed to create embedding provider ({}), using BM25-only search",
                        err
                    );
                    (None, None, Some(err.to_string()))
                }
            };

        let reranker = if provider.is_some() {
            crate::retrieval::create_dynamic_reranker(config, backend)
                .await
                .unwrap_or_else(|err| {
                    warn!("Failed to create reranker ({})", err);
                    None
                })
        } else {
            None
        };

        Self {
            provider,
            model_name,
            provider_error,
            reranker,
        }
    }

    pub fn bm25_only() -> Self {
        Self {
            provider: None,
            model_name: None,
            provider_error: None,
            reranker: None,
        }
    }

    pub fn embedding_provider(&self) -> Option<&CachedEmbeddingProvider<DynamicProvider>> {
        self.provider.as_ref()
    }

    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    pub async fn search(
        &self,
        index_dir: &Path,
        query: &str,
        intent: Option<&str>,
        config: &VeraConfig,
        filters: &SearchFilters,
        result_limit: usize,
    ) -> Result<(Vec<SearchResult>, SearchTimings)> {
        let total_start = Instant::now();
        // `query` (raw) drives BM25, classification, expansion and exact-match
        // augmentation. The intent prefix only enriches the semantic side
        // (embedding + reranker); sending it to BM25 makes Tantivy parse
        // `intent:` as a non-existent field and fail. See issue #20.
        let vector_query = build_query_with_intent(query, intent);
        // True when an intent prefix was actually applied (non-empty intent).
        let has_intent = matches!(vector_query, std::borrow::Cow::Owned(_));
        let fetch_limit = compute_fetch_limit(query, filters, result_limit);

        let Some(provider) = self.provider.as_ref() else {
            if let Some(error) = self.provider_error.as_deref() {
                warn!(
                    "embedding provider unavailable ({}), using BM25-only search",
                    error
                );
            }
            return run_bm25_only(
                index_dir,
                query,
                filters,
                fetch_limit,
                result_limit,
                total_start,
            );
        };

        let mut stored_dim = config.embedding.max_stored_dim;

        // Check metadata mismatch
        let metadata_path = index_dir.join("metadata.db");
        if let Ok(metadata_store) = crate::storage::metadata::MetadataStore::open(&metadata_path) {
            if let (Some(s_model), Some(s_dim)) = (
                metadata_store.get_index_meta("model_name").unwrap_or(None),
                metadata_store
                    .get_index_meta("embedding_dim")
                    .unwrap_or(None),
            ) {
                if let Some(model_name) = self.model_name.as_deref() {
                    if !crate::config::model_names_match_with_aliases(
                        &s_model,
                        model_name,
                        &config.embedding.model_aliases,
                    ) {
                        warn!(
                            "Index model '{}' does not match active model '{}'; using BM25-only search",
                            s_model, model_name
                        );
                        return run_bm25_only(
                            index_dir,
                            query,
                            filters,
                            fetch_limit,
                            result_limit,
                            total_start,
                        );
                    }
                }
                if let Ok(dim) = s_dim.parse::<usize>() {
                    if let Some(provider_dim) = provider.expected_dim() {
                        if provider_dim < dim {
                            warn!(
                                "Index dimension {} exceeds provider dimension {}; using BM25-only search",
                                dim, provider_dim
                            );
                            return run_bm25_only(
                                index_dir,
                                query,
                                filters,
                                fetch_limit,
                                result_limit,
                                total_start,
                            );
                        }
                    }
                    stored_dim = dim;
                }
            }
        }

        // Create optional reranker. An explicit intent is a semantic signal
        // only the reranker/embedding side can use, so it forces reranking on.
        let reranker_enabled =
            reranking_enabled(self.reranker.is_some(), has_intent, query, filters);

        // Classify query to adapt fusion parameters.
        let query_type = classify_query(query);
        let query_params = params_for_query_type(query_type);
        let rrf_k = query_params.rrf_k;
        let vector_candidates = effective_vector_candidates(fetch_limit, query_params);
        let rerank_candidates =
            effective_rerank_candidates(config.retrieval.rerank_candidates, result_limit);

        let ranking_stage = if reranker_enabled {
            RankingStage::PostRerank
        } else {
            RankingStage::Initial
        };

        let (results, hybrid_timings) = if reranker_enabled {
            let reranker = self
                .reranker
                .as_ref()
                .expect("reranker_enabled requires an initialized reranker");
            search_hybrid_reranked_with_augmentation(
                index_dir,
                provider,
                reranker,
                query,
                vector_query.as_ref(),
                filters,
                fetch_limit,
                result_limit,
                rrf_k,
                stored_dim,
                rerank_candidates,
                vector_candidates,
                crate::config::graph_augmentation_enabled(),
            )
            .await?
        } else {
            search_hybrid(
                index_dir,
                provider,
                query,
                vector_query.as_ref(),
                filters,
                fetch_limit,
                rrf_k,
                stored_dim,
                vector_candidates,
            )
            .await?
        };

        let mut timings = SearchTimings::from(hybrid_timings);

        let aug_start = Instant::now();
        let results =
            augment_exact_match_candidates(index_dir, query, results, ranking_stage, filters)?;
        timings.augmentation = Some(aug_start.elapsed());

        timings.total = Some(total_start.elapsed());
        Ok((apply_filters(results, filters, result_limit), timings))
    }
}

/// Execute a search against the index at `index_dir`.
///
/// Attempts hybrid search (BM25 + vector + optional reranking). Falls
/// back to BM25-only when embedding API is unavailable.
pub fn execute_search(
    index_dir: &Path,
    query: &str,
    intent: Option<&str>,
    config: &VeraConfig,
    filters: &SearchFilters,
    result_limit: usize,
    backend: InferenceBackend,
) -> Result<(Vec<SearchResult>, SearchTimings)> {
    let rt = tokio::runtime::Runtime::new()?;
    let context = rt.block_on(SearchContext::new(config, backend));
    rt.block_on(context.search(index_dir, query, intent, config, filters, result_limit))
}

/// Build the semantic query text used for embedding and reranking.
///
/// When an `--intent` is supplied it is prefixed as `intent: <intent> | <query>`
/// (intent whitespace collapsed) to steer the embedding model. The raw `query`
/// is returned unchanged when no usable intent is present. This prefixed form
/// must never reach BM25: Tantivy's `QueryParser` treats `intent:` as a field
/// query and there is no such field. See issue #20.
pub(crate) fn build_query_with_intent<'a>(
    query: &'a str,
    intent: Option<&str>,
) -> std::borrow::Cow<'a, str> {
    let intent = intent
        .map(|value| value.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|value| !value.is_empty());
    match intent {
        // Only allocate when an intent prefix is actually applied; otherwise
        // borrow the raw query.
        Some(intent) => std::borrow::Cow::Owned(format!("intent: {intent} | {query}")),
        None => std::borrow::Cow::Borrowed(query),
    }
}

/// Compute how many candidates to keep through fusion before final truncation.
///
/// Broad natural-language queries need a larger pool even without explicit
/// filters so deterministic ranking can surface structural chunks that raw RRF
/// scores placed outside the requested result window.
fn compute_fetch_limit(query: &str, filters: &SearchFilters, result_limit: usize) -> usize {
    let mut fetch_limit = if filters.is_empty() {
        result_limit
    } else {
        result_limit.saturating_mul(3).max(result_limit + 20)
    };

    // Path globs are applied post-retrieval, so we need a much larger pool
    // to ensure enough matching files survive filtering.
    if !filters.path_glob.is_empty() {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(10).max(result_limit + 100));
    }

    if filters.exact_paths.is_some() {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(12).max(result_limit + 200));
    }

    if needs_structural_overfetch(query, filters) {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(8).max(result_limit + 140));
    } else if matches!(classify_query(query), QueryType::NaturalLanguage) {
        fetch_limit = fetch_limit.max(result_limit.saturating_mul(3).max(result_limit + 40));
    }

    fetch_limit
}

fn needs_structural_overfetch(query: &str, filters: &SearchFilters) -> bool {
    matches!(classify_query(query), QueryType::NaturalLanguage)
        && query.split_whitespace().count() >= 4
        && filters.path_glob.is_empty()
        && filters.exact_paths.is_none()
        && filters.symbol_type.is_none()
        && !is_path_weighted_query(query)
}

fn effective_vector_candidates(
    fetch_limit: usize,
    query_params: crate::retrieval::query_classifier::QueryParams,
) -> usize {
    compute_vector_candidates(fetch_limit, query_params.vector_candidate_multiplier)
}

fn effective_rerank_candidates(configured: usize, result_limit: usize) -> usize {
    configured.max(result_limit)
}

/// Decide whether the cross-encoder reranker should run.
///
/// Skip heuristics (short identifier / path-weighted / exact-path or
/// symbol-type filtered lookups) are based on the raw query. But when the user
/// supplies an `--intent`, they are asking for semantic ranking, so the
/// reranker must run even for short raw queries — otherwise the intent-enriched
/// query would never reach the cross-encoder. See issue #20.
fn reranking_enabled(
    reranker_present: bool,
    has_intent: bool,
    query: &str,
    filters: &SearchFilters,
) -> bool {
    reranker_present && (has_intent || !should_skip_reranking(query, filters))
}

fn should_skip_reranking(query: &str, filters: &SearchFilters) -> bool {
    let word_count = query.split_whitespace().count();
    filters.exact_paths.is_some()
        || filters.symbol_type.is_some()
        || is_path_weighted_query(query)
        || (matches!(classify_query(query), QueryType::Identifier) && word_count <= 2)
}

fn run_bm25_only(
    index_dir: &Path,
    query: &str,
    filters: &SearchFilters,
    fetch_limit: usize,
    result_limit: usize,
    total_start: Instant,
) -> Result<(Vec<SearchResult>, SearchTimings)> {
    let bm25_start = Instant::now();
    let bm25_index =
        Bm25Index::open(&index_dir.join("bm25")).context("failed to open BM25 index for search")?;
    let metadata_store = MetadataStore::open(&index_dir.join("metadata.db"))
        .context("failed to open metadata store for search")?;
    let results = search_bm25_with_stores_and_filters(
        &bm25_index,
        &metadata_store,
        query,
        filters,
        fetch_limit,
    )?;
    let bm25_elapsed = bm25_start.elapsed();
    let aug_start = Instant::now();
    let results = augment_exact_match_candidates_with_store(
        &metadata_store,
        query,
        results,
        RankingStage::Initial,
        filters,
    )?;
    let timings = SearchTimings {
        bm25: Some(bm25_elapsed),
        augmentation: Some(aug_start.elapsed()),
        total: Some(total_start.elapsed()),
        ..Default::default()
    };
    Ok((apply_filters(results, filters, result_limit), timings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::bm25::{Bm25Document, Bm25Index};
    use crate::storage::metadata::MetadataStore;
    use crate::test_env::EnvVarGuard;
    use crate::types::Language;
    use crate::types::{Chunk, SymbolType};
    use tempfile::tempdir;

    /// Point the API embedding provider at a dead local port. The guard holds
    /// the environment lock and puts the three variables back on any unwind.
    fn set_test_embedding_env(model_id: &str) -> EnvVarGuard {
        EnvVarGuard::set(&[
            ("EMBEDDING_MODEL_BASE_URL", "http://127.0.0.1:0"),
            ("EMBEDDING_MODEL_ID", model_id),
            ("EMBEDDING_MODEL_API_KEY", "dummy-key"),
        ])
    }

    #[test]
    fn test_dimension_mismatch_and_inference() {
        let _env_guard = set_test_embedding_env("dummy-api-model");
        let dir = tempdir().unwrap();
        let index_dir = dir.path();

        let metadata_path = index_dir.join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();

        // 1. Test dimension mismatch (requires local model so provider_dim is Some(768))
        store
            .set_index_meta("model_name", "jina-embeddings-v5-text-nano-retrieval")
            .unwrap();
        store.set_index_meta("embedding_dim", "1024").unwrap(); // Mismatch: 1024 vs 768

        let config = VeraConfig::default();
        let filters = SearchFilters::default();

        // This attempts local provider creation first, then falls back to BM25 when possible.
        // In this synthetic test fixture the BM25 index is absent, so either path may surface.
        {
            let res = execute_search(
                index_dir,
                "test",
                None,
                &config,
                &filters,
                10,
                crate::config::InferenceBackend::OnnxJina(
                    crate::config::OnnxExecutionProvider::Cpu,
                ),
            );
            if let Err(err) = res {
                let err_msg = err.to_string();
                assert!(
                    err_msg.contains("tantivy")
                        || err_msg.contains("Failed to initialize local embedding provider")
                        || err_msg.contains("No such file")
                        || err_msg.contains("not found"),
                    "{}",
                    err_msg
                );
            }
        }

        // 2. Test metadata-dimension inference path (API provider returns None for expected_dim)
        // The dummy provider credentials are already set by the guard above.
        store
            .set_index_meta("model_name", "dummy-api-model")
            .unwrap();
        store.set_index_meta("embedding_dim", "123").unwrap();

        // Calling execute_search with is_local = false
        // It will pass the metadata check (model_name matches), skip mismatch check (expected_dim is None),
        // infer stored_dim = 123, and proceed to search.
        // Since the index is empty, it will return Ok([]) without making network calls.
        let res = execute_search(
            index_dir,
            "test",
            None,
            &config,
            &filters,
            10,
            crate::config::InferenceBackend::Api,
        );
        assert!(res.is_ok(), "Expected Ok but got {:?}", res);
    }

    #[test]
    fn model_metadata_mismatch_falls_back_to_bm25() {
        let _env_guard = set_test_embedding_env("active-api-model");
        let dir = tempdir().unwrap();
        let index_dir = dir.path();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "auth:0".to_string(),
                file_path: "src/auth.rs".to_string(),
                line_start: 1,
                line_end: 4,
                content: "pub fn authenticate_user() -> bool { true }".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("authenticate_user".to_string()),
            }])
            .unwrap();
        store.set_index_meta("model_name", "indexed-model").unwrap();
        store.set_index_meta("embedding_dim", "64").unwrap();

        let bm25 = Bm25Index::open(&index_dir.join("bm25")).unwrap();
        bm25.insert_batch(&[Bm25Document {
            chunk_id: "auth:0",
            file_path: "src/auth.rs",
            content: "pub fn authenticate_user() -> bool { true }",
            symbol_name: Some("authenticate_user"),
            language: "rust",
        }])
        .unwrap();

        let mut config = VeraConfig::default();
        config.embedding.timeout_secs = 1;
        config.embedding.max_retries = 0;
        let filters = SearchFilters::default();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let context = rt.block_on(SearchContext::new(
            &config,
            crate::config::InferenceBackend::Api,
        ));

        let (results, timings) = rt
            .block_on(context.search(index_dir, "authenticate user", None, &config, &filters, 10))
            .unwrap();

        assert_eq!(results[0].file_path, "src/auth.rs");
        assert!(timings.bm25.is_some());
        assert!(timings.vector.is_none());
    }

    #[test]
    fn bm25_only_search_fills_path_scoped_results_from_deeper_pool() {
        let dir = tempdir().unwrap();
        let index_dir = dir.path();
        let metadata_store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        let bm25 = Bm25Index::open(&index_dir.join("bm25")).unwrap();

        let mut chunks = Vec::new();
        for i in 0..160 {
            chunks.push(Chunk {
                id: format!("noise:{i}"),
                file_path: format!("other/dependency_injection_work_{i}.py"),
                line_start: 1,
                line_end: 4,
                content:
                    "def dependency_injection_work():\n    dependency injection work dependency injection work"
                        .to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("dependency_injection_work".to_string()),
            });
        }
        chunks.push(Chunk {
            id: "fastapi:dependency".to_string(),
            file_path: "fastapi/dependencies/utils.py".to_string(),
            line_start: 42,
            line_end: 55,
            content: "def solve_dependencies():\n    \"\"\"Resolve dependency injection for request handlers.\"\"\"\n    return values"
                .to_string(),
            language: Language::Python,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("solve_dependencies".to_string()),
        });

        metadata_store.insert_chunks(&chunks).unwrap();
        let lang_strings: Vec<String> = chunks.iter().map(|c| c.language.to_string()).collect();
        let docs: Vec<Bm25Document<'_>> = chunks
            .iter()
            .zip(lang_strings.iter())
            .map(|(chunk, language)| Bm25Document {
                chunk_id: &chunk.id,
                file_path: &chunk.file_path,
                content: &chunk.content,
                symbol_name: chunk.symbol_name.as_deref(),
                language,
            })
            .collect();
        bm25.insert_batch(&docs).unwrap();

        let filters = SearchFilters {
            path_glob: vec!["fastapi/**".to_string()],
            ..Default::default()
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let context = SearchContext::bm25_only();

        let (results, timings) = rt
            .block_on(context.search(
                index_dir,
                "how does dependency injection work",
                None,
                &VeraConfig::default(),
                &filters,
                5,
            ))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "fastapi/dependencies/utils.py");
        assert!(timings.bm25.is_some());
    }

    #[test]
    fn build_query_with_intent_formats_and_normalizes() {
        // No intent: raw query borrowed unchanged (this is what BM25 receives).
        let none = build_query_with_intent("test query", None);
        assert_eq!(none.as_ref(), "test query");
        assert!(matches!(none, std::borrow::Cow::Borrowed(_)));
        // Empty / whitespace-only intent collapses to no prefix (still borrowed).
        let empty = build_query_with_intent("test query", Some("   "));
        assert_eq!(empty.as_ref(), "test query");
        assert!(matches!(empty, std::borrow::Cow::Borrowed(_)));
        // Intent present: owned prefixed form with whitespace collapsed.
        let some = build_query_with_intent("test query", Some("find  auth\n handlers"));
        assert_eq!(some.as_ref(), "intent: find auth handlers | test query");
        assert!(matches!(some, std::borrow::Cow::Owned(_)));
    }

    #[test]
    fn reranking_runs_for_intent_searches_with_short_queries() {
        let filters = SearchFilters::default();
        // A short identifier query alone is skipped by the rerank heuristic.
        assert!(should_skip_reranking("authenticate", &filters));
        // Without intent that means reranking is off...
        assert!(!reranking_enabled(true, false, "authenticate", &filters));
        // ...but an intent forces the cross-encoder to run so it sees the
        // intent-enriched query (issue #20 regression guard).
        assert!(reranking_enabled(true, true, "authenticate", &filters));
        // No reranker present: always off regardless of intent.
        assert!(!reranking_enabled(false, true, "authenticate", &filters));
        // Natural-language query without intent still reranks normally.
        assert!(reranking_enabled(
            true,
            false,
            "how does auth work",
            &filters
        ));
    }

    #[test]
    fn path_glob_filters_do_not_skip_reranking() {
        let filters = SearchFilters {
            path_glob: vec!["crates/vera-core/**".to_string()],
            ..Default::default()
        };

        assert!(!should_skip_reranking(
            "how does dependency injection work",
            &filters
        ));
    }

    #[test]
    fn rerank_candidate_depth_is_independent_of_fetch_depth() {
        assert_eq!(effective_rerank_candidates(20, 5), 20);
        assert_eq!(
            effective_rerank_candidates(4, 5),
            5,
            "reranking must cover enough candidates to return the requested result count"
        );

        // Vector candidates use query_params multiplier without inflation
        let nl_params =
            params_for_query_type(crate::retrieval::query_classifier::QueryType::NaturalLanguage);
        let vc = effective_vector_candidates(10, nl_params);
        assert!(vc >= 50); // at least the minimum from compute_vector_candidates
    }

    #[test]
    fn broad_nl_queries_overfetch_before_ranking() {
        let filters = SearchFilters::default();

        assert_eq!(compute_fetch_limit("Config", &filters, 20), 20);
        assert_eq!(
            compute_fetch_limit("file type detection and filtering", &filters, 20),
            160
        );
        assert_eq!(
            compute_fetch_limit(
                "how are HTTP errors handled and returned to clients",
                &filters,
                5
            ),
            145
        );
    }

    #[test]
    fn exact_identifier_queries_skip_reranking() {
        assert!(should_skip_reranking("Config", &SearchFilters::default()));
        assert!(should_skip_reranking(
            "src/config.ts",
            &SearchFilters::default()
        ));
        assert!(!should_skip_reranking(
            "how are HTTP errors handled",
            &SearchFilters::default()
        ));
    }

    #[test]
    fn exact_identifier_lookup_finds_matching_symbol() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "sink:0".to_string(),
                file_path: "crates/searcher/src/sink.rs".to_string(),
                line_start: 102,
                line_end: 223,
                content: "pub trait Sink {}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Trait),
                symbol_name: Some("Sink".to_string()),
            }])
            .unwrap();

        let augmented = augment_exact_match_candidates(
            dir.path(),
            "Sink trait and its implementations",
            Vec::new(),
            RankingStage::Initial,
            &SearchFilters::default(),
        )
        .unwrap();

        assert!(
            augmented
                .iter()
                .any(|result| result.symbol_name.as_deref() == Some("Sink"))
        );
    }

    #[test]
    fn exact_identifier_prefers_public_type_definition() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "config:0".to_string(),
                    file_path: "crates/core/search.rs".to_string(),
                    line_start: 19,
                    line_end: 25,
                    content: "struct Config {\n    search_zip: bool,\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("Config".to_string()),
                },
                Chunk {
                    id: "config:1".to_string(),
                    file_path: "crates/regex/src/config.rs".to_string(),
                    line_start: 25,
                    line_end: 43,
                    content: "pub(crate) struct Config {\n    pub(crate) multi_line: bool,\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("Config".to_string()),
                },
                Chunk {
                    id: "config:2".to_string(),
                    file_path: "crates/searcher/src/searcher/mod.rs".to_string(),
                    line_start: 151,
                    line_end: 185,
                    content: "pub struct Config {\n    line_term: LineTerminator,\n    multi_line: bool,\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("Config".to_string()),
                },
            ])
            .unwrap();

        let augmented = augment_exact_match_candidates(
            dir.path(),
            "Config",
            Vec::new(),
            RankingStage::Initial,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(
            augmented[0].file_path,
            "crates/searcher/src/searcher/mod.rs"
        );
    }

    #[test]
    fn multi_query_exact_matches_are_promoted_after_fusion() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "kimi:0".to_string(),
                    file_path: "backend/crates/omnigate-auth/src/kimi.rs".to_string(),
                    line_start: 265,
                    line_end: 275,
                    content: "pub fn persist_kimi_auth_record() {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("persist_kimi_auth_record".to_string()),
                },
                Chunk {
                    id: "factory:0".to_string(),
                    file_path: "backend/crates/omnigate-auth/src/factory.rs".to_string(),
                    line_start: 286,
                    line_end: 296,
                    content: "pub fn persist_factory_auth_record() {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("persist_factory_auth_record".to_string()),
                },
                Chunk {
                    id: "lib:0".to_string(),
                    file_path: "backend/crates/omnigate-auth/src/lib.rs".to_string(),
                    line_start: 10,
                    line_end: 40,
                    content: "pub use crate::kimi::persist_kimi_auth_record;".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Module),
                    symbol_name: Some("omnigate_auth".to_string()),
                },
            ])
            .unwrap();

        let fused = vec![
            SearchResult {
                file_path: "backend/crates/omnigate-auth/src/lib.rs".to_string(),
                line_start: 10,
                line_end: 40,
                content: "pub use crate::kimi::persist_kimi_auth_record;".to_string(),
                language: Language::Rust,
                score: 0.0,
                symbol_name: Some("omnigate_auth".to_string()),
                symbol_type: Some(SymbolType::Module),
            },
            SearchResult {
                file_path: "backend/crates/omnigate-auth/src/factory.rs".to_string(),
                line_start: 286,
                line_end: 296,
                content: "pub fn persist_factory_auth_record() {}".to_string(),
                language: Language::Rust,
                score: 0.0,
                symbol_name: Some("persist_factory_auth_record".to_string()),
                symbol_type: Some(SymbolType::Function),
            },
            SearchResult {
                file_path: "backend/crates/omnigate-auth/src/kimi.rs".to_string(),
                line_start: 265,
                line_end: 275,
                content: "pub fn persist_kimi_auth_record() {}".to_string(),
                language: Language::Rust,
                score: 0.0,
                symbol_name: Some("persist_kimi_auth_record".to_string()),
                symbol_type: Some(SymbolType::Function),
            },
        ];

        let queries = vec![
            "persist_kimi_auth_record".to_string(),
            "persist_factory_auth_record".to_string(),
        ];
        let augmented = augment_multi_query_exact_matches(
            dir.path(),
            &queries,
            fused,
            &SearchFilters::default(),
            5,
        )
        .unwrap();

        assert_eq!(
            augmented[0].symbol_name.as_deref(),
            Some("persist_kimi_auth_record")
        );
        assert_eq!(
            augmented[1].symbol_name.as_deref(),
            Some("persist_factory_auth_record")
        );
    }

    #[test]
    fn multi_query_exact_matches_interleave_across_queries() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();

        let mut chunks: Vec<Chunk> = (0..5)
            .map(|i| Chunk {
                id: format!("common:{i}"),
                file_path: format!("src/common_{i}.rs"),
                line_start: 1,
                line_end: 10,
                content: "pub fn common_fn() {}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("common_fn".to_string()),
            })
            .collect();
        chunks.push(Chunk {
            id: "rare:0".to_string(),
            file_path: "src/rare.rs".to_string(),
            line_start: 1,
            line_end: 10,
            content: "pub fn rare_fn() {}".to_string(),
            language: Language::Rust,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("rare_fn".to_string()),
        });
        store.insert_chunks(&chunks).unwrap();

        let augmented = augment_multi_query_exact_matches(
            dir.path(),
            &["common_fn".to_string(), "rare_fn".to_string()],
            Vec::new(),
            &SearchFilters::default(),
            3,
        )
        .unwrap();

        // The first query keeps the top slot, but the second query must
        // contribute instead of being crowded out by the first query's rows.
        assert_eq!(augmented.len(), 3);
        assert_eq!(augmented[0].symbol_name.as_deref(), Some("common_fn"));
        assert_eq!(augmented[1].symbol_name.as_deref(), Some("rare_fn"));
    }

    #[test]
    fn bare_filename_query_gets_exact_file_augmentation() {
        let dir = tempdir().unwrap();
        let metadata_path = dir.path().join("metadata.db");
        let store = MetadataStore::open(&metadata_path).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "deep:0".to_string(),
                    file_path: "deep/nested/handler.py".to_string(),
                    line_start: 1,
                    line_end: 20,
                    content: "def handle(): pass".to_string(),
                    language: Language::Python,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("handle".to_string()),
                },
                Chunk {
                    id: "shallow:0".to_string(),
                    file_path: "src/handler.py".to_string(),
                    line_start: 1,
                    line_end: 20,
                    content: "def handle(): pass".to_string(),
                    language: Language::Python,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("handle".to_string()),
                },
            ])
            .unwrap();

        let augmented = augment_multi_query_exact_matches(
            dir.path(),
            &["handler.py".to_string()],
            Vec::new(),
            &SearchFilters::default(),
            5,
        )
        .unwrap();

        assert!(
            !augmented.is_empty(),
            "a bare filename query must inject exact file chunks"
        );
        assert_eq!(augmented[0].file_path, "src/handler.py");
    }
}
