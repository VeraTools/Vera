//! BM25 standalone search over indexed chunks.
//!
//! Provides keyword-based search using the Tantivy BM25 index, with results
//! hydrated from the metadata store. Exact keyword matches rank higher than
//! partial matches through Tantivy's native BM25 scoring combined with
//! symbol-name boosting.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::debug;

use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::types::{SearchFilters, SearchResult};

const FILTERED_RAW_MULTIPLIER: usize = 24;
const FILTERED_RAW_MIN_EXTRA: usize = 1_000;
const FILTERED_RAW_MAX: usize = 20_000;
const BM25_HYDRATION_PAGE_MAX: usize = 900;

/// Perform a BM25 keyword search over the indexed chunks.
///
/// Opens the BM25 index and metadata store from the index directory,
/// executes the query, and returns hydrated search results sorted by
/// BM25 score (descending).
///
/// # Arguments
/// - `index_dir` — Path to the `.vera` index directory
/// - `query` — The search query text
/// - `limit` — Maximum number of results to return
///
/// # Returns
/// A vector of `SearchResult` with full chunk metadata, sorted by score descending.
pub fn search_bm25(index_dir: &Path, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let bm25_dir = index_dir.join("bm25");
    let metadata_path = index_dir.join("metadata.db");

    let bm25_index = Bm25Index::open(&bm25_dir).context("failed to open BM25 index for search")?;
    let metadata_store =
        MetadataStore::open(&metadata_path).context("failed to open metadata store for search")?;

    search_bm25_with_stores(&bm25_index, &metadata_store, query, limit)
}

/// Perform BM25 search using pre-opened stores (useful for testing and reuse).
///
/// Searches the BM25 index for the given query, then hydrates each result
/// with full chunk metadata from the metadata store. Results are returned
/// sorted by BM25 score in descending order.
pub fn search_bm25_with_stores(
    bm25_index: &Bm25Index,
    metadata_store: &MetadataStore,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    search_bm25_with_stores_inner(
        bm25_index,
        metadata_store,
        query,
        limit,
        unfiltered_raw_candidate_limit(limit),
        None,
    )
}

/// Perform BM25 search using pre-opened stores and active filters.
///
/// Filtered searches scan a larger raw Tantivy pool, hydrate candidates in
/// metadata batches, and keep matching chunks until `limit` results are collected or candidates are exhausted.
pub fn search_bm25_with_stores_and_filters(
    bm25_index: &Bm25Index,
    metadata_store: &MetadataStore,
    query: &str,
    filters: &SearchFilters,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    if filters.is_empty() {
        return search_bm25_with_stores(bm25_index, metadata_store, query, limit);
    }

    search_bm25_with_stores_inner(
        bm25_index,
        metadata_store,
        query,
        limit,
        filtered_raw_candidate_limit(limit),
        Some(filters),
    )
}

fn search_bm25_with_stores_inner(
    bm25_index: &Bm25Index,
    metadata_store: &MetadataStore,
    query: &str,
    limit: usize,
    raw_limit: usize,
    filters: Option<&SearchFilters>,
) -> Result<Vec<SearchResult>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let bm25_results = bm25_index
        .search(query, raw_limit)
        .with_context(|| format!("BM25 search failed for query: {query}"))?;

    debug!(
        query = query,
        raw_results = bm25_results.len(),
        "BM25 search returned candidates"
    );

    let mut results = Vec::with_capacity(limit.min(bm25_results.len()));

    // The first `limit` candidates hydrate one row at a time through the
    // cached single-row statement: when every candidate passes (the common
    // case, especially unfiltered) this is the cheapest possible path and
    // fetches nothing that goes unused. If the head falls short (filter
    // rejection or missing metadata), the remaining pool hydrates in paged
    // batches, amortizing SQLite round trips over the long rejection tail.
    let (head, tail) = bm25_results.split_at(limit.min(bm25_results.len()));
    for bm25_result in head {
        let Some(chunk) = metadata_store
            .get_chunk(&bm25_result.chunk_id)
            .with_context(|| {
                format!(
                    "failed to fetch metadata for chunk: {}",
                    bm25_result.chunk_id
                )
            })?
        else {
            debug!(
                chunk_id = %bm25_result.chunk_id,
                "chunk metadata not found, skipping"
            );
            continue;
        };
        let result = chunk.into_search_result(f64::from(bm25_result.score));
        if filters.is_some_and(|filters| !filters.matches(&result)) {
            continue;
        }
        results.push(result);
    }

    if results.len() < limit && !tail.is_empty() {
        let hydration_page_size = limit.saturating_mul(4).clamp(256, BM25_HYDRATION_PAGE_MAX);

        for page in tail.chunks(hydration_page_size) {
            let ids: Vec<String> = page
                .iter()
                .map(|bm25_result| bm25_result.chunk_id.clone())
                .collect();
            let mut chunks_by_id = metadata_store.get_chunks_by_ids(&ids).with_context(|| {
                format!("failed to fetch metadata for {} BM25 candidates", ids.len())
            })?;

            for bm25_result in page {
                // `remove` moves the chunk out of the page map: every id is
                // visited once, so no clone is needed.
                let Some(chunk) = chunks_by_id.remove(&bm25_result.chunk_id) else {
                    debug!(
                        chunk_id = %bm25_result.chunk_id,
                        "chunk metadata not found, skipping"
                    );
                    continue;
                };

                let result = chunk.into_search_result(f64::from(bm25_result.score));

                if filters.is_some_and(|filters| !filters.matches(&result)) {
                    continue;
                }

                results.push(result);

                if results.len() >= limit {
                    break;
                }
            }

            if results.len() >= limit {
                break;
            }
        }
    }

    debug!(
        query = query,
        returned = results.len(),
        "BM25 search complete"
    );

    Ok(results)
}

fn unfiltered_raw_candidate_limit(limit: usize) -> usize {
    limit.saturating_mul(2).max(limit.saturating_add(10))
}

fn filtered_raw_candidate_limit(limit: usize) -> usize {
    if limit == 0 {
        0
    } else {
        limit
            .saturating_mul(FILTERED_RAW_MULTIPLIER)
            .max(limit.saturating_add(FILTERED_RAW_MIN_EXTRA))
            .min(FILTERED_RAW_MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::bm25::Bm25Document;
    use crate::types::{Chunk, Language, SymbolType};

    /// Create a set of sample chunks with varied content for testing.
    fn sample_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                id: "src/auth.rs:0".to_string(),
                file_path: "src/auth.rs".to_string(),
                line_start: 1,
                line_end: 15,
                content: "pub fn authenticate(user: &str, password: &str) -> Result<Token> {\n    \
                           let hash = hash_password(password);\n    \
                           verify_credentials(user, &hash)\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("authenticate".to_string()),
                part_index: None,
            },
            Chunk {
                id: "src/auth.rs:1".to_string(),
                file_path: "src/auth.rs".to_string(),
                line_start: 17,
                line_end: 25,
                content: "pub fn verify_credentials(user: &str, hash: &str) -> Result<Token> {\n    \
                           let stored = db.get_user_hash(user)?;\n    \
                           if stored == hash { Ok(Token::new()) } else { Err(AuthError) }\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("verify_credentials".to_string()),
                part_index: None,
            },
            Chunk {
                id: "src/config.py:0".to_string(),
                file_path: "src/config.py".to_string(),
                line_start: 1,
                line_end: 10,
                content: "class DatabaseConfig:\n    \
                           def __init__(self, host, port, name):\n        \
                           self.host = host\n        \
                           self.port = port\n        \
                           self.name = name"
                    .to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Class),
                symbol_name: Some("DatabaseConfig".to_string()),
                part_index: None,
            },
            Chunk {
                id: "src/server.ts:0".to_string(),
                file_path: "src/server.ts".to_string(),
                line_start: 1,
                line_end: 12,
                content: "function handleRequest(req: Request): Response {\n    \
                           const auth = authenticate(req.headers);\n    \
                           if (!auth) return new Response('Unauthorized', { status: 401 });\n    \
                           return processRequest(req);\n}"
                    .to_string(),
                language: Language::TypeScript,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("handleRequest".to_string()),
                part_index: None,
            },
            Chunk {
                id: "src/utils.rs:0".to_string(),
                file_path: "src/utils.rs".to_string(),
                line_start: 1,
                line_end: 8,
                content: "pub fn format_output(data: &[u8]) -> String {\n    \
                           String::from_utf8_lossy(data).to_string()\n}"
                    .to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("format_output".to_string()),
                part_index: None,
            },
            Chunk {
                id: "src/db.go:0".to_string(),
                file_path: "src/db.go".to_string(),
                line_start: 1,
                line_end: 10,
                content: "func ConnectDatabase(config DatabaseConfig) (*sql.DB, error) {\n    \
                           dsn := fmt.Sprintf(\"%s:%d/%s\", config.Host, config.Port, config.Name)\n    \
                           return sql.Open(\"postgres\", dsn)\n}"
                    .to_string(),
                language: Language::Go,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("ConnectDatabase".to_string()),
                part_index: None,
            },
        ]
    }

    /// Set up in-memory BM25 index and metadata store from sample chunks.
    fn setup_test_stores() -> (Bm25Index, MetadataStore) {
        let chunks = sample_chunks();

        let metadata_store = MetadataStore::open_in_memory().unwrap();
        metadata_store.insert_chunks(&chunks).unwrap();

        let bm25_index = Bm25Index::open_in_memory().unwrap();
        bm25_index.insert_chunks(&chunks).unwrap();

        (bm25_index, metadata_store)
    }

    #[test]
    fn search_returns_results_for_known_keywords() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "authenticate", 10).unwrap();
        assert!(
            !results.is_empty(),
            "should find results for 'authenticate'"
        );
    }

    #[test]
    fn results_ranked_by_score_descending() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "authenticate", 10).unwrap();
        assert!(
            results.len() >= 2,
            "need multiple results to verify ordering"
        );

        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "scores must be descending: {} >= {} at position {i}",
                results[i - 1].score,
                results[i].score,
            );
        }
    }

    #[test]
    fn exact_identifier_match_ranks_highest() {
        let (bm25, metadata) = setup_test_stores();
        // Search for exact function name "authenticate"
        let results = search_bm25_with_stores(&bm25, &metadata, "authenticate", 10).unwrap();
        assert!(!results.is_empty());

        // The function named "authenticate" should be the top result.
        assert_eq!(
            results[0].symbol_name.as_deref(),
            Some("authenticate"),
            "exact symbol name match should be the top result"
        );
        assert_eq!(results[0].file_path, "src/auth.rs");
    }

    #[test]
    fn results_include_chunk_metadata() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "DatabaseConfig", 10).unwrap();
        assert!(!results.is_empty());

        let top = &results[0];
        assert_eq!(top.file_path, "src/config.py");
        assert_eq!(top.line_start, 1);
        assert_eq!(top.line_end, 10);
        assert_eq!(top.language, Language::Python);
        assert_eq!(top.symbol_name.as_deref(), Some("DatabaseConfig"));
        assert_eq!(top.symbol_type, Some(SymbolType::Class));
        assert!(top.score > 0.0, "score should be positive");
        assert!(
            top.content.contains("class DatabaseConfig"),
            "content should be present"
        );
    }

    #[test]
    fn search_respects_limit() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "function", 2).unwrap();
        assert!(results.len() <= 2, "results should respect the limit");
    }

    #[test]
    fn search_no_results_returns_empty() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "xyznonexistent999", 10).unwrap();
        assert!(results.is_empty(), "no results for nonsense query");
    }

    #[test]
    fn search_finds_content_keywords() {
        let (bm25, metadata) = setup_test_stores();
        // "password" appears in the authenticate function's body
        let results = search_bm25_with_stores(&bm25, &metadata, "password", 10).unwrap();
        assert!(!results.is_empty());
        let found = results.iter().any(|r| r.file_path == "src/auth.rs");
        assert!(found, "should find auth.rs for 'password' keyword");
    }

    #[test]
    fn search_finds_symbol_name_directly() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "ConnectDatabase", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].symbol_name.as_deref(),
            Some("ConnectDatabase"),
            "searching for exact symbol name should find it"
        );
    }

    #[test]
    fn exact_match_ranks_higher_than_partial() {
        let (bm25, metadata) = setup_test_stores();
        // "handleRequest" is an exact symbol name; "Request" appears in content
        // of multiple chunks. The exact symbol match should rank higher.
        let results = search_bm25_with_stores(&bm25, &metadata, "handleRequest", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].symbol_name.as_deref(),
            Some("handleRequest"),
            "exact identifier match should rank highest"
        );
    }

    #[test]
    fn multiple_results_from_same_file() {
        let (bm25, metadata) = setup_test_stores();
        // Both functions in auth.rs mention credentials/auth concepts
        let results = search_bm25_with_stores(&bm25, &metadata, "credentials", 10).unwrap();
        let auth_results: Vec<_> = results
            .iter()
            .filter(|r| r.file_path == "src/auth.rs")
            .collect();
        assert!(
            !auth_results.is_empty(),
            "should find at least one result from auth.rs"
        );
    }

    #[test]
    fn scores_are_positive() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "authenticate", 10).unwrap();
        for result in &results {
            assert!(result.score > 0.0, "BM25 scores should be positive");
        }
    }

    #[test]
    fn search_across_languages() {
        let (bm25, metadata) = setup_test_stores();
        // "config" appears in Python's DatabaseConfig and Go's ConnectDatabase
        let results = search_bm25_with_stores(&bm25, &metadata, "config", 10).unwrap();
        assert!(!results.is_empty());

        let languages: Vec<_> = results.iter().map(|r| r.language).collect();
        // Should find results from multiple languages
        let has_python = languages.contains(&Language::Python);
        let has_go = languages.contains(&Language::Go);
        assert!(
            has_python || has_go,
            "should find results across languages for 'config'"
        );
    }

    #[test]
    fn result_content_matches_metadata() {
        let (bm25, metadata) = setup_test_stores();
        let results = search_bm25_with_stores(&bm25, &metadata, "format_output", 10).unwrap();
        assert!(!results.is_empty());

        let result = &results[0];
        assert_eq!(result.file_path, "src/utils.rs");
        assert_eq!(result.symbol_name.as_deref(), Some("format_output"));
        assert!(result.content.contains("format_output"));
        assert!(result.line_start > 0, "line_start should be 1-based");
        assert!(
            result.line_end >= result.line_start,
            "line_end >= line_start"
        );
    }

    #[test]
    fn filtered_raw_candidate_limit_is_bounded() {
        assert_eq!(filtered_raw_candidate_limit(0), 0);
        assert_eq!(filtered_raw_candidate_limit(10), 1_010);
        assert_eq!(filtered_raw_candidate_limit(1_000), 20_000);
    }

    #[test]
    fn filtered_search_finds_scoped_result_buried_by_off_scope_hits() {
        let metadata_store = MetadataStore::open_in_memory().unwrap();
        let bm25_index = Bm25Index::open_in_memory().unwrap();

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
                part_index: None,
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
            part_index: None,
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
        bm25_index.insert_batch(&docs).unwrap();

        let filters = SearchFilters {
            path_glob: vec!["fastapi/**".to_string()],
            ..Default::default()
        };
        let query = "how does dependency injection work";

        let shallow = search_bm25_with_stores(&bm25_index, &metadata_store, query, 10).unwrap();
        assert!(
            shallow.iter().all(|result| !filters.matches(result)),
            "unfiltered top results should be off-scope"
        );

        let filtered =
            search_bm25_with_stores_and_filters(&bm25_index, &metadata_store, query, &filters, 10)
                .unwrap();

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].file_path, "fastapi/dependencies/utils.py");
    }

    fn setup_language_filter_fixture(matching_count: usize) -> (Bm25Index, MetadataStore) {
        let metadata_store = MetadataStore::open_in_memory().unwrap();
        let bm25_index = Bm25Index::open_in_memory().unwrap();

        let noise_count = 640;
        let mut chunks = Vec::with_capacity(noise_count + matching_count);
        for i in 0..noise_count {
            chunks.push(Chunk {
                id: format!("noise:{i}"),
                file_path: format!("vendor/noise_{i}.py"),
                line_start: 1,
                line_end: 4,
                content: format!(
                    "def noise_{i}():\n    return \"paged hydration marker paged hydration marker paged hydration marker\""
                ),
                language: Language::Python,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("paged_hydration_marker".to_string()),
                part_index: None,
            });
        }
        for i in 0..matching_count {
            chunks.push(Chunk {
                id: format!("match:{i}"),
                file_path: format!("src/matches/match_{i}.rs"),
                line_start: 1,
                line_end: 4,
                content: format!(
                    "fn match_{i}() {{\n    let marker = \"paged hydration marker\";\n    marker\n}}"
                ),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some(format!("match_{i}")),
                part_index: None,
            });
        }

        metadata_store.insert_chunks(&chunks).unwrap();
        bm25_index.insert_chunks(&chunks).unwrap();

        (bm25_index, metadata_store)
    }

    #[test]
    fn paged_hydration_matches_unfiltered_manual_filtering_across_pages() {
        let (bm25_index, metadata_store) = setup_language_filter_fixture(12);
        let filters = SearchFilters {
            language: Some("rust".to_string()),
            ..Default::default()
        };
        let query = "paged hydration marker";
        let limit = 8;

        let raw_candidates = bm25_index
            .search(query, filtered_raw_candidate_limit(limit))
            .unwrap();
        let first_matching_position = raw_candidates
            .iter()
            .position(|candidate| candidate.chunk_id.starts_with("match:"))
            .expect("fixture should contain matching-language candidates");
        assert!(
            first_matching_position >= 256,
            "matching candidates should require hydration beyond the first page: {first_matching_position}"
        );

        let filtered = search_bm25_with_stores_and_filters(
            &bm25_index,
            &metadata_store,
            query,
            &filters,
            limit,
        )
        .unwrap();
        let expected: Vec<_> = search_bm25_with_stores(
            &bm25_index,
            &metadata_store,
            query,
            filtered_raw_candidate_limit(limit),
        )
        .unwrap()
        .into_iter()
        .filter(|result| filters.matches(result))
        .take(limit)
        .collect();

        let result_signature = |result: &SearchResult| {
            (
                result.file_path.clone(),
                result.line_start,
                result.line_end,
                result.content.clone(),
                result.language,
                result.score,
                result.symbol_name.clone(),
                result.symbol_type,
            )
        };
        let actual_signature: Vec<_> = filtered.iter().map(result_signature).collect();
        let expected_signature: Vec<_> = expected.iter().map(result_signature).collect();

        assert_eq!(filtered.len(), limit);
        assert_eq!(actual_signature, expected_signature);
    }

    #[test]
    fn filtered_search_returns_all_matches_when_candidates_are_exhausted() {
        let (bm25_index, metadata_store) = setup_language_filter_fixture(3);
        let filters = SearchFilters {
            language: Some("rust".to_string()),
            ..Default::default()
        };

        let results = search_bm25_with_stores_and_filters(
            &bm25_index,
            &metadata_store,
            "paged hydration marker",
            &filters,
            10,
        )
        .unwrap();

        assert_eq!(results.len(), 3);
        assert!(
            results
                .iter()
                .all(|result| result.language == Language::Rust)
        );
    }
}
