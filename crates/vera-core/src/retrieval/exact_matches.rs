//! Exact-match candidate augmentation.
//!
//! Short identifier and filename queries get supplemental candidates pulled
//! straight from the metadata store so direct symbol lookups stay near the
//! top even when BM25/vector ranking would bury them.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

use crate::chunk_text::file_name;
use crate::retrieval::apply_filters;
use crate::retrieval::query_utils::{
    content_declares_public_symbol, content_starts_with_impl, file_stem,
    looks_like_compound_identifier, looks_like_filename, path_depth, result_key, trim_query_token,
};
use crate::retrieval::ranking::{
    RankingStage, apply_query_ranking_multi_query, apply_query_ranking_with_filters,
    is_path_weighted_query,
};
use crate::storage::metadata::MetadataStore;
use crate::types::{Chunk, SearchFilters, SearchResult, SymbolType};

#[cfg(test)]
pub(crate) fn augment_exact_match_candidates(
    index_dir: &Path,
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let metadata_path = index_dir.join("metadata.db");
    let Ok(store) = MetadataStore::open(&metadata_path) else {
        return Ok(apply_query_ranking_with_filters(
            query, results, stage, filters,
        ));
    };
    let Ok(files) = store.indexed_files() else {
        return Ok(apply_query_ranking_with_filters(
            query, results, stage, filters,
        ));
    };
    augment_exact_match_candidates_with_store(&store, &files, query, results, stage, filters)
}

/// Maximum definition chunks one concept-matched file may contribute.
const CONCEPT_CHUNKS_PER_FILE: usize = 4;

pub(crate) fn augment_exact_match_candidates_with_store(
    store: &MetadataStore,
    indexed_files: &[String],
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let supplemental = collect_exact_match_candidates(store, indexed_files, query, 0)?;

    // Concept matches only fire when no exact filename or identifier matched.
    // They join the pool tail so ranking signals can decide their position.
    let concept = if supplemental.is_empty() {
        collect_concept_matched_files(store, indexed_files, query)?
            .into_iter()
            .map(|chunk| chunk.into_search_result(0.0))
            .collect()
    } else {
        Vec::new()
    };
    if supplemental.is_empty() && concept.is_empty() {
        return Ok(apply_query_ranking_with_filters(
            query, results, stage, filters,
        ));
    }
    let mut merged = merge_exact_matches(supplemental, results);
    append_new_candidates(&mut merged, concept);
    Ok(apply_query_ranking_with_filters(
        query, merged, stage, filters,
    ))
}

/// Re-apply exact-match boosting after multi-query fusion.
///
/// Each subquery may have a different exact identifier or filename target.
/// Adding those exact hits again after RRF keeps direct symbol lookups near the
/// top instead of letting merged scores bury them behind broad contextual hits.
pub fn augment_multi_query_exact_matches(
    index_dir: &Path,
    queries: &[String],
    results: Vec<SearchResult>,
    filters: &SearchFilters,
    result_limit: usize,
) -> Result<Vec<SearchResult>> {
    if queries.is_empty() {
        return Ok(apply_filters(results, filters, result_limit));
    }

    let metadata_path = index_dir.join("metadata.db");
    let Ok(store) = crate::storage::metadata::MetadataStore::open(&metadata_path) else {
        return Ok(apply_filters(results, filters, result_limit));
    };
    let indexed_files = store.indexed_files()?;

    let mut per_query: Vec<std::vec::IntoIter<SearchResult>> = Vec::with_capacity(queries.len());
    let mut concept_candidates = Vec::new();
    for (query_index, query) in queries.iter().enumerate() {
        let exact = collect_exact_match_candidates(&store, &indexed_files, query, query_index)?;
        if exact.is_empty() {
            concept_candidates.extend(
                collect_concept_matched_files(&store, &indexed_files, query)?
                    .into_iter()
                    .map(|chunk| chunk.into_search_result(0.0)),
            );
        }
        per_query.push(exact.into_iter());
    }

    // Interleave candidates across queries (round-robin) so a high-cardinality
    // first subquery cannot exhaust the result limit before later subqueries
    // contribute. Per-query ordering is preserved.
    let mut supplemental = Vec::new();
    loop {
        let mut progressed = false;
        for candidates in &mut per_query {
            if let Some(candidate) = candidates.next() {
                supplemental.push(candidate);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    if supplemental.is_empty() && concept_candidates.is_empty() {
        return Ok(apply_filters(results, filters, result_limit));
    }

    // Exact matches enter in front; concept matches join at the pool tail so
    // they compete on ranking signals instead of displacing fused results.
    // Ranking runs here too: the fused pool must pass the same scoring step
    // the single-query path applies, scored per subquery so one subquery's
    // exact match cannot crowd out another's (issue #121).
    let mut merged = merge_exact_matches(supplemental, results);
    append_new_candidates(&mut merged, concept_candidates);
    let ranked = apply_query_ranking_multi_query(queries, merged, RankingStage::Initial, filters);
    Ok(apply_filters(ranked, filters, result_limit))
}

#[derive(Debug)]
pub(crate) struct ExactMatchCandidate {
    order: ExactMatchOrder,
    result: SearchResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ExactMatchOrder {
    query_index: usize,
    match_kind: u8,
    exact_priority: (u8, u8, u8, u8),
    path_depth: usize,
    line_start: u32,
}

pub(crate) fn collect_exact_match_candidates(
    store: &crate::storage::metadata::MetadataStore,
    indexed_files: &[String],
    query: &str,
    query_index: usize,
) -> Result<Vec<SearchResult>> {
    let mut candidates = Vec::new();

    // Bare filename queries ("handler.py") are unambiguous direct lookups, so
    // they bypass the path-weighted gate that prose queries must pass.
    if let Some(filename) = extract_exact_filename(query)
        .filter(|_| is_path_weighted_query(query) || query.split_whitespace().count() == 1)
    {
        let mut matching_files: Vec<String> = indexed_files
            .iter()
            .filter(|path| file_name(path).eq_ignore_ascii_case(&filename))
            .cloned()
            .collect();
        matching_files.sort_by(|a, b| path_depth(a).cmp(&path_depth(b)).then(a.cmp(b)));

        for file_path in matching_files.into_iter().take(20) {
            for chunk in store.get_chunks_by_file(&file_path)? {
                candidates.push(ExactMatchCandidate {
                    order: ExactMatchOrder {
                        query_index,
                        match_kind: 0,
                        exact_priority: (0, 0, 0, 0),
                        path_depth: path_depth(&chunk.file_path),
                        line_start: chunk.line_start,
                    },
                    result: chunk.into_search_result(0.0),
                });
            }
        }
    }

    if let Some(identifier_case) = extract_exact_identifier_case(query).as_deref() {
        let mut chunks = store.get_chunks_by_symbol_name_case_sensitive(identifier_case)?;
        let identifier = identifier_case.to_ascii_lowercase();
        let mut fallback_chunks = store.get_chunks_by_symbol_name(&identifier)?;
        fallback_chunks.retain(|chunk| chunk.symbol_name.as_deref() != Some(identifier_case));
        if uppercase_identifier_query(identifier_case) {
            fallback_chunks.retain(|chunk| {
                !matches!(
                    chunk.symbol_type,
                    Some(SymbolType::Method | SymbolType::Function | SymbolType::Module)
                )
            });
        }
        chunks.extend(fallback_chunks);

        // Scan for definition chunks in files whose stem matches the identifier.
        // This catches definitions that weren't found by symbol name lookup,
        // e.g. when the file is sessions.py and the query is "Session".
        let seen_files: HashSet<String> = chunks.iter().map(|c| c.file_path.clone()).collect();
        let stem_chunks =
            collect_stem_matched_definitions(store, indexed_files, &identifier, &seen_files)?;
        chunks.extend(stem_chunks);

        for chunk in chunks {
            let order = ExactMatchOrder {
                query_index,
                match_kind: 1,
                exact_priority: exact_match_priority(query, identifier_case, &chunk),
                path_depth: path_depth(&chunk.file_path),
                line_start: chunk.line_start,
            };
            candidates.push(ExactMatchCandidate {
                order,
                result: chunk.into_search_result(0.0),
            });
        }
    }

    candidates.sort_by(|left, right| {
        left.order
            .cmp(&right.order)
            .then(left.result.file_path.cmp(&right.result.file_path))
    });

    Ok(candidates
        .into_iter()
        .map(|candidate| candidate.result)
        .collect())
}

/// Find definition chunks from files whose stem matches the identifier.
///
/// When searching for "Session", this finds files like `sessions.py`,
/// `session.rs`, etc. and returns their definition chunks. Only scans
/// definition-type symbols to avoid pulling in noisy mentions.
pub(crate) fn collect_stem_matched_definitions(
    store: &crate::storage::metadata::MetadataStore,
    indexed_files: &[String],
    identifier: &str,
    already_seen: &HashSet<String>,
) -> Result<Vec<crate::types::Chunk>> {
    let identifier_lower = identifier.to_ascii_lowercase();
    let mut results = Vec::new();

    for file_path in indexed_files {
        if already_seen.contains(file_path.as_str()) {
            continue;
        }
        let fname = file_name(file_path).to_ascii_lowercase();
        let stem = file_stem(&fname);

        // Match stem to identifier: exact, plural, or prefix overlap.
        let is_stem_match = stem == identifier_lower
            || stem.strip_suffix('s') == Some(&identifier_lower)
            || identifier_lower.strip_suffix('s') == Some(stem)
            || (stem.len() >= 4 && identifier_lower.starts_with(stem))
            || (identifier_lower.len() >= 4 && stem.starts_with(&identifier_lower));

        if !is_stem_match {
            continue;
        }

        let chunks = store.get_chunks_by_file(file_path)?;
        for chunk in chunks {
            if is_definition_chunk(&chunk) {
                results.push(chunk);
            }
        }
    }

    Ok(results)
}

/// Find files whose stem matches a query keyword for NL queries.
///
/// When searching "configuration loading", this finds `config.py` because
/// "config" shares a prefix with "configuration". Only returns definition
/// chunks to avoid noise. Limited to short queries to avoid false positives.
pub(crate) fn collect_concept_matched_files(
    store: &crate::storage::metadata::MetadataStore,
    indexed_files: &[String],
    query: &str,
) -> Result<Vec<crate::types::Chunk>> {
    let keywords: Vec<String> = query
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|w| w.len() >= 4 && !is_concept_stopword(w))
        .collect();

    if keywords.is_empty() || keywords.len() > 5 {
        return Ok(Vec::new());
    }

    let mut matched_files = Vec::new();

    for file_path in indexed_files {
        let fname = file_name(file_path).to_ascii_lowercase();
        let stem = file_stem(&fname);
        if stem.len() < 4 {
            continue;
        }

        // Match file stem to query keywords: exact, singular/plural, or
        // prefix overlap. Both the stem and keyword must be non-stopwords
        // and the shorter must be at least 5 chars.
        if is_concept_stopword(stem) {
            continue;
        }
        let keyword_match = keywords.iter().any(|kw| {
            stem == kw
                || stem.strip_suffix('s') == Some(kw.as_str())
                || kw.strip_suffix('s') == Some(stem)
                || (stem.len() >= 5 && kw.starts_with(stem))
                || (kw.len() >= 5 && stem.starts_with(kw.as_str()))
        });

        if keyword_match {
            matched_files.push(file_path.clone());
        }
    }

    // Sort by path depth (prefer shallower files) and limit to avoid noise.
    matched_files.sort_by(|a, b| path_depth(a).cmp(&path_depth(b)).then(a.cmp(b)));
    matched_files.truncate(3);

    let mut results = Vec::new();
    for file_path in &matched_files {
        let chunks = store.get_chunks_by_file(file_path)?;
        let mut contributed = 0;
        for chunk in chunks {
            // Only inject definition chunks to avoid flooding results with
            // every line of a concept-matched file.
            if is_definition_chunk(&chunk) {
                results.push(chunk);
                contributed += 1;
                // Cap per-file contribution: one file must not fill the pool.
                if contributed >= CONCEPT_CHUNKS_PER_FILE {
                    break;
                }
            }
        }
    }

    Ok(results)
}

/// Words too common to be useful for file-stem matching.
pub(crate) fn is_concept_stopword(word: &str) -> bool {
    matches!(
        word,
        "error"
            | "errors"
            | "type"
            | "types"
            | "data"
            | "file"
            | "files"
            | "code"
            | "test"
            | "tests"
            | "util"
            | "utils"
            | "help"
            | "helper"
            | "helpers"
            | "base"
            | "core"
            | "main"
            | "init"
            | "index"
            | "model"
            | "models"
            | "form"
            | "format"
            | "value"
            | "values"
            | "path"
            | "paths"
            | "node"
            | "item"
            | "items"
            | "work"
            | "with"
            | "from"
            | "that"
            | "this"
            | "what"
            | "when"
            | "does"
            | "have"
            | "been"
            | "into"
            | "about"
            | "through"
            | "between"
            | "inside"
    )
}

pub(crate) fn merge_exact_matches(
    supplemental: Vec<SearchResult>,
    results: Vec<SearchResult>,
) -> Vec<SearchResult> {
    let mut merged = Vec::with_capacity(supplemental.len() + results.len());
    let mut seen = HashSet::new();

    for result in supplemental.into_iter().chain(results) {
        if seen.insert(result_key(&result)) {
            merged.push(result);
        }
    }

    merged
}

/// Append candidates at the pool tail, skipping duplicates of entries already
/// in the pool. Appended candidates get the lowest base ranks, so they only
/// surface when ranking signals (stem/keyword/definition matches) justify it.
pub(crate) fn append_new_candidates(pool: &mut Vec<SearchResult>, candidates: Vec<SearchResult>) {
    let mut seen: HashSet<_> = pool.iter().map(result_key).collect();
    for candidate in candidates {
        if seen.insert(result_key(&candidate)) {
            pool.push(candidate);
        }
    }
}

pub(crate) fn extract_exact_filename(query: &str) -> Option<String> {
    query
        .split_whitespace()
        .map(trim_query_token)
        .filter(|token| !token.is_empty())
        .find(|token| looks_like_filename(token))
        .map(|token| file_name(token).to_ascii_lowercase())
}

pub(crate) fn extract_exact_identifier_case(query: &str) -> Option<String> {
    let single_token_query = query.split_whitespace().count() == 1;
    query
        .split_whitespace()
        .map(trim_query_token)
        .filter(|token| !token.is_empty())
        .find(|token| {
            (!looks_like_filename(token) || contains_namespace_separator(token))
                && (looks_like_compound_identifier(token) || single_token_query)
        })
        .map(ToString::to_string)
}

fn contains_namespace_separator(token: &str) -> bool {
    if token.contains("::") || token.contains("->") || token.contains('\\') {
        return true;
    }

    let segments: Vec<_> = token.split('.').collect();
    segments.len() >= 3
        || (segments.len() == 2
            && segments[1]
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase()))
}

pub(crate) fn query_mentions_implementation(query: &str) -> bool {
    let lower = query.to_ascii_lowercase();
    lower.contains("implement")
        || lower.contains("registration")
        || lower.contains("mounted")
        || lower.contains("mounting")
}

pub(crate) fn exact_match_priority(
    query: &str,
    identifier_case: &str,
    chunk: &Chunk,
) -> (u8, u8, u8, u8) {
    let exact_case = u8::from(chunk.symbol_name.as_deref() != Some(identifier_case));
    let implementation_rank =
        if query_mentions_implementation(query) && chunk_looks_like_impl(chunk) {
            0
        } else {
            1
        };
    let visibility_rank = u8::from(!content_declares_public_symbol(&chunk.content));
    let type_mismatch_rank = if identifier_case
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
        && matches!(
            chunk.symbol_type,
            Some(SymbolType::Method | SymbolType::Function)
        )
        && chunk.symbol_name.as_deref() != Some(identifier_case)
    {
        1
    } else {
        0
    };

    (
        exact_case,
        implementation_rank,
        visibility_rank,
        type_mismatch_rank,
    )
}

pub(crate) fn chunk_looks_like_impl(chunk: &Chunk) -> bool {
    chunk
        .symbol_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("impl"))
        || content_starts_with_impl(&chunk.content)
}

/// Definition-type symbols eligible for exact-match injection. Intentionally
/// excludes `Method` (unlike ranking's definition predicate).
pub(crate) fn is_definition_chunk(chunk: &Chunk) -> bool {
    matches!(
        chunk.symbol_type,
        Some(
            SymbolType::Class
                | SymbolType::Struct
                | SymbolType::Trait
                | SymbolType::Interface
                | SymbolType::Enum
                | SymbolType::Function
                | SymbolType::Module
        )
    )
}

pub(crate) fn uppercase_identifier_query(identifier: &str) -> bool {
    identifier
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_chunk(symbol_name: &str, symbol_type: SymbolType) -> Chunk {
        Chunk {
            id: format!("src/{symbol_name}:0"),
            file_path: "src/session.rs".to_string(),
            line_start: 1,
            line_end: 4,
            content: format!("pub struct {symbol_name} {{}}"),
            language: crate::types::Language::Rust,
            symbol_type: Some(symbol_type),
            symbol_name: Some(symbol_name.to_string()),
            part_index: None,
        }
    }

    #[test]
    fn extracts_lowercase_single_word_symbols() {
        assert_eq!(
            extract_exact_identifier_case("authenticate"),
            Some("authenticate".to_string())
        );
        assert_eq!(extract_exact_identifier_case("Cargo.toml"), None);
    }

    #[test]
    fn extracts_scope_qualified_symbols_with_dots() {
        assert_eq!(
            extract_exact_identifier_case("std::io::Error"),
            Some("std::io::Error".to_string())
        );
    }

    #[test]
    fn symbol_identity_does_not_match_a_longer_name() {
        let store = MetadataStore::open_in_memory().unwrap();
        let mut chunk = test_chunk("SessionStore", SymbolType::Struct);
        chunk.file_path = "src/other.rs".to_string();
        store.insert_chunks(&[chunk]).unwrap();

        let files = store.indexed_files().unwrap();
        let results = collect_exact_match_candidates(&store, &files, "Session", 0).unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn exact_name_augmentation_reaches_split_symbol() {
        use crate::config::VeraConfig;
        use crate::embedding::test_helpers::MockProvider;
        use crate::indexing::index_repository;

        async fn index_repo(files: &[(&str, &str)]) -> tempfile::TempDir {
            let dir = tempfile::TempDir::new().unwrap();
            for (path, content) in files {
                let abs = dir.path().join(path);
                if let Some(parent) = abs.parent() {
                    std::fs::create_dir_all(parent).unwrap();
                }
                std::fs::write(abs, content).unwrap();
            }
            let provider = MockProvider::new(8);
            let config = VeraConfig::default();
            index_repository(dir.path(), &provider, &config, "mock-model")
                .await
                .unwrap();
            dir
        }

        let mut lines = vec!["export const MixerConsole: React.FC = () => {".to_string()];
        for i in 0..210 {
            lines.push(format!("  const line{i} = {i};"));
        }
        lines.push("  return null;".to_string());
        lines.push("};".to_string());
        let content = lines.join("\n");
        let dir = index_repo(&[("src/mixer.tsx", &content)]).await;
        let index_dir = crate::indexing::index_dir(dir.path());
        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        let files = store.indexed_files().unwrap();
        let results = collect_exact_match_candidates(&store, &files, "MixerConsole", 0).unwrap();
        assert!(
            !results.is_empty(),
            "exact-name augmentation must reach split symbol"
        );
        assert!(
            results
                .iter()
                .all(|r| r.symbol_name.as_deref() == Some("MixerConsole"))
        );
        // Top results include split parts.
        let top = &results[0];
        assert!(top.part_index.is_some(), "split part must carry part_index");
        assert_eq!(top.symbol_name.as_deref(), Some("MixerConsole"));
    }
}
