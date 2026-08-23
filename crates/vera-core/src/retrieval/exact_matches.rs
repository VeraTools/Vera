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
    content_declares_public_symbol, content_starts_with_impl, looks_like_compound_identifier,
    looks_like_filename, path_depth, result_key, trim_query_token,
};
use crate::retrieval::ranking::{
    RankingStage, apply_query_ranking_with_filters, is_path_weighted_query,
};
use crate::storage::metadata::MetadataStore;
use crate::types::{Chunk, SearchFilters, SearchResult, SymbolType};

pub(crate) fn augment_exact_match_candidates(
    index_dir: &Path,
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let metadata_path = index_dir.join("metadata.db");
    let Ok(store) = MetadataStore::open_existing(&metadata_path) else {
        return Ok(apply_query_ranking_with_filters(
            query, results, stage, filters,
        ));
    };
    augment_exact_match_candidates_with_store(&store, query, results, stage, filters)
}

pub(crate) fn augment_exact_match_candidates_with_store(
    store: &MetadataStore,
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let supplemental = collect_exact_match_candidates(store, query, 0)?;
    if supplemental.is_empty() {
        return Ok(apply_query_ranking_with_filters(
            query, results, stage, filters,
        ));
    }

    let merged = merge_exact_matches(supplemental, results);

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

    let mut per_query: Vec<std::vec::IntoIter<SearchResult>> = Vec::with_capacity(queries.len());
    for (query_index, query) in queries.iter().enumerate() {
        per_query.push(collect_exact_match_candidates(&store, query, query_index)?.into_iter());
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

    if supplemental.is_empty() {
        return Ok(apply_filters(results, filters, result_limit));
    }

    Ok(apply_filters(
        merge_exact_matches(supplemental, results),
        filters,
        result_limit,
    ))
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
    query: &str,
    query_index: usize,
) -> Result<Vec<SearchResult>> {
    let mut candidates = Vec::new();

    // Bare filename queries ("handler.py") are unambiguous direct lookups, so
    // they bypass the path-weighted gate that prose queries must pass.
    if let Some(filename) = extract_exact_filename(query)
        .filter(|_| is_path_weighted_query(query) || query.split_whitespace().count() == 1)
    {
        let mut matching_files: Vec<String> = store
            .indexed_files()?
            .into_iter()
            .filter(|path| file_name(path).eq_ignore_ascii_case(&filename))
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
        let stem_chunks = collect_stem_matched_definitions(store, &identifier, &seen_files)?;
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

    // For queries with no filename or identifier match, scan for files whose
    // stem matches a query keyword. This catches NL queries like
    // "configuration loading" → config.py, "testing client" → testing.py.
    if candidates.is_empty() {
        let concept_chunks = collect_concept_matched_files(store, query)?;
        for chunk in concept_chunks {
            candidates.push(ExactMatchCandidate {
                order: ExactMatchOrder {
                    query_index,
                    match_kind: 2,
                    exact_priority: (0, 0, 0, 0),
                    path_depth: path_depth(&chunk.file_path),
                    line_start: chunk.line_start,
                },
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
    identifier: &str,
    already_seen: &HashSet<String>,
) -> Result<Vec<crate::types::Chunk>> {
    let identifier_lower = identifier.to_ascii_lowercase();
    let mut results = Vec::new();

    let all_files = store.indexed_files()?;
    for file_path in &all_files {
        if already_seen.contains(file_path.as_str()) {
            continue;
        }
        let fname = file_name(file_path).to_ascii_lowercase();
        let stem = fname.rsplit_once('.').map(|(s, _)| s).unwrap_or(&fname);

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

    let all_files = store.indexed_files()?;
    let mut matched_files = Vec::new();

    for file_path in &all_files {
        let fname = file_name(file_path).to_ascii_lowercase();
        let stem = fname.rsplit_once('.').map(|(s, _)| s).unwrap_or(&fname);
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
        for chunk in chunks {
            // Only inject definition chunks to avoid flooding results with
            // every line of a concept-matched file.
            if is_definition_chunk(&chunk) {
                results.push(chunk);
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
            (!looks_like_filename(token) || token.contains("::"))
                && (looks_like_compound_identifier(token) || single_token_query)
        })
        .map(ToString::to_string)
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
    let visibility_rank = u8::from(!chunk_is_public_symbol(chunk));
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

pub(crate) fn chunk_is_public_symbol(chunk: &Chunk) -> bool {
    content_declares_public_symbol(&chunk.content)
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
}
