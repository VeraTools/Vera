//! Iterative (multi-hop) search: runs an initial semantic search, extracts
//! symbol names from the top results, and performs follow-up searches to
//! find related code. Merges and deduplicates all results.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

use crate::config::VeraConfig;
use crate::types::{SearchFilters, SearchResult};

use super::query_utils::result_key;
use super::search_service::{SearchContext, SearchTimings};

/// Run an iterative (multi-hop) search.
///
/// 1. Execute the initial query via `context.search`.
/// 2. Extract unique symbol names from the top results.
/// 3. Run follow-up searches for each extracted symbol.
/// 4. Merge and deduplicate, preserving the original result order first.
#[allow(clippy::too_many_arguments)]
pub async fn execute_iterative_search_with_context(
    context: &SearchContext,
    index_dir: &Path,
    query: &str,
    intent: Option<&str>,
    config: &VeraConfig,
    filters: &SearchFilters,
    result_limit: usize,
    hops: usize,
) -> Result<(Vec<SearchResult>, SearchTimings)> {
    let fetch_per_hop = result_limit;

    let (initial_results, timings) = context
        .search(index_dir, query, intent, config, filters, fetch_per_hop)
        .await?;

    if hops == 0 || initial_results.is_empty() {
        return Ok((initial_results, timings));
    }

    let mut seen = HashSet::new();
    let mut merged: Vec<SearchResult> = Vec::new();

    for r in &initial_results {
        let key = result_key(r);
        if seen.insert(key) {
            merged.push(r.clone());
        }
    }

    // Extract symbol names from initial results for follow-up queries.
    let follow_up_symbols = select_follow_up_symbols(&initial_results);

    for symbol in &follow_up_symbols {
        let (hop_results, _) = context
            .search(
                index_dir,
                symbol,
                intent,
                config,
                filters,
                fetch_per_hop / 2,
            )
            .await?;

        for r in hop_results {
            let key = result_key(&r);
            if seen.insert(key) {
                merged.push(r);
            }
        }
    }

    merged.truncate(result_limit);
    Ok((merged, timings))
}

fn select_follow_up_symbols(initial_results: &[SearchResult]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut symbols = Vec::new();

    for result in initial_results {
        let Some(name) = result.symbol_name.as_deref() else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        // Skip generic names that would produce noisy results.
        if matches!(
            lower.as_str(),
            "main" | "new" | "default" | "test" | "init" | "run" | "setup"
        ) {
            continue;
        }
        if seen.insert(name.to_string()) {
            symbols.push(name.to_string());
            if symbols.len() == 5 {
                break;
            }
        }
    }

    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    fn result_with_symbol(symbol_name: &str) -> SearchResult {
        SearchResult {
            file_path: format!("{symbol_name}.rs"),
            line_start: 1,
            line_end: 1,
            content: String::new(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some(symbol_name.to_string()),
            symbol_type: None,
            part_index: None,
        }
    }

    #[test]
    fn follow_up_symbol_selection_is_deterministic_and_ranked() {
        let results = [
            result_with_symbol("first"),
            result_with_symbol("second"),
            result_with_symbol("first"),
            result_with_symbol("main"),
            result_with_symbol("third"),
            result_with_symbol("fourth"),
            result_with_symbol("fifth"),
            result_with_symbol("sixth"),
        ];

        let first = select_follow_up_symbols(&results);
        let second = select_follow_up_symbols(&results);

        assert_eq!(first, second);
        assert_eq!(first, vec!["first", "second", "third", "fourth", "fifth"]);
    }
}
