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

/// Follow-up symbols for the next hop: deduped, generic noise names skipped,
/// capped at five, in first-appearance order of the ranked initial results.
/// A `HashSet` round-trip here used to make the choice process-random whenever
/// more than five distinct symbols qualified, so two runs of the identical
/// query could search different second hops and return different results.
fn follow_up_symbols_from(initial_results: &[SearchResult]) -> Vec<String> {
    let mut seen = HashSet::new();
    initial_results
        .iter()
        .filter_map(|r| r.symbol_name.clone())
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            // Skip generic names that would produce noisy results.
            !matches!(
                lower.as_str(),
                "main" | "new" | "default" | "test" | "init" | "run" | "setup"
            )
        })
        .filter(|name| seen.insert(name.clone()))
        .take(5)
        .collect()
}

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

    // Extract symbol names from initial results for follow-up queries, keeping
    // ranked first-appearance order so the same query picks the same symbols
    // in every process.
    let follow_up_symbols = follow_up_symbols_from(&initial_results);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn result_with_symbol(name: &str) -> SearchResult {
        SearchResult {
            file_path: format!("src/{name}.rs"),
            line_start: 1,
            line_end: 10,
            content: format!("fn {name}() {{}}"),
            language: crate::types::Language::Rust,
            score: 1.0,
            symbol_name: Some(name.to_string()),
            symbol_type: None,
        }
    }

    /// More than five qualifying symbols must select the first five in ranked
    /// order, not a process-random subset (the old HashSet round-trip).
    #[test]
    fn follow_up_symbols_take_first_five_in_ranked_order() {
        let results: Vec<_> = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
            .iter()
            .map(|n| result_with_symbol(n))
            .collect();

        assert_eq!(
            follow_up_symbols_from(&results),
            vec!["s1", "s2", "s3", "s4", "s5"]
        );
    }

    /// Duplicates and generic noise names are dropped without disturbing the
    /// order of the survivors.
    #[test]
    fn follow_up_symbols_dedupe_and_skip_generic_names() {
        let mut results = vec![
            result_with_symbol("verify_token"),
            result_with_symbol("main"),
            result_with_symbol("session_store"),
            result_with_symbol("verify_token"),
        ];
        // `main` sits between the two keepers: removing it must not reorder
        // what is left.
        results.insert(2, result_with_symbol("init"));

        assert_eq!(
            follow_up_symbols_from(&results),
            vec!["verify_token", "session_store"]
        );
    }

    /// A fixture with five or fewer qualifying symbols passes through intact.
    #[test]
    fn follow_up_symbols_keep_everything_at_or_under_the_cap() {
        let results: Vec<_> = ["alpha", "beta"]
            .iter()
            .map(|n| result_with_symbol(n))
            .collect();

        assert_eq!(follow_up_symbols_from(&results), vec!["alpha", "beta"]);
    }
}
