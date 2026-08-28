//! Experimental structural graph augmentation for retrieval.
//!
//! The graph lookups are intentionally supplemental. A failed lookup is
//! treated as an empty result so older or partially-built indexes do not make
//! ordinary search fail.

use std::collections::HashSet;
use std::path::Path;

use tracing::debug;

use crate::retrieval::references::search_callers;
use crate::retrieval::type_relations::search_explicit_implementations;
use crate::types::{SearchFilters, SearchResult, SymbolType};

const MAX_SEEDS: usize = 5;
const MAX_LOOKUP_RESULTS: usize = 5;
const MAX_AUGMENTED_CANDIDATES: usize = 20;
const RRF_K: f64 = 60.0;

const CALLER_DENYLIST: &[&str] = &[
    "new",
    "default",
    "fmt",
    "display",
    "debug",
    "clone",
    "drop",
    "to_string",
    "from",
    "into",
    "len",
    "get",
    "set",
    "main",
    "run",
    "test",
    "init",
    "new_instance",
];

/// Expand a fused pool with caller and explicit implementation results.
///
/// The synchronous graph APIs perform bounded SQLite and source-file reads.
/// They are called directly here because the fixed seed and per-lookup caps
/// bound the amount of supplemental work. There is deliberately no timeout:
/// cancelling a blocking lookup would require detached work and could leave
/// an index read running after the search has returned.
pub(crate) fn augment_pool(
    index_dir: &Path,
    pool: &mut Vec<SearchResult>,
    filters: &SearchFilters,
) -> usize {
    let seed_symbols = select_seed_symbols(pool);
    if seed_symbols.is_empty() {
        return 0;
    }

    let mut candidates = Vec::with_capacity(seed_symbols.len() * MAX_LOOKUP_RESULTS * 2);
    for symbol in seed_symbols {
        if !caller_lookup_is_skipped(&symbol) {
            match search_callers(index_dir, &symbol, MAX_LOOKUP_RESULTS, filters) {
                Ok(results) => candidates.extend(results.into_iter().take(MAX_LOOKUP_RESULTS)),
                Err(error) => debug!(
                    symbol = %symbol,
                    error = %error,
                    "graph callers lookup failed; skipping lookup"
                ),
            }
        }

        match search_explicit_implementations(index_dir, &symbol, MAX_LOOKUP_RESULTS, filters) {
            Ok(results) => candidates.extend(results.into_iter().take(MAX_LOOKUP_RESULTS)),
            Err(error) => debug!(
                symbol = %symbol,
                error = %error,
                "graph implementations lookup failed; skipping lookup"
            ),
        }
    }

    inject_augmented_candidates(pool, candidates)
}

/// Select up to five distinct eligible symbols from the highest-scoring pool
/// results. The original pool order is not changed.
fn select_seed_symbols(pool: &[SearchResult]) -> Vec<String> {
    let mut ranked: Vec<&SearchResult> = pool.iter().collect();
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut seen = HashSet::new();
    ranked
        .into_iter()
        .filter_map(|result| {
            let symbol_type = result.symbol_type?;
            if !is_seed_symbol_type(symbol_type) {
                return None;
            }

            let symbol = base_symbol_name(result.symbol_name.as_deref()?);
            if symbol.is_empty() {
                return None;
            }
            seen.insert(symbol.to_string()).then(|| symbol.to_string())
        })
        .take(MAX_SEEDS)
        .collect()
}

fn base_symbol_name(symbol: &str) -> &str {
    symbol
}

fn is_seed_symbol_type(symbol_type: SymbolType) -> bool {
    matches!(
        symbol_type,
        SymbolType::Function
            | SymbolType::Method
            | SymbolType::Struct
            | SymbolType::Enum
            | SymbolType::Class
            | SymbolType::Trait
            | SymbolType::Interface
    )
}

fn caller_lookup_is_skipped(symbol: &str) -> bool {
    symbol.chars().count() <= 2
        || CALLER_DENYLIST
            .iter()
            .any(|denied| symbol.eq_ignore_ascii_case(denied))
}

/// Append graph candidates after deduplicating them against the organic pool.
///
/// This pure step is kept separate from the SQLite lookups so score, dedup,
/// and cap behavior can be tested without constructing a full index fixture.
fn inject_augmented_candidates(
    pool: &mut Vec<SearchResult>,
    candidates: impl IntoIterator<Item = SearchResult>,
) -> usize {
    let pool_len = pool.len();
    let sentinel_score = 1.0 / (RRF_K + pool_len as f64 + 1.0);
    let mut seen: HashSet<(String, u32)> = pool
        .iter()
        .map(|result| (result.file_path.clone(), result.line_start))
        .collect();
    let mut added = 0;

    for mut candidate in candidates {
        if added >= MAX_AUGMENTED_CANDIDATES {
            break;
        }

        let key = (candidate.file_path.clone(), candidate.line_start);
        if !seen.insert(key) {
            continue;
        }

        candidate.score = sentinel_score;
        pool.push(candidate);
        added += 1;
    }

    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    fn result(
        file_path: &str,
        line_start: u32,
        score: f64,
        symbol_name: Option<&str>,
        symbol_type: Option<SymbolType>,
    ) -> SearchResult {
        SearchResult {
            file_path: file_path.to_string(),
            line_start,
            line_end: line_start + 2,
            content: format!("content for {file_path}"),
            language: Language::Rust,
            score,
            symbol_name: symbol_name.map(str::to_string),
            symbol_type,
            part_index: None,
        }
    }

    #[test]
    fn selects_only_eligible_symbols_from_top_five() {
        let pool = vec![
            result("block.rs", 1, 1.0, None, Some(SymbolType::Block)),
            result(
                "module.rs",
                1,
                0.9,
                Some("module"),
                Some(SymbolType::Module),
            ),
            result(
                "function.rs",
                1,
                0.8,
                Some("function_name"),
                Some(SymbolType::Function),
            ),
            result(
                "trait.rs",
                1,
                0.7,
                Some("TraitName"),
                Some(SymbolType::Trait),
            ),
            result(
                "class.rs",
                1,
                0.6,
                Some("ClassName"),
                Some(SymbolType::Class),
            ),
            result(
                "fifth-eligible.rs",
                1,
                0.5,
                Some("fifth_eligible"),
                Some(SymbolType::Function),
            ),
        ];

        assert_eq!(
            select_seed_symbols(&pool),
            vec![
                "function_name".to_string(),
                "TraitName".to_string(),
                "ClassName".to_string(),
                "fifth_eligible".to_string(),
            ]
        );
    }

    #[test]
    fn caller_relation_adds_definition_chunk_with_sentinel_score() {
        use crate::parsing::references::RawReference;
        use crate::storage::metadata::MetadataStore;
        use crate::types::Chunk;
        use tempfile::tempdir;

        let repo_dir = tempdir().unwrap();
        let index_dir = repo_dir.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        std::fs::write(
            repo_dir.path().join("caller.rs"),
            "fn caller() {\n    target();\n}\n",
        )
        .unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "seed:0".to_string(),
                    file_path: "seed.rs".to_string(),
                    line_start: 1,
                    line_end: 3,
                    content: "fn target() {}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("target".to_string()),
                    part_index: None,
                },
                Chunk {
                    id: "caller:0".to_string(),
                    file_path: "caller.rs".to_string(),
                    line_start: 1,
                    line_end: 3,
                    content: "fn caller() {\n    target();\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("caller".to_string()),
                    part_index: None,
                },
            ])
            .unwrap();
        let references = [RawReference {
            callee: "target".to_string(),
            caller: Some("caller".to_string()),
            qualifier: None,
            line: 2,
        }];
        store
            .insert_parse_artifacts_batch_borrowed(&[("caller.rs", &references)], &[])
            .unwrap();

        let mut pool = vec![result(
            "seed.rs",
            1,
            0.5,
            Some("target"),
            Some(SymbolType::Function),
        )];

        assert_eq!(
            augment_pool(&index_dir, &mut pool, &SearchFilters::default()),
            1
        );
        assert_eq!(pool.len(), 2);
        assert_eq!(pool[1].file_path, "caller.rs");
        assert_eq!(pool[1].line_start, 1);
        assert_eq!(pool[1].symbol_name.as_deref(), Some("caller"));
        assert_eq!(pool[1].score, 1.0 / 62.0);
    }

    #[test]
    fn injects_with_sentinel_and_deduplicates_by_file_and_start_line() {
        let mut pool = vec![result(
            "organic.rs",
            10,
            0.5,
            Some("seed"),
            Some(SymbolType::Function),
        )];
        let candidates = vec![
            result(
                "organic.rs",
                10,
                1.0,
                Some("seed"),
                Some(SymbolType::Function),
            ),
            result(
                "caller.rs",
                20,
                1.0,
                Some("caller"),
                Some(SymbolType::Function),
            ),
            result(
                "caller.rs",
                20,
                0.8,
                Some("caller"),
                Some(SymbolType::Function),
            ),
        ];

        assert_eq!(inject_augmented_candidates(&mut pool, candidates), 1);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool[0].score, 0.5);
        assert_eq!(pool[1].file_path, "caller.rs");
        assert_eq!(pool[1].score, 1.0 / 62.0);
    }

    #[test]
    fn caps_augmented_candidates_at_twenty() {
        let mut pool = vec![result(
            "seed.rs",
            1,
            1.0,
            Some("seed"),
            Some(SymbolType::Function),
        )];
        let candidates = (0..25).map(|index| {
            result(
                &format!("caller-{index}.rs"),
                1,
                1.0,
                Some("caller"),
                Some(SymbolType::Function),
            )
        });

        assert_eq!(inject_augmented_candidates(&mut pool, candidates), 20);
        assert_eq!(pool.len(), 21);
    }

    #[test]
    fn skips_callers_for_short_and_ubiquitous_names() {
        for symbol in ["new", "DEFAULT", "fmt", "x", "ab"] {
            assert!(
                caller_lookup_is_skipped(symbol),
                "{symbol} should be skipped"
            );
        }
        assert!(!caller_lookup_is_skipped("authenticate_user"));
    }

    #[test]
    fn normalizes_split_chunk_seed_names() {
        let pool = vec![
            SearchResult {
                file_path: "large.rs".to_string(),
                line_start: 1,
                line_end: 3,
                content: "content".to_string(),
                language: Language::Rust,
                score: 1.0,
                symbol_name: Some("LargeFunction".to_string()),
                symbol_type: Some(SymbolType::Function),
                part_index: Some(1),
            },
            SearchResult {
                file_path: "large.rs".to_string(),
                line_start: 20,
                line_end: 22,
                content: "content".to_string(),
                language: Language::Rust,
                score: 0.9,
                symbol_name: Some("LargeFunction".to_string()),
                symbol_type: Some(SymbolType::Function),
                part_index: Some(2),
            },
        ];

        assert_eq!(select_seed_symbols(&pool), vec!["LargeFunction"]);
    }
}
