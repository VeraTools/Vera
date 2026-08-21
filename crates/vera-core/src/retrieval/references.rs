//! Exact call-graph retrieval helpers.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

use crate::corpus::{ContentClass, classify_content};
use crate::path_containment::{canonical_project_root, resolve_indexed_path};
use crate::retrieval::file_scan::{
    allows_class, language_for_path, line_context_snippet, symbol_for_line,
};
use crate::storage::metadata::MetadataStore;
use crate::types::{SearchFilters, SearchResult};

/// Search exact call sites of `symbol` using the persisted call graph.
pub fn search_callers(
    index_dir: &Path,
    symbol: &str,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    if limit == 0 {
        anyhow::bail!("limit must be greater than zero");
    }

    let metadata_path = index_dir.join("metadata.db");
    let store = MetadataStore::open(&metadata_path)?;
    let repo_root = canonical_project_root(index_dir)?;
    let callers = store.find_callers(symbol)?;
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for caller in callers {
        if results.len() >= limit {
            break;
        }

        let language = language_for_path(&caller.file_path);
        if !filters.matches_file(&caller.file_path, language) {
            continue;
        }

        let Some(file_abs) = resolve_indexed_path(&repo_root, &caller.file_path) else {
            continue;
        };
        let content = match crate::discovery::read_source_lossy(&file_abs) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let class = classify_content(&caller.file_path, language, &content);
        if !allows_class(filters, class) {
            continue;
        }
        if matches!(filters.include_generated, Some(false))
            && matches!(class, ContentClass::Generated)
        {
            continue;
        }

        let chunks = store.get_chunks_by_file(&caller.file_path)?;
        let (snippet, line_start, line_end) = line_context_snippet(&content, caller.line, 2);
        let (symbol_name, symbol_type) = symbol_for_line(Some(&chunks), caller.line);
        if !filters.matches_symbol_type(symbol_type) {
            continue;
        }

        let key = format!("{}:{}:{}", caller.file_path, line_start, line_end);
        if !seen.insert(key) {
            continue;
        }

        results.push(SearchResult {
            file_path: caller.file_path,
            line_start,
            line_end,
            content: snippet,
            language,
            score: 1.0,
            symbol_name,
            symbol_type,
        });
    }

    Ok(results)
}
