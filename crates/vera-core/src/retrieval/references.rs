//! Exact call-graph retrieval helpers.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

use crate::corpus::{ContentClass, classify_content};
use crate::path_containment::canonical_project_root;
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
    search_callers_through(index_dir, symbol, None, limit, filters)
}

/// Search call sites, optionally limited to calls made through one receiver.
///
/// `receiver` is matched against the text before the dot at the call site, so
/// `state.add_url_rule()` and `app.add_url_rule()` can be told apart even
/// though both are stored under the callee name `add_url_rule`.
pub fn search_callers_through(
    index_dir: &Path,
    symbol: &str,
    receiver: Option<&str>,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    if limit == 0 {
        anyhow::bail!("limit must be greater than zero");
    }

    let metadata_path = index_dir.join("metadata.db");
    let store = MetadataStore::open(&metadata_path)?;
    let repo_root = canonical_project_root(index_dir)?;
    let root_dir = crate::discovery::open_root_dir(&repo_root)?;
    let max_file_size_bytes = super::configured_max_file_size_bytes(&store);
    let callers = store.find_callers_through(symbol, receiver)?;
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

        let content = match crate::discovery::read_source_lossy_capped(
            &root_dir,
            Path::new(&caller.file_path),
            max_file_size_bytes,
        ) {
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
