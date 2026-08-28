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
        let (symbol_name, symbol_type, part_index) = symbol_for_line(Some(&chunks), caller.line);
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
            part_index,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn mixer_content() -> String {
        let mut lines = vec!["import React from 'react';".to_string()];
        lines.push("export function TrackBadge() { return null; }".to_string());
        lines.push("export function TransportBar() { return null; }".to_string());
        lines.push("export const MixerConsole: React.FC = () => {".to_string());
        for i in 0..210 {
            lines.push(format!("  const line{i} = {i};"));
        }
        lines.push("  return null;".to_string());
        lines.push("};".to_string());
        lines.push("export function AudioWorkspace() {".to_string());
        lines.push("  return MixerConsole({});".to_string());
        lines.push("}".to_string());
        lines.join("\n")
    }

    #[tokio::test]
    async fn references_resolve_used_split_symbol() {
        let content = mixer_content();
        let dir = index_repo(&[("src/mixer.tsx", &content)]).await;
        let index_dir = crate::indexing::index_dir(dir.path());

        let store =
            crate::storage::metadata::MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        let chunks = store.get_chunks_by_symbol_name("MixerConsole").unwrap();
        assert!(chunks.len() >= 2, "MixerConsole must be split");

        let callers =
            search_callers(&index_dir, "MixerConsole", 10, &SearchFilters::default()).unwrap();
        assert_eq!(
            callers.len(),
            1,
            "used split symbol should have exactly one caller, got {callers:?}"
        );
        assert_eq!(callers[0].file_path, "src/mixer.tsx");
        // Caller should be attributed to AudioWorkspace with correct part handling.
        assert_eq!(
            callers[0].symbol_name.as_deref(),
            Some("AudioWorkspace"),
            "call site should be attributed to AudioWorkspace"
        );
        // part_index for caller (AudioWorkspace unsplit) should be None.
        assert_eq!(callers[0].part_index, None);
    }

    #[tokio::test]
    async fn dead_code_omits_used_split_symbol_and_reports_unused_once() {
        let mixer = mixer_content();
        // Add UnusedPanel split and used.
        let mut unused_lines = vec!["export const UnusedPanel: React.FC = () => {".to_string()];
        for i in 0..210 {
            unused_lines.push(format!("  const u{i} = {i};"));
        }
        unused_lines.push("  return null;".to_string());
        unused_lines.push("};".to_string());
        let unused_content = unused_lines.join("\n");

        let dir = index_repo(&[
            ("src/mixer.tsx", &mixer),
            ("src/unused_panel.tsx", &unused_content),
        ])
        .await;
        let index_dir = crate::indexing::index_dir(dir.path());
        let store =
            crate::storage::metadata::MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        let dead = store.find_dead_symbols().unwrap();
        let mixer_entries: Vec<_> = dead
            .iter()
            .filter(|d| d.symbol_name == "MixerConsole")
            .collect();
        assert!(
            mixer_entries.is_empty(),
            "used split symbol MixerConsole must be omitted from dead-code, got {mixer_entries:?}"
        );
        let _track_entries: Vec<_> = dead
            .iter()
            .filter(|d| d.symbol_name == "TrackBadge")
            .collect();
        // TrackBadge is not called anywhere, but it's exported; check expected?
        // Instead focus on UnusedPanel which is split and unused.
        let unused_entries: Vec<_> = dead
            .iter()
            .filter(|d| d.symbol_name == "UnusedPanel")
            .collect();
        assert_eq!(
            unused_entries.len(),
            1,
            "unused split symbol must appear exactly once, got {unused_entries:?} (total dead: {dead:?})"
        );
        // Ensure grouping is by (symbol,file) not part.
        let dup_check: Vec<_> = store.get_chunks_by_symbol_name("UnusedPanel").unwrap();
        assert!(
            dup_check.len() >= 2,
            "UnusedPanel should be split into >=2 rows"
        );
    }
}
