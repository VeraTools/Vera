//! Exact explicit type-relation retrieval helpers.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

use crate::corpus::{ContentClass, classify_content};
use crate::parsing::signatures;
use crate::path_containment::canonical_project_root;
use crate::retrieval::file_scan::{
    allows_class, language_for_path, line_context_snippet, smallest_symbol_chunk_for_line,
    symbol_for_line,
};
use crate::storage::metadata::MetadataStore;
use crate::types::{SearchFilters, SearchResult, SymbolType};

pub fn search_explicit_implementations(
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
    let root_dir = crate::discovery::open_root_dir(&repo_root)?;
    let max_file_size_bytes = super::configured_max_file_size_bytes(&store);
    let symbol = super::structural::normalize_impl_target(symbol);
    let relations = store.find_type_relations(&symbol)?;
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for relation in relations {
        if results.len() >= limit {
            break;
        }

        let language = language_for_path(&relation.file_path);
        if !filters.matches_file(&relation.file_path, language) {
            continue;
        }

        let content = match crate::discovery::read_source_lossy_capped(
            &root_dir,
            Path::new(&relation.file_path),
            max_file_size_bytes,
        ) {
            Ok(content) => content,
            Err(e) => {
                tracing::debug!("skipping {}: {e}", relation.file_path);
                continue;
            }
        };
        let class = classify_content(&relation.file_path, language, &content);
        if !allows_class(filters, class) {
            continue;
        }
        if matches!(filters.include_generated, Some(false))
            && matches!(class, ContentClass::Generated)
        {
            continue;
        }

        let chunks = store.get_chunks_by_file(&relation.file_path)?;
        let line = relation.line;
        let mut line_start = line;
        let mut line_end = line;
        let mut snippet = None;
        let mut symbol_type = None;
        let mut enclosing_part_index: Option<u32> = None;

        if let Some(chunk) = smallest_symbol_chunk_for_line(&chunks, line) {
            line_start = chunk.line_start;
            line_end = chunk.line_end;
            snippet = Some(signatures::extract_signature_for_path(
                &chunk.content,
                language,
                &chunk.file_path,
            ));
            symbol_type = match chunk.symbol_type {
                Some(SymbolType::Block) | None => None,
                other => other,
            };
            enclosing_part_index = chunk.part_index;
        }

        let content = snippet.unwrap_or_else(|| {
            let (snippet, start, end) = line_context_snippet(&content, line, 2);
            line_start = start;
            line_end = end;
            snippet
        });

        let key = format!(
            "{}:{}:{}:{}",
            relation.file_path,
            line_start,
            line_end,
            relation.owner.to_ascii_lowercase()
        );
        if !seen.insert(key) {
            continue;
        }

        let fallback_info = symbol_for_line(Some(&chunks), line);
        let fallback_symbol = fallback_info.1;
        if enclosing_part_index.is_none() {
            enclosing_part_index = fallback_info.2;
        }
        let final_symbol_type = symbol_type.or(match fallback_symbol {
            Some(SymbolType::Block) | None => None,
            other => other,
        });
        if !filters.matches_symbol_type(final_symbol_type) {
            continue;
        }

        results.push(SearchResult {
            file_path: relation.file_path,
            line_start,
            line_end,
            content,
            language,
            score: 1.0,
            symbol_name: Some(relation.owner),
            symbol_type: final_symbol_type,
            part_index: enclosing_part_index,
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

    #[tokio::test]
    async fn explicit_implementations_keep_working_for_split_symbols() {
        // Loader target is normal, but owner Repo is large enough to split.
        let mut repo_lines = vec![
            "pub trait Loader {}".to_string(),
            "pub struct Repo {".to_string(),
        ];
        for i in 0..210 {
            repo_lines.push(format!("  field{i}: i32,"));
        }
        repo_lines.push("}".to_string());
        repo_lines.push("impl Loader for Repo {".to_string());
        repo_lines.push("  fn load(&self) {}".to_string());
        repo_lines.push("}".to_string());
        let repo_content = repo_lines.join("\n");

        let dir = index_repo(&[("src/repo.rs", &repo_content)]).await;
        let index_dir = crate::indexing::index_dir(dir.path());
        let store =
            crate::storage::metadata::MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        let repo_chunks = store.get_chunks_by_symbol_name("Repo").unwrap();
        // Repo should be split if large; but impl chunk may be associated
        // with Repo's enclosing chunk after indexing.
        assert!(!repo_chunks.is_empty());

        let results =
            search_explicit_implementations(&index_dir, "Loader", 10, &SearchFilters::default())
                .unwrap();
        assert!(
            results
                .iter()
                .any(|r| r.symbol_name.as_deref() == Some("Repo")),
            "impl owner Repo should be found via bare name, got {results:?}"
        );
        // Ensure no per-part duplication for same impl line.
        let repo_impls: Vec<_> = results
            .iter()
            .filter(|r| r.symbol_name.as_deref() == Some("Repo"))
            .collect();
        assert_eq!(
            repo_impls.len(),
            1,
            "split owner should not duplicate impl entries"
        );
        assert!(
            !repo_impls[0]
                .symbol_name
                .as_deref()
                .unwrap()
                .contains(" (part ")
        );
    }
}
