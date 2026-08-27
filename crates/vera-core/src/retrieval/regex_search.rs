//! Regex pattern search over indexed files.
//!
//! Searches file contents using a regex pattern, returning matches with
//! file paths, line numbers, and surrounding context. Operates on the
//! indexed file list from the metadata store, respecting the same file
//! scope as semantic search.

use std::path::Path;

use anyhow::Result;
use regex::RegexBuilder;

use crate::corpus::{ContentClass, classify_content};
use crate::path_containment::canonical_project_root;
use crate::retrieval::file_scan::{
    allows_class, bounded_byte_snippet, language_for_path, sort_files_by_scan_priority,
    symbol_for_line,
};
use crate::retrieval::ranking::{RankingStage, apply_query_ranking_with_filters};
use crate::storage::metadata::MetadataStore;
use crate::types::{Chunk, SearchFilters, SearchResult};

/// Search indexed files for a regex pattern.
///
/// Reads the file list from the metadata store, then greps each file.
/// Returns results as `SearchResult` for compatibility with the rest of
/// the pipeline.
pub fn search_regex(
    index_dir: &Path,
    pattern: &str,
    limit: usize,
    case_insensitive: bool,
    context_lines: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let regex = RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .map_err(|e| anyhow::anyhow!("Invalid regex pattern: {e}"))?;

    let metadata_path = index_dir.join("metadata.db");
    let store = MetadataStore::open(&metadata_path)?;
    let mut files = store.indexed_files()?;
    sort_files_by_scan_priority(&mut files, filters);

    // Resolve the project root (index_dir is .vera/, parent is project root).
    let project_root = canonical_project_root(index_dir)?;
    let root_dir = crate::discovery::open_root_dir(&project_root)?;
    let max_file_size_bytes = super::configured_max_file_size_bytes(&store);

    let mut results = Vec::new();

    for file_rel in &files {
        if results.len() >= limit {
            break;
        }

        let language = language_for_path(file_rel);
        if !filters.matches_file(file_rel, language) {
            continue;
        }

        let file_chunks = if filters.symbol_type.is_some() {
            let chunks = store.get_chunks_by_file(file_rel)?;
            if !chunks
                .iter()
                .any(|chunk| filters.matches_symbol_type(chunk.symbol_type))
            {
                continue;
            }
            Some(chunks)
        } else {
            None
        };

        let content = match crate::discovery::read_source_lossy_capped(
            &root_dir,
            Path::new(file_rel),
            max_file_size_bytes,
        ) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let class = classify_content(file_rel, language, &content);

        if !allows_class(filters, class) {
            continue;
        }

        if matches!(filters.include_generated, Some(false))
            && matches!(class, ContentClass::Generated)
        {
            continue;
        }

        if matches!(class, ContentClass::Generated) {
            collect_minified_matches(
                &mut results,
                &regex,
                file_rel,
                &content,
                limit,
                filters,
                file_chunks.as_deref(),
            );
            continue;
        }

        let lines: Vec<&str> = content.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if results.len() >= limit {
                break;
            }
            if !regex.is_match(line) {
                continue;
            }

            let ctx_start = i.saturating_sub(context_lines);
            let ctx_end = (i + context_lines + 1).min(lines.len());
            let snippet = lines[ctx_start..ctx_end].join("\n");
            let (symbol_name, symbol_type) =
                symbol_for_line(file_chunks.as_deref(), (i + 1) as u32);

            let result = SearchResult {
                file_path: file_rel.clone(),
                line_start: (ctx_start + 1) as u32,
                line_end: ctx_end as u32,
                content: snippet,
                score: 1.0,
                symbol_name,
                symbol_type,
                language,
            };

            if !filters.matches_symbol_type(symbol_type) {
                continue;
            }

            results.push(result);
        }
    }

    let mut results =
        apply_query_ranking_with_filters(pattern, results, RankingStage::Initial, filters);
    results.truncate(limit);
    Ok(results)
}

fn collect_minified_matches(
    results: &mut Vec<SearchResult>,
    regex: &regex::Regex,
    file_rel: &str,
    content: &str,
    limit: usize,
    filters: &SearchFilters,
    file_chunks: Option<&[Chunk]>,
) {
    let language = language_for_path(file_rel);

    for found in regex.find_iter(content) {
        if results.len() >= limit {
            break;
        }
        let (snippet, line_start, line_end) =
            bounded_byte_snippet(content, found.start(), found.end(), 220);
        let (symbol_name, symbol_type) = symbol_for_line(file_chunks, line_start);
        let result = SearchResult {
            file_path: file_rel.to_string(),
            line_start,
            line_end,
            content: snippet,
            score: 1.0,
            symbol_name,
            symbol_type,
            language,
        };

        if !filters.matches_symbol_type(symbol_type) {
            continue;
        }

        results.push(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::metadata::MetadataStore;
    use crate::types::{Chunk, Language, SymbolType};

    #[test]
    fn invalid_regex_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let index_dir = tmp.path().join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();
        let result = search_regex(
            &index_dir,
            "[invalid",
            10,
            false,
            2,
            &SearchFilters::default(),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid regex"));
    }

    #[test]
    fn source_files_are_scanned_before_docs() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(index_dir.join("docs")).unwrap_or(());
        std::fs::create_dir_all(repo_root.join("src")).unwrap();
        std::fs::create_dir_all(repo_root.join("docs")).unwrap();
        std::fs::write(
            repo_root.join("src/hotkeys.ts"),
            "export const keybind = true;\n",
        )
        .unwrap();
        std::fs::write(repo_root.join("docs/hotkeys.md"), "keybind docs\n").unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "src:0".to_string(),
                    file_path: "src/hotkeys.ts".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: "export const keybind = true;".to_string(),
                    language: Language::TypeScript,
                    symbol_type: None,
                    symbol_name: None,
                },
                Chunk {
                    id: "docs:0".to_string(),
                    file_path: "docs/hotkeys.md".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: "keybind docs".to_string(),
                    language: Language::Markdown,
                    symbol_type: None,
                    symbol_name: None,
                },
            ])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "keybind",
            5,
            false,
            0,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(results[0].file_path, "src/hotkeys.ts");
    }

    #[test]
    fn minified_matches_return_bounded_snippets() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(repo_root.join("dist")).unwrap();
        std::fs::create_dir_all(&index_dir).unwrap();

        let content = format!(
            "var a=0;{}targetSymbol();{}",
            "x=1;".repeat(400),
            "y=2;".repeat(400)
        );
        std::fs::write(repo_root.join("dist/app.min.js"), &content).unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "dist:0".to_string(),
                file_path: "dist/app.min.js".to_string(),
                line_start: 1,
                line_end: 1,
                content: content.clone(),
                language: Language::JavaScript,
                symbol_type: None,
                symbol_name: None,
            }])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "targetSymbol",
            5,
            false,
            0,
            &SearchFilters {
                include_generated: Some(true),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].content.len() < content.len());
        assert!(results[0].content.contains("targetSymbol"));
    }

    #[test]
    fn content_class_generated_files_survive_runtime_scope_filtering() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(&index_dir).unwrap();

        let content = format!("const targetSymbol=1;{}", "a=1;".repeat(400));
        std::fs::write(repo_root.join("app.min.js"), &content).unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[Chunk {
                id: "app:0".to_string(),
                file_path: "app.min.js".to_string(),
                line_start: 1,
                line_end: 1,
                content: content.clone(),
                language: Language::JavaScript,
                symbol_type: None,
                symbol_name: None,
            }])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "targetSymbol",
            5,
            false,
            0,
            &SearchFilters {
                scope: Some(crate::types::SearchScope::Runtime),
                include_generated: Some(true),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "app.min.js");
        assert!(results[0].content.contains("targetSymbol"));
    }

    #[test]
    fn language_and_path_filters_apply_before_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(repo_root.join("src")).unwrap();
        std::fs::create_dir_all(repo_root.join("tests")).unwrap();
        std::fs::create_dir_all(&index_dir).unwrap();

        std::fs::write(
            repo_root.join("src/main.rs"),
            "const NEEDLE: &str = \"needle\";\n",
        )
        .unwrap();
        std::fs::write(repo_root.join("tests/helper.py"), "needle = 'needle'\n").unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "src:0".to_string(),
                    file_path: "src/main.rs".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: "const NEEDLE: &str = \"needle\";".to_string(),
                    language: Language::Rust,
                    symbol_type: None,
                    symbol_name: None,
                },
                Chunk {
                    id: "tests:0".to_string(),
                    file_path: "tests/helper.py".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: "needle = 'needle'".to_string(),
                    language: Language::Python,
                    symbol_type: None,
                    symbol_name: None,
                },
            ])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "needle",
            1,
            false,
            0,
            &SearchFilters {
                language: Some("python".to_string()),
                path_glob: vec!["tests/**".to_string()],
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "tests/helper.py");
        assert_eq!(results[0].language, Language::Python);
    }

    #[test]
    fn symbol_type_filter_matches_enclosing_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(repo_root.join("src")).unwrap();
        std::fs::create_dir_all(&index_dir).unwrap();

        let content = "struct TokenConfig {\n    token: String,\n}\n\nfn build_token() {\n    let token = String::new();\n}\n";
        std::fs::write(repo_root.join("src/lib.rs"), content).unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "struct:0".to_string(),
                    file_path: "src/lib.rs".to_string(),
                    line_start: 1,
                    line_end: 3,
                    content: "struct TokenConfig {\n    token: String,\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Struct),
                    symbol_name: Some("TokenConfig".to_string()),
                },
                Chunk {
                    id: "fn:0".to_string(),
                    file_path: "src/lib.rs".to_string(),
                    line_start: 5,
                    line_end: 7,
                    content: "fn build_token() {\n    let token = String::new();\n}".to_string(),
                    language: Language::Rust,
                    symbol_type: Some(SymbolType::Function),
                    symbol_name: Some("build_token".to_string()),
                },
            ])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "token",
            10,
            true,
            0,
            &SearchFilters {
                symbol_type: Some("function".to_string()),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|result| result.symbol_type == Some(SymbolType::Function))
        );
        assert!(
            results
                .iter()
                .all(|result| result.symbol_name.as_deref() == Some("build_token"))
        );
        assert_eq!(results[0].file_path, "src/lib.rs");
    }

    #[test]
    fn stored_paths_outside_the_project_root_are_not_read() {
        let tmp = tempfile::tempdir().unwrap();
        // The canary sits next to the project root, so both escape shapes reach
        // it without the test touching anything outside the temp dir.
        let canary = tmp.path().join("canary.txt");
        std::fs::write(&canary, "CANARY_SECRET lives outside the repository\n").unwrap();

        let repo_root = tmp.path().join("repo");
        let index_dir = repo_root.join(".vera");
        std::fs::create_dir_all(repo_root.join("src")).unwrap();
        std::fs::create_dir_all(&index_dir).unwrap();
        std::fs::write(
            repo_root.join("src/lib.rs"),
            "fn read() { CANARY_SECRET; }\n",
        )
        .unwrap();

        let store = MetadataStore::open(&index_dir.join("metadata.db")).unwrap();
        store
            .insert_chunks(&[
                Chunk {
                    id: "inside:0".to_string(),
                    file_path: "src/lib.rs".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: "fn read() { CANARY_SECRET; }".to_string(),
                    language: Language::Rust,
                    symbol_type: None,
                    symbol_name: None,
                },
                Chunk {
                    id: "absolute:0".to_string(),
                    file_path: canary.to_str().unwrap().to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: String::new(),
                    language: Language::Unknown,
                    symbol_type: None,
                    symbol_name: None,
                },
                Chunk {
                    id: "traversal:0".to_string(),
                    file_path: "../canary.txt".to_string(),
                    line_start: 1,
                    line_end: 1,
                    content: String::new(),
                    language: Language::Unknown,
                    symbol_type: None,
                    symbol_name: None,
                },
            ])
            .unwrap();

        let results = search_regex(
            &index_dir,
            "CANARY_SECRET",
            10,
            false,
            0,
            &SearchFilters::default(),
        )
        .unwrap();

        assert!(
            results
                .iter()
                .all(|result| !result.content.contains("outside the repository")),
            "a stored path escaped the project root: {results:?}"
        );
        assert_eq!(
            results
                .iter()
                .map(|result| result.file_path.as_str())
                .collect::<Vec<_>>(),
            vec!["src/lib.rs"]
        );
    }
}
