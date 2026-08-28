//! Shared query-parsing and path utilities used across ranking and search.

/// Count directory separators in a path.
pub(crate) fn path_depth(path: &str) -> usize {
    path.matches('/').count() + path.matches('\\').count()
}

/// Return a filename without its final extension.
pub(crate) fn file_stem(filename: &str) -> &str {
    filename
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(filename)
}

/// Strip non-identifier punctuation from the edges of a query token.
pub(crate) fn trim_query_token(token: &str) -> &str {
    token.trim_matches(|ch: char| {
        !ch.is_ascii_alphanumeric() && !matches!(ch, '.' | '_' | '-' | '/')
    })
}

/// Check whether a token looks like a compound identifier (snake_case, CamelCase, or `::` path).
pub(crate) fn looks_like_compound_identifier(token: &str) -> bool {
    token.contains('_') || token.contains("::") || token.chars().any(|ch| ch.is_ascii_uppercase())
}

/// Check whether a (lowercased) token looks like a filename.
pub(crate) fn looks_like_filename(token: &str) -> bool {
    matches!(
        token,
        "dockerfile" | "makefile" | "cmakelists.txt" | "nginx.conf"
    ) || token.contains('.')
}

/// Check whether the first non-empty content line declares a public symbol
/// (pub/export/public/class/interface).
pub(crate) fn content_declares_public_symbol(content: &str) -> bool {
    content.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(
            trimmed.starts_with("pub ")
                || trimmed.starts_with("export ")
                || trimmed.starts_with("public ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("interface "),
        )
    }) == Some(true)
}

/// Check whether the first non-empty content line starts an impl block.
pub(crate) fn content_starts_with_impl(content: &str) -> bool {
    content
        .lines()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim_start().starts_with("impl "))
}

/// Generate a unique key for a search result to detect overlaps.
///
/// Uses file_path + line_start + line_end as a composite key, since
/// SearchResult doesn't carry the chunk ID but these fields uniquely
/// identify a chunk within the index.
pub(crate) fn result_key(result: &crate::types::SearchResult) -> String {
    format!(
        "{}:{}:{}",
        result.file_path, result.line_start, result.line_end
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Language, SearchResult};

    #[test]
    fn result_key_format() {
        let r = SearchResult {
            file_path: "src/main.rs".to_string(),
            line_start: 10,
            line_end: 20,
            content: String::new(),
            score: 1.0,
            symbol_name: None,
            symbol_type: None,
            language: Language::Rust,
            part_index: None,
        };
        assert_eq!(result_key(&r), "src/main.rs:10:20");
    }
}
