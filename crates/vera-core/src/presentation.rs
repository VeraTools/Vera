//! Output presentation helpers shared by the CLI and MCP frontends.

use serde::Serialize;
use std::borrow::Cow;

use crate::types::{SearchResult, SymbolType};

/// Compact JSON representation that drops low-signal fields (`score`, `language`)
/// and omits null optional fields. This is the default for AI agent consumption.
#[derive(Serialize)]
pub struct CompactResult<'a> {
    pub file_path: &'a str,
    pub line_start: u32,
    pub line_end: u32,
    pub content: Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_type: Option<&'a SymbolType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_index: Option<u32>,
}

impl<'a> CompactResult<'a> {
    pub fn from_search_result(r: &'a SearchResult) -> Self {
        Self {
            file_path: &r.file_path,
            line_start: r.line_start,
            line_end: r.line_end,
            content: Cow::Borrowed(r.content.as_str()),
            symbol_name: r.symbol_name.as_deref(),
            symbol_type: r.symbol_type.as_ref(),
            part_index: r.part_index,
        }
    }
}

/// Truncate `content` to fit within `allowed` bytes, breaking at a line boundary.
pub fn truncate_to_budget(content: &str, allowed: usize) -> Cow<'_, str> {
    if content.len() <= allowed {
        return Cow::Borrowed(content);
    }
    const TRUNCATION_MARKER: &str = "\n[...truncated]";
    let can_include_marker = allowed >= TRUNCATION_MARKER.len();
    let content_budget = if can_include_marker {
        allowed - TRUNCATION_MARKER.len()
    } else {
        allowed
    };
    let end = content
        .char_indices()
        .take_while(|(i, c)| *i + c.len_utf8() <= content_budget)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);

    if !can_include_marker {
        return Cow::Owned(content[..end].to_string());
    }

    let break_at = content[..end].rfind('\n').unwrap_or(end);
    let mut truncated = content[..break_at].to_string();
    truncated.push_str(TRUNCATION_MARKER);
    Cow::Owned(truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_to_budget_short_passthrough() {
        let short = "short content";
        let result = truncate_to_budget(short, 1000);
        assert_eq!(result, short);
    }

    #[test]
    fn truncate_to_budget_long_truncates() {
        let long = "line1\nline2\nline3\n".repeat(100);
        let result = truncate_to_budget(&long, 100);
        assert!(result.len() <= 100);
        assert!(result.ends_with("[...truncated]"));
    }

    #[test]
    fn truncate_to_budget_clamps_small_budgets() {
        let result = truncate_to_budget("a long line", 5);
        assert!(result.len() <= 5);
    }

    #[test]
    fn json_carries_bare_name_plus_part_index_and_display_shows_parts() {
        use crate::types::{Language, SearchResult, SymbolType, display_symbol_name};
        let bare = "MixerConsole";
        for part in 1..=3 {
            let r = SearchResult {
                file_path: "src/mixer.tsx".to_string(),
                line_start: part * 10,
                line_end: part * 10 + 5,
                content: "content".to_string(),
                language: Language::TypeScript,
                score: 1.0,
                symbol_name: Some(bare.to_string()),
                symbol_type: Some(SymbolType::Function),
                part_index: Some(part),
            };
            let cr = CompactResult::from_search_result(&r);
            assert_eq!(cr.symbol_name, Some(bare));
            assert_eq!(cr.part_index, Some(part));
            let json = serde_json::to_string(&cr).unwrap();
            assert!(json.contains("\"symbol_name\":\"MixerConsole\""));
            assert!(json.contains(&format!("\"part_index\":{part}")));
            assert!(!json.contains(" (part "));

            let display = r.display_name().unwrap();
            assert_eq!(display, display_symbol_name(bare, Some(part)));
            assert!(display.contains(&format!("(part {part})")));
        }
        // Unsplit stays bare with no part_index.
        let unsplit = SearchResult {
            file_path: "src/small.rs".to_string(),
            line_start: 1,
            line_end: 2,
            content: "content".to_string(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some("SmallFn".to_string()),
            symbol_type: Some(SymbolType::Function),
            part_index: None,
        };
        let cr = CompactResult::from_search_result(&unsplit);
        assert_eq!(cr.symbol_name, Some("SmallFn"));
        assert_eq!(cr.part_index, None);
        let json = serde_json::to_string(&cr).unwrap();
        assert!(!json.contains("part_index"));
        assert_eq!(unsplit.display_name().as_deref(), Some("SmallFn"));

        // Literal " (part N)" verbatim unsplit case.
        let lit = SearchResult {
            file_path: "src/lit.rs".to_string(),
            line_start: 1,
            line_end: 1,
            content: "content".to_string(),
            language: Language::Rust,
            score: 1.0,
            symbol_name: Some("foo (part 2)".to_string()),
            symbol_type: Some(SymbolType::Function),
            part_index: None,
        };
        assert_eq!(lit.display_name().as_deref(), Some("foo (part 2)"));
        let cr = CompactResult::from_search_result(&lit);
        assert_eq!(cr.symbol_name, Some("foo (part 2)"));
        assert_eq!(cr.part_index, None);
    }
}
