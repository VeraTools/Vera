//! Shared text builders for semantic and lexical indexing.
//!
//! Vera stores raw chunks in metadata, but both the dense and lexical stages
//! benefit from a richer representation that includes path, symbol, and light
//! structural hints. Keep the heuristics simple and deterministic.

use std::collections::BTreeSet;

use crate::types::{Chunk, SymbolType};

const MAX_METADATA_VALUE_CHARS: usize = 240;
const MAX_METADATA_TOKEN_CHARS: usize = 80;

/// Build the structured text used for semantic embeddings.
///
/// When `max_bytes` is non-zero the code content is truncated at a line
/// boundary so the total output (metadata + code) stays within budget.
/// This prevents oversized chunks from exceeding the embedding model's
/// context window.
pub fn build_embedding_text(chunk: &Chunk) -> String {
    build_structured_text(chunk, 0)
}

/// Like [`build_embedding_text`] but caps the total output at `max_bytes`.
pub fn build_embedding_text_bounded(chunk: &Chunk, max_bytes: usize) -> String {
    build_structured_text(chunk, max_bytes)
}

/// Build the structured text indexed by BM25.
///
/// Includes all metadata from the embedding text plus split identifier tokens
/// so compound names like `parseConfig` match queries for "parse" or "config".
pub fn build_bm25_text(chunk: &Chunk) -> String {
    let mut text = build_structured_text(chunk, 0);
    let split_tokens = extract_split_identifiers(&chunk.content);
    if !split_tokens.is_empty() {
        text.push_str("\nTokens: ");
        text.push_str(&split_tokens);
    }
    text
}

fn build_structured_text(chunk: &Chunk, max_bytes: usize) -> String {
    let mut parts = Vec::new();
    let filename = file_name(&chunk.file_path);
    let path_tokens = normalize_path_tokens(&chunk.file_path);
    let signature = extract_signature(chunk);

    parts.push(format!("Language: {}", chunk.language));
    parts.push(format!("Path: {path_tokens}"));
    parts.push(format!("Filename: {filename}"));

    if let Some(symbol_line) = symbol_line(chunk) {
        parts.push(symbol_line);
    }

    if let Some(signature) = signature.as_deref() {
        parts.push(format!("Signature: {signature}"));
    }

    if let Some(summary) = extract_summary(chunk) {
        parts.push(format!("Summary: {summary}"));
    }

    let parameters = extract_parameters(signature.as_deref());
    if !parameters.is_empty() {
        parts.push(format!("Parameters: {}", parameters.join(", ")));
    }

    if let Some(return_type) = extract_return_type(signature.as_deref()) {
        parts.push(format!("Returns: {return_type}"));
    }

    let imports = extract_imports(chunk);
    if !imports.is_empty() {
        parts.push(format!("Imports: {}", imports.join(", ")));
    }

    let calls = extract_calls(chunk);
    if !calls.is_empty() {
        parts.push(format!("Calls: {}", calls.join(", ")));
    }

    let flow_hints = extract_flow_hints(chunk);
    if !flow_hints.is_empty() {
        parts.push(format!("Flow: {}", flow_hints.join(", ")));
    }

    let label = if chunk.language.is_document_like() {
        "Document"
    } else {
        "Code"
    };

    // Strip embedded data blobs (hex arrays, base85 fonts, numeric tables)
    // before applying the byte budget so the token budget is spent on
    // semantically meaningful code.
    let collapsed = collapse_data_lines(&chunk.content);

    // When a byte budget is set, truncate the code content so the total
    // (metadata header + code) fits. Metadata is kept intact since it's
    // small and high-value for retrieval; only the code body is trimmed.
    let content = if max_bytes > 0 {
        let header = parts.join("\n");
        // header + "\n" + "Code:\n" = overhead before the actual content
        let overhead = header.len() + 1 + label.len() + 2;
        let content_budget = max_bytes.saturating_sub(overhead);
        if content_budget == 0 {
            String::new()
        } else {
            truncate_at_line_boundary(&collapsed, content_budget)
        }
    } else {
        collapsed
    };

    parts.push(format!("{label}:\n{content}"));

    let structured = parts.join("\n");
    if max_bytes > 0 && structured.len() > max_bytes {
        truncate_at_line_boundary(&structured, max_bytes)
    } else {
        structured
    }
}

/// Truncate `text` to at most `max_bytes`, cutting at the last newline
/// boundary that fits. Returns the full string when it already fits.
fn truncate_at_line_boundary(text: &str, max_bytes: usize) -> String {
    if max_bytes == 0 || text.len() <= max_bytes {
        return text.to_string();
    }
    // Find the last newline at or before the byte budget.
    let floor = floor_char_boundary(text, max_bytes);
    let cut = &text[..floor];
    let end = cut.rfind('\n').unwrap_or(floor);
    text[..end].to_string()
}

fn symbol_line(chunk: &Chunk) -> Option<String> {
    let name = chunk.symbol_name.as_ref()?;
    match chunk.symbol_type {
        Some(SymbolType::Function) => Some(format!("Symbol: function {name}")),
        Some(SymbolType::Method) => Some(format!("Symbol: method {name}")),
        Some(SymbolType::Class) => Some(format!("Symbol: class {name}")),
        Some(SymbolType::Struct) => Some(format!("Symbol: struct {name}")),
        Some(SymbolType::Enum) => Some(format!("Symbol: enum {name}")),
        Some(SymbolType::Trait) => Some(format!("Symbol: trait {name}")),
        Some(SymbolType::Interface) => Some(format!("Symbol: interface {name}")),
        Some(SymbolType::TypeAlias) => Some(format!("Symbol: type {name}")),
        Some(SymbolType::Constant) => Some(format!("Symbol: constant {name}")),
        Some(SymbolType::Variable) => Some(format!("Symbol: variable {name}")),
        Some(SymbolType::Module) => Some(format!("Symbol: module {name}")),
        Some(SymbolType::Block) | None => Some(format!("Symbol: {name}")),
    }
}

pub(crate) fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

pub(crate) fn normalize_path_tokens(path: &str) -> String {
    let filename = file_name(path);
    let mut normalized = String::with_capacity(path.len() * 2);
    let chars: Vec<char> = path.chars().collect();

    for (idx, ch) in chars.iter().enumerate() {
        match ch {
            '/' | '\\' | '_' | '-' | '.' => {
                if !normalized.ends_with(' ') && !normalized.is_empty() {
                    normalized.push(' ');
                }
            }
            c if c.is_uppercase() => {
                if idx > 0 && chars[idx - 1].is_lowercase() && !normalized.ends_with(' ') {
                    normalized.push(' ');
                }
                normalized.push(c.to_ascii_lowercase());
            }
            c => normalized.push(c.to_ascii_lowercase()),
        }
    }

    let compact = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() {
        filename.to_string()
    } else if compact.contains(filename) {
        compact
    } else {
        format!("{compact} {filename}")
    }
}

fn extract_signature(chunk: &Chunk) -> Option<String> {
    if chunk.language.is_document_like() {
        return None;
    }

    let lines: Vec<&str> = chunk.content.lines().collect();
    let start = signature_start_index(chunk, &lines)?;

    let mut signature = String::new();
    let mut paren_balance = 0i32;

    for line in lines.iter().skip(start).take(3) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if !signature.is_empty() {
            signature.push(' ');
        }
        signature.push_str(trimmed);
        paren_balance += trimmed.matches('(').count() as i32;
        paren_balance -= trimmed.matches(')').count() as i32;
        if paren_balance <= 0
            && (trimmed.ends_with('{')
                || trimmed.ends_with(':')
                || trimmed.ends_with(';')
                || trimmed.contains("=>"))
        {
            break;
        }
    }

    let signature = signature.split_whitespace().collect::<Vec<_>>().join(" ");
    let signature = abbreviate_middle(&signature, MAX_METADATA_VALUE_CHARS);
    (!signature.is_empty()).then_some(signature)
}

fn extract_summary(chunk: &Chunk) -> Option<String> {
    if chunk.language.is_document_like() {
        let line = chunk
            .content
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())?;
        return Some(
            line.trim_start_matches('#')
                .trim_start_matches("//")
                .trim_start_matches('*')
                .trim()
                .to_string(),
        )
        .filter(|line| !line.is_empty())
        .map(|line| abbreviate_middle(&line, MAX_METADATA_VALUE_CHARS));
    }

    let lines: Vec<&str> = chunk.content.lines().collect();
    let signature_idx = signature_start_index(chunk, &lines)?;

    let mut summary_lines = Vec::new();
    let mut in_block_string = false;

    for line in lines.iter().skip(signature_idx + 1).take(6) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !summary_lines.is_empty() {
                break;
            }
            continue;
        }

        let is_triple_quote = trimmed.starts_with("\"\"\"")
            || trimmed.starts_with("'''")
            || trimmed.ends_with("\"\"\"")
            || trimmed.ends_with("'''");

        if in_block_string || is_triple_quote {
            let cleaned = trimmed.trim_matches('"').trim_matches('\'').trim();
            if !cleaned.is_empty() {
                summary_lines.push(cleaned.to_string());
            }
            if is_triple_quote {
                in_block_string = !in_block_string;
            }
            continue;
        }

        if looks_like_comment(trimmed) {
            summary_lines.push(strip_comment_markers(trimmed).to_string());
            continue;
        }

        break;
    }

    (!summary_lines.is_empty())
        .then(|| abbreviate_middle(&summary_lines.join(" "), MAX_METADATA_VALUE_CHARS))
}

fn extract_parameters(signature: Option<&str>) -> Vec<String> {
    let Some(signature) = signature else {
        return Vec::new();
    };
    let Some(open) = signature.find('(') else {
        return Vec::new();
    };
    let Some(close) = signature[open + 1..].find(')') else {
        return Vec::new();
    };
    let inner = &signature[open + 1..open + 1 + close];
    inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(clean_parameter_name)
        .map(|part| abbreviate_middle(&part, MAX_METADATA_TOKEN_CHARS))
        .filter(|part| !part.is_empty())
        .collect()
}

fn clean_parameter_name(parameter: &str) -> String {
    let before_default = parameter.split('=').next().unwrap_or(parameter).trim();
    let before_type = before_default
        .split(':')
        .next()
        .unwrap_or(before_default)
        .trim();
    let before_pattern = before_type
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim();
    before_pattern
        .rsplit_once(' ')
        .map(|(_, name)| name)
        .unwrap_or(before_pattern)
        .trim_matches(|c: char| matches!(c, ',' | ')' | '{' | '}'))
        .to_string()
}

fn extract_return_type(signature: Option<&str>) -> Option<String> {
    let signature = signature?;
    if let Some((_, return_type)) = signature.split_once("->") {
        return Some(
            return_type
                .trim()
                .trim_end_matches('{')
                .trim_end_matches(':')
                .trim_end_matches(';')
                .trim()
                .to_string(),
        )
        .filter(|rt| !rt.is_empty())
        .map(|rt| abbreviate_middle(&rt, MAX_METADATA_TOKEN_CHARS));
    }
    None
}

fn extract_imports(chunk: &Chunk) -> Vec<String> {
    let mut imports = BTreeSet::new();

    for line in chunk.content.lines() {
        let trimmed = line.trim();
        let remainder = if let Some(rest) = trimmed.strip_prefix("use ") {
            Some(rest)
        } else if let Some(rest) = trimmed.strip_prefix("import ") {
            Some(rest)
        } else if let Some(rest) = trimmed.strip_prefix("from ") {
            Some(rest)
        } else {
            trimmed.strip_prefix("#include ")
        };

        if let Some(rest) = remainder {
            let cleaned = rest
                .trim_end_matches(';')
                .trim_matches('"')
                .trim_matches('\'')
                .trim_matches('<')
                .trim_matches('>');
            let normalized = cleaned
                .split(['{', '}', '(', ')', ',', ';'])
                .flat_map(|segment| segment.split_whitespace())
                .filter(|segment| !segment.is_empty() && *segment != "as")
                .take(8)
                .collect::<Vec<_>>()
                .join(" ");
            let normalized = abbreviate_middle(&normalized, MAX_METADATA_TOKEN_CHARS);
            if !normalized.is_empty() {
                imports.insert(normalized);
            }
        }
    }

    imports.into_iter().collect()
}

fn extract_calls(chunk: &Chunk) -> Vec<String> {
    let mut calls = BTreeSet::new();
    let symbol_name = chunk.symbol_name.as_deref().unwrap_or_default();

    for line in chunk.content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || looks_like_comment(trimmed) {
            continue;
        }

        for (idx, ch) in trimmed.char_indices() {
            if ch != '(' || idx == 0 {
                continue;
            }

            let candidate = trimmed[..idx]
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | ':' | '.' | '!'))
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>();

            if candidate.is_empty() {
                continue;
            }

            let short = candidate
                .trim_end_matches('!')
                .rsplit([':', '.'])
                .next()
                .unwrap_or(candidate.as_str());

            if short.is_empty()
                || short == symbol_name
                || is_call_keyword(short)
                || short.chars().all(|c| c.is_ascii_uppercase())
            {
                continue;
            }

            let normalized = if candidate.contains('.') {
                short.to_string()
            } else {
                candidate.trim_end_matches('!').to_string()
            };
            let normalized = abbreviate_middle(&normalized, MAX_METADATA_TOKEN_CHARS);
            if !normalized.is_empty() {
                calls.insert(normalized);
            }
        }
    }

    calls.into_iter().take(10).collect()
}

fn abbreviate_middle(value: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= max_chars {
        return value.to_string();
    }

    if max_chars <= 3 {
        return chars.into_iter().take(max_chars).collect();
    }

    let head_len = (max_chars - 3) / 2;
    let tail_len = max_chars - 3 - head_len;
    let head: String = chars.iter().take(head_len).collect();
    let tail: String = chars[chars.len().saturating_sub(tail_len)..]
        .iter()
        .collect();
    format!("{head}...{tail}")
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn extract_flow_hints(chunk: &Chunk) -> Vec<&'static str> {
    let content = chunk.content.to_ascii_lowercase();
    let mut hints = Vec::new();

    if ["if ", " else", "match ", "switch ", "case "]
        .iter()
        .any(|needle| content.contains(needle))
    {
        hints.push("branches");
    }

    if ["for ", "while ", "loop ", ".iter(", ".map(", ".filter("]
        .iter()
        .any(|needle| content.contains(needle))
    {
        hints.push("loops");
    }

    if [
        "try", "catch", "except", "raise", "throw", "err(", "result<", "bail!", "panic!", "abort(",
    ]
    .iter()
    .any(|needle| content.contains(needle))
    {
        hints.push("error handling");
    }

    hints
}

fn signature_start_index(chunk: &Chunk, lines: &[&str]) -> Option<usize> {
    let symbol_name = chunk.symbol_name.as_deref();

    lines
        .iter()
        .position(|line| looks_like_signature_line(line.trim(), symbol_name))
        .or_else(|| {
            lines.iter().position(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty()
                    && !trimmed.starts_with('@')
                    && !looks_like_comment(trimmed)
                    && !looks_like_import(trimmed)
            })
        })
}

fn looks_like_signature_line(line: &str, symbol_name: Option<&str>) -> bool {
    if line.is_empty()
        || line.starts_with('@')
        || looks_like_comment(line)
        || looks_like_import(line)
    {
        return false;
    }

    if let Some(name) = symbol_name
        && line.contains(name)
    {
        return true;
    }

    [
        "fn ",
        "pub fn ",
        "async fn ",
        "pub async fn ",
        "def ",
        "class ",
        "struct ",
        "enum ",
        "trait ",
        "impl ",
        "interface ",
        "function ",
        "type ",
        "const ",
        "let ",
        "var ",
        "module ",
        "namespace ",
    ]
    .iter()
    .any(|prefix| line.starts_with(prefix))
}

fn looks_like_comment(line: &str) -> bool {
    line.starts_with("//")
        || line.starts_with("/*")
        || line.starts_with('*')
        || line.starts_with('#')
        || line.starts_with("--")
}

fn looks_like_import(line: &str) -> bool {
    line.starts_with("use ")
        || line.starts_with("import ")
        || line.starts_with("from ")
        || line.starts_with("#include ")
}

fn strip_comment_markers(line: &str) -> &str {
    line.trim_start_matches("//")
        .trim_start_matches("/*")
        .trim_start_matches('*')
        .trim_start_matches('#')
        .trim_start_matches("--")
        .trim()
}

/// Minimum consecutive data lines required before collapsing.
/// Prevents false positives on isolated unusual lines.
const MIN_DATA_RUN: usize = 4;

/// Collapse runs of data-dense lines (hex arrays, base85/base64 strings,
/// numeric lookup tables) into short placeholders. These lines waste
/// embedding tokens with zero semantic value for code search.
fn collapse_data_lines(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut result = Vec::with_capacity(lines.len());
    let mut i = 0;

    while i < lines.len() {
        if is_data_line(lines[i]) {
            let run_start = i;
            while i < lines.len() && is_data_line(lines[i]) {
                i += 1;
            }
            let run_len = i - run_start;
            if run_len >= MIN_DATA_RUN {
                result.push(format!("[{run_len} lines of data]"));
            } else {
                for item in lines.iter().take(i).skip(run_start) {
                    result.push(item.to_string());
                }
            }
        } else {
            result.push(lines[i].to_string());
            i += 1;
        }
    }

    result.join("\n")
}

/// A line is considered "data" if it's at least 40 characters and contains
/// at most one identifier (4+ consecutive alphabetic characters). The
/// threshold of 4 avoids false positives from hex literals (`0xFF` produces
/// a 3-char alpha run `xFF`). This catches hex arrays, encoded strings
/// (base85/base64 blobs), and numeric tables while leaving normal code
/// untouched.
fn is_data_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 40 {
        return false;
    }
    let mut identifiers = 0usize;
    let mut alpha_run = 0usize;
    for c in trimmed.chars() {
        if c.is_alphabetic() {
            alpha_run += 1;
        } else {
            if alpha_run >= 4 {
                identifiers += 1;
            }
            alpha_run = 0;
        }
    }
    if alpha_run >= 4 {
        identifiers += 1;
    }
    identifiers <= 1
}

fn is_call_keyword(candidate: &str) -> bool {
    matches!(
        candidate,
        "if" | "for"
            | "while"
            | "match"
            | "switch"
            | "return"
            | "fn"
            | "def"
            | "class"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "catch"
            | "except"
            | "try"
            | "use"
            | "import"
            | "from"
            | "loop"
            | "macro_rules"
            | "where"
            | "new"
    )
}

/// Extract identifiers from code, split camelCase/PascalCase/snake_case into
/// sub-tokens, and return them as a single space-separated string.
///
/// Example: code containing `parseConfig` and `get_user_by_id` produces
/// "parse config get user by id".
fn extract_split_identifiers(content: &str) -> String {
    use std::collections::BTreeSet;

    let mut sub_tokens = BTreeSet::new();

    for token in content
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|t| t.len() >= 4)
    {
        let parts = split_identifier(token);
        if parts.len() >= 2 {
            for part in &parts {
                if part.len() >= 2 {
                    sub_tokens.insert(part.to_ascii_lowercase());
                }
            }
        }
    }

    sub_tokens.into_iter().collect::<Vec<_>>().join(" ")
}

/// Split a single identifier into sub-tokens via camelCase/PascalCase or
/// snake_case boundaries. Returns the parts (lowercased). Returns a single
/// element if no split is possible.
///
/// Examples:
///   "parseConfig"      -> ["parse", "config"]
///   "XMLParser"        -> ["xml", "parser"]
///   "get_user_by_id"   -> ["get", "user", "by", "id"]
///   "simple"           -> ["simple"]
pub(crate) fn split_identifier(token: &str) -> Vec<String> {
    // snake_case: split on underscores.
    if token.contains('_') {
        return token
            .split('_')
            .filter(|p| !p.is_empty())
            .map(|p| p.to_ascii_lowercase())
            .collect();
    }

    // camelCase / PascalCase: split on case transitions.
    let chars: Vec<char> = token.chars().collect();
    let mut parts = Vec::new();
    let mut start = 0;

    for i in 1..chars.len() {
        let split = if chars[i].is_uppercase() && chars[i - 1].is_lowercase() {
            // aB -> a | B
            true
        } else if i + 1 < chars.len()
            && chars[i].is_uppercase()
            && chars[i - 1].is_uppercase()
            && chars[i + 1].is_lowercase()
        {
            // ABc -> A | Bc (end of acronym)
            true
        } else {
            false
        };

        if split {
            let part: String = chars[start..i].iter().collect();
            if !part.is_empty() {
                parts.push(part.to_ascii_lowercase());
            }
            start = i;
        }
    }

    let last: String = chars[start..].iter().collect();
    if !last.is_empty() {
        parts.push(last.to_ascii_lowercase());
    }

    if parts.len() < 2 {
        vec![token.to_ascii_lowercase()]
    } else {
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Language, SymbolType};

    fn rust_chunk() -> Chunk {
        Chunk {
            id: "src/auth.rs:0".to_string(),
            file_path: "src/auth.rs".to_string(),
            line_start: 1,
            line_end: 8,
            content: "use crate::db::UserStore;\n\
pub fn authenticate(user: &str, password: &str) -> Result<Token> {\n\
    let stored = UserStore::load(user)?;\n\
    if stored.verify(password) {\n\
        Ok(Token::new(user))\n\
    } else {\n\
        Err(AuthError::InvalidCredentials)\n\
    }\n\
}"
            .to_string(),
            language: Language::Rust,
            symbol_type: Some(SymbolType::Function),
            symbol_name: Some("authenticate".to_string()),
            part_index: None,
        }
    }

    #[test]
    fn normalize_path_tokens_splits_words() {
        let normalized = normalize_path_tokens("src/utils/HttpClientHelper.rs");
        assert!(normalized.contains("src utils http client helper"));
        assert!(normalized.contains("HttpClientHelper.rs"));
    }

    #[test]
    fn embedding_text_includes_structural_hints() {
        let text = build_embedding_text(&rust_chunk());
        assert!(text.contains("Symbol: function authenticate"));
        assert!(text.contains("Signature: pub fn authenticate"));
        assert!(text.contains("Parameters: user, password"));
        assert!(text.contains("Imports: crate::db::UserStore"));
        assert!(text.contains("Calls:"));
        assert!(text.contains("UserStore::load"));
        assert!(text.contains("verify"));
        assert!(text.contains("Token::new"));
        assert!(text.contains("Flow: branches, error handling"));
    }

    #[test]
    fn bm25_text_keeps_filename_signal() {
        let chunk = Chunk {
            id: "Cargo.toml:0".to_string(),
            file_path: "Cargo.toml".to_string(),
            line_start: 1,
            line_end: 4,
            content: "[workspace]\nmembers = [\"crates/vera-core\"]".to_string(),
            language: Language::Toml,
            symbol_type: Some(SymbolType::Block),
            symbol_name: Some("Cargo.toml".to_string()),
            part_index: None,
        };
        let text = build_bm25_text(&chunk);
        assert!(text.contains("Filename: Cargo.toml"));
        assert!(text.contains("Path: cargo toml Cargo.toml"));
        assert!(text.contains("Document:\n[workspace]"));
    }

    #[test]
    fn abbreviate_middle_keeps_prefix_and_suffix() {
        let shortened = abbreviate_middle("very_long_generated_identifier_name_for_testing", 16);
        assert_eq!(shortened.len(), 16);
        assert!(shortened.starts_with("very_l"));
        assert!(shortened.ends_with("testing"));
    }

    #[test]
    fn embedding_text_truncates_pathological_call_metadata() {
        let mut chunk = rust_chunk();
        chunk.content = format!("pub fn authenticate() {{\n    {}();\n}}", "a".repeat(200));

        let text = build_embedding_text(&chunk);
        let calls_line = text
            .lines()
            .find(|line| line.starts_with("Calls:"))
            .expect("Calls metadata should be present");
        assert!(!calls_line.contains(&"a".repeat(120)));
    }

    #[test]
    fn truncate_at_line_boundary_handles_multibyte_utf8() {
        let text = "first line\n漢字漢字";
        let max_bytes = "first line\n".len() + 1;

        assert_eq!(truncate_at_line_boundary(text, max_bytes), "first line");
    }

    #[test]
    fn embedding_text_bounded_never_exceeds_byte_budget() {
        let content = (0..120)
            .map(|i| format!("use crate::very::long::module::path{i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunk = Chunk {
            id: "src/lib.rs:0".to_string(),
            file_path: "src/lib.rs".to_string(),
            line_start: 1,
            line_end: 120,
            content,
            language: Language::Rust,
            symbol_type: Some(SymbolType::Block),
            symbol_name: None,
            part_index: None,
        };

        let text = build_embedding_text_bounded(&chunk, 220);
        assert!(
            text.len() <= 220,
            "bounded embedding text should not exceed max bytes"
        );
    }

    #[test]
    fn is_data_line_detects_hex_arrays() {
        assert!(is_data_line(
            "    0xFF, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,"
        ));
    }

    #[test]
    fn is_data_line_detects_numeric_tables() {
        assert!(is_data_line(
            "    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,"
        ));
    }

    #[test]
    fn is_data_line_detects_base85_strings() {
        assert!(is_data_line(
            r#"    "7])#######hV0qs'/###[),##/l:$#Q6>##5[n42>c-TH`->>#/e]6Nds7j7()]""#
        ));
    }

    #[test]
    fn is_data_line_ignores_normal_code() {
        assert!(!is_data_line(
            "    let result = calculate_hash(input, &config)?;"
        ));
        assert!(!is_data_line("    if (window->DrawList) { return; }"));
        assert!(!is_data_line("    return x;"));
    }

    #[test]
    fn is_data_line_ignores_short_lines() {
        assert!(!is_data_line("0xFF, 0x00,"));
        assert!(!is_data_line("    123,"));
    }

    #[test]
    fn collapse_data_lines_replaces_long_runs() {
        let mut lines = Vec::new();
        lines.push("static const unsigned char font_data[] = {".to_string());
        for i in 0..10 {
            lines.push(format!(
                "    0x{i:02X}, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,"
            ));
        }
        lines.push("};".to_string());
        let content = lines.join("\n");

        let collapsed = collapse_data_lines(&content);
        assert!(
            collapsed.contains("[10 lines of data]"),
            "should collapse 10 data lines, got: {collapsed}"
        );
        assert!(
            collapsed.contains("font_data"),
            "should preserve the declaration line"
        );
        assert!(
            collapsed.contains("};"),
            "should preserve the closing brace"
        );
    }

    #[test]
    fn collapse_data_lines_keeps_short_runs() {
        let content = "int x = 1;\n\
            0xFF, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,\n\
            0xFF, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,\n\
            int y = 2;";
        let collapsed = collapse_data_lines(content);
        assert!(
            !collapsed.contains("lines of data"),
            "runs shorter than MIN_DATA_RUN should be kept"
        );
    }

    #[test]
    fn collapse_data_lines_preserves_normal_code() {
        let content = "fn main() {\n\
            let config = Config::new();\n\
            let result = process(&config)?;\n\
            println!(\"{result}\");\n\
        }";
        let collapsed = collapse_data_lines(content);
        assert_eq!(collapsed, content, "normal code should be unchanged");
    }

    #[test]
    fn embedding_text_collapses_embedded_data() {
        let mut data_lines: Vec<String> = Vec::new();
        data_lines.push("static const unsigned char font[] = {".to_string());
        for _ in 0..20 {
            data_lines.push(
                "    0xFF, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,".to_string(),
            );
        }
        data_lines.push("};".to_string());

        let chunk = Chunk {
            id: "imgui_draw.cpp:211".to_string(),
            file_path: "Source/Include/dear-imgui/imgui_draw.cpp".to_string(),
            line_start: 211,
            line_end: 232,
            content: data_lines.join("\n"),
            language: Language::Cpp,
            symbol_type: Some(SymbolType::Variable),
            symbol_name: Some("font".to_string()),
            part_index: None,
        };

        let text = build_embedding_text(&chunk);
        assert!(
            text.contains("[20 lines of data]"),
            "embedding text should collapse data lines"
        );
        assert!(
            !text.contains("0xFF"),
            "embedding text should not contain raw hex data"
        );
    }
}
