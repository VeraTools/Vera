//! Extract function/class signatures by stripping bodies from code snippets.
//!
//! Used by `--compact` mode to return only declarations without implementation
//! details. Parses the content with tree-sitter, finds the outermost symbol
//! node, and returns everything before the body.

use tree_sitter::Parser;

use crate::types::Language;

use super::languages::tree_sitter_grammar_for_path;

/// Body node kinds per language family. The first match wins.
fn body_node_kinds(lang: Language) -> &'static [&'static str] {
    match lang {
        // Tier 1A: core languages
        Language::Rust => &[
            "block",
            "field_declaration_list",
            "enum_variant_list",
            "declaration_list",
        ],
        Language::TypeScript | Language::JavaScript => {
            &["statement_block", "class_body", "interface_body"]
        }
        Language::Python => &["block"],
        Language::Go => &["block", "field_declaration_list"],
        Language::Java => &["block", "class_body", "interface_body", "enum_body"],
        Language::C | Language::Cpp => &["compound_statement", "field_declaration_list"],
        Language::CSharp => &["block", "declaration_list"],
        Language::Kotlin => &["class_body", "function_body"],
        Language::Swift => &["function_body", "class_body"],
        Language::Ruby => &["body_statement"],
        Language::Php => &["compound_statement", "declaration_list"],
        Language::Scala => &["block", "template_body"],
        Language::Dart => &["block", "class_body"],
        Language::Lua | Language::Luau => &["block"],
        Language::Zig => &["block"],
        Language::Elixir => &["do_block"],
        Language::Haskell => &["where_clause"],
        Language::Bash | Language::Fish | Language::Zsh => &["compound_statement", "do_group"],
        // Tier 2A: additional code languages
        Language::ObjectiveC => &["compound_statement", "field_declaration_list"],
        Language::Perl => &["block"],
        Language::Julia => &["block"],
        Language::OCaml => &["let_binding"],
        Language::Groovy => &["block", "class_body"],
        Language::Clojure => &["list_lit"],
        Language::CommonLisp => &["list_lit"],
        Language::Erlang => &["clause_body"],
        Language::FSharp => &["block"],
        Language::Fortran => &["body"],
        Language::PowerShell => &["script_block"],
        Language::R => &["brace_list"],
        Language::DLang => &["block_statement", "struct_body"],
        Language::Scheme | Language::Racket => &["list"],
        Language::Elm => &["let_in_expr"],
        Language::Glsl | Language::Hlsl => &["compound_statement"],
        Language::Nix => &["attrset_expression"],
        // Tier 1B/2B: structural/frontend
        Language::Sql => &["begin_block"],
        Language::Svelte | Language::Vue | Language::Astro => &["script_element"],
        Language::Css | Language::Scss => &["block"],
        _ => &[],
    }
}

/// Placeholder that replaces the stripped body.
fn body_placeholder(lang: Language) -> &'static str {
    match lang {
        Language::Python | Language::Elixir => " ...",
        Language::Ruby | Language::Erlang => "\n  # ...\nend",
        Language::Lua | Language::Luau => "\n  -- ...\nend",
        Language::Bash | Language::Fish | Language::Zsh => "\n  # ...\ndone",
        Language::Fortran => "\n  ! ...\nend",
        Language::Haskell | Language::Elm => " = ...",
        Language::Clojure | Language::CommonLisp | Language::Scheme | Language::Racket => " ...)",
        Language::Nix => " { ... }",
        _ => " { ... }",
    }
}

/// Extract the signature from a code snippet by stripping the body.
///
/// Returns `None` if the language has no grammar, parsing fails, or no
/// body node is found (caller should use the fallback).
fn extract_signature_inner(content: &str, lang: Language, file_path: &str) -> Option<String> {
    let grammar = tree_sitter_grammar_for_path(lang, file_path)?;
    let kinds = body_node_kinds(lang);
    if kinds.is_empty() {
        return None;
    }

    let mut parser = Parser::new();
    parser.set_language(&grammar).ok()?;
    let tree = parser.parse(content, None)?;

    // Walk the tree looking for the first body node at depth <= 3.
    let mut cursor = tree.root_node().walk();
    let body_start = find_body_start(&mut cursor, kinds, 0, 3)?;

    // Trim trailing whitespace before the body, keep the signature.
    let sig = content[..body_start].trim_end();
    let placeholder = body_placeholder(lang);
    Some(format!("{sig}{placeholder}"))
}

/// Recursively find the byte offset where the first matching body node starts.
fn find_body_start(
    cursor: &mut tree_sitter::TreeCursor,
    kinds: &[&str],
    depth: usize,
    max_depth: usize,
) -> Option<usize> {
    if depth > max_depth {
        return None;
    }
    if cursor.goto_first_child() {
        loop {
            let node = cursor.node();
            if kinds.contains(&node.kind()) {
                return Some(node.start_byte());
            }
            if let Some(offset) = find_body_start(cursor, kinds, depth + 1, max_depth) {
                return Some(offset);
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
    None
}

/// Fallback: return the first `max_lines` lines with a truncation marker.
fn first_n_lines(content: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= max_lines {
        return content.to_string();
    }
    let mut out: String = lines[..max_lines].join("\n");
    let remaining = lines.len() - max_lines;
    out.push_str(&format!("\n[... {remaining} more lines]"));
    out
}

/// Extract a compact signature from a code snippet using the language grammar.
///
/// This preserves the original public API. Call [`extract_signature_for_path`]
/// when the source path is available and the grammar dialect may depend on its
/// extension.
pub fn extract_signature(content: &str, lang: Language) -> String {
    extract_signature_inner(content, lang, "").unwrap_or_else(|| first_n_lines(content, 3))
}

/// Extract a compact signature using the grammar selected for `file_path`.
///
/// Tries tree-sitter body stripping first. Falls back to first 3 lines
/// for unsupported languages or non-symbol chunks.
///
/// `file_path` selects the grammar dialect, the same way indexing does: a
/// `.tsx` file must be parsed with the tsx grammar. Parsing JSX with the plain
/// TypeScript grammar can fail to locate the body at all, in which case this
/// silently degrades to the raw first-three-lines fallback.
pub fn extract_signature_for_path(content: &str, lang: Language, file_path: &str) -> String {
    extract_signature_inner(content, lang, file_path).unwrap_or_else(|| first_n_lines(content, 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typescript_interface_signature() {
        let code = "export interface Repo extends Base {\n    get(id: string): Item;\n}\n";
        let sig = extract_signature(code, Language::TypeScript);
        assert_eq!(sig, "export interface Repo extends Base { ... }");
    }

    #[test]
    fn tsx_signature_uses_the_tsx_grammar() {
        // JSX ahead of a declaration derails the plain TypeScript grammar
        // badly enough that no body node is found within the depth limit, so
        // `extract_signature` silently degrades to the raw first-three-lines
        // fallback instead of a stripped signature. Selecting the grammar by
        // path, as indexing does, keeps it working.
        let code = "const A = <p>{ q }</p>;\nfunction Later(x: number) {\n  return x;\n}\n";

        let tsx = extract_signature_for_path(code, Language::TypeScript, "Widget.tsx");
        assert_eq!(
            tsx,
            "const A = <p>{ q }</p>;\nfunction Later(x: number) { ... }"
        );

        // The same content under a .ts path keeps the old behaviour, which is
        // correct: there the angle brackets really are type syntax. Assert the
        // exact `first_n_lines` output rather than merely "not the tsx one" —
        // a differently malformed signature would satisfy a negative check.
        let ts = extract_signature_for_path(code, Language::TypeScript, "Widget.ts");
        assert_eq!(
            ts,
            "const A = <p>{ q }</p>;\nfunction Later(x: number) {\n  return x;\n[... 1 more lines]"
        );
    }

    #[test]
    fn rust_pub_trait_signature() {
        let code = "pub trait Child: Parent + Send {\n    fn child(&self);\n}\n";
        let sig = extract_signature(code, Language::Rust);
        assert_eq!(sig, "pub trait Child: Parent + Send { ... }");
    }

    #[test]
    fn rust_function_signature() {
        let code = "pub fn authenticate(credentials: &Credentials) -> Result<Token> {\n    let user = db.find_user(&credentials.username)?;\n    validate(user)?;\n    Ok(Token::new())\n}";
        let sig = extract_signature(code, Language::Rust);
        assert!(sig.contains("pub fn authenticate"));
        assert!(sig.contains("{ ... }"));
        assert!(!sig.contains("find_user"));
    }

    #[test]
    fn python_function_signature() {
        let code = "def authenticate(credentials: Credentials) -> Token:\n    user = db.find_user(credentials.username)\n    validate(user)\n    return Token()";
        let sig = extract_signature(code, Language::Python);
        assert!(sig.contains("def authenticate"));
        assert!(sig.contains("..."));
        assert!(!sig.contains("find_user"));
    }

    #[test]
    fn typescript_function_signature() {
        let code = "function authenticate(credentials: Credentials): Token {\n    const user = db.findUser(credentials.username);\n    return new Token();\n}";
        let sig = extract_signature(code, Language::TypeScript);
        assert!(sig.contains("function authenticate"));
        assert!(sig.contains("{ ... }"));
        assert!(!sig.contains("findUser"));
    }

    #[test]
    fn go_function_signature() {
        let code = "func authenticate(creds Credentials) (Token, error) {\n\tuser := db.FindUser(creds.Username)\n\treturn Token{}, nil\n}";
        let sig = extract_signature(code, Language::Go);
        assert!(sig.contains("func authenticate"));
        assert!(sig.contains("{ ... }"));
        assert!(!sig.contains("FindUser"));
    }

    #[test]
    fn java_method_signature() {
        let code = "public Token authenticate(Credentials creds) {\n    User user = db.findUser(creds.username);\n    return new Token();\n}";
        let sig = extract_signature(code, Language::Java);
        assert!(sig.contains("public Token authenticate"));
        assert!(sig.contains("{ ... }"));
        assert!(!sig.contains("findUser"));
    }

    #[test]
    fn c_function_signature() {
        let code = "int authenticate(const char* user, const char* pass) {\n    int result = check_db(user, pass);\n    return result;\n}";
        let sig = extract_signature(code, Language::C);
        assert!(sig.contains("int authenticate"));
        assert!(sig.contains("{ ... }"));
        assert!(!sig.contains("check_db"));
    }

    #[test]
    fn fallback_for_unknown_language() {
        let code = "line1\nline2\nline3\nline4\nline5\nline6";
        let sig = extract_signature(code, Language::Unknown);
        assert!(sig.contains("line1"));
        assert!(sig.contains("line3"));
        assert!(sig.contains("[... 3 more lines]"));
        assert!(!sig.contains("line4"));
    }

    #[test]
    fn short_content_unchanged() {
        let code = "const X: i32 = 42;";
        let sig = extract_signature(code, Language::Rust);
        // No body to strip, fallback returns as-is since <= 3 lines.
        assert_eq!(sig, code);
    }

    #[test]
    fn rust_struct_signature() {
        let code = "pub struct Config {\n    pub name: String,\n    pub value: i32,\n}";
        let sig = extract_signature(code, Language::Rust);
        assert!(sig.contains("pub struct Config"));
        assert!(sig.contains("{ ... }"));
        assert!(!sig.contains("pub name"));
    }
}
