//! Extract call-site references from tree-sitter ASTs.
//!
//! Walks the AST looking for function/method call expressions and records
//! the callee name, source file, and line number. These lightweight edges
//! power `vera references`, `vera impact`, and `vera dead-code`.

use crate::types::Language;

/// A raw reference (call site) extracted from the AST.
#[derive(Debug, Clone)]
pub struct RawReference {
    /// Name of the called symbol (e.g., function or method name).
    pub callee: String,
    /// Name of the enclosing symbol that contains this call (if known).
    pub caller: Option<String>,
    /// 1-based line number of the call site.
    pub line: u32,
}

/// Extract call-site references from a parsed tree.
///
/// Returns a list of `(callee_name, caller_name, line)` tuples representing
/// direct function/method calls found in the source. Does not attempt type
/// resolution or dynamic dispatch analysis.
pub fn extract_references(
    tree: &tree_sitter::Tree,
    source: &[u8],
    lang: Language,
) -> Vec<RawReference> {
    let symbols = super::extractor::extract_symbols(tree, source, lang);
    extract_references_with_symbols(tree, source, lang, &symbols)
}

/// Extract call-site references using pre-computed symbols.
///
/// Use this when symbols have already been extracted (e.g., during unified
/// parsing) to avoid redundant AST walks.
pub fn extract_references_with_symbols(
    tree: &tree_sitter::Tree,
    source: &[u8],
    lang: Language,
    symbols: &[super::extractor::RawSymbol],
) -> Vec<RawReference> {
    let mut refs = Vec::new();
    collect_calls(tree.root_node(), source, lang, symbols, &mut refs);
    refs
}

/// Node kinds that represent function/method calls per language.
fn is_call_node(lang: Language, kind: &str) -> bool {
    match lang {
        Language::Rust => matches!(kind, "call_expression" | "macro_invocation"),
        Language::TypeScript | Language::JavaScript => {
            matches!(kind, "call_expression" | "new_expression") || is_jsx_element_node(kind)
        }
        Language::Python => kind == "call",
        Language::Go => kind == "call_expression",
        Language::Java | Language::Kotlin | Language::CSharp => {
            matches!(
                kind,
                "method_invocation" | "object_creation_expression" | "invocation_expression"
            )
        }
        Language::C | Language::Cpp => kind == "call_expression",
        Language::Ruby => matches!(kind, "call" | "method_call"),
        Language::Swift => kind == "call_expression",
        Language::Php => matches!(kind, "function_call_expression" | "method_call_expression"),
        Language::Elixir => kind == "call",
        Language::Scala => matches!(kind, "call_expression" | "apply_expression"),
        Language::Lua | Language::Luau => kind == "function_call",
        Language::Dart => matches!(kind, "function_expression_invocation" | "method_invocation"),
        Language::Zig => kind == "call_expression",
        Language::Haskell => kind == "function_application",
        Language::Perl => kind == "function_call_expression",
        Language::Julia => kind == "call_expression",
        Language::OCaml => kind == "application_expression",
        Language::Groovy => matches!(kind, "method_call_expression" | "function_call"),
        Language::Erlang => matches!(kind, "call" | "function_call"),
        Language::FSharp => kind == "application_expression",
        Language::PowerShell => kind == "command_expression",
        Language::R => kind == "call",
        Language::DLang => kind == "call_expression",
        Language::Elm => kind == "function_call_expr",
        _ => false,
    }
}

/// Whether a node kind is a JSX element tag that names the component it renders.
///
/// `jsx_closing_element` is deliberately excluded so `<Foo></Foo>` records one
/// reference rather than two.
fn is_jsx_element_node(kind: &str) -> bool {
    matches!(kind, "jsx_opening_element" | "jsx_self_closing_element")
}

/// Whether a JSX tag names a host element rather than a component in scope.
///
/// JSX resolves a *bare* tag beginning with a lowercase ASCII letter, such as
/// `<div>`, to an intrinsic element string. Anything else is a value in scope,
/// and only those are call sites; recording host elements would add an edge per
/// HTML tag in the repository.
///
/// The rule mirrors TypeScript's `isIntrinsicJsxName`, which is
/// `first char in a..z || name contains '-'`. Both halves matter:
///
/// - the range is ASCII on purpose. A Unicode-aware lowercase test would call
///   `<éWidget />` a host element and drop a real component reference.
/// - a hyphen makes the tag intrinsic whatever its case, because that is the
///   custom-element form and TypeScript emits `<My-element />` as the string
///   `"My-element"` rather than a value lookup.
///
/// The decision has to come from the tag node, not from the extracted callee.
/// `extract_callee` returns the rightmost identifier, so `<Icons.arrow />` yields
/// `arrow`, and judging that name alone would discard a real component
/// reference. A dotted tag is a member expression on a value in scope whatever
/// the case of its property.
fn is_jsx_host_element(node: &tree_sitter::Node, source: &[u8]) -> bool {
    if !is_jsx_element_node(node.kind()) {
        return false;
    }
    let Some(name) = node.child_by_field_name("name") else {
        return false;
    };
    if name.kind() == "member_expression" {
        return false;
    }
    name.utf8_text(source).is_ok_and(|text| {
        text.starts_with(|ch: char| ch.is_ascii_lowercase()) || text.contains('-')
    })
}

/// Extract the callee name from a call node.
fn extract_callee(node: &tree_sitter::Node, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            // Direct function call: foo()
            "identifier" | "simple_identifier" => {
                return child.utf8_text(source).ok().map(|s| s.to_string());
            }
            // Method call: obj.method() — extract the method name
            "field_expression"
            | "member_expression"
            | "attribute"
            | "selector_expression"
            | "member_access_expression"
            | "navigation_expression" => {
                let mut inner = child.walk();
                for field_child in child.children(&mut inner) {
                    if matches!(
                        field_child.kind(),
                        "field_identifier"
                            | "property_identifier"
                            | "identifier"
                            | "simple_identifier"
                    ) {
                        // Take the last identifier (the method name)
                        if let Ok(text) = field_child.utf8_text(source) {
                            // Only return if this is the rightmost identifier
                            if field_child.end_byte() == child.end_byte()
                                || field_child.start_byte()
                                    > child.start_byte() + child.byte_range().len() / 2
                            {
                                return Some(text.to_string());
                            }
                        }
                    }
                }
                // Fallback: try named child "name" or "field"
                if let Some(name_node) = child.child_by_field_name("field") {
                    return name_node.utf8_text(source).ok().map(|s| s.to_string());
                }
                if let Some(name_node) = child.child_by_field_name("name") {
                    return name_node.utf8_text(source).ok().map(|s| s.to_string());
                }
            }
            // Scoped identifier: Foo::bar()
            "scoped_identifier" | "qualified_identifier" => {
                let mut inner = child.walk();
                let mut last_ident = None;
                for sc_child in child.children(&mut inner) {
                    if sc_child.kind() == "identifier" || sc_child.kind() == "type_identifier" {
                        last_ident = sc_child.utf8_text(source).ok().map(|s| s.to_string());
                    }
                }
                return last_ident;
            }
            _ => {}
        }
    }
    // Try the "function" or "name" field directly
    if let Some(func_node) = node.child_by_field_name("function") {
        if func_node.kind() == "identifier" || func_node.kind() == "simple_identifier" {
            return func_node.utf8_text(source).ok().map(|s| s.to_string());
        }
        // Nested field access
        if let Some(field) = func_node.child_by_field_name("field") {
            return field.utf8_text(source).ok().map(|s| s.to_string());
        }
        if let Some(name) = func_node.child_by_field_name("name") {
            return name.utf8_text(source).ok().map(|s| s.to_string());
        }
    }
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node.utf8_text(source).ok().map(|s| s.to_string());
    }
    if let Some(method_node) = node.child_by_field_name("method") {
        return method_node.utf8_text(source).ok().map(|s| s.to_string());
    }
    None
}

/// Find which symbol (from the extracted symbols list) encloses a given byte offset.
fn find_enclosing_symbol(
    symbols: &[super::extractor::RawSymbol],
    byte_offset: usize,
) -> Option<String> {
    symbols
        .iter()
        .filter(|s| s.start_byte <= byte_offset && byte_offset < s.end_byte)
        .min_by_key(|s| s.end_byte - s.start_byte)
        .and_then(|s| s.name.clone())
}

/// Collect call references using a single TreeCursor for the entire traversal.
fn collect_calls(
    root: tree_sitter::Node<'_>,
    source: &[u8],
    lang: Language,
    symbols: &[super::extractor::RawSymbol],
    refs: &mut Vec<RawReference>,
) {
    let mut cursor = root.walk();
    loop {
        let node = cursor.node();
        if is_call_node(lang, node.kind()) {
            if let Some(callee) = extract_callee(&node, source) {
                if !is_jsx_host_element(&node, source) {
                    let caller = find_enclosing_symbol(symbols, node.start_byte());
                    refs.push(RawReference {
                        callee,
                        caller,
                        line: node.start_position().row as u32 + 1,
                    });
                }
            }
        }
        // Depth-first: try child, then sibling, then backtrack.
        if cursor.goto_first_child() {
            continue;
        }
        if cursor.goto_next_sibling() {
            continue;
        }
        loop {
            if !cursor.goto_parent() {
                return;
            }
            if cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::languages::tree_sitter_grammar_for_path;
    use tree_sitter::Parser;

    fn parse_and_extract(source: &str, lang: Language) -> Vec<RawReference> {
        // An extensionless path resolves to the plain per-language grammar.
        parse_and_extract_for_path(source, "source", lang)
    }

    fn parse_and_extract_for_path(source: &str, path: &str, lang: Language) -> Vec<RawReference> {
        let grammar = tree_sitter_grammar_for_path(lang, path).expect("grammar should exist");
        let mut parser = Parser::new();
        parser.set_language(&grammar).unwrap();
        let tree = parser.parse(source, None).unwrap();
        extract_references(&tree, source.as_bytes(), lang)
    }

    #[test]
    fn rust_function_calls() {
        let source = r#"
fn main() {
    foo();
    bar::baz();
    obj.method();
}

fn foo() {}
"#;
        let refs = parse_and_extract(source, Language::Rust);
        let callees: Vec<&str> = refs.iter().map(|r| r.callee.as_str()).collect();
        assert!(
            callees.contains(&"foo"),
            "should find foo call: {callees:?}"
        );
        assert!(
            refs.iter().any(|r| r.caller.as_deref() == Some("main")),
            "caller should be main"
        );
    }

    #[test]
    fn python_function_calls() {
        let source = r#"
def main():
    foo()
    bar.baz()

def foo():
    pass
"#;
        let refs = parse_and_extract(source, Language::Python);
        let callees: Vec<&str> = refs.iter().map(|r| r.callee.as_str()).collect();
        assert!(
            callees.contains(&"foo"),
            "should find foo call: {callees:?}"
        );
    }

    #[test]
    fn javascript_function_calls() {
        let source = r#"
function main() {
    foo();
    obj.method();
    new Bar();
}

function foo() {}
"#;
        let refs = parse_and_extract(source, Language::JavaScript);
        let callees: Vec<&str> = refs.iter().map(|r| r.callee.as_str()).collect();
        assert!(
            callees.contains(&"foo"),
            "should find foo call: {callees:?}"
        );
    }

    #[test]
    fn go_function_calls() {
        let source = r#"
package main

func main() {
    foo()
    fmt.Println("hello")
}

func foo() {}
"#;
        let refs = parse_and_extract(source, Language::Go);
        let callees: Vec<&str> = refs.iter().map(|r| r.callee.as_str()).collect();
        assert!(
            callees.contains(&"foo"),
            "should find foo call: {callees:?}"
        );
    }

    #[test]
    fn tsx_jsx_elements_are_call_sites() {
        let source = r#"
import { Hello } from "./jsx";

export function App() {
    return (
        <div className="wrap">
            <Hello name="world" />
            <Panel.Header title="hi"></Panel.Header>
            <Icons.arrow />
            <éWidget />
            <My-element />
        </div>
    );
}
"#;
        let refs = parse_and_extract_for_path(source, "src/app.tsx", Language::TypeScript);
        let callees: Vec<&str> = refs.iter().map(|r| r.callee.as_str()).collect();
        assert!(
            callees.contains(&"Hello"),
            "JSX element should be a call site: {callees:?}"
        );
        assert!(
            callees.contains(&"Header"),
            "dotted JSX element should be a call site: {callees:?}"
        );
        assert!(
            callees.contains(&"arrow"),
            "a dotted tag is a value in scope whatever the case of its \
             property, so <Icons.arrow /> is a call site: {callees:?}"
        );
        assert!(
            !callees.contains(&"div"),
            "host elements should not be call sites: {callees:?}"
        );
        assert!(
            callees.contains(&"éWidget"),
            "only ASCII a-z marks an intrinsic tag, so <éWidget /> is a \
             component reference: {callees:?}"
        );
        assert!(
            !callees.contains(&"My-element"),
            "a hyphen makes a tag intrinsic whatever its case, matching \
             TypeScript's isIntrinsicJsxName: {callees:?}"
        );
        assert_eq!(
            refs.iter().filter(|r| r.callee == "Header").count(),
            1,
            "paired tags should record one reference, not two"
        );
        assert!(
            refs.iter()
                .any(|r| r.callee == "Hello" && r.caller.as_deref() == Some("App")),
            "caller should be App: {refs:?}"
        );
    }

    #[test]
    fn jsx_in_javascript_is_a_call_site() {
        let source = "export function App() { return <Hello name=\"world\" />; }\n";
        let refs = parse_and_extract(source, Language::JavaScript);
        let callees: Vec<&str> = refs.iter().map(|r| r.callee.as_str()).collect();
        assert!(
            callees.contains(&"Hello"),
            "JSX element should be a call site: {callees:?}"
        );
    }

    #[test]
    fn empty_source_returns_no_refs() {
        let refs = parse_and_extract("", Language::Rust);
        assert!(refs.is_empty());
    }
}
