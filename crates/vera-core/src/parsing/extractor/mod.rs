//! AST symbol extraction rules per language.
//!
//! Defines which tree-sitter node types correspond to which [`SymbolType`]
//! for each supported language. Walks the AST to extract top-level symbols.

use crate::types::Language;
use crate::types::SymbolType;

pub(crate) mod classify;
pub(crate) mod names;
mod special_forms;

#[cfg(test)]
mod tests;

use classify::classify_node;
use names::extract_name;
use special_forms::*;

/// A raw symbol extracted from the AST before chunking.
#[derive(Debug, Clone)]
pub struct RawSymbol {
    /// Name of the symbol (e.g., function name, class name).
    pub name: Option<String>,
    /// Type of symbol.
    pub symbol_type: SymbolType,
    /// 0-based byte offset of the symbol start in the source.
    pub start_byte: usize,
    /// 0-based byte offset of the symbol end in the source.
    pub end_byte: usize,
    /// 0-based start row in the source.
    pub start_row: usize,
    /// 0-based end row in the source.
    pub end_row: usize,
}

impl RawSymbol {
    /// Build a symbol spanning the given node.
    fn at(node: &tree_sitter::Node<'_>, name: Option<String>, symbol_type: SymbolType) -> Self {
        Self {
            name,
            symbol_type,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            start_row: node.start_position().row,
            end_row: node.end_position().row,
        }
    }
}

/// Maps a tree-sitter node kind to a [`SymbolType`] for the given language.
///
/// Returns `None` if the node kind is not a top-level symbol we extract.
/// Extract top-level symbols from a parsed tree.
///
/// Walks the AST, identifying nodes that match the language's extraction rules.
/// Returns symbols sorted by their position in the source.
pub fn extract_symbols(tree: &tree_sitter::Tree, source: &[u8], lang: Language) -> Vec<RawSymbol> {
    let mut symbols = Vec::new();
    let mut cursor = tree.root_node().walk();
    collect_symbols_cursor(&mut cursor, source, lang, &mut symbols, 1);
    symbols.sort_by_key(|s| s.start_byte);
    symbols
}

/// Extract reStructuredText heading titles from the AST.
///
/// Returns a sorted list of `(start_row, title_text)` pairs. Rows are 0-based.
pub fn extract_rst_section_titles(tree: &tree_sitter::Tree, source: &[u8]) -> Vec<(u32, String)> {
    fn walk(node: tree_sitter::Node<'_>, source: &[u8], out: &mut Vec<(u32, String)>) {
        if node.kind() == "title" {
            if let Ok(text) = node.utf8_text(source) {
                let title = text.split_whitespace().collect::<Vec<_>>().join(" ");
                if !title.is_empty() {
                    out.push((node.start_position().row as u32, title));
                }
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            walk(child, source, out);
        }
    }

    let mut titles = Vec::new();
    walk(tree.root_node(), source, &mut titles);
    titles.sort_by_key(|(row, _)| *row);
    titles.dedup_by(|a, b| a.0 == b.0);
    titles
}

/// Iterates siblings using a single `TreeCursor`, avoiding per-level cursor
/// allocation. Delegates to [`collect_symbols`] for per-node logic.
fn collect_symbols_cursor(
    cursor: &mut tree_sitter::TreeCursor,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
    depth: usize,
) {
    if depth > 6 {
        return;
    }
    if !cursor.goto_first_child() {
        return;
    }
    loop {
        collect_symbols(cursor.node(), source, lang, symbols, depth);
        if !cursor.goto_next_sibling() {
            break;
        }
    }
    cursor.goto_parent();
}

/// Recursively collect symbols from AST nodes.
///
/// `depth` limits how deep we recurse to avoid extracting deeply nested items
/// as top-level symbols. We go up to depth 6 to handle patterns like:
/// - export_statement > function_declaration (TS/JS)
/// - decorated_definition > function_definition (Python)
/// - impl_item > function_item (Rust methods)
/// - source_file > document > definition > type_system_definition >
///   type_definition > object_type_definition (GraphQL)
fn collect_symbols(
    node: tree_sitter::Node<'_>,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
    depth: usize,
) {
    if depth > 6 {
        return;
    }

    let kind = node.kind();

    // Handle Go type_declaration → recurse into type_spec children
    if lang == Language::Go && kind == "type_declaration" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "type_spec" {
                let sym_type = refine_go_type_spec(&child, source);
                let name = extract_name(&child, source);
                symbols.push(RawSymbol::at(&child, name, sym_type));
            }
        }
        return;
    }

    // Handle R binary_operator (x <- function() { ... }) as named function
    if lang == Language::R && kind == "binary_operator" {
        if let Some(rhs) = node.child_by_field_name("rhs") {
            if rhs.kind() == "function_definition" {
                let name = extract_name(&node, source);
                symbols.push(RawSymbol::at(&node, name, SymbolType::Function));
                return;
            }
        }
    }

    // Handle TS/JS function-valued bindings: the name lives on the declarator,
    // and a function initializer makes the binding a function, not a variable.
    if (lang == Language::TypeScript || lang == Language::JavaScript)
        && matches!(kind, "lexical_declaration" | "variable_declaration")
    {
        if let Some(sym) = extract_js_function_binding(&node, source) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Zig variable_declaration -> struct_declaration
    if lang == Language::Zig && kind == "variable_declaration" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "struct_declaration" {
                let name = extract_name(&node, source);
                symbols.push(RawSymbol::at(&node, name, SymbolType::Struct));
                return;
            }
        }
    }

    // Handle Scheme S-expressions: (define (name ...) ...)
    if lang == Language::Scheme && kind == "list" {
        if let Some(sym) = extract_scheme_define(&node, source) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Racket S-expressions: (define (name ...) ...), (module ...), (struct ...)
    if lang == Language::Racket && kind == "list" {
        if let Some(sym) = extract_racket_define(&node, source) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Clojure S-expressions: (defn name ...), (ns name), (defmacro name ...), (def name ...)
    if lang == Language::Clojure && kind == "list_lit" {
        if let Some(sym) = extract_lisp_define(&node, source, CLOJURE_DEFINE_KINDS) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Common Lisp: defun has defun_header with name; defclass via list_lit
    if lang == Language::CommonLisp && kind == "defun" {
        if let Some(sym) = extract_commonlisp_defun(&node, source) {
            symbols.push(sym);
            return;
        }
    }
    if lang == Language::CommonLisp && kind == "list_lit" {
        if let Some(sym) = extract_lisp_define(&node, source, COMMONLISP_DEFINE_KINDS) {
            symbols.push(sym);
            return;
        }
    }

    // Handle Elixir calls
    if lang == Language::Elixir && kind == "call" {
        if let Some(target) = node.child_by_field_name("target") {
            if let Ok(text) = target.utf8_text(source) {
                let sym_type = match text {
                    "defmodule" => Some(SymbolType::Module),
                    "def" | "defp" | "defmacro" => Some(SymbolType::Function),
                    _ => None,
                };
                if let Some(st) = sym_type {
                    let name = extract_elixir_name(&node, source);
                    symbols.push(RawSymbol::at(&node, name, st));
                    if text == "defmodule" {
                        if let Some(do_block) = get_elixir_do_block(&node) {
                            let mut cursor = do_block.walk();
                            collect_symbols_cursor(&mut cursor, source, lang, symbols, depth + 1);
                        }
                    }
                    return;
                }
            }
        }
    }

    if let Some(mut sym_type) = classify_node(lang, kind) {
        // Refine HCL blocks
        if lang == Language::Hcl && kind == "block" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "identifier" {
                    if let Ok(text) = child.utf8_text(source) {
                        sym_type = match text {
                            "resource" | "data" => SymbolType::Struct,
                            "variable" | "output" => SymbolType::TypeAlias,
                            "module" => SymbolType::Module,
                            _ => SymbolType::Struct,
                        };
                    }
                    break;
                }
            }
        }

        // Refine Kotlin and Swift class_declaration
        if (lang == Language::Kotlin || lang == Language::Swift) && kind == "class_declaration" {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                let ckind = child.kind();
                if ckind == "enum" || ckind == "enum_class_body" {
                    sym_type = SymbolType::Enum;
                } else if ckind == "interface" {
                    sym_type = SymbolType::Interface;
                } else if ckind == "struct" {
                    sym_type = SymbolType::Struct;
                }
            }
        }

        // For Rust impl and trait blocks, extract methods inside but also keep
        // the whole block
        if lang == Language::Rust && (kind == "impl_item" || kind == "trait_item") {
            extract_rust_block_methods(node, source, lang, symbols, sym_type);
            return;
        }

        // A Rust `mod name { ... }` is a container, not a leaf. Record the
        // module, then keep walking so the items declared inside it are
        // extracted as symbols of their own instead of being swallowed by the
        // module's span. `mod name;` has no body and simply yields nothing more.
        //
        // Descend into the `declaration_list` rather than the `mod_item`, the
        // way `extract_rust_block_methods` does: recursing from the `mod_item`
        // spends one depth level on the module and a second on its body list,
        // so three levels of nesting would exhaust the shared budget before
        // reaching the items inside.
        if lang == Language::Rust && kind == "mod_item" {
            let name = extract_name(&node, source);
            symbols.push(RawSymbol::at(&node, name, sym_type));
            if let Some(body) = node.child_by_field_name("body") {
                let mut cursor = body.walk();
                collect_symbols_cursor(&mut cursor, source, lang, symbols, depth + 1);
            }
            return;
        }

        // For Python classes, extract methods as separate chunks instead of
        // keeping the entire class body as a single chunk.
        if lang == Language::Python && kind == "class_definition" {
            extract_python_class_methods(node, source, symbols);
            return;
        }

        // For Objective-C classes, extract methods as separate chunks
        if lang == Language::ObjectiveC
            && (kind == "class_interface"
                || kind == "class_implementation"
                || kind == "category_interface"
                || kind == "category_implementation")
        {
            let name = extract_name(&node, source);
            symbols.push(RawSymbol::at(&node, name, sym_type));
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(child_sym) = classify_node(lang, child.kind()) {
                    let child_name = extract_name(&child, source);
                    symbols.push(RawSymbol::at(&child, child_name, child_sym));
                }
                // Also check inside implementation_definition
                if child.kind() == "implementation_definition" {
                    let mut inner = child.walk();
                    for inner_child in child.children(&mut inner) {
                        if let Some(inner_sym) = classify_node(lang, inner_child.kind()) {
                            let inner_name = extract_name(&inner_child, source);
                            symbols.push(RawSymbol::at(&inner_child, inner_name, inner_sym));
                        }
                    }
                }
            }
            return;
        }

        // For C# namespace, we want to recurse inside.
        if lang == Language::CSharp
            && (kind == "namespace_declaration" || kind == "file_scoped_namespace_declaration")
        {
            let name = extract_name(&node, source);
            symbols.push(RawSymbol::at(&node, name, sym_type));
            let mut cursor = node.walk();
            collect_symbols_cursor(&mut cursor, source, lang, symbols, depth + 1);
            return;
        }

        // For class-like declarations in languages where methods live inside
        // class/interface/enum bodies, extract nested methods as separate symbols.
        if (lang == Language::CSharp
            || lang == Language::Php
            || lang == Language::Dart
            || lang == Language::TypeScript
            || lang == Language::JavaScript
            || lang == Language::Java)
            && (kind == "class_declaration"
                || kind == "class_definition"
                || kind == "interface_declaration"
                || kind == "enum_declaration")
        {
            extract_general_class_methods(node, source, lang, symbols, sym_type);
            return;
        }

        // For Protobuf services, extract rpc methods:
        if lang == Language::Protobuf && (kind == "service" || kind == "service_definition") {
            let name = extract_name(&node, source);
            symbols.push(RawSymbol::at(&node, name, sym_type));

            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                // Sometime tree-sitter has a body node containing rpcs
                if child.kind() == "service_body" || child.kind() == "block" {
                    let mut inner_cursor = child.walk();
                    for inner_child in child.children(&mut inner_cursor) {
                        if inner_child.kind() == "rpc"
                            || inner_child.kind() == "rpc_definition"
                            || inner_child.kind() == "rpc_declaration"
                        {
                            if let Some(rpc_type) = classify_node(lang, inner_child.kind()) {
                                let end_byte = inner_child.end_byte();
                                let end_row = inner_child.end_position().row;
                                let rpc_name = extract_name(&inner_child, source);
                                symbols.push(RawSymbol {
                                    name: rpc_name,
                                    symbol_type: rpc_type,
                                    start_byte: inner_child.start_byte(),
                                    end_byte,
                                    start_row: inner_child.start_position().row,
                                    end_row,
                                });
                            }
                        }
                    }
                } else if child.kind() == "rpc"
                    || child.kind() == "rpc_definition"
                    || child.kind() == "rpc_declaration"
                {
                    if let Some(rpc_type) = classify_node(lang, child.kind()) {
                        let end_byte = child.end_byte();
                        let end_row = child.end_position().row;
                        let rpc_name = extract_name(&child, source);
                        symbols.push(RawSymbol {
                            name: rpc_name,
                            symbol_type: rpc_type,
                            start_byte: child.start_byte(),
                            end_byte,
                            start_row: child.start_position().row,
                            end_row,
                        });
                    }
                }
            }
            return;
        }

        let mut end_byte = node.end_byte();
        let mut end_row = node.end_position().row;

        if lang == Language::Dart && (kind == "function_signature" || kind == "method_signature") {
            if let Some(next_sibling) = node.next_sibling() {
                if next_sibling.kind() == "function_body" {
                    end_byte = next_sibling.end_byte();
                    end_row = next_sibling.end_position().row;
                }
            }
        }

        let name = extract_name(&node, source);
        symbols.push(RawSymbol {
            name,
            symbol_type: sym_type,
            start_byte: node.start_byte(),
            end_byte,
            start_row: node.start_position().row,
            end_row,
        });
        return;
    }

    // Recurse into children for wrapper nodes
    let mut cursor = node.walk();
    collect_symbols_cursor(&mut cursor, source, lang, symbols, depth + 1);
}
