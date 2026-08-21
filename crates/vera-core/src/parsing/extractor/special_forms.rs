//! Special-form extractors for languages whose grammars need custom
//! handling (lisp defines, elixir do-blocks, go type specs, class methods).

use super::RawSymbol;
use super::classify::classify_node;
use super::names::extract_name;
use crate::types::{Language, SymbolType};

/// Extract a symbol from a Scheme `list` node if it starts with `define` or `define-syntax`.
///
/// Scheme AST: `list` → first child `symbol` "define" → second child is either
/// a `list` (procedure: name is its first `symbol`) or a `symbol` (variable definition).
pub(super) fn extract_scheme_define(
    node: &tree_sitter::Node<'_>,
    source: &[u8],
) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let children: Vec<_> = node.children(&mut cursor).collect();

    // Need at least: ( symbol <something> )
    // Find the first `symbol` child (skip parentheses)
    let first_sym = children.iter().find(|c| c.kind() == "symbol")?;
    let keyword = first_sym.utf8_text(source).ok()?;

    let sym_type = match keyword {
        "define" | "define-syntax" => SymbolType::Function,
        _ => return None,
    };

    // The name is in the next meaningful child after the keyword
    let after_keyword: Vec<_> = children
        .iter()
        .skip_while(|c| c.start_byte() <= first_sym.start_byte())
        .filter(|c| c.kind() != "(" && c.kind() != ")")
        .collect();

    let name = if let Some(next) = after_keyword.first() {
        if next.kind() == "list" {
            // (define (hello ...) ...) → name is first symbol in the inner list
            let mut inner_cursor = next.walk();
            next.children(&mut inner_cursor)
                .find(|c| c.kind() == "symbol")
                .and_then(|c| c.utf8_text(source).ok().map(|s| s.to_string()))
        } else if next.kind() == "symbol" {
            // (define x 42) → name is the symbol directly
            next.utf8_text(source).ok().map(|s| s.to_string())
        } else {
            None
        }
    } else {
        None
    };

    Some(RawSymbol::at(node, name, sym_type))
}

/// Extract a symbol from a Racket `list` node for define/module/struct forms.
pub(super) fn extract_racket_define(
    node: &tree_sitter::Node<'_>,
    source: &[u8],
) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let children: Vec<_> = node.children(&mut cursor).collect();

    let first_sym = children.iter().find(|c| c.kind() == "symbol")?;
    let keyword = first_sym.utf8_text(source).ok()?;

    let sym_type = match keyword {
        "define" | "define-syntax" => SymbolType::Function,
        "module" | "module*" | "module+" => SymbolType::Module,
        "struct" => SymbolType::Struct,
        _ => return None,
    };

    let after_keyword: Vec<_> = children
        .iter()
        .skip_while(|c| c.start_byte() <= first_sym.start_byte())
        .filter(|c| c.kind() != "(" && c.kind() != ")")
        .collect();

    let name = if let Some(next) = after_keyword.first() {
        if next.kind() == "list" {
            let mut inner_cursor = next.walk();
            next.children(&mut inner_cursor)
                .find(|c| c.kind() == "symbol")
                .and_then(|c| c.utf8_text(source).ok().map(|s| s.to_string()))
        } else if next.kind() == "symbol" {
            next.utf8_text(source).ok().map(|s| s.to_string())
        } else {
            None
        }
    } else {
        None
    };

    Some(RawSymbol::at(node, name, sym_type))
}

/// Keyword-to-symbol-type table for Clojure `list_lit` define forms.
pub(super) const CLOJURE_DEFINE_KINDS: &[(&[&str], SymbolType)] = &[
    (&["defn", "defmacro", "defn-"], SymbolType::Function),
    (&["ns"], SymbolType::Module),
    (&["def", "defonce"], SymbolType::Variable),
];

/// Keyword-to-symbol-type table for Common Lisp `list_lit` define forms.
pub(super) const COMMONLISP_DEFINE_KINDS: &[(&[&str], SymbolType)] = &[
    (&["defclass"], SymbolType::Class),
    (
        &["defvar", "defparameter", "defconstant"],
        SymbolType::Variable,
    ),
    (&["defpackage"], SymbolType::Module),
    (
        &["defmacro", "defgeneric", "defmethod"],
        SymbolType::Function,
    ),
];

/// Extract a symbol from a Lisp `list_lit` define form.
///
/// AST: `list_lit` → first `sym_lit` child text is the keyword → second
/// `sym_lit` is the name.
pub(super) fn extract_lisp_define(
    node: &tree_sitter::Node<'_>,
    source: &[u8],
    keyword_types: &[(&[&str], SymbolType)],
) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let sym_lits: Vec<_> = node
        .children(&mut cursor)
        .filter(|c| c.kind() == "sym_lit")
        .collect();

    if sym_lits.len() < 2 {
        return None;
    }

    let keyword = sym_lits[0].utf8_text(source).ok()?;
    let sym_type = keyword_types
        .iter()
        .find_map(|(keywords, ty)| keywords.contains(&keyword).then_some(*ty))?;

    let name = sym_lits[1].utf8_text(source).ok().map(|s| s.to_string());

    Some(RawSymbol::at(node, name, sym_type))
}

/// Extract a symbol from a Common Lisp `defun` node.
///
/// CL AST: `defun` → `defun_header` child → `sym_lit` child is the name.
pub(super) fn extract_commonlisp_defun(
    node: &tree_sitter::Node<'_>,
    source: &[u8],
) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let name = node
        .children(&mut cursor)
        .find(|c| c.kind() == "defun_header")
        .and_then(|header| {
            let mut hcursor = header.walk();
            header
                .children(&mut hcursor)
                .find(|c| c.kind() == "sym_lit")
                .and_then(|s| s.utf8_text(source).ok().map(|t| t.to_string()))
        });

    Some(RawSymbol::at(node, name, SymbolType::Function))
}

pub(super) fn extract_elixir_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    let mut args = None;
    for child in node.children(&mut cursor) {
        if child.kind() == "arguments" {
            args = Some(child);
            break;
        }
    }

    if let Some(args_node) = args {
        if args_node.named_child_count() > 0 {
            if let Some(first_arg) = args_node.named_child(0) {
                if first_arg.kind() == "call" {
                    if let Some(target) = first_arg.child_by_field_name("target") {
                        return target.utf8_text(source).ok().map(|s| s.to_string());
                    }
                }
                let mut inner_cursor = first_arg.walk();
                for child in first_arg.children(&mut inner_cursor) {
                    if child.kind() == "identifier" || child.kind() == "alias" {
                        return child.utf8_text(source).ok().map(|s| s.to_string());
                    }
                }
                return first_arg.utf8_text(source).ok().map(|s| s.to_string());
            }
        }
    }
    None
}

pub(super) fn get_elixir_do_block<'a>(
    node: &tree_sitter::Node<'a>,
) -> Option<tree_sitter::Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| child.kind() == "do_block")
}

/// Extract function-valued `const`, `let`, and `var` bindings as named function
/// symbols.
///
/// The name sits on the `variable_declarator`, not on the declaration itself,
/// so the generic name lookup finds nothing and the symbol is stored unnamed.
/// That is what makes function-valued bindings such as React components and
/// utilities unreachable from `structural definitions`.
///
/// `let` and `var` bindings are covered too: the reported symptom was about
/// `const` because that is the dominant style, but the missing name is a
/// property of the declarator, not of the keyword, so scoping this to `const`
/// would leave the same bug in place for the other two.
///
/// Returns `None` for anything else, including multi-declarator statements,
/// destructuring patterns and bindings to plain values, so those keep their
/// existing chunk shape.
///
/// The symbol spans the declaration node, which is what every other declaration
/// kind in this extractor does. `export` is part of the enclosing
/// `export_statement`, so it sits outside that span exactly as it already does
/// for `export`-ed functions, classes and interfaces. Chunk content is expanded
/// to whole lines afterwards, so a single-line `export const f = ...` still reads
/// with its `export`, while a declaration split across lines does not.
pub(super) fn extract_js_function_binding(
    node: &tree_sitter::Node<'_>,
    source: &[u8],
) -> Option<RawSymbol> {
    let mut cursor = node.walk();
    let mut declarators = node
        .children(&mut cursor)
        .filter(|child| child.kind() == "variable_declarator");
    let declarator = declarators.next()?;
    if declarators.next().is_some() {
        return None;
    }

    let value = declarator.child_by_field_name("value")?;
    if !matches!(
        value.kind(),
        "arrow_function" | "function_expression" | "generator_function"
    ) {
        return None;
    }

    let name = extract_name(&declarator, source)?;
    Some(RawSymbol::at(node, Some(name), SymbolType::Function))
}

/// Refine a Go type_spec into the correct SymbolType based on the type child.
pub(super) fn refine_go_type_spec(node: &tree_sitter::Node<'_>, source: &[u8]) -> SymbolType {
    if let Some(type_child) = node.child_by_field_name("type") {
        match type_child.kind() {
            "struct_type" => return SymbolType::Struct,
            "interface_type" => return SymbolType::Interface,
            _ => {}
        }
    }
    // Check the text for common patterns
    let text = node.utf8_text(source).unwrap_or("");
    if text.contains("struct") {
        SymbolType::Struct
    } else if text.contains("interface") {
        SymbolType::Interface
    } else {
        SymbolType::TypeAlias
    }
}

/// Extract individual methods from a Rust `impl` or `trait` block as separate
/// symbols, keeping the block itself indexable as `container_type`.
pub(super) fn extract_rust_block_methods(
    block_node: tree_sitter::Node<'_>,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
    container_type: SymbolType,
) {
    let name = extract_name(&block_node, source);
    symbols.push(RawSymbol::at(&block_node, name, container_type));

    let mut cursor = block_node.walk();

    for child in block_node.children(&mut cursor) {
        if child.kind() == "declaration_list" {
            let mut inner_cursor = child.walk();
            for item in child.children(&mut inner_cursor) {
                if item.kind() == "function_item" {
                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol::at(&item, name, SymbolType::Method));
                } else if let Some(sym_type) = classify_node(lang, item.kind()) {
                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol::at(&item, name, sym_type));
                }
            }
        }
    }
}

/// Extract methods from a Python `class_definition` as separate symbols.
///
/// Similar to how Rust `impl` methods are extracted: the class body is
/// walked for `function_definition` nodes (methods) which become individual
/// [`Method`] chunks, while the class itself remains indexable as a full
/// [`Class`] symbol for definition-oriented queries.
pub(super) fn extract_python_class_methods(
    class_node: tree_sitter::Node<'_>,
    source: &[u8],
    symbols: &mut Vec<RawSymbol>,
) {
    let name = extract_name(&class_node, source);
    symbols.push(RawSymbol::at(&class_node, name, SymbolType::Class));

    // The class body is a `block` child.
    let mut cursor = class_node.walk();
    for child in class_node.children(&mut cursor) {
        if child.kind() == "block" {
            let mut inner_cursor = child.walk();
            for item in child.children(&mut inner_cursor) {
                // Direct function_definition in class body = method.
                if item.kind() == "function_definition" {
                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol::at(&item, name, SymbolType::Method));
                }
                // Decorated methods: decorated_definition > function_definition
                else if item.kind() == "decorated_definition" {
                    let mut dec_cursor = item.walk();
                    for dec_child in item.children(&mut dec_cursor) {
                        if dec_child.kind() == "function_definition" {
                            let name = extract_name(&dec_child, source);
                            symbols.push(RawSymbol {
                                name,
                                symbol_type: SymbolType::Method,
                                // Use the decorated_definition range to include
                                // the decorator in the chunk.
                                start_byte: item.start_byte(),
                                end_byte: item.end_byte(),
                                start_row: item.start_position().row,
                                end_row: item.end_position().row,
                            });
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn extract_general_class_methods(
    class_node: tree_sitter::Node<'_>,
    source: &[u8],
    lang: Language,
    symbols: &mut Vec<RawSymbol>,
    class_sym_type: SymbolType,
) {
    // ALWAYS push the class/wrapper itself
    let name = extract_name(&class_node, source);
    symbols.push(RawSymbol::at(&class_node, name, class_sym_type));

    let mut cursor = class_node.walk();

    for child in class_node.children(&mut cursor) {
        let ckind = child.kind();
        if ckind.contains("body") || ckind.contains("block") || ckind == "declaration_list" {
            let mut inner_cursor = child.walk();
            for item in child.children(&mut inner_cursor) {
                if let Some(sym_type) = classify_node(lang, item.kind()) {
                    let mut end_byte = item.end_byte();
                    let mut end_row = item.end_position().row;

                    // Dart detached method body
                    if lang == Language::Dart && item.kind() == "method_signature" {
                        if let Some(next) = item.next_sibling() {
                            if next.kind() == "function_body" {
                                end_byte = next.end_byte();
                                end_row = next.end_position().row;
                            }
                        }
                    }

                    let name = extract_name(&item, source);
                    symbols.push(RawSymbol {
                        name,
                        symbol_type: sym_type,
                        start_byte: item.start_byte(),
                        end_byte,
                        start_row: item.start_position().row,
                        end_row,
                    });
                } else if lang == Language::Dart && item.kind() == "class_member" {
                    let mut cm_cursor = item.walk();
                    for cm_child in item.children(&mut cm_cursor) {
                        if let Some(sym_type) = classify_node(lang, cm_child.kind()) {
                            let mut end_byte = cm_child.end_byte();
                            let mut end_row = cm_child.end_position().row;

                            if cm_child.kind() == "method_signature" {
                                if let Some(next) = cm_child.next_sibling() {
                                    if next.kind() == "function_body" {
                                        end_byte = next.end_byte();
                                        end_row = next.end_position().row;
                                    }
                                }
                            }

                            let name = extract_name(&cm_child, source);
                            symbols.push(RawSymbol {
                                name,
                                symbol_type: sym_type,
                                start_byte: cm_child.start_byte(),
                                end_byte,
                                start_row: cm_child.start_position().row,
                                end_row,
                            });
                        }
                    }
                } else if lang == Language::Java && class_sym_type == SymbolType::Enum {
                    // Java enum methods can be nested under enum_body_declarations
                    // wrappers instead of appearing as direct enum_body children.
                    let mut java_cursor = item.walk();
                    for java_child in item.children(&mut java_cursor) {
                        if let Some(sym_type) = classify_node(lang, java_child.kind()) {
                            let name = extract_name(&java_child, source);
                            symbols.push(RawSymbol::at(&java_child, name, sym_type));
                            continue;
                        }

                        let mut java_inner_cursor = java_child.walk();
                        for java_inner in java_child.children(&mut java_inner_cursor) {
                            if let Some(sym_type) = classify_node(lang, java_inner.kind()) {
                                let name = extract_name(&java_inner, source);
                                symbols.push(RawSymbol::at(&java_inner, name, sym_type));
                            }
                        }
                    }
                }
            }
        }
    }
}
