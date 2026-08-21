//! Symbol-name extraction from tree-sitter nodes.

/// Node kinds whose text is a usable symbol name in the extract_name fallback.
const PRIMARY_NAME_KINDS: &[&str] = &[
    "identifier",
    "type_identifier",
    "property_identifier",
    "simple_identifier",
    "word",
    "constant",
];

/// Superset of PRIMARY_NAME_KINDS accepted by name_from_node (adds
/// field_identifier plus the generic `name`/`variable` wrapper kinds).
const NAME_KINDS: &[&str] = &[
    "identifier",
    "type_identifier",
    "property_identifier",
    "field_identifier",
    "simple_identifier",
    "word",
    "constant",
    "name",
    "variable",
];

/// Extract the name of a symbol from a tree-sitter node.
///
/// Looks for the first `name` or `identifier`-type child node.
pub(crate) fn extract_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    if node.kind() == "impl_item" {
        return extract_impl_name(node, source);
    }

    // Fortran containers carry their names on their header child rather than
    // directly on the body node.
    let header_kind = match node.kind() {
        "module" => Some("module_statement"),
        "program" => Some("program_statement"),
        _ => None,
    };
    if let Some(header_kind) = header_kind {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == header_kind {
                let mut header_cursor = child.walk();
                for name in child.children(&mut header_cursor) {
                    if name.kind() == "name" {
                        return name.utf8_text(source).ok().map(str::to_string);
                    }
                }
            }
        }
    }

    // HCL block name (second child that is an identifier or string_lit)
    if node.kind() == "block" {
        let mut cursor = node.walk();
        let mut found_type = false;
        for child in node.children(&mut cursor) {
            if !found_type && child.kind() == "identifier" {
                found_type = true;
                continue;
            }
            if found_type && (child.kind() == "string_lit" || child.kind() == "identifier") {
                return child
                    .utf8_text(source)
                    .ok()
                    .map(|s| s.trim_matches('"').to_string());
            }
        }
    }

    // Try common name field patterns
    for field in &["name", "declarator"] {
        if let Some(child) = node.child_by_field_name(field) {
            return name_from_node(&child, source);
        }
    }

    // Protobuf names
    if node.kind() == "message"
        || node.kind() == "enum"
        || node.kind() == "service"
        || node.kind() == "rpc"
        || node.kind().ends_with("_definition")
    {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind().ends_with("_name") {
                let mut inner = child.walk();
                for c in child.children(&mut inner) {
                    if c.kind() == "identifier" {
                        return c.utf8_text(source).ok().map(|s| s.to_string());
                    }
                }
                return child.utf8_text(source).ok().map(|s| s.to_string());
            }
        }
    }
    // Dart: method_signature -> function_signature -> name
    if node.kind() == "method_signature" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "function_signature" {
                if let Some(name_child) = child.child_by_field_name("name") {
                    return name_from_node(&name_child, source);
                }
            }
        }
    }
    // Fallback: look for first identifier child
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if PRIMARY_NAME_KINDS.contains(&child.kind()) {
            return Some(child.utf8_text(source).ok()?.to_string());
        }
    }
    None
}

fn extract_impl_name(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let text = node.utf8_text(source).ok()?;
    let header = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    if !header.starts_with("impl") {
        return None;
    }

    let mut header = header
        .trim_end_matches('{')
        .trim_end_matches("where")
        .trim();
    if let Some((prefix, _)) = header.split_once(" where ") {
        header = prefix.trim();
    }

    let cleaned = header.split_whitespace().collect::<Vec<_>>().join(" ");
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Extract a name string from a node, handling nested patterns.
fn name_from_node(node: &tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    let kind = node.kind();
    // Direct identifier nodes
    if NAME_KINDS.contains(&kind) {
        return Some(node.utf8_text(source).ok()?.to_string());
    }
    // Pointer declarators, reference declarators, etc. (C/C++)
    if kind.contains("declarator") {
        if let Some(inner) = node.child_by_field_name("declarator") {
            return name_from_node(&inner, source);
        }
        // Or a direct identifier child
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" || child.kind() == "field_identifier" {
                return Some(child.utf8_text(source).ok()?.to_string());
            }
        }
    }
    None
}
