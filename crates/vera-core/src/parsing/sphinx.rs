//! Sphinx-oriented preprocessing for reStructuredText.
//!
//! Tree-sitter gives us structural parsing (sections, directives, inline nodes),
//! but it does not resolve Sphinx semantics such as ``.. include::``.
//! This module performs lightweight source normalization before chunking:
//! - Recursively inline ``.. include::`` files (with cycle/depth guards)
//! - Normalize inline role syntax like ``:doc:`...``` into plain text

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use regex::Regex;

use crate::path_containment::{self, Containment};

const MAX_INCLUDE_DEPTH: usize = 16;

/// Preprocess RST text for chunking and embedding.
pub fn preprocess_rst(source: &str, current_file: &Path, repo_root: &Path) -> Result<String> {
    let mut stack = vec![current_file.to_path_buf()];
    let mut include_cache = HashMap::new();

    let expanded = resolve_includes_recursive(
        source,
        current_file,
        repo_root,
        &mut stack,
        &mut include_cache,
        0,
    )?;

    let with_toctree = normalize_toctree_blocks(&expanded);

    Ok(normalize_roles(&with_toctree))
}

fn include_re() -> &'static Regex {
    static INCLUDE_RE: OnceLock<Regex> = OnceLock::new();
    INCLUDE_RE.get_or_init(|| Regex::new(r"(?m)^[ \t]*\.\.\s+include::\s+(.+?)\s*$").unwrap())
}

fn role_re() -> &'static Regex {
    static ROLE_RE: OnceLock<Regex> = OnceLock::new();
    ROLE_RE.get_or_init(|| Regex::new(r":([A-Za-z0-9_:.+-]+):`([^`]+)`").unwrap())
}

fn toctree_entry_re() -> &'static Regex {
    static TOCTREE_ENTRY_RE: OnceLock<Regex> = OnceLock::new();
    TOCTREE_ENTRY_RE.get_or_init(|| Regex::new(r"^(.+?)\s*<([^>]+)>$").unwrap())
}

fn resolve_includes_recursive(
    text: &str,
    current_file: &Path,
    repo_root: &Path,
    stack: &mut Vec<PathBuf>,
    include_cache: &mut HashMap<PathBuf, String>,
    depth: usize,
) -> Result<String> {
    if depth >= MAX_INCLUDE_DEPTH {
        return Ok(text.to_string());
    }

    let mut output = String::with_capacity(text.len());
    let mut last_end = 0usize;

    for captures in include_re().captures_iter(text) {
        let full_match = captures
            .get(0)
            .expect("include regex always has full match");
        output.push_str(&text[last_end..full_match.start()]);

        let raw_ref = captures
            .get(1)
            .expect("include regex always has path capture")
            .as_str();
        let include_ref = strip_wrapping_quotes(raw_ref.trim());

        let replacement = match resolve_include_path(include_ref, current_file, repo_root)? {
            Some(include_path) => {
                if stack.contains(&include_path) {
                    full_match.as_str().to_string()
                } else {
                    let include_text = if let Some(cached) = include_cache.get(&include_path) {
                        cached.clone()
                    } else {
                        let bytes = std::fs::read(&include_path).with_context(|| {
                            format!("failed to read include: {}", include_path.display())
                        })?;
                        let decoded = String::from_utf8_lossy(&bytes).to_string();
                        include_cache.insert(include_path.clone(), decoded.clone());
                        decoded
                    };

                    stack.push(include_path.clone());
                    let resolved = resolve_includes_recursive(
                        &include_text,
                        &include_path,
                        repo_root,
                        stack,
                        include_cache,
                        depth + 1,
                    )?;
                    stack.pop();
                    resolved
                }
            }
            None => full_match.as_str().to_string(),
        };

        output.push_str(&replacement);
        last_end = full_match.end();
    }

    output.push_str(&text[last_end..]);
    Ok(output)
}

fn resolve_include_path(
    include_ref: &str,
    current_file: &Path,
    repo_root: &Path,
) -> Result<Option<PathBuf>> {
    let candidate = if include_ref.starts_with('/') {
        repo_root.join(include_ref.trim_start_matches('/'))
    } else {
        current_file.parent().unwrap_or(repo_root).join(include_ref)
    };

    let canonical_repo_root = repo_root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize repo root: {}", repo_root.display()))?;

    Ok(
        match path_containment::resolve_within(&canonical_repo_root, &candidate) {
            Containment::Inside(path) => Some(path),
            Containment::Escaped | Containment::Unresolved => None,
        },
    )
}

fn normalize_roles(text: &str) -> String {
    role_re()
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let role = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            let body = caps.get(2).map(|m| m.as_str()).unwrap_or("").trim();
            normalize_role(role, body)
        })
        .into_owned()
}

fn normalize_toctree_blocks(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.is_empty() {
        return String::new();
    }

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0usize;

    while i < lines.len() {
        let Some(base_indent) = toctree_indent(lines[i]) else {
            out.push(lines[i].to_string());
            i += 1;
            continue;
        };

        let mut j = i + 1;
        let mut options: Vec<(String, Option<String>)> = Vec::new();
        let mut entries: Vec<(String, String)> = Vec::new();

        while j < lines.len() {
            let line = lines[j];
            if line.trim().is_empty() {
                j += 1;
                continue;
            }

            let indent = leading_indent(line);
            if indent <= base_indent {
                break;
            }

            let trimmed = line.trim();
            if let Some((key, value)) = parse_toctree_option(trimmed) {
                options.push((key, value));
            } else if let Some((label, target)) = parse_toctree_entry(trimmed) {
                entries.push((label, target));
            }

            j += 1;
        }

        if options.is_empty() && entries.is_empty() {
            for original in &lines[i..j] {
                out.push((*original).to_string());
            }
        } else {
            out.push("[directive type=toctree]".to_string());
            for (key, value) in options {
                if let Some(value) = value {
                    out.push(format!(
                        "[directive_option key={key} value={}]",
                        normalize_inline_whitespace(&value)
                    ));
                } else {
                    out.push(format!("[directive_option key={key} value=true]"));
                }
            }

            for (label, target) in entries {
                out.push(format!(
                    "[link type=doc target={}] {}",
                    normalize_inline_whitespace(&target),
                    normalize_inline_whitespace(&label)
                ));
            }
        }

        i = j;
    }

    out.join("\n")
}

fn toctree_indent(line: &str) -> Option<usize> {
    let trimmed = line.trim_start_matches([' ', '\t']);
    if !trimmed.starts_with(".. toctree::") {
        return None;
    }
    Some(line.len() - trimmed.len())
}

fn leading_indent(line: &str) -> usize {
    line.len() - line.trim_start_matches([' ', '\t']).len()
}

fn parse_toctree_option(trimmed: &str) -> Option<(String, Option<String>)> {
    if !trimmed.starts_with(':') {
        return None;
    }

    let rest = &trimmed[1..];
    let (key, value) = rest.split_once(':')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }

    let value = value.trim();
    if value.is_empty() {
        Some((key.to_string(), None))
    } else {
        Some((key.to_string(), Some(value.to_string())))
    }
}

fn parse_toctree_entry(trimmed: &str) -> Option<(String, String)> {
    if trimmed.is_empty() || trimmed.starts_with(':') || trimmed.starts_with(".. ") {
        return None;
    }

    if let Some(caps) = toctree_entry_re().captures(trimmed) {
        let label = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
        let target = caps.get(2).map(|m| m.as_str()).unwrap_or("").trim();
        if target.is_empty() {
            return None;
        }
        let label = if label.is_empty() { target } else { label };
        return Some((label.to_string(), target.to_string()));
    }

    Some((trimmed.to_string(), trimmed.to_string()))
}

fn normalize_role(role: &str, body: &str) -> String {
    let role_lower = role.to_ascii_lowercase();
    if role_lower == "doc" || role_lower == "ref" {
        let (label, target) = if let Some((label, target)) = parse_role_target(body) {
            (
                normalize_inline_whitespace(&label),
                normalize_inline_whitespace(&target),
            )
        } else {
            let value = normalize_inline_whitespace(body);
            (value.clone(), value)
        };

        return format!("[link type={role_lower} target={target}] {label}");
    }

    normalize_role_body(body)
}

fn normalize_role_body(body: &str) -> String {
    if let Some((label, target)) = parse_role_target(body) {
        let label = normalize_inline_whitespace(&label);
        let target = normalize_inline_whitespace(&target);
        if label == target {
            label
        } else {
            format!("{label} ({target})")
        }
    } else {
        normalize_inline_whitespace(body)
    }
}

fn normalize_inline_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_role_target(body: &str) -> Option<(String, String)> {
    if !body.ends_with('>') {
        return None;
    }

    let start = body.rfind('<')?;
    if start == 0 {
        return None;
    }

    let label = body[..start].trim();
    let target = body[start + 1..body.len() - 1].trim();
    if target.is_empty() {
        return None;
    }

    if label.is_empty() {
        Some((target.to_string(), target.to_string()))
    } else {
        Some((label.to_string(), target.to_string()))
    }
}

fn strip_wrapping_quotes(input: &str) -> &str {
    let bytes = input.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &input[1..input.len() - 1];
        }
    }
    input
}

#[cfg(test)]
mod tests {
    use super::preprocess_rst;
    use std::fs;

    #[test]
    fn preprocess_inlines_relative_include() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("sub/part.rst"), "Included content\n").unwrap();

        let source_path = root.join("guide.rst");
        let source = "Title\n=====\n\n.. include:: sub/part.rst\n";
        let processed = preprocess_rst(source, &source_path, root).unwrap();

        assert!(processed.contains("Included content"));
        assert!(!processed.contains(".. include:: sub/part.rst"));
    }

    #[test]
    fn preprocess_inlines_root_relative_include() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::create_dir_all(root.join("docs/includes")).unwrap();
        fs::write(
            root.join("docs/includes/common.rst.inc"),
            "Common snippet\n",
        )
        .unwrap();

        let source_path = root.join("docs/guide.rst");
        let source = ".. include:: /docs/includes/common.rst.inc\n";
        let processed = preprocess_rst(source, &source_path, root).unwrap();

        assert!(processed.contains("Common snippet"));
    }

    #[test]
    fn preprocess_normalizes_sphinx_roles() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let source_path = root.join("guide.rst");
        let source = "See :doc:`Routing </routing>` and :ref:`parameters <config-parameters>`.";
        let processed = preprocess_rst(source, &source_path, root).unwrap();

        assert!(processed.contains("[link type=doc target=/routing] Routing"));
        assert!(processed.contains("[link type=ref target=config-parameters] parameters"));
        assert!(!processed.contains(":doc:`"));
        assert!(!processed.contains(":ref:`"));
    }

    #[test]
    fn preprocess_normalizes_multiline_roles() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let source_path = root.join("guide.rst");
        let source = "See :doc:`Config component\n</components/config>` for details.";
        let processed = preprocess_rst(source, &source_path, root).unwrap();

        assert!(processed.contains("[link type=doc target=/components/config] Config component"));
    }

    #[test]
    fn preprocess_normalizes_role_without_custom_label() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let source_path = root.join("guide.rst");
        let source = "See :doc:`/components/config` for details.";
        let processed = preprocess_rst(source, &source_path, root).unwrap();

        assert!(processed.contains("[link type=doc target=/components/config] /components/config"));
    }

    #[test]
    fn preprocess_normalizes_toctree_directive() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let source_path = root.join("guide.rst");
        let source = r#".. toctree::
   :maxdepth: 2
   :caption: Components

   /components/config
   Routing </routing>
"#;

        let processed = preprocess_rst(source, &source_path, root).unwrap();

        assert!(processed.contains("[directive type=toctree]"));
        assert!(processed.contains("[directive_option key=maxdepth value=2]"));
        assert!(processed.contains("[directive_option key=caption value=Components]"));
        assert!(processed.contains("[link type=doc target=/components/config] /components/config"));
        assert!(processed.contains("[link type=doc target=/routing] Routing"));
        assert!(!processed.contains(".. toctree::"));
    }
}
