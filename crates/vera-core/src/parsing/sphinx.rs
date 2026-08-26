//! Sphinx-oriented preprocessing for reStructuredText.
//!
//! Tree-sitter gives us structural parsing (sections, directives, inline nodes),
//! but it does not resolve Sphinx semantics such as ``.. include::``.
//! This module performs lightweight source normalization before chunking:
//! - Recursively inline ``.. include::`` files (with cycle/depth/size guards)
//! - Normalize inline role syntax like ``:doc:`...``` into plain text

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use regex::Regex;

use crate::path_containment::{self, Containment};

const MAX_INCLUDE_DEPTH: usize = 16;
const INCLUDE_OUTPUT_BUDGET_MULTIPLIER: u64 = 4;

/// Preprocess RST text for chunking and embedding.
pub fn preprocess_rst(source: &str, current_file: &Path, repo_root: &Path) -> Result<String> {
    preprocess_rst_with_limit(
        source,
        current_file,
        repo_root,
        crate::config::DEFAULT_MAX_FILE_SIZE_BYTES,
    )
}

/// Preprocess RST text with a configured source and include-size limit.
///
/// Expanded output is capped at four times `max_file_size_bytes`. The input
/// file is already limited to that size by discovery, so this allows normal
/// include composition while bounding recursive fan-out.
pub(crate) fn preprocess_rst_with_limit(
    source: &str,
    current_file: &Path,
    repo_root: &Path,
    max_file_size_bytes: u64,
) -> Result<String> {
    // Seed the stack with the canonicalized path: resolve_include_path
    // returns canonicalized paths (resolve_within), so a non-canonical seed
    // (macOS /var -> /private/var, symlinked checkouts) never compares equal
    // and include cycles expand to MAX_INCLUDE_DEPTH instead of terminating.
    let stack_seed = current_file
        .canonicalize()
        .unwrap_or_else(|_| current_file.to_path_buf());
    let mut stack = vec![stack_seed];
    let mut expansion_cache = HashMap::new();
    let canonical_repo_root = repo_root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize repo root: {}", repo_root.display()))?;
    let mut context = IncludeResolutionContext {
        canonical_repo_root: &canonical_repo_root,
        stack: &mut stack,
        expansion_cache: &mut expansion_cache,
        output_budget: include_output_budget(max_file_size_bytes),
        max_file_size_bytes,
    };

    let expanded =
        resolve_includes_recursive(source, current_file, 0, context.output_budget, &mut context)?;

    let with_toctree = normalize_toctree_blocks(&expanded);

    Ok(normalize_roles(&with_toctree))
}

fn include_output_budget(max_file_size_bytes: u64) -> usize {
    max_file_size_bytes
        .saturating_mul(INCLUDE_OUTPUT_BUDGET_MULTIPLIER)
        .min(usize::MAX as u64) as usize
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

struct IncludeResolutionContext<'a> {
    canonical_repo_root: &'a Path,
    stack: &'a mut Vec<PathBuf>,
    expansion_cache: &'a mut HashMap<PathBuf, Arc<str>>,
    output_budget: usize,
    max_file_size_bytes: u64,
}

fn resolve_includes_recursive(
    text: &str,
    current_file: &Path,
    depth: usize,
    output_budget: usize,
    context: &mut IncludeResolutionContext<'_>,
) -> Result<String> {
    let mut remaining_output_bytes = output_budget;

    if depth >= MAX_INCLUDE_DEPTH {
        let mut output = String::new();
        append_with_budget(&mut output, text, &mut remaining_output_bytes);
        return Ok(output);
    }

    let mut output = String::with_capacity(text.len().min(remaining_output_bytes));
    let mut last_end = 0usize;

    for captures in include_re().captures_iter(text) {
        if remaining_output_bytes == 0 {
            break;
        }

        let full_match = captures
            .get(0)
            .expect("include regex always has full match");
        append_with_budget(
            &mut output,
            &text[last_end..full_match.start()],
            &mut remaining_output_bytes,
        );

        if remaining_output_bytes == 0 {
            break;
        }

        let raw_ref = captures
            .get(1)
            .expect("include regex always has path capture")
            .as_str();
        let include_ref = strip_wrapping_quotes(raw_ref.trim());

        let replacement =
            match resolve_include_path(include_ref, current_file, context.canonical_repo_root)? {
                Some(include_path) => {
                    if context.stack.contains(&include_path) {
                        None
                    } else if let Some(cached) = context.expansion_cache.get(&include_path) {
                        Some(Arc::clone(cached))
                    } else {
                        match read_include_file(&include_path, context.max_file_size_bytes)? {
                            Some(include_text) => {
                                context.stack.push(include_path.clone());
                                let resolved = resolve_includes_recursive(
                                    &include_text,
                                    &include_path,
                                    depth + 1,
                                    context.output_budget,
                                    context,
                                )?;
                                context.stack.pop();
                                let resolved: Arc<str> = Arc::from(resolved);
                                context
                                    .expansion_cache
                                    .insert(include_path.clone(), Arc::clone(&resolved));
                                Some(resolved)
                            }
                            None => None,
                        }
                    }
                }
                None => None,
            };

        append_with_budget(
            &mut output,
            replacement.as_deref().unwrap_or(full_match.as_str()),
            &mut remaining_output_bytes,
        );
        last_end = full_match.end();
    }

    append_with_budget(&mut output, &text[last_end..], &mut remaining_output_bytes);
    Ok(output)
}

fn read_include_file(path: &Path, max_file_size_bytes: u64) -> Result<Option<String>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to read include: {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(max_file_size_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read include: {}", path.display()))?;

    if bytes.len() as u64 > max_file_size_bytes {
        return Ok(None);
    }

    Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
}

fn append_with_budget(output: &mut String, text: &str, remaining: &mut usize) {
    let byte_count = text.len().min(*remaining);
    let mut end = byte_count;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    if end == 0 && byte_count > 0 {
        *remaining = 0;
        return;
    }
    output.push_str(&text[..end]);
    *remaining -= end;
}

fn resolve_include_path(
    include_ref: &str,
    current_file: &Path,
    canonical_repo_root: &Path,
) -> Result<Option<PathBuf>> {
    let candidate = if include_ref.starts_with('/') {
        canonical_repo_root.join(include_ref.trim_start_matches('/'))
    } else {
        current_file
            .parent()
            .unwrap_or(canonical_repo_root)
            .join(include_ref)
    };

    Ok(
        match path_containment::resolve_within(canonical_repo_root, &candidate) {
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
    use super::{preprocess_rst, preprocess_rst_with_limit};
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
    fn preprocess_bounds_dag_fanout_output() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let source_path = root.join("root.rst");
        let levels = ["a", "b", "c", "d", "e", "f", "g", "h"];
        for (index, level) in levels.iter().enumerate() {
            let content = match levels.get(index + 1) {
                Some(next) => format!(".. include:: {next}.rst\n").repeat(3),
                None => "leaf content\n".to_string(),
            };
            fs::write(root.join(format!("{level}.rst")), content).unwrap();
        }

        let source = ".. include:: a.rst\n".repeat(3);
        fs::write(&source_path, &source).unwrap();
        let processed = preprocess_rst_with_limit(&source, &source_path, root, 64).unwrap();

        assert!(processed.len() <= 256);
        assert!(processed.contains("leaf content"));
    }

    #[test]
    fn preprocess_skips_oversized_include() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let source_path = root.join("root.rst");
        let include_path = root.join("large.rst");
        let source = "before\n.. include:: large.rst\nafter\n";

        fs::write(&include_path, "x".repeat(33)).unwrap();
        let processed = preprocess_rst_with_limit(source, &source_path, root, 32).unwrap();

        assert!(processed.contains("before"));
        assert!(processed.contains(".. include:: large.rst"));
        assert!(processed.contains("after"));
        assert!(!processed.contains(&"x".repeat(33)));
    }

    #[test]
    fn preprocess_terminates_on_include_cycle() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let source_path = root.join("root.rst");
        let child_path = root.join("child.rst");

        fs::write(&child_path, "child\n.. include:: root.rst\n").unwrap();
        let source = "root\n.. include:: child.rst\n";
        fs::write(&source_path, source).unwrap();
        let processed = preprocess_rst_with_limit(source, &source_path, root, 128).unwrap();

        assert!(processed.contains("root"));
        assert!(processed.contains("child"));
        assert!(processed.contains(".. include:: root.rst"));
        assert!(processed.len() < 128 * 4);
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
