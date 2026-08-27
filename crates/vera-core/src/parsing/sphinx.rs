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
        ancestor_ids: Vec::new(),
        stack_ids: FileSet::default(),
        file_ids: HashMap::new(),
        expansion_cache: &mut expansion_cache,
        output_budget: include_output_budget(max_file_size_bytes),
        max_file_size_bytes,
    };

    // Intern the seed that was actually pushed, not `current_file`: they differ
    // whenever canonicalization changes the path (macOS /var, symlinked
    // checkouts), and every dependency id comes from the canonical form that
    // `resolve_include_path` returns.
    let seed = context.stack[0].clone();
    let root_id = context.file_id(&seed);
    context.ancestor_ids.push(root_id);
    context.stack_ids.insert(root_id);

    let (expanded, _deps) =
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
    /// The same ancestors as `stack`, kept as ids so the bitset can be rebuilt
    /// after a pop without cloning and re-hashing every ancestor path.
    ancestor_ids: Vec<usize>,
    /// `ancestor_ids` as a bitset, so validity checks are word operations
    /// rather than repeated path comparisons.
    stack_ids: FileSet,
    file_ids: HashMap<PathBuf, usize>,
    expansion_cache: &'a mut HashMap<PathBuf, CachedExpansion>,
    output_budget: usize,
    max_file_size_bytes: u64,
}

/// A set of include files, as a bitset over ids interned per preprocess call.
///
/// These sets are unioned at every level of the recursion, so representation
/// dominates: with `HashSet<PathBuf>` the same 16-file cyclic fixture that
/// takes 88 ms here took 4.1 s, and forcing every cache lookup to hit did not
/// move that number — the cost was copying and hashing paths, not re-expanding.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct FileSet {
    words: Vec<u64>,
}

impl FileSet {
    fn insert(&mut self, id: usize) {
        let (word, bit) = (id / 64, id % 64);
        if self.words.len() <= word {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << bit;
    }

    fn union_with(&mut self, other: &FileSet) {
        if self.words.len() < other.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (slot, word) in self.words.iter_mut().zip(&other.words) {
            *slot |= word;
        }
    }

    fn intersects(&self, other: &FileSet) -> bool {
        self.words.iter().zip(&other.words).any(|(a, b)| a & b != 0)
    }

    fn is_subset_of(&self, other: &FileSet) -> bool {
        self.words
            .iter()
            .enumerate()
            .all(|(i, word)| word & !other.words.get(i).copied().unwrap_or(0) == 0)
    }
}

/// A cached expansion, with the conditions under which it may be reused.
///
/// The path alone is not injective: whether a cycle guard fires inside an
/// expansion is decided by the chain the file was reached through, so one
/// branch's result must not be replayed into a branch with different ancestors.
///
/// Keying on the whole chain would fix that and give back the fan-out bound
/// this cache exists for: a shared include reached through two parents has two
/// different chains, so a convergent graph re-expands once per distinct path.
/// Measured on a shared subtree behind 32 parents, that was 18.1 ms against
/// 4.2 ms with the cache reused.
///
/// So the entry records what its content actually depended on:
///
/// - `suppressed`: includes left as literal directives because they were on the
///   stack. The result holds only where those are still ancestors.
/// - `inlined`: includes expanded within it. The result holds only where none of
///   them is an ancestor, since there they would have been suppressed instead.
///
/// In an acyclic graph `suppressed` is empty and no inlined file can be an
/// ancestor, so every branch reuses the entry exactly as before.
struct CachedExpansion {
    text: Arc<str>,
    deps: ExpansionDeps,
    /// The depth this expansion was produced at. Only consulted when the depth
    /// cap truncated it, where the result describes that depth rather than the
    /// file. Refusing to cache those outright instead costs an exponential
    /// re-expansion once a graph is deep enough to reach the cap: on the
    /// 16-file cyclic fixture that was 5.5 s against 88 ms.
    built_at_depth: usize,
}

impl CachedExpansion {
    fn is_valid_for(&self, stack: &FileSet, depth: usize) -> bool {
        if self.deps.truncated_by_depth {
            // Describes the cap at the depth it was built at, nothing more.
            if self.built_at_depth != depth {
                return false;
            }
        } else if depth + self.deps.height >= MAX_INCLUDE_DEPTH {
            // Fully expanded, but its deepest descendant would not fit here.
            // Replaying it would return more than this position should hold.
            return false;
        }
        self.deps.is_valid_for(stack)
    }
}

impl<'a> IncludeResolutionContext<'a> {
    fn push_ancestor(&mut self, path: PathBuf, id: usize) {
        self.stack.push(path);
        self.ancestor_ids.push(id);
        self.stack_ids.insert(id);
    }

    /// Drop the innermost ancestor and rebuild the bitset from the ids that
    /// remain.
    ///
    /// A bitset cannot be un-set by id alone — the same file can appear at more
    /// than one depth — and the stack is at most `MAX_INCLUDE_DEPTH` deep, so
    /// rebuilding from ids is cheap and cannot drift from `stack`.
    fn pop_ancestor(&mut self) {
        self.stack.pop();
        self.ancestor_ids.pop();
        self.stack_ids = FileSet::default();
        for id in &self.ancestor_ids {
            self.stack_ids.insert(*id);
        }
    }

    fn file_id(&mut self, path: &Path) -> usize {
        if let Some(id) = self.file_ids.get(path) {
            return *id;
        }
        let id = self.file_ids.len();
        self.file_ids.insert(path.to_path_buf(), id);
        id
    }
}

/// What one expansion depended on, propagated up so callers can cache safely.
#[derive(Debug, Default, Clone)]
struct ExpansionDeps {
    suppressed: FileSet,
    inlined: FileSet,
    /// Truncated by the depth cap. That is a property of how deep this path
    /// already was rather than of the file, so such a result may only be reused
    /// at the same depth.
    truncated_by_depth: bool,
    /// Levels of include nesting below this expansion, 0 for one with none.
    ///
    /// A fully expanded entry is still depth-sensitive: replayed at a site deep
    /// enough that fresh evaluation would have hit the cap, it returns more
    /// than the expansion at that position should contain, and the result then
    /// depends on which branch populated the cache first.
    height: usize,
}

impl ExpansionDeps {
    fn absorb(&mut self, other: &ExpansionDeps) {
        self.suppressed.union_with(&other.suppressed);
        self.inlined.union_with(&other.inlined);
        self.truncated_by_depth |= other.truncated_by_depth;
        self.height = self.height.max(other.height + 1);
    }

    /// Whether an expansion with these dependencies is valid under `stack`.
    fn is_valid_for(&self, stack: &FileSet) -> bool {
        self.suppressed.is_subset_of(stack) && !self.inlined.intersects(stack)
    }
}

fn resolve_includes_recursive(
    text: &str,
    current_file: &Path,
    depth: usize,
    output_budget: usize,
    context: &mut IncludeResolutionContext<'_>,
) -> Result<(String, ExpansionDeps)> {
    let mut remaining_output_bytes = output_budget;
    let mut deps = ExpansionDeps::default();

    if depth >= MAX_INCLUDE_DEPTH {
        let mut output = String::new();
        append_with_budget(&mut output, text, &mut remaining_output_bytes);
        deps.truncated_by_depth = true;
        return Ok((output, deps));
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
                    let include_id = context.file_id(&include_path);
                    if context.stack.contains(&include_path) {
                        deps.suppressed.insert(include_id);
                        None
                    } else if let Some(cached) = context
                        .expansion_cache
                        .get(&include_path)
                        .filter(|cached| cached.is_valid_for(&context.stack_ids, depth + 1))
                    {
                        deps.inlined.insert(include_id);
                        deps.absorb(&cached.deps);
                        Some(Arc::clone(&cached.text))
                    } else {
                        match read_include_file(&include_path, context.max_file_size_bytes)? {
                            Some(include_text) => {
                                context.push_ancestor(include_path.clone(), include_id);
                                let (resolved, child_deps) = resolve_includes_recursive(
                                    &include_text,
                                    &include_path,
                                    depth + 1,
                                    context.output_budget,
                                    context,
                                )?;
                                context.pop_ancestor();
                                let resolved: Arc<str> = Arc::from(resolved);
                                context.expansion_cache.insert(
                                    include_path.clone(),
                                    CachedExpansion {
                                        text: Arc::clone(&resolved),
                                        deps: child_deps.clone(),
                                        built_at_depth: depth + 1,
                                    },
                                );
                                deps.inlined.insert(include_id);
                                deps.absorb(&child_deps);
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
    Ok((output, deps))
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
    use std::path::Path;

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

    /// Build the diamond-with-a-back-edge fixture and expand `a.rst`.
    ///
    /// `b` and `c` both include `d`; `d` includes `b`. Whether `d`'s expansion
    /// suppresses `b` depends on which branch reached it, so `d` is exactly the
    /// file whose result must not be shared between branches.
    fn expand_diamond(root: &Path, first: &str, second: &str) -> String {
        fs::write(root.join("b.rst"), "B-BODY-UNIQUE\n.. include:: d.rst\n").unwrap();
        fs::write(root.join("c.rst"), "C-BODY\n.. include:: d.rst\n").unwrap();
        fs::write(root.join("d.rst"), "D-BODY\n.. include:: b.rst\n").unwrap();

        let source = format!("A-BODY\n.. include:: {first}\n.. include:: {second}\n");
        let source_path = root.join("a.rst");
        fs::write(&source_path, &source).unwrap();
        preprocess_rst_with_limit(&source, &source_path, root, 4096).unwrap()
    }

    #[test]
    fn diamond_includes_do_not_depend_on_directive_order() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();

        let b_then_c = expand_diamond(first.path(), "b.rst", "c.rst");
        let c_then_b = expand_diamond(second.path(), "c.rst", "b.rst");

        // Every branch that can reach b.rst must inline it. Caching d.rst's
        // expansion under its path alone replayed the branch where b was
        // suppressed into the branch where it was not, so one of these was 1.
        assert_eq!(
            b_then_c.matches("B-BODY-UNIQUE").count(),
            2,
            "b.rst is reachable through both branches:\n{b_then_c}"
        );
        assert_eq!(
            c_then_b.matches("B-BODY-UNIQUE").count(),
            2,
            "b.rst is reachable through both branches:\n{c_then_b}"
        );

        // Reordering two directives may reorder the output, but it must not
        // change what the output contains — that is what reaches content_hash.
        let sorted = |text: &str| {
            let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            lines.sort_unstable();
            lines.join("\n")
        };
        assert_eq!(
            sorted(&b_then_c),
            sorted(&c_then_b),
            "same files, same include graph, different directive order:\n--- b,c ---\n{b_then_c}\n--- c,b ---\n{c_then_b}"
        );
    }

    #[test]
    fn a_shared_include_without_a_cycle_is_still_cached() {
        // The negative half: only expansions whose content depended on the call
        // path are excluded. A plain shared include has no cycle guard inside
        // it, so it must still be cached and must still expand everywhere.
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(root.join("shared.rst"), "SHARED-BODY\n").unwrap();
        fs::write(root.join("x.rst"), "X-BODY\n.. include:: shared.rst\n").unwrap();
        fs::write(root.join("y.rst"), "Y-BODY\n.. include:: shared.rst\n").unwrap();

        let source = "TOP\n.. include:: x.rst\n.. include:: y.rst\n";
        let source_path = root.join("top.rst");
        fs::write(&source_path, source).unwrap();
        let processed = preprocess_rst_with_limit(source, &source_path, root, 4096).unwrap();

        assert_eq!(
            processed.matches("SHARED-BODY").count(),
            2,
            "a cycle-free shared include expands on every branch:\n{processed}"
        );
        assert!(processed.contains("X-BODY") && processed.contains("Y-BODY"));
    }

    /// Build a tree where `shared.rst` is reachable both near the root and at
    /// the bottom of a chain long enough that expanding it there would cross
    /// `MAX_INCLUDE_DEPTH`, and expand it with the two branches in `order`.
    fn expand_shallow_and_deep(root: &Path, order: [&str; 2]) -> String {
        // shared.rst has a subtree of its own, so where it is expanded decides
        // how much of that subtree fits under the cap.
        for i in 0..4 {
            fs::write(
                root.join(format!("sub{i}.rst")),
                format!("SUB{i}\n.. include:: sub{}.rst\n", i + 1),
            )
            .unwrap();
        }
        fs::write(root.join("sub4.rst"), "SUB-LEAF\n").unwrap();
        fs::write(root.join("shared.rst"), "SHARED\n.. include:: sub0.rst\n").unwrap();

        // A chain that reaches shared.rst just under the cap.
        let chain = super::MAX_INCLUDE_DEPTH - 2;
        for i in 0..chain {
            let next = if i + 1 == chain {
                "shared.rst".to_string()
            } else {
                format!("deep{}.rst", i + 1)
            };
            fs::write(
                root.join(format!("deep{i}.rst")),
                format!("DEEP{i}\n.. include:: {next}\n"),
            )
            .unwrap();
        }

        let source = format!(
            "TOP\n.. include:: {}\n.. include:: {}\n",
            order[0], order[1]
        );
        let source_path = root.join("top.rst");
        fs::write(&source_path, &source).unwrap();
        preprocess_rst_with_limit(&source, &source_path, root, 4096).unwrap()
    }

    #[test]
    fn a_cached_expansion_is_not_replayed_where_the_depth_cap_would_bite() {
        let shallow_first = tempfile::tempdir().unwrap();
        let deep_first = tempfile::tempdir().unwrap();

        // Same files, same graph. Only which branch populates the cache first
        // differs, and that must not decide what the other branch gets: an
        // entry expanded near the root reaches deeper than one expanded at the
        // bottom of the chain is allowed to.
        let a = expand_shallow_and_deep(shallow_first.path(), ["shared.rst", "deep0.rst"]);
        let b = expand_shallow_and_deep(deep_first.path(), ["deep0.rst", "shared.rst"]);

        let sorted = |text: &str| {
            let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            lines.sort_unstable();
            lines.join("\n")
        };
        assert_eq!(
            sorted(&a),
            sorted(&b),
            "cache population order changed the expansion:\n--- shallow first ---\n{a}\n--- deep first ---\n{b}"
        );
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
