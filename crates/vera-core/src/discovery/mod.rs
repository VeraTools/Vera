//! File discovery: walk directory tree, respect .gitignore, exclude defaults.
//!
//! This module discovers source files for indexing by:
//! - Walking the directory tree using the `ignore` crate (respects .gitignore)
//! - Applying default exclusion patterns (.git, node_modules, target, etc.)
//! - Detecting and skipping binary files (content heuristics + extension checks)
//! - Handling large files (skip above configurable threshold)
//!
//! # Architecture
//!
//! Uses the `ignore` crate which natively supports:
//! - `.gitignore` patterns (hierarchical, all levels)
//! - Global gitignore
//! - `.ignore` files
//! - Custom override patterns

use std::path::Component;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::Match;
use ignore::WalkBuilder;
use ignore::gitignore::{Gitignore, GitignoreBuilder, Glob as GitignoreGlob};
use ignore::overrides::{Override, OverrideBuilder};
use tracing::{debug, warn};

use crate::CancellationToken;
use crate::config::IndexingConfig;

/// A discovered source file ready for indexing.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    /// Absolute path to the file.
    pub absolute_path: PathBuf,
    /// Repository-relative path (for chunk metadata).
    pub relative_path: String,
    /// File size in bytes.
    pub size: u64,
}

/// Result of the file discovery process.
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Files that passed all filters and are ready to index.
    pub files: Vec<DiscoveredFile>,
    /// Number of files skipped because they were binary.
    pub binary_skipped: usize,
    /// Number of files skipped because they exceeded the size threshold.
    pub large_skipped: usize,
    /// Relative paths and sizes (bytes) of files skipped because they exceeded the size threshold.
    pub large_skipped_paths: Vec<(String, u64)>,
    /// Number of files skipped due to read errors (permissions, etc.).
    pub error_skipped: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PathDecision {
    Indexed,
    Excluded,
    Missing,
    Directory,
    OutsideRoot,
}

impl std::fmt::Display for PathDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Indexed => "indexed",
            Self::Excluded => "excluded",
            Self::Missing => "missing",
            Self::Directory => "directory",
            Self::OutsideRoot => "outside_root",
        };
        write!(f, "{value}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PathReason {
    Indexed,
    Missing,
    Directory,
    NonRegularFile,
    OutsideRoot,
    DefaultExclude,
    CliExclude,
    Veraignore,
    DotIgnore,
    Gitignore,
    GitExclude,
    GitGlobal,
    RstIncludeFragment,
    TooLarge,
    EmptyFile,
    BinaryExtension,
    BinaryContent,
}

impl std::fmt::Display for PathReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Indexed => "indexed",
            Self::Missing => "missing",
            Self::Directory => "directory",
            Self::NonRegularFile => "non_regular_file",
            Self::OutsideRoot => "outside_root",
            Self::DefaultExclude => "default_exclude",
            Self::CliExclude => "cli_exclude",
            Self::Veraignore => "veraignore",
            Self::DotIgnore => "dot_ignore",
            Self::Gitignore => "gitignore",
            Self::GitExclude => "git_exclude",
            Self::GitGlobal => "git_global",
            Self::RstIncludeFragment => "rst_include_fragment",
            Self::TooLarge => "too_large",
            Self::EmptyFile => "empty_file",
            Self::BinaryExtension => "binary_extension",
            Self::BinaryContent => "binary_content",
        };
        write!(f, "{value}")
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PathExplanation {
    pub input_path: String,
    pub absolute_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_path: Option<String>,
    pub decision: PathDecision,
    pub reason: PathReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct IgnoreStrategy {
    use_gitignore: bool,
    use_veraignore: bool,
}

#[derive(Debug, Clone)]
struct IgnoreMatchExplanation {
    reason: PathReason,
    source: Option<String>,
    pattern: Option<String>,
    details: Option<String>,
}

#[derive(Debug, Clone)]
enum IgnoreDecision {
    None,
    Whitelist(IgnoreMatchExplanation),
    Ignore(IgnoreMatchExplanation),
}

/// Known binary file extensions (skip without content inspection).
///
/// Categories: compiled objects, archives, images, audio/video, fonts,
/// documents, databases, JVM bytecode, .NET, lock files, compiled Python, WASM.
#[rustfmt::skip]
const BINARY_EXTENSIONS: &[&str] = &[
    "o", "obj", "a", "lib", "so", "dylib", "dll", "exe", "com",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "zst",
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "tiff", "tif",
    "mp3", "mp4", "avi", "mov", "wav", "flac", "ogg", "webm", "mkv",
    "ttf", "otf", "woff", "woff2", "eot",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "db", "sqlite", "sqlite3",
    "class", "jar", "war", "ear",
    "pdb", "nupkg",
    "lockb",
    "pyc", "pyo",
    "wasm",
    "bin", "dat", "pak",
];

/// Read a source file as UTF-8, replacing invalid sequences lossily.
///
/// Discovery already excludes binary files, so a file that reaches this
/// helper is considered text; strict `read_to_string` would skip text files
/// with stray invalid bytes (mixed encodings, latin-1 notes). Lossy decoding
/// indexes them instead of silently dropping them from the index.
pub fn read_source_lossy(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    match String::from_utf8(bytes) {
        Ok(text) => Ok(text),
        Err(err) => Ok(String::from_utf8_lossy(err.as_bytes()).into_owned()),
    }
}

/// The directory exclusions discovery applies, reusable without walking a tree.
///
/// Built from the same [`build_overrides`] patterns `discover_files` installs on
/// its walker, so a caller that has to classify a single path (the MCP file
/// watcher deciding whether an event is worth an update cycle) cannot drift from
/// what indexing actually skips.
pub struct ExclusionMatcher {
    root: PathBuf,
    overrides: Override,
}

impl ExclusionMatcher {
    /// Build a matcher rooted at `root` for the given indexing configuration.
    pub fn new(root: &Path, config: &IndexingConfig) -> Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("failed to resolve path: {}", root.display()))?;
        let overrides = build_overrides(&root, config)?;
        Ok(Self { root, overrides })
    }

    /// True if `path` lies in, or is, a directory discovery excludes.
    ///
    /// Paths outside the root are reported as not excluded: the caller cannot
    /// classify them against these patterns, and treating them as excluded would
    /// silently drop work.
    ///
    /// The default patterns are directory-only (`!target/`), so `path` is probed
    /// rather than assumed to be a directory: `discover_files` indexes a regular
    /// file named `build` or `target`, and classifying it as excluded here would
    /// make the watcher ignore edits to a file that is in the index. A path that
    /// no longer exists is treated as a file for the same reason, so deleting
    /// such a file still triggers an update; its parents are matched as
    /// directories either way, which keeps deletions under `target/` excluded.
    pub fn is_excluded(&self, path: &Path) -> bool {
        let Ok(relative) = path.strip_prefix(&self.root) else {
            return false;
        };
        if path.is_dir() && !self.overrides.matched(relative, true).is_none() {
            return true;
        }
        override_matches_path_or_any_parents(&self.overrides, relative)
    }
}

/// Discover source files in a directory tree.
///
/// Walks the directory respecting .gitignore patterns and default exclusions.
/// Skips binary files and files exceeding the size threshold.
pub fn discover_files(root: &Path, config: &IndexingConfig) -> Result<DiscoveryResult> {
    discover_files_with_cancellation(root, config, &CancellationToken::new())
}

/// Discover source files while cooperatively observing cancellation.
pub fn discover_files_with_cancellation(
    root: &Path,
    config: &IndexingConfig,
    cancellation: &CancellationToken,
) -> Result<DiscoveryResult> {
    cancellation.check()?;
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve path: {}", root.display()))?;

    let strategy = determine_ignore_strategy(&root, config)?;

    let mut walker = WalkBuilder::new(&root);
    walker
        .hidden(false)
        .git_ignore(strategy.use_gitignore)
        .git_global(strategy.use_gitignore)
        .git_exclude(strategy.use_gitignore)
        .require_git(false)
        .ignore(!config.no_ignore);

    if strategy.use_veraignore {
        walker.add_custom_ignore_filename(".veraignore");
    }

    // Add default directory exclusions and CLI --exclude patterns as overrides.
    let overrides = build_overrides(&root, config)?;
    walker.overrides(overrides);

    let mut files = Vec::new();
    let mut binary_skipped = 0usize;
    let mut large_skipped = 0usize;
    let mut large_skipped_paths = Vec::new();
    let mut error_skipped = 0usize;

    for entry in walker.build() {
        cancellation.check()?;
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                warn!("file discovery error: {err}");
                error_skipped += 1;
                continue;
            }
        };

        // Keep only regular files. `file_type()` reports the entry's own type,
        // so a symlink is skipped rather than indexed through its target: the
        // walker does not follow links, and resolving one here would pull
        // content from outside the repository under an in-repository path.
        let path = entry.path();
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        // Sphinx include fragments are inlined into parent RST documents,
        // so indexing them separately duplicates content by default.
        if !config.no_default_excludes && is_rst_include_fragment(path) {
            debug!("skipping rst include fragment: {}", path.display());
            continue;
        }

        // Get file metadata for size check.
        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(err) => {
                warn!("cannot read metadata for {}: {err}", path.display());
                error_skipped += 1;
                continue;
            }
        };

        let size = metadata.len();

        // Skip large files.
        if size > config.max_file_size_bytes {
            debug!(
                "skipping large file: {} ({} bytes > {} max)",
                path.display(),
                size,
                config.max_file_size_bytes
            );
            let rel = path
                .strip_prefix(&root)
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            large_skipped_paths.push((rel, size));
            large_skipped += 1;
            continue;
        }

        // Skip empty files.
        if size == 0 {
            continue;
        }

        // Skip binary files by extension.
        if is_binary_extension(path) {
            debug!("skipping binary (extension): {}", path.display());
            binary_skipped += 1;
            continue;
        }

        // Skip binary files by content detection (read first 8KB).
        match is_binary_content(path) {
            Ok(true) => {
                debug!("skipping binary (content): {}", path.display());
                binary_skipped += 1;
                continue;
            }
            Ok(false) => {}
            Err(err) => {
                warn!(
                    "cannot read file for binary check: {} — {err}",
                    path.display()
                );
                error_skipped += 1;
                continue;
            }
        }

        // Compute repository-relative path.
        let relative_path = match path.strip_prefix(&root) {
            Ok(rel) => rel.to_string_lossy().to_string(),
            Err(_) => path.to_string_lossy().to_string(),
        };

        files.push(DiscoveredFile {
            absolute_path: path.to_path_buf(),
            relative_path,
            size,
        });
    }

    // Sort for deterministic output.
    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    Ok(DiscoveryResult {
        files,
        binary_skipped,
        large_skipped,
        large_skipped_paths,
        error_skipped,
    })
}

/// Explain why a path would or would not be indexed.
pub fn explain_path(root: &Path, path: &Path, config: &IndexingConfig) -> Result<PathExplanation> {
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve path: {}", root.display()))?;
    let display_input = path.display().to_string();
    let Some((root_relative, relative)) = normalized_relative_path(&root, path) else {
        return Ok(PathExplanation {
            input_path: display_input,
            absolute_path: if path.is_absolute() {
                path.display().to_string()
            } else {
                root.join(path).display().to_string()
            },
            relative_path: None,
            decision: PathDecision::OutsideRoot,
            reason: PathReason::OutsideRoot,
            source: None,
            pattern: None,
            details: Some(format!(
                "path is outside the repository root {}",
                root.display()
            )),
        });
    };
    let candidate = root.join(&relative);
    let absolute_display = candidate.display().to_string();
    let relative_path = relative.to_string_lossy().to_string();

    // `symlink_metadata` refuses to follow only the final component, so an
    // intermediate symlinked directory would still be traversed by every check
    // below. The walker never descends through one, so nothing beneath it is
    // enumerated; reject the whole path before reading metadata, ignore rules
    // or content through the link. Which form of the path the scan walks is
    // platform-dependent, since only POSIX resolves a `..` against the directory
    // it has already reached.
    match scan_ancestors(&root, ancestor_scan_path(root_relative, &relative))? {
        AncestorScan::Symlink(link) => {
            return Ok(path_excluded(
                display_input,
                absolute_display,
                relative_path,
                IgnoreMatchExplanation {
                    reason: PathReason::NonRegularFile,
                    source: Some("discovery".to_string()),
                    pattern: None,
                    details: Some(format!(
                        "the ancestor {} is a symbolic link, and symbolic links are not followed during indexing",
                        link.display()
                    )),
                },
            ));
        }
        AncestorScan::UnresolvableAncestor => {
            return Ok(path_missing(display_input, absolute_display, relative_path));
        }
        AncestorScan::Clear => {}
    }

    // Inspect the link itself, not its target, so this diagnostic reports the
    // same decision the walker makes.
    let metadata = match std::fs::symlink_metadata(&candidate) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(path_missing(display_input, absolute_display, relative_path));
        }
        Err(err) => {
            return Err(err)
                .with_context(|| format!("failed to read metadata for {}", candidate.display()));
        }
    };

    if metadata.is_dir() {
        return Ok(PathExplanation {
            input_path: display_input,
            absolute_path: absolute_display,
            relative_path: Some(relative_path),
            decision: PathDecision::Directory,
            reason: PathReason::Directory,
            source: None,
            pattern: None,
            details: Some("directories are not indexed directly".to_string()),
        });
    }

    if !metadata.is_file() {
        return Ok(path_excluded(
            display_input,
            absolute_display,
            relative_path,
            IgnoreMatchExplanation {
                reason: PathReason::NonRegularFile,
                source: Some("discovery".to_string()),
                pattern: None,
                details: Some(if metadata.is_symlink() {
                    "symbolic links are not followed during indexing".to_string()
                } else {
                    "only regular files are indexed".to_string()
                }),
            },
        ));
    }

    if let Some(explanation) = explain_override_match(&root, &relative, config)? {
        return Ok(path_excluded(
            display_input,
            absolute_display,
            relative_path,
            explanation,
        ));
    }

    let mut whitelist = IgnoreDecision::None;
    if !config.no_ignore {
        let strategy = determine_ignore_strategy(&root, config)?;

        for decision in [
            explain_custom_ignore_match(&root, &relative, strategy.use_veraignore)?,
            explain_named_ignore_match(&root, &relative, ".ignore", PathReason::DotIgnore)?,
            if strategy.use_gitignore {
                explain_named_ignore_match(&root, &relative, ".gitignore", PathReason::Gitignore)?
            } else {
                IgnoreDecision::None
            },
            if strategy.use_gitignore {
                explain_git_exclude_match(&root, &candidate)?
            } else {
                IgnoreDecision::None
            },
            if strategy.use_gitignore {
                explain_git_global_match(&candidate)?
            } else {
                IgnoreDecision::None
            },
        ] {
            match decision {
                IgnoreDecision::Ignore(explanation) => {
                    return Ok(path_excluded(
                        display_input,
                        absolute_display,
                        relative_path,
                        explanation,
                    ));
                }
                IgnoreDecision::Whitelist(explanation) => {
                    whitelist = IgnoreDecision::Whitelist(explanation)
                }
                IgnoreDecision::None => {}
            }
        }
    }

    if !config.no_default_excludes && is_rst_include_fragment(&candidate) {
        return Ok(path_excluded(
            display_input,
            absolute_display,
            relative_path,
            IgnoreMatchExplanation {
                reason: PathReason::RstIncludeFragment,
                source: Some("default excludes".to_string()),
                pattern: Some("*.rst.inc".to_string()),
                details: Some(
                    "reStructuredText include fragments are skipped because their content is indexed through the parent document"
                        .to_string(),
                ),
            },
        ));
    }

    if metadata.len() > config.max_file_size_bytes {
        return Ok(path_excluded(
            display_input,
            absolute_display,
            relative_path,
            IgnoreMatchExplanation {
                reason: PathReason::TooLarge,
                source: Some("indexing config".to_string()),
                pattern: None,
                details: Some(format!(
                    "file size {} exceeds the configured limit {}",
                    metadata.len(),
                    config.max_file_size_bytes
                )),
            },
        ));
    }

    if metadata.len() == 0 {
        return Ok(path_excluded(
            display_input,
            absolute_display,
            relative_path,
            IgnoreMatchExplanation {
                reason: PathReason::EmptyFile,
                source: Some("discovery".to_string()),
                pattern: None,
                details: Some("empty files are skipped during indexing".to_string()),
            },
        ));
    }

    if is_binary_extension(&candidate) {
        return Ok(path_excluded(
            display_input,
            absolute_display,
            relative_path,
            IgnoreMatchExplanation {
                reason: PathReason::BinaryExtension,
                source: Some("binary extension list".to_string()),
                pattern: candidate
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| format!("*.{ext}")),
                details: Some(
                    "known binary file extensions are skipped before content inspection"
                        .to_string(),
                ),
            },
        ));
    }

    if is_binary_content(&candidate)? {
        return Ok(path_excluded(
            display_input,
            absolute_display,
            relative_path,
            IgnoreMatchExplanation {
                reason: PathReason::BinaryContent,
                source: Some("binary content heuristic".to_string()),
                pattern: None,
                details: Some(
                    "the first 8KB contains a null byte, so the file is treated as binary"
                        .to_string(),
                ),
            },
        ));
    }

    let (source, pattern, details) = match whitelist {
        IgnoreDecision::Whitelist(explanation) => {
            (explanation.source, explanation.pattern, explanation.details)
        }
        IgnoreDecision::Ignore(_) | IgnoreDecision::None => (None, None, None),
    };

    Ok(PathExplanation {
        input_path: display_input,
        absolute_path: absolute_display,
        relative_path: Some(relative_path),
        decision: PathDecision::Indexed,
        reason: PathReason::Indexed,
        source,
        pattern,
        details: details.or_else(|| Some("path would be indexed".to_string())),
    })
}

/// What a walk of `relative`'s ancestors found, ignoring the final component.
enum AncestorScan {
    /// Every ancestor exists and none of them is a symbolic link.
    Clear,
    /// The first ancestor that is a symbolic link, repository-relative.
    Symlink(PathBuf),
    /// An ancestor is absent or is not a directory, so the path cannot resolve
    /// at all.
    UnresolvableAncestor,
}

/// Which form of the path the ancestor scan should walk: the one whose
/// ancestors the operating system would actually visit.
///
/// POSIX resolves a path left to right and applies each `..` to the directory it
/// has already reached, so the ancestors are the ones written down:
/// `link/../file` traverses `link`, and `regular/../file` stops at `regular`
/// with `ENOTDIR`. Win32 canonicalizes `..` out of an ordinary path before the
/// file system sees it, so `link\..\file` opens `file` and `regular\..\file`
/// succeeds; scanning the written form there would report on ancestors Windows
/// never visits.
///
/// Only the placement of `..` differs, which is why this selects an input rather
/// than gating the scan. The scan's other two outcomes hold on both platforms:
/// Windows will not open `regular\nested\file` either, and the walker declines a
/// reparse point exactly as it declines a symbolic link. The Windows branch is
/// reasoned from Win32's documented path canonicalization and is not exercised
/// by the tests below, which run only on Unix.
fn ancestor_scan_path<'a>(as_written: &'a Path, collapsed: &'a Path) -> &'a Path {
    if cfg!(unix) { as_written } else { collapsed }
}

/// Walk the ancestors of `relative` under `root`, stopping at the first that is
/// a symbolic link or that the operating system could not resolve through.
///
/// `relative` comes from `ancestor_scan_path`, which decides whether this
/// platform has already collapsed its `..` segments. The walk keeps its own
/// lexical prefix so that a `..` still present counts as one more component
/// following whatever precedes it.
///
/// An unresolvable ancestor is reported separately rather than skipped, because
/// the kernel cannot resolve through it either: `absent/../link/file` is missing,
/// not a link, and normalizing `absent/..` away would otherwise leave the caller
/// inspecting a path the operating system would never reach. An ancestor that
/// exists but is not a directory is the same outcome by a different errno, since
/// `regular/../file` stops at `regular` with `ENOTDIR`.
fn scan_ancestors(root: &Path, relative: &Path) -> Result<AncestorScan> {
    let mut prefix = PathBuf::new();
    let mut components = relative.components().peekable();
    while let Some(component) = components.next() {
        match component {
            Component::CurDir => continue,
            Component::ParentDir => {
                if !prefix.pop() {
                    return Ok(AncestorScan::Clear);
                }
                continue;
            }
            Component::Normal(part) => prefix.push(part),
            Component::RootDir | Component::Prefix(_) => return Ok(AncestorScan::Clear),
        }
        if components.peek().is_none() {
            break;
        }
        let ancestor = root.join(&prefix);
        match std::fs::symlink_metadata(&ancestor) {
            Ok(metadata) if metadata.is_symlink() => return Ok(AncestorScan::Symlink(prefix)),
            Ok(metadata) if !metadata.is_dir() => {
                return Ok(AncestorScan::UnresolvableAncestor);
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(AncestorScan::UnresolvableAncestor);
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("failed to read metadata for {}", ancestor.display())
                });
            }
        }
    }
    Ok(AncestorScan::Clear)
}

/// Resolve `path` against `root`, returning both the root-relative path as it
/// was written and its lexically normalized form.
///
/// Callers that inspect the filesystem need the former, because normalization
/// discards the `..` segments the kernel would resolve through; callers that
/// match ignore rules or report the path need the latter.
fn normalized_relative_path<'a>(root: &Path, path: &'a Path) -> Option<(&'a Path, PathBuf)> {
    let relative = if path.is_absolute() {
        path.strip_prefix(root).ok()?
    } else {
        path
    };
    Some((relative, normalize_relative_path(relative)?))
}

fn normalize_relative_path(path: &Path) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    return None;
                }
            }
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(normalized)
}

fn determine_ignore_strategy(root: &Path, config: &IndexingConfig) -> Result<IgnoreStrategy> {
    let veraignore_path = root.join(".veraignore");
    if config.no_ignore {
        return Ok(IgnoreStrategy {
            use_gitignore: false,
            use_veraignore: false,
        });
    }
    if veraignore_path.exists() {
        let content = std::fs::read_to_string(&veraignore_path)
            .with_context(|| format!("failed to read {}", veraignore_path.display()))?;
        let has_include = content.lines().any(|l| l.trim() == "#include .gitignore");
        return Ok(IgnoreStrategy {
            use_gitignore: has_include,
            use_veraignore: true,
        });
    }
    Ok(IgnoreStrategy {
        use_gitignore: true,
        use_veraignore: false,
    })
}

fn build_overrides(root: &Path, config: &IndexingConfig) -> Result<Override> {
    let mut overrides = OverrideBuilder::new(root);
    if !config.no_default_excludes {
        for pattern in &config.default_excludes {
            overrides
                .add(&format!("!{pattern}/"))
                .with_context(|| format!("invalid exclusion pattern: {pattern}"))?;
        }
        overrides.add("!.veraignore")?;
    }
    for pattern in &config.extra_excludes {
        overrides
            .add(&format!("!{pattern}"))
            .with_context(|| format!("invalid --exclude pattern: {pattern}"))?;
    }
    overrides
        .build()
        .context("failed to build override patterns")
}

fn simple_explanation(
    reason: PathReason,
    source: &str,
    pattern: &str,
    details: &str,
) -> IgnoreMatchExplanation {
    IgnoreMatchExplanation {
        reason,
        source: Some(source.to_string()),
        pattern: Some(pattern.to_string()),
        details: Some(details.to_string()),
    }
}

fn explain_override_match(
    root: &Path,
    relative: &Path,
    config: &IndexingConfig,
) -> Result<Option<IgnoreMatchExplanation>> {
    if !config.no_default_excludes {
        for pattern in &config.default_excludes {
            let mut builder = OverrideBuilder::new(root);
            builder
                .add(&format!("!{pattern}/"))
                .with_context(|| format!("invalid exclusion pattern: {pattern}"))?;
            let matcher = builder
                .build()
                .context("failed to build override matcher")?;
            if override_matches_path_or_any_parents(&matcher, relative) {
                return Ok(Some(simple_explanation(
                    PathReason::DefaultExclude,
                    "default excludes",
                    pattern,
                    "matched Vera's built-in indexing exclusions",
                )));
            }
        }

        let mut builder = OverrideBuilder::new(root);
        builder.add("!.veraignore")?;
        let matcher = builder
            .build()
            .context("failed to build .veraignore override")?;
        if override_matches_path_or_any_parents(&matcher, relative) {
            return Ok(Some(simple_explanation(
                PathReason::DefaultExclude,
                "default excludes",
                ".veraignore",
                "the .veraignore file itself is never indexed",
            )));
        }
    }

    for pattern in &config.extra_excludes {
        let mut builder = OverrideBuilder::new(root);
        builder
            .add(&format!("!{pattern}"))
            .with_context(|| format!("invalid --exclude pattern: {pattern}"))?;
        let matcher = builder
            .build()
            .context("failed to build CLI override matcher")?;
        if override_matches_path_or_any_parents(&matcher, relative) {
            return Ok(Some(simple_explanation(
                PathReason::CliExclude,
                "--exclude",
                pattern,
                "matched a CLI exclusion pattern",
            )));
        }
    }

    Ok(None)
}

fn override_matches_path_or_any_parents(matcher: &Override, relative: &Path) -> bool {
    if !matcher.matched(relative, false).is_none() {
        return true;
    }

    let mut current = relative.parent();
    while let Some(parent) = current {
        if !matcher.matched(parent, true).is_none() {
            return true;
        }
        current = parent.parent();
    }
    false
}

fn explain_custom_ignore_match(
    root: &Path,
    relative: &Path,
    use_veraignore: bool,
) -> Result<IgnoreDecision> {
    if !use_veraignore {
        return Ok(IgnoreDecision::None);
    }
    explain_named_ignore_match(root, relative, ".veraignore", PathReason::Veraignore)
}

fn explain_named_ignore_match(
    root: &Path,
    relative: &Path,
    filename: &str,
    reason: PathReason,
) -> Result<IgnoreDecision> {
    for dir in path_ancestors_from_child(relative) {
        let ignore_file = root.join(&dir).join(filename);
        if !ignore_file.exists() {
            continue;
        }
        let (matcher, err) = Gitignore::new(&ignore_file);
        if let Some(err) = err {
            warn!(file = %ignore_file.display(), error = %err, "ignore file parse warning");
        }
        let candidate = relative.strip_prefix(&dir).unwrap_or(relative);
        let matched = gitignore_matches_path_or_any_parents(&matcher, candidate, false);
        match matched {
            Match::Ignore(glob) => {
                return Ok(IgnoreDecision::Ignore(glob_explanation(reason, glob)));
            }
            Match::Whitelist(glob) => {
                return Ok(IgnoreDecision::Whitelist(glob_explanation(reason, glob)));
            }
            Match::None => {}
        }
    }
    Ok(IgnoreDecision::None)
}

fn explain_git_exclude_match(root: &Path, absolute: &Path) -> Result<IgnoreDecision> {
    let exclude = root.join(".git").join("info").join("exclude");
    if !exclude.exists() {
        return Ok(IgnoreDecision::None);
    }
    let mut builder = GitignoreBuilder::new(root);
    if let Some(err) = builder.add(&exclude) {
        warn!(file = %exclude.display(), error = %err, "git exclude parse warning");
    }
    let matcher = builder
        .build()
        .context("failed to build git exclude matcher")?;
    let candidate = absolute.strip_prefix(root).unwrap_or(absolute);
    Ok(
        match gitignore_matches_path_or_any_parents(&matcher, candidate, false) {
            Match::Ignore(glob) => {
                IgnoreDecision::Ignore(glob_explanation(PathReason::GitExclude, glob))
            }
            Match::Whitelist(glob) => {
                IgnoreDecision::Whitelist(glob_explanation(PathReason::GitExclude, glob))
            }
            Match::None => IgnoreDecision::None,
        },
    )
}

fn explain_git_global_match(absolute: &Path) -> Result<IgnoreDecision> {
    let (matcher, err) = Gitignore::global();
    if let Some(err) = err {
        warn!(error = %err, "global gitignore parse warning");
    }
    Ok(
        match gitignore_matches_path_or_any_parents(&matcher, absolute, false) {
            Match::Ignore(glob) => {
                IgnoreDecision::Ignore(glob_explanation(PathReason::GitGlobal, glob))
            }
            Match::Whitelist(glob) => {
                IgnoreDecision::Whitelist(glob_explanation(PathReason::GitGlobal, glob))
            }
            Match::None => IgnoreDecision::None,
        },
    )
}

fn glob_explanation(reason: PathReason, glob: &GitignoreGlob) -> IgnoreMatchExplanation {
    IgnoreMatchExplanation {
        reason,
        source: glob.from().map(|path| path.display().to_string()),
        pattern: Some(glob.original().to_string()),
        details: Some(if glob.is_whitelist() {
            "matched a whitelist pattern".to_string()
        } else {
            "matched an ignore pattern".to_string()
        }),
    }
}

fn path_excluded(
    input_path: String,
    absolute_path: String,
    relative_path: String,
    explanation: IgnoreMatchExplanation,
) -> PathExplanation {
    PathExplanation {
        input_path,
        absolute_path,
        relative_path: Some(relative_path),
        decision: PathDecision::Excluded,
        reason: explanation.reason,
        source: explanation.source,
        pattern: explanation.pattern,
        details: explanation.details,
    }
}

fn path_missing(
    input_path: String,
    absolute_path: String,
    relative_path: String,
) -> PathExplanation {
    PathExplanation {
        input_path,
        absolute_path,
        relative_path: Some(relative_path),
        decision: PathDecision::Missing,
        reason: PathReason::Missing,
        source: None,
        pattern: None,
        details: Some("path does not exist".to_string()),
    }
}

fn path_ancestors_from_child(path: &Path) -> Vec<PathBuf> {
    let mut ancestors = Vec::new();
    let mut current = path.parent();
    while let Some(parent) = current {
        ancestors.push(parent.to_path_buf());
        current = parent.parent();
    }
    // The loop already produces an empty PathBuf for the root when
    // `parent()` returns "" on a relative path. Only push if missing.
    if ancestors
        .last()
        .is_none_or(|last| !last.as_os_str().is_empty())
    {
        ancestors.push(PathBuf::new());
    }
    ancestors
}

fn gitignore_matches_path_or_any_parents<'a>(
    matcher: &'a Gitignore,
    path: &Path,
    is_dir: bool,
) -> Match<&'a GitignoreGlob> {
    match matcher.matched(path, is_dir) {
        Match::None => {}
        decision => return decision,
    }

    let mut current = path.parent();
    while let Some(parent) = current {
        match matcher.matched(parent, true) {
            Match::None => current = parent.parent(),
            decision => return decision,
        }
    }

    Match::None
}

/// Check if a file has a known binary extension.
fn is_binary_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            let lower = ext.to_lowercase();
            BINARY_EXTENSIONS.contains(&lower.as_str())
        })
}

fn is_rst_include_fragment(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".rst.inc"))
}

/// Check if a file contains binary content by reading the first 8KB.
///
/// Uses the null-byte heuristic: if any null bytes are found in the
/// first 8KB, the file is considered binary.
fn is_binary_content(path: &Path) -> Result<bool> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("cannot open for binary check: {}", path.display()))?;

    let mut buf = [0u8; 8192];
    let n = file.read(&mut buf)?;

    // Null byte detection: any \0 in the sample means binary.
    Ok(buf[..n].contains(&0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn default_config() -> IndexingConfig {
        IndexingConfig::default()
    }

    /// The matcher exists so the MCP watcher classifies a path the same way
    /// `discover_files` does. The failing direction is a *file* whose name
    /// equals an excluded *directory*: `build`, `target`, `dist` and friends are
    /// ordinary extensionless script names, and discovery indexes them because
    /// its patterns are directory-only.
    #[test]
    fn matcher_agrees_with_discovery_on_a_file_named_like_an_excluded_dir() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("build"), "#!/bin/sh\ncargo build\n").unwrap();
        fs::create_dir_all(dir.path().join("target/debug")).unwrap();
        fs::write(dir.path().join("target/debug/app.o"), "object").unwrap();

        let config = default_config();
        let discovered: Vec<String> = discover_files(dir.path(), &config)
            .unwrap()
            .files
            .into_iter()
            .map(|f| f.relative_path)
            .collect();
        // Presence first: the assertion below is meaningless if discovery
        // skipped the file for some unrelated reason.
        assert!(
            discovered.contains(&"build".to_string()),
            "discovery indexes an extensionless file named `build`: {discovered:?}"
        );
        assert!(
            !discovered.contains(&"target/debug/app.o".to_string()),
            "discovery skips the target/ directory: {discovered:?}"
        );

        let root = dir.path().canonicalize().unwrap();
        let matcher = ExclusionMatcher::new(&root, &config).unwrap();
        assert!(
            !matcher.is_excluded(&root.join("build")),
            "a regular file named `build` is indexed, so it must not be classified as excluded"
        );
        assert!(
            matcher.is_excluded(&root.join("target/debug/app.o")),
            "a file under target/ is not indexed, so it must be classified as excluded"
        );
        assert!(
            matcher.is_excluded(&root.join("target")),
            "the target directory itself must be classified as excluded"
        );
    }

    #[test]
    fn matcher_excludes_index_and_vendor_dirs_but_not_sources() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let matcher = ExclusionMatcher::new(&root, &default_config()).unwrap();

        for excluded in [".vera/metadata.db", ".git/HEAD", "node_modules/x/i.js"] {
            assert!(
                matcher.is_excluded(&root.join(excluded)),
                "{excluded} must be excluded"
            );
        }
        assert!(!matcher.is_excluded(&root.join("src/main.rs")));
        assert!(
            !matcher.is_excluded(Path::new("/somewhere/else/main.rs")),
            "a path outside the root cannot be classified, so it is not excluded"
        );
    }

    #[test]
    fn matcher_applies_extra_excludes() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mut config = default_config();
        config.extra_excludes = vec!["vendor/**".to_string()];
        let matcher = ExclusionMatcher::new(&root, &config).unwrap();

        assert!(matcher.is_excluded(&root.join("vendor/lib/x.rs")));
        assert!(!matcher.is_excluded(&root.join("src/x.rs")));
    }

    #[test]
    fn read_source_lossy_decodes_invalid_utf8() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("notes.txt");
        // Latin-1 "café" with an invalid UTF-8 byte sequence.
        fs::write(&path, b"caf\xe9\nplain text\n").unwrap();

        let content = read_source_lossy(&path).unwrap();
        assert!(content.contains("caf\u{FFFD}"));
        assert!(content.contains("plain text"));

        let good = dir.path().join("ok.rs");
        fs::write(&good, "fn main() {}\n").unwrap();
        assert_eq!(read_source_lossy(&good).unwrap(), "fn main() {}\n");
    }

    #[test]
    fn discovers_source_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("lib.py"), "def hello(): pass").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 2);

        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"lib.py"));
        assert!(names.contains(&"main.rs"));
    }

    #[test]
    fn respects_gitignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive data").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&"secret.txt"),
            "gitignored file should be excluded"
        );
    }

    #[test]
    fn default_exclusions_active() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

        // Create excluded directories with files.
        let nm = dir.path().join("node_modules");
        fs::create_dir_all(&nm).unwrap();
        fs::write(nm.join("dep.js"), "module.exports = {}").unwrap();

        let git = dir.path().join(".git");
        fs::create_dir_all(&git).unwrap();
        fs::write(git.join("config"), "[core]").unwrap();

        let target = dir.path().join("target");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("out.rs"), "compiled").unwrap();

        let pycache = dir.path().join("__pycache__");
        fs::create_dir_all(&pycache).unwrap();
        fs::write(pycache.join("mod.pyc"), "bytecode").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();

        assert!(names.contains(&"main.rs"), "main.rs should be included");
        assert!(
            !names.iter().any(|n| n.starts_with("node_modules")),
            "node_modules should be excluded"
        );
        assert!(
            !names.iter().any(|n| n.starts_with(".git")),
            ".git should be excluded"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("target")),
            "target should be excluded"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("__pycache__")),
            "__pycache__ should be excluded"
        );
    }

    #[test]
    fn skips_binary_by_extension() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("image.png"), "not really png").unwrap();
        fs::write(dir.path().join("program.exe"), "not really exe").unwrap();
        fs::write(dir.path().join("archive.zip"), "not really zip").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "main.rs");
        assert_eq!(result.binary_skipped, 3);
    }

    #[test]
    fn skips_binary_by_content() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        // Write a file with null bytes (binary content).
        let binary_content = b"some text\x00more text";
        fs::write(dir.path().join("data.txt"), binary_content).unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "main.rs");
        assert_eq!(result.binary_skipped, 1);
    }

    #[test]
    fn skips_large_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("small.rs"), "fn main() {}").unwrap();

        // Create a file larger than the default threshold (1MB).
        let large_content = "x".repeat(1_100_000);
        fs::write(dir.path().join("huge.rs"), large_content).unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "small.rs");
        assert_eq!(result.large_skipped, 1);
    }

    #[test]
    fn configurable_size_threshold() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("small.rs"), "fn a() {}").unwrap();
        fs::write(dir.path().join("medium.rs"), "x".repeat(500)).unwrap();

        let config = IndexingConfig {
            max_file_size_bytes: 100,
            ..Default::default()
        };

        let result = discover_files(dir.path(), &config).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "small.rs");
        assert_eq!(result.large_skipped, 1);
    }

    #[test]
    fn skips_empty_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("empty.rs"), "").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].relative_path, "main.rs");
    }

    #[test]
    fn discovers_nested_files() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("src").join("utils");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("helper.rs"), "fn help() {}").unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        assert_eq!(result.files.len(), 2);

        let paths: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(paths.contains(&"main.rs"));
        assert!(paths.contains(&"src/utils/helper.rs"));
    }

    #[test]
    fn binary_extension_detection() {
        assert!(is_binary_extension(Path::new("file.png")));
        assert!(is_binary_extension(Path::new("file.PNG")));
        assert!(is_binary_extension(Path::new("file.exe")));
        assert!(is_binary_extension(Path::new("file.zip")));
        assert!(is_binary_extension(Path::new("file.wasm")));
        assert!(!is_binary_extension(Path::new("file.rs")));
        assert!(!is_binary_extension(Path::new("file.py")));
        assert!(!is_binary_extension(Path::new("file.ts")));
        assert!(!is_binary_extension(Path::new("no_extension")));
    }

    #[test]
    fn result_is_deterministically_sorted() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("z.rs"), "fn z() {}").unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        fs::write(dir.path().join("m.rs"), "fn m() {}").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let paths: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert_eq!(paths, vec!["a.rs", "m.rs", "z.rs"]);
    }

    #[test]
    fn venv_and_dist_excluded() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("app.py"), "def main(): pass").unwrap();

        let venv = dir.path().join(".venv");
        fs::create_dir_all(&venv).unwrap();
        fs::write(venv.join("activate"), "#!/bin/bash").unwrap();

        let dist = dir.path().join("dist");
        fs::create_dir_all(&dist).unwrap();
        fs::write(dist.join("bundle.js"), "var x=1").unwrap();

        let build = dir.path().join("build");
        fs::create_dir_all(&build).unwrap();
        fs::write(build.join("output.js"), "compiled").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let paths: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();

        assert!(paths.contains(&"app.py"));
        assert!(
            !paths.iter().any(|p| p.starts_with(".venv")),
            ".venv should be excluded"
        );
        assert!(
            !paths.iter().any(|p| p.starts_with("dist")),
            "dist should be excluded"
        );
        assert!(
            !paths.iter().any(|p| p.starts_with("build")),
            "build should be excluded"
        );
    }

    #[test]
    fn rst_include_fragments_excluded_by_default() {
        let dir = TempDir::new().unwrap();
        let docs = dir.path().join("docs");
        let includes = docs.join("includes");
        fs::create_dir_all(&includes).unwrap();

        fs::write(
            docs.join("index.rst"),
            ".. include:: includes/common.rst.inc\n",
        )
        .unwrap();
        fs::write(includes.join("common.rst.inc"), "Shared include fragment\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();

        assert!(names.contains(&"docs/index.rst"));
        assert!(
            !names.contains(&"docs/includes/common.rst.inc"),
            "rst include fragment should be excluded by default"
        );
    }

    #[test]
    fn rst_include_fragments_included_when_default_excludes_disabled() {
        let dir = TempDir::new().unwrap();
        let docs = dir.path().join("docs");
        let includes = docs.join("includes");
        fs::create_dir_all(&includes).unwrap();

        fs::write(
            docs.join("index.rst"),
            ".. include:: includes/common.rst.inc\n",
        )
        .unwrap();
        fs::write(includes.join("common.rst.inc"), "Shared include fragment\n").unwrap();

        let mut config = default_config();
        config.no_default_excludes = true;

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();

        assert!(names.contains(&"docs/index.rst"));
        assert!(
            names.contains(&"docs/includes/common.rst.inc"),
            "rst include fragment should be included with no_default_excludes"
        );
    }

    #[test]
    fn gitignore_patterns_with_wildcards() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("test.log"), "log data").unwrap();
        fs::write(dir.path().join("app.log"), "more logs").unwrap();
        fs::write(dir.path().join(".gitignore"), "*.log\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(!names.contains(&"test.log"));
        assert!(!names.contains(&"app.log"));
    }

    #[test]
    fn handles_invalid_path() {
        let result = discover_files(Path::new("/nonexistent/path"), &default_config());
        assert!(result.is_err());
    }

    #[test]
    fn veraignore_overrides_gitignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join("notes.md"), "local notes").unwrap();
        // .gitignore excludes secret.txt and notes.md
        fs::write(dir.path().join(".gitignore"), "secret.txt\nnotes.md\n").unwrap();
        // .veraignore only excludes secret.txt (notes.md should now be indexed)
        fs::write(dir.path().join(".veraignore"), "secret.txt\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            names.contains(&"notes.md"),
            "notes.md should be indexed when .veraignore overrides .gitignore"
        );
        assert!(
            !names.contains(&"secret.txt"),
            "secret.txt should still be excluded by .veraignore"
        );
    }

    #[test]
    fn veraignore_include_gitignore_directive() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join("draft.md"), "draft notes").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();
        // .veraignore includes gitignore rules AND adds draft.md exclusion
        fs::write(
            dir.path().join(".veraignore"),
            "#include .gitignore\ndraft.md\n",
        )
        .unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&"secret.txt"),
            "gitignore rules should apply via #include"
        );
        assert!(
            !names.contains(&"draft.md"),
            ".veraignore extra pattern should apply"
        );
    }

    #[test]
    fn no_veraignore_uses_gitignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&"secret.txt"),
            "gitignore should apply when no .veraignore"
        );
    }

    #[test]
    fn extra_excludes_from_cli() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        let vendor = dir.path().join("vendor");
        fs::create_dir_all(&vendor).unwrap();
        fs::write(vendor.join("dep.rs"), "fn dep() {}").unwrap();

        let mut config = default_config();
        config.extra_excludes = vec!["vendor/**".to_string()];

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.iter().any(|n| n.starts_with("vendor")),
            "--exclude should exclude vendor"
        );
    }

    #[test]
    fn no_ignore_disables_gitignore_and_veraignore() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();

        let mut config = default_config();
        config.no_ignore = true;

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            names.contains(&"secret.txt"),
            "no_ignore should disable gitignore"
        );
    }

    #[test]
    fn no_default_excludes_disables_smart_defaults() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        let nm = dir.path().join("node_modules");
        fs::create_dir_all(&nm).unwrap();
        fs::write(nm.join("dep.js"), "module.exports = {}").unwrap();

        let mut config = default_config();
        config.no_default_excludes = true;

        let result = discover_files(dir.path(), &config).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            names.iter().any(|n| n.starts_with("node_modules")),
            "no_default_excludes should allow node_modules"
        );
    }

    #[test]
    fn veraignore_itself_excluded_from_indexing() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join(".veraignore"), "*.log\n").unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert!(names.contains(&"main.rs"));
        assert!(
            !names.contains(&".veraignore"),
            ".veraignore should not be indexed"
        );
    }

    #[test]
    fn explain_path_reports_gitignore_match() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive data").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();

        let explanation =
            explain_path(dir.path(), Path::new("secret.txt"), &default_config()).unwrap();
        assert_eq!(explanation.decision, PathDecision::Excluded);
        assert_eq!(explanation.reason, PathReason::Gitignore);
        assert_eq!(explanation.pattern.as_deref(), Some("secret.txt"));
    }

    #[test]
    fn explain_path_reports_default_exclude() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("dist")).unwrap();
        fs::write(dir.path().join("dist").join("bundle.js"), "var x = 1;").unwrap();

        let explanation =
            explain_path(dir.path(), Path::new("dist/bundle.js"), &default_config()).unwrap();
        assert_eq!(explanation.decision, PathDecision::Excluded);
        assert_eq!(explanation.reason, PathReason::DefaultExclude);
        assert_eq!(explanation.pattern.as_deref(), Some("dist"));
    }

    #[test]
    fn explain_path_reports_veraignore_override() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("secret.txt"), "sensitive data").unwrap();
        fs::write(dir.path().join("notes.md"), "draft").unwrap();
        fs::write(dir.path().join(".gitignore"), "secret.txt\n").unwrap();
        fs::write(dir.path().join(".veraignore"), "notes.md\n").unwrap();

        let secret = explain_path(dir.path(), Path::new("secret.txt"), &default_config()).unwrap();
        assert_eq!(secret.decision, PathDecision::Indexed);

        let notes = explain_path(dir.path(), Path::new("notes.md"), &default_config()).unwrap();
        assert_eq!(notes.decision, PathDecision::Excluded);
        assert_eq!(notes.reason, PathReason::Veraignore);
        assert_eq!(notes.pattern.as_deref(), Some("notes.md"));
    }

    #[test]
    fn explain_path_reports_whitelist_match() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join(".gitignore"), "*.rs\n!main.rs\n").unwrap();

        let explanation =
            explain_path(dir.path(), Path::new("main.rs"), &default_config()).unwrap();
        assert_eq!(explanation.decision, PathDecision::Indexed);
        assert_eq!(explanation.pattern.as_deref(), Some("!main.rs"));
        assert!(
            explanation
                .source
                .as_deref()
                .is_some_and(|source| source.ends_with(".gitignore"))
        );
    }

    #[test]
    fn explain_path_rejects_paths_that_escape_root() {
        let dir = TempDir::new().unwrap();
        let outside = dir.path().parent().unwrap().join("vera-outside-root.rs");
        fs::write(&outside, "fn outside() {}\n").unwrap();

        let explanation = explain_path(
            dir.path(),
            Path::new("../vera-outside-root.rs"),
            &default_config(),
        )
        .unwrap();
        assert_eq!(explanation.decision, PathDecision::OutsideRoot);
        assert_eq!(explanation.reason, PathReason::OutsideRoot);

        fs::remove_file(outside).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_file_outside_root_is_not_discovered() {
        let outside = TempDir::new().unwrap();
        let canary = outside.path().join("canary.rs");
        fs::write(&canary, "fn canary_outside_root() {}\n").unwrap();

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        std::os::unix::fs::symlink(&canary, dir.path().join("linked.rs")).unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["main.rs"],
            "a symlink to a file outside the root must not be discovered"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_file_inside_root_is_not_indexed_twice() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("main.rs");
        fs::write(&real, "fn main() {}").unwrap();
        std::os::unix::fs::symlink(&real, dir.path().join("alias.rs")).unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let names: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["main.rs"],
            "a symlink to an in-repository file must not be indexed a second time"
        );
    }

    #[cfg(unix)]
    #[test]
    fn explain_path_rejects_symlinks_to_regular_files() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("main.rs");
        fs::write(&real, "fn main() {}").unwrap();
        std::os::unix::fs::symlink(&real, dir.path().join("alias.rs")).unwrap();

        let explanation =
            explain_path(dir.path(), Path::new("alias.rs"), &default_config()).unwrap();
        assert_eq!(explanation.decision, PathDecision::Excluded);
        assert_eq!(explanation.reason, PathReason::NonRegularFile);
    }

    #[cfg(unix)]
    #[test]
    fn explain_path_rejects_files_below_a_symlinked_directory() {
        let outside = TempDir::new().unwrap();
        let outside_dir = outside.path().join("secrets");
        fs::create_dir_all(outside_dir.join("nested")).unwrap();
        fs::write(
            outside_dir.join("nested/secret.rs"),
            "fn outside_secret() {}\n",
        )
        .unwrap();

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        std::os::unix::fs::symlink(&outside_dir, dir.path().join("linkdir")).unwrap();

        let result = discover_files(dir.path(), &default_config()).unwrap();
        let discovered: Vec<&str> = result
            .files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        assert_eq!(discovered, vec!["main.rs"]);

        // The symlink is an intermediate component, not the final one, so
        // `symlink_metadata` on the candidate would follow it.
        let explanation = explain_path(
            dir.path(),
            Path::new("linkdir/nested/secret.rs"),
            &default_config(),
        )
        .unwrap();
        assert_eq!(
            explanation.decision,
            PathDecision::Excluded,
            "explain_path must agree with discovery, which never enumerated this path"
        );
        assert_eq!(explanation.reason, PathReason::NonRegularFile);
    }

    #[cfg(unix)]
    #[test]
    fn explain_path_rejects_a_symlinked_ancestor_erased_by_a_parent_segment() {
        let outside = TempDir::new().unwrap();
        fs::create_dir(outside.path().join("vault")).unwrap();
        fs::write(
            outside.path().join("secret.rs"),
            "fn outside_the_repository() {}\n",
        )
        .unwrap();

        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("secret.rs"),
            "fn inside_the_repository() {}\n",
        )
        .unwrap();
        std::os::unix::fs::symlink(outside.path().join("vault"), dir.path().join("link")).unwrap();

        // The kernel resolves left to right, so `link/../secret.rs` traverses
        // `link` and names `<outside>/secret.rs`. Collapsing `..` lexically
        // first rewrites it to the unrelated in-repository `secret.rs` and
        // leaves no ancestor for the symlink check to see.
        let explanation = explain_path(
            dir.path(),
            Path::new("link/../secret.rs"),
            &default_config(),
        )
        .unwrap();
        assert_eq!(
            explanation.decision,
            PathDecision::Excluded,
            "a parent segment must not erase the symlinked ancestor it traverses"
        );
        assert_eq!(explanation.reason, PathReason::NonRegularFile);
    }

    #[cfg(unix)]
    #[test]
    fn explain_path_reports_a_missing_ancestor_as_missing() {
        let outside = TempDir::new().unwrap();
        let secrets = outside.path().join("secrets");
        fs::create_dir_all(secrets.join("nested")).unwrap();
        fs::write(secrets.join("nested/secret.rs"), "fn outside_secret() {}\n").unwrap();

        let dir = TempDir::new().unwrap();
        std::os::unix::fs::symlink(&secrets, dir.path().join("link")).unwrap();

        // The kernel cannot resolve through `absent`, so the path is missing.
        // Collapsing `absent/..` away instead would leave the checks below
        // inspecting a candidate that reaches through `link`.
        let explanation = explain_path(
            dir.path(),
            Path::new("absent/../link/nested/secret.rs"),
            &default_config(),
        )
        .unwrap();
        assert_eq!(explanation.decision, PathDecision::Missing);
        assert_eq!(explanation.reason, PathReason::Missing);
    }

    // Windows collapses `..` lexically before the filesystem sees the path, so
    // the `ENOTDIR` this pins is POSIX resolution order, which is what
    // `ancestor_scan_path` selects for.
    #[cfg(unix)]
    #[test]
    fn explain_path_reports_a_non_directory_ancestor_as_missing() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("regular"), "not a directory\n").unwrap();
        fs::write(dir.path().join("target.rs"), "fn target() {}\n").unwrap();

        // The kernel stops at `regular` with `ENOTDIR` and never reaches
        // `target.rs`, so the path as written cannot resolve. Collapsing
        // `regular/..` away instead would report on the unrelated `target.rs`.
        assert_eq!(
            std::fs::symlink_metadata(dir.path().join("regular/../target.rs"))
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::NotADirectory
        );

        let explanation = explain_path(
            dir.path(),
            Path::new("regular/../target.rs"),
            &default_config(),
        )
        .unwrap();
        assert_eq!(explanation.decision, PathDecision::Missing);
        assert_eq!(explanation.reason, PathReason::Missing);
    }

    #[test]
    fn explain_path_reports_a_path_that_does_not_exist_as_missing() {
        let dir = TempDir::new().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();

        let explanation =
            explain_path(dir.path(), Path::new("src/absent.rs"), &default_config()).unwrap();
        assert_eq!(explanation.decision, PathDecision::Missing);
        assert_eq!(explanation.reason, PathReason::Missing);
    }

    #[cfg(unix)]
    #[test]
    fn explain_path_rejects_non_regular_files() {
        let dir = TempDir::new().unwrap();
        let device = dir.path().join("null-device");
        std::os::unix::fs::symlink("/dev/null", &device).unwrap();

        let explanation =
            explain_path(dir.path(), Path::new("null-device"), &default_config()).unwrap();
        assert_eq!(explanation.decision, PathDecision::Excluded);
        assert_eq!(explanation.reason, PathReason::NonRegularFile);
    }
}
