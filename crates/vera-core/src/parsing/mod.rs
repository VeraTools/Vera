//! Source code parsing using tree-sitter.
//!
//! This module is responsible for:
//! - Loading tree-sitter grammars for supported languages
//! - Parsing source files into ASTs
//! - Extracting symbol-level chunks (functions, classes, structs, etc.)
//! - Tier 0 fallback chunking for unsupported languages
//!
//! # Architecture
//!
//! - [`languages`] — Grammar loading and language detection
//! - [`extractor`] — AST node extraction rules per language
//! - [`chunker`] — Symbol-to-chunk conversion and large symbol splitting

pub mod chunker;
pub mod extractor;
pub mod languages;
pub mod references;
pub mod signatures;
pub mod sphinx;
pub mod type_relations;

use anyhow::{Context, Result};
use tree_sitter::Parser;

use crate::config::IndexingConfig;
use crate::types::{Chunk, Language};

/// Parsing diagnostics captured alongside chunk extraction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParseDiagnostics {
    /// Whether the tree-sitter parse tree contained error nodes.
    pub tree_has_error: bool,
    /// Whether Vera had to fall back to Tier 0 line chunking.
    pub used_tier0_fallback: bool,
}

/// Apply both byte- and char-budget splits, gated behind the ~750-char knob.
///
/// Byte splitting is always applied (max_chunk_bytes default 24576). Char
/// splitting is gated by `chunk_max_chars_effective()` so default behavior
/// remains byte-identical when the knob is off (DEFAULT OFF).
fn apply_splits(chunks: Vec<Chunk>, config: &IndexingConfig) -> Vec<Chunk> {
    let chunks = chunker::split_oversized_chunks(chunks, config.max_chunk_bytes);
    let max_chars = config.chunk_max_chars_effective();
    if max_chars != 0 {
        chunker::split_oversized_chunks_by_chars(chunks, max_chars)
    } else {
        chunks
    }
}

/// Parse a source file and return both code chunks and parser diagnostics.
pub fn parse_file_with_diagnostics(
    source: &str,
    file_path: &str,
    language: Language,
    config: &IndexingConfig,
) -> Result<(Vec<Chunk>, Vec<references::RawReference>, ParseDiagnostics)> {
    // Special-case formats that don't use standard symbol extraction.
    if language == Language::Markdown {
        let chunks = chunker::markdown_section_chunks(source, file_path);
        return Ok((apply_splits(chunks, config), Vec::new(), ParseDiagnostics::default()));
    }
    if language == Language::Rst {
        let (chunks, diagnostics) = parse_rst_section_chunks(source, file_path)?;
        return Ok((apply_splits(chunks, config), Vec::new(), diagnostics));
    }
    if language.prefers_file_chunking() {
        let chunks = chunker::whole_file_chunk(source, file_path, language);
        return Ok((apply_splits(chunks, config), Vec::new(), ParseDiagnostics::default()));
    }
    if uses_indexing_tier0_fallback(language) {
        let chunks = chunker::tier0_line_chunks(source, file_path, language);
        return Ok((
            apply_splits(chunks, config),
            Vec::new(),
            ParseDiagnostics {
                used_tier0_fallback: true,
                ..ParseDiagnostics::default()
            },
        ));
    }

    let grammar = match languages::tree_sitter_grammar_for_path(language, file_path) {
        Some(g) => g,
        None => {
            let chunks = chunker::tier0_line_chunks(source, file_path, language);
            return Ok((
                apply_splits(chunks, config),
                Vec::new(),
                ParseDiagnostics {
                    used_tier0_fallback: true,
                    ..ParseDiagnostics::default()
                },
            ));
        }
    };

    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .context("failed to load tree-sitter grammar")?;

    let tree = parser
        .parse(source, None)
        .context("tree-sitter parsing returned None")?;
    let tree_has_error = tree.root_node().has_error();

    // Single symbol extraction pass reused for both chunking and references.
    let symbols = extractor::extract_symbols(&tree, source.as_bytes(), language);
    let refs =
        references::extract_references_with_symbols(&tree, source.as_bytes(), language, &symbols);
    let chunks = chunker::chunks_from_symbols(&symbols, source, file_path, language, config);

    let used_tier0_fallback = chunks.is_empty() && !source.trim().is_empty();
    let chunks = if used_tier0_fallback {
        chunker::tier0_line_chunks(source, file_path, language)
    } else {
        chunks
    };

    Ok((
        apply_splits(chunks, config),
        refs,
        ParseDiagnostics {
            tree_has_error,
            used_tier0_fallback,
        },
    ))
}

fn uses_indexing_tier0_fallback(language: Language) -> bool {
    matches!(language, Language::Haskell)
}

/// Parse a source file and produce chunks (without references).
///
/// Convenience wrapper around [`parse_file_with_diagnostics`] for callers that only need chunks.
pub fn parse_and_chunk(
    source: &str,
    file_path: &str,
    language: Language,
    config: &IndexingConfig,
) -> Result<Vec<Chunk>> {
    let (chunks, _refs, _diagnostics) =
        parse_file_with_diagnostics(source, file_path, language, config)?;
    Ok(chunks)
}

fn parse_rst_section_chunks(
    source: &str,
    file_path: &str,
) -> Result<(Vec<Chunk>, ParseDiagnostics)> {
    let grammar = languages::tree_sitter_grammar(Language::Rst)
        .context("missing tree-sitter grammar for reStructuredText")?;

    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .context("failed to load reStructuredText grammar")?;

    let tree = parser
        .parse(source, None)
        .context("tree-sitter parsing returned None")?;
    let tree_has_error = tree.root_node().has_error();

    let headings = extractor::extract_rst_section_titles(&tree, source.as_bytes());
    if headings.is_empty() {
        return Ok((
            chunker::tier0_line_chunks(source, file_path, Language::Rst),
            ParseDiagnostics {
                tree_has_error,
                used_tier0_fallback: true,
            },
        ));
    }

    Ok((
        chunker::rst_section_chunks(source, file_path, &headings),
        ParseDiagnostics {
            tree_has_error,
            used_tier0_fallback: false,
        },
    ))
}

/// Parse a source file and extract call-site references only.
///
/// Only works for languages with tree-sitter grammars. Returns an empty
/// vec for unsupported languages or parse failures.
pub fn parse_and_extract_references(
    source: &str,
    language: Language,
) -> Vec<references::RawReference> {
    let grammar = match languages::tree_sitter_grammar(language) {
        Some(g) => g,
        None => return Vec::new(),
    };
    let mut parser = Parser::new();
    if parser.set_language(&grammar).is_err() {
        return Vec::new();
    }
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };
    references::extract_references(&tree, source.as_bytes(), language)
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod metadata_tests;
