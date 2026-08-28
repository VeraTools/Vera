//! Persistent storage backends for Vera's index.
//!
//! This module provides three storage components:
//! - [`metadata::MetadataStore`] — SQLite-based chunk metadata storage
//! - [`vector::VectorStore`] — dual sqlite-vec and flat SIMD vector storage
//! - [`bm25::Bm25Index`] — Tantivy-based BM25 full-text search index
//!
//! These are composed by the indexing pipeline and retrieval engine.

pub mod bm25;
pub mod metadata;
pub mod vector;

/// Maximum number of SQL parameters in a single query (`IN (...)`).
///
/// SQLite's default `SQLITE_MAX_VARIABLE_NUMBER` is 999; batching at 900
/// leaves headroom for other parameters in the same statement.
pub const SQL_PARAMETER_BATCH: usize = 900;

/// Build a comma-separated list of `?` placeholders for `count` parameters.
pub fn sql_placeholders(count: usize) -> String {
    std::iter::repeat_n("?", count)
        .collect::<Vec<_>>()
        .join(",")
}
