//! Vera Core — parsing, indexing, retrieval, storage, and embedding engine.
//!
//! This crate provides the core functionality for Vera's code indexing and
//! retrieval pipeline. It is consumed by `vera-cli` (CLI interface) and
//! `vera-mcp` (MCP server).
//!
//! # Module overview
//!
//! - [`parsing`] — Tree-sitter based source code parsing and AST extraction.
//! - [`indexing`] — Building and maintaining the search index (BM25, vectors).
//! - [`retrieval`] — Hybrid search, fusion, and reranking pipeline.
//! - [`storage`] — Persistent storage backends (SQLite + sqlite-vec, Tantivy).
//! - [`embedding`] — Embedding generation via external API providers.

mod cancellation;
pub mod chunk_text;
pub mod corpus;
pub mod discovery;
pub mod embedding;
pub mod git_scope;

pub use cancellation::CancellationToken;

/// Install the rustls crypto provider (ring). Safe to call multiple times.
pub fn init_tls() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

pub mod indexing;
pub mod parsing;
pub(crate) mod path_containment;
pub mod presentation;
pub mod retrieval;
pub mod storage;

/// Shared types used across multiple modules.
pub mod types;

/// Configuration types and defaults.
pub mod config;

pub mod local_models;

/// Index statistics collection.
pub mod stats;
