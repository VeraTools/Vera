//! Vera MCP Server — Model Context Protocol interface for Vera.
//!
//! Exposes Vera's indexing and retrieval capabilities as MCP tools over
//! JSON-RPC 2.0 with stdio transport:
//! - `search_code` — hybrid search (auto-indexes and watches on first use)
//! - `get_stats` — retrieve index statistics
//! - `get_overview` — summarize project structure
//! - `regex_search` — regex search over indexed files
//!
//! # Usage
//!
//! Start the server with `vera mcp`. The server reads JSON-RPC messages
//! from stdin and writes responses to stdout. Logs go to stderr.

pub mod protocol;
pub mod saved_config;
pub mod server;
pub mod staleness;
#[cfg(test)]
mod test_support;
pub mod tools;
pub mod watcher;
