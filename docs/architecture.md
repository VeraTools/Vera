# Architecture

## Workspace Crates

| Crate | Purpose | Entry point |
|-------|---------|-------------|
| `vera-core` | Parsing, indexing, storage, embedding, retrieval | `lib.rs` |
| `vera-cli` | CLI (clap derive macros) | `main.rs` |
| `vera-mcp` | MCP server (JSON-RPC over stdio) | `server.rs` |
| `vera-serve` | HTTP inference server (OpenAI/Cohere-compatible) | `lib.rs` |
| `eval` | Benchmark harness and metrics | `src/main.rs` |
| `tree-sitter-{sql,proto,vue,dockerfile,astro}` | Vendored C grammars (built via `cc`) | `build.rs` |

## vera-core modules

### `parsing/`: Language parsing & symbol extraction

Files: `mod.rs` (public API), `languages.rs` (grammar dispatch), `extractor.rs` (AST node → SymbolType), `chunker.rs` (symbol-aware, whole-file, and Tier 0 chunking).

Data flow: file → grammar lookup → tree-sitter parse (+ diagnostics) → node classification → chunk production.

### `embedding/`: Embedding generation

`EmbeddingProvider` trait with three implementations:
- `ApiEmbeddingProvider`: HTTP calls to OpenAI-compatible endpoints
- `LocalEmbeddingProvider`: ONNX Runtime inference with Jina v5 nano
- `Model2VecProvider`: Potion Code static embeddings on CPU

`DynamicProvider` dispatches between them at runtime based on `VERA_BACKEND` or backend CLI flags.

### `retrieval/`: Search pipeline

1. Query enters `search_service.rs`
2. BM25 (`bm25.rs`) and vector search (`vector.rs`) run in parallel
3. Results fused via RRF (`hybrid.rs`, k=60). `fuse_rrf_multi` generalizes fusion to N ranked lists.
4. Query-aware ranking and candidate shaping apply deterministic priors (`ranking.rs`, `search_service.rs`)
5. Top candidates reranked by cross-encoder (`reranker.rs` or `local_reranker.rs`)
6. Final `Vec<SearchResult>` returned

Deep search (`--deep`): `rag_fusion.rs` runs a cheap BM25 pre-filter to collect symbol names and file paths, then passes these as context hints to the LLM (`completion_client.rs`) which decomposes the query into targeted sub-queries (default 2). Sub-queries reuse the active `SearchContext`, and results merge with weighted RRF (original query gets 2x weight). Falls back to iterative symbol-following when no completion endpoint is configured.

Structural search:
- `references.rs`: exact caller lookups from the persisted call graph, returned as search-style snippets
- `structural.rs`: agent-oriented structural intents for definitions, env reads, routes, SQL, and explicit implementation lookups

### `storage/`: Persistent storage

- `metadata.rs`: SQLite: chunk metadata, file paths, content hashes, and persisted file-level index state used for health reporting
- `bm25.rs`: Tantivy: full-text BM25 index
- `vector.rs`: sqlite-vec: embedding vectors

All stored in `.vera/` at the project root.

### `indexing/`: Index build & update

- `pipeline.rs`: Full build: discover → parse → chunk → embed → store, including persisted parse diagnostics
- `update.rs`: Incremental: hash-based change detection, re-process only modified files, and refresh file-level index state

### Other modules

- `types.rs`: `Language` enum (60+ variants), `SearchResult`, `CodeChunk`, `SymbolType`
- `config.rs`: `RetrievalConfig`, `IndexConfig` defaults
- `local_models.rs`: Manages local embedding presets, custom ONNX embedding configs, and ORT/model assets under the Vera data directory (XDG-compliant)
- `discovery/`: File discovery with gitignore support, binary/size filtering
- `git_scope.rs`: Resolves `--changed`, `--since`, and `--base` into exact paths relative to the indexed directory, which may sit below the git repository root
- `chunk_text.rs`: Line-boundary text splitting for byte-budget enforcement

## vera-cli

`main.rs` parses args via clap. `commands/` contains the CLI subcommand implementations and helpers: `agent`, `config`, `doctor`, `explain_path`, `grep`, `index`, `mcp`, `overview`, `references` (also used by `dead-code`), `repair`, `search`, `serve`, `setup`, `stats`, `structural`, `uninstall`, `update`, `upgrade`, and `watch`.

## vera-mcp

`server.rs` routes JSON-RPC requests. `tools.rs` implements seven MCP tools: `search_code`, `get_stats`, `get_overview`, `regex_search`, `structural_search`, `find_references`, and `explain_path`. `search_code`, `structural_search`, and `find_references` auto-index and start a file watcher on first use. Search, references, and overview tools also accept changed-file git scopes.

## Adding a new language

1. Add `Language` variant in `types.rs` (alphabetical order)
2. Add extension mapping in `Language::from_extension()`
3. Add grammar dependency to `vera-core/Cargo.toml`
4. Wire grammar in `parsing/languages.rs` → `tree_sitter_grammar()`
5. Add node classifier in `parsing/extractor.rs` → `classify_node()`
6. Write tests: extension mapping, grammar loading, symbol extraction
