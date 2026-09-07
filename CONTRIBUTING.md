# Contributing

## Build from Source

Rust 1.88+ required (see `Cargo.toml` `rust-version` for exact MSRV).

```bash
git clone https://github.com/VeraTools/Vera.git
cd Vera
bash scripts/bootstrap-vendored-grammars.sh  # Downloads the vendored tree-sitter grammars used by CI.
cargo build
```

## Run Tests

```bash
cargo test --workspace       # all tests
cargo test -p vera-core      # core crate only
```

## Lint & Format

```bash
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

## Project Layout

| Crate | What it does |
|-------|-------------|
| `vera-core` | Parsing, indexing, storage, embedding, retrieval pipeline |
| `vera-cli` | CLI interface (clap) |
| `vera-mcp` | MCP server (JSON-RPC over stdio) |
| `vera-serve` | HTTP inference server |
| `eval` | Benchmark harness and evaluation tasks |

The core engine lives in `vera-core`. Most changes happen here:

- `parsing/`: tree-sitter grammars, AST chunking, symbol extraction
- `embedding/`: embedding providers (API, Potion Code, and local ONNX)
- `retrieval/`: BM25, vector search, RRF fusion, reranking
- `storage/`: SQLite metadata, Tantivy BM25 index, sqlite-vec vectors, and the flat SIMD vector sidecar
- `indexing/`: index build and incremental update pipeline

For how the pipeline fits together, see [docs/how-it-works.md](docs/how-it-works.md).

## Adding a Language

See [docs/architecture.md](docs/architecture.md#adding-a-new-language) for the step-by-step checklist. The full language list is at [docs/supported-languages.md](docs/supported-languages.md).

## Conventions

- **Error handling:** `anyhow::Result` in CLI code, `thiserror` for typed errors in `vera-core`
- **Async:** `tokio` runtime for I/O-bound work
- **Tests:** `#[cfg(test)]` modules at the bottom of source files, `tempfile` for filesystem tests
- **Commits:** `type(scope): description`: e.g. `feat(lang): add HTML support`, `fix(retrieval): handle empty query`

## Running Benchmarks

```bash
bash eval/setup-semble-corpus.sh                    # clone the pinned Semble corpus
cargo build --release
cargo run --release --bin vera-eval -- run \
  --tool vera-potion \
  --tasks-dir eval/tasks/semble \
  --corpus eval/semble-corpus.toml \
  --json-only
```

Benchmark details: [docs/benchmarks.md](docs/benchmarks.md).

## Branch Policy

Pull requests target `master` directly.
