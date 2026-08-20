# Features

Vera (Vector Enhanced Reranking Agent) is a code search tool that combines BM25 keyword matching, vector similarity, and cross-encoder reranking into a single retrieval pipeline. This page covers everything it can do.

## Search Pipeline

### Hybrid Retrieval (BM25 + Vector)

Every query runs two retrieval paths in parallel:

- **BM25 keyword search** via Tantivy. Handles exact identifiers, config keys, and literal strings. Sub-millisecond latency.
- **Vector similarity search** via sqlite-vec. Catches conceptual matches even when the exact words don't appear in the code.

Results from both paths merge through Reciprocal Rank Fusion (RRF), so a result that scores well in both lists rises to the top. Full details: [how-it-works.md](how-it-works.md).

### Cross-Encoder Reranking

After fusion, the top candidates go through a cross-encoder that reads query and candidate together as a single pair. This is the most impactful stage: it lifts MRR@10 from 0.39 to 0.60 (54% improvement). Most code search tools skip this step entirely.

Vera supports local cross-encoders (Jina) and remote reranking endpoints (Jina, Cohere, or Voyage AI `rerank-2` with `RERANKER_MODEL_BASE_URL=https://api.voyageai.com/v1`, with wire format handled automatically).

Large candidate sets are automatically batched (default 20 per request, configurable via `VERA_MAX_RERANK_BATCH`). Individual documents exceeding the reranker's context window are truncated at the last newline boundary before the character limit (default 4800, configurable via `VERA_MAX_RERANK_DOC_CHARS`). Both settings work automatically with no required configuration.

### Multi-Query Search

`vera search` accepts multiple quoted queries at once. Run 2-3 varied queries to capture different aspects of what you're looking for (e.g., "OAuth token refresh", "JWT expiry handling", "auth middleware"). Results merge with reciprocal rank fusion, which cuts round-trips and improves recall when one phrasing misses relevant code.

### Intent-Based Reranking

`vera search --intent "goal"` lets you describe your higher-level goal separately from the search query. Vera folds that intent into the retrieval query so ranking can target what you actually need, not just the short keyword you typed. This is useful when the query is ambiguous or too short to convey enough context.

### Deep Search

`vera search "query" --deep` runs a BM25 pre-filter to collect real symbol names and file paths from the index, feeds those as context hints to an LLM completion endpoint, and decomposes the query into targeted sub-queries that each search for a different code location or concept. Sub-queries run in parallel, and results merge via weighted Reciprocal Rank Fusion (the original query counts double). This finds code across multiple relevant locations rather than just rephrasing the same intent.

Requires a completion endpoint: set `VERA_COMPLETION_BASE_URL` and `VERA_COMPLETION_MODEL_ID` (any OpenAI-compatible chat endpoint works, including local llama.cpp). When no completion endpoint is configured, `--deep` falls back to iterative symbol-following: it extracts symbol names from top results and searches for those symbols automatically.

### Compact Mode

`vera search "query" --compact` strips function and class bodies from results, returning only signatures (name, parameters, return type). This fits more results into fewer tokens, making it useful for broad exploration before drilling into specific implementations. Works with `vera grep` too. Falls back to the first 3 lines for languages or chunks where body stripping isn't applicable.

### Git-Aware Search Scopes

When a task is limited to modified files or a PR diff, scope the search before broadening the query:

- `--changed`: modified, staged, and untracked files in the current working tree
- `--since <rev>`: files changed since a specific revision
- `--base <rev>`: files changed since `merge-base(HEAD, <rev>)`

These flags work with `vera search`, `vera grep`, and `vera overview`.

### Query-Aware Ranking

A deterministic ranking stage between fusion and reranking handles cases that dense retrieval alone is bad at: exact filename queries, path-heavy config lookups, noisy test/docs matches, and broad natural-language queries that need structural results. It also pulls in related implementation blocks or same-file context when the initial hit is too narrow.

### Corpus-Aware Search Scopes

Filter results by corpus category with `--scope`:

| Scope | What it includes |
|-------|-----------------|
| `source` | Application source code (default bias) |
| `docs` | Markdown, READMEs, ADRs, guides |
| `runtime` | Extracted runtime trees, bundled app code |
| `all` | Everything, no filtering |

Vera favors source files by default. Docs, runtime extracts, and generated/minified files are still indexed but deprioritized unless you explicitly request them. Add `--include-generated` to include dist/minified/generated artifacts.

### Search Filters

Narrow results by language (`--lang rust`), file path glob (`--path "src/**/*.rs"`), symbol type (`--type function`), corpus scope (`--scope source`), and result count (`--limit 5`). Repeat `--path` for multiple patterns. Path patterns use OR semantics; other filters combine with AND semantics.

## Parsing and Indexing

### Tree-Sitter Structural Parsing

65 languages supported, 61 with tree-sitter grammars compiled into the binary. Functions, classes, structs, traits, interfaces, methods, and `impl` blocks are extracted as discrete chunks. Results map to actual symbol boundaries, not arbitrary line ranges. The remaining 4 formats (TOML, YAML, JSON, and Markdown) use text-based chunking.

Symbol-aware chunking scores 2.3x higher MRR on symbol lookup than sliding-window chunking (0.55 vs 0.24), while using 14% fewer tokens. Full list: [supported-languages.md](supported-languages.md).

### Adaptive Chunking

Large symbols (>200 lines) are split at logical boundaries: closing braces, semicolons, blank lines. This preserves readability instead of cutting at arbitrary line counts. Languages without a tree-sitter grammar fall back to sliding-window chunking. Module-level gaps between symbols are kept as chunks when they carry useful retrieval context.

Chunks that exceed the embedding model's input limit are automatically split in a post-processing pass. API mode uses a 24KB byte budget (roughly 6K-7K tokens, safe for any modern embedding model). Local mode uses the model's own tokenizer and max_length. Override with `VERA_MAX_CHUNK_BYTES` if needed.

### Incremental Updates

`vera update .` compares content hashes against the stored index and only re-parses, re-chunks, and re-embeds changed files. For small changes this takes seconds, not minutes.

Pass `--max-files <N>` to bound the number of added or modified files processed in a single run. Any remaining files are reported as deferred in the update summary so you can process them in subsequent runs.

### File Watching

`vera watch .` monitors the project for file changes and triggers incremental index updates automatically (debounced at 2s). Keeps the index fresh during long coding sessions without manual intervention. Progress logs print to stderr so you can see when updates start, complete, or skip.

### Flexible Exclusions

Vera respects `.gitignore` by default. For more control, `.veraignore` (gitignore syntax) gives full control over what gets indexed. Use `#include .gitignore` at the top to layer extra exclusions on top of gitignore rules. One-off `--exclude` flags, `--no-ignore`, and `--no-default-excludes` are also available.

### Path Explainability

`vera explain-path path/to/file` explains the decisive reason a file is or is not indexed. It reports default excludes, `.veraignore`, `.ignore`, `.gitignore`, size limits, binary detection, missing files, and more. Use this instead of guessing when a file is unexpectedly missing from search results.

### Progress Reporting

Indexing and updating show an interactive progress display with file discovery, classification, parsing, and embedding generation phases. Pass `--no-progress` to disable interactive progress bars, or use `--json` for machine consumption.

### Verbose Indexing

`vera index . --verbose` shows detailed skipped-file output for categories Vera discovers after walking the tree, such as oversized files. For exact exclusion debugging, use `vera explain-path path/to/file`.

### Index Health

`vera stats` shows persisted file-level index health. `vera stats --json` returns the same data in machine-readable form:

- files whose tree-sitter parse tree contained error nodes
- files that fell back to Tier 0 line chunking
- files that failed to parse and therefore produced no indexed chunks

This makes parser regressions and partial indexing visible instead of silent.

## Code Intelligence

### Call Graph and Reference Finding

`vera references foo` finds all callers of a symbol as search-style snippets. `vera references foo --callees` finds what a symbol calls. Add `--changed`, `--since`, or `--base` when you want exact call relationships limited to a diff. The call graph is built during indexing from tree-sitter AST analysis, so lookups are instant.

### Agent-Oriented Structural Search

`vera structural <intent> [query]` covers the common structural tasks agents hit repeatedly without forcing raw tree-sitter syntax.

- `definitions <symbol>` finds symbol definitions by name
- `env [NAME]` finds environment variable reads, optionally narrowed to one variable
- `routes` finds common HTTP route registrations
- `sql` finds common SQL execution sites
- `impls <symbol>` finds explicit implementations, conformances, and inheritance declarations

`impls` only returns explicit declarations. It does not guess implicit interface satisfaction in languages where that would require semantic analysis.

Use this as the default structural workflow. Use `vera references` for exact caller/callee questions.

### Dead Code Detection

`vera dead-code` finds functions and methods with no callers. Excludes common entry points (`main`, `new`, `default`, etc.) to reduce noise. Useful for codebase cleanup and understanding which code is actually reachable.

### Project Overview

`vera overview` generates an architecture summary: language breakdown, directory structure, entry points, symbol type distribution, complexity hotspots, and detected project conventions (frameworks, patterns, config files). Designed for agent onboarding so an AI agent can understand a codebase before diving in.

### Regex Search

`vera grep "pattern"` runs regex search over indexed files with configurable context lines, case sensitivity, and the same corpus filters as `vera search` (`--lang`, `--path`, `--type`, `--scope`). Repeat `--path` to match any of several patterns. It complements semantic search for exact string matching, import statements, TODOs, and known identifiers.

## Model Backend

### Local-First, Model-Agnostic

Indexing, storage, and search always stay on your machine. The backend choice only affects where embeddings and reranking run:

- **Potion Code CPU**: `vera setup --potion-code` downloads static code embeddings and runs without ONNX Runtime.
- **Jina ONNX GPU**: `vera setup --onnx-jina-cuda` or another `--onnx-jina-*` flag downloads curated ONNX models. The full pipeline (BM25 + vector + rerank) runs without external calls.
- **API mode**: Point at any OpenAI-compatible endpoint (remote APIs or local servers like llama.cpp). Only model calls leave your machine. Query prefixes for asymmetric embedding models (Qwen3, CodeRankEmbed, E5, BGE) are auto-detected from the model ID. Override with `EMBEDDING_QUERY_PREFIX` for unsupported models. See [llama-cpp-setup.md](llama-cpp-setup.md) for a step-by-step guide.

### Curated Local Models

Potion Code is the CPU-first local model:

| Model | Role |
|-------|------|
| [minishlab/potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) | Static code embedding model |

The Jina ONNX backends use these local models:

| Model | Role |
|-------|------|
| [jina-embeddings-v5-text-nano-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) | Embedding model |
| [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | Cross-encoder reranker |

An optional [CodeRankEmbed](https://huggingface.co/Zenabius/CodeRankEmbed-onnx) preset is available for embedding-heavy or no-rerank experiments. Details: [models.md](models.md).

### GPU Acceleration

Auto-detected during setup. Supported backends:

| Flag | Hardware |
|------|----------|
| `--potion-code` | CPU-only local inference |
| `--onnx-jina-cuda` | NVIDIA (CUDA 12+) |
| `--onnx-jina-rocm` | AMD (Linux, ROCm) |
| `--onnx-jina-directml` | Any DirectX 12 GPU (Windows) |
| `--onnx-jina-coreml` | Apple Silicon (macOS) |
| `--onnx-jina-openvino` | Intel GPU/iGPU (Linux) |

Vera downloads the matching ONNX Runtime build automatically. For OpenVINO and ROCm, it installs via pip into a managed venv.

### Adaptive GPU Batching

Local ONNX indexing shapes micro-batches from actual token lengths rather than using a fixed batch size. Long inputs shrink roughly with the square of padded sequence length. Vera learns safe batch windows per length bucket from real successes and allocation failures, persists them across runs, and warm-starts on the next session. On constrained GPUs, pass `--low-vram` to force conservative settings.

### Custom Local Embeddings

Swap the Jina ONNX embedding model without changing the rest of that pipeline. Point at a Hugging Face repo, a direct URL, or a local directory with custom pooling, query prefix, and dimension settings. The local reranker stays on the curated Jina model.

## Output and Integration

### Token-Efficient Output

Default markdown codeblock format cuts ~35-40% tokens vs JSON. On a 20-query benchmark, Vera's chunk-level output averages 67% fewer tokens than loading the full files containing the same results. Most queries see 75-95% reduction.

### Response Truncation

Output is progressively truncated to fit a total character budget (`retrieval.max_output_chars`, default 12,000) so results do not blow up LLM context windows. Lower-ranked results are truncated first, at a line boundary, with a `[...truncated]` marker. Short results pass through unchanged.

### Multiple Output Formats

| Flag | Output |
|------|--------|
| *(default)* | Markdown codeblocks with file path, line range, and symbol metadata |
| `--json` | Compact single-line JSON |
| `--raw` | Verbose human-readable output for `search`, `grep`, and `references`. Works before or after the subcommand. |
| `--timing` | Timing info to stderr (`search`: per-stage, `grep`: total). Works before or after the subcommand. |

## MCP Server

`vera mcp` exposes a small set of high-value tools over JSON-RPC (stdio), compatible with any MCP client:

| Tool | What it does |
|------|-------------|
| `search_code` | Hybrid search with multi-query, intent, all filters, and git-scoped changed-file search. Auto-indexes and starts watcher on first use. |
| `get_stats` | File count, chunk count, index size, language breakdown, and index health |
| `get_overview` | Architecture overview with conventions detection and optional git-scoped filtering |
| `regex_search` | Regex search with context lines, scope controls, and git-scoped filtering |
| `structural_search` | Agent-oriented structural intents for definitions, env reads, routes, SQL, and explicit implementation lookups |
| `find_references` | Exact callers or callees from the persisted call graph, with optional git-scoped filtering |
| `explain_path` | Explain why a file is or is not indexed |

Tool descriptions include explicit WHEN TO USE / WHEN NOT TO USE guidance so AI agents route queries to the right tool automatically.

Docker images available for CPU, CUDA, ROCm, and OpenVINO. Details: [docker.md](docker.md).

## Agent Integration

### Skill Files for 31 Agent Clients

`vera agent install` installs skill files that teach AI agents how to write effective queries, when to use semantic search vs regex, and how to interpret results. Supports Junie, Claude Code, Cursor, Windsurf, Copilot, Cline, Roo Code, and 24 more agent clients. Skills install globally or per-project.

### Agent Config Snippets

During setup, Vera offers to add a usage snippet to your project's agent config file (`AGENTS.md`, `CLAUDE.md`, `COPILOT.md`, `.cursorrules`, `.clinerules`, `.windsurfrules`) so agents discover Vera automatically.

### Syncing Stale Skills

`vera agent sync` refreshes stale agent skill installs to match the current binary version and updates Vera-owned project markdown snippets: sections wrapped in `<!-- vera:begin -->` / `<!-- vera:end -->` markers, plus unmarked legacy sections whose text still matches the generated snippet exactly (these are migrated into markers on sync). Editing anything in the section, or deleting it, takes it out of sync scope permanently. When Vera notices stale installs during normal CLI use, it refreshes skill files only, silently, and leaves project markdown unchanged.

## CLI Tooling

### Interactive Setup Wizard

`vera setup` walks through backend selection, agent skill installation, and optional project indexing in one command. Skip the wizard with flags for non-interactive use.

### Backend Management

`vera backend` manages the model backend separately from the full setup wizard. Switch GPU backends, pick Potion Code CPU, swap ONNX embedding models, or reconfigure API endpoints without re-running setup.

### HTTP Inference Server

`vera serve` starts a local HTTP daemon exposing OpenAI-compatible and Cohere-compatible inference endpoints for standard Vera clients and external tools:

- `POST /v1/embeddings`: OpenAI-compatible embedding format
- `POST /v1/rerank`: Cohere/Jina-compatible reranker format
- `GET /v1/health`: liveness probe and active model metadata

Key flags:
- `--port <PORT>`: TCP port to listen on (default: 3000)
- `--host <HOST>`: bind address (default: 127.0.0.1)
- `--api-key <KEY>`: require Bearer token authentication (or set `VERA_SERVE_KEY`)
- `--idle-timeout <SECS>`: seconds of inactivity before unloading models from memory (0 = reload per request, -1 = keep loaded indefinitely, default: 0)
- Backend flags: `--potion-code`, `--onnx-jina-cuda`, `--onnx-jina-rocm`, `--onnx-jina-coreml`, `--onnx-jina-openvino`, `--onnx-jina-directml`, or `--api`

### Diagnostics

`vera doctor` reports the saved and active backend, installed version, checks GitHub for newer releases, and detects missing, corrupt, or truncated ONNX model caches. If model assets are damaged, it suggests running `vera repair --<backend>`; runtime embedding load errors provide the same repair hint. `--probe` adds a deeper read-only local backend check. `--json` outputs machine-readable diagnostics. It exits 1 when any check fails, matching the `overall_ok` field in the JSON report; warnings leave the exit code at 0. `vera repair` re-fetches missing or corrupt local assets.

### Self-Updating

`vera upgrade` inspects the current update plan and shows the exact command it would run. `--apply` executes it. Vera checks for new releases once per day and prints a hint when a newer version is available.

### Configuration and Stats

`vera config` shows the current configuration. `vera stats` shows index statistics plus persisted health signals such as tree-sitter errors, Tier 0 fallback, and parse failures.

To allow switching between equivalent embedding model names without triggering a re-index, configure `embedding.model_aliases` in config or export `VERA_EMBEDDING_MODEL_ALIASES` (semicolon-separated groups of comma-separated aliases, such as `canonical,alias;other,other-alias`).

### Uninstalling

`vera uninstall` removes Vera's data directory (models, ONNX Runtime libs, config), agent skill files, and the PATH shim. Per-project indexes (`.vera/` in each project) are left in place.

### Cross-Platform

Single static binary for Linux (x86_64, aarch64), macOS (x86_64, aarch64), and Windows (x86_64). Install via npm (`bunx @vera-ai/cli install`), pip (`uvx vera-ai install`), prebuilt binary, Docker, or build from source.

A fully static musl-linked binary (`x86_64-unknown-linux-musl`) is available for environments without standard shared libraries (NixOS, Alpine, minimal containers). It has zero runtime dependencies. The npm and pip wrappers auto-detect musl-based systems and select the correct binary. To override target selection manually, set `VERA_TARGET` (e.g., `VERA_TARGET=x86_64-unknown-linux-musl bunx @vera-ai/cli install`). The chosen target is stored in `~/.vera/install.json` so upgrades preserve it.

## Benchmarks

21-task benchmark across `ripgrep`, `flask`, `fastify`, and `turborepo`:

| Metric | ripgrep | cocoindex | ColGREP (149M) | Vera |
|--------|---------|-----------|----------------|------|
| Recall@1 | 0.15 | 0.16 | 0.57 | **0.72** |
| Recall@5 | 0.28 | 0.37 | 0.67 | **0.78** |
| MRR@10 | 0.26 | 0.35 | 0.62 | **0.91** |
| nDCG@10 | 0.29 | 0.52 | 0.56 | **0.84** |

Full methodology and additional comparisons: [benchmarks.md](benchmarks.md).
