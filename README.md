<div align="center">

<img width="1584" height="539" alt="vera" src="https://github.com/user-attachments/assets/c866fc70-b1e6-400b-aaf7-fa68721a4955" />

# Vera

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/VeraTools/Vera/blob/master/LICENSE)
[![CI](https://github.com/VeraTools/Vera/actions/workflows/ci.yml/badge.svg)](https://github.com/VeraTools/Vera/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@vera-ai/cli)](https://www.npmjs.com/package/@vera-ai/cli)
[![PyPI](https://img.shields.io/pypi/v/vera-ai)](https://pypi.org/project/vera-ai/)
[![GitHub release](https://img.shields.io/github/v/release/VeraTools/Vera?include_prereleases&sort=semver)](https://github.com/VeraTools/Vera/releases)
[![Languages](https://img.shields.io/badge/languages-65%2B-green.svg)](docs/supported-languages.md)

[Docs](docs/README.md)
·
[Install Guide](docs/installation.md)
·
[Features](docs/features.md)
·
[Query Guide](docs/query-guide.md)
·
[Benchmarks](docs/benchmarks.md)
·
[How It Works](docs/how-it-works.md)
·
[Models](docs/models.md)
·
[Supported Languages](docs/supported-languages.md)

**Local, symbol-aware code search for developers and AI agents.**

Hybrid BM25 + vector search with optional reranking, 65 languages, one static binary. Indexes stay on your machine; results come back as symbol-bounded chunks with file paths, line ranges, and scores.

<sub>**V**ector **E**nhanced **R**eranking **A**gent</sub>

</div>

![vera search demo](docs/assets/vera-demo.gif)

## Quick Start

**1. Install**
```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

**2. Set up and index**

Zero-setup local (CPU, no key, no GPU):
```bash
vera setup --potion-code --index .
```

Best measured search quality (one OpenRouter key, Qwen preset):
```bash
vera setup --api --index .
```

<details><summary>GPU and other backends</summary>

```bash
vera setup                                  # Interactive wizard, indexes this project by default
vera setup --onnx-jina-coreml --index .     # Apple Silicon (M1/M2/M3/M4)
vera setup --onnx-jina-cuda --index .       # NVIDIA GPU
vera setup --onnx-jina-rocm --index .       # AMD GPU (ROCm, Linux)
vera setup --onnx-jina-openvino --index .   # Intel GPU (OpenVINO, Linux)
vera setup --onnx-jina-directml --index .   # DirectX 12 GPU (Windows)
```

The wizard also offers presets for OpenAI, Jina, and Voyage. The Qwen preset uses `qwen/qwen3-embedding-8b` + `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1` with a single shared key and the generic reranker protocol.

</details>

**3. Search**
```bash
vera search "authentication logic"
```

If the current project has no index, interactive search offers to create one. JSON and non-interactive searches still return the missing-index error.

**4. Keep `.vera/` out of git**
```bash
echo '.vera/' >> .gitignore
```
The index can be large and is machine-local.

See [What's New](docs/whats-new.md) for release notes.

## What Sets Vera Apart

| | |
|---|---|
| **Token-efficient for agents** | Returns symbol-bounded chunks, not entire files. 75-95% fewer tokens on typical queries. In a blind-graded agent benchmark, a mid-tier model with Vera reached the same answer quality while consuming 27-48% less context (48% with the Qwen API embedding+reranker pair, 27% with local Potion defaults). |
| **Single binary, 65 languages** | One static binary with 61 tree-sitter grammars compiled in. No Python, no language servers, no per-language toolchains. |
| **Fast at query time, tiny on disk** | 6.4 ms median query latency on the 1,251-task suite (local Potion Code defaults) with a 4.7 GB index for 63 repositories (6.8x smaller than Semble's 32 GB). |
| **Updates, not just re-indexes** | Incremental updates and watch mode keep the index current as files change. Persistent indexes survive restarts and are reused when identity checks pass. |
| **Built-in code intelligence** | Call graph analysis, reference finding, dead code detection, and project overview, all from the same index. |
| **Generalizes off its training set** | Leads Semble on the independent contamination set (10 fresh repositories, locally generated ground truth) and on recall@5, while trailing by 0.008 nDCG on Semble's own 63-repo benchmark. Details in [Benchmarks](#benchmarks). |

Vera started as a fork of Pampax. When the design stopped fitting what I wanted from a code search tool, I rebuilt it from the ground up, with each choice backed by research, benchmarking, and the [ADRs](docs/adr/000-decision-summary.md) in this repo. The full [feature list](docs/features.md) covers everything Vera can do.

## Choosing a Backend

Vera itself is always local: the index lives in `.vera/` per project, config and models in the Vera data directory (see [Installation](docs/installation.md#set-up-a-backend)). The backend choice only affects where embeddings and reranking run.

API mode works with any OpenAI-compatible endpoint and needs no local compute. The [models guide](docs/models.md) and [installation guide](docs/installation.md) cover provider options, setup flags, Docker, and building from source.

## Requirements

- Linux x86_64/aarch64 (glibc or musl), macOS x86_64/arm64, or Windows x86_64.
- No runtime dependencies; Python and Node are only needed to run the installer wrappers.
- The default local Potion model runs on CPU.
- `.vera/` size scales with the repository; the 63-repository benchmark used 4.7 GB.

## Privacy

Local modes send nothing off-machine. API mode sends chunk text and queries to the configured endpoint. The update check contacts GitHub once a day and is disabled with `VERA_NO_UPDATE_CHECK=1`.

## Vera vs. Other Tools

| Tool | Concept queries | Exact/regex | Offline | Works for agents (MCP/CLI) |
|---|---|---|---|---|
| ripgrep | No | Yes | Yes | CLI |
| LSP “find references” | Limited to language-server semantics | Symbol-aware | Yes | Client-dependent |
| Editor semantic search or Sourcegraph-style | Often | Varies | Varies | Client-dependent |
| Vera | Yes, through hybrid search | Yes, through `vera grep` and structural queries | Yes in local modes | MCP and CLI |

## Use with AI Agents

`vera agent install` installs the Vera skill for supported coding agents and can add a short usage snippet to your project's `AGENTS.md`, `CLAUDE.md`, `COPILOT.md`, or editor rules file.

```bash
vera agent install
vera agent install --client all
```

### MCP

Claude Code:
```bash
claude mcp add vera -- vera mcp
```

Cursor, Windsurf, and generic MCP clients:
```json
{"mcpServers":{"vera":{"command":"vera","args":["mcp"]}}}
```

Codex:
```toml
[mcp_servers.vera]
command = "vera"
args = ["mcp"]
```

Vera exposes `search_code`, `get_stats`, `get_overview`, `regex_search`, `structural_search`, `find_references`, and `explain_path`. See [MCP integration](docs/mcp.md) for client-specific setup and tool details.

If you use the [skills CLI](https://github.com/vercel-labs/skills), you can install Vera there too:

```bash
npx skills add VeraTools/Vera
```

## Usage

### Core Workflow

```bash
vera search "authentication logic"
vera update .
```

### Search Patterns

```bash
vera search "error handling" --lang rust
vera search "routes" --path "src/**/*.ts" --path "tests/**/*.ts"
vera search "handler" --type function --limit 5
vera search "OAuth token refresh" "JWT expiry handling" "auth middleware"
vera search "config" --intent "find where database connection strings are loaded"
vera search "config loading" --deep
vera search "auth" --compact
vera search "token validation" --changed
vera search "config loading" --base origin/main
vera structural definitions parse_config
vera structural env DATABASE_URL
vera structural routes --path "src/**/*.ts"
vera structural impls Loader
vera references parse_config --changed
```

Repeat `--path` to match any of several file path patterns. Path patterns use OR semantics; other filters still combine with AND semantics.

### Common Tasks

| Task | Command |
|------|---------|
| Regex or exact text | `vera grep "fn\s+main"` |
| Common structural tasks | `vera structural routes` / `vera structural env DATABASE_URL` / `vera structural impls Loader` |
| Explain why a file is missing from the index | `vera explain-path path/to/file` |
| Inspect index health | `vera stats --json` |
| Find callers | `vera references foo` |
| Find callees | `vera references foo --callees` |
| Find dead code | `vera dead-code` |
| Get a project overview | `vera overview` |
| Scope a search to changed files | `vera search "query" --changed` |
| Keep the index fresh | `vera watch .` |
| Run local HTTP inference server | `vera serve` |
| Check your setup | `vera doctor` |
| Repair missing local assets | `vera repair` |
| Install agent skills | `vera agent install` |

See the [query guide](docs/query-guide.md) for search tips, the [feature list](docs/features.md) for the full command surface, and `vera --help` for CLI details.

### Output

Defaults to markdown codeblocks (the most token-efficient format for AI agents):

````
```src/auth/login.rs:42-68 function:authenticate
pub fn authenticate(credentials: &Credentials) -> Result<Token> { ... }
```
````

Use `--json` for compact JSON. `--raw` works with `vera search`, `vera grep`, and `vera references`; `--timing` works with `vera search` and `vera grep`. You can place them before or after the subcommand (for example, `vera --timing search "auth"` or `vera references parse_config --raw`).

### Excluding Files

Vera respects `.gitignore` by default. Create a `.veraignore` file (gitignore syntax) for more control, or use `--exclude` flags. Details: [docs/features.md](docs/features.md#flexible-exclusions).

If a file is missing from the index and you need the exact reason, run:

```bash
vera explain-path path/to/file
```

## Benchmarks

Semble benchmark comparison on 1,251 tasks across 63 repositories (Vera v1.4.0 row measured 2026-09-03 on AMD Ryzen 7 9800X3D; Semble column from the 2026-08-23 comparison on the same task set and embeddings):

| Tool | nDCG@10 | R@1 | R@5 | R@10 | MRR | Query p50 | Index time | Index size |
|------|---------|------|------|-------|-----|-----------|------------|------------|
| Vera | 0.8437 | 0.6713 | **0.9189** | 0.9502 | 0.8258 | 6.4 ms | 115 s | **4.7 GB** |
| Semble 0.5.5, full rerank stack | **0.8514** | **0.6747** | 0.9177 | **0.9656** | **0.8348** | **2.3 ms** | **100 s** | 32 GB |

Both tools used the same `minishlab/potion-code-16M-v2` embeddings, harness, graded relevance, and suffix-corrected path matching in the scorer.

The full-suite gap is on Semble's own development corpus. On the 320-task tuning subset and the independent 10-repository contamination set, Vera leads (`0.8538` vs `0.8494`, `0.7674` vs `0.7655`); Recall@5 favors Vera on the full suite. See [full benchmark results](docs/benchmarks.md#current-results).

For agents in real coding sessions, Vera returns symbol-bounded chunks (75-95% fewer tokens than file reads), ships incremental updates and watch mode so the index tracks edits, and in a blind-graded four-arm agent benchmark a mid-tier model reached the same answer quality while consuming 27% less context with local Potion defaults and 48% less with the Qwen API embedding+reranker pair.

## Status and Community

Vera has a stable v1.x CLI. The MCP surface is intentionally small.

Report problems in [Issues](https://github.com/VeraTools/Vera/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
