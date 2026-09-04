<div align="center">

<img width="1584" height="539" alt="vera" src="https://github.com/user-attachments/assets/c866fc70-b1e6-400b-aaf7-fa68721a4955" />

# Vera

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/VeraTools/Vera/blob/master/LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![GitHub release](https://img.shields.io/github/v/release/VeraTools/Vera?include_prereleases&sort=semver)](https://github.com/VeraTools/Vera/releases)
[![Languages](https://img.shields.io/badge/languages-65%2B-green.svg)](docs/supported-languages.md)

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

**V**ector **E**nhanced **R**eranking **A**gent

Code search that combines BM25 keyword matching, vector similarity, and optional cross-encoder reranking. Supports 65 languages (61 with tree-sitter parsing), runs locally, returns structured results with file paths, line ranges, symbol metadata, and relevance scores.

</div>

## What's New

See [What's New](docs/whats-new.md) for release highlights from v1.0 onward, including search-quality measurements, local model changes, agent workflows, performance work, and reliability fixes.

## Quick Start

**1. Install**
```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

**2. Set up and index** (pick one)

Recommended: the Qwen preset via OpenRouter (best measured search quality, single API key):
```bash
vera setup --api --index .          # choose the Qwen preset, paste one key
```

The zero-setup local option (runs on CPU, no key, no GPU):
```bash
vera setup --potion-code --index .
```

Other backends:
```bash
vera setup                                  # Interactive wizard, indexes this project by default
vera setup --onnx-jina-coreml --index .     # Apple Silicon (M1/M2/M3/M4)
vera setup --onnx-jina-cuda --index .       # NVIDIA GPU
vera setup --onnx-jina-rocm --index .       # AMD GPU (ROCm, Linux)
vera setup --onnx-jina-openvino --index .   # Intel GPU (OpenVINO, Linux)
vera setup --onnx-jina-directml --index .   # DirectX 12 GPU (Windows)
```
The wizard also offers presets for OpenAI, Jina, and Voyage. The Qwen preset uses `qwen/qwen3-embedding-8b` + `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1` with a single shared key and the generic reranker protocol.

**3. Search**
```bash
vera search "authentication logic"
```

If the current project has no index, interactive search offers to create one. JSON and non-interactive searches still return the missing-index error.

The default local embedding model is [`minishlab/potion-code-16M-v2`](https://huggingface.co/minishlab/potion-code-16M-v2), a static embedding model that runs locally on CPU on any supported machine; no GPU or ONNX Runtime needed. Jina ONNX and CodeRankEmbed are opt-in alternatives. For the highest measured search quality, use the Qwen preset through OpenRouter instead; see [docs/models.md](docs/models.md).

## What Sets Vera Apart

| | |
|---|---|
| **Wins where it was never tuned** | Trails Semble by 0.008 nDCG on Semble's own benchmark set, but leads on the independent contamination set (10 fresh repositories, locally generated ground truth) and on recall@5. Vera refuses ground-truth-specific tuning, which costs home-field points and buys generalization. |
| **Fast at query time, tiny on disk** | 6.4 ms median query latency on the 1,251-task suite (local Potion Code defaults) with a 4.7 GB index for 63 repositories (6.8x smaller than Semble's 32 GB). Filter-during-scan, enabled by default in v1.4.0, halves filtered-query latency with ranking unchanged. |
| **Updates, not just re-indexes** | Incremental updates and watch mode keep the index current as files change. Persistent indexes survive restarts and are reused when identity checks pass. |
| **Single binary, 65 languages** | One static binary with 61 tree-sitter grammars compiled in. No Python, no language servers, no per-language toolchains. |
| **Built-in code intelligence** | Call graph analysis, reference finding, dead code detection, and project overview, all from the same index. |
| **Token-efficient for agents** | Returns symbol-bounded chunks, not entire files. 75-95% fewer tokens on typical queries. In a blind-graded agent benchmark, a mid-tier model with Vera reached the same answer quality while reading 17% fewer input tokens. |

Vera started after weeks of working on Pampax, a project I forked because it and other similar tools were missing what I wanted. I kept running into deep-rooted bugs, less-than-ideal design decisions, and thought I could build something better from the ground up. Every design choice comes from careful research, learning from other projects, benchmarking and evaluation. Take a look at the full [feature list](docs/features.md) to see everything Vera can do.

## Installation

Use the quick start above if you just want to get going. This section helps you pick the right backend.

```bash
bunx @vera-ai/cli install   # or: npx -y @vera-ai/cli install / uvx vera-ai install
```

### Pick Your Backend

Vera itself is always local: the index lives in `.vera/` per project, config and models in `$XDG_DATA_HOME/vera` (or `~/.vera` for existing installs). The backend choice only affects where embeddings and reranking run.

Pick the `vera setup` flag that matches your hardware from the quick start above. The full hardware-to-command matrix, step-by-step instructions, API provider options, Docker, and building from source live in the [Installation Guide](docs/installation.md).

API mode works with any OpenAI-compatible endpoint and needs no local compute. Use `vera setup --api --yes` with `EMBEDDING_MODEL_*` variables for non-interactive setup. The Qwen preset (`qwen/qwen3-embedding-8b` + `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1`) needs only one shared key and configures the generic reranker protocol automatically. Jina ONNX and CodeRankEmbed are opt-in alternatives. Reranking is opt-in and disabled by default. After the first index, `vera update .` only re-embeds changed files, so incremental updates are fast on any backend. Full details: [docs/models.md](docs/models.md).

<details>
<summary>MCP server</summary>

```bash
vera mcp   # or: bunx @vera-ai/cli mcp / uvx vera-ai mcp
```
Exposes `search_code`, `get_stats`, `get_overview`, `regex_search`, `structural_search`, `find_references`, and `explain_path`. `search_code`, `structural_search`, and `find_references` auto-index and start a file watcher on first use if no index exists.
The MCP surface stays intentionally small; use the CLI skill path when you need the full command set.

</details>

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

How to read this table honestly: the full-suite gap (`0.8437` vs `0.8514`) is measured on Semble's own benchmark, whose 63 repositories are also the development corpus Semble's ranker was tuned against; MinishLab publishes both Semble and the Potion embedding model. On the 320-task tuning subset Vera scores `0.8538` against `0.8494`, and on the independent contamination set, 10 repositories disjoint from Semble's 63 with locally generated ground truth, Vera scores `0.7674` against `0.7655`. Vera deliberately tunes ranking signals only through preregistered ablations and refuses ground-truth-specific rules, which costs home-field points on this table and is the same discipline that shows up as a win the moment the evaluation leaves Semble's corpus. Recall@5, the metric that matters most for feeding candidates to a model or reranker, favors Vera on the full suite (`0.9189` vs `0.9177`).

For agents in real coding sessions, the measurable effects are context and workflow, not just ranking: Vera returns symbol-bounded chunks (75-95% fewer tokens than file reads), ships incremental updates and watch mode so the index tracks edits, and in a blind-graded agent benchmark a mid-tier model reached the same answer quality while reading 17% fewer input tokens with Vera installed.

The Vera row is from the 9800X3D host and the Semble column predates the CPU change, so latency columns are indicative, not a controlled comparison. See [docs/benchmarks.md](docs/benchmarks.md) for the screening tables, the agent-level benchmark, and historical comparisons.

Full methodology and version history: [docs/benchmarks.md](docs/benchmarks.md).

## Configure Your AI Agent

`vera agent install` installs the Vera skill for supported coding agents and can add a short usage snippet to your project's `AGENTS.md`, `CLAUDE.md`, `COPILOT.md`, or editor rules file.

```bash
vera agent install
vera agent install --client all
```

If you use the [skills CLI](https://github.com/vercel-labs/skills), you can install Vera there too:

```bash
npx skills add VeraTools/Vera
```

If you skipped the prompt and want to add the instructions manually, use the snippet in the [Installation Guide](docs/installation.md#set-up-agent-skills).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
