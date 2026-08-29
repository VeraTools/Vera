# @vera-ai/cli

Code search for AI agents. Vera indexes your codebase using tree-sitter parsing and hybrid search (BM25 + vector similarity + optional cross-encoder reranking), then returns ranked code snippets as Markdown codeblocks by default, or JSON with `--json`.

This package downloads and wraps the native Vera binary for your platform. On musl-based Linux (Alpine, NixOS), the correct static binary is selected automatically. Set `VERA_TARGET` to override target detection (e.g., `VERA_TARGET=x86_64-unknown-linux-musl npm install -g @vera-ai/cli`).

The default local embedding model is `minishlab/potion-code-16M-v2`; it runs locally on CPU on any supported machine, no GPU or ONNX Runtime needed. In the current Semble comparison, Vera v1.2.0 scored `0.8450` nDCG@10 versus Semble 0.5.5 at `0.8514`, using the same embeddings, harness, graded relevance, and suffix-corrected path matching. Full details live in the main repo docs.

## Install

```bash
npm install -g @vera-ai/cli
```

## Quick Start

```bash
vera setup --potion-code --index .
vera search "authentication logic"
```

`vera setup` with no flags runs an interactive wizard and offers to index the current project, defaulting to yes. An interactive search also offers to create a missing index. `vera setup --api` prompts for an OpenAI-compatible endpoint and key; the wizard offers presets for OpenAI, Jina, Voyage, and Qwen via OpenRouter, with the Qwen preset needing only one shared key (`qwen/qwen3-embedding-8b` + `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1`). Use `--yes` with `EMBEDDING_MODEL_*` variables for non-interactive setup. `vera agent install` manages skill files for your coding agents and can update `AGENTS.md` / `CLAUDE.md` style project instructions.

## Common Tasks

| Task | Command |
|------|---------|
| Use the interactive setup wizard | `vera setup` |
| Use the default local model | `vera setup --potion-code` |
| Configure API mode | `vera setup --api` |
| Use a local NVIDIA backend | `vera setup --onnx-jina-cuda` |
| Search semantically | `vera search "authentication middleware"` |
| Search only changed files | `vera search "authentication middleware" --changed` |
| Common structural tasks | `vera structural routes` / `vera structural env DATABASE_URL` / `vera structural impls Loader` |
| Find callers or callees | `vera references foo` / `vera references foo --callees` |
| Explain why a file is missing | `vera explain-path path/to/file` |
| Inspect index health | `vera stats --json` |
| Keep the index up to date | `vera update .` |
| Watch for file changes | `vera watch .` |
| Run local HTTP inference server | `vera serve` |
| Diagnose setup issues | `vera doctor` |
| Run the deeper local probe | `vera doctor --probe` |
| Repair missing local assets | `vera repair` |
| Inspect binary upgrades | `vera upgrade` |
| Install agent skills | `vera agent install` |

For the full backend matrix, model options, Docker setup, and troubleshooting, see the main [README](https://github.com/VeraTools/Vera) and [Installation Guide](https://github.com/VeraTools/Vera/blob/master/docs/installation.md).

## What you get

- **65 languages** (61 with tree-sitter AST parsing)
- **Hybrid search**: BM25 keyword + vector similarity, fused with Reciprocal Rank Fusion
- **Opt-in cross-encoder reranking** for precision, disabled by default
- **Git-aware scopes and index debugging**: `--changed` / `--since` / `--base`, `explain-path`, and index health in `vera stats`
- **Markdown codeblock output** by default with file paths, line ranges, and optional symbol info (use `--json` for compact JSON; `--raw` works with `vera search`, `vera grep`, and `vera references`; `--timing` works with `vera search` and `vera grep`, before or after the subcommand)

For full documentation, including local model options and manual install steps, see the [GitHub repo](https://github.com/VeraTools/Vera).
