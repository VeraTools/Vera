# vera-ai

Code search for AI agents. Vera indexes your codebase using tree-sitter parsing and hybrid search (BM25 + vector similarity + optional cross-encoder reranking), then returns ranked code snippets as Markdown codeblocks by default, or JSON with `--json`.

This package downloads and wraps the native Vera binary for your platform. On musl-based Linux (Alpine, NixOS), the correct static binary is selected automatically. Set `VERA_TARGET` to override target detection (e.g., `VERA_TARGET=x86_64-unknown-linux-musl uvx vera-ai install`).

The default local embedding model is `minishlab/potion-code-16M-v2`; it runs locally on CPU on any supported machine, no GPU or ONNX Runtime needed. In the current Semble comparison, Vera v1.4.0 scored `0.8437` nDCG@10 versus Semble 0.5.5 at `0.8514` on Semble's own tuning corpus, and leads on the independent contamination set (`0.7674` vs `0.7655`) and on recall@5; Vera's index is 6.8x smaller (4.7 GB vs 32 GB). For the highest measured search quality, use the Qwen preset through OpenRouter. Full details live in the main repo docs.

## Install

```bash
pip install vera-ai
```

## Quick Start

```bash
vera-ai setup --potion-code --index .
vera-ai search "authentication logic"
```

`vera-ai setup` with no flags runs an interactive wizard and offers to index the current project, defaulting to yes. An interactive search also offers to create a missing index. `vera-ai setup --api` prompts for an OpenAI-compatible endpoint and key; the wizard offers presets for OpenAI, Jina, Voyage, and Qwen via OpenRouter, with the Qwen preset needing only one shared key (`qwen/qwen3-embedding-8b` + `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1`). Use `--yes` with `EMBEDDING_MODEL_*` variables for non-interactive setup. `vera-ai agent install` manages skill files for your coding agents and can update `AGENTS.md` / `CLAUDE.md` style project instructions.

## Common Tasks

| Task | Command |
|------|---------|
| Use the interactive setup wizard | `vera-ai setup` |
| Use the default local model | `vera-ai setup --potion-code` |
| Configure API mode | `vera-ai setup --api` |
| Use a local NVIDIA backend | `vera-ai setup --onnx-jina-cuda` |
| Search semantically | `vera-ai search "authentication middleware"` |
| Search only changed files | `vera-ai search "authentication middleware" --changed` |
| Common structural tasks | `vera-ai structural routes` / `vera-ai structural env DATABASE_URL` / `vera-ai structural impls Loader` |
| Find callers or callees | `vera-ai references foo` / `vera-ai references foo --callees` |
| Explain why a file is missing | `vera-ai explain-path path/to/file` |
| Inspect index health | `vera-ai stats --json` |
| Keep the index up to date | `vera-ai update .` |
| Watch for file changes | `vera-ai watch .` |
| Run local HTTP inference server | `vera-ai serve` |
| Diagnose setup issues | `vera-ai doctor` |
| Run the deeper local probe | `vera-ai doctor --probe` |
| Repair missing local assets | `vera-ai repair` |
| Inspect binary upgrades | `vera-ai upgrade` |
| Install agent skills | `vera-ai agent install` |

For the full backend matrix, model options, Docker setup, and troubleshooting, see the main [README](https://github.com/VeraTools/Vera) and [Installation Guide](https://github.com/VeraTools/Vera/blob/master/docs/installation.md).

## What you get

- **65 languages** (61 with tree-sitter AST parsing)
- **Hybrid search**: BM25 keyword + vector similarity, fused with Reciprocal Rank Fusion
- **Opt-in cross-encoder reranking** for precision, disabled by default
- **Git-aware scopes and index debugging**: `--changed` / `--since` / `--base`, `explain-path`, and index health in `vera-ai stats`
- **Markdown codeblock output** by default with file paths, line ranges, and optional symbol info (use `--json` for compact JSON; `--raw` works with `vera-ai search`, `vera-ai grep`, and `vera-ai references`; `--timing` works with `vera-ai search` and `vera-ai grep`, before or after the subcommand)

For full documentation, including local model options and manual install steps, see the [GitHub repo](https://github.com/VeraTools/Vera).
