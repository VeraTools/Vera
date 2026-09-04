# Installation Guide

For the short path, use the [Quick Start](../README.md#quick-start) in the README.

## Install the Binary

Pick whichever package manager you have:

```bash
bunx @vera-ai/cli install               # Bun
npx -y @vera-ai/cli install             # npm
uvx vera-ai install                     # Python (uv)
pip install vera-ai && vera-ai install  # Python (pip)
```

The installer downloads the `vera` binary for your platform, writes a shim to a user bin directory, and delegates to `vera agent install`, which launches an interactive scope and client selector to install skill files. After that, `vera` is a standalone command.

<details>
<summary>Other install methods</summary>

**Prebuilt binaries:**
Download from [GitHub Releases](https://github.com/VeraTools/Vera/releases) for Linux (x86_64, aarch64), macOS (x86_64, aarch64), or Windows (x86_64). For Alpine, NixOS, or minimal containers without glibc, use the `x86_64-unknown-linux-musl` archive (fully static, zero runtime dependencies). The npm/pip wrappers auto-detect musl systems; to force a specific target, set `VERA_TARGET=x86_64-unknown-linux-musl` before running the install command.

**Build from source** (Rust 1.88+):
```bash
git clone https://github.com/VeraTools/Vera.git && cd Vera
bash scripts/bootstrap-vendored-grammars.sh   # downloads the four grammars that are not tracked in git
cargo build --release
cp target/release/vera ~/.local/bin/
vera setup
```

**Docker** (MCP server):
```bash
docker run --rm -i -v $(pwd):/workspace ghcr.io/veratools/vera:cpu
```
CPU, CUDA, ROCm, and OpenVINO images available. See [docker.md](docker.md).

**Manual install:** [manual-install.md](manual-install.md)

</details>

## Set Up a Backend

Vera's index and search always run locally. The "backend" only controls where embedding and reranking models run.

The default embedding model is `minishlab/potion-code-16M-v2`, a static embedding model that runs locally on CPU on any supported machine; no GPU or ONNX Runtime needed. Jina ONNX and CodeRankEmbed are opt-in alternatives.

### API Mode

Models run on a remote server. No downloads, no GPU required, works on any hardware. You just need an API key from any OpenAI-compatible provider.

```bash
vera setup --api
```

Vera will prompt you for your endpoint URL, model ID, and API key. These get saved to Vera's config so you only enter them once. API mode is an alternative to the default local model.

Many providers offer free tiers or generous trial credits. Any OpenAI-compatible embedding endpoint works. Some options:

| Provider | Free tier? | Notes |
|----------|-----------|-------|
| [Jina AI](https://jina.ai/) | Yes (1M tokens free) | Remote embedding and reranking endpoints |
| [OpenAI](https://platform.openai.com/) | Trial credits | `text-embedding-3-small` or `text-embedding-3-large` |
| [Voyage AI](https://www.voyageai.com/) | Free tier available | Code-optimized models (`voyage-code-3`, `rerank-2`) |
| [Qwen via OpenRouter](https://openrouter.ai/) | Paid usage | `qwen/qwen3-embedding-8b` + `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1` (preset in `vera setup`) |
| [Cohere](https://cohere.com/) | Trial key | `embed-english-v3.0` |

For non-interactive setup, set the environment variables directly and add `--yes`:

```bash
export EMBEDDING_MODEL_BASE_URL=https://api.jina.ai/v1
export EMBEDDING_MODEL_ID=jina-embeddings-v3
export EMBEDDING_MODEL_API_KEY=your-key

# Optional: reranker for better precision (Jina or Voyage AI)
export RERANKER_MODEL_BASE_URL=https://api.jina.ai/v1
export RERANKER_MODEL_ID=jina-reranker-v2-base-multilingual
export RERANKER_MODEL_API_KEY=your-key

# Or for Voyage AI:
# export RERANKER_MODEL_BASE_URL=https://api.voyageai.com/v1
# export RERANKER_MODEL_ID=rerank-2
# export RERANKER_MODEL_API_KEY=your-key

# Or for Qwen via OpenRouter (paid usage, generic protocol):
# export EMBEDDING_MODEL_BASE_URL=https://openrouter.ai/api/v1
# export EMBEDDING_MODEL_ID=qwen/qwen3-embedding-8b
# export EMBEDDING_MODEL_API_KEY=your-openrouter-key
# export RERANKER_MODEL_BASE_URL=https://openrouter.ai/api/v1
# export RERANKER_MODEL_ID=qwen/qwen3-reranker-8b
# export RERANKER_MODEL_API_KEY=your-openrouter-key

vera setup --api --yes
# Optional reranker protocol overrides (without a TTY)
# vera config set retrieval.reranker_protocol generic
# vera config set retrieval.reranker_endpoint_path "/rerank"
```

Vera automatically handles Voyage AI's rerank wire format when `RERANKER_MODEL_BASE_URL` points to `https://api.voyageai.com/v1`. The Qwen preset uses the generic wire format (`top_n`/`results`) via `https://openrouter.ai/api/v1`.

Only model calls leave your machine. Indexing, storage, and search remain local.

### CPU Local Mode

The default `minishlab/potion-code-16M-v2` model runs locally on CPU on any supported machine; no GPU or ONNX Runtime needed.

```bash
vera setup --potion-code
```

Use this when you want the default local model. It also runs on CPU-only machines, and the interactive `vera setup` wizard selects it as the default local backend.

### GPU Local Mode

Jina ONNX is an opt-in local backend. Vera downloads the Jina embedding model and local reranker, then uses your GPU provider. No API key is needed, and the setup works offline after the download.

**Pick the right command for your hardware:**

| You have | Command | What happens |
|----------|---------|-------------|
| Not sure | `vera setup` | Interactive wizard auto-detects your hardware |
| CPU only | `vera setup --potion-code` | Uses the default `minishlab/potion-code-16M-v2` model |
| Apple Silicon (M1/M2/M3/M4) | `vera setup --onnx-jina-coreml` | Uses CoreML GPU acceleration |
| NVIDIA GPU | `vera setup --onnx-jina-cuda` | Uses CUDA. Fastest local option |
| AMD GPU (Linux) | `vera setup --onnx-jina-rocm` | Uses ROCm |
| Intel GPU (Linux) | `vera setup --onnx-jina-openvino` | Uses OpenVINO |
| DirectX 12 GPU (Windows) | `vera setup --onnx-jina-directml` | Uses DirectML |

For custom ONNX models, GPU-specific tuning, and inference speed comparisons, see [models.md](models.md).

## Verify Your Setup

```bash
vera doctor          # checks config, models, and connectivity
vera doctor --probe  # deeper local backend diagnostics
```

## Index and Search

Add `--index .` to an explicit setup command to configure the backend and index the current project in one step:

```bash
vera setup --potion-code --index .
vera search "authentication logic"
```

The interactive `vera setup` wizard also offers to index the current project and defaults to yes. If you skip indexing, an interactive `vera search` offers to create the missing index. JSON and non-interactive searches return the existing missing-index error instead of prompting.

See the [query guide](query-guide.md) for tips on writing effective queries.

## Set Up Agent Skills

Vera can install skill files so your AI coding agents know how to use it:

```bash
vera agent install              # interactive: choose scope + agents
vera agent install --client all # non-interactive: all agents, global
```

This is optional but recommended if you use AI coding agents. The interactive flow can also update your project's `AGENTS.md`, `CLAUDE.md`, `COPILOT.md`, `.cursorrules`, `.clinerules`, or `.windsurfrules` file with a short Vera usage snippet.

<details>
<summary>Add the instructions manually</summary>

```markdown
## Code Search

<!-- vera:begin -->

Use Vera before opening many files or running broad text search when you need to find where logic lives or how a feature works.

- `vera search "query"` for semantic code search. Describe behavior: "JWT validation", not "auth". If one phrasing misses, try 2-3 varied queries or add `--intent "goal"`.
- `vera search ... --changed`, `--since <rev>`, or `--base <rev>` when the task is limited to modified files or a PR diff
- `vera grep "pattern"` for exact text or regex in indexed files
- `vera structural definitions <symbol>`, `vera structural env <NAME>`, `vera structural routes`, or `vera structural impls <symbol>` for common structural tasks and explicit type relationships
- `vera explain-path path/to/file` to explain why a file is or is not indexed
- `vera references <symbol>` for callers and `vera references <symbol> --callees` for callees
- `vera overview` for a project summary (languages, entry points, hotspots). Add `--changed`, `--since <rev>`, or `--base <rev>` to scope it to modified files.
- `vera stats --json` for index health, including tree-sitter error, parse-failure, and Tier 0 fallback counts
- `vera search --deep "query"` for RAG-fusion query expansion + merged ranking
- Narrow `vera search` or `vera grep` with `--lang`, `--path`, `--type`, or `--scope docs`
- `vera watch .` to auto-update the index, or `vera update .` after edits (`vera index .` if `.vera/` is missing)
- For detailed usage, query patterns, and troubleshooting, read the Vera skill file installed by `vera agent install`
<!-- vera:end -->
```

`vera structural impls <symbol>` only finds explicit declarations such as `implements`, `extends`, `with`, `:`, or `impl Trait for Type`. It does not infer implicit interface satisfaction.

</details>

<details>
<summary>Use the Vercel skills CLI instead</summary>

```bash
npx skills add VeraTools/Vera
```

</details>

## Updating

Vera checks for new releases daily and prints a hint when one is available.

```bash
vera upgrade              # dry run: shows what would happen
vera upgrade --apply      # applies the update
```

After an upgrade, Vera automatically syncs stale agent skill installs. Set `VERA_NO_UPDATE_CHECK=1` to disable the automatic check.

If you are having trouble updating, reinstall with the package manager you originally used:

```bash
# Bun
bun install -g @vera-ai/cli && bunx @vera-ai/cli install
# npm
npm install -g @vera-ai/cli && npx @vera-ai/cli install
# uv
uvx vera-ai install
# pip
pip install --upgrade vera-ai && vera-ai install
```

## Uninstalling

```bash
vera uninstall   # removes the entire Vera data directory including the config dir, skill files, PATH shim, binary caches, and downloaded model weights
```

## Troubleshooting

- Run `vera doctor` to diagnose issues.
- Run `vera doctor --probe` for deeper local backend diagnostics.
- Wrong backend? Run `vera setup` again with a different flag.
- Slow opt-in Jina indexing on CPU? Switch to `--potion-code`, `--api`, or a GPU backend.
- See [troubleshooting.md](troubleshooting.md) for more.
