# Local Models

Vera has two local backend families:

- Potion Code CPU: static embeddings through `model2vec-rs`. No ONNX Runtime.
- Jina ONNX GPU: local embedding model plus the curated local reranker through ONNX Runtime.

`vera setup` downloads model assets into the Vera data directory (`$XDG_DATA_HOME/vera/models/`, or `~/.vera/models/` on existing installs). Jina ONNX backends also install the matching ONNX Runtime library into `lib/`.

## Curated Embedding Options

| Option | Command | Notes |
| --- | --- | --- |
| Potion Code | `vera setup --potion-code` | CPU-first local backend. Uses [`minishlab/potion-code-16M`](https://huggingface.co/minishlab/potion-code-16M), returns 256-dim embeddings, and skips ONNX Runtime. |
| Jina v5 nano retrieval | `vera setup --onnx-jina-cuda` or another `--onnx-jina-*` flag | GPU local backend. Faster indexing than larger ONNX embedding models and strongest end-to-end benchmark coverage in Vera so far. The retrieval variant is asymmetric, so Vera prefixes queries with `Query:` and indexed passages with `Document:`. |
| CodeRankEmbed | `vera setup --onnx-jina-cuda --code-rank-embed` | Optional ONNX embedding preset. Useful when you want a code-specific bi-encoder or you are testing without reranking. On Vera's short 6-task no-rerank check it beat the Jina preset on retrieval quality, but indexing was much slower. |

The Jina ONNX family uses this local reranker:

| Model | Role |
| --- | --- |
| [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | Local cross-encoder reranker |

Potion Code currently uses vector/BM25 fusion plus Vera's deterministic ranking heuristics instead of the ONNX reranker.

## CodeRankEmbed Comparison

Short no-rerank check on 6 tasks across `flask` and `ripgrep` with CUDA ONNX:

| Model | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG | Search p50 | Flask index | Ripgrep index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jina preset | 0.5556 | 0.5556 | 0.5556 | 0.8462 | 0.6442 | 761.9 ms | 5.8 s | 11.9 s |
| CodeRankEmbed preset | 0.7222 | 0.7222 | 0.7222 | 1.0000 | 0.8108 | 611.4 ms | 14.7 s | 29.1 s |

This is why CodeRankEmbed ships as an option, not the default. It showed better pure embedding retrieval on that small no-rerank slice, but Vera's full local benchmark still uses the Jina preset because the full reranked pipeline is already very strong and Jina indexes faster.

See [benchmarks.md](benchmarks.md) for the full benchmark context.

## Custom Local Embedding Models

You can point Vera at a different ONNX embedding model without changing the local reranker. These flags apply to the Jina ONNX backend family, not Potion Code.

### Hugging Face Repo Or URL

```bash
vera setup --onnx-jina-cuda \
  --embedding-repo Zenabius/CodeRankEmbed-onnx \
  --embedding-pooling cls \
  --embedding-no-onnx-data \
  --embedding-query-prefix "Represent this query for searching relevant code:"
```

`--embedding-repo` also accepts a full Hugging Face URL such as `https://huggingface.co/Zenabius/CodeRankEmbed-onnx`.

### Local Directory

```bash
vera setup --onnx-jina-cuda \
  --embedding-dir /path/to/model-dir \
  --embedding-onnx-file onnx/model_quantized.onnx \
  --embedding-tokenizer-file tokenizer.json \
  --embedding-dim 768
```

Use this when you already downloaded or exported the model yourself.

## Flags

| Flag | Meaning |
| --- | --- |
| `--code-rank-embed` | Select the built-in CodeRankEmbed preset |
| `--embedding-repo <repo-or-url>` | Download a custom embedding model from Hugging Face |
| `--embedding-dir <dir>` | Use a local directory instead of downloading from Hugging Face |
| `--embedding-onnx-file <path>` | Relative path to the ONNX file inside the repo or directory |
| `--embedding-onnx-data-file <path>` | Relative path to an ONNX external data file |
| `--embedding-no-onnx-data` | Use models that do not ship an external data file |
| `--embedding-tokenizer-file <path>` | Relative path to the tokenizer file |
| `--embedding-dim <n>` | Embedding dimension stored in the index |
| `--embedding-pooling <mode>` | Pooling method for token-level outputs: `mean`, `cls`, or `last-token` |
| `--embedding-max-length <n>` | Tokenizer truncation length |
| `--embedding-query-prefix <text>` | Optional prefix prepended to local embedding queries |
| `--embedding-document-prefix <text>` | Optional prefix prepended to indexed passages |

## Required Files

For a custom ONNX embedding model, Vera needs:

- an ONNX model file
- a tokenizer file
- optionally an ONNX external data file

The defaults are:

| Asset | Default path |
| --- | --- |
| ONNX model | `onnx/model_quantized.onnx` |
| ONNX external data | `onnx/model_quantized.onnx_data` |
| Tokenizer | `tokenizer.json` |

If your model uses different names, pass the matching `--embedding-*` flags.

## Inference Speed

Potion Code is the CPU local option. Use Jina ONNX with CUDA, ROCm, CoreML, DirectML, or OpenVINO when you have a supported GPU. After the first index, `vera update .` only re-embeds changed files, so updates are fast on any backend.

| Backend | Hardware | Time | Notes |
|---------|----------|------|-------|
| CUDA | RTX 4080 | **~8 s** | Recommended for large repos |
| API mode | Remote GPU | ~56 s | Requires API key, no local compute |
| Jina ONNX CPU | Ryzen 5 7600X3D (6c/12t) | ~6 min | Compatibility path. Use Potion Code for CPU-only machines |

## API Mode

Interactive setup prompts for the endpoint URL, model ID, API key, and optional reranker:

```bash
vera setup --api
```

For non-interactive setup, export the API values first and add `--yes`:

```bash
export EMBEDDING_MODEL_BASE_URL=https://your-embedding-api/v1
export EMBEDDING_MODEL_ID=your-embedding-model
export EMBEDDING_MODEL_API_KEY=your-api-key

# Optional reranker
export RERANKER_MODEL_BASE_URL=https://your-reranker-api/v1
export RERANKER_MODEL_ID=your-reranker-model
export RERANKER_MODEL_API_KEY=your-api-key

vera setup --api --yes
```

Only model calls leave your machine. Indexing, storage, and search remain local.

## Model Aliases

When switching between endpoints or local model names that provide identical embeddings, configure aliases so Vera does not require a full re-index.

Set `embedding.model_aliases` in config or export `VERA_EMBEDDING_MODEL_ALIASES`:

```bash
export VERA_EMBEDDING_MODEL_ALIASES="jina-embeddings-v3,jina-v3;text-embedding-3-small,openai-small"
```

The syntax uses semicolon-separated groups of comma-separated equivalent model names. Each group must contain at least two names.

## Apple Silicon Memory and Batching

On macOS Apple Silicon, CoreML auto-detects unified memory by reading `sysctl hw.memsize`. Vera treats half of system RAM as the available GPU pool for auto-scaling and caps the CoreML auto batch size at 64 to keep macOS and other applications responsive.

## Notes

- Custom ONNX options only affect Jina ONNX local embeddings. API mode and Potion Code are unchanged.
- Query prefixes only apply to ONNX local embedding queries, not API embeddings. Document prefixes apply to indexed passages, and only on the local ONNX path.
- The stored embedding model identity covers every `--embedding-*` setting, not just the model name, so changing pooling or either prefix also requires a re-index.
- If you switch local embedding models without configured aliases, re-index the repo so the stored vectors match the active model.
- If your network blocks CLI downloads, use [manual-install.md](manual-install.md).
