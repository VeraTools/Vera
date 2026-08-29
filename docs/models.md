# Local Models

Vera's default embedding model is [`minishlab/potion-code-16M-v2`](https://huggingface.co/minishlab/potion-code-16M-v2), a static embedding model that runs locally on CPU on any supported machine; no GPU or ONNX Runtime needed. Jina ONNX and CodeRankEmbed are opt-in alternatives.

Vera has two local backend families:

- Potion Code: the default local embedding backend.
- Jina ONNX: an opt-in local embedding backend with a local reranker.

`vera setup` downloads model assets into the Vera data directory (`$XDG_DATA_HOME/vera/models/`, or `~/.vera/models/` on existing installs). Jina ONNX backends also install the matching ONNX Runtime library into `lib/`.

## Curated Embedding Options

| Option | Command | Notes |
| --- | --- | --- |
| Potion Code | `vera setup --potion-code` | Default local embedding model: [`minishlab/potion-code-16M-v2`](https://huggingface.co/minishlab/potion-code-16M-v2). Runs locally on CPU on any supported machine; no GPU or ONNX Runtime needed. |
| Jina v5 nano retrieval | `vera setup --onnx-jina-cuda` or another `--onnx-jina-*` flag | Opt-in GPU local backend. The retrieval variant is asymmetric, so Vera prefixes queries with `Query:` and indexed passages with `Document:`. |
| CodeRankEmbed | `vera setup --onnx-jina-cuda --code-rank-embed` | Optional ONNX embedding preset for code-specific or embedding-only experiments. Current screening results are below. |

When local reranking is enabled, the Jina ONNX family provides this built-in cross-encoder:

| Model | Role |
| --- | --- |
| [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | Local cross-encoder reranker |

With reranking disabled, the default Potion Code path uses vector/BM25 fusion plus Vera's deterministic ranking heuristics.

## Embedding Screening

These screening results (2026-08-21/22, pre-dating the ranking improvements) use Vera's hybrid retrieval with reranking disabled. The first set has 320 Semble tasks; the independent set has 180 tasks from separate repositories.

| Model | 320-task nDCG@10 | Independent nDCG@10 | 320-task p50 | Independent p50 | 320-task index | Independent index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `potion-code-16M-v2` | 0.7930 | 0.7019 | 13.5 ms | 10.1 ms | 56 s | 26 s |
| `jina-embeddings-v5` | 0.7956 | 0.7149 | 44.9 ms | 13.3 ms | 147 s | 61 s |
| `CodeRankEmbed` | 0.7949 | 0.7069 | 21.3 ms | 14.8 ms | 375 s | 153 s |

Potion Code remains the recommended default because it runs locally on all supported machines and has the lowest screening index time among these embedding alternatives. The full benchmark tables and methodology are in [benchmarks.md](benchmarks.md).

## Reranking

Reranking is opt-in through `retrieval.reranking_enabled` and is off by default. The no-reranker heuristic stack is the baseline for normal searches.

In the 2026-08-23 dual-set screening, every tested cross-encoder scored below that baseline. Scores are `320-task subset / independent set` nDCG@10:

| Reranking option | nDCG@10 |
| --- | ---: |
| No reranker | 0.8540 / 0.7644 |
| `jina-reranker-v2-base-multilingual` | 0.8473 / 0.7557 |
| `mxbai-rerank-xsmall-v1` | 0.8497 / 0.7564 |
| `gte-reranker-modernbert-base` | 0.8472 / 0.7525 |

`mxbai-rerank-xsmall-v1` is the recommended opt-in reranker when you need a cross-encoder. Configure the local model with:

```bash
vera config set retrieval.reranking_enabled true

export LOCAL_RERANKER_REPO=mixedbread-ai/mxbai-rerank-xsmall-v1
export LOCAL_RERANKER_REVISION=b5c6e9da73abc3711f593f705371cdbe9e0fe422
export LOCAL_RERANKER_ONNX_FILE=onnx/model_quantized.onnx
export LOCAL_RERANKER_TOKENIZER_FILE=tokenizer.json
```

## CodeRankEmbed Comparison

See the [canonical CodeRankEmbed comparison](benchmarks.md#optional-coderankembed-preset) for the 6-task results and context.

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

### Custom Local Reranker

The local cross-encoder can be replaced with an ONNX-compatible model from Hugging Face. The variables above select the recommended `mxbai-rerank-xsmall-v1` export. Leave `LOCAL_RERANKER_REVISION` unset to use the `main` ref and the legacy cache path. A non-empty revision stores assets under `models/<repo>/revisions/<revision>/`, so different model revisions do not overwrite one another.

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

The default Potion Code model runs on all supported machines. Use Jina ONNX with CUDA, ROCm, CoreML, DirectML, or OpenVINO when you want an opt-in alternative. After the first index, `vera update .` only re-embeds changed files, so updates are fast on any backend.

| Backend | Hardware | Time | Notes |
|---------|----------|------|-------|
| CUDA | RTX 4080 | **~8 s** | Recommended for large repos |
| API mode | Remote GPU | ~56 s | Requires API key, no local compute |
| Jina ONNX CPU | Ryzen 5 7600X3D (6c/12t) | ~6 min | Compatibility path. Use Potion Code for CPU-only machines |

## API Mode

Interactive setup prompts for the endpoint URL, model ID, API key, and optional reranker. The wizard offers presets for OpenAI, Jina, Voyage, and Qwen (OpenRouter) with exact prefills; the Qwen preset uses `qwen/qwen3-embedding-8b` + `qwen/qwen3-reranker-8b` via `https://openrouter.ai/api/v1` with a single shared key (paid usage) and the generic rerank protocol (`top_n`/`results`) by default:

```bash
vera setup --api
```

The reranker step also configures the wire protocol (auto, generic, or voyage), endpoint path override, and optional task instruction. Qwen via OpenRouter relies on the generic protocol unless overridden. Custom proxies can select `generic` or `voyage` via `retrieval.reranker_protocol` without hostname spoofing.

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

# Reranker protocol and endpoint overrides (also configurable without a TTY)
vera config set retrieval.reranker_protocol generic
vera config set retrieval.reranker_endpoint_path "/rerank"
vera config set retrieval.reranker_task_instruction "Given a query, retrieve relevant code"
```

For Qwen via OpenRouter non-interactively:

```bash
export EMBEDDING_MODEL_BASE_URL=https://openrouter.ai/api/v1
export EMBEDDING_MODEL_ID=qwen/qwen3-embedding-8b
export EMBEDDING_MODEL_API_KEY=your-openrouter-key
export RERANKER_MODEL_BASE_URL=https://openrouter.ai/api/v1
export RERANKER_MODEL_ID=qwen/qwen3-reranker-8b
export RERANKER_MODEL_API_KEY=your-openrouter-key
vera setup --api --yes
vera config set retrieval.reranker_protocol generic
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

- Custom ONNX options only affect opt-in Jina ONNX local embeddings. API mode and the default Potion Code model are unchanged.
- Query and document prefixes apply to both API mode and local ONNX embeddings. API mode reads `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX`, with model-ID auto-detection when they are unset. Local ONNX uses the corresponding `--embedding-query-prefix` and `--embedding-document-prefix` setup flags.
- The stored embedding model identity covers every `--embedding-*` setting, not just the model name, so changing pooling or either prefix also requires a re-index.
- If you switch local embedding models without configured aliases, re-index the repo so the stored vectors match the active model.
- If your network blocks CLI downloads, use [manual-install.md](manual-install.md).
