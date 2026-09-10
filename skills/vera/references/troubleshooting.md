# Troubleshooting

## `no index found ... or any parent directory`

Cause:

- the repository has not been indexed yet
- the command is running outside any indexed directory

Fix:

```sh
vera index .
```

Run it from the repository root. Search, grep, structural, references, overview, stats, and update commands resolve the nearest `.vera/` up the directory tree, so they work from subdirectories too.

## Results Are Stale

Cause:

- code changed after the last index

Fix:

```sh
vera update .
```

Works from any subdirectory; it refreshes the repository root's index.

## Local ONNX Inference Fails

Check:

```sh
vera doctor
```

Common causes:

- ONNX Runtime auto-download failed (check network, or set `ORT_DYLIB_PATH`)
- local model assets have not been downloaded yet
- GPU backend missing drivers (CUDA 12+ for `--onnx-jina-cuda`, ROCm for `--onnx-jina-rocm`, DirectX 12 for `--onnx-jina-directml`, macOS Apple Silicon for `--onnx-jina-coreml`)
- non-CPU ONNX session loading failed after dependency checks passed; Vera retries runtime embedding and reranker setup on CPU and logs a warning

Helpful commands:

```sh
vera setup                        # re-download models + ORT (CPU)
vera setup --onnx-jina-cuda       # re-download with CUDA ORT
vera doctor                       # basic health check
vera doctor --probe               # deeper ONNX session check
vera doctor --json                # machine-readable diagnostics
vera repair                       # re-fetch missing local assets without full setup
vera backend                      # switch GPU/model backend without re-running setup
```

On constrained GPUs, pass `--low-vram` to `vera index` to force conservative batch settings.

## API Mode Fails

Re-run setup and enter the endpoint URL, model ID, and API key:

```sh
vera setup --api
```

For non-interactive setup, check these variables before running `vera setup --api --yes`:

- `EMBEDDING_MODEL_BASE_URL`
- `EMBEDDING_MODEL_ID`
- `EMBEDDING_MODEL_API_KEY`

Optional reranker values must either all be present or all be absent in non-interactive setup.

## Too Much Noise

Try one of these:

- add `--lang`
- add `--path`
- add `--type`
- reduce `--limit`
- rewrite the query to describe behavior, not just a vague topic

## Exact Match Requested

Use `vera grep` for exact text or regex inside indexed files:

```sh
vera grep "EMBEDDING_MODEL_BASE_URL"
vera grep "TODO\(" -i
vera grep "queryClient|invalidateQueries" --path "frontend/src/**"
```

Use `rg` for file names, counting matches, or files outside the Vera index.

## Debugging Exclusion Rules

If unexpected files are indexed or missing from results:

```sh
vera explain-path path/to/file
```

Use `vera stats --json` if you need to inspect parse failures, tree-sitter error nodes, or Tier 0 fallback counts across the whole index.

Check `.veraignore` syntax (gitignore format). Remember: `.veraignore` replaces `.gitignore` rules entirely unless you add `#include .gitignore` at the top to merge both. When using `#include .gitignore`, only add patterns that aren't already in `.gitignore`.
