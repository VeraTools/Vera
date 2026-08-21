# Troubleshooting

## Building from source fails on macOS with "symbol(s) not found for architecture arm64"

On macOS 14+, the Xcode SDK compiles C dependencies (tree-sitter grammars, ring) for the running OS version, but Cargo's default minimum for `aarch64-apple-darwin` is macOS 11.0. The 11.0 ABI lacks newer symbols, so linking fails.

Set the deployment target to your running macOS version before building:

```bash
export MACOSX_DEPLOYMENT_TARGET="$(sw_vers -productVersion)"
cargo build --release
```

## "No index found in current directory"

Either the repository hasn't been indexed yet, or you're running the command from the wrong directory.

```bash
vera index .
```

Make sure you're in the repository root (the directory containing `.vera/`).

## Results feel stale or outdated

Code changed after the last index. Update it:

```bash
vera update .
```

## Local ONNX inference isn't working

Run the diagnostic command first:

```bash
vera doctor
vera doctor --probe
vera doctor --probe --json
vera upgrade
```

Common causes:

- Models haven't been downloaded yet. Run `vera setup`, `vera setup --potion-code`, or the matching `--onnx-jina-*` setup command
- If assets are missing, corrupt, or truncated in the ONNX model cache, run `vera repair --<backend>` (such as `vera repair --onnx-jina-cuda` or `vera repair --potion-code`). `vera doctor` and embedding load errors detect damaged model files and print the matching repair command hint
- ONNX Runtime auto-download failed. Check network, or set `ORT_DYLIB_PATH` to a manually installed library. Potion Code does not use ONNX Runtime
- If your network only allows browser downloads, use [manual-install.md](manual-install.md)
- GPU backend not working. Make sure the required drivers are installed (CUDA 12+ for `--onnx-jina-cuda`, ROCm for `--onnx-jina-rocm`, DirectX 12 for `--onnx-jina-directml`). CoreML (`--onnx-jina-coreml`) requires macOS on Apple Silicon. OpenVINO (`--onnx-jina-openvino`) and ROCm (`--onnx-jina-rocm`) are installed automatically via pip; if the automatic install fails, install manually (`pip install onnxruntime-openvino` or `pip install onnxruntime-rocm`), then set `ORT_DYLIB_PATH` to the `libonnxruntime.so` inside the package. If GPU init still fails, use `--potion-code` for CPU-only local inference or fix the provider-specific dependencies.
- CoreML reranker runs on CPU on Apple Silicon. CoreML accelerates the embedding model but **not** the reranker: no prebuilt reranker ONNX export is CoreML-compatible (the quantized Jina model has ops the CoreML EP cannot execute, and the fp16 export uses a float16 input dtype the CoreML EP rejects). Vera ships the quantized reranker for CoreML, which is the fastest CPU path, and selects the CPU execution provider for the reranker session explicitly. Do not assume ONNX Runtime falls back on its own: with the CoreML EP registered it still assigns a fused subgraph to CoreML, which builds successfully and then fails at inference with "Unable to compute the prediction using a neural network model (error code: -1)". `vera doctor --probe` flags this with a `probe-reranker-coreml-cpu` warning. If you set up CoreML on an older Vera version that downloaded `onnx/model_fp16.onnx`, run `vera repair --onnx-jina-coreml` to swap it for `onnx/model_quantized.onnx`. If reranking is too slow, disable it with `vera config set retrieval.reranking_enabled false`.
- On CUDA backends, Vera now uses the detected toolkit/runtime libraries (`CUDA_PATH`, CUDA's `version.json` or `version.txt`, `nvcc --version`, and on Linux also `LD_LIBRARY_PATH` or `ldconfig`) to choose the CUDA 12 vs CUDA 13 ONNX Runtime build instead of the driver's maximum supported version. If you switch CUDA toolkits, rerun `vera repair --onnx-jina-cuda` to refresh the downloaded runtime.
- On Windows DirectML, the ONNX Runtime is downloaded from NuGet (not GitHub releases). If the download fails, check your network or set `ORT_DYLIB_PATH` to a manually obtained `onnxruntime.dll` from the [Microsoft.ML.OnnxRuntime.DirectML NuGet package](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML).
- `vera doctor` flags missing, corrupt, or truncated model caches or runtime, shows the saved and active backend, prints the installed Vera version, suggests `vera repair --<backend>`, and checks for newer releases. `vera doctor --probe` adds a deeper read-only local backend probe and does not download or repair missing assets. Both exit 1 when a check fails, so `vera doctor && vera index .` stops on a broken setup; warnings such as a missing config file or an unindexed working directory leave the exit code at 0. `vera repair` is the write path if you need Vera to re-fetch local assets. `vera upgrade` shows the binary update plan and can apply it when the install method is known.
- If a non-CPU ONNX session fails after dependency checks pass, Vera retries runtime embedding and reranker setup on CPU and logs a warning. Fix the GPU provider issue or switch to `--potion-code` if CPU fallback is too slow.

If API mode hits an `exceed_context_size_error` during indexing, update to the latest Vera build. Current releases split and shrink pathological embedding inputs instead of aborting the whole batch on one oversized chunk.

If Vera now fails fast with a message like `CUDA backend selected, but required libraries are missing`, the ONNX Runtime CUDA provider was downloaded but your system linker cannot find the CUDA or cuDNN shared libraries it depends on. Install the required userspace libraries, refresh the linker cache if needed, then rerun `vera doctor --probe`.

## API mode isn't working

Re-run setup and enter the endpoint URL, model ID, and API key when prompted:

```bash
vera setup --api
```

For non-interactive setup, check that all three environment variables are set before running `vera setup --api --yes`:

- `EMBEDDING_MODEL_BASE_URL`
- `EMBEDDING_MODEL_ID`
- `EMBEDDING_MODEL_API_KEY`

If the active embedding model name differs from the one stored in the index, re-index the repo. For verified equivalent model names, configure `embedding.model_aliases` with `vera config set`, or set `VERA_EMBEDDING_MODEL_ALIASES` as semicolon-separated groups of comma-separated aliases, such as `canonical,alias`.

If you're using a reranker in non-interactive setup, its three variables (`RERANKER_MODEL_BASE_URL`, `RERANKER_MODEL_ID`, `RERANKER_MODEL_API_KEY`) must either all be set or all be absent. Partial configuration will fail.

If the provider returns a batch-size error such as `at most 100 requests can be in one batch`, lower the embedding batch size:

```bash
vera config set embedding.batch_size 100
```

Current builds also clamp known provider limits automatically. Gemini embedding endpoints are capped at 100 inputs per request.

If the provider returns `429` or `quota exceeded`, that is a provider-side limit. `embedding.max_concurrent_requests` only reduces how many requests Vera sends in parallel; it does not raise your API quota. Lower concurrency if you are hitting short burst limits, or wait for quota reset / enable billing if the project is out of quota.

## GPU runs out of memory during indexing

Vera auto-detects VRAM and adjusts batch size, but very low-VRAM GPUs (4 GB or less) may still run out of memory. Use the `--low-vram` flag:

```bash
vera index . --low-vram
```

This forces batch size 1 and caps the ONNX Runtime memory arena to 1 GB. You can also manually tune batch size with `vera config set embedding.batch_size 1`.

On newer builds, Vera does not send every local GPU batch to ONNX at the configured `embedding.batch_size`. It tokenizes first, shrinks long-sequence micro-batches, and learns safer limits per sequence-length bucket from real runs. Those learned windows are stored in `adaptive-batch-scaler.json` inside Vera's data directory and reused on later runs for the same backend, device, and model. CoreML uses sequence-length batching without model-size cold-start dampening because Apple Silicon uses unified memory. If a pathological batch still trips an allocation error, Vera retries it at a smaller size instead of aborting the whole index.

If you still see repeated retries or very slow indexing, lower `embedding.batch_size` manually or use `--low-vram`.

## Too many irrelevant results

Try narrowing your search:

- `--lang rust`: filter by language
- `--path "src/**/*.ts"`: filter by file path; repeat it to match any of several patterns
- `--type function`: filter by symbol type
- `--limit 5`: return fewer results
- Rewrite the query to be more specific about the behavior you're looking for

See the [query guide](query-guide.md) for more tips on writing effective queries.

## Need an exact text match?

Use `vera grep` for exact string or regex matching across indexed files:

```bash
vera grep "EMBEDDING_MODEL_BASE_URL"
vera grep "TODO\(" -i
vera grep "queryClient|invalidateQueries" --path "frontend/src/**"
```

Vera uses Rust regex syntax. Use `|` for alternation. `\|` matches a literal pipe.

Use `rg` when you need to count matches, search filenames, or scan files outside the index:

```bash
rg "TODO\(" -n
rg --files | rg "docker"
```

## Search misses new files or recent edits

`vera search` and `vera grep` only read indexed files. If the index is behind the working tree, Vera warns on stderr and shows how many files were added, modified, or deleted since the last refresh.

Refresh the index with:

```bash
vera update .
```

If you are editing continuously, keep the index current with:

```bash
vera watch .
```
