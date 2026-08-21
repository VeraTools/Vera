# Vera Benchmarks

This page tracks the v1.0 full-pipeline results and ablations first, then older snapshots kept for historical comparison.

## v1.0 Full-Pipeline Benchmark

The v1.0.0-rc run used the full 1,251-task Semble suite across 63 repos on the `vera-cuda` lane: hybrid BM25+vector retrieval, RRF fusion, and a local ONNX cross-encoder reranker on CUDA, measured 2026-08-16 against the v1.0.0 release candidate. The older "BM25 scoped filters" full-suite row (`nDCG@10` 0.7267, measured in May 2026) was a BM25-only lane using an older harness. Cross-date comparisons are approximate.

### Model Lanes And Partial Runs

`vera-eval` keeps the historical lanes `vera-bm25`, `vera-cuda`, `vera-cpu`, and `vera-potion`. Use `--task-id` and `--category` to select a task slice without changing the task files or ground truth:

```bash
cargo run -p vera-eval -- run --tool vera-bm25 \
  --task-id symbol-lookup-001,symbol-lookup-002 \
  --category symbol_lookup --json-only
```

For candidate models, pass a JSON or TOML file with one or more lanes. Each lane records its backend, model source, pooling, prefixes, dimensions, maximum length, reranking choice, and optional revision. The evaluator emits one report per lane, or a JSON array when more than one lane is requested.

JSON example:

```json
{
  "lanes": [
    {
      "name": "bitnet-270m",
      "backend": "custom-onnx",
      "repo": "microsoft/bitnet-embedding-270m",
      "onnx_file": "onnx/model.onnx",
      "tokenizer_file": "tokenizer.json",
      "pooling": "last-token",
      "query_prefix": "Represent this query for searching relevant code:",
      "document_prefix": "Represent this code for retrieval:",
      "dim": 640,
      "max_length": 512,
      "rerank": false,
      "revision": "<commit-or-tag>"
    },
    {
      "name": "potion-control",
      "backend": "potion",
      "rerank": false
    }
  ]
}
```

The `custom-onnx` backend defaults to CPU; set `execution_provider` to `cuda`, `rocm`, `coreml`, `openvino`, or another supported provider when needed. `repo` downloads from Hugging Face and `dir` points at an existing local model directory. API lanes use `model_id`, `query_prefix`, and `environment` for endpoint settings. Every report includes the resolved lane contract, Vera Git SHA, timestamp, command and redacted environment summary, corpus SHAs, and a SHA-256 identity of the selected task IDs.

The lower-level Python runner supports the same task controls for its existing retrieval modes:

```bash
python3 benchmarks/scripts/run_vera_benchmarks.py \
  --modes bm25-only hybrid-norerank \
  --task-id intent-001,intent-002 --category intent --skip-index
```

### Main Results

| Variant | nDCG@10 | Recall@1 | Recall@5 | Mean search latency |
|---------|---------|----------|----------|---------------------|
| Baseline (pre-ablation HEAD) | 0.7209 | 0.5336 | 0.8088 | 310 ms |
| v1.0.0-rc (C1+C2 merged) | 0.7327 | 0.5476 | 0.8144 | 2421 ms |

### Ablations

All ablations were measured on the full 1,251-task suite against the stated base.

| Ablation | What it changes | nDCG delta | R@1 delta | Latency effect | Decision |
|----------|-----------------|------------|-----------|----------------|----------|
| C1: rerank no-surplus skip | Skip reranking when the fused pool has no surplus over the requested limit | -0.0002 | +0.0000 | -2.4 ms mean | Shipped |
| C2: rerank path-glob searches | Always rerank path-scoped searches (removes the old skip heuristic) | +0.0113 | +0.0132 | scoped queries now pay rerank cost | Shipped |
| GA: structural graph augmentation | Append bounded caller/implementation chunks into the rerank pool (`VERA_GRAPH_AUGMENT=1`) | +0.0047 | +0.0080 | +2018 ms mean (reranks the expanded pool) | Merged off by default: gain too small for the latency, R@5 did not move |

Artifacts:

- [2026-08-16-vera-cuda-v1-full.json](../benchmarks/results/semble/2026-08-16-vera-cuda-v1-full.json) (shipped configuration, full suite)

The baseline, C1, C2, and GA variant JSONs were pruned from the tree; they remain in git history.

### Agent-Level Benchmark

This benchmark used 10 cross-file tracing questions about the Flask codebase. The question set and harness are documented in [benchmarks/agent-bench/README.md](../benchmarks/agent-bench/README.md). Each question was answered by a fresh `droid exec` agent in two arms: `with-vera`, with a Vera index and agent skill installed; and `control`, with the Vera CLI blocked and exiting 127. Answers were graded blind against a verified answer key by a judge model using a 0-10 rubric per question. Efficiency metrics came from the agent harness stream: tool calls, input tokens, and wall time.

| Tested model | Arm | Mean score | Tool calls | Input tokens | Wall time |
|--------------|-----|------------|------------|--------------|-----------|
| claude-opus-5 (medium effort) | with-vera | 10.0/10 | 186 | 298 | 1367 s |
| claude-opus-5 (medium effort) | control | 10.0/10 | 173 | 212 | 1252 s |
| kimi-k3 (medium effort) | with-vera | 10.0/10 | 219 | 230,567 | 1312 s |
| kimi-k3 (medium effort) | control | 9.9/10 | 190 | 278,041 | 1198 s |

The opus table's input-token counts include only non-cached tokens because nearly everything there was cache reads. The kimi lane is the honest context-size comparison: with Vera, the agent pulled 17% fewer input tokens (230.6k vs 278.0k) to reach the same answer quality.

On a small, well-organized repo, a frontier model answers these questions perfectly with plain grep+read, so quality parity is expected. Vera's measurable effect at this scale is reduced context consumption for the mid-tier model, at roughly equal wall time with slightly more tool calls. Larger and less familiar codebases are where the retrieval advantage should grow. Treat this as a floor, not a ceiling.

Limitations: 10 questions, 1 repo, 1 run per cell, and no statistical power claims.

## Current Local Release Benchmark

This is the benchmark used to measure the `v0.7.0` retrieval pipeline.

- 21 tasks
- 4 repos: `ripgrep`, `flask`, `fastify`, `turborepo`
- local Jina embedding + reranker stack
- CUDA ONNX backend
- same pinned corpora and the same local-binary harness for every version below

### Accuracy Improvements From `v0.4.0` To `v0.7.0`

| Version | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|--------|----------|----------|-----------|--------|---------|
| `v0.4.0` | 0.2421 | 0.5040 | 0.5159 | 0.5016 | 0.4570 |
| `v0.5.0` | 0.3135 | 0.5635 | 0.6349 | 0.5452 | 0.5293 |
| `v0.7.0` | **0.7183** | **0.7778** | **0.8254** | **0.9095** | **0.8361** |

From `v0.4.0` to `v0.7.0`, Vera improved by:

- `+0.4762` Recall@1
- `+0.2738` Recall@5
- `+0.3095` Recall@10
- `+0.4079` MRR@10
- `+0.3791` nDCG@10

The raw per-version JSONs were pruned from the tree; they remain in git history. See [`v0.7.0` accuracy improvements](./releases/v0.7.0-accuracy-improvements.md) for the per-version breakdown.

### Current Performance Snapshot

`v0.7.0` local Jina CUDA ONNX results:

| Measure | Result |
|---------|--------|
| Search latency p50 | `3716 ms` |
| Search latency p95 | `4185 ms` |

### Recent Local Tuning Loop

These runs were used for retrieval tuning after `v0.7.0`. They are useful for regression tracking, but they are not the public release snapshot above.

Method:

- `benchmarks/scripts/run_local_binary_benchmarks.py` against the same 21-task, 4-repo corpus
- forced model paths on CUDA so quantized and fp16 runs did not silently switch models
- judged by the full suite, not Vera usage rate or one benchmark hole

Artifacts:

- pre-fix quantized: [c7bdc09-jina-cuda-onnx-quantized-embed](../benchmarks/results/local-binaries/c7bdc09-jina-cuda-onnx-quantized-embed.json)
- pre-fix fp16: [c7bdc09-jina-cuda-onnx-fp16-embed](../benchmarks/results/local-binaries/c7bdc09-jina-cuda-onnx-fp16-embed.json)
- current fp16 candidate-pool fix: [candidate-pool-fix-rerank50-jina-cuda-onnx-fp16-embed](../benchmarks/results/local-binaries/candidate-pool-fix-rerank50-jina-cuda-onnx-fp16-embed.json)
- current quantized candidate-pool fix: [oom-fix-jina-cuda-onnx-quantized-embed](../benchmarks/results/local-binaries/oom-fix-jina-cuda-onnx-quantized-embed.json)
- current quantized dynamic scaler: [dynamic-scaler-jina-cuda-onnx-quantized-embed](../benchmarks/results/local-binaries/dynamic-scaler-jina-cuda-onnx-quantized-embed.json)

Current fp16 candidate-pool fix vs pre-fix fp16:

| Metric | Pre-fix fp16 | Current fp16 |
|--------|--------------|--------------|
| Recall@1 | 0.7183 | **0.7659** |
| Recall@5 | 0.8254 | **0.8968** |
| Recall@10 | 0.8254 | **0.8968** |
| MRR@10 | 0.9206 | **0.9683** |
| nDCG@10 | 0.8425 | **0.9027** |

What changed:

- `intent-004` (`file type detection and filtering`) moved from a miss to a perfect hit by returning `crates/ignore/src/types.rs:224-301` instead of a tiny helper method
- `cross-file-002` improved because the deeper pool also kept the second relevant blueprint registration chunk alive long enough to rank
- no task regressed in the fp16 full-suite rerun

Tradeoff:

- search latency went up on the fp16 tuning run (`p50 4103 ms`, `p95 10772 ms`)
- most of the extra cost came from broad intent queries that now search a deeper candidate pool before truncation

Quantized note:

- the full forced-quantized 21-task rerun now completes and matches the fp16 aggregate metrics on this suite
- on this machine, quantized ended up slightly faster on search (`p50 3617 ms` vs `4103 ms`) but slower on indexing
- the original blocker was a large `turborepo` embedding batch that hit a CUDA ONNX allocation spike inside `MultiHeadAttention`; Vera now retries those local batches at smaller sizes instead of aborting the index
- the dynamic sequence-aware scaler keeps the same aggregate metrics as `oom-fix-jina-cuda-onnx-quantized-embed`, then trims quantized indexing time on every benchmark repo in the same 21-task run (`ripgrep 13.08s -> 12.96s`, `fastify 15.24s -> 14.57s`, `turborepo 55.28s -> 54.71s`, `flask 6.37s -> 5.82s`)
- the scaler now also persists learned GPU windows across runs in `~/.vera/adaptive-batch-scaler.json`; when you compare cold indexing throughput, clear that file first or run all candidates against the same warmed state

### Semble Comparison

Vera is benchmarked against [Semble](https://github.com/MinishLab/semble), a Python code search tool using `potion-code-16M` static embeddings.

**320-task subset** (16 repos, used for tuning iteration):

| Tool | Backend | Recall@1 | Recall@10 | MRR | nDCG@10 | Search p50 | Search p95 | Index time |
|------|---------|----------|-----------|-----|---------|------------|------------|------------|
| Semble | Potion Code CPU | **0.6630** | **0.9479** | **0.8223** | **0.8311** | **1.43 ms** | **15.41 ms** | 26.06 s |
| Vera | BM25 scoped filters | 0.5802 | 0.9172 | 0.7618 | 0.7752 | 5.05 ms | 23.33 ms | 10.40 s |
| Vera | BM25 ranked (v4) | 0.5792 | 0.8781 | 0.7520 | 0.7567 | 2.92 ms | 10.37 ms | **10.28 s** |
| Vera | Potion Code CPU (v5) | 0.5792 | 0.8797 | 0.7490 | 0.7550 | 10.11 ms | 41.08 ms | 16.62 s |
| Vera | Potion Code CPU (v4) | 0.5792 | 0.8797 | 0.7490 | 0.7550 | 10.68 ms | 49.14 ms | 16.72 s |
| Vera | BM25 ranked (v3) | 0.5573 | 0.8750 | 0.7376 | 0.7477 | 2.92 ms | 10.37 ms | 10.28 s |
| Vera | Potion Code CPU (v3) | 0.5510 | 0.8891 | 0.7340 | 0.7468 | 13.95 ms | 54.26 ms | 16.40 s |
| Vera | Jina CUDA ONNX | 0.5276 | 0.8578 | 0.7058 | 0.7233 | 23.50 ms | 6236.60 ms | 151.20 s |

v3 = English stemming, concept-to-filename augmentation. v4 = stronger definition boost, content-based definition detection, embedded symbol extraction, proportional stem matching, stronger noise penalties. Potion v4 also benefits from parallelized BM25+embedding (24% p50 improvement). Potion v5 keeps v4 retrieval metrics and lowers vector search overfetch from 2x to 1.5x of the requested candidate pool, which trims Potion CPU p95 latency on the subset.

**Full 1,251-task Semble suite** (63 repos, gate for parity claims):

| Metric | Vera BM25 (v3) | Vera BM25 (v4) | Vera BM25 scoped filters |
|--------|---------------:|---------------:|-------------------------:|
| nDCG@10 | 0.7010 | 0.7074 | **0.7267** |
| Recall@1 | 0.5357 | **0.5449** | 0.5448 |
| Recall@5 | 0.7748 | 0.7832 | **0.8144** |
| Recall@10 | 0.8128 | 0.8160 | **0.8654** |
| MRR | 0.6857 | 0.6943 | **0.7043** |

v4 per-category nDCG: symbol_lookup 0.8944, intent 0.6987 (+0.0087), cross_file 0.6141 (+0.0051). All categories improved. 100 task improvements vs 72 regressions across the full suite.

Scoped-filtered BM25 expands the raw Tantivy pool only when search filters are active, then hydrates and keeps matching chunks before final ranking. On the full suite, zero-hit@10 tasks dropped from 188 to 139. Recall@10 had 82 task improvements and 4 regressions. Search p95 rose from 24.96 ms to 54.94 ms.

The Jina CUDA run uses CUDA ONNX Runtime via `ORT_DYLIB_PATH`. Do not run this lane against the CPU ONNX Runtime when comparing latency.

The May subset and full-suite JSONs were pruned from the tree; they remain in git history.

### Optional CodeRankEmbed Preset

Vera now ships CodeRankEmbed as an optional local embedding preset. This is the short no-rerank sanity check used to decide whether it was worth exposing as a first-class option:

- 6 tasks
- 2 repos: `flask`, `ripgrep`
- local CUDA ONNX backend
- reranking disabled on purpose to expose embedding differences directly

| Model | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG | Search p50 | Flask index | Ripgrep index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jina preset | 0.5556 | 0.5556 | 0.5556 | 0.8462 | 0.6442 | 761.9 ms | 5.8 s | 11.9 s |
| CodeRankEmbed preset | 0.7222 | 0.7222 | 0.7222 | 1.0000 | 0.8108 | 611.4 ms | 14.7 s | 29.1 s |

Takeaway: CodeRankEmbed was clearly stronger on this small no-rerank slice, but it indexed much slower. The default local benchmark and docs still center Jina because Vera's full reranked pipeline is already very strong and the shorter indexing time matters in practice.

## Vera vs ColGREP

These ColGREP numbers are the earlier reference results recorded on the same 21-task, 4-repo suite. They remain useful as a retrieval quality reference because they show how the current Vera pipeline compares with a late-interaction code search system on the same workload.

| Metric | Vera `v0.7.0` | ColGREP (149M) | ColGREP Edge (17M) |
|--------|---------------|----------------|--------------------|
| Recall@1 | **0.7183** | 0.5710 | 0.5240 |
| Recall@5 | **0.7778** | 0.6670 | 0.5710 |
| Recall@10 | **0.8254** | 0.7140 | 0.7140 |
| MRR@10 | **0.9095** | 0.6170 | 0.5660 |
| nDCG@10 | **0.8361** | 0.5610 | 0.5240 |

Indexing time, 4 repos combined:

| Tool | Total time | Hardware |
|------|-----------|----------|
| Vera `v0.7.0` | `~70 s` | RTX 4080 |
| ColGREP (149M, CPU) | `~180 s` | Ryzen 5 7600X3D 6c/12t |
| ColGREP Edge (17M, CPU) | `~160 s` | Ryzen 5 7600X3D 6c/12t |

ColGREP's late-interaction design was a useful reference while improving Vera's own ranking and chunk selection.

## Legacy Public API Benchmark

This is the older public benchmark snapshot that still appears in older docs and release notes.

- 17 tasks
- 3 repos: `ripgrep`, `flask`, `fastify`
- mixed API and local runs

### Retrieval Quality

| Metric | ripgrep | cocoindex-code | vector-only | Vera hybrid |
|--------|---------|----------------|-------------|-------------|
| Recall@1 | 0.1548 | 0.1587 | 0.0952 | **0.4265** |
| Recall@5 | 0.2817 | 0.3730 | 0.4921 | **0.6961** |
| Recall@10 | 0.3651 | 0.5040 | 0.6627 | **0.7549** |
| MRR@10 | 0.2625 | 0.3517 | 0.2814 | **0.6009** |
| nDCG@10 | 0.2929 | 0.5206 | 0.7077 | **0.8008** |

### Local vs API Models

The local Jina models were competitive with the much larger Qwen3-Embedding-8B API model on that older 17-task benchmark:

| Metric | Jina local (ONNX) | Qwen3-8B (API) |
|--------|-------------------|----------------|
| MRR@10 | **0.68** | 0.60 |
| Recall@5 | 0.65 | **0.73** |
| Recall@10 | 0.73 | **0.75** |
| nDCG@10 | 0.72 | **0.81** |

### Performance Snapshot

From the same older benchmark set:

| Measure | Result |
|---------|--------|
| BM25-only search p95 | `3.5 ms` |
| Hybrid search p95 | `6749 ms` |
| `ripgrep` index time | `65.1 s` |
| `flask` index time | `20.2 s` |
| `fastify` index time | `41.8 s` |

## Limits And Caveats

- The current release benchmark is deterministic and fully local, which makes it better for regression gating.
- The legacy public snapshot is still useful for older comparisons, but it should not be treated as the current retrieval baseline.
- Benchmark numbers in this repository show comparative behavior, not a promise that another machine or codebase will land on the same values.

## Related Docs

- [`v0.7.0` accuracy improvements](./releases/v0.7.0-accuracy-improvements.md)
- [Indexing performance note](../benchmarks/indexing-performance.md)
- [Reproduction guide](../benchmarks/reports/reproduction-guide.md)
