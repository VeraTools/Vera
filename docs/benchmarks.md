# Vera Benchmarks

This page tracks the current Semble comparison first, then historical snapshots and ablations.

## Provenance

Semble-derived runs use the task set from [Semble v0.5.5](https://github.com/MinishLab/semble/tree/v0.5.5) at commit `9218491`: 1,251 tasks across 63 pinned repositories. The converter and corpus manifests pin that commit and record a deterministic task-content hash. The 320-task tuning subset is an unchanged 16-repository slice of the same release.

Vera reports `vera-graded-2-1-task-mean-v1`: primary targets have relevance 2, secondary targets have relevance 1, duplicate matching chunks receive credit once, and scores are averaged over tasks. Historical Semble published tables use binary relevance and repository/language macro averages. The 2026-09-03 comparison below uses the same graded scorer for both tools. New JSON reports record both the Semble snapshot and metric contract; older reports without those fields are `unknown-legacy` and use Vera's graded calculation.

## 2026-09-03 Semble Comparison (Vera row refreshed for v1.4.0)

The Semble benchmark uses 1,251 tasks across 63 repositories. Both tools used the same `minishlab/potion-code-16M-v2` embeddings, harness, graded relevance, and suffix-corrected path matching in the scorer.

| Tool | nDCG@10 | R@1 | R@5 | R@10 | MRR | Query p50 | Index time | Index size |
|------|---------|------|------|-------|-----|-----------|------------|------------|
| Vera v1.4.0 | 0.8437 | 0.6713 | **0.9189** | 0.9502 | 0.8258 | 6.4 ms | 115 s | **4.7 GB** |
| Semble 0.5.5, full rerank stack | **0.8514** | **0.6747** | 0.9177 | **0.9656** | **0.8348** | **2.3 ms** | **100 s** | 32 GB |

The Vera row was re-measured on 2026-09-03 at head 98e6e50 on the AMD Ryzen 7 9800X3D host with shipped defaults (local potion-code-16M-v2 embeddings, no reranker) in three interleaved full-suite runs; the Semble column is unchanged from the 2026-08-23 comparison. The v1.2.0 row read 0.8450 on the old CPU, so quality is within run-to-run variance and nDCG parity held within 0.000005 in the filter-during-scan on/off ablation. Filter-during-scan in v1.4.0 halved filtered-query p50 from ~12 ms to 6.4 ms and p95 from ~139 ms to 65 ms (measured in-process by the eval harness) with that parity.

On the 320-task tuning subset Vera scored `0.8538` versus Semble at `0.8494` nDCG; on the contamination-check independent set (10 repositories disjoint from Semble's 63, locally generated tasks) Vera scored `0.7674` versus Semble at `0.7655`. The aggregate table is the comparison summary; repo-level variation is not a substitute for the full result.

Vector search uses an exact flat SIMD scan over memory-mapped vectors, dual-written alongside sqlite-vec (`VERA_VECTOR_SCAN=vec0` selects the old path). On the largest corpus repo (zig, 336k chunks) the vector stage dropped from ~460 ms to ~21 ms per query; quality is unchanged because the scan is exact. Index size grew because vectors are stored in both backends during the transition.

Embedding alternatives and the dual-set reranker screening are documented in [models.md](models.md).

## Historical v1.0 Full-Pipeline Benchmark

The v1.0.0-rc run used the full Semble v0.5.5 task set on the `vera-cuda` lane: hybrid BM25+vector retrieval, RRF fusion, and a local ONNX cross-encoder reranker on CUDA, measured 2026-08-16 against the v1.0.0 release candidate. The older "BM25 scoped filters" full-suite row (`nDCG@10` 0.7267, measured in May 2026) was a BM25-only lane using an older harness. Cross-date comparisons are approximate.

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
      "name": "harrier-270m",
      "backend": "custom-onnx",
      "repo": "onnx-community/harrier-oss-v1-270m-ONNX",
      "onnx_file": "onnx/model.onnx",
      "onnx_data_file": "onnx/model.onnx_data",
      "tokenizer_file": "tokenizer.json",
      "pooling": "last-token",
      "query_prefix": "Instruct: Given a code search query, retrieve relevant code passages that answer the query\nQuery:",
      "dim": 640,
      "max_length": 512,
      "rerank": false,
      "revision": "d59c919d0159aea2c19ed7d04288fcdd048d0f9c"
    },
    {
      "name": "potion-control",
      "backend": "potion",
      "rerank": false
    }
  ]
}
```

The `custom-onnx` backend defaults to CPU; set `execution_provider` to `cuda`, `rocm`, `coreml`, `openvino`, or another supported provider when needed. `repo` downloads from Hugging Face and `dir` points at an existing local model directory. Set `revision` to an immutable commit SHA to pin the download: pinned revisions resolve from that ref, cache under `models/<repo>/revisions/<revision>/`, and join the model identity, so an index built on one revision is never reused for another. Omitted means `main`. `revision` is only valid on Hugging Face repo lanes, not on `dir`, `bm25`, `api`, or `potion` lanes. API lanes use `model_id`, `query_prefix`, and `environment` for endpoint settings. Every report includes the resolved lane contract, Vera Git SHA, timestamp, command and redacted environment summary, corpus SHAs, and a SHA-256 identity of the selected task IDs.

The lower-level Python runner supports the same task controls for its existing retrieval modes:

```bash
python3 benchmarks/scripts/run_vera_benchmarks.py \
  --modes bm25-only hybrid-norerank \
  --task-id intent-001,intent-002 --category intent --skip-index
```

### Historical Embedding Model Screening (2026-08-21)

This screening predates the current comparison above. All lanes run hybrid BM25+vector retrieval with the reranker disabled, so the table isolates the embedding model. Every model is pinned to an immutable commit SHA (see `revision` above). Measured at Vera commit `69bc514` on one CUDA GPU.

Subset (320 tasks, 16 repos):

| Lane | nDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms | Index s | Storage |
|------|---------|----------|----------|-----------|-----|--------|--------|---------|---------|
| BM25 (no embeddings) | 0.7792 | 0.5911 | 0.8609 | 0.9141 | 0.7683 | 5.0 | 17.1 | 12 | 323 MB |
| Potion Code 16M | 0.7656 | 0.5786 | 0.8547 | 0.9000 | 0.7508 | 13.2 | 57.6 | 17 | 418 MB |
| potion-code-16M-v2 (API) | 0.7930 | 0.5990 | 0.8812 | 0.9328 | 0.7769 | 13.5 | 56.8 | 56 | 419 MB |
| F2LLM-v2-80M (API) | 0.7897 | 0.6068 | 0.8875 | 0.9141 | 0.7774 | 20.3 | 62.0 | 163 | 419 MB |
| F2LLM-v2-160M (API) | 0.7932 | 0.6021 | 0.8859 | 0.9234 | 0.7807 | 29.7 | 103.5 | 300 | 609 MB |
| F2LLM-v2-330M (API) | 0.7971 | 0.6052 | 0.8938 | 0.9328 | 0.7805 | 31.3 | 90.8 | 427 | 609 MB |
| jina v5 nano | 0.7956 | 0.6005 | 0.8859 | 0.9328 | 0.7788 | 44.9 | 148.6 | 147 | 672 MB |
| CodeRankEmbed 137M | 0.7949 | 0.6021 | 0.8875 | 0.9328 | 0.7774 | 21.3 | 84.7 | 375 | 670 MB |
| Harrier 270M | 0.7938 | 0.6068 | 0.8813 | 0.9281 | 0.7799 | 24.2 | 77.4 | 2391 | 609 MB |
| Harrier 0.6B fp32 | 0.7971 | 0.5974 | 0.8906 | 0.9422 | 0.7791 | 27.4 | 94.3 | 7082 | 800 MB |
| BitNet-270M i2_s (API) | 0.7743 | 0.5849 | 0.8656 | 0.9062 | 0.7620 | 51.0 | 124.5 | 18061 | 607 MB |

Full suite (1,251 tasks, 63 repos):

| Lane | nDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms | Index s |
|------|---------|----------|----------|-----------|-----|--------|--------|---------|
| jina v5 nano | 0.7452 | 0.5560 | 0.8320 | 0.8846 | 0.7184 | 22.9 | 128.8 | 1823 |
| F2LLM-v2-330M (API) | 0.7425 | 0.5564 | 0.8343 | 0.8778 | 0.7164 | 57.9 | 222.4 | 3446 |
| CodeRankEmbed 137M | 0.7401 | 0.5512 | 0.8311 | 0.8802 | 0.7123 | 27.0 | 124.5 | 3308 |
| potion-code-16M-v2 (API) | 0.7388 | 0.5516 | 0.8296 | 0.8766 | 0.7129 | 16.8 | 84.6 | 443 |

Read: all neural lanes except BitNet sit within ~0.005 nDCG of each other on both scales; no code-specialization premium showed up at Vera's current 512-token embedding truncation. BitNet-270M lands below even BM25, and its 1.58-bit GGUF only loads in the microsoft/BitNet fork (upstream llama.cpp rejects the custom tensor types; the fork's CUDA path asserts), so it is out on both axes. In this historical screening, the practical choice turned on license and operational cost more than retrieval quality. Harrier's index times (16x and 48x Jina's on the subset) disqualified both sizes for the default.

Precision notes: jina lanes run its fp16 export on GPU (the automatic quantized-to-fp16 swap); CodeRankEmbed and Harrier run fp32 (CodeRankEmbed ships no fp16 export; Harrier's fp16 export uses GroupQueryAttention with attention bias, which the ONNX Runtime CUDA kernel rejects). Harrier 0.6B fp32 has multi-shard external weights (`model.onnx_data_1`), which Vera's downloader does not support yet; it ran from a pinned local directory instead. The `(API)` rows are models Vera cannot run natively, served by a local OpenAI-compatible shim and measured through the `api` lane backend: potion-code-16M-v2 (model2vec, CPU) and the F2LLM-v2 family (transformers, bf16 on CUDA, 512-token truncation to match the other lanes), all pinned by upstream commit (e9d2a44; 19a4fd85, 0e04993a, 1b8f0301). BitNet-270M ran on the microsoft/BitNet fork (0b341e58) `llama-server` on CPU with the i2_s GGUF pinned at 5f1c2fd, behind a proxy that truncates inputs to 510 tokens (the server errors instead of truncating) and restarts the server on its concurrency crashes.

Artifacts (uncommitted, under `benchmarks/results/`): `harrier-screening-20260821T050037Z-subset-{vera-bm25,vera-potion,vera-cuda}.json`, `harrier-screening-20260821T082401Z-subset-harrier-270m-cuda.json`, `harrier-screening-20260821T093941Z-subset-harrier-0.6b-fp32-cuda.json`, `harrier-screening-20260821T111613Z-subset-coderankembed-cuda.json`, `harrier-screening-20260821T133638Z-subset-f2llm-80m-api.json`, `harrier-screening-20260821T135016Z-subset-potion-v2-api.json`, `harrier-screening-20260821T173032Z-subset-f2llm-160m-api.json`, `harrier-screening-20260821T173032Z-subset-f2llm-330m-api.json`, `harrier-screening-20260821T114648Z-full-vera-cuda.json`, `harrier-screening-20260821T114654Z-full-coderankembed-cuda.json`, `harrier-screening-20260821T135235Z-full-potion-v2-api.json`, `harrier-screening-20260821T174547Z-full-f2llm-330m-api.json`, `harrier-screening-20260821T180030Z-subset-bitnet-270m-api.json`.

### Independent Contamination Check (2026-08-22)

MinishLab publishes both potion-code-16M-v2 and Semble, so Semble alone cannot clear potion-v2 of evaluation contamination. This check reruns the top candidates on an independently built set: 10 pinned repositories disjoint from Semble's 63 (celery, jinja, rayon, viper, lodash, ky, sidekiq, okhttp, zlib, nlohmann/json), with 180 locally generated Semble-style tasks (100 intent, 50 cross_file, 30 symbol_lookup) and verified file-level ground truth. Same harness, same no-rerank contract, same revisions.

| Lane | nDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR | p50 ms | Index s |
|------|---------|----------|----------|-----------|-----|--------|---------|
| jina v5 nano | 0.7149 | 0.3889 | 0.7972 | 0.8750 | 0.7148 | 13.3 | 61 |
| CodeRankEmbed 137M | 0.7069 | 0.4000 | 0.7731 | 0.8546 | 0.7212 | 14.8 | 153 |
| F2LLM-v2-330M (API) | 0.7061 | 0.3944 | 0.7704 | 0.8574 | 0.7178 | 21.6 | 167 |
| potion-code-16M-v2 (API) | 0.7019 | 0.3935 | 0.7741 | 0.8574 | 0.7050 | 10.1 | 26 |
| BM25 (no embeddings) | 0.6506 | 0.3630 | 0.7278 | 0.7852 | 0.6682 | 3.3 | 5 |

Read: the historical Semble ordering carries over (Jina ahead, then CodeRankEmbed, F2LLM-330M, and potion-v2 within ~0.005 of each other, BM25 far behind). If potion-v2 were inflated by its Semble relationship it should look relatively stronger there; instead its gap to Jina widens from -0.0026 (Semble subset) to -0.0130 on fresh repos. The screening comparison shows no home-field advantage. All neural lanes drop 0.06-0.09 in absolute terms on this set (smaller repos, LLM-written queries), and the neural-over-BM25 margin roughly triples, which is the opposite of what task leakage would produce. potion-v2 remains the best permissively licensed candidate and indexes 6x faster than the runners-up.

Note: a first Jina pass accidentally ran with the reranker enabled (the historical bare `--tool vera-cuda` lane enabled reranking; the screening lane spec sets `rerank: false`). It scored 0.7309, which was the reranker's contribution, and its p50 was ~2.6 s from per-query reranking. The row above is the corrected no-rerank run.

Artifacts (uncommitted, under `benchmarks/results/`): `indep-20260822T150853Z-{bm25,potion-v2-api,jina-cuda,coderankembed-cuda,f2llm-330m-api}.json`. The corpus manifest (`eval/indep-corpus.toml`) and task files (`eval/tasks/indep/`) live in the bench worktree.

### Historical Main Results

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

## Historical v0.7.0 Benchmark

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

The raw per-version JSONs were pruned from the tree; they remain in git history.

### Historical Performance Snapshot

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

The tuning-run JSON artifacts were pruned from the tree; see git history for the original files.

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

### Historical Semble Comparison

Vera is benchmarked on Semble v0.5.5's task set. Rows labeled Semble were produced by Semble's own harness; rows labeled Vera use Vera's graded metric contract described in [Provenance](#provenance).

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

This six-task snapshot is historical. See [models.md](models.md#embedding-screening) for the current embedding comparison.

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

- [Query-aware retrieval ADR](./adr/005-query-aware-retrieval.md)
- [Model selection and screening](./models.md)
