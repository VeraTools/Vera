# Vera Benchmarks

This page tracks the current Semble comparison first, then historical snapshots and ablations.

## Provenance

Semble-derived runs use the task set from [Semble v0.5.5](https://github.com/MinishLab/semble/tree/v0.5.5) at commit `9218491`: 1,251 tasks across 63 pinned repositories. The converter and corpus manifests pin that commit and record a deterministic task-content hash. The 320-task tuning subset is an unchanged 16-repository slice of the same release.

Vera reports `vera-graded-2-1-task-mean-v1`: primary targets have relevance 2, secondary targets have relevance 1, duplicate matching chunks receive credit once, and scores are averaged over tasks. Historical Semble published tables use binary relevance and repository/language macro averages. The 2026-09-03 comparison below uses the same graded scorer for both tools. New JSON reports record both the Semble snapshot and metric contract; older reports without those fields are `unknown-legacy` and use Vera's graded calculation.

## Limits And Caveats

- The current release benchmark is deterministic and fully local, which makes it better for regression gating.
- The legacy public snapshot is still useful for older comparisons, but it should not be treated as the current retrieval baseline.
- Benchmark numbers in this repository show comparative behavior, not a promise that another machine or codebase will land on the same values.

## Current Results

### 2026-09-03 Semble Comparison (Vera row refreshed for v1.4.0)

The Semble benchmark uses 1,251 tasks across 63 repositories. Both tools used the same `minishlab/potion-code-16M-v2` embeddings, harness, graded relevance, and suffix-corrected path matching in the scorer.

| Tool | nDCG@10 | R@1 | R@5 | R@10 | MRR | Query p50 | Index time | Index size |
|------|---------|------|------|-------|-----|-----------|------------|------------|
| Vera v1.4.0 | 0.8437 | 0.6713 | **0.9189** | 0.9502 | 0.8258 | 6.4 ms | 115 s | **4.7 GB** |
| Semble 0.5.5, full rerank stack | **0.8514** | **0.6747** | 0.9177 | **0.9656** | **0.8348** | **2.3 ms** | **100 s** | 32 GB |

The Vera row was re-measured on 2026-09-03 at head 98e6e50 on the AMD Ryzen 7 9800X3D host with shipped defaults (local potion-code-16M-v2 embeddings, no reranker) in three interleaved full-suite runs; the Semble column is unchanged from the 2026-08-23 comparison. The v1.2.0 row read 0.8450 on the old CPU, so quality is within run-to-run variance and nDCG parity held within 0.000005 in the filter-during-scan on/off ablation. Filter-during-scan in v1.4.0 halved filtered-query p50 from ~12 ms to 6.4 ms and p95 from ~139 ms to 65 ms (measured in-process by the eval harness) with that parity.

On the 320-task tuning subset Vera scored `0.8538` versus Semble at `0.8494` nDCG; on the contamination-check independent set (10 repositories disjoint from Semble's 63, locally generated tasks) Vera scored `0.7674` versus Semble at `0.7655`. The aggregate table is the comparison summary; repo-level variation is not a substitute for the full result.

Vector search uses an exact flat SIMD scan over memory-mapped vectors, dual-written alongside sqlite-vec (`VERA_VECTOR_SCAN=vec0` selects the old path). On the largest corpus repo (zig, 336k chunks) the vector stage dropped from ~460 ms to ~21 ms per query; quality is unchanged because the scan is exact. Index size grew because vectors are stored in both backends during the transition.

Embedding alternatives and the dual-set reranker screening are documented in [models.md](models.md).

## Related Docs

- [Query-aware retrieval ADR](./adr/005-query-aware-retrieval.md)
- [Model selection and screening](./models.md)

- [Benchmark history](benchmarks-history.md)
