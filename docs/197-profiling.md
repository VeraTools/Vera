# Issue #197: Persistent-index query-latency profile

**Workload (stated in #197):** Persistent stores: Tantivy BM25 (`<repo>/.vera/bm25/`), SQLite metadata (`metadata.db` hydration), flat mmap vector sidecar (`vectors.f32` + `vectors.db` vec0 fallback). Queries reuse `SearchContext`/`SearchStores` within a process. Embedding via Potion is sub-millisecond and not the bottleneck.

**Baseline:** `9.93 ms` p50 / `73.26 ms` p95 on the full Semble suite (1,251 tasks) at `072c725` (v1.2.0). Semble in-memory baseline: `2.3 ms` p50. Gap ~7.6 ms is architectural (persistent vs in-process NumPy flat scan).

## Measured stage breakdown (persistent path, warm process)

Source: issue #197 body + comment `2528f2e` warm measurements + local synthetic 2,000-chunk micro-benchmark (MockProvider dim 8, 200 files x 4 chunks, Tantivy + SQLite + flat mmap, `SearchStores` warm).

- **Vector stage ~60% of p50:** on `zig` (336k chunks) the flat mmap scan dominates warm p50: `2528f2e` reported vector stage ~10 ms p50 on zig, versus Semble's in-memory vectors at ~2 ms. Micro-benchmark on 2k chunks: vector KNN + batch hydration `~1.2 ms` (embedding `0.2 ms` + storage `1.0 ms`); scales linearly with index size. Cold first-touch mmap page-in adds ~10 to 20 ms p95 (issue notes "cold page-in").
- **BM25 hydration ~30% of p50:** Tantivy search itself is sub-ms; hydration of top-k chunk metadata from `metadata.db` is the cost. Issue notes BM25 hydration is the second hot spot. On zig, BM25 hydration for `limit 10` head + tail paging costs ~3 ms p50. Synthetic 2k-chunk baseline: BM25 `search + head hydration (10 x single-row get_chunk)` ~0.9 ms (tail paging not exercised for `limit 10` unfiltered, head used 10 individual `SELECT ... WHERE id=?` round trips). After this PR the head is batched into one `get_chunks_by_ids` call, mirroring the vector path, saving N-1 round trips (~0.5 ms on 2k, more on large limits).
- **Augmentation stage (exact-match):** previously ~18.4 ms p50 on zig from `SELECT DISTINCT file_path` scan; cached in `2528f2e` to ~1.4 ms via `MetadataDbStamp` (length+mtime of `metadata.db` + `-wal`). Synthetic 2k-chunk: cache hit `~15 µs`, miss (first call) `~8 ms` (distinct scan over 2k rows). Cache is stamp-guarded so edits invalidate.
- **Per-query index-meta reads:** `search_service::search` did 3x `get_index_meta` (`model_name`, `embedding_dim`, `document_prefix`) per query via `bm25_metadata` lock, even when the index hasn't changed. Micro-benchmark: 100x3 reads = `~45 ms` => `0.45 ms` per query (~4 to 5% of p50). These reads were pure overhead when the stamp is unchanged; now cached against the same `MetadataDbStamp` used for `indexed_files` (see implemented wins below).
- **Query-time source reads:** `383d21f` capped reads at `max_file_size_bytes` already bound memory; baseline kept. RST expansion cache is cycle-keyed, also preserved.
- **Embedding:** Potion query embedding ~0.2 ms, not a lever (issue explicitly says do not pursue CUDA).

## Cache-miss / hydration round-trip summary

- `indexed_files` cache miss: one `SELECT DISTINCT file_path FROM chunks` => tens of ms on large indexes (zig ~18 ms). Hit: stamp check only (~15 µs). Miss rate is file-modification driven; stamp includes `-wal` so concurrent indexer commits invalidate.
- `vector_store` cache miss: reopen `vectors.db` + mmap; hit: stamp equality check (db_len+mtime+manifest). Miss rate is dimension or file-change driven.
- BM25 hydration: baseline head `limit` rows via `limit` single-row queries (N round trips), tail paged at 256 to 900 per `get_chunks_by_ids` batch; after this PR head is also batched (1 round trip for limit <=900). Vector hydration: single `get_chunks_by_ids` batch for all `limit` rows => 1 round trip. Head was the outlier before batching.
- `has_filter_matches` (vec0 diagnostic) scans full table via `SELECT ... FROM chunks` and iterates in Rust; filtered queries pay full scan (2k rows ~0.8 ms, 336k rows ~100 ms estimated). This is only on filtered vec0 path; flat filtered fetches whole index instead, but diagnostic still scans.
- `get_index_meta` 3x per query: each is a single-row `SELECT value FROM index_metadata WHERE key=?` (now cached).

## Implemented bounded latency wins (staleness-safe)

Index meta cache and LRU are bounded (stamp-guarded and LRU-capped) and re-verify freshness on every query; BM25 head-batch is bounded by its 900 page limit and preserves BM25 order. All build on the `383d21f` baseline (capped reads + cycle-keyed RST) and do not weaken it.

1. **Cache index meta against `MetadataDbStamp`:** same stamp as `indexed_files`; avoids 3 SQLite reads per warm query (~0.45 ms). Cited by meta-cache change.
2. **Batch BM25 head hydration:** as described in Cache-miss / hydration round-trip summary. Cited by BM25 batch change.
3. **Bounded multi-repo resident store:** `SearchContext` now caches up to 4 `SearchStores` via LRU (previously exactly one `Option`). Cross-repo agent sessions (for example querying 4 repos round-robin) paid open cost per switch (~5 to 10 ms: Bm25Index open + MetadataStore open + VectorStore open). LRU of 4 keeps hot repos resident while capping memory (4x mmap handles, 4x BM25 readers). Eviction is LRU, stamp-checked for freshness via `open_stamp`. Cited by LRU change.

## Methodology notes

- Warm vs cold separated: warm is second query in same process after stores opened and mmap touched; cold is first query in fresh process (page-in). Semble comparison uses warm persistent path vs Semble in-memory.
- Synthetic workload uses `MockProvider` (dim 8) to isolate storage cost from model cost, as issue notes model is not the bottleneck.
- Full Semble suite numbers are from issue body (9.93 ms p50) and `2528f2e` comment (zig warm augmentation 18.4 to 1.4 ms, total 31.1 to 14.1 ms, full suite p95 73.3 to 64.5 ms).

## PR-head verification (f3ec282)

Synthetic 2k-chunk micro-benchmark at PR-head `f3ec282543723bbc63542c969871db3c0c6ec20b`: BM25 head batch p50 ~0.4 ms vs baseline 0.9 ms (N single-row), meta cache saves ~0.45 ms per query, augmentation cache hit ~15 µs. Ranking unchanged (same chunks, same order, same scores), so full Semble nDCG@10 is identical to baseline full suite artifact `benchmarks/results/2026-08-16-vera-cuda-v1-full.json` (p50 9.93 ms, p95 73.26 ms). This PR is latency-only with no scoring change, validated by 17 BM25 tests (including `paged_hydration_matches_unfiltered_manual_filtering_across_pages` and `filtered_search_finds_scoped_result`) plus hybrid and search_service tests (1104 total, `memory envelope: capacity=4 len=4` recorded in `memory_bounded_across_multiple_indexed_repos_with_recorded_envelope`). Full Semble re-run not required for benchmark integrity per AGENTS.md (no ranking signal tuned to ground truth).
