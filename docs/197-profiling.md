# Issue #197 — Persistent-index query-latency profile

**Workload (stated in #197):** Persistent stores: Tantivy BM25 (`<repo>/.vera/bm25/`), SQLite metadata (`metadata.db` hydration), flat mmap vector sidecar (`vectors.f32` + `vectors.db` vec0 fallback). Queries reuse `SearchContext`/`SearchStores` within a process. Embedding via Potion is sub-millisecond and not the bottleneck.

**Baseline:** `9.93 ms` p50 / `73.26 ms` p95 on the full Semble suite (1,251 tasks) at `072c725` (v1.2.0). Semble in-memory baseline: `2.3 ms` p50. Gap ~7.6 ms is architectural (persistent vs in-process NumPy flat scan).

## Measured stage breakdown (persistent path, warm process)

Source: issue #197 body + comment `2528f2e` warm measurements + local synthetic 2,000-chunk micro-benchmark (MockProvider dim 8, 200 files x 4 chunks, Tantivy + SQLite + flat mmap, `SearchStores` warm).

- **Vector stage ~60% of p50** — on `zig` (336k chunks) the flat mmap scan dominates warm p50: `2528f2e` reported vector stage ~10 ms p50 on zig, versus Semble's in-memory vectors at ~2 ms. Micro-benchmark on 2k chunks: vector KNN + batch hydration `~1.2 ms` (embedding `0.2 ms` + storage `1.0 ms`); scales linearly with index size. Cold first-touch mmap page-in adds ~10–20 ms p95 (issue notes "cold page-in").
- **BM25 hydration ~30% of p50** — Tantivy search itself is sub-ms; hydration of top-k chunk metadata from `metadata.db` is the cost. Issue notes BM25 hydration is the second hot spot. On zig, BM25 hydration for `limit 10` head + tail paging costs ~3 ms p50. Synthetic 2k-chunk: BM25 `search + head hydration (10 x single-row get_chunk)` ~0.9 ms, tail paging not exercised for `limit 10` unfiltered. The head uses 10 individual `SELECT ... WHERE id=?` round trips.
- **Augmentation stage (exact-match)** — previously ~18.4 ms p50 on zig from `SELECT DISTINCT file_path` scan; cached in `2528f2e` to ~1.4 ms via `MetadataDbStamp` (length+mtime of `metadata.db` + `-wal`). Synthetic 2k-chunk: cache hit `~15 µs`, miss (first call) `~8 ms` (distinct scan over 2k rows). Cache is stamp-guarded so edits invalidate.
- **Per-query index-meta reads** — `search_service::search` does 3× `get_index_meta` (`model_name`, `embedding_dim`, `document_prefix`) per query via `bm25_metadata` lock, even when the index hasn't changed. Micro-benchmark: 100×3 reads = `~45 ms` → `0.45 ms` per query (~4–5% of p50). These reads are pure overhead when the stamp is unchanged; they can be cached against the same `MetadataDbStamp` used for `indexed_files`.
- **Query-time source reads** — `382d...` capped reads at `max_file_size_bytes` already bound memory; baseline kept. RST expansion cache is cycle-keyed, also preserved.
- **Embedding** — Potion query embedding ~0.2 ms, not a lever (issue explicitly says do not pursue CUDA).

## Cache-miss / hydration round-trip summary

- `indexed_files` cache miss: one `SELECT DISTINCT file_path FROM chunks` → tens of ms on large indexes (zig ~18 ms). Hit: stamp check only (~15 µs). Miss rate is file-modification driven; stamp includes `-wal` so concurrent indexer commits invalidate.
- `vector_store` cache miss: reopen `vectors.db` + mmap; hit: stamp equality check (db_len+mtime+manifest). Miss rate is dimension or file-change driven.
- BM25 hydration: head `limit` rows via `limit` single-row queries (N round trips). Tail paged at 256–900 per `get_chunks_by_ids` batch. Vector hydration: single `get_chunks_by_ids` batch for all `limit` rows → 1 round trip. BM25 head is the outlier with N round trips.
- `has_filter_matches` (vec0 diagnostic) scans full table via `SELECT ... FROM chunks` and iterates in Rust; filtered queries pay full scan (2k rows ~0.8 ms, 336k rows ~100 ms estimated). This is only on filtered vec0 path; flat filtered fetches whole index instead, but diagnostic still scans.
- `get_index_meta` 3× per query: each is a single-row `SELECT value FROM index_metadata WHERE key=?`.

## What remains for a bounded latency win (and is staleness-safe)

1. **Cache index meta against `MetadataDbStamp`** — same stamp as `indexed_files`; avoids 3 SQLite reads per warm query (~0.45 ms). Cited by meta-cache change.
2. **Batch BM25 head hydration** — replace N single-row `get_chunk` calls with one `get_chunks_by_ids` batch for the head, mirroring vector path. Saves N-1 round trips (~0.5 ms on synthetic, more on large indexes with higher limit). Cited by BM25 batch change.
3. **Bounded multi-repo resident store** — `SearchContext` currently caches exactly one `SearchStores` (`Option<(PathBuf, Arc<SearchStores>)>`). Cross-repo agent sessions (e.g., querying 4 repos round-robin) pay open cost per switch (~5–10 ms: Bm25Index open + MetadataStore open + VectorStore open). A bounded LRU of 4 keeps hot repos resident while capping memory (4× mmap handles, ~4× BM25 readers). Eviction is LRU, stamp-checked for freshness. Cited by LRU change.

All three are bounded (stamp-guarded or LRU-capped) and re-verify freshness on every query; they build on the 383d21f baseline (capped reads + cycle-keyed RST) and do not weaken it.

## Methodology notes

- Warm vs cold separated: warm = second query in same process after stores opened and mmap touched; cold = first query in fresh process (page-in). Semble comparison uses warm persistent path vs Semble in-memory.
- Synthetic workload uses `MockProvider` (dim 8) to isolate storage cost from model cost, as issue notes model is not the bottleneck.
- Full Semble suite numbers are from issue body (9.93 ms p50) and `2528f2e` comment (zig warm augmentation 18.4→1.4 ms, total 31.1→14.1 ms, full suite p95 73.3→64.5 ms).
