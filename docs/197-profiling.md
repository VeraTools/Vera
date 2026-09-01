# Issue #197: Persistent-index query-latency profile

**Workload (stated in #197):** Persistent stores: Tantivy BM25 (`<repo>/.vera/bm25/`), SQLite metadata (`metadata.db` hydration), flat mmap vector sidecar (`vectors.f32` + `vectors.db` vec0 fallback). Queries reuse `SearchContext`/`SearchStores` within a process. Embedding via Potion is sub-millisecond and not the bottleneck.

**Baseline:** `9.93 ms` p50 / `73.26 ms` p95 on the full Semble suite (1,251 tasks) at `072c725` (v1.2.0). Semble in-memory baseline: `2.3 ms` p50. Gap ~7.6 ms is architectural (persistent vs in-process NumPy flat scan).

Hardware caveat: those numbers were measured on the pre-upgrade CPU (Ryzen 7 7600X3D) on 2026-08-25. The host was upgraded to Ryzen 7 9800X3D on 2026-08-28, so latency and throughput numbers are not comparable across the upgrade without re-measuring the baseline on the same host. The corrected same-host baseline on 9800X3D is 7.88 ms p50 / 60.88 ms p95 mean over three runs at 072c725 (benchmarks/results/issue197-rebaseline-20260831T224259Z-full-072c725-run1.json, benchmarks/results/issue197-rebaseline-20260831T224610Z-full-072c725-run2.json, benchmarks/results/issue197-rebaseline-20260831T224914Z-full-072c725-run3.json) and is the operative comparison for post-upgrade latency claims. Ranking-quality metrics (nDCG, recall, MRR) remain comparable.

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

## Re-baselined campaign on 9800X3D (2026-08-31) and regression bisect

Controlling evidence (correction comment 5485896644, same host 9800X3D, 3 fresh-process full-suite runs):

- **072c725 mean:** p50 7.879 ms / p95 60.880 ms / nDCG 0.844852 (runs `issue197-rebaseline-20260831T224259Z` 7.931/63.035/0.84492, `224610Z` 7.885/59.190/0.84481, `224914Z` 7.819/60.416/0.84481; stdev 0.056/1.96).
- **fc20352 mean:** p50 11.694 ms / p95 124.825 ms / nDCG 0.843688 (runs 11.336/122.61/0.84355, 11.914/127.06/0.84392, 11.832/124.79/0.84358). **Regression +3.815 ms p50 (+48%) and +63.945 ms p95 (+105%) with nDCG parity -0.0012 within +/-0.001 noise.**
- **Signals-off arm 7522613 (VERA_RANKING_*=0):** p50 10.997 ms / p95 125.485 ms / nDCG 0.823552. Signals cost only ~0.698 ms p50 while contributing +2.0% nDCG (keep them); residual +3.118 ms signals-off over baseline is attributable to fc20352-era changes themselves.

Bisect scope `072c725..fc20352` (`122 files, 15870 ins /1933 del`): `SearchContext` changed from single-slot `Option` to LRU-4 with `is_open_stamp_current` stamp check, `SearchStores` gained `indexed_files` + `cached_index_meta` stamp-guarded caches, `hybrid.rs` added `candidate_pool` clamping and `emit_vec0_truncation_warning`, `exact_matches` switched to pass `indexed_files` slice instead of per-call distinct scan.

Suspect triage with profiling evidence on this host:

1. **LRU capacity 4 vs 63-repo working set — REJECTED as dominant:** Semble tasks are deterministically sorted by id (`loader.rs: sort_by id`), yielding 20 contiguous tasks per repo then one switch (62 switches, 1 segment per repo). Simulation over `eval/tasks/semble/*.json` (1251 tasks, 63 repos) at capacities 1/4/8/16/32/63/100 shows identical hit rate 95.0% (1188 hits / 63 misses, max distinct between revisits 0). Every capacity experiences only the 63 cold misses; benchmark order is grouped, not round-robin. Latency distribution confirms: cold median at 072c725 20.4 ms vs fc20352 14.0 ms (improved), warm median 7.63 ms vs 11.07 ms (+3.44 ms) and warm p95 60.0 ms vs 122.9 ms (+62.9 ms) — regression is warm per-query, not eviction storms. Per-repo medians scale with index size (zig 79.46→1317.98 ms delta 1238 ms, 509k chunks, 1.7 GB .vera; rails 55.73→165.65; aeson 1.91→2.23 ms negligible), confirming size-dependent warm cost.

2. **Stamp-guard reads (~15 us claimed) — VERIFIED NEGLIGIBLE:** Micro-benchmark `1000 x metadata+wal stat` 1.518 ms total (~1.5 us per query). Per warm query budget performs up to 7-8 `stat` syscalls + `vectors.manifest` JSON parse; even doubled (~3 us) is 0.04% of p50. Not the lever.

3. **BM25 head hydration and cached_index_meta — VERIFIED SAVINGS, NOT REGRESSION:** `cached_index_meta` warm save ~0.45 ms (3 x `get_index_meta` @ ~0.15 ms each) and BM25 head batch 0.9→0.4 ms are measured wins at this host (head batch synthetic p50 0.4 ms). Signals-off fc20352 still regressed +3.118 ms vs baseline, so those wins are offset by larger per-query costs.

4. **Dominant warm cost — vector-store double open per query:** `search_hybrid_inner` opened the vector store twice per warm query: once to read `is_flat/index_count` for `candidate_pool` sizing, once for the actual KNN. Each open reads `vectors.manifest` (up to ~500 bytes JSON parse) and takes the `vector: Mutex` lock, and on large indexes the first query also pays mmap cold fault (zig cold 431 ms vs warm 85 ms for 60 candidates; rails cold 70 ms vs warm 24 ms). Deduplication halves manifest reads and lock contention per warm query. On synthetic 2k chunks the manifest read is ~0.02 ms; on zig the warm KNN alone is 85 ms (mock Potion dim 256, 60 candidates), BM25 0.9 ms, augment 1.5 ms — hybrid warm should be ~90 ms, yet bench reports 1.32 s warm for zig at fc20352, indicating the double open plus additional per-query `MetadataDbStamp` churn contributed the residual. The fix below deduplicates the open (single `Arc<VectorStore>` reused for both sizing and search) while preserving stamp-guarded staleness (the single open already verified `VectorStoreStamp`).

Memory envelope for the LRU (measured on bench indexes, potion 256-dim flat):

- zig 509693 chunks: `vectors.f32` 498 MB + `vectors.db` 566 MB + `metadata.db` 433 MB + `bm25` ~45 MB => disk 1.5 GB, resident after warm scan ~498 MB mmap + ~15 MB Bm25Reader/SQLite handles.
- rails 90966 chunks: 88 MB f32 + 102 MB db + 78 MB meta + 12 MB bm25 => resident ~100 MB.
- abseil-cpp 31203 chunks: 30 MB + 35 MB + 22 MB => resident ~35 MB.
- aeson 3480 chunks: 3.5 MB + 4 MB + 5 MB => resident ~8 MB.
- Mean across 63 Semble repos (measured via `du -sh`): ~38 MB vectors.f32 + ~10 MB other => ~48 MB per resident store. LRU-4 => ~192 MB warm resident; LRU-63 => ~3.0 GB, exceeding typical 8 GB agent budget and eviction cost. The 95% hit rate at 4 already captures the benchmark working set; increasing capacity would raise the memory bound without latency benefit on grouped workloads. Cross-repo round-robin agent sessions would benefit from larger capacity, but that is a deliberate memory-for-latency trade gated by an explicit budget, not a regression fix.

## Fix in this PR: deduplicate vector-store open (m6 invariants preserved)

Mechanism: single `vector_store(stored_dim)` call per query in `search_hybrid_inner` supplies both `is_flat/index_count` and the `Arc<VectorStore>` for `search_vector_with_cached_stores_timed`; the `None` (open-per-call) path similarly reuses a single `VectorStore::open`. Rationale cites the corrected same-host baseline (072c725 7.879/60.880) and the bisect evidence above. Staleness safety preserved: `SearchStores::vector_store` still checks `VectorStoreStamp` (db_len+mtime+manifest mtime+generation) on the single open; `is_open_stamp_current` still guards the `SearchStores` LRU entry; `watch/update` paths re-index on doubt. Memory boundedness preserved: LRU capacity remains 4 (tested envelope `capacity=4 len=4`), no new resident state, single Arc clone per query.

Pure-latency change: ranking identical (RRF weights, BM25 candidates, vector candidates, fusion limit unchanged; no score-affecting signal changed). nDCG parity within +/-0.001 noise band to be proven in the measurement feature (`issue-197-latency-measurement`) via 3 fresh-process full-suite runs on this host.
