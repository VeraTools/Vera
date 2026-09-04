# How Vera Works

Vera's search pipeline retrieves candidates, fuses results, applies deterministic ranking, and optionally reranks. Every stage was chosen based on benchmarks against real codebases, not assumptions.

## Parsing: Tree-Sitter Chunks

Vera parses source files into ASTs using tree-sitter grammars compiled into the binary. Instead of splitting code into arbitrary line ranges, it extracts discrete structural units such as functions, classes, structs, traits, interfaces, methods, and `impl` blocks.

For config and document-like files, Vera uses whole-file chunks instead. Module-level gaps between symbols are also kept as chunks when they carry useful retrieval context.

Each chunk carries metadata: file path, line range, language, symbol name, and symbol type. This means search results map to actual code boundaries, not random slices.

Large symbols (>200 lines) are split at logical boundaries. When a definition exceeds the chunk size limit (about 200 lines), the chunker splits it into multiple chunks that share the bare symbol name with a distinct part index. Storage keeps the bare name; display renders it as `name (part N)`. This preserves identity across split parts: `vera structural definitions` finds split symbols by bare name, `vera references` resolves their single call site, `vera dead-code` deduplicates parts by (symbol, file), and JSON output carries the bare name with a `part_index` field. Languages without a tree-sitter grammar fall back to sliding-window chunking. See [features.md](features.md#tree-sitter-structural-parsing) for chunking benchmarks.

During parsing, Vera also records file-level diagnostics such as tree-sitter error nodes, Tier 0 fallback, and outright parse failures. `vera stats` surfaces these later as index-health signals instead of silently dropping them on the floor.

## Retrieval: BM25 + Vector Search

Two retrieval paths run in parallel for every query:

**BM25 (keyword matching)** uses a Tantivy index over structured chunk text, including content, symbol names, file paths, and filename/path tokens. It handles exact identifier and config-style lookups. searching for `parse_config` finds that exact function.

**Vector search (semantic matching)** embeds the query and compares it against pre-computed chunk embeddings. Vera writes every vector to sqlite-vec and to a flat `.vera/vectors.f32` sidecar. The default path memory-maps the sidecar and computes exact Euclidean L2 with SimSIMD; `VERA_VECTOR_SCAN=vec0` selects sqlite-vec for rollback and ablation. Deleted SQLite rowids remain represented by `.vera/vectors.tombs`, so they are skipped without moving later rows. Watch-mode updates write only changed rows at their rowid-derived offsets, while initial builds and recovery atomically publish full sidecar files. The flat file is fsynced before the generation manifest is renamed last; an absent or inconsistent sidecar is rebuilt from `vec_chunks`. This catches conceptual matches. searching "authentication middleware" finds relevant auth code even if those exact words don't appear.

When a query carries path, glob, or language filters, the flat SIMD scan applies those filters during the scan via an eligibility map. Ineligible chunks are skipped before hydration instead of being hydrated and filtered afterward. This is enabled by default (`retrieval.vector_filter_during_scan`, env `VERA_VECTOR_FILTER_DURING_SCAN`). On the full 1,251-task Semble suite it halved filtered-query latency (p50 from about 12 ms to 6.4 ms) with ranking parity verified by an on/off differential suite.

Neither path alone is sufficient. BM25 misses semantic matches. Vector search misses exact identifiers and ranks poorly. Combining them covers both.

## Fusion: Reciprocal Rank Fusion

Results from both retrieval paths are merged using Reciprocal Rank Fusion (RRF). RRF scores each result based on its rank in each list:

```
score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))
```

A result that ranks high in both lists gets a high fused score. A result that ranks high in only one list still appears, but lower. The constant `k` (default: 60) controls how much weight goes to top-ranked vs. lower-ranked results.

RRF is simple, parameter-light, and doesn't need training data. It consistently outperforms either retrieval path alone.

## Query-Aware Ranking

After fusion, Vera applies lightweight deterministic ranking logic before final reranking.

Current ranking combines BM25 and vector results with RRF (`k=60`), then applies a file-coherence boost, keyword path boost, content coverage, content-symbol definition boost, and exact-match concept pool tail injection capped at 4 definitions per file.

Each boost is individually toggleable via config and environment: `retrieval.ranking_filename_stem_boost` (`VERA_RANKING_FILENAME_STEM_BOOST`, with `retrieval.ranking_filename_stem_min_ratio` / `VERA_RANKING_FILENAME_STEM_MIN_RATIO` and `retrieval.ranking_filename_stem_skip_symbol_queries` / `VERA_RANKING_FILENAME_STEM_SKIP_SYMBOL_QUERIES`), `retrieval.ranking_definition_boost` (`VERA_RANKING_DEFINITION_BOOST`), and `retrieval.ranking_recall_pool_expansion` (`VERA_RANKING_RECALL_POOL_EXPANSION`). Each shipped signal survived a preregistered ablation on the full suite plus an independent set; signals that did not clear the 0.5 percent bar were recorded as negative results and not shipped.

This stage handles cases that dense retrieval alone is bad at:

- exact filename and exact identifier queries
- path-heavy config lookups
- noisy test and docs matches
- broad natural-language queries that need structural results instead of tiny helpers
- same-file and cross-file answer completion

For broad intent queries, Vera also keeps a deeper fused candidate pool before final truncation. This matters when raw RRF pushes the right `struct` or `impl` block just outside the requested top N and a tiny helper would otherwise win by default.

This is also where Vera adds a small amount of query-aware candidate expansion, such as pulling in related implementation blocks or same-file structural context when the initial hit is too narrow.

## Reranking: Cross-Encoder

Reranking is opt-in through `retrieval.reranking_enabled` and is off by default. When enabled, the top fused candidates are sent to a cross-encoder reranker. Unlike embeddings (which encode query and document separately), the cross-encoder reads the query and each candidate together as a single pair, scoring relevance jointly.

The default no-reranker path ends with the deterministic ranking stage. The 2026-08-23 dual-set cross-encoder screening scored every tested reranker below that heuristic baseline. See [models.md](models.md#reranking) for the scores and the recommended local override.

With Jina ONNX local models, the reranker runs on-device via ONNX Runtime. With Potion Code embeddings, enabling reranking uses the local CPU reranker unless an API reranker is configured; when reranking is off, deterministic ranking is the final stage. With API mode, reranking calls your configured endpoint. Obvious filename and path-dominant queries can skip reranking when lexical evidence is already decisive.

Large candidate sets are batched automatically to stay within the reranker's request limits. Oversized documents are truncated at newline boundaries before scoring. See [features.md](features.md#cross-encoder-reranking) for configuration details.

API rerankers use an explicit wire protocol, `retrieval.reranker_protocol`, with `generic` (`top_n`/`results` payload) and `voyage` (`top_k`/`data` payload) variants. When unset, hostname auto-detection picks a sensible default and can be overridden. Related knobs include `retrieval.reranker_task_instruction` and `retrieval.reranker_task_field` for providers that accept a task instruction, `retrieval.reranker_return_documents`, and `retrieval.reranker_rate_limit_wait_secs` (env `VERA_RERANK_RATE_LIMIT_WAIT_SECS`) which caps how long the client waits out 429 quota windows before degrading to unreranked results. Permanent 4xx errors are not retried.

## Storage

Everything lives in two places:

- **`.vera/`** in the project root. SQLite metadata (chunks, file hashes, file-level index state), Tantivy BM25 index, sqlite-vec vectors, and the flat vector sidecar (`vectors.f32`, `vectors.tombs`, and `vectors.manifest`). One directory per project.
- **`$XDG_DATA_HOME/vera/models/`** (or `~/.vera/models/` on existing installs): cached local model assets. Downloaded once by `vera setup`.

The index is a SQLite database, a Tantivy directory, and the flat vector sidecar files. No external services, no daemons, no background processes.

## Incremental Updates

`vera update .` compares content hashes against stored metadata, re-processes only changed files, and refreshes the persisted file-level index state. That keeps parse failures and fallback counts visible across incremental runs. See [features.md](features.md#incremental-updates) for details.

## Pipeline Summary

```
Query
  ├─→ BM25 search (Tantivy)        ──→ ranked candidates
  └─→ Vector search (flat SIMD; sqlite-vec fallback) ──→ ranked candidates
                                          │
                                    RRF fusion
                                          │
                             query-aware ranking and expansion
                                          │
                                    top candidates
                                          │
                            optional cross-encoder rerank
                                          │
                                    final ranked results
```

| Stage | What it does | Why it matters |
|-------|-------------|----------------|
| Tree-sitter parsing | Extracts symbols as chunks | Results map to real code boundaries |
| BM25 | Exact keyword matching | Catches identifiers, fast |
| Vector search | Semantic similarity | Catches conceptual matches |
| RRF fusion | Merges both result lists | Covers both exact and semantic |
| Query-aware ranking | Applies deterministic priors and candidate shaping | Fixes exact-match, config, and cross-file failure modes |
| Optional cross-encoder rerank | Joint query-document scoring | Adds a second relevance signal when enabled |
