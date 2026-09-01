# What's New

Release highlights from v1.0 onward. For the current benchmark tables and methodology, see [benchmarks.md](benchmarks.md). For the full command surface, see [features.md](features.md).

## v1.3.0

### Search correctness

- Split symbols keep their identity. The chunker stores the bare symbol name plus a distinct part index, so `vera structural definitions` finds split symbols by bare name, `vera references` resolves their single call site, `vera dead-code` deduplicates parts by (symbol, file), exact-name augmentation reaches split chunks, JSON carries bare name with part index and text display keeps `(part N)` with single-sourced formatting.
- Filtered vector search above 4,096 candidates no longer truncates. Path, language and exact-path filters reach low-ranked island results that previously returned zero hits on the over-cap fixture, with deterministic ordering, batched hydration beyond 999 SQL parameters, and order preservation. Flat scan is now unbounded, the spurious below-cap truncation warning is gone, and vec0 diagnostics fire only when clamping could lose results with an actionable message naming the backend and remedy.
- The per-query vector-store open is now deduplicated, halving manifest reads per query while preserving stamp-guarded staleness and the LRU-4 memory bound.

### Indexing progress and reuse

- The embedding progress bar is now honest. While parsing is still in progress it shows open-ended work so far with no percentage, then switches to a fixed total at `ParsingDone` and fills monotonically to 100%. Cancellation and mid-run failures no longer imply success, small single-window repos show a fixed total directly, and non-TTY, `--no-progress` and `--json` modes are unchanged.
- Evaluation lanes can reuse a current index when identity gates pass. `reuse_index: true` skips indexing only when the on-disk index matches embedding model name (including `model_aliases` and `VERA_EMBEDDING_MODEL_ALIASES`), document prefix, staleness, embedding dimension, content-affecting indexing config, and format version, with correct size accounting and BM25 never reusing.

### Ranking and retrieval

- Three ranking signals for issue #196 are now toggleable with mechanism-first rationales: filename-stem boost, definition boost, and recall-pool expansion. Each has a config knob and `VERA_RANKING_*` env override, implemented separately from measurement and proven by dual-set ablations on the 320-task subset and 180-task independent set with full-suite confirmation before any quality claim.
- Three additional hypotheses (multiplicative path penalties, candidate-pool multiplier, 750-char chunks) are implemented as default-off knobs with correct index-identity wiring. Dual-set ablations on the 320-task subset and 180-task independent set plus full 1,251-task confirmation showed each below the 0.5% full-suite aggregate bar or with regression, so all three stay default off with honest negatives recorded. The chunk arm cites the prior 2048 window and cap negatives and reports its own index-time and storage cost.
- Reranker protocol now cleanly separates generic (`top_n` / `results`) from Voyage (`top_k` / `data`) with explicit config override over hostname auto-detection, and resilience covers permanent 4xx no-retry, capped `Retry-After` and `X-RateLimit-Reset` waits, cancellation, and graceful degradation.

### Setup and first-run

- `vera setup` ships a Qwen (OpenRouter) preset in the interactive `vera setup` flow with single-key hardening, with `installation.md` and `models.md` updated in the same commits. The first-run flow is streamlined to a single key with auto protocol selection, and `README.md` plus both package READMEs move together for user-facing changes.
- `vera doctor` now tolerates blank env values and probes DirectML more accurately, `vera uninstall` reports shim classification more precisely, and `retrieval.max_output_chars` help correctly shows `0 = unlimited`.

### Evaluation and provenance

- Result JSONs now record host CPU model from `/proc/cpuinfo` and the three `VERA_RANKING_*` env values in `version_info.environment`, so future hardware changes and signals arms are detectable from artifacts alone with a graceful non-Linux fallback.
- `docs/197-profiling.md` now carries a hardware caveat: the 9.93 ms p50 and 73.26 ms p95 numbers were measured on Ryzen 7 7600X3D and are not comparable to post-2026-08-28 measurements on Ryzen 7 9800X3D without the same-host re-baseline (mean 7.88 ms p50 and 60.88 ms p95 on three `072c725` runs).

### Compatibility

No breaking CLI changes. Existing indexes open and search as before. New indexes carry split-symbol part indices and improved filtered-vector behavior, and the evaluation harness remains backward compatible with old result JSONs.

## v1.2.0

### Query latency

- The ranked exact-match augmentation no longer re-scans the chunk metadata for the indexed file list on every query. The list is cached per open index and revalidated with a database-plus-WAL stamp, so edits still invalidate it immediately. On the zig repository (about 336,000 chunks) the augmentation stage dropped from p50 18.4 ms to 1.4 ms and total warm p50 from 31.1 ms to 14.1 ms. Full-suite p95 improved from 73.3 ms to 64.5 ms.

### Indexing throughput

- Full builds now overlap the parse, embed, and store stages: parsing runs one window ahead on a worker, and a dedicated store thread owns all staging writes behind a bounded channel, so the embedding thread never waits on SQLite or Tantivy. Tantivy writes go through a single bulk writer with one final commit, and fresh vector builds use a two-statement insert path.
- Full Semble corpus indexing dropped from 213.5 s to about 137 s (-36%); the zig repository alone went from 62.7 s to 36.7 s. Peak RSS stays within the windowed budget (533 MB on zig). Search output over the resulting index was verified byte-identical to the previous pipeline.

### Ranking

- The content-based symbol definition boost no longer treats fixture definitions in test, example, and bench trees as definition sites unless the query asks for them. A `TaskRegistry` lookup previously ranked a test fixture above the real class; the fix keys off directory components and conventional test-file names, so first-class modules like click's `testing.py` keep their boost.

### CI and supply chain

- Pull requests now run a CI workflow: rustfmt, clippy with warnings as errors, the full test suite, cargo deny, cargo machete, and cargo audit, plus a pinned MSRV check job.
- `cargo deny` now fails on unsound advisories from transitive dependencies, not just workspace crates. The one currently known instance (RUSTSEC-2026-0253, `lru` via tantivy) is documented in `deny.toml` with a reachability analysis and a removal trigger.
- The declared MSRV is corrected to 1.88: the tree has used let-chains for some time, so 1.86 could never build it. The lockfile pins `ordered-float` 5.0.0 so no locked dependency requires more than 1.88.

### Indexing memory and crash recovery

- Full indexing now processes bounded parse, embed, and store windows instead of holding every chunk and embedding for the repository in memory. On a 638 MB corpus with 27,252 chunks, measured peak RSS dropped from 2,746 MB to 640 MB.
- Full builds publish through a `.vera.build` staging directory and keep the previous index in `.vera.old` until the swap completes. Startup removes stale staging directories and restores the previous index when a crash interrupted publication.
- Discovery and file watchers always exclude `.vera.build` and `.vera.old`, even when default excludes are disabled.
- The windowed build was verified byte-identical to the previous single-pass build on curl: 27,252 chunks and 4,256 file hashes matched exactly.

### Search and storage

- Filtered BM25 search now hydrates the likely result head one row at a time and pages only the rejection tail. Interleaved A/B probes kept unfiltered p50 at parity while reducing filtered p95 from 55.1 ms to 51.8 ms.
- `vera stats`, call-graph queries, and update staleness checks can open an existing index read-only with schema validation instead of creating or mutating an index on read paths.
- Filtered overview statistics use SQL aggregate queries with bounded parameter batches instead of hydrating every row.
- Vector counts now come from stored chunks, and vector deletions propagate so stale mappings do not inflate index statistics.

### Model and API compatibility

- API embedding providers support document prefixes through `EMBEDDING_DOCUMENT_PREFIX`, with model-ID auto-detection when the variable is unset.
- The indexed document-prefix identity is stored in index metadata. If the active prefix changes, search falls back to BM25-only and update asks for a re-index instead of mixing vectors from different input formats.
- ROCm GPU detection accepts both `rocm-smi` CSV layouts, derives free VRAM when only total and used columns exist, uses a stable GPU fingerprint, and handles near-zero free VRAM conservatively.

### Parsing, CLI, MCP, and serve fixes

- Markdown chunking respects code fences, Python inheritance extraction is more accurate, and archive-backed token scoping no longer leaks across files.
- `vera doctor` no longer treats intentionally blank environment values as missing, `vera uninstall` reports partial failures accurately, and `vera agent install` refuses to prompt on a non-interactive terminal.
- Update checks compare full semantic versions, helper output honors `max_output_chars` in JSON and raw modes, and CLI state writes use a plain atomic rename.
- MCP watchers are scoped per working directory, coalesce repeated updates, and ignore index staging writes. `vera serve` caps inbound frame size.

### Compatibility

Existing v1.1.0 indexes still open. If an index predates an active API document prefix, Vera deliberately disables the vector side for that index until it is rebuilt rather than mixing embedding spaces. There are no breaking CLI changes.

## v1.1.0

### Search quality and model choice

- The default local embedding model is `minishlab/potion-code-16M-v2`, pinned to revision `e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b`. It runs locally on CPU without ONNX Runtime or a GPU.
- The final Semble comparison used the same model, harness, and graded metric contract for both tools. Vera scored `0.8441` nDCG@10 versus Semble's `0.8514` on the full 1,251-task suite, `0.8540` versus `0.8494` on the 320-task subset, and `0.7644` versus `0.7655` on the independent contamination-check set.
- Deterministic ranking gained file-coherence, keyword-path, content-coverage, definition, and concept-pool signals through dual-set ablations. Signals that did not clear the keep threshold were dropped.
- Cross-encoder reranking remains opt-in. The dual-set screening found no tested cross-encoder above the no-reranker baseline; `mxbai-rerank-xsmall-v1` is documented as the recommended local override when reranking is needed.

### Query performance

- Vector search uses an exact flat SIMD scan over memory-mapped vectors, with sqlite-vec retained as the write path and fallback. Query p50 dropped from 15.1 ms to 9.9 ms on the Semble comparison.
- On the largest tested repository, zig with about 336,000 chunks, the vector stage dropped from roughly 460 ms to 21 ms per query.
- Search stores are reused across queries, reranker construction is lazy, and local rerank batches are sorted by length to reduce padding waste.

### Local model supply chain

- Local model downloads can be pinned to immutable Hugging Face revisions. Built-in assets carry commit pins and compiled-in SHA-256 verification.
- Custom downloads persist digest sidecars, ORT archives are verified before extraction, and multi-shard ONNX external weights are supported.
- The local reranker can be replaced through `LOCAL_RERANKER_*` environment variables.
- Source reads are descriptor-relative through `cap-std`, reducing path traversal risk in repository file handling.

### Indexing, parsing, and MCP reliability

- RST include fan-out is bounded and cached, preventing include graphs from exploding index work.
- Jina v5 uses last-token pooling plus its Query and Document prefixes. Lua function-valued assignments, Bash case arms, JSX call sites, nested Rust modules, and container-member boundaries are extracted more accurately.
- Update publication is cancellation-safe, parallelized where safe, and writes freshness metadata only after the stores are rebuilt.
- MCP multi-query results are fused instead of concatenated, stale-index warnings are surfaced, and the watcher reuses its runtime and provider across debounce cycles.
- `vera serve` caches one model per slot and keeps in-flight loads alive when the initiating client disconnects.

### Platform and release engineering

- The dependency stack moved to Tantivy 0.26.1 and MSRV 1.86.
- Docker publishing covers CPU, CUDA, ROCm, and OpenVINO images. The release workflow packages the same Linux binary into the images instead of compiling a second time.
- The ROCm base image is mirrored by digest into GHCR, and failed Docker publishes can retry through the runner daemon instead of a persistent builder cache.

## v1.0.1

- `vera index` gained `--no-progress`, and progress output is gated on an interactive stderr.
- Flag-driven setup works without a terminal and validates blank or partial API credential sets.
- Agent configuration sync is quiet and preserves user edits outside Vera's managed markers.
- Upgrade reporting verifies the applied binary version instead of trusting installer exit codes.
- CoreML reranking uses the supported CPU path instead of the problematic fp16 path.
- sqlite-vec KNN requests are clamped to the backend limit without silently disabling vector search.
- Release and Docker workflows moved to Blacksmith runners.

## v1.0

Vera 1.0 is the feature-complete milestone. The hybrid search pipeline, code-intelligence commands, agent integrations, and local inference backends are all in place, and a wave of community PRs and user-reported fixes hardened them along the way.

### Search quality

The v1.0.0 release candidate was measured on the full 1,251-task Semble v0.5.5 snapshot across 63 repositories, using hybrid BM25 and vector retrieval, RRF fusion, and a local ONNX cross-encoder reranker on the `vera-cuda` lane. These numbers use Vera's graded metric contract; [benchmark provenance](benchmarks.md#provenance) explains how it differs from Semble's published metric.

| Metric | v1.0.0-rc |
|--------|-----------|
| nDCG@10 | `0.7327` |
| Recall@1 | `0.5476` |
| Recall@5 | `0.8144` |
| Mean search latency | `2421 ms` |

The mean latency is reranker-dominated. The BM25-only scoped-filter reference lane is about `54 ms` p95, so the two numbers describe different pipeline costs.

The gains came from a reworked default retrieval pipeline: BM25 stemming and identifier-aware tokenization, stronger definition ranking, concept-to-filename augmentation, adaptive RRF weighting, file saturation decay, and parallel hybrid retrieval, plus a fix for scoped BM25 candidate starvation. Release ablations then determined which further ranking changes belonged in the default pipeline:

| Change | Release decision |
|--------|------------------|
| C1: rerank no-surplus skip | Shipped as a latency guard. Vera skips reranking when the fused pool has no surplus over the requested result limit, with a `-0.0002` nDCG delta and `-2.4 ms` mean latency effect. |
| C2: rerank path-glob searches | Shipped. Path-scoped searches remain eligible for reranking, improving full-suite nDCG by `+0.0113`. |
| Structural graph augmentation | Merged as an experimental opt-in under `VERA_GRAPH_AUGMENT=1`. It adds bounded caller and implementation chunks to the rerank pool, gaining `+0.0047` nDCG at roughly `+83%` mean latency. It is off by default because the latency cost outweighs the gain and Recall@5 did not move. |

### Agent-level benchmark

The benchmark used 10 cross-file Flask questions, fresh agents, and two A/B arms: `with-vera` had a local index and project skill installed, while `control` had the Vera CLI blocked by a shim that exits 127. A judge graded answers blind against a verified answer key.

| Tested model | With Vera | Control | Observed result |
|--------------|-----------|---------|-----------------|
| `claude-opus-5` | `10.0/10` | `10.0/10` | Quality parity and efficiency parity on this run |
| `kimi-k3` | `10.0/10` | `9.9/10` | With Vera used 17% fewer input tokens: `230.6k` versus `278.0k` |

This is a small workload signal, not a general performance claim. The question set, reproduction commands, raw measurements, and limitations are in the [agent benchmark README](../benchmarks/agent-bench/README.md).

### Code intelligence and agent workflows

- `vera structural` runs agent-oriented structural search intents: `definitions`, `env`, `routes`, `sql`, and `impls`. Implementations, conformances, inheritance, and Rust trait relations are indexed explicitly, so `impls` answers from the index instead of text matching.
- Git-aware search scopes restrict queries to a diff: `--changed` for working-tree changes, `--since <rev>` for files changed since a revision, and `--base <rev>` for files changed since the merge base with a revision. They work with `vera search`, `vera grep`, `vera overview`, and `vera references`.
- `vera explain-path <path>` reports why a file is or is not indexed.
- Stale-index detection warns when the index no longer matches the working tree, and `vera stats` reports persisted parser-health metrics, so damaged or partial indexes are visible instead of silently degrading results.
- Repeat `--path` to search several path patterns with OR semantics. Other filters still combine with AND semantics, so `--lang`, `--type`, and `--scope` continue to narrow the combined path match.
- Function and method symbol types are aliases, so `--type function` and `--type method` can be used interchangeably when selecting callable symbols.

### MCP server

The MCP server grew from four tools to seven: `structural_search`, `find_references`, and `explain_path` join the existing tools, and the search tools gained the same path filters and git scopes as the CLI.

### Models and local backends

- `vera setup --potion-code` selects the default `minishlab/potion-code-16M-v2` static embedding model. It runs locally on CPU on any supported machine; no GPU or ONNX Runtime needed.
- Loaded embedding and reranker models are cached and reused across repeated searches, multi-query and deep search, and MCP calls instead of being reloaded per query.
- Voyage AI rerank endpoints are supported, including the `rerank-2` API format.
- `VERA_EMBEDDING_MODEL_ALIASES` lets compatible deployment names share an index after the normal dimension check. Alias groups are separated with semicolons and names within a group with commas.
- Local mode checks ONNX model integrity and gives `vera doctor` and `vera repair` enough information to recover damaged or incomplete assets.
- CoreML embedding batches scale with available Apple Silicon unified memory.
- OpenAI-compatible embedding providers can report a token-limit error and have the oversized input truncated and retried automatically.

### HTTP inference server

`vera serve` starts a local HTTP inference server exposing OpenAI-compatible embeddings (`POST /v1/embeddings`), Cohere/Jina-compatible reranking (`POST /v1/rerank`), and a health endpoint, with bearer-token authentication, `--host`, `--port`, and `--idle-timeout` model caching. Requests are cancelled when clients disconnect, empty keys cannot bypass authentication, and internal errors are redacted from responses.

### Agent configuration sync

`vera agent install` writes the skill files for supported agent clients. Managed synchronization refreshes Vera's own marked sections in agent configuration files (AGENTS.md, CLAUDE.md, and similar) during normal CLI use and preserves user edits outside the managed markers.

### Indexing and updates

- Index and update commands show phase progress for discovery, parsing, and embedding. Use `--no-progress` or `--json` when a machine-readable or quiet interface is needed (`vera update` at v1.0.0; `vera index` gained `--no-progress` in v1.0.1).
- `vera update . --max-files <N>` bounds the added or modified files processed in one run and reports deferred files for a later update.
- Indexing and update runs are cancellable: interrupting a run stops in-flight embedding work, and bounded remote calls keep abandoned runs from consuming API quota.
- Update failure handling is more atomic: a failed update keeps the previously indexed parse and chunk data instead of dropping both.

### Fixes

- Bare directory patterns such as `--path src/app` now match files below that directory instead of returning no results.
- `--intent` no longer leaks its intent prefix into BM25 field queries, fixing the BM25 error path for intent searches.
- Files with invalid UTF-8 are read lossily during indexing and retrieval, so they are no longer silently skipped.
- Unicode file paths are handled correctly by path glob matching and grep byte-to-character offsets.
- Embedding cache keys include the model namespace, preventing cached vectors from one model from being reused for another model.
- TypeScript and JavaScript class-method indexing, Java enum-method indexing, TypeScript interface methods, and generic type-relation handling were corrected, improving `references` and structural results.
- `crossbeam-epoch` was updated to address the tracked RustSec advisory.

Many of these fixes came from community PRs and Discord user reports; thank you to everyone who filed and fixed them.

### Upgrade notes

Source builds enforce the vendored tree-sitter grammar bootstrap step: if the grammar sources are missing, the build fails early and prints the bootstrap command instead of failing later at link time.

```bash
bash scripts/bootstrap-vendored-grammars.sh
cargo build --release
```

The script downloads the four grammar sources that are not tracked in git. Prebuilt package installs do not need this source-build step. See the [installation guide](installation.md) for the complete source-build instructions.

There are no breaking CLI changes in v1.0. Existing search, indexing, update, agent, and MCP workflows keep their command names and flags.

Existing v1.0 indexes open and search as before. Reindex with `vera index .` to pick up the new stemmed and identifier-tokenized BM25 fields, caller and type-relation data, and index-health metrics. Until then, search quality on old indexes stays at pre-v1 levels.

### Distribution

Vera is available as prebuilt binaries on GitHub Releases, the `@vera-ai/cli` npm package, the `vera-ai` PyPI package, and Docker images on `ghcr.io/veratools/vera` (`cpu`, `cuda`, `rocm`, and since v1.0.1 `openvino`). See the [installation guide](installation.md) and the [Docker guide](docker.md).
