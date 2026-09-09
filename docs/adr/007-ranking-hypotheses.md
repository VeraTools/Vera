# ADR 007: Issue #196 Remaining Hypotheses: Multiplicative Path Penalty, Candidate-Pool Multiplier, 750-Char Chunks (DEFAULT OFF)

Status: implemented behind toggleable knobs, DEFAULT OFF; measurement owned by `issue-196-hypotheses-measurement` (separate PR)

## Context

After ADR 006 / #239 (three ranking signals shipped toggleable, default `true` with env overrides), three remaining #196 hypotheses remain untested. The Semble full-suite gap (Vera 0.8451 vs 0.8514 nDCG@10, 0.74% relative) concentrates in `intent` and `cross_file` with worst-case repos (nvm/zig/rails/redis/nlohmann-json/redux/axum). Failure taxonomy: recall misses (correct file never reaches ranking), ranking misses (correct file in pool but outranked by docs/tests/fixtures), noise. No new benchmark tuning may occur in the implementation PR; mechanism-first rationales only. This ADR follows the #239 / ADR-006 pattern exactly.

## Decision

Implement the three remaining #196 hypotheses as individually toggleable config knobs + env overrides, **DEFAULT OFF**, with mechanism-first rationales, strictly separated from measurement (no result JSONs in this PR):

- `retrieval.ranking_multiplicative_path_penalty` (`VERA_RANKING_MULTIPLICATIVE_PATH_PENALTY`, aliases `VERA_RANKING_PATH_PENALTY`): bool, default `false` (OFF), factor `0.3×` when enabled
- `retrieval.ranking_candidate_pool_multiplier` (`VERA_RANKING_CANDIDATE_POOL_MULTIPLIER`, aliases `VERA_RANKING_POOL_MULTIPLIER` / `VERA_RANKING_CANDIDATE_POOL_SIZE_MULTIPLIER`, numeric `5` also accepted as truthy): bool, default `false` (OFF), factor `5×` `top_k` when enabled
- `indexing.chunk_max_chars` (`VERA_INDEXING_CHUNK_MAX_CHARS`, aliases `VERA_MAX_CHUNK_CHARS` / `VERA_CHUNK_MAX_CHARS` / `VERA_INDEXING_MAX_CHUNK_CHARS`, serde alias `max_chunk_chars`): `usize`, default `0` (OFF, 0 means disabled), `750` when enabled

Naming follows the existing `retrieval.ranking_*` / `indexing.max_chunk_*` / `VERA_RANKING_*` / `VERA_INDEXING_*` conventions. `VeraConfig` threads through `score_prior` / `score_pool` / `apply_query_ranking` / `compute_fetch_limit_with_config` and through `parsing::chunker` plus `indexing_config` identity gates. Wrappers preserve compatibility via `VeraConfig::default()` (which already respects env overrides). Env overrides are authoritative over file values.

Implementation lives on branch `feat/issue196-hypotheses` (worktree) and merges only on stable + MSRV (1.88) green at the exact head.

## Mechanism Rationale Per Hypothesis

Parameters are carried from mechanism reasoning only; no benchmark tuning in this PR.

### 1. Multiplicative path penalties for tests/compat/examples at ~0.3×

**Current state:** `retrieval/ranking/score.rs` penalizes `tests/` / `compat` / `examples` additively: `Test -0.95`, `compat -0.65`, `Example|Bench -0.55`, `Docs -0.55/-0.95`, `Archive -0.85`, `Generated -0.95`, etc., plus `definition_site_role_blocked` gating for definition boosts. Additive penalties subtract a constant regardless of retrieval confidence.

**Hypothesis:** boilerplate, fixture, and example directories (`tests/`, `testdata/`, `__tests__/`, `spec/`, `fixture(s)`, `example(s)`, `demo(s)`, `bench(es)`, `benchmark(s)`, `sample(s)`, `compat`/`legacy`/`shim`/`polyfill`) contain keyword-dense non-implementation content (copied snippets, mocked configs, usage samples) that matches many NL queries incidentally. A **multiplicative** `0.3×` penalty demotes these candidates **proportionally to retrieval confidence**, preserving ordering among non-penalized candidates (all keep their relative scores) while consistently demoting penalized ones by the same factor. Two equally-scoring candidates, `src/session.rs` vs `tests/fixtures/session.rs`, retain `src/` > `tests/` by factor 3.3, rather than by a fixed subtract that may be washed out at high scores or dominate at low scores.

**Gating:** must respect the existing boost-directory gating used by `ranking_definition_boost` (`definition_site_role_blocked` / `wants_test_paths` / `wants_example_paths` / `wants_compat_paths`). When the query explicitly asks for tests/compat/examples (e.g. “legacy compat session”), the penalty does not fire. This mirrors the definition boost's directory checks so explicit requests are not penalized.

**Implementation:** new `apply_multiplicative_path_penalty` in `score.rs` multiplies `scores[i] *= 0.3` for `ContentClass::Test` / `Example` / `Bench` or `definition_site_role_blocked`-detected test/example dirs, or `is_compat_path`, when the corresponding `wants_*` is false. Called from `score_pool_with_config` after `apply_coherence_boost` / `apply_keyword_path_boost` / `apply_content_symbol_boost`, gated by `ranking_multiplicative_path_penalty_enabled()`. Additive penalties remain unchanged; when the knob is on, both additive and multiplicative apply (distinct mechanisms, additive -0.95 plus ×0.3). When OFF (default), behavior is unchanged.

No weight was tuned: `0.3` is carried from the mechanism hypothesis (strong enough to demote below `src/` for equally-scoring candidates, `1/0.3 ≈ 3.3`, without collapsing test fixtures to zero) and is not derived from ground-truth or score chasing.

### 2. Larger candidate-pool multiplier: the 5× `top_k` hypothesis

**Current state:** `compute_fetch_limit_with_config` computes `fetch_limit` from filters (`result_limit`, `3×`/`10×`/`12×` for `path_glob`/`exact_paths`) and, when `ranking_recall_pool_expansion` is enabled, a **table-driven** recall expansion: `needs_structural_overfetch` (NL ≥4 words, not path-weighted, no filters) → `8×`, else broad NL → `3×`. Full-suite delta for that signal was **0.54%** (reported in #239). Vector candidate pooling (`candidate_pool`) further bounds the fetch via the KNN cap (4427 flat / KNN selector for extreme limits).

**Hypothesis:** a **bare** `5×` `top_k` pool multiplier gives ranking signals more correct-but-low-ranked files to promote, without the conditional table. Intent and cross-file queries are open-ended; BM25/vector may place the correct file at rank 40 while the fetch limit only keeps 5–20 candidates. Expanding the pool uniformly (not gated on `needs_structural_overfetch`) trades a modest, bounded increase in candidates (5×) for higher recall that ranking can then refine. This is the issue's `5× top_k` hypothesis, now wired as a pool-size knob.

**Non-duplication:** the new knob does **not** duplicate `ranking_recall_pool_expansion`. The existing signal is **table-driven and conditional** (structural 8× gated on ≥4 words and exact query shape; full-suite delta 0.54% documented in #239). The new knob is a **bare 5× `top_k`** applied **uniformly before** the conditional table (`fetch_limit.max(result_limit*5 + 50)`) so both knobs compose without duplicating logic: when both are on, the fetch limit is the max of `5×`, `8×`, and filter-driven expansions; when only the new knob is on, NL structural queries still expand to 5× even if the recall table is off. The ADR explicitly cross-references the 0.54% delta to keep the distinction measurable in the separate measurement PR.

**Implementation:** in `retrieval/search_service.rs` `compute_fetch_limit_with_config`, after filter-driven `fetch_limit` is computed, if `ranking_candidate_pool_multiplier_enabled()` then `fetch_limit = fetch_limit.max(result_limit*5)` (with `+50` minimum headroom, `result_limit*5`). This runs **before** the `ranking_recall_pool_expansion` early return, so the bare multiplier survives even when recall expansion is disabled. `candidate_pool_multiplier_factor()` returns `5` when enabled, `1` otherwise. Gated DEFAULT OFF.

Parameter `5` is carried from the mechanism hypothesis (5× top_k as named in the issue) and not re-tuned to benchmark scores.

### 3. ~750-character chunks: finer embedding locality (touches index format / identity)

**Prior negative (#67):** two larger-window / larger-cap experiments were run and kept as honest negatives with cost columns:

- **window-2048 on jina:** `-0.23%` nDCG for `+88%` index time (stored, hypothesis rejected)
- **2048-byte cap on potion:** `-1.24%` subset / `-0.21%` full nDCG, `+24%` index time, `+15%` storage (kept status quo `512/24576`)

The status quo remains `max_chunk_lines 200` / `max_chunk_bytes 24576` (24 KB, ~6–7K tokens; local embedders see first 512 tokens).

**Hypothesis (opposite direction):** **smaller** `~750` character chunks give **finer embedding locality** than the 24 KB byte cap. Long chunks blur multiple concepts (imports + several functions + trailing prose) into one embedding; a retrieval query matching one concept must compete with noise from the rest of the chunk. Splitting at `~750` chars (line-boundary aware) isolates each concept into its own retrievable unit while preserving symbol coherence (the same `bare_name` + `part_index` shape as the byte path). This is the opposite direction of #67's 2048, so it is expected to have different trade-offs; cost columns (index time, storage) must be reported alongside quality in the measurement PR.

**Index format / identity:** this touches the index format. `chunk_max_chars` is wired through the **content-affecting** `indexing_config` identity keys: the `max_chunk_lines` / `max_chunk_bytes` pattern already exists in `eval/src/lanes.rs` provenance gates and `eval/src/vera_adapter.rs` `indexing_config_matches`. `freshness.rs` stores the full `IndexingConfig` JSON (including `chunk_max_chars`) via `record_index_snapshot`; `vera_adapter.rs` checks `chunk_max_chars` (with `max_chunk_chars` / `max_chunk_characters` aliases) and treats missing as `0` (DEFAULT OFF) for backward compatibility so existing indexes reuse when the knob is off. Throughput-only keys (`batch_size`, `max_concurrent_requests`, `timeout_secs`, etc.) are intentionally **not** part of the identity.

**Gating for byte-identical default:** the chunker change is **gated** behind the knob so default behavior is **byte-identical when off**. `parsing/chunker.rs` adds `split_oversized_chunks_by_chars(chunks, max_chars)` (mirrors `split_oversized_chunks` but uses `char` counts, `0` is a no-op). `parsing/mod.rs` introduces `apply_splits()` which does `split_oversized_chunks` (always) then `split_oversized_chunks_by_chars` only when `config.chunk_max_chars_effective() != 0`. When `chunk_max_chars == 0` (DEFAULT OFF), the char path is not taken, producing identical chunk boundaries to master.

Parameter `750` is carried from the mechanism hypothesis (finer locality than 24 KB, smaller than #67's 2048) and not tuned to benchmark scores.

## What Was NOT Done (Implementation / Measurement Separation)

Per VAL-ISSUE-025 / VAL-ISSUE-026:

- No `benchmarks/results/*.json` was committed in this PR; no measured quality claim (subset, full-suite, or independent) appears here.
- No with/without ablation was executed to tune knobs; every new knob defaults **OFF** so the signals-off reference (post-m8 rebaseline, `issue-196-measurement` will evaluate hypotheses against that baseline on the 320-task subset and 180-task independent set at named commits with `vera_git_sha` provenance).
- No heuristic constants (0.3, 5, 750) were derived from ground-truth inspection or score-chasing; they are carried from the issue's mechanism reasoning.
- No tuning of existing signals (`KEYWORD_PATH_WEIGHT`, `COVERAGE_WEIGHT`, `FILE_SATURATION`, etc.) occurred.
- The PR is implementation-only; measurement artifacts (result JSONs with `index_time_secs` / `storage_size_bytes` cost columns, and the chunk-arm's explicit cross-reference to #67's `-0.23%` / `+88%` and `-1.24%/ -0.21%` / `+24%` / `+15%` numbers) belong to the separate measurement PR which also satisfies VAL-ISSUE-027 through VAL-ISSUE-030.

## Toggleability and Ablation Cost

All three knobs are individually toggleable via config file and env var (env authoritative over file):

- `retrieval.ranking_multiplicative_path_penalty`: `VERA_RANKING_MULTIPLICATIVE_PATH_PENALTY` (bool); covers `tests/` / `compat` / `examples`; 0.3× multiplicative; respects `wants_*` gating
- `retrieval.ranking_candidate_pool_multiplier`: `VERA_RANKING_CANDIDATE_POOL_MULTIPLIER` (bool, numeric `5` also truthy; aliases `VERA_RANKING_POOL_MULTIPLIER`, `VERA_RANKING_CANDIDATE_POOL_SIZE_MULTIPLIER`); bare 5× `top_k`; interacts with `compute_fetch_limit_with_config`; distinct from the 0.54% recall-pool signal
- `indexing.chunk_max_chars`: `VERA_INDEXING_CHUNK_MAX_CHARS` (aliases `VERA_MAX_CHUNK_CHARS` / `VERA_CHUNK_MAX_CHARS` / `VERA_INDEXING_MAX_CHUNK_CHARS`; serde aliases `max_chunk_chars`); char-budget split; content-affecting identity; DEFAULT OFF gives byte-identical chunking

Tests prove the contracts:

- `config::tests::issue196_hypotheses_default_off`: fresh `VeraConfig::default()` keeps all three OFF (bool false, `chunk_max_chars == 0`); legacy JSON without the keys deserializes to OFF
- `config::tests::issue196_hypotheses_env_flips_independently`: each `VERA_*` env flips only its knob (`penalty` `1` → `0.3×` enabled but pool/chunk stay OFF; `pool` `1`/`5` → `5×` enabled but others stay OFF; `chunk` `750` → char budget enabled but retrieval stays OFF); alias envs also flip
- `parsing::chunker::tests::chunk_max_chars_off_is_byte_identical`: `split_oversized_chunks_by_chars(..., 0)` is identity; default `IndexingConfig` produces identical chunks to the 24 KB path
- `parsing::chunker::tests::chunk_max_chars_on_splits_finer_than_byte_cap`: `750` splits into ≥2 sub-chunks each ≤750 chars, with `bare_name` + `part_index` shape mirroring the byte path, and the suffix is verbatim (`display_symbol_name` is single source)
- `retrieval::ranking::tests::multiplicative_path_penalty_is_toggleable`: disabled keeps base rank (penalty OFF), enabled demotes `tests/fixtures/` below `src/` even when `tests/` starts first, and respects `wants_test_paths` gating (`"tests"` in query → no demotion)
- `retrieval::ranking::tests::multiplicative_path_penalty_is_multiplicative_not_additive`: direct `apply_multiplicative_path_penalty` multiplies `10.0 → 3.0`, `7.0 → 2.1` (compat), `4.0 → 1.2` (example), not a subtract
- `retrieval::search_service::tests::candidate_pool_multiplier_is_toggleable_and_independent`: DEFAULT OFF leaves `compute_fetch_limit("Config", 20)` at `20`; bare `5×` inflates even identifier queries to `≥100`; NL similarly `≥100`; both knobs compose; filter-driven `path_glob` still maximal

## Ground-Truth Non-Inspection Statement

No rule, constant, or weight in this PR was derived from inspecting benchmark ground-truth answers or expected file paths. Hypotheses were formed from the failure taxonomy (recall vs ranking vs noise) and from general code-search intuition (module labels, definition anchors, open-ended query recall, chunk locality vs #67). No hardcoded repo or file-path lists matching benchmark expectations were introduced.

## Consequences

- New unit tests pin toggle + byte-identical + multiplicative behavior; existing ranking/chunking tests continue to pass with defaults.
- No benchmark-derived tuning debt is introduced; measurement can proceed independently and per-hypothesis dual-set (subset + independent) ablations with cost columns can be attributed cleanly before any default flip (gates per VAL-ISSUE-027 through VAL-ISSUE-030, and the 0.5% bar).
- Benchmark-integrity position: **implementation PR contains no result JSONs and no measured quality claims; parameters are from mechanism reasoning only** (VAL-ISSUE-026). Measurement PR will name its evaluated `vera_git_sha` and report `index_time_secs` + `storage_size_bytes` for the chunk arm with explicit #67 prior cross-reference.

## References

- #196 gap and failure taxonomy; #239 / ADR 006 (toggleable-signal pattern, `VeraConfig` threading, `compute_fetch_limit` gating, `Lanes` provenance `VERA_RANKING_*` + `host.cpu_model`)
- #67 prior negatives: `window-2048` / `2048-byte cap` with index-time / storage cost columns
- #243 hardware caveat (7600X3D → 9800X3D warm 9.93 → 16.6 ms p50; rebaseline target `072c725` re-measured same-host)
- Implementation files: `crates/vera-core/src/config.rs` (knobs + `env_bool`/`env_usize` + `enabled()` helpers), `crates/vera-core/src/retrieval/ranking/score.rs` + `mod.rs` (0.3×), `crates/vera-core/src/retrieval/search_service.rs` (5×), `crates/vera-core/src/parsing/chunker.rs` + `mod.rs` (750-char), `eval/src/vera_adapter.rs` + `eval/src/lanes.rs` (identity + provenance)
