# ADR 006: Issue #196 Ranking Signals (Implementation Only, No Measurement)

Status: implemented behind toggleable flags; measurement owned by `issue-196-measurement`

## Context

Issue #196 tracks the full-suite Semble quality gap (Vera 0.8451 nDCG@10 vs Semble 0.8514). The gap concentrates in `intent` and `cross_file` categories with worst-case repos (nvm/zig/rails/redis/nlohmann-json/redux/axum). Failure taxonomy from screening (not per-task ground truth): recall misses (correct file never reaches ranking), ranking misses (correct file in pool but outranked by docs/tests), and noise (benchmark tasks where expectation itself is noisy).

Prior screening identified candidate signals (path penalties, definition boosts, file-stem matching, candidate-pool sizing) but required mechanism-first implementation with no benchmark-derived tuning in the same PR as measurement.

## Decision

Implement three ranking signals in `crates/vera-core` behind individually toggleable config flags, default `true`, with env overrides for cheap ablations:

- `retrieval.ranking_filename_stem_boost` (`VERA_RANKING_FILENAME_STEM_BOOST`)
- `retrieval.ranking_definition_boost` (`VERA_RANKING_DEFINITION_BOOST`)
- `retrieval.ranking_recall_pool_expansion` (`VERA_RANKING_RECALL_POOL_EXPANSION`)

Config plumbing threads `VeraConfig` through `score_prior`, `score_pool`, `apply_query_ranking`, `compute_fetch_limit`, and exact-match augmentation. Wrappers preserve backward-compatible signatures using `VeraConfig::default()` (which already respects env overrides).

## Mechanism Rationale Per Signal

Each rationale is grounded in how developers search code, not in nDCG movement.

### 1. Filename-stem keyword boost (`ranking_filename_stem_boost`)

Existing `apply_keyword_path_boost` is now gated behind this flag.

Rationale: file names and parent directory names are human-chosen module labels. When a developer asks "session renewal and validation flow", a file named `session.rs` or in directory `rendering/` is more likely to implement that concept than a keyword-dense README or test helper that mentions "session" incidentally. The pool-relative boost rewards files whose stem matches a query keyword (≥3 chars, exact or prefix, 5% match ratio threshold) proportional to the pool's best score, so strong retrieval confidence preserves the signal while weak pools do not over-boost. This targets ranking misses where the correct source file was in the pool but buried under docs/tests.

No heuristic constants were retuned: weight `KEYWORD_PATH_WEIGHT` and match ratio logic are unchanged; the signal only gains a toggle.

### 2. Definition-content boost (`ranking_definition_boost`)

Gates both NL definition-boost blocks in `score_prior`:

- metadata-based definition boost (symbol name overlaps query keywords, long-keyword ≥5 chars, overlap ratio, 1.0–1.5x tier when file stem aligns with symbol)
- content-based definition boost (`content_defines_query_keyword` / `content_defines_symbol`)

Rationale: definitions are canonical anchors. Developers searching for a concept ("CliRunner", "TaskRegistry", "session") usually want the definition site (`class Session`, `struct Types`) rather than incidental mentions, fixture data in `tests/`, or usage samples in `bench/`. The boost is directory-gated (blocked for `tests/`, `examples/`, `bench/` unless query explicitly wants them) and stem-aligned (stronger when file name matches the symbol, e.g. `CliRunner` in `testing.py` still counts because `src/click/testing.py` is a first-class module, not a test fixture). This targets ranking misses where incidental docs/test hits outrank the definition.

No constants were tuned to benchmark ground truth: weights (1.0, 1.5, 0.3) and the 5-char keyword filter are unchanged from prior implementation; the signal only gains a toggle.

### 3. Recall-pool expansion (`ranking_recall_pool_expansion`)

Gates structural and NL overfetch in `compute_fetch_limit_with_config`:

- filter-driven expansion (always on, even when this flag is off): `path_glob` or non-empty filters inflate the pool 3× or 10× to survive post-retrieval filtering.
- recall-driven expansion (gated): broad NL queries (≥4 words), `file type` / `how are ...` patterns, and structural queries expand up to 8× so low-ranking but correct files enter ranking at all.

Rationale: intent and cross-file queries are open-ended; the correct answer may be a low-BM25 file that only becomes rankable after the query-aware signals are applied. A tight fetch limit prunes it before ranking can promote it. Expanding the pool for recall-oriented query shapes (not for exact identifiers, which are already handled by exact-match augmentation) trades bounded extra BM25/vector work for the chance that ranking signals see the right file. This targets recall misses, the dominant failure mode for `intent` and `cross_file` categories.

Parameter `result_limit * 8` and the NL word-count thresholds (≥4) are carried unchanged from the pre-existing `compute_fetch_limit` heuristic; they were chosen from mechanism reasoning (open-ended queries need headroom) and are not re-tuned to benchmark scores in this PR.

## What Was NOT Done (Implementation/Measurement Separation)

This PR is implementation only. The following are explicitly out of scope and owned by `issue-196-measurement`:

- No benchmark runs (320-task subset, 1,251-task full suite, or 180-task independent contamination set) were executed to tune or select signals.
- No ablation with/without measurements were recorded here; every signal defaults `true` so existing behavior is preserved, and env flags allow the measurement PR to run cheap ablations without code changes.
- No signal parameters (weights, thresholds, pool multipliers) were derived from ground-truth inspection or score-chasing; they are carried from the pre-existing ranking code or from the mechanism reasoning above.
- Result JSONs, corpus downloads, or eval harness changes do not appear in this PR.

## Ground-Truth Non-Inspection Statement

Implementation notes state the failure-category source (recall vs ranking vs noise) without per-task ground-truth citation. No rule, constant, or weight in this PR was derived from inspecting benchmark ground-truth answers or expected file paths. Hypotheses were formed from the failure taxonomy (recall misses for `intent`/`cross_file`, ranking misses where docs/tests outrank source, and known noise categories) and from general code-search intuition (module labels, definition anchors, open-ended query recall). No hardcoded path lists matching benchmark expectations were introduced.

## Toggleability and Ablation Cost

All three flags are individually toggleable via config file (`retrieval.ranking_*`) and env var (`VERA_RANKING_*` with `1/0 true/false yes/no on/off`):

- Disabling a flag via `VERA_RANKING_X=0` flips the in-process behavior without recompilation, enabling the measurement lane to run with/without arms cheaply.
- Env overrides are authoritative over file values for these flags (file → env precedence), matching the harness need for per-run ablation without editing the config.
- Legacy configs missing the flags deserialize to `true` for backward compatibility (no behavior change for existing indexes).
- Unit tests in `config.rs`, `retrieval/ranking/tests.rs`, and `retrieval/search_service.rs` verify round-trip, legacy defaults, env precedence, and that disabling each signal preserves base retrieval order while disabling the targeted boost.

## Consequences

- New unit tests pin the toggle behavior; existing ranking tests continue to pass with defaults.
- No benchmark-derived tuning debt is introduced; measurement can proceed independently and per-signal ablations can be attributed cleanly.
- The ADR and PR body together satisfy VAL-ISSUE-001 (mechanism rationale precedes every signal) and VAL-ISSUE-010 (implementation/measurement separated).
