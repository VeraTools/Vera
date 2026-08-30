# Nemotron Reranker Acceptance Bar — Pre-Registered (2026-08-29)

**Date (UTC):** 2026-08-29T07:38:00Z
**Vera commit:** `6caae4c1f01da99fe89619b119c6dbd4268f29b7` (master)
**Semble subset task revision:** `60bac08d1e9d71f6d9208f6c15fcc7c354fab85d2eca5a95b6da9b6dc6608190` (320 tasks, Semble 0.5.5 commit 9218491)
**Corpus:** `eval/semble-subset-corpus.toml` (16 repos)
**Author:** reranker-screening worker (m5-presets-screening)
**Status:** PRE-REGISTERED — written BEFORE any screening result exists for this evaluation window. File mtime and git commit timestamp are evidence of predating screening lanes.

## Purpose

This document pre-registers the acceptance bar for the **Nemotron free reranker** (`nvidia/llama-nemotron-rerank-vl-1b-v2:free`) paired with the **Nemotron free embedding** (`nvidia/nemotron-3-embed-1b:free`). Per `features.json:reranker-screening` and `validation-contract VAL-SCREEN-001/010` and `VAL-SETUP-010`, the subsequent Nemotron include/reject decision **MUST** cite this bar and the measured shortfall/win. No reranker-related default change ships without full-suite (1,251) + independent-set (180) evidence.

The bar is evaluated on the **320-task subset** primary slice. A credible finalist may be promoted to full-suite validation.

## Protocol Constraints for Comparison (identical retrieval inputs)

All deltas below are measured between **same Vera commit, same embedding model, same candidate depth, same document format/truncation, same task revision** — affirmatively recorded per artifact via `version_info.vera_git_sha`, `version_info.task_set`, `version_info.config` (`rerank_candidates=50`, `reranker_max_doc_chars=4800`, `max_rerank_batch`, `reranker_protocol`, `document_prefix`), and `version_info.lane`. Where the harness cannot emit a field, the run-note log carries the affirmation.

- Embedding model: `nvidia/nemotron-3-embed-1b:free` (API via OpenRouter `https://openrouter.ai/api/v1`)
- Reranker candidates: 50 (default, unchanged until ablation evidence)
- Document character budget: 4800 (default)
- Rereranker batch: 20 (harness default at 6caae4c)
- Task set: 320 (harrier-screening subset)
- Free endpoints see ONLY public benchmark corpus (Semble repos, `.bench/semble-repos`) — no private user code.

## Acceptance Criteria

**INCLUDE** (ship as default / preset candidate) only if **ALL** of 1–4 pass. Otherwise **REJECT** (do not change defaults; keep existing potion/Vera embeddings as defaults).

### 1) Primary retrieval win — nDCG@10

- `nDCG( Nemotron embed + Nemotron rerank ) - nDCG( Nemotron embed alone ) >= +0.007` absolute on the 320-task subset
- Measured under identical inputs as above; no other config delta between paired lanes except `rerank:true/false` + reranker env (`RERANKER_MODEL_ID`, `RERANKER_MODEL_BASE_URL`, `RERANKER_MODEL_API_KEY`, `VERA_RERANK_RATE_LIMIT_WAIT_SECS`).

Rationale: +0.007 nDCG exceeds noise on the 320 slice and is the minimum to justify reranker latency/cost; previous potion local reranker work used similar 0.005–0.01 bands.

### 2) Secondary retrieval non-regression and meaningful R@1 gain

- `R@1( rerank ) - R@1( no rerank ) >= +0.010` absolute
- No metric regresses beyond: `R@5` delta >= −0.005, `R@10` delta >= −0.005, `MRR` delta >= −0.005
- Per-category check: no task category (symbol_lookup / intent / cross_file / config / disambiguation) regresses > −0.02 nDCG.

### 3) Latency and reliability ceiling (free-tier execution)

- `p50 total` for rerank lane `<= 2500 ms`; `p95 total` for rerank lane `<= 6000 ms` on the benchmark host (RTX 4080 + API)
- Reranker failure/degrade rate `<= 10 %` of tasks (≤32 / 320 returning unreranked due to 429/timeout/error). Warnings of the form `Warning: reranker unavailable` are counted as failures.
- Measured p50/p95 include embedding + vector/BM25 + rerank overhead. Values are workload-specific but must stay interactive for `vera search` UX.

### 4) Absolute floor — not worse than credible baseline

- `nDCG( Nemotron embed + rerank ) >= 0.800` on the 320-task subset

Reference points (harrier-screening 320-task subset at neighboring commits, same harness, same metric contract `vera-graded-2-1-task-mean-v1`):
- `vera-bm25` 0.7792 @ 6caae4c-ancestors
- `vera-potion` 0.7655 (pre-v2) / potion-v2 local expected ~0.85 on full; local potion at 0.8451 full is the system baseline
- `vera-cuda` 0.7956
- `f2llm-80m-api` 0.7897

The 0.800 floor ensures Nemotron reranked output is not below the current hybrid (cuda) tier despite being free.

### 5) Cost and integrity gates (not reranker-specific but required for decision validity)

- No duplicate full-corpus paid lanes for this decision (320 only for screening).
- Reserve floor $1.863... never breached; every paid call (screening lanes + direct probes + VAL-SETUP-011/VAL-FIRST-002 traffic) appears itemized in `/home/lamim/.cache/vera-away/cost-log.md` and reconciles with overall OpenRouter balance delta.
- Free reranker lanes only touch public Semble corpus (affirmatively logged).

## Decision Template

The post-screening decision record (under `benchmarks/` or `.agents/`, cited by `VAL-SETUP-010`) will state:

> **Decision:** {INCLUDE | REJECT}
> **Bar:** this file `benchmarks/nemotron-acceptance-bar-2026-08-29.md` commit `<sha>` dated 2026-08-29
> **Measured:** list the four numbers above vs thresholds, naming the two paired result JSONs (`harrier-screening-<UTC>Z-subset-nemotron-embed-free.json` and `harrier-screening-<UTC>Z-subset-nemotron-embed-free-rerank.json` or equivalent harrier-screening naming) plus any Qwen reference lanes.
> **Artifacts:** absolute paths to result JSONs, run-note logs under `/home/lamim/.cache/vera-lanes/`, cost-log entries.

If INCLUDE, the record also cites full-suite (1,251) + independent-set (180) follow-up evidence before any default is flipped (ship gate VAL-SCREEN-009).

## Out-of-Scope for This Bar

- Qwen3 embedding (`qwen/qwen3-embedding-8b`) and Qwen reranker (`qwen/qwen3-reranker-8b`) are reference lanes only; their numbers do not decide the Nemotron bar, but provide context on paid reranker headroom.
- Ablations (document format variants, candidate-depth sweep 25/50/100/128/200, prior weight, ordering/fusion, batch 20-vs-0 with failure behavior, task-instruction control, CoREB/zerank) are recorded in the separate ablation register per VAL-SCREEN-008.

## Evidence Preservation

- This file's `mtime` and its git commit timestamp must precede every screening result JSON's `mtime` and every screening result's `timestamp` field inside the JSON.
- Result JSONs follow `benchmarks/results/harrier-screening-<UTC>Z-subset-<lane>.json` with `version_info.vera_git_sha` = committed Vera SHA, `version_info.task_set.task_ids_sha256` = `60bac08d...`, `version_info.config` showing only reranker delta between the paired comparison lanes.
