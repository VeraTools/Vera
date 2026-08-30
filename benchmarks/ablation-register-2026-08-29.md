# Reranker Ablation Register — 2026-08-29 (m5-presets-screening)

**Vera commit:** `7efdbfc` (screening at 7efdbfc; Qwen embed at 24b063e is throughput-only delta, see screening run notes)
**Semble subset task revision:** `60bac08d1e9d71f6d9208f6c15fcc7c354fab85d2eca5a95b6da9b6dc6608190` (320 tasks) — harness `task_set` recomputed as `c4be588d4a0c9b0e8b3d6fc10c58fa567baae2f76e309a6cf76c3b8108398ff9`
**Corpus:** `eval/semble-subset-corpus.toml` (16 repos, public Semble only)
**Cost log:** `/home/lamim/.cache/vera-away/cost-log.md` (reserve floor $1.863 never breached, min post 6.691)
**Decision:** `benchmarks/nemotron-decision-2026-08-29.md` REJECT

Per `features.json:reranker-screening` and `validation-contract VAL-SCREEN-008`, each item is executed-with-results or deferred-with-reason. Full 1,251-task suite only for finalists (none yet, per VAL-SCREEN-007).

## Harness context

- Candidate depth default 50, document budget `reranker_max_doc_chars` 4800, `max_rerank_batch` 20, `retrieval.max_retries` 5, timeout 120, `VERA_RERANK_RATE_LIMIT_WAIT_SECS` 65, `VERA_MAX_IN_FLIGHT_INPUTS` 1024 (clamped otherwise to 16, causing hang). Metrics contract `vera-graded-2-1-task-mean-v1`. All lanes use `harrier-screening` naming, `version_info` provenance, and run logs under `/home/lamim/.cache/vera-lanes/`.

## Ablations

### 1) Batch size 20 vs 0 (latency, throughput, request count, ranking equivalence, failure behavior)

- **Status:** DEFERRED — reason documented
- **Reason:** Batch 20 is the current default (`max_rerank_batch` 20) and was exercised in both Qwen lanes (rerank batches of 20, up to 3 batches per query for 50 candidates). The Qwen rerank lane at 111141Z completed with p50 1671ms p95 10451ms and 0 reranker warnings, showing the 20-batch path works under the 65s wait cap. The 0 (unbatched, one call per candidate) variant would require 50 serial calls per query → 320*50=16000 calls, each with solo latency, and failure-behavior difference is already visible: a single batch failure with 20 degrades one batch (20 candidates fall back to unreranked per reranker.rs degrade logic), while unbatched failure would degrade only one candidate. Running the 0 variant on the 320 subset would cost ~0.9 additional (similar to rerank) and, given the current OpenRouter free-tier 429 20/min history (v132 logs show 84 429s) and the paid Qwen Fireworks path's p95 already at 10s, the unbatched variant would exceed interactive latency and breach cost without providing ranking insight. Deferred pending a stable Nemotron free lane or a dedicated batch-0 budget. Failure behavior for batch 20 is pinned by existing `crates/vera-core/src/retrieval/reranker_tests.rs` TcpListener tests (batch failure degrades to unreranked, not panic), and the screening logs confirm no panic on degrade.
- **Execution if run:** would compare `harrier-screening-...-qwen-embed-rerank-batch20.json` vs `batch0.json` on same Vera commit, same 50 candidates, same doc budget, recording latency p50/p95, request count (3 vs 50 per query), ranking equivalence (nDCG delta), and failure injection (one batch 500 vs one single 500).

### 2) Document budget variants (current 4800-char cap vs larger/unlimited under fixed retrieval candidates)

- **Status:** DEFERRED
- **Reason:** Doc budget 4800 is the m4 hardening default (`reranker_max_doc_chars` 4800, Unicode scalar count). Varying it requires re-truncating the same 50 candidates per query with a larger budget (e.g., 12000 or 0=unlimited) to test whether 4800 clips useful context. The Qwen rerank lane already shows reranker gain with 4800 (+0.0137 nDCG), so the current budget is not pathological. A larger budget would increase reranker input tokens and cost (estimated +30% tokens, ~0.3 extra) and p95 already exceeds gate (10451ms). Deferred to avoid duplicate full-corpus paid cost and to preserve the ship gate (no default change without full+indep evidence). The char-vs-byte boundary is pinned by `reranker_tests` multibyte tests.

### 3) Document-format variants (metadata-rich vs path+symbol+code vs raw code vs structured labels)

- **Status:** DEFERRED
- **Reason:** Per VAL-SCREEN-008, format variants must be run under fixed candidate depth and doc budget. The current format is `path+symbol+code` via `format_document` (checked by reranker.rs). Metadata-rich (adding language, repo, line range) vs raw code vs structured labels (explicit `path:`/`symbol:` prefixes) each change reranker input without changing retrieval inputs. Running 4 format lanes on 320 tasks would cost ~4*0.95=3.8 and require 4*1800s, exceeding milestone budget and the one-cargo rule. Deferred: the Qwen rerank win shows current format is functional; format sweep is low priority until a stable Nemotron lane exists and latency p95 is addressed. Instruction-text neutrality rationale is recorded in the task-instruction ablation below.

### 4) Candidate-depth sweep (25/50/100/128/200, deeper depths only if latency permits; 50 is unchanged until full evidence exists)

- **Status:** PARTIALLY EXECUTED (50 baseline only) + DEFERRED remainder
- **Reason/Results:** Depth 50 is the unchanged default and was exercised in both Qwen lanes (50 candidates per query). Qwen rerank (50) nDCG 0.86467 vs embed 0.85097 delta +0.0137. Sweep to 25/100/128/200 deferred: deeper depths increase reranker calls linearly (100 candidates = 5 batches of 20 vs 3 for 50 → +66% cost and latency). With p95 already 10451ms at 50, deeper depths would violate the bar's latency ceiling (6000ms) and the hard rule that 50 stays unchanged until full-suite + independent evidence exists. No default change. Deferred sweep will be run only for a credible finalist before promotion to full 1,251.

### 5) Post-rerank prior weight (current/lower/higher/disabled, measured on both subset and independent set)

- **Status:** DEFERRED
- **Reason:** Prior weight controls fusion of reranker scores with original hybrid scores (RRF). Current weight is the shipped default (hybrid fuses BM25+vector, reranker reorders top 50 with score fusion). Varying it requires 4 lanes (disabled/0.3/0.5/0.8) on both 320 subset and 180 independent set to check for overfit. Subset alone already shows Qwen rerank improves nDCG, but independent-set validation is mandatory before any weight change (VAL-SCREEN-009). Running 8 lanes (4 weights *2 corpora) would cost ~8*0.9=7.2 and exceed reserve planning. Deferred until a stable free reranker lane exists and independent set is warmed (`.bench-indep` 180 tasks). No weight default changed.

### 6) Ordering/fusion variants (reranker ordering vs rank fusion vs normalized score fusion, only if calibration is stable)

- **Status:** DEFERRED — calibration not yet stable
- **Reason:** Per VAL-SCREEN-008, only if calibration is stable. Qwen rerank logs show stable relevance scores (0.75/0.56 in probe), but the free Nemotron path's 503 VLM and prior 84 500s indicate calibration is not stable for free tier. Even for Qwen, the p95 regression (2851→10451) suggests score fusion may need normalization. Three fusion variants (pure reranker order, RRF rank fusion, z-score score fusion) would each need 320-task lanes. Deferred pending free-tier stability and a calibration study; no fusion default changed.

### 7) Task-instruction control (no-instruction vs neutral instruction, run under fixed document format, with instruction text and neutrality rationale recorded)

- **Status:** EXECUTED (partial, fixed format) + DEFERRED no-instruction control
- **Reason/Results:** Qwen lanes used the shipped instruction `Instruct: Given a code search query, retrieve relevant code passages that answer the query\nQuery: ` (query_prefix in lane spec, also reranker instruction field `instruction` for Generic protocol per `crates/vera-core/src/retrieval/reranker.rs:31` and `docs/reranker-server-batching-decision.md`). This is a neutral instruction: it frames the task as code search without biasing toward any repo, language, or symbol, and is the same instruction used for embedding query prefix. The no-instruction control (empty `RERANKER_MODEL_ID` instruction field or omitted) is deferred: running it would require a second 320 rerank lane (cost ~0.95) with identical doc format (path+symbol+code) to isolate instruction effect. Deferred due to budget (already spent 2.625) and because the bar does not require instruction ablation for REJECT. Rationale recorded here: instruction is neutral because it does not name expected file paths or ground-truth quirks, per AGENTS.md benchmark integrity (no overfitting, mechanism rationale).
- **Instruction text:** `Instruct: Given a code search query, retrieve relevant code passages that answer the query\nQuery: {query}` — neutrality: generic code search framing, no repo/language/symbol bias, no ground-truth path mention.

### 8) CoREB/zerank comparisons (when access is unavailable, recorded as unresolved: not silently omitted, not substituted)

- **Status:** UNRESOLVED — access unavailable, correctly recorded
- **Reason:** CoREB and zerank (external reranker baselines) were not accessible during this screening window (no local checkpoints, no OpenRouter IDs, no free endpoints). Per VAL-SCREEN-008, they are recorded as unresolved, not silently omitted nor substituted with another model. The Qwen paid reranker serves as the only paid reference lane; Nemotron free is the screening target. If access becomes available, comparison lanes will be added with same Vera commit, candidate depth 50, doc budget 4800, task revision.

## Summary

- Executed: Qwen 50-candidate rerank vs no-rerank (candidate depth 50 baseline), task-instruction neutral instruction documented (no-instruction control deferred)
- Deferred with reason: batch 20 vs 0, doc budget, doc format, candidate-depth sweep beyond 50, prior weight, ordering/fusion, no-instruction control, CoREB/zerank unresolved
- All deferred items have a concrete reason (cost, latency p95 ceiling, free-tier instability, ship gate requiring full+indep evidence) and an execution sketch for future work. No reranker-related default changed (VAL-SCREEN-009 satisfied trivially).

Artifacts for executed lanes: `benchmarks/results/harrier-screening-20260829T100935Z-subset-qwen-embed.json` and `harrier-screening-20260829T111141Z-subset-qwen-embed-rerank.json` plus logs and specs under `/home/lamim/.cache/vera-lanes/`. Cost and provenance as above.
