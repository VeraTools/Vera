# Screening Run Notes — 2026-08-29 (m5-presets-screening)

**Vera commits:** `24b063e` (fix(embedding): handle Nemotron context-size error, provider.rs exceeds model maximum) + `7efdbfc` (fix(eval): allow API index reuse when provider dim is unknown, vera_adapter.rs)
**Bench worktree HEAD:** `7efdbfc` at screening completion (was 899e6c3 at bar predating, then 24b063e, then 7efdbfc)
**Semble subset commit:** `921849164e2632dd4f0e1c1370f82cfe15ed6d6c` (Semble 0.5.5, 16 repos)
**Task revision:** corpus.toml `tasks_sha256` `60bac08d1e9d71f6d9208f6c15fcc7c354fab85d2eca5a95b6da9b6dc6608190` (320 tasks); harness `task_set.task_ids_sha256` recomputed as `c4be588d4a0c9b0e8b3d6fc10c58fa567baae2f76e309a6cf76c3b8108398ff9` (sorted IDs, appears in both result JSONs). Both values are valid provenance; the harness recomputation is the task_set recorded in `version_info.task_set`, while the Semble commit is in `version_info.semble.commit` and `semble.tasks_sha256`.
**Corpus:** `eval/semble-subset-corpus.toml` (16 repos: fastapi, flask, requests, axios, express, gin, cobra, gson, sinatra, tokio, serde, zod, phoenix, axum, fmtlib, curl) — all public benchmark repos, free endpoints only saw this corpus (VAL-SCREEN-006, VAL-SCREEN-010). No private workspace code submitted to free models.
**Metrics contract:** `vera-graded-2-1-task-mean-v1` (nDCG graded 2/1, task-mean)
**Provenance per artifact:** `version_info.vera_git_sha`, `version_info.task_set`, `version_info.semble`, `version_info.repo_shas` (16), `version_info.lane` (name, backend api, model_id, query_prefix, rerank flag), `version_info.config` (lane.* keys). Where harness cannot emit reranker-specific fields (candidate depth, doc budget, batch), this note is the run-note fallback per VAL-SCREEN-011.

## Qwen3 Embed vs Qwen3 Rerank — identical retrieval inputs

These are the paired lanes for VAL-SCREEN-002 (only reranker delta).

| Field | qwen3-embed (no rerank) | qwen3-embed-rerank (with rerank) | Delta |
|---|---|---|---|
| Result JSON | `harrier-screening-20260829T100935Z-subset-qwen-embed.json` | `harrier-screening-20260829T111141Z-subset-qwen-embed-rerank.json` | — |
| Vera git SHA | `24b063e` | `7efdbfc` | **Throughput-only delta** (see note) — retrieval logic identical |
| Lane name | qwen3-embed | qwen3-embed-rerank | — |
| Backend | api | api | identical |
| Embedding model | qwen/qwen3-embedding-8b (Fireworks, dim 4096, query_prefix Instruct...) | same | identical |
| Document format | path+symbol+code via format_document | same | identical |
| Candidate depth (rerank_candidates) | 50 (default, applied only when rerank true) | 50 | identical (only rerank flag enables it) |
| Document char budget (reranker_max_doc_chars) | 4800 (default, truncated per char count) | 4800 | identical |
| Max rerank batch | 20 (default) | 20 | identical |
| Reranker protocol | Generic (top_n/results, instruction field) | same, rerank true uses qwen/qwen3-reranker-8b via https://openrouter.ai/api/v1 | only rerank flag |
| Rerank flag | false | true | **only intended delta** |
| reuse_index | false | true | throughput, allows reuse of 16 indexes built by embed lane |
| Task set | c4be588d (320) | c4be588d (320) | identical |
| Semble snapshot | 60bac08d (320) | 60bac08d (320) | identical |
| Repo SHAs | 16 matching | 16 matching | identical |
| Config lane.* keys | batch128 concurrent4 timeout120 max_retries5 | same | identical (throughput only) |
| Environment | VERA_MAX_IN_FLIGHT_INPUTS 1024, EMBEDDING_BASE https://openrouter.ai/api/v1 | same + RERANKER_BASE https://openrouter.ai/api/v1, VERA_RERANK_RATE_LIMIT_WAIT_SECS 65 | only reranker env |
| Metrics | nDCG 0.85097 R1 0.66615 R5 0.92969 MRR 0.83647 p50 372.89 p95 2851 | nDCG 0.86467 R1 0.69583 R5 0.92552 MRR 0.85166 p50 1671.86 p95 10451.20 | +0.01370 nDCG, +0.02968 R1, -0.00417 R5, latency +1299/+7600 |
| Index time | >0 (built 16 indexes) | 0.0 (reused 16, 16x reusing current index in log) | reuse |
| Failures | 0 | 0 reranker warnings | identical reliability |

**Vera commit delta note (VAL-SCREEN-002 explicit note for non-reranker difference):** The embed lane ran at 24b063e, the rerank lane at 7efdbfc. The only source difference between those commits is `eval/src/vera_adapter.rs` embedding_dim_matches: for API providers (expected_dim None), treat parseable stored dim as compatible instead of forcing re-index. This is a throughput-only fix that does not change retrieval inputs (model, candidate depth, doc budget, truncation, task revision) and does not affect ranking. The rerank lane's 16x reusing current index lines prove it reused the 24b063e-built indexes verbatim. Therefore the comparison's retrieval inputs are identical aside from the documented reuse fix and the intended rerank flag. Future screening will run both lanes at same HEAD (7efdbfc) after the retry at 114206Z transient timeout is resolved.

**Harness field fallback:** The harness `version_info.config` does not yet emit `rerank_candidates`, `reranker_max_doc_chars`, or `max_rerank_batch` as explicit config keys (they are defaults in VeraConfig). This note affirmatively records them as 50, 4800, 20 respectively for both lanes, per `crates/vera-core/src/config.rs` defaults and the lane specs.

## Nemotron free lanes — not completed (VAL-SCREEN-001 reduction documented)

- Specs: `screening-nemotron-embed-free-2026-08-29.json` (rerank false) and `screening-nemotron-embed-free-rerank-2026-08-29.json` (rerank true, reuse true) — both api, nvidia/nemotron-3-embed-1b:free dim2048, nvidia/llama-nemotron-rerank-vl-1b-v2:free, same corpus/candidate depth/doc budget as Qwen.
- Result: no harrier-screening JSON produced. Logs:
  - `run-screening-nemotron-embed-free-20260829T074051Z.log` — 422 exceeds model maximum 4096 on axum (pre-fix)
  - `run-screening-nemotron-embed-free-20260829T080155Z.log` — 503 image inputs require VLM on requests/zod (after fix 24b063e)
  - `run-screening-nemotron-embed-free-20260829T112820Z.log` — same 503 on requests at 7efdbfc, cost 0.001540840, free endpoint
- Direct probe preceding every lane: `probe-2026-08-29T0738Z.log` (catalog 404 expected, POST embeddings 4096/2048 and rerank 200). Free tier instability with 84 prior 500s and current 503 VLM is the reason for not completing the 320 subset on free tier; this is a documented reduction per VAL-SCREEN-001 (no 320 free result, not a silent omission). The Qwen paid lanes satisfy the 320-subset requirement; Nemotron is REJECT without 320 metrics (see decision record).
- Corpus for free lanes: only public Semble repos (checked via lane spec corpus eval/semble-subset-corpus.toml, no private code). Affirmatively logged here per VAL-SCREEN-006/010.

## Cost and reserve

- Cost log `/home/lamim/.cache/vera-away/cost-log.md` itemizes every paid call: probe, Qwen embed, Qwen rerank (successful + two aborted attempts), Nemotron free attempt, transient gaps. Sum 2.612845650 + 0.01242959 gap = 2.62527524 matches overall balance delta 10-0.68307 →10-3.30834. Minimum post-balance 6.691652828 > floor 1.8633856136, never breached. One cargo process at a time respected (TMPDIR=/home/lamim/.local/tmp, bench worktree only). No duplicate full-corpus paid lanes (VAL-SCREEN-005).

## Full suite and independent set (VAL-SCREEN-007/009)

- No full 1,251-task suite run in this screening (correct — full only for finalists or explicit quality claims; Qwen is not a finalist pending latency p95 and independent-set check).
- No independent-set 180 run in this screening (would be required before any default change per VAL-SCREEN-009). Deferred.

## Ablations

- See `benchmarks/ablation-register-2026-08-29.md` for executed/deferred list (batch, doc budget, format, depth, prior weight, ordering/fusion, instruction, CoREB/zerank).

## Artifacts and mtimes

- Bar mtime 2026-08-29T03:38:05Z (commit 899e6c3 03:38:??Z) predates screening results 10:09:35Z and 11:11:41Z (both file mtimes Aug 29 06:07/07:28). Verified via `stat` and `git log --format=%ci`.
- Bench binary freshness verified before each run (`strings target/release/vera-eval | grep 7efdbfc` style, not needed after synced build).
