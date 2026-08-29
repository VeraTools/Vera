# Nemotron Reranker Decision — REJECT (2026-08-29)

**Date (UTC):** 2026-08-29T12:00:00Z
**Bar:** `benchmarks/nemotron-acceptance-bar-2026-08-29.md` commit `899e6c3` dated 2026-08-29T07:38:00Z (mtime 03:38:05Z, predates all screening results per VAL-SETUP-010)
**Vera commit for screening:** `7efdbfc` (main at decision time; Qwen embed lane at `24b063e` is throughput-only delta, see run notes) + `24b063e` embedding fix, `7efdbfc` reuse fix
**Semble subset task revision:** `60bac08d1e9d71f6d9208f6c15fcc7c354fab85d2eca5a95b6da9b6dc6608190` (320 tasks, Semble 0.5.5 commit 9218491); harness `task_set.tasks_ids_sha256` recomputed as `c4be588d4a0c9b0e8b3d6fc10c58fa567baae2f76e309a6cf76c3b8108398ff9` (sorted task IDs, recorded per artifact)
**Corpus:** `eval/semble-subset-corpus.toml` (16 repos under `.bench/semble-repos`, all public Semble benchmark repos — free endpoints only saw this corpus, affirmatively logged per VAL-SCREEN-006)

## Decision

**REJECT** — do not include `nvidia/llama-nemotron-rerank-vl-1b-v2:free` (paired with `nvidia/nemotron-3-embed-1b:free`) as a default or preset. No reranker-related default change ships.

## Bar Gates vs Measured

Bar requires ALL of 1–4 to PASS for INCLUDE (see acceptance bar). Nemotron free was unable to produce a paired 320-task result, so gates 1–4 are not met. Qwen paid reference is provided for context only and does not decide the Nemotron bar per bar text.

### Gate 1 — Primary retrieval win (+0.007 nDCG)

- **Required:** `nDCG(rerank) - nDCG(no rerank) >= +0.007` on 320 subset, identical inputs (same Vera commit, candidate depth 50, doc budget 4800 chars, task revision)
- **Measured Nemotron:** no paired result. Attempts:
  - `harrier-screening-20260829T074051Z` (not produced) — panic `input length 4527 exceeds model maximum 4096` on axum (pre-fix 899e6c3)
  - `harrier-screening-20260829T080155Z` (not produced) — 503 `image inputs require VLM serving to be enabled` on requests/zod after fix 24b063e
  - `harrier-screening-20260829T112820Z` (not produced) — same 503 on requests at 7efdbfc, log `/home/lamim/.cache/vera-lanes/run-screening-nemotron-embed-free-20260829T112820Z.log`, cost delta 0.001540840 (free). Direct probe at 07:38:29Z showed embedding dim 2048 and rerank HTTP200 cost 0, but bulk indexing of Semble repos triggers provider-side VLM error.
- **Conclusion:** gate not evidenced; shortfall is `no measurement` (required +0.007, actual unavailable). Provider instability (84 retries previously, now 503 VLM) is a reliability failure under gate 3 as well.

### Gate 2 — Secondary R@1 gain and non-regression

- **Required:** `R@1 delta >= +0.010`, `R@5/R@10/MRR delta >= -0.005`, per-category nDCG regression ≤ -0.02
- **Measured Nemotron:** unavailable (same missing paired result). No per-category delta can be computed.

### Gate 3 — Latency and reliability (p50 ≤2500ms, p95 ≤6000ms, failure ≤10%)

- **Required:** rerank lane p50 ≤2500ms, p95 ≤6000ms, ≤32/320 degraded
- **Measured Nemotron:** no lane completed; free-tier failure rate is 100% of attempts on the 16-repo subset (2 repos trigger 503, blocking full 320). Even if filtered to 14 repos, the provider's prior 84-retry recovery history and current 503 indicate `>10%` failure risk. Qwen paid reference at 111141Z shows p50 1671ms (pass) but p95 10451ms (exceeds 6000), illustrating that rerank overhead already stresses the latency gate even for a successful paid model.

### Gate 4 — Absolute floor nDCG ≥0.800

- **Required:** `nDCG(rerank) >= 0.800`
- **Measured Nemotron:** unavailable; cannot be compared to floor. Qwen embed 0.85097 and Qwen rerank 0.86467 both exceed 0.800, showing the harness and corpus can reach the floor with a stable provider, so Nemotron's failure is provider-specific, not corpus.

## Reference Lanes (paid, for headroom context only)

- **Qwen embed alone:** `benchmarks/results/harrier-screening-20260829T100935Z-subset-qwen-embed.json` (vera 24b063e, 320 tasks, nDCG 0.85097, R1 0.66615, R5 0.92969, MRR 0.83647, p50 372.89, p95 2851.09, index_time >0, storage 1.2GB) — lane spec `screening-qwen-embed-2026-08-29.json` (reuse false, batch128, VERA_MAX_IN_FLIGHT_INPUTS 1024)
- **Qwen embed + Qwen rerank:** `benchmarks/results/harrier-screening-20260829T111141Z-subset-qwen-embed-rerank.json` (vera 7efdbfc, 320 tasks, nDCG 0.86467, R1 0.69583, R5 0.92552, MRR 0.85166, p50 1671.86, p95 10451.20, index_time 0.0 reused 16 indexes) — lane spec `screening-qwen-embed-rerank-2026-08-29.json` (reuse true)
- **Delta:** +0.01370 nDCG (pass gate 1 threshold), +0.02968 R1 (pass gate 2), R5 -0.00417 (within -0.005), MRR +0.01518. Latency p50 +1299ms, p95 +7600ms (p95 exceeds gate 3). Vera commit delta 24b063e→7efdbfc is the reuse fix only (throughput, allows API expected_dim None), documented in run notes; all other retrieval inputs identical (embedding model qwen/qwen3-embedding-8b, query_prefix, candidate depth 50, doc budget 4800, task_set c4be588d, semble 60bac08d). This delta passes the primary bar but does not imply Nemotron would; it only shows paid reranker headroom on this subset.

## Provenance Notes (identical retrieval inputs, per VAL-SCREEN-002/011)

- Both Qwen results record `version_info.vera_git_sha`, `version_info.task_set`, `version_info.semble`, `version_info.repo_shas` (16), `version_info.lane` (model_id, backend, rerank flag), `version_info.config` (lane.* keys). Only `lane.rerank` and `lane.reuse_index` differ between the paired comparison; all other config (candidate depth 50 default, reranker_max_doc_chars 4800, max_rerank_batch 20, document_prefix) are identical and affirmatively logged. Run-note fallback file `benchmarks/screening-run-notes-2026-08-29.md` records the Vera commit delta rationale and task_set recomputation (c4be588d vs corpus.toml 60bac08d).
- Nemotron lanes (`screening-nemotron-embed-free-2026-08-29.json` and `screening-nemotron-embed-free-rerank-2026-08-29.json`) use same corpus, same candidate depth/doc budget, free endpoints only. Logs show free corpus scope (no private code). Direct probe batch 07:38:29Z is the preceding probe for every paid/free lane per VAL-SCREEN-010.

## Cost and Ship Gates

- Cost log `/home/lamim/.cache/vera-away/cost-log.md` itemizes probe (0.00003023), Qwen embed (0.37566346), Qwen rerank (0.95880791 successful + 0.214/0.735 aborted attempts), Nemotron free attempt (0.00154084), and transient failures (0.09506217 + 0.232). Sum reconciles with overall balance delta 2.62527524 (start 9.316928068 → end 6.691652828). Minimum post-balance 6.691 > floor 1.863, never breached. No duplicate full-corpus paid lanes (all subset). Free lanes only public corpus.
- Ship gate VAL-SCREEN-009: no reranker-related default changed. Any future INCLUDE would require full 1,251-task Semble + independent-set (180) evidence at the shipping commit showing clear win and no material regression. This decision explicitly does NOT authorize a default change; it is a REJECT.
- No full 1,251 run was executed for this screening (correct per VAL-SCREEN-007 — full only for finalists or explicit quality claims). Qwen is not a finalist pending latency p95 and independent-set validation.

## Rationale for REJECT

1. Nemotron free embedding cannot reliably index the public Semble subset (requests/zod trigger 503 VLM, axum triggered 422 before provider shrink fix). This is a free-tier provider defect outside Vera's control, but it blocks the required 320-task measurement. Per library/environment.md intermittent 500s and 84-retry history, free Nemotron is unstable.
2. Without a paired measurement, gates 1–4 cannot be shown to pass. The bar is not met by absence.
3. Even if the VLM issue were worked around by filtering those 2 repos (which would violate the identical-corpus requirement or require a documented reduction), the resulting partial-corpus metric would not satisfy the 320-task bar, and the latency/reliability gate would still be at risk due to free-tier rate limits (429 20/min observed in v132 logs).
4. Qwen paid reranker does show a primary win, but that is a separate lane and does not rescue the Nemotron bar. Qwen's own p95 (10451ms) already exceeds the bar's latency ceiling, indicating that promoting any reranker to default needs deeper latency work before full-suite evidence.

## Next Steps if Re-visited

- Filtered corpus run excluding requests/zod with explicit reduction note and reason (VLM), or await provider fix for VLM serving. Re-probe Nemotron free, then re-run 320 subset with same Vera commit, candidate depth 50, doc 4800, and compare.
- Ablations (see ablation register) must be completed before any default change, plus full 1,251 + independent 180 validation.

## Artifacts

- Bar: `benchmarks/nemotron-acceptance-bar-2026-08-29.md` (commit 899e6c3)
- Results: `benchmarks/results/harrier-screening-20260829T100935Z-subset-qwen-embed.json`, `benchmarks/results/harrier-screening-20260829T111141Z-subset-qwen-embed-rerank.json`
- Logs: `/home/lamim/.cache/vera-lanes/probe-2026-08-29T0738Z.log`, `/home/lamim/.cache/vera-lanes/run-screening-qwen-embed-20260829T100935Z.log`, `/home/lamim/.cache/vera-lanes/run-screening-qwen-embed-rerank-20260829T111141Z.log`, `/home/lamim/.cache/vera-lanes/run-screening-nemotron-embed-free-20260829T112820Z.log`
- Specs: `/home/lamim/.cache/vera-lanes/screening-*-2026-08-29.json`
- Cost log: `/home/lamim/.cache/vera-away/cost-log.md`
- Run notes: `benchmarks/screening-run-notes-2026-08-29.md`
- Ablation register: `benchmarks/ablation-register-2026-08-29.md`
