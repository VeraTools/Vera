# ADR 008: Enable filter-during-scan by default for filtered vector queries (issue #197)

Status: accepted (2026-09-03); shipped in the default-flip PR following the r5 preregistered decision round

## Context

Filtered vector queries (path glob, exact path, or language filters) used to hydrate the whole flat vector index and filter afterward. Issue #197 shipped the filter-during-scan mechanism default-off in PR #255 (commit 1ff24cd): a lazy per-store eligibility map (chunk row to path-id and language, built once per store generation) gates the flat SIMD scan so only eligible rows compete for top-K and only the filtered top-K hydrates. Differential byte-identity tests proved flag-on equals flag-off results, and PR #256 (commit b2e5f6c) hardened the path (typed fallback, scan guard, bulk matcher).

The default stayed OFF pending evidence. Three preregistered rounds supplied it:

- **r3** (at 1ff24cd, comment 5503479257): mechanism control PASS, flag-on beat flag-off by p50 5.077 ms and p95 66.260 ms with same-head nDCG parity 0.000067 (within ±0.001). The issue acceptance gate failed on absolute nDCG only: flag-on 0.843733 vs the 072c725 same-host baseline floor 0.843852, short 0.000119. Latency gates passed (p50 6.313 ≤ 7.38, p95 60.397 ≤ 65.88).
- **r4** (comment 5504100591): drift localization across 2528f2e..b2e5f6c found the absolute nDCG shortfall NOT LOCALIZABLE to a single commit (largest neighbor delta 0.000501, below the 0.00090 rule). It is cumulative intentional lineage work: 5d66972 split-symbol #226 (-0.000501), 964a2d5 filtered-vector fix #227 (-0.000404), 15b35d3 ranking signals #239 (-0.000145), b2e5f6c hardening #256 (-0.000340, mostly carried). Decisive cross-check: flag-off at b2e5f6c measured 0.843734, reproducing r3's flag-on 0.843733. The absolute gate fails both arms identically, so it cannot adjudicate the flag.
- **r5** (at 98e6e50, comment 5532835495, the decisive round): 3 flag-on + 3 flag-off fresh-process full-suite runs, interleaved, one binary, env-only toggles. All three gates passed and the verdict was FLIP.

## Re-anchor rationale

r5 re-judged the nDCG criterion as same-head paired parity (|mean(on) - mean(off)| ≤ 0.001, band inherited unchanged from r3's mechanism-control rule) instead of absolute parity with the 072c725 baseline. The causal question for a default flip is whether turning the flag on at the same head changes ranking relative to off at that head; that is the comparison a default actually ships. An absolute gate that fails both arms identically (per r4) measures lineage drift, not the flag's effect. A permanent-OFF third option was considered and rejected on the same grounds: discarding a proven mechanism over head-vs-head drift the flag does not cause. The absolute latency criterion was kept, and the absolute nDCG delta is disclosed alongside every verdict rather than being hidden by the re-anchor.

## r5 evidence (AMD Ryzen 7 9800X3D, 1,251-task Semble suite, local Potion Code, 3 runs per arm)

| metric | flag-on mean | flag-off mean | delta (off - on) | gate | verdict |
|--------|--------------|---------------|------------------|------|---------|
| p50 latency | 6.352505 ms | 12.003688 ms | 5.651183 ms | > 0.6 ms better | PASS |
| p95 latency | 65.033730 ms | 139.123443 ms | 74.089713 ms | > 5 ms better | PASS |
| nDCG@10 | 0.843699 | 0.843694 | 0.000004893 | ≤ 0.001 | PASS |

Absolute latency acceptance (gate c, vs the corrected same-host 072c725 baseline 7.879 / 60.880 ms): flag-on p50 6.352505 ≤ 7.38 PASS (1.026 ms under); p95 65.033730 ≤ 65.88 PASS (0.846 ms under, within the preregistered non-regression band (60.88, 65.88], not flat, not improved).

## Decision

Flip `retrieval.vector_filter_during_scan` default to `true` (`default_vector_filter_during_scan()` in `crates/vera-core/src/config.rs`). The env var `VERA_VECTOR_FILTER_DURING_SCAN` stays authoritative over both the config-file value and the default, so the off path remains one env var away for rollback or ablation. No production logic changed beyond the default.

## Mandatory absolute-delta disclosure

Shipped full-suite nDCG is about 0.0011 below the issue-opening state (0.844852 → about 0.8437). This delta is present identically with the flag off and is attributed by r4's preregistered localization to intentional lineage work (#226 split-symbol, #227 filtered-vector fix, #239 ranking signals), not to this flag. Same-head on/off parity passed at 0.000004893 (±0.001 band). This decision does not claim absolute nDCG parity with the 072c725 baseline.

## Correctness, staleness, and memory bounds

- Exactness: differential tests in `crates/vera-core/src/retrieval/filter_scan_tests.rs` prove flag-on equals flag-off output (17 tests, including the 5-case byte-identity matrix on the 4,427-chunk overcap fixture).
- Staleness: the eligibility map is per store plus generation, revalidated against the metadata and vector-store stamps on every use; generation-change, tombstone, update-invalidation, and corrupted-map fallback tests (val_001, val_013, val_014, val_015) cover it, along with the watch/incremental suites.
- Memory: at most 4 indexed repositories stay resident (LRU-4 cap in `search_service.rs`), each holding one eligibility map with two arrays sized to max_rowid (path-ids and languages) plus one distinct-path table; maps are stamp-invalidated, never accumulated.

## Consequences

- Filtered queries on the flat backend hydrate only the filtered top-K. Full-suite filtered latency drops by roughly half at p50 and more at p95.
- Users who need the old behavior set `VERA_VECTOR_FILTER_DURING_SCAN=0` or `retrieval.vector_filter_during_scan = false` in config.
- The sqlite-vec path (`VERA_VECTOR_SCAN=vec0`) ignores the flag (val_019) and is unaffected.
- Legacy config files without the field deserialize to the new default (true); explicit `false` in a config file still wins over the default, and env still wins over the file.

## References

- Issue #197 and its r3/r4/r5 preregistration comments (5503479257, 5504100591, 5532835495)
- Comparison artifact: `/home/lamim/.cache/vera-lanes/issue197r5-comparison.json` and `.md` (means, stdevs, gates, margins)
- Result JSONs: `benchmarks/results/issue197r5-20260903T*-full-98e6e50-*.json` (bench worktree, untracked)
- Implementation: `crates/vera-core/src/config.rs`, `crates/vera-core/src/retrieval/hybrid.rs`, `crates/vera-core/src/retrieval/search_service.rs`, `crates/vera-core/src/storage/eligibility.rs`
- ADR 006 / ADR 007 for the toggleable-knob and implementation/measurement-separation patterns this flip followed
