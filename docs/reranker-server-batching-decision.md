# Reranker server-side batching decision

**Date:** 2026-08-29
**Status:** Declined — no separate capability field

## Investigated protocols

- Generic: SiliconFlow, Jina, Cohere, OpenRouter Qwen/vLLM-compatible `/rerank` (`top_n` + `results`, `return_documents: false` by default, `instruction` supported via `instruction` field)
- Voyage AI: `top_k` + `data` (`https://api.voyageai.com`)

Sources checked: current vLLM docs (`docs.vllm.ai/models/pooling_models/scoring` Cohere Rerank API, `instruction` field for both `/score` and `/rerank`), OpenRouter rerank docs (Python/TS SDK README, model list), Voyage rerank docs referenced via OpenRouter collection, Qwen3-Reranker model cards (HF `Qwen/Qwen3-Reranker-8B`), SiliconFlow/Jina/Cohere generic OpenAI-style rerank compatibility.

## Findings

No investigated provider documents a distinct “server-side batching preference” flag that the client must set. All providers accept arbitrary `top_n`/`top_k` and a documents array; none expose a `batch` or `prefer_server_batching` field in the rerank request. vLLM’s rerank API (which Vera targets for self-hosted Qwen3-Reranker) handles long document lists server-side without a client hint; sending the whole candidate set in one request is simply `max_rerank_batch = 0` in Vera’s existing model (0 = one unbatched request, preserved from `e6bb1fa5`).

## Decision

Do not add a separate `reranker_server_batching` capability field. The existing `retrieval.max_rerank_batch` (and its alias `retrieval.reranker_max_doc_chars` family) already represents the client-vs-server choice:

- `20` (default) = client-side batching, each batch ≤ 20 docs, scores merged with global index remapping.
- `0` = single unbatched request, letting the server batch internally.

If a future provider introduces an explicit server-batching hint (e.g. `batch: false` or `prefer_server_batching`), it should be added as a small capability field with a captured-request test, per the handoff. This note satisfies VAL-RERANK-031’s “explicitly declined with a written decision note” branch.
