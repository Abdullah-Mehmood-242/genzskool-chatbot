# ADR-0003: Hybrid retrieval (dense + BM25, RRF fusion)

- Status: Accepted
- Date: 2026-11-12

## Context
Users ask with exact tokens ("eBay", "CapCut", "Make.com", "Rs. 30,000", "GS-DM-001") and with
vague intent ("I want to earn online with no experience"). Dense embeddings handle the latter;
lexical matching handles exact product names. With fake (hash-based) embeddings in offline mode,
BM25 also carries retrieval quality on its own.

## Decision
- Dense search via the `DenseRetriever` port (Qdrant).
- Sparse search via `SparseRetriever` using `rank_bm25` over the chunk texts loaded from
  the active chunk files ingestion writes: `data/processed/registry.json` names the active
  `index_version` per document and `data/processed/<doc_id>.v<N>.chunks.jsonl` holds its chunks
  (ADR-0011). Dense results are filtered to the same active versions. Rebuilt on startup and on
  index-version change.
- Fuse with Reciprocal Rank Fusion: `score(d) = sum 1 / (fusion_k + rank_i(d))`, `fusion_k = 60`.
- Then metadata filter, optional reranker (port, disabled by default), then `score_threshold`.
- RRF scores are rank-based, so the threshold is applied to a normalised score
  (fused score divided by the max possible fused score) to stay comparable across configs.
- Query rewrite is a port, disabled by default.

## Consequences
- (+) Robust to both keyword and semantic queries; works offline.
- (-) BM25 index lives in process memory per retrieval replica; fine at this corpus size.
