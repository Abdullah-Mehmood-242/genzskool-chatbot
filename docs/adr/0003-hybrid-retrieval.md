# ADR-0003: Hybrid retrieval (dense + BM25, RRF fusion)

- Status: Accepted (revised 2026-11-12 after measuring on the real KB)
- Date: 2026-11-12

## Context
Users ask with exact tokens ("eBay", "CapCut", "Make.com", "Rs. 30,000", "GS-DM-001") and with
vague intent ("I want to earn online with no experience"). Dense embeddings handle the latter;
lexical matching handles exact product names. With fake (hash-based) embeddings in offline mode,
BM25 carries most of the quality. Retrieval must also return **nothing** for off-topic questions:
an empty list is what lets the advisor refuse instead of force-fitting a course.

## Decision
Pipeline (`SearchService`): normalise -> (optional rewrite) -> metadata filter -> dense + BM25 in
parallel -> RRF -> (optional rerank) -> threshold -> top-k.

- **Corpus**: the artifacts ingestion writes. `data/processed/registry.json` names the active
  `index_version` per document and `<doc_id>.v<N>.chunks.jsonl` holds its chunks (ADR-0011). The
  file provider re-reads them only when the registry changes (inode + mtime + size, because the
  atomic replace gives a new inode even when v1 -> v2 keeps the same size). No service call is needed.
- **Filters** are evaluated in the domain to a set of allowed chunk ids; both retrievers are
  restricted to it. In Qdrant that is a `HasId` filter, which applies the filters exactly and also
  pins dense results to the active versions.
- **Query analysis** (same rules for query and corpus): NFKC, lower-case, thousands separators
  removed (`30,000` -> `30000`), explicit lemma map, English + Roman Urdu stopwords, and
  deterministic synonym expansion from `config/retrieval.yaml` ("kitne paise" -> fee, "reels" ->
  video editing). The LLM rewrite port exists but ships disabled with a no-op adapter.
- **BM25** via `rank_bm25`, subclassed to use Lucene's IDF `ln(1 + (N - n + 0.5)/(n + 0.5))`.
  rank_bm25's default IDF is 0 for a term in exactly half the documents and negative above, so a
  common but meaningful word like "fee" could silently stop matching as the corpus grows.
- **Candidate gates** (what makes off-topic queries return `[]`):
  - dense: cosine >= `dense_min_score` (model-dependent: 0.3 for fake embeddings, where unrelated
    chunks measured 0.15-0.28);
  - BM25: the chunk must contain >= `min_term_matches` (2) of the query's content words, or at
    least `min_term_coverage` (50%) of them, synonyms counting. One stray common word ("today")
    is not relevance.
- **RRF**: `score(d) = sum 1/(fusion_k + rank)`, `fusion_k = 60`, normalised by the best possible
  score (`n_lists/(k+1)`) to [0, 1]. Then `score_threshold` (0.3) and `top_k` (5).
- **Reranker** port: no-op default; Cohere/Voyage over HTTP or a local cross-encoder when enabled.
  On reranker failure the fused order is kept.
- **Degradation**: if the dense retriever fails (Qdrant down), search continues on BM25 alone and
  reports `dense_error` in `retrieval_debug`; `/readyz` shows `vector_store: false`. Only if every
  retriever fails is the response 503.
- **Privacy**: query text is never logged (it may contain phone numbers); logs carry counts,
  reasons and latency only.

## Measured (fake embeddings, real KB, `data/eval/retrieval_cases.jsonl`)
42 answerable + 6 off-topic cases: hit-rate@5 = 1.00, MRR >= 0.8, off-topic -> empty 6/6. The same
gate runs in-process (`services/retrieval/tests/eval`) and over HTTP against real Qdrant (e2e).
The cases were written while tuning; Phase 6 adds an independent golden set.

## Consequences
- (+) Robust to keyword and semantic queries; works offline; refusals are data-driven.
- (+) New index versions are picked up on the next request.
- (-) Thresholds are tuned per embedding model; when switching to a real provider, re-run the
  retrieval eval and adjust `dense_min_score`.
- (-) The BM25 index lives in process memory per replica; fine at this corpus size.
