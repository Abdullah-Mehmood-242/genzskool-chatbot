# ADR-0012: Embedding clients live in libs/common

- Status: Accepted
- Date: 2026-11-12

## Context
Ingestion embeds documents and retrieval embeds queries. They must use the same provider, model,
dimension and, for the offline `fake` provider, the exact same hashing. Two copies of the clients
would eventually drift (a tokenizer tweak in one service), and dense retrieval would break with no
error, only worse results. The rule "libs/common holds contracts only, no business logic" argues
for duplication; correctness argues against it.

## Decision
- `common.embeddings` holds the embedding clients: `FakeEmbedder` (signed feature hashing with
  blake2b, so it is identical across processes and machines), `VoyageEmbedder` and `OpenAIEmbedder`
  (plain `httpx` REST, no SDK dependency), `SentenceTransformersEmbedder` (optional package, lazy
  import) and the `build_embedder` factory.
- This is infrastructure plumbing, not business logic: no course, fee or advisor rule lives there
  (enforced by `tests/architecture/test_repo_structure.py::test_common_lib_has_no_business_logic_keywords`).
- Each service still owns its port (`EmbeddingProvider` in its `domain/`) and a thin adapter in
  `infrastructure/` that delegates to the shared client, so the dependency rule is unchanged.
- One contract test suite (`libs/common/tests/test_embeddings.py`) runs against every provider.
  A cross-process test proves the fake embeddings are stable.
- Error mapping: 429/5xx/network errors become `TransientProviderError` (retried); 401/403/4xx
  become `ProviderError` (not retried); timeouts become `UpstreamTimeout` (retried). Messages never
  contain the API key.

## Consequences
- (+) Ingestion and retrieval cannot disagree on embeddings.
- (+) No vendor SDKs needed for Voyage/OpenAI embeddings; smaller images.
- (-) A change to `common.embeddings` affects both services at once (intended).
