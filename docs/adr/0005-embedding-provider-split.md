# ADR-0005: Embedding provider separate from LLM provider; fake providers

- Status: Accepted
- Date: 2026-11-12

## Context
Anthropic (the default LLM) has no embeddings endpoint. The stack must also run with no keys.

## Decision
- Two independent settings: `LLM_PROVIDER` (`anthropic | openai | ollama | fake`) and
  `EMBEDDING_PROVIDER` (`voyage | openai | sentence_transformers | fake`), each with its own key.
- `fake` embeddings: deterministic feature hashing of lower-cased word unigrams and bigrams into
  `EMBEDDING_DIM` dimensions, L2-normalised. It gives real (lexical) similarity, so retrieval
  metrics are meaningful offline.
- `fake` LLM: deterministic, template-based answer built only from the evidence passed in
  (extracts the relevant fee/mentor lines and cites them). It is grounded by construction, which
  lets the evaluation gate run in CI without keys.
- Real SDKs (`anthropic`, `openai`, `voyageai`, `sentence-transformers`) are imported lazily
  inside their adapters, so they are optional installs and the default image stays small.
- Adapters are chosen in one factory per service, keyed by the setting. Adding a provider =
  new adapter + one factory entry (Open/Closed).

## Consequences
- (+) `docker compose up` and CI work offline.
- (-) Changing embedding model or dimension requires a full re-ingest (new `index_version`);
  documented in the runbook.
