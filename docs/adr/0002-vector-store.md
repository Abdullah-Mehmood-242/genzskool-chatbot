# ADR-0002: Qdrant as the vector store

- Status: Accepted
- Date: 2026-11-12

## Context
We need dense search with payload (metadata) filtering, a Docker image, and a Python client.
The corpus has ~20 chunks, so scale is irrelevant; filtering and operational simplicity matter.

## Decision
Use Qdrant (Docker, `qdrant/qdrant`) behind the `VectorStore` port (ingestion) and
`DenseRetriever` port (retrieval). Ship an `InMemoryVectorStore` adapter (cosine similarity in
pure Python) for unit tests and offline runs. `qdrant-client` also supports a local in-process
mode (`QdrantClient(":memory:")`) used by contract tests to prove the Qdrant adapter without
Docker.

Index versioning uses a payload field `index_version`; reads filter on the active version.

## Consequences
- (+) Payload filters map directly to our metadata (`fee_pkr <= budget`, `category`, `level`).
- (+) Swapping to pgvector/Chroma means writing one adapter that passes the port contract tests.
- (-) One more container to run. Acceptable.
