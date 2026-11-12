# ADR-0001: Microservices with hexagonal internals

- Status: Accepted
- Date: 2026-11-12

## Context
The knowledge base is ~5 PDF pages (~3-5k tokens). A single process with a modular structure
would serve it with lower latency, fewer failure modes and far less operational cost. The
project brief nonetheless requires microservices, SOLID and clean architecture.

## Decision
Build five independently deployable services (`api_gateway`, `ingestion`, `retrieval`,
`advisor`, `evaluation`). Each service is small and **hexagonal** inside (`domain`,
`application`, `infrastructure`, `api`). Services share only DTO contracts
(`libs/common/contracts`) and talk over HTTP/JSON.

## Consequences
- (+) Clear seams: ingestion can be re-run without touching the chat path; retrieval can be
  tuned and evaluated in isolation; the gateway owns all edge concerns.
- (+) Because every use case depends on ports, the services could be collapsed into a modular
  monolith by replacing the HTTP client adapters (`HttpRetrievalClient`) with in-process
  adapters that call the use case directly. No domain or application code changes.
- (-) Extra network hops (gateway -> advisor -> retrieval) add latency (~5-20 ms per hop on a
  docker network) and need timeouts, retries and breakers.
- (-) More deployment surface: 5 images, 5 health checks.
- Revisit if: operating cost matters more than the architectural exercise, or traffic stays
  below a few requests per second. Then collapse into one process as described above.
