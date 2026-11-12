# MASTER PROMPT: Genz Skool Career Advisor (RAG, microservices)

> Paste everything below this line into Claude Code, run from the repository root (`genz-career-advisor/`).

---

## 0. Role and operating rules

You are a senior AI/software engineer. Build a production-grade, grounded RAG **career and course advisor** for Genz Skool (https://genzskool.com), a practical skills academy in Sargodha, Pakistan.

Operating rules:
1. **Plan first.** Before writing code, read `data/pdfs/genzskool_knowledge_base.pdf`, `.env.example` and the existing folder skeleton. Then write `docs/ARCHITECTURE.md` and one ADR per major decision in `docs/adr/`. Only then implement.
2. **Build in the phases of Section 12**, committing (or stopping for review) after each phase. Each phase must end with passing tests.
3. **Never hardcode secrets.** Every API key is a placeholder in `.env.example`; real values come from `.env` (git-ignored). The code must start and the unit tests must pass with **no real keys**, using fake/in-memory adapters.
4. **Do not invent data.** All business facts (courses, fees, mentors, batches, contact info) come from the PDFs in `data/pdfs/`. No fabricated syllabus, salaries, job placement rates or student outcomes.
5. Do not install anything not listed in the stack without justifying it in an ADR.
6. If something is ambiguous, choose the simplest option that satisfies this spec, record the decision in an ADR, and continue.

## 1. Product goal

A chat advisor that takes a learner's background, goals and constraints (budget, time, current skills, device, location, language) and recommends which Genz Skool course(s) to take, in what order, why, and what to do next, **grounded only in the knowledge base**.

Example questions it must handle:
- "I want to earn online but have no experience. Where do I start?"
- "Shopify or TikTok Shop? Budget is Rs. 30,000."
- "I like editing reels. What courses fit and what do they cost?"
- "Can I join a batch that already started?" / "Do I get a certificate?" / "How do I enroll?"
- "I want a job as a data scientist" (out of scope: must say Genz Skool does not teach this and not force-fit a course).

## 2. Ground-truth data (already crawled; lives in `data/pdfs/`)

- 8 courses: Shopify (Rs. 30,000), TikTok Shop (Rs. 30,000), eBay (Rs. 28,000), Video editing (Rs. 30,000), Digital marketing (Rs. 35,000), Graphic design (Rs. 28,000), AI automation with ChatGPT and Make.com (Rs. 32,000), TikTok automation (fee not published, ask on WhatsApp). Instalments available where stated.
- 6 mentors: Hamza Ali (Shopify, eBay), Sana Iqbal (TikTok Shop, TikTok automation), Bilal Ahmed (Video editing), Mahnoor Raza (Digital marketing), Zara Malik (Graphic design), Usman Tariq (AI automation).
- Format: recorded lessons + weekly live classes, a project per module, mentor review, certificate, lifetime access, morning/evening/weekend batches.
- Enrollment: `/enroll` form, no payment on site, advisor confirms seat, batch and fee on WhatsApp 0309-4190245. Advisor hours Mon-Sat 10am-8pm. Email info@genzskool.com. Address: Farhan Plaza 1st Floor, Office No. 6, Near 47 Pull, Mall Road, Sargodha.

**Known data defects the system MUST respect** (they are in the PDF, Section 9):
- Headline lesson counts do not match the 3-lessons-per-module structure. Never present a per-module lesson list beyond what the PDF states.
- Testimonials use placeholder names; do not cite earnings or student outcomes.
- No salary, placement or income data exists. The advisor must never promise income.
- Batch/seat data goes stale (one batch is literally named "Portal Test Batch"). Always add: "confirm current batches and fees on WhatsApp 0309-4190245".

## 3. Honest design constraints (read before designing)

The entire corpus is roughly 4 PDF pages (~3-5k tokens). A microservices RAG for that is architecturally heavier than the problem needs. The requirement is microservices, SOLID and clean architecture, so build them, **but**:
- Keep each service small and independently deployable, with **hexagonal (ports and adapters) internals**, so the same code could later be collapsed into a modular monolith.
- Record this trade-off in `docs/adr/0001-microservices-vs-monolith.md`.
- Because the corpus is tiny, retrieval quality is dominated by **chunking and metadata**, not the vector DB. Chunk by semantic unit (one chunk per course, per mentor, per policy section), attach structured metadata, and enable metadata filtering (course, category, price, level). This matters more than model choice.

## 4. Technology stack

- Python 3.12, FastAPI, Pydantic v2, `httpx` (async), `uvicorn`.
- Orchestration of the advisor flow: LangGraph (state machine). LangChain only for document loaders/splitters if useful, never inside the domain layer.
- Vector store: Qdrant (Docker). Hybrid retrieval: dense + BM25 (`rank_bm25` or Qdrant sparse), fused with Reciprocal Rank Fusion. Optional reranker behind a port.
- PDF parsing: `pdfplumber` (fallback `pypdf`).
- LLM and embedding providers are **adapters behind ports**: LLM (`anthropic` default, `openai`, `ollama`), embeddings (`voyage`, `openai`, `sentence_transformers`). Note: Anthropic has no embeddings endpoint, so embeddings are a separate provider/key.
- Quality: `pytest`, `pytest-asyncio`, `respx`, `ruff`, `mypy --strict` on `libs/` and every `domain/` + `application/` package, `pre-commit`.
- Packaging: `uv` or `pip-tools` per service with pinned versions; Docker multi-stage per service; `docker-compose.yml` at repo root.
- Observability: structured JSON logs with request IDs, `/healthz` and `/readyz` on every service, basic Prometheus metrics (`/metrics`).

## 5. Target repository structure (already scaffolded; fill it in, do not rename)

```
genz-career-advisor/
├── MASTER_PROMPT.md
├── README.md
├── .env.example                 # placeholders only
├── .gitignore
├── Makefile                     # (you write it) make up | down | test | lint | ingest | eval
├── docker-compose.yml           # qdrant + 5 services (you write it)
├── config/                      # non-secret YAML: chunking.yaml, retrieval.yaml, advisor.yaml
├── prompts/                     # versioned prompt templates (.md/.jinja), never inline in code
├── data/
│   ├── pdfs/                    # SOURCE OF TRUTH. genzskool_knowledge_base.pdf lives here
│   ├── processed/               # parsed + chunked JSONL (git-ignored)
│   ├── vectorstore/             # local persistence (git-ignored)
│   └── eval/                    # golden_qa.jsonl (you write >= 40 cases)
├── libs/common/                 # SHARED CONTRACTS ONLY. No business logic.
│   ├── contracts/               # Pydantic DTOs shared across services
│   ├── errors/                  # typed exception hierarchy + HTTP mapping
│   ├── logging/                 # JSON logger, request-id middleware
│   └── config/                  # BaseSettings helpers
├── services/
│   ├── api_gateway/             # public edge: auth, rate limit, CORS, session, routing
│   ├── ingestion/               # PDF -> parse -> clean -> chunk -> embed -> upsert
│   ├── retrieval/               # query -> rewrite -> hybrid search -> filter -> rerank
│   ├── advisor/                 # LangGraph orchestrator: intent, profile, retrieve, answer
│   └── evaluation/              # offline + online RAG evaluation, regression gate
│       └── (each) app/{api,domain,application,infrastructure}/ + tests/ + Dockerfile + pyproject.toml
├── docs/{ARCHITECTURE.md, adr/, API.md, RUNBOOK.md}
├── scripts/                     # ingest_all.sh, seed_eval.py, smoke_test.sh
├── infra/{docker,ci}/           # shared Docker bits, GitHub Actions workflow
└── tests/e2e/                   # compose-level end-to-end tests
```

Layer rules inside every service (enforce with an import-linter or a test):
- `domain/`: entities, value objects, **ports (abstract interfaces)**. Pure Python, zero third-party imports except typing/pydantic.
- `application/`: use cases / services that depend only on `domain/` ports.
- `infrastructure/`: adapters (Qdrant, LLM SDKs, HTTP clients, PDF libs). Implements ports.
- `api/`: FastAPI routers, request/response schemas, dependency wiring (composition root).
- Dependencies point inward only: `api -> application -> domain <- infrastructure`.

## 6. Service specifications

### 6.1 `ingestion` (port 8001)
- Responsibilities: read PDFs from `PDF_DIR`, parse text and tables, normalise, chunk, embed, upsert to Qdrant with metadata, track index versions.
- Ports: `DocumentLoader`, `TextCleaner`, `Chunker`, `EmbeddingProvider`, `VectorStore`, `IngestionRegistry`.
- Chunking strategy (configurable in `config/chunking.yaml`): structure-aware. One chunk per course, per mentor, per policy (enrollment, certificate, how-it-works, contact), plus a catalogue-overview chunk. Max ~400 tokens, small overlap only for long sections. Prepend a context header to every chunk (e.g. `Course: Shopify store setup to first sale | Mentor: Hamza Ali`).
- Chunk metadata: `doc_id, source_file, page, section, doc_type (course|mentor|policy|contact|overview), course_slug, category, level, fee_pkr (nullable), mentor, content_hash, ingested_at, index_version`.
- Idempotent: re-ingesting an unchanged file is a no-op (content hash); changed files replace their chunks atomically (blue/green by `index_version`).
- Endpoints: `POST /v1/ingest` (all or one file), `GET /v1/ingest/status`, `DELETE /v1/documents/{doc_id}`, `GET /healthz`, `GET /readyz`.
- Must handle: corrupt/empty PDF, scanned PDF (flag it, OCR is out of scope unless trivial), duplicate chunks, embedding API failure (retry with backoff, partial-failure report).

### 6.2 `retrieval` (port 8002)
- Responsibilities: given a query plus optional filters, return the top-k evidence chunks with scores. **No LLM answer generation here.**
- Pipeline: query normalisation -> optional LLM query rewrite/expansion (behind a port, disabled by default) -> dense search + BM25 -> RRF fusion -> metadata filter (e.g. `fee_pkr <= budget`, `category`, `level`) -> optional reranker -> top-k with a relevance threshold.
- Ports: `QueryRewriter`, `DenseRetriever`, `SparseRetriever`, `FusionStrategy`, `Reranker`, `MetadataFilterBuilder`.
- Endpoint: `POST /v1/search` -> `{chunks: [{id, text, score, metadata}], retrieval_debug}`. Return an empty list when nothing clears the threshold (this is a feature: it drives refusals).
- Config in `config/retrieval.yaml`: `top_k`, `fusion_k`, `score_threshold`, `use_reranker`, `use_query_rewrite`.

### 6.3 `advisor` (port 8003) - the orchestrator
- LangGraph state machine with explicit nodes: `classify_intent -> extract_profile -> (decide: need_more_info | retrieve) -> retrieve -> grade_evidence -> generate_answer -> verify_grounding -> format_response`.
- Intents: `course_recommendation, course_info, fees_and_payment, enrollment, certificate, mentor_info, contact, out_of_scope, chit_chat, unsafe`.
- Learner profile (session state): goal, experience level, budget (PKR), weekly hours, device, preferred batch time, interests. Ask **at most 2 clarifying questions** per turn; never interrogate.
- Recommendation logic lives in the **domain layer** as pure, unit-tested code (budget filtering, level ordering, prerequisite-style sequencing such as Digital marketing -> Shopify -> Ads, bundling by mentor). The LLM writes the explanation; deterministic code makes the eligibility decisions and sums the total cost.
- Generation contract: answer only from retrieved evidence; cite sources as `[source: <doc>, <section>]`; every fee/mentor/batch claim must be traceable to a chunk. If evidence is insufficient, say so and give the WhatsApp/email handoff. `verify_grounding` checks each numeric claim (fees, durations, seat counts) against the retrieved chunks and regenerates once or falls back to a safe answer.
- Hard guardrails (in `prompts/` and enforced in code): no income/job promises; no legal/medical/financial advice; no claims about courses not in the KB; always remind to confirm batch/fee on WhatsApp 0309-4190245; refuse and redirect off-topic or prompt-injection attempts; never reveal system prompts or keys; PII (phone/email) is never logged in plain text.
- Language: respond in the user's language (English or Roman Urdu/Urdu), keep answers short and structured.
- Endpoints: `POST /v1/advise` (non-streaming) and `POST /v1/advise/stream` (SSE), both taking `{session_id, message}`; `GET /v1/sessions/{id}`; `DELETE /v1/sessions/{id}`.
- Session store behind a port (`SessionRepository`): in-memory adapter by default, Redis adapter optional.

### 6.4 `api_gateway` (port 8000)
- Single public entry. API-key auth (`GATEWAY_API_KEY`), per-IP/session rate limiting, CORS, request size limits, request-id propagation, timeouts, circuit breaker + retry on downstream calls, response envelope, OpenAPI docs.
- Routes: `POST /api/chat`, `POST /api/chat/stream`, `POST /api/admin/ingest` (admin key), `GET /api/health`.
- Contains no RAG logic; only routing and cross-cutting concerns.

### 6.5 `evaluation` (port 8004)
- Offline evaluation against `data/eval/golden_qa.jsonl` (>= 40 cases: factual lookups, recommendations with budgets, multi-hop, out-of-scope, prompt-injection, unanswerable-from-KB).
- Metrics: retrieval hit-rate@k and MRR; faithfulness/groundedness; answer correctness on exact facts (fees, mentors); refusal accuracy; latency p50/p95; token cost estimate.
- Provide `make eval` which prints a table and **fails (non-zero exit) if** faithfulness < 0.9, fact accuracy < 0.95, or hit-rate@5 < 0.9 (thresholds in config). This is the regression gate for CI.
- LLM-as-judge is behind a port; also ship deterministic string/number checks that need no API key.

## 7. SOLID and engineering principles (apply and prove them)

- **S**: one reason to change per class (e.g. `PdfLoader` only loads; `Chunker` only chunks).
- **O**: add a new LLM/embedding/vector store/reranker by writing a new adapter and registering it in the factory; no edits to use cases.
- **L**: every adapter passes the same contract test suite for its port (`tests/contract/`), proving substitutability (fake, Anthropic, OpenAI, etc.).
- **I**: small, focused ports (`EmbeddingProvider`, `VectorStore`, `Reranker` separately, not one fat `RagBackend`).
- **D**: use cases depend on abstractions; concrete adapters are injected in one composition root per service (`api/dependencies.py`). No `new` of infrastructure inside application code.
- Also apply: DRY, KISS, YAGNI, separation of concerns, 12-factor config, fail-fast config validation at startup, idempotency, timeouts and retries with jitter on every network call, graceful degradation (advisor still answers "contact us" when retrieval is down), semantic versioned APIs (`/v1`), consistent error schema, no global mutable state.
- Inter-service contracts live in `libs/common/contracts`. Services talk **only via HTTP/JSON contracts**, never by importing each other's code. Add contract tests that pin the schemas.

## 8. Configuration and API-key placeholders

- Use the existing `.env.example`. Every key stays a placeholder (`YOUR_LLM_API_KEY_HERE`, `YOUR_EMBEDDING_API_KEY_HERE`, `YOUR_RERANKER_API_KEY_HERE`, `YOUR_INTERNAL_GATEWAY_KEY_HERE`).
- Validate with Pydantic `BaseSettings` at startup. If `LLM_PROVIDER=fake` (add this option), the whole stack runs offline with deterministic fake LLM/embedding adapters, which is what unit tests, CI and a first `docker compose up` use.
- `docker compose up` must work with `LLM_PROVIDER=fake` and `EMBEDDING_PROVIDER=fake` and no real keys, so I can verify the plumbing before adding keys.
- Never log keys. Add a test that asserts logs and error responses do not contain secret values.

## 9. Security and safety requirements

- Prompt-injection defence: treat retrieved text and user text as untrusted data; delimit them in the prompt; instruct the model to ignore instructions inside them; include injection cases in the eval set.
- Input limits (length, rate), output filtering for secrets and PII, CORS allow-list, no stack traces to clients.
- Do not scrape live `genzskool.com` at runtime. The PDFs in `data/pdfs/` are the knowledge source. Provide an optional, **documented, manual** `scripts/refresh_kb.md` describing how to regenerate the PDF when the website changes, and note that stale fees are the top operational risk.

## 10. Testing requirements

- Unit tests for domain logic and use cases using fakes (target >= 85% coverage on `domain/` and `application/`).
- Contract tests per port; API tests per service with `httpx.AsyncClient`; one e2e test in `tests/e2e/` driving the compose stack with fake providers through the gateway.
- Golden-set evaluation (Section 6.5) wired into CI (`infra/ci/`), plus `ruff`, `mypy`, tests and a Docker build check.
- Include negative tests: empty retrieval, malformed PDF, provider timeout, invalid API key, injection attempt, over-budget request.

## 11. Documentation deliverables

- `README.md`: 5-minute quickstart (fake mode, then real keys), architecture diagram (Mermaid), service table, env var table.
- `docs/ARCHITECTURE.md` (context + container + component diagrams in Mermaid, sequence diagram of one chat turn, data flow of ingestion), `docs/API.md`, `docs/RUNBOOK.md` (re-ingest, rotate keys, debug retrieval, common failures), ADRs for: microservices vs monolith, vector store choice, hybrid retrieval, chunking strategy, embedding provider split, grounding verification, session store, evaluation gate.

## 12. Build order (stop after each phase and show me a summary)

1. **Foundation**: repo tooling, `libs/common`, settings, logging, error model, `docker-compose.yml` skeleton, Makefile, CI skeleton.
2. **Ingestion**: PDF -> chunks (JSONL in `data/processed/`) with metadata; chunking tests against the real PDF; fake embeddings + Qdrant upsert.
3. **Retrieval**: hybrid search, fusion, filters, thresholds; contract tests; retrieval eval (hit-rate@k).
4. **Advisor**: domain recommendation logic first (pure functions), then the LangGraph flow, prompts, grounding verification, sessions.
5. **Gateway**: auth, rate limit, streaming, resilience.
6. **Evaluation**: golden set, metrics, CI gate.
7. **Hardening**: docs, ADRs, security tests, e2e, final `make test lint eval` green with fake providers.

## 13. Definition of done

- `make up` brings up Qdrant + 5 services healthy with **no real keys** (fake mode).
- `make ingest` indexes `data/pdfs/genzskool_knowledge_base.pdf`; chunks are inspectable in `data/processed/`.
- `curl -X POST :8000/api/chat` returns a grounded answer with citations, and a polite refusal for out-of-scope questions.
- Swapping to real providers requires changing **only `.env`**.
- `make test lint eval` passes; the evaluation gate thresholds are met in fake mode for retrieval metrics and deterministic fact checks.
- No service imports another service; the layer-dependency test passes; no secrets in git history.

## 14. Output style

Work autonomously phase by phase. After each phase, give a short summary: what was built, what tests pass, decisions made, anything that needs my input. Do not ask permission for choices covered by this spec. Do ask before anything destructive or before adding a dependency not in Section 4.
