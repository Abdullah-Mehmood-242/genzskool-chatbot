# ADR-0013: Advisor flow, LLM adapters and streaming after verification

- Status: Accepted
- Date: 2026-11-13

## Context
The advisor must give grounded, short answers in English or Roman Urdu, never promise income, never
invent fees, and keep working without keys (CI) and when retrieval is down.

## Decision
**Flow.** A LangGraph state machine with the spec's nodes:
`classify_intent -> extract_profile -> (need_more_info | retrieve) -> retrieve -> grade_evidence ->
generate_answer -> verify_grounding -> format_response`. Refusals (unsafe, out of scope, regulated
advice) and greetings go straight to `format_response`: no retrieval, no LLM call.

**Deterministic first.** Intent, language, the learner profile (budget like "30k"/"30 hazar"),
eligibility, ordering, totals and mentor bundles are pure domain code (`understanding.py`,
`recommend.py`), unit-tested on the real catalogue. The LLM receives the decided plan and only
words it. At most two clarifying questions, asked only when no interest is known and the goal does
not already imply a path, and never asked twice.

**Grounding.** `verify_grounding` checks every number, mentor name, promise, regulated-advice phrase
and prompt-leak in a draft (ADR-0006). One regeneration with the violations listed; then the
deterministic `AnswerComposer`, which builds answers only from evidence lines and plan numbers
(grounded by construction; a property test covers every intent x profile x language). LLM errors,
refusals (`stop_reason == "refusal"`) and empty drafts use the composer too.

**Injection defence.** Evidence, history and the learner's message are HTML-escaped inside
`<evidence>`, `<conversation>` and `<user_message>` tags, and the system prompt says they are data.
Messages that try to override instructions, ask for keys, or forge our tags are refused before any
prompt is built. Answers are PII-filtered (only the official contacts survive); session history
stores learners' messages with phones/emails redacted; queries are never logged.

**LLM adapters** (`LLMClient` port, one factory):
- `fake`: the composer; deterministic, used offline and in CI.
- `anthropic`: the official `anthropic` SDK (pinned 2.8.0, API version `2026-10-21`). No
  `temperature` (rejected by current models), `output_config.effort` (default `low`: short chat
  answers), default `max_tokens` 16000 because thinking counts toward it, and the server-side refusal
  fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`) on the models that support
  it, switchable with `LLM_REFUSAL_FALLBACK`. Recommended model: `claude-opus-5-5`.
- `openai` (Chat Completions, or any OpenAI-compatible URL) and `ollama`: plain `httpx`.
All map failures to `TransientProviderError` / `UpstreamTimeout` (retryable) or `ProviderError`.

**Streaming.** `POST /v1/advise/stream` (SSE) emits `status` events per graph step, then the
*verified* answer in pieces, then `done` with the full response. LLM tokens are not streamed raw:
showing an unverified fee and then retracting it would defeat the grounding guarantee.

**Degradation.** Retrieval is called through a client with timeouts, retry with jitter and a circuit
breaker. If it fails, the advisor answers with the WhatsApp/email hand-off (HTTP 200), and
retrieval is deliberately not part of the advisor's readiness check.

## Consequences
- (+) Fees, totals and eligibility are always right or the answer falls back to evidence lines.
- (+) Fully testable offline; the same tests drive the real adapters through mocked transports.
- (-) Time-to-first-token on the stream equals the verification time (sub-second with the fake;
  one LLM round trip with a real model).
- (-) Rule-based intent and profile extraction need the keyword lists in `config/advisor.yaml`
  kept current when courses change.
