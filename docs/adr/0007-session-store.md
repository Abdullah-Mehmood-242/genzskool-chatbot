# ADR-0007: Session store behind a port

- Status: Accepted
- Date: 2026-11-12

## Context
The advisor keeps a learner profile and recent turns per `session_id`.

## Decision
`SessionRepository` port with an `InMemorySessionRepository` (default, TTL eviction, max
sessions bound) and an optional `RedisSessionRepository` adapter, selected by `SESSION_STORE`
(`memory | redis`). Redis is not added to the default compose stack (YAGNI); the adapter imports
`redis` lazily.

PII: the profile stores no phone or email. Session contents are never logged; the log redaction
filter masks phone numbers and emails anywhere in log lines.

## Consequences
- (+) Zero extra infrastructure for development.
- (-) In-memory sessions are lost on restart and are per-replica. Run one advisor replica or use
  Redis for horizontal scaling.
