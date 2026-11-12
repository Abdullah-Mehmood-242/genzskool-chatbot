# ADR-0011: Auto-ingest (startup run, folder watcher, drift self-heal)

- Status: Accepted
- Date: 2026-11-12

## Context
The spec ships `POST /v1/ingest` and `make ingest`. Relying on a human to run them means a fresh
`docker compose up` serves an empty index, and an updated PDF keeps serving stale fees (the top
operational risk) until someone remembers. A wiped Qdrant volume would silently empty retrieval.

## Decision
The ingestion service runs an `AutoIngestor` (application layer) in its FastAPI lifespan:

| Trigger | Behaviour |
|---|---|
| Startup (`AUTO_INGEST_ON_STARTUP=true`) | `ingest()` all PDFs in the background. `/readyz` reports `startup_ingest=false` until it has run; `/healthz` stays green so startup is never blocked. |
| Folder watcher (`AUTO_INGEST_INTERVAL_S`, default 15 in compose, 0 = off) | Every interval, compare cheap fingerprints (name, size, mtime_ns). Any difference triggers `ingest()`. Content hashes keep a touched-but-identical file a no-op. |
| Deletion (`AUTO_INGEST_PRUNE=true`) | Documents whose PDF disappeared are removed from Qdrant, the registry and `data/processed/`. **Guard:** if the folder is empty while documents are indexed, pruning is skipped (a broken mount is likelier than intent). |
| Index drift | Each tick checks that the active version's point count matches the registry. If Qdrant lost points (e.g. a wiped volume), the documents are re-ingested. |

- Polling, not inotify: it works identically on Linux, Windows bind mounts and Docker Desktop, and
  needs no new dependency.
- Failures (e.g. embedding provider down, a corrupt PDF) never stop the loop. A failed run leaves a
  *retry pending*: the next tick re-runs even if the folder looks unchanged (trigger `retry`). This
  also covers "bad file added, then removed", where the folder returns to its last good state. The
  interval backs off as `interval * 2^failures`, capped at 8x.
- Manual `POST /v1/ingest` and auto-ingest share one `asyncio.Lock`: never two ingests at once.
- State is visible in `GET /v1/ingest/status` (`auto_ingest`: runs, last trigger, last error, failures).

## Consequences
- (+) `docker compose up` alone yields a populated index; dropping a new PDF into `data/pdfs/`
  makes it searchable within one interval; deleting one stops it being served.
- (+) Self-healing after Qdrant data loss.
- (-) A poll every 15 s (a directory listing plus one count query per document): negligible at
  this corpus size.
- (-) One ingestion replica must own auto-ingest. Running several would race (harmlessly, thanks to
  idempotency and the lock per process, but wastefully). Scale retrieval, not ingestion.
