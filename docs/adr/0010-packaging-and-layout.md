# ADR-0010: Shared-library packaging and per-service `app` package

- Status: Accepted
- Date: 2026-11-12

## Context
The scaffold places every service's code in `services/<name>/app/`, so all five services use the
same top-level package name `app`. The shared library lives in `libs/common/`.

## Decision
- `libs/common` is the importable package `common` (`common.contracts`, `common.errors`, ...),
  reached by putting `libs/` on `PYTHONPATH`. In Docker it is copied to `/app/libs/common`.
- Each service is tested, linted and type-checked **in its own process** with
  `PYTHONPATH=libs:services/<name>`, so the duplicate `app` package name never collides. The
  Makefile loops over services.
- Each service pins exact `==` versions in `requirements.txt`, which is the single source:
  `pyproject.toml` reads it dynamically and the Docker build installs it directly. `uv`/`pip-tools`
  are not required.
- Besides the scaffolded `contracts/ errors/ logging/ config/`, `libs/common` adds two
  cross-cutting (non-business) subpackages: `resilience/` (retry with jitter, circuit breaker)
  and `observability/` (service app factory: request id, body limit, access log, `/healthz`,
  `/readyz`, `/metrics`). Keeping them shared avoids five copies (DRY).
- `common/logging` shadows the stdlib name if `libs/common` itself is put on `sys.path`; only
  `libs/` may be on the path (the task runner enforces this).
- Python 3.12 everywhere (spec): Docker images use `python:3.12-slim`, the local venv is 3.12
  (`.python-version`), and ruff/mypy target `py312`.

## Consequences
- (+) Matches the scaffold without renaming.
- (-) `pytest` cannot be run once over the whole repo; use `make test`.
