# ADR-0009: Supporting dependencies outside Section 4

- Status: Accepted
- Date: 2026-11-12

## Context
Section 4 lists the main stack. Some requirements imply small libraries that are not named.

## Decision
| Package | Why it is required | Alternative rejected |
|---|---|---|
| `pydantic-settings` | Spec requires Pydantic `BaseSettings`; in Pydantic v2 it lives in this package. | none |
| `PyYAML` (+ `types-PyYAML`) | Spec requires YAML config in `config/`. | TOML (spec says YAML) |
| `prometheus-client` | Spec requires `/metrics` in Prometheus format. | Hand-written exposition format (error-prone) |
| `pytest-cov` | Spec sets a >= 85% coverage target on domain/application. | none |

Not added: `import-linter` (layer rules are enforced by an AST-based test instead),
`tenacity` (retry with jitter is ~30 lines in `libs/common`), `redis` (optional adapter, lazy import).

## Consequences
All four are small, widely used and pinned in each service's `pyproject.toml`.
