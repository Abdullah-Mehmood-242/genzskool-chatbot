# ADR-0008: Evaluation gate with deterministic checks

- Status: Accepted
- Date: 2026-11-12

## Context
RAG regressions (a chunking change dropping the eBay fee, a prompt change adding income
promises) must fail CI, without needing API keys.

## Decision
- Golden set `data/eval/golden_qa.jsonl` (>= 40 cases) with, per case: `question`, `category`,
  `expected_doc_ids` (for retrieval), `must_contain` / `must_not_contain` facts, `expect_refusal`.
- Metrics: hit-rate@k, MRR, faithfulness (share of answer numeric claims grounded in evidence),
  fact accuracy (`must_contain` satisfied), refusal accuracy, latency p50/p95, token cost estimate.
- Deterministic checks run always; LLM-as-judge sits behind a `Judge` port and is off by default.
- `make eval` fails (exit 1) if faithfulness < 0.9, fact accuracy < 0.95 or hit-rate@5 < 0.9.
  Thresholds in `config/evaluation.yaml`.
- The evaluation can run in-process against the use cases (CI, no Docker) or over HTTP against
  running services (staging).

## Consequences
- (+) Every PR is gated on grounded facts.
- (-) In fake mode, faithfulness of the fake LLM is high by construction; it validates plumbing
  and retrieval, not real model behaviour. Run `make eval` with real keys before releases.
