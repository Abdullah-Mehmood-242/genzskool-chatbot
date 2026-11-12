"""Retrieval evaluation gate on the real KB (fake embeddings): hit-rate@5, MRR, empty-on-off-topic.

Cases live in data/eval/retrieval_cases.jsonl so the evaluation service can reuse them (Phase 6).
Thresholds mirror config/evaluation.yaml (hit_rate_at_5 >= 0.9).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import CONFIG_DIR, EVAL_CASES

from app.api.dependencies import build_container
from app.domain.models import Corpus
from app.infrastructure.corpus import StaticCorpusProvider
from app.settings import Settings
from common.config import load_settings
from common.contracts import SearchRequest

CASES: list[dict[str, Any]] = [json.loads(line) for line in EVAL_CASES.read_text(encoding="utf-8").splitlines() if line]
POSITIVE = [c for c in CASES if not c["expect_empty"]]
NEGATIVE = [c for c in CASES if c["expect_empty"]]
THRESHOLDS = yaml.safe_load((CONFIG_DIR / "evaluation.yaml").read_text(encoding="utf-8"))["thresholds"]
K = 5


@pytest.fixture(scope="module")
def results(kb_corpus: Corpus) -> dict[str, list[str]]:
    import asyncio

    # Module-scoped: runs before the function-scoped env fixture, so settings are explicit here.
    settings = load_settings(
        Settings,
        _env_file=None,
        config_dir=CONFIG_DIR,
        vector_store="memory",
        embedding_provider="fake",
        embedding_dim=1024,
    )
    container = build_container(settings, corpus=StaticCorpusProvider(kb_corpus))

    async def run() -> dict[str, list[str]]:
        out = {}
        for case in CASES:
            resp = await container.service.search(SearchRequest(query=case["query"], top_k=K))
            out[case["id"]] = [c.metadata.section for c in resp.chunks]
        return out

    return asyncio.run(run())


def test_eval_set_is_well_formed() -> None:
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert len(POSITIVE) >= 40
    assert len(NEGATIVE) >= 5
    from conftest import load_fixture_chunks

    sections = {c.metadata.section for c in load_fixture_chunks()}
    for case in POSITIVE:
        assert case["expected_sections"], case["id"]
        assert set(case["expected_sections"]) <= sections, f"{case['id']} expects unknown sections"


def test_hit_rate_at_5_meets_gate(results: dict[str, list[str]]) -> None:
    hits = [c["id"] for c in POSITIVE if set(results[c["id"]][:K]) & set(c["expected_sections"])]
    rate = len(hits) / len(POSITIVE)
    misses = {c["id"]: results[c["id"]] for c in POSITIVE if c["id"] not in hits}
    assert rate >= THRESHOLDS["hit_rate_at_5"], f"hit@5={rate:.3f}; misses={misses}"


def test_mrr(results: dict[str, list[str]]) -> None:
    total = 0.0
    for case in POSITIVE:
        ranks = [i for i, s in enumerate(results[case["id"]][:K], 1) if s in case["expected_sections"]]
        total += 1 / ranks[0] if ranks else 0.0
    mrr = total / len(POSITIVE)
    assert mrr >= 0.8, f"MRR={mrr:.3f}"


def test_off_topic_queries_return_empty(results: dict[str, list[str]]) -> None:
    leaked = {c["id"]: results[c["id"]] for c in NEGATIVE if results[c["id"]]}
    assert not leaked, f"off-topic queries returned evidence: {leaked}"


@pytest.mark.parametrize("case", POSITIVE, ids=[c["id"] for c in POSITIVE])
def test_case(results: dict[str, list[str]], case: dict[str, Any]) -> None:
    """Per-case visibility: which exact query regressed (the aggregate gate above is what must pass)."""
    got = results[case["id"]][:K]
    assert set(got) & set(case["expected_sections"]), f"{case['query']!r} -> {got}"
