"""The shared fixture tests/fixtures/kb/ must equal what this service's pipeline produces today.

Retrieval (and later advisor/evaluation) tests search over that fixture. If the chunker changes,
regenerate it with `python scripts/dev.py fixtures` and review the diff.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "kb"


def test_fixture_matches_current_chunker_output() -> None:
    spec = importlib.util.spec_from_file_location("build_fixtures", REPO / "scripts" / "build_fixtures.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.render(module.build_chunks())
    for name, content in expected.items():
        path = FIXTURE / name
        assert path.exists(), f"missing {name}: run `python scripts/dev.py fixtures`"
        on_disk = path.read_text(encoding="utf-8")
        if name.endswith(".json"):
            assert json.loads(on_disk) == json.loads(content), (
                "stale registry fixture: run `python scripts/dev.py fixtures`"
            )
        else:
            assert on_disk == content, "stale chunk fixture: run `python scripts/dev.py fixtures` and review the diff"
    assert sorted(p.name for p in FIXTURE.iterdir()) == sorted(expected)
