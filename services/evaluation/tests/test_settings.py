from __future__ import annotations

from pathlib import Path

from app.settings import Settings
from common.config import load_settings


def test_defaults() -> None:
    s = load_settings(Settings, _env_file=None)
    assert s.service_name == "evaluation"
    assert s.golden_set_path == Path("data/eval/golden_qa.jsonl")
    assert s.retrieval_url.endswith(":8002")
    assert s.advisor_url.endswith(":8003")
