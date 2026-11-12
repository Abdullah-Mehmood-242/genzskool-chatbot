from __future__ import annotations

from pathlib import Path

from common.config import ServiceSettings


class Settings(ServiceSettings):
    service_name: str = "evaluation"
    retrieval_url: str = "http://retrieval:8002"
    advisor_url: str = "http://advisor:8003"
    golden_set_path: Path = Path("data/eval/golden_qa.jsonl")
