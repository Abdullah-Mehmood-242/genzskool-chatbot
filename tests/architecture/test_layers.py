"""Enforce hexagonal layer rules and service isolation across all services."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from layer_rules import check_service

ROOT = Path(__file__).resolve().parents[2]
SERVICES = sorted(p for p in (ROOT / "services").iterdir() if (p / "app").is_dir())


def test_all_five_services_present() -> None:
    assert {p.name for p in SERVICES} == {"api_gateway", "ingestion", "retrieval", "advisor", "evaluation"}


@pytest.mark.parametrize("service", SERVICES, ids=lambda p: p.name)
def test_layer_dependencies_point_inward(service: Path) -> None:
    violations = check_service(service, ROOT)
    assert not violations, "\n".join(str(v) for v in violations)


@pytest.mark.parametrize("service", SERVICES, ids=lambda p: p.name)
def test_composition_root_exists(service: Path) -> None:
    assert (service / "app" / "api" / "dependencies.py").is_file()


def _make_service(tmp: Path, files: dict[str, str]) -> Path:
    svc = tmp / "services" / "demo"
    for rel, body in files.items():
        f = svc / "app" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return svc


@pytest.mark.parametrize(
    ("files", "expected_reason"),
    [
        ({"domain/x.py": "import httpx\n"}, "third-party"),
        ({"domain/x.py": "from app.infrastructure.db import Repo\n"}, "layer boundary"),
        ({"application/x.py": "from app.api.routes import r\n"}, "layer boundary"),
        ({"application/x.py": "import qdrant_client\n"}, "third-party"),
        ({"infrastructure/x.py": "from app.api import deps\n"}, "layer boundary"),
        ({"domain/x.py": "from common.observability import create_service_app\n"}, "layer boundary"),
        ({"api/x.py": "from services.retrieval.app import main\n"}, "must not import each other"),
        ({"domain/x.py": "from . import y\n"}, "relative"),
    ],
)
def test_checker_detects_violations(tmp_path: Path, files: dict[str, str], expected_reason: str) -> None:
    svc = _make_service(tmp_path, files)
    violations = check_service(svc, tmp_path)
    assert violations
    assert expected_reason in violations[0].reason


def test_checker_accepts_valid_layout(tmp_path: Path) -> None:
    domain_src = "\n".join(
        ["from abc import ABC", "from pydantic import BaseModel", "from common.contracts import Chunk"]
    )
    svc = _make_service(
        tmp_path,
        {
            "domain/ports.py": domain_src,
            "application/uc.py": "from app.domain.ports import X\nfrom langgraph.graph import StateGraph\n",
            "infrastructure/a.py": "import httpx\nfrom app.domain.ports import X\nfrom common.logging import redact\n",
            "api/dependencies.py": "from app.infrastructure.a import A\nfrom app.application.uc import U\n",
        },
    )
    assert check_service(svc, tmp_path) == []
