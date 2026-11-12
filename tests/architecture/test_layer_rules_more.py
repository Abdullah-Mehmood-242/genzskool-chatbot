"""Exhaustive cases for the layer checker itself."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from layer_rules import Violation, check_service


def _svc(tmp: Path, files: dict[str, str]) -> Path:
    svc = tmp / "services" / "demo"
    for rel, body in files.items():
        f = svc / "app" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return svc


VIOLATIONS = [
    # domain
    ("domain/x.py", "import app\n", "layer boundary"),
    ("domain/x.py", "import fastapi\n", "third-party"),
    ("domain/x.py", "from app.application.uc import U\n", "layer boundary"),
    ("domain/x.py", "from app.settings import Settings\n", "layer boundary"),
    ("domain/x.py", "from common.logging import redact\n", "layer boundary"),
    ("domain/x.py", "from common.config import load_yaml\n", "layer boundary"),
    ("domain/x.py", "import langgraph\n", "third-party"),
    ("domain/sub/deep.py", "import httpx\n", "third-party"),
    # application
    ("application/x.py", "from app.infrastructure.a import A\n", "layer boundary"),
    ("application/x.py", "from app.settings import Settings\n", "layer boundary"),
    ("application/x.py", "from common.observability import create_service_app\n", "layer boundary"),
    ("application/x.py", "from common.config import load_settings\n", "layer boundary"),
    ("application/x.py", "import httpx\n", "third-party"),
    ("application/x.py", "import langchain\n", "third-party"),
    # infrastructure
    ("infrastructure/x.py", "from app.application.uc import U\n", "layer boundary"),
    ("infrastructure/x.py", "from app.settings import Settings\n", "layer boundary"),
    ("infrastructure/x.py", "from app.api.dependencies import build_container\n", "layer boundary"),
    # isolation and structure
    ("main.py", "import services.retrieval.app.main\n", "must not import each other"),
    ("api/x.py", "from services import retrieval\n", "must not import each other"),
    ("infrastructure/x.py", "from .y import z\n", "relative"),
    ("api/x.py", "from ..domain import y\n", "relative"),
    ("utils/helpers.py", "import os\n", "must live in"),
    ("rogue.py", "import os\n", "must live in"),
]


@pytest.mark.parametrize(("rel", "src", "reason"), VIOLATIONS, ids=[f"{r}:{s.strip()}" for r, s, _ in VIOLATIONS])
def test_violation_detected(tmp_path: Path, rel: str, src: str, reason: str) -> None:
    violations = check_service(_svc(tmp_path, {rel: src}), tmp_path)
    assert len(violations) == 1, violations
    assert reason in violations[0].reason


ALLOWED = [
    ("domain/x.py", "from __future__ import annotations\nimport abc, dataclasses, re, typing\n"),
    ("domain/x.py", "import os.path\nfrom collections.abc import Sequence\n"),
    (
        "domain/x.py",
        "from pydantic import BaseModel\nfrom common.contracts import Chunk\nfrom common.errors import NotFound\n",
    ),
    ("domain/x.py", "import app.domain.models as m\nfrom app.domain import ports\n"),
    ("application/x.py", "from common.resilience import retry_async\nfrom common.logging import get_request_id\n"),
    ("application/x.py", "from app.domain.ports import P\nfrom app.application.other import O\n"),
    ("infrastructure/x.py", "import qdrant_client\nimport httpx\nfrom common.config import is_placeholder\n"),
    ("infrastructure/x.py", "from app.domain.ports import P\nfrom app.infrastructure.base import B\n"),
    ("api/x.py", "from app.settings import Settings\nfrom app.infrastructure.a import A\nimport fastapi\n"),
    ("main.py", "from app.api.dependencies import build_container\nimport uvicorn\n"),
    ("settings.py", "from common.config import ServiceSettings\nimport pydantic\n"),
    ("__init__.py", '"""doc"""\n'),
]


@pytest.mark.parametrize(("rel", "src"), ALLOWED, ids=[r for r, _ in ALLOWED])
def test_allowed_imports(tmp_path: Path, rel: str, src: str) -> None:
    assert check_service(_svc(tmp_path, {rel: src}), tmp_path) == []


def test_all_violations_in_a_file_are_reported_with_lines(tmp_path: Path) -> None:
    src = "import os\nimport httpx\nimport qdrant_client\nfrom app.api import x\n"
    violations = check_service(_svc(tmp_path, {"domain/x.py": src}), tmp_path)
    assert [v.line for v in violations] == [2, 3, 4]


def test_imports_inside_functions_are_checked(tmp_path: Path) -> None:
    src = "def f():\n    import httpx\n    return httpx\n"
    assert len(check_service(_svc(tmp_path, {"domain/x.py": src}), tmp_path)) == 1


def test_violation_str_is_actionable() -> None:
    v = Violation("services/a/app/domain/x.py", 3, "domain", "httpx", "third-party import not allowed in this layer")
    assert (
        str(v) == "services/a/app/domain/x.py:3 [domain] imports 'httpx': third-party import not allowed in this layer"
    )


def test_syntax_errors_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(SyntaxError):
        check_service(_svc(tmp_path, {"domain/x.py": "def broken(:\n"}), tmp_path)


def test_empty_service_has_no_violations(tmp_path: Path) -> None:
    (tmp_path / "services" / "demo" / "app").mkdir(parents=True)
    assert check_service(tmp_path / "services" / "demo", tmp_path) == []
