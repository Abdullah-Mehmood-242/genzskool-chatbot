"""Spec-mandated structure, docs, tooling and CI shape (Sections 4, 5, 10, 11 of MASTER_PROMPT)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SERVICES = ["api_gateway", "ingestion", "retrieval", "advisor", "evaluation"]
LAYERS = ["api", "application", "domain", "infrastructure"]


@pytest.mark.parametrize("svc", SERVICES)
def test_service_layout(svc: str) -> None:
    base = ROOT / "services" / svc
    for layer in LAYERS:
        assert (base / "app" / layer / "__init__.py").is_file(), f"{svc}/{layer}"
    for f in (
        "app/main.py",
        "app/settings.py",
        "app/api/dependencies.py",
        "Dockerfile",
        "pyproject.toml",
        "requirements.txt",
    ):
        assert (base / f).is_file(), f"{svc}/{f}"
    assert list((base / "tests").glob("test_*.py"))


@pytest.mark.parametrize("svc", SERVICES)
def test_service_pyproject(svc: str) -> None:
    data = tomllib.loads((ROOT / "services" / svc / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "genz-" + svc.replace("_", "-")
    assert data["project"]["requires-python"] == ">=3.12"
    assert data["project"]["dynamic"] == ["dependencies"]
    assert data["tool"]["setuptools"]["dynamic"]["dependencies"]["file"] == ["requirements.txt"]


def test_common_lib_layout() -> None:
    for sub in ("contracts", "errors", "logging", "config", "resilience", "observability"):
        assert (ROOT / "libs" / "common" / sub / "__init__.py").is_file()
    assert (ROOT / "libs" / "common" / "py.typed").is_file()


def test_common_lib_has_no_business_logic_keywords() -> None:
    """libs/common is contracts-only: no course names, fees or advisor rules."""
    banned = re.compile(r"(?i)\b(shopify|tiktok|ebay|hamza|sana iqbal|30,000|budget_filter|def recommend\w*\()")
    for path in (ROOT / "libs" / "common").rglob("*.py"):
        if "tests" in path.parts:
            continue
        assert not banned.search(path.read_text(encoding="utf-8")), path


def test_root_tooling_config() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["ruff"]["target-version"] == "py312"
    assert data["tool"]["mypy"]["strict"] is True
    assert data["tool"]["mypy"]["python_version"] == "3.12"
    assert data["tool"]["pytest"]["ini_options"]["asyncio_mode"] == "auto"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_makefile_has_spec_targets() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z][a-z-]*):", text, flags=re.MULTILINE))
    assert {"up", "down", "test", "lint", "ingest", "eval"} <= targets
    assert "--wait" in text  # make up waits for health checks


def test_makefile_uses_tabs() -> None:
    for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if line.startswith(" ") and line.strip() and not line.strip().startswith("#"):
            pytest.fail(f"recipe line must start with a tab: {line!r}")


def test_ci_workflow_shape() -> None:
    ci = yaml.safe_load((ROOT / "infra/ci/ci.yml").read_text(encoding="utf-8"))
    assert set(ci["jobs"]) == {"quality", "docker-build", "e2e"}
    assert ci["env"]["LLM_PROVIDER"] == "fake"
    assert ci["env"]["EMBEDDING_PROVIDER"] == "fake"
    assert ci["env"]["PYTHON_VERSION"] == "3.12"
    steps = " ".join(s.get("run", "") for s in ci["jobs"]["quality"]["steps"])
    for cmd in ("dev.py lint", "dev.py typecheck", "dev.py test", "dev.py eval"):
        assert cmd in steps
    assert ci["jobs"]["docker-build"]["strategy"]["matrix"]["service"] == SERVICES
    assert "secrets." not in (ROOT / "infra/ci/ci.yml").read_text(encoding="utf-8"), "CI must not need secrets"


def test_precommit_guards() -> None:
    cfg = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    ids = {h["id"] for repo in cfg["repos"] for h in repo["hooks"]}
    assert {"detect-private-key", "check-added-large-files", "ruff", "mypy", "no-env-file"} <= ids


def test_dockerignore_excludes_secrets_and_bulk() -> None:
    lines = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".env", ".venv", ".git", "data"} <= lines


def test_gitattributes_keeps_pdf_binary_and_text_lf() -> None:
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.pdf binary" in text
    assert "* text=auto eol=lf" in text


REQUIRED_ADR_TOPICS = {
    "microservices": "microservices",
    "vector store": "vector-store",
    "hybrid retrieval": "hybrid-retrieval",
    "chunking": "chunking",
    "embedding provider split": "embedding-provider",
    "grounding verification": "grounding",
    "session store": "session-store",
    "evaluation gate": "evaluation-gate",
}


@pytest.mark.parametrize(("topic", "slug"), REQUIRED_ADR_TOPICS.items())
def test_required_adr_exists(topic: str, slug: str) -> None:
    matches = list((ROOT / "docs" / "adr").glob(f"*{slug}*.md"))
    assert len(matches) == 1, f"ADR for {topic}"
    text = matches[0].read_text(encoding="utf-8")
    for heading in ("## Context", "## Decision", "## Consequences"):
        assert heading in text, f"{matches[0].name} lacks {heading}"
    assert "Status: Accepted" in text


def test_adr_numbering_is_contiguous() -> None:
    nums = sorted(int(p.name[:4]) for p in (ROOT / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md"))
    assert nums == list(range(1, len(nums) + 1))


def test_architecture_doc_links_every_adr_and_has_diagrams() -> None:
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for p in (ROOT / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md"):
        assert f"| {p.name[:4]} |" in text, f"ARCHITECTURE.md does not list ADR {p.name[:4]}"
    assert text.count("```mermaid") >= 5
    for kind in ("sequenceDiagram", "stateDiagram", "flowchart"):
        assert kind in text


def test_ports_documented_consistently() -> None:
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for svc, port in zip(SERVICES, [8000, 8001, 8002, 8003, 8004], strict=True):
        assert f"`{svc}` | {port}" in text
