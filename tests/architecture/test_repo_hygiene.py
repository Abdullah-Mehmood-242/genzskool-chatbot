"""Static checks on packaging, deployment and secret hygiene (no Docker needed)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PORTS = {"api_gateway": 8000, "ingestion": 8001, "retrieval": 8002, "advisor": 8003, "evaluation": 8004}
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _requirements(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


@pytest.mark.parametrize(
    "req_file",
    [
        *sorted((ROOT / "services").glob("*/requirements.txt")),
        ROOT / "requirements-dev.txt",
        ROOT / "libs/common/requirements.txt",
    ],
    ids=lambda p: p.relative_to(ROOT).as_posix(),
)
def test_every_dependency_is_pinned_exactly(req_file: Path) -> None:
    reqs = _requirements(req_file)
    assert reqs, f"{req_file} is empty"
    unpinned = [r for r in reqs if not re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[A-Za-z0-9_.+\-]+", r)]
    assert not unpinned, f"unpinned in {req_file.name}: {unpinned}"


def test_common_requirements_are_a_subset_of_every_service() -> None:
    common = set(_requirements(ROOT / "libs/common/requirements.txt"))
    for svc in PORTS:
        missing = common - set(_requirements(ROOT / "services" / svc / "requirements.txt"))
        assert not missing, f"{svc} misses common deps {missing}"


def test_same_package_has_same_version_everywhere() -> None:
    versions: dict[str, set[str]] = {}
    for f in [*ROOT.glob("services/*/requirements.txt"), ROOT / "requirements-dev.txt"]:
        for r in _requirements(f):
            name, ver = r.split("==")
            versions.setdefault(name.split("[")[0].lower(), set()).add(ver)
    drift = {k: v for k, v in versions.items() if len(v) > 1}
    assert not drift, f"version drift: {drift}"


@pytest.mark.parametrize(("svc", "port"), PORTS.items())
def test_dockerfile_matches_service(svc: str, port: int) -> None:
    text = (ROOT / "services" / svc / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12-slim" in text
    assert f"EXPOSE {port}" in text
    assert f"SERVICE_PORT={port}" in text
    assert f"COPY services/{svc}/app /app/service/app" in text
    assert f"COPY services/{svc}/requirements.txt" in text
    assert "USER app" in text, "containers must not run as root"
    assert "HEALTHCHECK" in text
    assert "--factory" in text
    # `exec` makes uvicorn PID 2 under tini (not a child of sh) so SIGTERM reaches it: graceful shutdown.
    assert 'CMD ["sh", "-c", "exec uvicorn ' in text


@pytest.mark.parametrize(("svc", "port"), PORTS.items())
def test_compose_service_wiring(svc: str, port: int) -> None:
    service = COMPOSE["services"][svc]
    assert service["build"]["dockerfile"] == f"services/{svc}/Dockerfile"
    assert service["build"]["context"] == "."
    published = [p for p in service["ports"] if p.endswith(f":{port}")]
    assert published, f"{svc} must publish {port}"
    if svc != "api_gateway":
        assert all(p.startswith("127.0.0.1:") for p in service["ports"]), "internal services bind to localhost only"
    assert all(dep["condition"] == "service_healthy" for dep in service.get("depends_on", {}).values())


def test_compose_dependency_graph() -> None:
    deps = {name: set(s.get("depends_on", {})) for name, s in COMPOSE["services"].items()}
    assert deps["api_gateway"] == {"advisor", "ingestion"}
    assert deps["advisor"] == {"retrieval"}
    assert deps["retrieval"] == {"qdrant"}
    assert deps["ingestion"] == {"qdrant"}
    assert "healthcheck" in COMPOSE["services"]["qdrant"]


def test_compose_defaults_to_fake_providers() -> None:
    raw = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "LLM_PROVIDER: ${LLM_PROVIDER:-fake}" in raw
    assert "EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-fake}" in raw


def test_ingestion_and_retrieval_share_embedding_config() -> None:
    env_i = COMPOSE["services"]["ingestion"]["environment"]
    env_r = COMPOSE["services"]["retrieval"]["environment"]
    for key in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIM", "QDRANT_COLLECTION"):
        assert env_i[key] == env_r[key], key


def _env_example() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def test_env_example_keys_are_placeholders_only() -> None:
    env = _env_example()
    secret_keys = [k for k in env if k.endswith(("_API_KEY", "_KEY")) and k != "QDRANT_API_KEY"]
    assert {"LLM_API_KEY", "EMBEDDING_API_KEY", "RERANKER_API_KEY", "GATEWAY_API_KEY", "ADMIN_API_KEY"} <= set(
        secret_keys
    )
    for k in secret_keys:
        assert re.fullmatch(r"YOUR_[A-Z_]+_HERE", env[k]), f"{k} must be a placeholder"
    assert env["QDRANT_API_KEY"] == ""


def test_env_example_defaults_to_offline_mode() -> None:
    env = _env_example()
    assert env["LLM_PROVIDER"] == "fake"
    assert env["EMBEDDING_PROVIDER"] == "fake"


@pytest.mark.parametrize("name", ["chunking", "retrieval", "advisor", "evaluation"])
def test_config_yaml_is_a_mapping(name: str) -> None:
    data = yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data


def test_evaluation_thresholds_match_spec() -> None:
    t = yaml.safe_load((ROOT / "config/evaluation.yaml").read_text(encoding="utf-8"))["thresholds"]
    assert t == {"faithfulness": 0.90, "fact_accuracy": 0.95, "hit_rate_at_5": 0.90}


def test_knowledge_base_pdf_present() -> None:
    assert (ROOT / "data/pdfs/genzskool_knowledge_base.pdf").stat().st_size > 1000


def test_gitignore_protects_secrets_and_artifacts() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (".env", ".venv/", "data/processed/*", "data/vectorstore/*"):
        assert entry in ignored


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)  # noqa: S603, S607


def test_no_env_file_or_secret_tracked_by_git() -> None:
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git work tree")
    assert _git("check-ignore", "-q", ".env").returncode == 0, ".env must be git-ignored"
    tracked = _git("ls-files").stdout.splitlines()
    assert ".env" not in tracked
    pattern = re.compile(r"(sk-ant-[A-Za-z0-9\-_]{20,}|sk-[A-Za-z0-9]{32,}|pa-[A-Za-z0-9\-_]{30,})")
    for rel in tracked:
        path = ROOT / rel
        if path.suffix in {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".example", ".json", ".sh"}:
            assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), f"key-like string in {rel}"


def test_qdrant_server_matches_client_minor_version() -> None:
    image = re.search(r"qdrant/qdrant:v(\d+)\.(\d+)\.\d+", (ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert image, "qdrant image must be pinned to an exact vX.Y.Z tag"
    for svc in ("ingestion", "retrieval"):
        pin = next(
            r for r in _requirements(ROOT / "services" / svc / "requirements.txt") if r.startswith("qdrant-client==")
        )
        major, minor = pin.split("==")[1].split(".")[:2]
        assert (major, minor) == image.groups(), f"{svc}: qdrant-client {pin} vs server v{'.'.join(image.groups())}"
