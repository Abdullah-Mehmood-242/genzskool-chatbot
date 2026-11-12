"""Phase 1 e2e: the compose stack is healthy, wired, hardened and observable (fake providers, no keys)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import DOCKER, PORTS, compose, exec_in, url

SERVICES = list(PORTS)


@pytest.mark.parametrize("svc", SERVICES)
def test_ops_endpoints_from_host(svc: str) -> None:
    h = httpx.get(url(svc, "/healthz"), headers={"X-Request-ID": "e2e-request-0001"})
    assert h.status_code == 200
    assert h.json()["service"] == svc
    assert h.json()["status"] == "ok"
    assert h.headers["X-Request-ID"] == "e2e-request-0001"
    assert httpx.get(url(svc, "/readyz")).status_code == 200
    m = httpx.get(url(svc, "/metrics"))
    assert m.status_code == 200
    assert "http_requests_total" in m.text


@pytest.mark.parametrize("svc", SERVICES)
def test_error_schema_and_body_limit_in_container(svc: str) -> None:
    nf = httpx.get(url(svc, "/v1/nope"))
    assert nf.status_code == 404
    assert nf.json()["error"]["code"] == "not_found"
    big = httpx.post(url(svc, "/healthz"), content=b"x" * 200_000)
    assert big.status_code == 413


@pytest.mark.parametrize("svc", SERVICES)
def test_openapi_served(svc: str) -> None:
    spec = httpx.get(url(svc, "/openapi.json")).json()
    assert spec["info"]["title"] == f"genz-{svc}"


def test_all_containers_healthy(ps: dict[str, dict[str, object]]) -> None:
    assert set(ps) == {*SERVICES, "qdrant"}
    for name, row in ps.items():
        assert row["State"] == "running", name
        assert row["Health"] == "healthy", name


@pytest.mark.parametrize("svc", SERVICES)
def test_container_runs_as_non_root_on_python_312(svc: str, docker_available: None) -> None:
    assert exec_in(svc, "id", "-u") == "10001"
    assert exec_in(svc, "python", "-c", "import sys; print(sys.version_info[:2])") == "(3, 12)"


@pytest.mark.parametrize("svc", SERVICES)
def test_image_contains_no_secrets_or_tests(svc: str, docker_available: None) -> None:
    listing = exec_in(svc, "sh", "-c", "ls -a /app /app/service; find /app -name '.env' -o -name 'tests' -type d")
    assert ".env" not in listing.split()
    assert "/app/service/tests" not in listing
    assert "/app/libs/common/tests" not in listing


@pytest.mark.parametrize("svc", SERVICES)
def test_container_env_is_offline_mode(svc: str, docker_available: None) -> None:
    env = exec_in(svc, "env")
    if svc in ("ingestion", "retrieval"):
        assert "EMBEDDING_PROVIDER=fake" in env
    if svc == "advisor":
        assert "LLM_PROVIDER=fake" in env
    assert "APP_ENV=dev" in env


def test_pdf_mount_is_read_only_and_processed_is_writable(docker_available: None) -> None:
    assert exec_in("ingestion", "ls", "/app/data/pdfs") == "genzskool_knowledge_base.pdf"
    ro = compose("exec", "-T", "ingestion", "sh", "-c", "touch /app/data/pdfs/x 2>&1; echo rc=$?", check=False)
    assert "rc=0" not in ro.stdout
    rw = exec_in(
        "ingestion", "sh", "-c", "touch /app/data/processed/.probe && rm /app/data/processed/.probe && echo ok"
    )
    assert rw == "ok"
    ro_r = compose("exec", "-T", "retrieval", "sh", "-c", "touch /app/data/processed/x 2>&1; echo rc=$?", check=False)
    assert "rc=0" not in ro_r.stdout


PY_GET = "import sys,urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=3).status)"


@pytest.mark.parametrize(
    ("src", "target"),
    [
        ("ingestion", "http://qdrant:6333/readyz"),
        ("retrieval", "http://qdrant:6333/readyz"),
        ("advisor", "http://retrieval:8002/healthz"),
        ("api_gateway", "http://advisor:8003/healthz"),
        ("api_gateway", "http://ingestion:8001/healthz"),
        ("evaluation", "http://retrieval:8002/healthz"),
        ("evaluation", "http://advisor:8003/healthz"),
    ],
)
def test_service_discovery_on_docker_network(src: str, target: str, docker_available: None) -> None:
    assert exec_in(src, "python", "-c", PY_GET, target) == "200"


def test_qdrant_version_matches_client(docker_available: None) -> None:
    out = exec_in(
        "ingestion",
        "python",
        "-c",
        "import urllib.request,json;print(json.load(urllib.request.urlopen('http://qdrant:6333/'))['version'])",
    )
    client = exec_in("ingestion", "python", "-c", "import importlib.metadata as m; print(m.version('qdrant-client'))")
    assert out.split(".")[:2] == client.split(".")[:2]


def test_internal_ports_not_exposed_publicly(ps: dict[str, dict[str, object]]) -> None:
    for name, row in ps.items():
        publishers = row.get("Publishers") or []
        for p in publishers:  # type: ignore[union-attr]
            if p["PublishedPort"] and name != "api_gateway":
                assert p["URL"] in ("127.0.0.1",), f"{name} exposed on {p['URL']}"


def test_container_logs_are_json_with_request_ids(docker_available: None) -> None:
    rid = "e2e-logcheck-0001"
    httpx.get(url("advisor", "/v1/log-probe"), headers={"X-Request-ID": rid})
    time.sleep(0.5)
    logs = compose("logs", "--no-log-prefix", "--since", "1m", "advisor").stdout
    lines = [ln for ln in logs.splitlines() if rid in ln]
    assert lines, "access log for the probe request not found"
    record = json.loads(lines[-1])
    assert record["service"] == "advisor"
    assert record["request_id"] == rid
    assert record["status"] == 404


CRASH = "\n".join(
    [
        "import os, signal",
        "for pid in filter(str.isdigit, os.listdir('/proc')):",
        "    try:",
        "        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().split(bytes(1))",
        "    except OSError:",
        "        continue",
        "    if any(part.endswith(b'/uvicorn') for part in cmd):  # python /opt/venv/bin/uvicorn ...",
        "        os.kill(int(pid), signal.SIGKILL)",
    ]
)


def _restart_count(service: str) -> int:
    container_id = compose("ps", "-q", service).stdout.strip()
    assert DOCKER
    out = subprocess.run(  # noqa: S603
        [DOCKER, "inspect", "-f", "{{.RestartCount}}", container_id], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


def test_crashed_service_is_restarted(docker_available: None) -> None:
    """SIGKILL the uvicorn process inside the container (a real crash); the restart policy must recover it."""
    before = _restart_count("evaluation")
    compose("exec", "-T", "evaluation", "python", "-c", CRASH, check=False)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if _restart_count("evaluation") > before:
            try:
                if httpx.get(url("evaluation", "/healthz"), timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
        time.sleep(1)
    pytest.fail(f"evaluation was not restarted and healthy within 90s (restarts before={before})")


@pytest.mark.parametrize("svc", SERVICES)
def test_every_container_log_line_is_json(svc: str, docker_available: None) -> None:
    logs = compose("logs", "--no-log-prefix", svc).stdout
    lines = [ln for ln in logs.splitlines() if ln.strip()]
    assert lines
    bad = [ln for ln in lines if not ln.startswith("{")]
    assert not bad, f"non-JSON log lines from {svc}: {bad[:3]}"
    for ln in lines:
        assert json.loads(ln)["service"] == svc


def _wait_healthy(service: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url(service, "/healthz"), timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    pytest.fail(f"{service} not healthy after {timeout}s")


def test_graceful_shutdown_on_sigterm(docker_available: None) -> None:
    """docker stop must reach uvicorn (exec form), exit cleanly and fast, and log the shutdown."""
    start = time.monotonic()
    compose("stop", "-t", "15", "evaluation")
    elapsed = time.monotonic() - start
    try:
        container_id = compose("ps", "-aq", "evaluation").stdout.strip()
        assert DOCKER
        state = subprocess.run(  # noqa: S603
            [DOCKER, "inspect", "-f", "{{.State.ExitCode}}", container_id], capture_output=True, text=True, check=True
        ).stdout.strip()
        logs = compose("logs", "--no-log-prefix", "--tail", "20", "evaluation").stdout
        # uvicorn re-raises the captured SIGTERM after a clean shutdown -> 143. 137 would mean SIGKILL.
        assert state in ("0", "143"), f"exit code {state}: process was killed, not shut down"
        assert elapsed < 10, f"stop took {elapsed:.1f}s (signal not handled; Docker waited for SIGKILL)"
        assert "Finished server process" in logs
    finally:
        compose("start", "evaluation")
        _wait_healthy("evaluation")
