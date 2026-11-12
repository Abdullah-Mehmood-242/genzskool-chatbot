"""Shared fixtures for compose-level end-to-end tests. Skips cleanly when the stack is not running."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
PORTS = {"api_gateway": 8000, "ingestion": 8001, "retrieval": 8002, "advisor": 8003, "evaluation": 8004}
HOST = os.environ.get("E2E_HOST", "127.0.0.1")


def _docker() -> str | None:
    found = shutil.which("docker")
    if found:
        return found
    default = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")
    return str(default) if default.exists() else None


DOCKER = _docker()


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    assert DOCKER, "docker CLI not found"
    return subprocess.run(  # noqa: S603 - fixed argv
        [DOCKER, "compose", *args], cwd=ROOT, capture_output=True, text=True, check=check, timeout=180
    )


def exec_in(service: str, *cmd: str) -> str:
    return compose("exec", "-T", service, *cmd).stdout.strip()


def _stack_up() -> bool:
    try:
        return httpx.get(f"http://{HOST}:8000/healthz", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session", autouse=True)
def stack() -> None:
    if not _stack_up():
        pytest.skip("compose stack is not running (make up)")


@pytest.fixture(scope="session")
def docker_available() -> None:
    if DOCKER is None:
        pytest.skip("docker CLI not available")


@pytest.fixture(scope="session")
def ps(docker_available: None) -> dict[str, dict[str, object]]:
    out = compose("ps", "--format", "json").stdout
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    return {str(r["Service"]): r for r in rows}


def url(service: str, path: str) -> str:
    return f"http://{HOST}:{PORTS[service]}{path}"
