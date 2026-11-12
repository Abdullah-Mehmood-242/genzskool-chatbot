#!/usr/bin/env python
"""Boot every service as a real uvicorn process (no Docker) and probe its ops endpoints.

Usage: python scripts/smoke_local.py   (exit code 0 = all services healthy)
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORTS = {"ingestion": 8001, "retrieval": 8002, "advisor": 8003, "api_gateway": 8000, "evaluation": 8004}
BOOT_TIMEOUT_S = 30


def _start(svc: str, port: int) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(ROOT / "libs"), str(ROOT / "services" / svc)]),
        LLM_PROVIDER="fake",
        EMBEDDING_PROVIDER="fake",
        LOG_LEVEL="WARNING",
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-access-log",
    ]
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # noqa: S603


def _wait_healthy(port: int) -> bool:
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    return False


def _probe(svc: str, port: int) -> list[str]:
    base = f"http://127.0.0.1:{port}"
    problems: list[str] = []
    h = httpx.get(f"{base}/healthz", headers={"X-Request-ID": "smoke-0000001"})
    if h.json().get("service") != svc:
        problems.append(f"healthz reports service={h.json().get('service')}")
    if h.headers.get("X-Request-ID") != "smoke-0000001":
        problems.append("request id not echoed")
    if httpx.get(f"{base}/readyz").status_code != 200:
        problems.append("readyz not 200")
    m = httpx.get(f"{base}/metrics")
    if m.status_code != 200 or "http_requests_total" not in m.text:
        problems.append("metrics missing")
    nf = httpx.get(f"{base}/does-not-exist")
    if nf.status_code != 404 or nf.json().get("error", {}).get("code") != "not_found":
        problems.append("404 does not use the error schema")
    big = httpx.post(f"{base}/healthz", content=b"x" * 200_000, headers={"Content-Type": "application/json"})
    if big.status_code != 413:
        problems.append(f"oversized body returned {big.status_code}, expected 413")
    return problems


def _ports_in_use() -> list[int]:
    busy = []
    for port in PORTS.values():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(port)
    return busy


def main() -> int:
    busy = _ports_in_use()
    if busy:
        # Otherwise the probes would hit whatever already listens there (e.g. the compose stack) and pass falsely.
        print(f"FAIL ports already in use: {busy}. Stop the compose stack first (make down).")
        return 2
    procs = {svc: _start(svc, port) for svc, port in PORTS.items()}
    failed = False
    try:
        for svc, port in PORTS.items():
            if not _wait_healthy(port):
                procs[svc].terminate()
                out = procs[svc].communicate(timeout=5)[0].decode(errors="replace")
                print(f"FAIL {svc}:{port} did not become healthy\n{out[-2000:]}")
                failed = True
                continue
            problems = _probe(svc, port)
            print(("FAIL " if problems else "ok   ") + f"{svc}:{port} " + "; ".join(problems))
            failed |= bool(problems)
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
