#!/usr/bin/env python
"""Cross-platform task runner (the Makefile delegates here; also usable directly on Windows).

Each service is tested/type-checked in its OWN process with its own PYTHONPATH, because every
service uses the package name ``app`` (see docs/adr/0010-packaging-and-layout.md).

Usage:
    python scripts/dev.py test [target ...]     # targets: common, architecture, e2e, <service>
    python scripts/dev.py lint                  # ruff check + format check
    python scripts/dev.py fmt                   # ruff fix + format
    python scripts/dev.py typecheck             # mypy --strict on libs/ + domain/ + application/
    python scripts/dev.py ingest [--force]      # trigger ingestion on the running stack
    python scripts/dev.py fixtures              # regenerate tests/fixtures/kb from the real PDF
    python scripts/dev.py smoke-local           # boot every service with uvicorn and probe it
    python scripts/dev.py check                 # lint + typecheck + test
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBS = ROOT / "libs"
SERVICES = ["ingestion", "retrieval", "advisor", "api_gateway", "evaluation"]
COVERAGE_MIN = 85


def _env(*paths: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    env.setdefault("LLM_PROVIDER", "fake")
    env.setdefault("EMBEDDING_PROVIDER", "fake")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd.relative_to(ROOT) if cwd != ROOT else '.'})", flush=True)
    return subprocess.call(cmd, cwd=cwd, env=env or _env(LIBS))  # noqa: S603 - fixed argv, no shell


def _pytest(args: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(ROOT / "pyproject.toml"),
            "--rootdir",
            str(cwd),
            *args,
        ],
        cwd=cwd,
        env=env,
    )


def test(targets: list[str]) -> int:
    targets = targets or ["common", "architecture", *SERVICES]
    rc = 0
    for t in targets:
        if t == "common":
            cov = ["--cov=common", "--cov-report=term", f"--cov-fail-under={COVERAGE_MIN}"]
            rc |= _pytest(["common/tests", *cov], cwd=LIBS, env=_env(LIBS))
        elif t == "architecture":
            rc |= _pytest(["tests/architecture"], cwd=ROOT, env=_env(LIBS))
        elif t == "e2e":
            rc |= _pytest(["tests/e2e"], cwd=ROOT, env=_env(LIBS))
        elif t in SERVICES:
            svc = ROOT / "services" / t
            cov = ["--cov=app.domain", "--cov=app.application", "--cov-report=term", f"--cov-fail-under={COVERAGE_MIN}"]
            has_code = any(
                p.stat().st_size > 0 for layer in ("domain", "application") for p in (svc / "app" / layer).rglob("*.py")
            )
            rc |= _pytest(["tests", *(cov if has_code else [])], cwd=svc, env=_env(LIBS, svc))
        else:
            print(f"unknown test target: {t}", file=sys.stderr)
            return 2
    return rc


def lint() -> int:
    ruff = [sys.executable, "-m", "ruff"]
    return _run([*ruff, "check", "."]) | _run([*ruff, "format", "--check", "."])


def fmt() -> int:
    ruff = [sys.executable, "-m", "ruff"]
    return _run([*ruff, "check", "--fix", "."]) | _run([*ruff, "format", "."])


def typecheck() -> int:
    mypy = [sys.executable, "-m", "mypy", "--config-file", str(ROOT / "pyproject.toml")]
    rc = _run([*mypy, "common"], cwd=LIBS, env=_env(LIBS))
    for svc_name in SERVICES:
        svc = ROOT / "services" / svc_name
        env = _env(LIBS, svc)
        env["MYPYPATH"] = str(LIBS)
        rc |= _run([*mypy, "app/domain", "app/application"], cwd=svc, env=env)
    return rc


def ingest(args: list[str]) -> int:
    """POST /v1/ingest to the running ingestion service (Windows-friendly twin of ingest_all.sh)."""
    import json
    import urllib.error
    import urllib.request

    url = os.environ.get("INGESTION_URL", "http://127.0.0.1:8001") + "/v1/ingest"
    body = json.dumps({"force": "--force" in args}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 - local service URL
            report = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        print(f"ingest request to {url} failed ({exc}); is the stack up? (make up)", file=sys.stderr)
        return 1
    for f in report["files"]:
        extra = f["error"] or "; ".join(f["warnings"])
        print(f"{f['outcome']:<16} {f['file']:<45} chunks={f['chunks']:<4} v={f['index_version']} {extra}")
    for doc in report["pruned"]:
        print(f"{'pruned':<16} {doc}")
    print("OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "test":
        return test(rest)
    if cmd == "lint":
        return lint()
    if cmd == "fmt":
        return fmt()
    if cmd == "typecheck":
        return typecheck()
    if cmd == "ingest":
        return ingest(rest)
    if cmd == "fixtures":
        svc = ROOT / "services" / "ingestion"
        return _run([sys.executable, str(ROOT / "scripts" / "build_fixtures.py")], env=_env(LIBS, svc))
    if cmd == "smoke-local":
        return _run([sys.executable, str(ROOT / "scripts" / "smoke_local.py")])
    if cmd == "check":
        return lint() | typecheck() | test(rest)
    print(f"unknown command: {cmd}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
