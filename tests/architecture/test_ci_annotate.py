"""scripts/ci_annotate.py turns JUnit failures into GitHub annotations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci_annotate.py"

JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.e2e.test_x" name="test_ok"/>
  <testcase classname="tests.e2e.test_x" name="test_fail"><failure message="assert 1 == 2">long
trace 100%</failure></testcase>
  <testcase classname="tests.e2e.test_y" name="test_err"><error message="boom">x</error></testcase>
  <testcase classname="tests.e2e.test_z" name="test_skip"><skipped/></testcase>
</testsuite></testsuites>"""


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)  # noqa: S603


def test_annotates_failures_and_errors_only(tmp_path: Path) -> None:
    report = tmp_path / "r.xml"
    report.write_text(JUNIT, encoding="utf-8")
    out = _run(str(report)).stdout.splitlines()
    errors = [ln for ln in out if ln.startswith("::error")]
    assert len(errors) == 2
    assert errors[0].startswith("::error title=tests.e2e.test_x::test_fail::assert 1 == 2%0Along%0Atrace 100%25")
    assert "test_err" in errors[1]
    assert out[-1] == "2 failed test(s)"


def test_limit_and_missing_report(tmp_path: Path) -> None:
    report = tmp_path / "r.xml"
    report.write_text(JUNIT, encoding="utf-8")
    out = _run(str(report), "1").stdout
    assert out.count("::error") == 1
    assert "1 more failed tests" in out
    missing = _run(str(tmp_path / "nope.xml"))
    assert missing.returncode == 0
    assert "::warning" in missing.stdout
