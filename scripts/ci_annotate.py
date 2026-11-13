#!/usr/bin/env python
"""Turn failed tests in a JUnit XML report into GitHub Actions error annotations.

Annotations are visible on the run page (and via the public checks API) without downloading the
full job log, so a CI failure can be diagnosed at a glance.

Usage: python scripts/ci_annotate.py <junit.xml> [max_annotations]
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _escape(text: str) -> str:
    # GitHub workflow-command escaping for the message part.
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def failures(report: Path) -> list[tuple[str, str]]:
    root = ET.parse(report).getroot()  # noqa: S314
    out: list[tuple[str, str]] = []
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                detail = (node.get("message") or "") + "\n" + (node.text or "")[-1500:]
                out.append((name, detail.strip()))
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    report = Path(argv[0])
    limit = int(argv[1]) if len(argv) > 1 else 10
    if not report.exists():
        print(f"::warning title=no test report::{report} not found (tests may not have started)")
        return 0
    found = failures(report)
    for name, detail in found[:limit]:
        print(f"::error title={_escape(name)}::{_escape(detail)}")
    if len(found) > limit:
        print(f"::warning title=more failures::{len(found) - limit} more failed tests not annotated")
    print(f"{len(found)} failed test(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
