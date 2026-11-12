"""Static import rules for the hexagonal layers (an AST scan; no import-linter dependency).

Dependencies point inward only:  api -> application -> domain <- infrastructure
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

STDLIB = set(sys.stdlib_module_names) | {"__future__"}

# Third-party packages each layer may import (beyond stdlib and its allowed first-party modules).
THIRD_PARTY_ALLOWED: dict[str, set[str]] = {
    "domain": {"pydantic"},
    "application": {"pydantic", "langgraph"},
    "infrastructure": {"*"},
    "api": {"*"},
}

# First-party prefixes each layer may import.
FIRST_PARTY_ALLOWED: dict[str, tuple[str, ...]] = {
    "domain": ("app.domain", "common.contracts", "common.errors"),
    "application": (
        "app.domain",
        "app.application",
        "common.contracts",
        "common.errors",
        "common.logging",
        "common.resilience",
    ),
    "infrastructure": ("app.domain", "app.infrastructure", "common"),
    "api": ("app", "common"),
}


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    layer: str
    module: str
    reason: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} [{self.layer}] imports '{self.module}': {self.reason}"


def _imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative imports are forbidden: they hide layer crossings
                found.append((node.lineno, "." * node.level + (node.module or "")))
            elif node.module:
                found.append((node.lineno, node.module))
    return found


ROOT_FILES = {"__init__.py", "main.py", "settings.py"}  # composition entrypoints directly under app/


def _layer_of(path: Path, app_root: Path) -> str | None:
    """Return the layer name, None for an allowed root file, or 'unknown' for anything else."""
    rel = path.relative_to(app_root).parts
    if len(rel) == 1:
        return None if rel[0] in ROOT_FILES else "unknown"
    return rel[0] if rel[0] in THIRD_PARTY_ALLOWED else "unknown"


def check_service(service_dir: Path, repo_root: Path) -> list[Violation]:
    app_root = service_dir / "app"
    violations: list[Violation] = []
    for path in sorted(app_root.rglob("*.py")):
        layer = _layer_of(path, app_root)
        rel = path.relative_to(repo_root).as_posix()
        if layer == "unknown":
            reason = "code must live in api/, application/, domain/ or infrastructure/ (or main.py/settings.py)"
            violations.append(Violation(rel, 1, "unknown", "-", reason))
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, module in _imports(tree):
            top = module.split(".")[0]
            if module.startswith("."):
                violations.append(Violation(rel, line, layer or "root", module, "relative imports are not allowed"))
                continue
            if top == "services":
                violations.append(Violation(rel, line, layer or "root", module, "services must not import each other"))
                continue
            if layer is None:  # app/main.py, app/settings.py: composition entrypoints
                continue
            if top in STDLIB:
                continue
            if top in ("app", "common"):
                if not any(module == p or module.startswith(p + ".") for p in FIRST_PARTY_ALLOWED[layer]):
                    violations.append(Violation(rel, line, layer, module, "crosses a layer boundary"))
                continue
            allowed = THIRD_PARTY_ALLOWED[layer]
            if "*" not in allowed and top not in allowed:
                violations.append(Violation(rel, line, layer, module, "third-party import not allowed in this layer"))
    return violations
