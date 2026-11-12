"""The task runner (scripts/dev.py) builds the right commands; subprocesses are stubbed."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dev_runner", ROOT / "scripts" / "dev.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dev(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[dict[str, object]]]:
    module = _load()
    calls: list[dict[str, object]] = []

    def fake_call(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    return module, calls


def _joined(call: dict[str, object]) -> str:
    return " ".join(call["cmd"])  # type: ignore[arg-type]


def test_no_args_prints_usage(
    dev: tuple[ModuleType, list[dict[str, object]]], capsys: pytest.CaptureFixture[str]
) -> None:
    module, calls = dev
    assert module.main([]) == 2
    assert "Usage" in capsys.readouterr().out
    assert calls == []


def test_unknown_command(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    assert module.main(["deploy-prod"]) == 2
    assert calls == []


def test_unknown_test_target(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, _ = dev
    assert module.main(["test", "nonexistent"]) == 2


def test_default_test_runs_every_target_in_own_process(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    assert module.main(["test"]) == 0
    cwds = [Path(str(c["cwd"])) for c in calls]
    assert cwds[0] == ROOT / "libs"
    assert cwds[1] == ROOT
    assert cwds[2:] == [ROOT / "services" / s for s in module.SERVICES]
    assert len(calls) == 2 + len(module.SERVICES)


def test_common_target_has_coverage_gate(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    module.main(["test", "common"])
    cmd = _joined(calls[0])
    assert "--cov=common" in cmd
    assert f"--cov-fail-under={module.COVERAGE_MIN}" in cmd
    assert "common/tests" in cmd


def test_service_pythonpath_is_isolated(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    module.main(["test", "advisor"])
    env = calls[0]["env"]
    assert isinstance(env, dict)
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts == [str(ROOT / "libs"), str(ROOT / "services" / "advisor")]
    assert not any("retrieval" in p for p in parts)


def test_service_coverage_only_when_layers_have_code(
    dev: tuple[ModuleType, list[dict[str, object]]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, calls = dev
    fake_root = tmp_path
    for svc in module.SERVICES:
        for layer in ("domain", "application"):
            (fake_root / "services" / svc / "app" / layer).mkdir(parents=True)
            (fake_root / "services" / svc / "app" / layer / "__init__.py").write_text("")
    (fake_root / "services" / "advisor" / "app" / "domain" / "rules.py").write_text("X = 1\n")
    monkeypatch.setattr(module, "ROOT", fake_root)
    module.main(["test", "advisor", "retrieval"])
    assert "--cov=app.domain" in _joined(calls[0])
    assert "--cov=app.domain" not in _joined(calls[1])


def test_env_defaults_to_fake_providers_without_overriding(
    dev: tuple[ModuleType, list[dict[str, object]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = dev
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    env = module._env(ROOT)
    assert env["LLM_PROVIDER"] == "fake"
    assert env["EMBEDDING_PROVIDER"] == "voyage"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_lint_runs_check_and_format_check(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    module.main(["lint"])
    assert [c["cmd"][-2:] for c in calls] == [["check", "."], ["--check", "."]]  # type: ignore[index]


def test_fmt_fixes(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    module.main(["fmt"])
    assert "--fix" in _joined(calls[0])
    assert _joined(calls[1]).endswith("format .")


def test_typecheck_covers_libs_and_every_inner_layer(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    module.main(["typecheck"])
    assert _joined(calls[0]).endswith("common")
    for call, svc in zip(calls[1:], module.SERVICES, strict=True):
        assert call["cwd"] == ROOT / "services" / svc
        assert _joined(call).endswith("app/domain app/application")
        assert call["env"]["MYPYPATH"] == str(ROOT / "libs")  # type: ignore[index]


def test_smoke_local_command(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    module.main(["smoke-local"])
    assert _joined(calls[0]).endswith("smoke_local.py")


def test_failures_propagate(dev: tuple[ModuleType, list[dict[str, object]]], monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = dev
    results = iter([0, 1, 0, 0, 0, 0, 0])
    monkeypatch.setattr(module.subprocess, "call", lambda *a, **k: next(results))
    assert module.main(["test"]) != 0


def test_check_runs_lint_typecheck_and_tests(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    assert module.main(["check", "common"]) == 0
    cmds = [_joined(c) for c in calls]
    assert any("ruff check" in c for c in cmds)
    assert any("mypy" in c for c in cmds)
    assert any("pytest" in c for c in cmds)


def test_uses_current_interpreter(dev: tuple[ModuleType, list[dict[str, object]]]) -> None:
    module, calls = dev
    module.main(["lint"])
    assert calls[0]["cmd"][0] == sys.executable  # type: ignore[index]
