"""Load the versioned prompt set (prompts/<version>/) and the system clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from app.domain.compose import Messages, PromptTemplates
from app.domain.ports import Clock
from common.errors import ConfigurationError

REQUIRED_MESSAGES = frozenset(
    {
        "greeting",
        "refusal_out_of_scope",
        "refusal_advice",
        "refusal_unsafe",
        "insufficient_evidence",
        "retrieval_down",
        "handoff",
        "confirm_reminder",
        "sources_label",
        "clarify_interest",
        "clarify_budget",
        "plan_intro",
        "plan_intro_budget",
        "plan_total_now",
        "plan_total_all",
        "plan_unknown_fee",
        "plan_bundle",
        "plan_next_step",
        "plan_none",
        "plan_over_budget",
        "fee_unpublished",
        "mentor_label",
        "fees_intro",
    }
)


@dataclass(frozen=True)
class PromptSet:
    version: str
    templates: PromptTemplates
    messages: Messages


def load_prompt_set(prompts_dir: Path, version: str) -> PromptSet:
    base = prompts_dir / version
    try:
        templates = PromptTemplates(
            system=(base / "system.md").read_text(encoding="utf-8"),
            answer=(base / "answer.md").read_text(encoding="utf-8"),
            regenerate=(base / "regenerate.md").read_text(encoding="utf-8"),
        )
        raw = yaml.safe_load((base / "messages.yaml").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Prompt file missing in {version}: {Path(exc.filename or '').name}") from None
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{version}/messages.yaml must be a mapping.")
    missing = REQUIRED_MESSAGES - set(raw)
    if missing:
        raise ConfigurationError(f"{version}/messages.yaml is missing: {', '.join(sorted(missing))}")
    if any(not isinstance(v, dict) or "en" not in v for v in raw.values()):
        raise ConfigurationError(f"Every message in {version}/messages.yaml needs an 'en' variant.")
    return PromptSet(version=version, templates=templates, messages=Messages(raw))


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)
