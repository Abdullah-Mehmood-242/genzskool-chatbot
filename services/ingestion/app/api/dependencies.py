"""Composition root: the ONLY place where concrete adapters are chosen and wired to ports."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter

from app.settings import Settings
from common.observability import ReadinessCheck


@dataclass
class Container:
    settings: Settings

    def readiness_checks(self) -> dict[str, ReadinessCheck]:
        return {}

    def routers(self) -> list[APIRouter]:
        return []


def build_container(settings: Settings) -> Container:
    return Container(settings=settings)
