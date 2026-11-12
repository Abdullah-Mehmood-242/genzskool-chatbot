"""Composition root: the ONLY place where concrete adapters are chosen and wired to ports."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI

from app.settings import Settings
from common.observability import ReadinessCheck


@dataclass
class Container:
    settings: Settings

    def readiness_checks(self) -> dict[str, ReadinessCheck]:
        return {}

    def routers(self) -> list[APIRouter]:
        return []

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        yield


def build_container(settings: Settings) -> Container:
    return Container(settings=settings)
