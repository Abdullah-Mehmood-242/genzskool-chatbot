"""Service entrypoint. Run with: ``uvicorn app.main:create_app --factory``."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.dependencies import build_container
from app.settings import Settings
from common.config import load_settings
from common.logging import configure_logging
from common.observability import create_service_app


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        # Log as JSON from the first line: settings validation itself may warn (or fail fast).
        configure_logging(Settings.model_fields["service_name"].default, "INFO")
        settings = load_settings(Settings)
    configure_logging(settings.service_name, settings.log_level)
    container = build_container(settings)
    app = create_service_app(
        service=settings.service_name,
        version=settings.service_version,
        readiness_checks=container.readiness_checks(),
        max_body_bytes=settings.max_body_bytes,
    )
    app.state.settings = settings
    app.state.container = container
    for router in container.routers():
        app.include_router(router)
    return app
