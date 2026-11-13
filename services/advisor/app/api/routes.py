"""HTTP API v1 for the advisor."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Path, Request, Response
from fastapi.responses import StreamingResponse

from app.application.advise import AdviseService
from common.contracts import AdviseRequest, AdviseResponse, ErrorDetail, ErrorResponse, SessionView
from common.errors import AppError
from common.logging import get_request_id

router = APIRouter(prefix="/v1", tags=["advisor"])
logger = logging.getLogger(__name__)
SessionId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]


def _service(request: Request) -> AdviseService:
    service: AdviseService = request.app.state.container.service
    return service


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/advise", response_model=AdviseResponse, summary="One conversational turn")
async def advise(request: Request, body: AdviseRequest) -> AdviseResponse:
    return await _service(request).advise(body)


@router.post(
    "/advise/stream", summary="One turn as Server-Sent Events", responses={200: {"content": {"text/event-stream": {}}}}
)
async def advise_stream(request: Request, body: AdviseRequest) -> StreamingResponse:
    """Events: `status` (progress per step), `answer` (pieces of the verified answer), `done` (the full
    AdviseResponse), or `error` (the shared error schema). Answers are streamed only after verification."""
    service = _service(request)
    request_id = get_request_id()

    async def events() -> AsyncIterator[str]:
        try:
            async for event, data in service.stream(body):
                yield sse(event, data)
        except AppError as exc:
            yield sse(
                "error",
                ErrorResponse(
                    error=ErrorDetail(code=exc.code, message=exc.message, request_id=request_id)
                ).model_dump(),
            )
        except Exception:
            logger.exception("stream_failed")
            yield sse(
                "error",
                ErrorResponse(
                    error=ErrorDetail(
                        code="internal_error", message="An internal error occurred.", request_id=request_id
                    )
                ).model_dump(),
            )

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/sessions/{session_id}", response_model=SessionView, summary="What the advisor remembers")
async def get_session(request: Request, session_id: SessionId) -> SessionView:
    return await _service(request).get_session(session_id)


@router.delete("/sessions/{session_id}", status_code=204, summary="Forget a session")
async def delete_session(request: Request, session_id: SessionId) -> Response:
    await _service(request).delete_session(session_id)
    return Response(status_code=204)
