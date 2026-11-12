"""HTTP API v1 for ingestion."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Path, Request, Response

from app.application.auto_ingest import AutoIngestor
from app.application.ingest import IngestionService
from common.contracts import DocumentStatus, IngestReport, IngestRequest, IngestStatus

router = APIRouter(prefix="/v1", tags=["ingestion"])


def _service(request: Request) -> IngestionService:
    service: IngestionService = request.app.state.container.service
    return service


def _auto(request: Request) -> AutoIngestor:
    auto: AutoIngestor = request.app.state.container.auto_ingestor
    return auto


@router.post("/ingest", response_model=IngestReport, summary="Ingest all PDFs, or one file")
async def ingest(request: Request, body: Annotated[IngestRequest | None, Body()] = None) -> IngestReport:
    """Idempotent: unchanged files are skipped unless ``force`` is true. Always 200; check ``ok`` per file."""
    body = body or IngestRequest()
    return await _service(request).ingest(body.file, force=body.force)


@router.get("/ingest/status", response_model=IngestStatus, summary="Indexed documents and auto-ingest state")
async def status(request: Request) -> IngestStatus:
    docs = [
        DocumentStatus(
            doc_id=r.doc_id,
            source_file=r.source_file,
            content_hash=r.content_hash,
            active_index_version=r.active_index_version,
            chunks=r.chunks,
            ingested_at=r.ingested_at,
        )
        for r in _service(request).status()
    ]
    return IngestStatus(documents=docs, auto_ingest=_auto(request).status)


@router.delete("/documents/{doc_id}", status_code=204, summary="Remove a document from the index")
async def delete_document(request: Request, doc_id: Annotated[str, Path(pattern=r"^[a-z0-9_]{1,100}$")]) -> Response:
    await _service(request).delete_document(doc_id)
    return Response(status_code=204)
