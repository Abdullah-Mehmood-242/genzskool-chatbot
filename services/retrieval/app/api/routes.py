"""HTTP API v1 for retrieval. No answer generation happens here."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.application.search import SearchService
from app.domain.ports import CorpusProvider
from common.contracts import SearchRequest, SearchResponse

router = APIRouter(prefix="/v1", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse, summary="Hybrid search over the active index")
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    """Returns evidence chunks with normalised scores. An empty list means nothing was relevant enough."""
    service: SearchService = request.app.state.container.service
    return await service.search(body)


@router.get("/index", summary="What the retrieval service is currently serving")
async def index(request: Request) -> dict[str, Any]:
    provider: CorpusProvider = request.app.state.container.corpus
    corpus = await provider.current()
    return {
        "documents": [{"doc_id": d, "index_version": v} for d, v in corpus.versions],
        "chunks": len(corpus),
        "signature": corpus.signature,
        "warnings": list(corpus.warnings),
    }
