"""Retrieval service API contracts (v1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from common.contracts.chunks import ChunkMetadata, DocType, Level


class SearchFilters(BaseModel):
    """Optional metadata filters. All set fields are ANDed."""

    max_fee_pkr: int | None = Field(default=None, ge=0, description="Keeps chunks with fee <= this, or no fee.")
    doc_types: list[DocType] | None = None
    course_slugs: list[str] | None = None
    category: str | None = None
    level: Level | None = None
    mentor: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: SearchFilters | None = None
    debug: bool = False


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float = Field(description="Normalised relevance score in [0, 1].")
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    chunks: list[RetrievedChunk] = Field(description="Empty when nothing clears the threshold.")
    retrieval_debug: dict[str, Any] = Field(default_factory=dict)
