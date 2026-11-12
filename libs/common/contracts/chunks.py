"""Chunk schema: written by ingestion, read by retrieval (Qdrant payload + processed JSONL)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DocType(StrEnum):
    COURSE = "course"
    MENTOR = "mentor"
    POLICY = "policy"
    CONTACT = "contact"
    OVERVIEW = "overview"


class Level(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Stable id of the source document (file stem).")
    source_file: str
    page: int = Field(ge=1)
    section: str
    doc_type: DocType
    course_slug: str | None = None
    category: str | None = None
    level: Level | None = None
    fee_pkr: int | None = Field(default=None, ge=0, description="Null when the fee is not published.")
    mentor: str | None = None
    content_hash: str
    ingested_at: datetime
    index_version: int = Field(ge=1)


class Chunk(BaseModel):
    """One retrievable unit of text with its metadata."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Deterministic id: hash of doc_id + section + content.")
    text: str
    metadata: ChunkMetadata
