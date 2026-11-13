"""Pydantic DTOs shared across services. Contracts only: no business logic."""

from common.contracts.advisor import (
    MAX_MESSAGE_CHARS,
    AdviseRequest,
    AdviseResponse,
    Citation,
    Intent,
    LearnerProfileDTO,
    RecommendedCourse,
    SessionView,
    Usage,
)
from common.contracts.chunks import Chunk, ChunkMetadata, DocType, Level
from common.contracts.errors import Envelope, ErrorDetail, ErrorResponse, HealthStatus
from common.contracts.ingestion import (
    AutoIngestStatus,
    AutoIngestTrigger,
    DocumentStatus,
    FileOutcome,
    FileReport,
    IngestReport,
    IngestRequest,
    IngestStatus,
)
from common.contracts.retrieval import RetrievedChunk, SearchFilters, SearchRequest, SearchResponse

__all__ = [
    "MAX_MESSAGE_CHARS",
    "AdviseRequest",
    "AdviseResponse",
    "AutoIngestStatus",
    "AutoIngestTrigger",
    "Chunk",
    "ChunkMetadata",
    "Citation",
    "DocType",
    "DocumentStatus",
    "Envelope",
    "ErrorDetail",
    "ErrorResponse",
    "FileOutcome",
    "FileReport",
    "HealthStatus",
    "IngestReport",
    "IngestRequest",
    "IngestStatus",
    "Intent",
    "LearnerProfileDTO",
    "Level",
    "RecommendedCourse",
    "RetrievedChunk",
    "SearchFilters",
    "SearchRequest",
    "SearchResponse",
    "SessionView",
    "Usage",
]
