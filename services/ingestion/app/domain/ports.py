"""Ports (abstract interfaces) for ingestion. Small and focused (Interface Segregation).

Adapters in ``app.infrastructure`` implement them; use cases in ``app.application`` depend only on
these abstractions (Dependency Inversion).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from app.domain.models import (
    ChunkResult,
    CleanedDocument,
    IndexRecord,
    ParsedDocument,
    SourceFile,
    SourceFingerprint,
)
from common.contracts import Chunk

Vector = list[float]


class DocumentSource(ABC):
    """Where source PDFs live (a directory today; object storage tomorrow)."""

    @abstractmethod
    def list_files(self) -> list[SourceFile]: ...

    @abstractmethod
    def resolve(self, file: str) -> SourceFile:
        """Validate a user-supplied file name and return it. Raises InvalidFileName or NotFound."""

    @abstractmethod
    def read(self, source: SourceFile) -> bytes: ...

    @abstractmethod
    def fingerprints(self) -> list[SourceFingerprint]: ...


class DocumentLoader(ABC):
    """Bytes -> pages of text and tables. Raises CorruptDocument / EmptyDocument."""

    @abstractmethod
    def load(self, data: bytes, source: SourceFile, content_hash: str) -> ParsedDocument: ...


class TextCleaner(ABC):
    @abstractmethod
    def clean(self, doc: ParsedDocument) -> CleanedDocument: ...


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: CleanedDocument, *, index_version: int, ingested_at: datetime) -> ChunkResult: ...


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...


class VectorStore(ABC):
    @abstractmethod
    async def ensure_collection(self, dim: int) -> None:
        """Create the collection if missing. Raises ConfigurationError on a dimension mismatch."""

    @abstractmethod
    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    @abstractmethod
    async def count(self, doc_id: str, index_version: int) -> int: ...

    @abstractmethod
    async def delete_version(self, doc_id: str, index_version: int) -> None: ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    async def healthy(self) -> bool: ...


class IngestionRegistry(ABC):
    """Persistent record of the active index version per document (the blue/green switch)."""

    @abstractmethod
    def get(self, doc_id: str) -> IndexRecord | None: ...

    @abstractmethod
    def put(self, record: IndexRecord) -> None: ...

    @abstractmethod
    def delete(self, doc_id: str) -> None: ...

    @abstractmethod
    def list(self) -> list[IndexRecord]: ...


class ChunkArtifactStore(ABC):
    """Inspectable chunk files (data/processed/). Also the BM25 corpus for retrieval."""

    @abstractmethod
    def write(self, doc_id: str, index_version: int, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def read(self, doc_id: str, index_version: int) -> list[Chunk]: ...

    @abstractmethod
    def delete(self, doc_id: str, index_version: int | None = None) -> None:
        """Delete one version, or every version when ``index_version`` is None."""


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime: ...
