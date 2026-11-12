"""Ports for retrieval (one small interface per concern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence

from app.domain.models import Candidate, Corpus, FusedCandidate, NormalizedQuery
from common.contracts import Chunk, ChunkMetadata, SearchFilters

MetadataPredicate = Callable[[ChunkMetadata], bool]


class CorpusProvider(ABC):
    """Supplies the active chunks; reloads when ingestion activates a new index version."""

    @abstractmethod
    async def current(self) -> Corpus: ...


class QueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, query: NormalizedQuery) -> NormalizedQuery: ...


class DenseRetriever(ABC):
    @abstractmethod
    async def search(self, query_text: str, corpus: Corpus, allowed_ids: Collection[str], k: int) -> list[Candidate]:
        """Top-k by vector similarity, restricted to ``allowed_ids``. Scores are cosine similarities."""

    @abstractmethod
    async def healthy(self) -> bool: ...


class SparseRetriever(ABC):
    @abstractmethod
    def search(self, terms: Sequence[str], corpus: Corpus, allowed_ids: Collection[str], k: int) -> list[Candidate]:
        """Top-k by keyword relevance (BM25), restricted to ``allowed_ids``; zero-score chunks excluded."""


class FusionStrategy(ABC):
    @abstractmethod
    def fuse(self, ranked: Mapping[str, Sequence[Candidate]]) -> list[FusedCandidate]: ...


class Reranker(ABC):
    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @abstractmethod
    async def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        """Relevance in [0, 1] for each chunk, same order as given."""


class MetadataFilterBuilder(ABC):
    @abstractmethod
    def build(self, filters: SearchFilters | None) -> MetadataPredicate: ...
