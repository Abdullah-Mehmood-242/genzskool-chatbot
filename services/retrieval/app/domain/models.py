"""Retrieval domain models. Pure Python."""

from __future__ import annotations

from dataclasses import dataclass, field

from common.contracts import Chunk


@dataclass(frozen=True)
class Corpus:
    """The active chunks of every indexed document (as named by the ingestion registry)."""

    chunks: tuple[Chunk, ...] = ()
    versions: tuple[tuple[str, int], ...] = ()  # (doc_id, active index_version)
    signature: str = "empty"  # changes whenever the active set changes
    warnings: tuple[str, ...] = ()
    by_id: dict[str, Chunk] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_id", {c.id: c for c in self.chunks})

    def __len__(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    text: str  # normalised query text (+ synonym words) used for dense search
    terms: tuple[str, ...]  # content terms from the user's words
    expanded_terms: tuple[str, ...]  # terms + synonyms, used for BM25
    term_groups: tuple[tuple[str, ...], ...] = ()  # per user term: (term, *synonyms), for coverage

    def coverage(self, chunk_tokens: set[str]) -> tuple[int, float]:
        """How many of the user's terms (or their synonyms) the chunk contains, and that as a fraction."""
        if not self.term_groups:
            return 0, 0.0
        matched = sum(1 for group in self.term_groups if any(t in chunk_tokens for t in group))
        return matched, matched / len(self.term_groups)


@dataclass(frozen=True)
class Candidate:
    """One retriever's opinion about one chunk. ``rank`` is 1-based within that retriever."""

    chunk_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: str
    score: float  # normalised to [0, 1]
    contributions: dict[str, Candidate] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20
    fusion_k: int = 60
    score_threshold: float = 0.3
    dense_min_score: float = 0.15
    sparse_min_score: float = 0.0
    min_term_matches: int = 2
    min_term_coverage: float = 0.5
    use_reranker: bool = False
    use_query_rewrite: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.candidate_k < self.top_k:
            raise ValueError("need 1 <= top_k <= candidate_k")
        if self.fusion_k < 1:
            raise ValueError("fusion_k must be >= 1")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0.0 <= self.dense_min_score <= 1.0:
            raise ValueError("dense_min_score must be in [0, 1]")
        if self.sparse_min_score < 0:
            raise ValueError("sparse_min_score must be >= 0")
        if self.min_term_matches < 1:
            raise ValueError("min_term_matches must be >= 1")
        if not 0.0 <= self.min_term_coverage <= 1.0:
            raise ValueError("min_term_coverage must be in [0, 1]")

    def passes_coverage(self, matched: int, coverage: float) -> bool:
        """Keyword candidates need enough of the query: one stray common word is not relevance."""
        return matched >= self.min_term_matches or (matched > 0 and coverage >= self.min_term_coverage)
