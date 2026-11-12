"""Pure ranking logic: query normalisation, metadata filtering, Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.domain.models import Candidate, FusedCandidate, NormalizedQuery
from app.domain.ports import FusionStrategy, MetadataFilterBuilder, MetadataPredicate
from app.domain.text import analyze, normalize
from common.contracts import ChunkMetadata, SearchFilters


class QueryNormalizer:
    """Normalise a query and expand it with configured synonyms (deterministic, no LLM)."""

    def __init__(self, synonyms: Mapping[str, Sequence[str]] | None = None) -> None:
        self.synonyms: dict[str, tuple[str, ...]] = {}
        for key, values in (synonyms or {}).items():
            analyzed_key = analyze(key)
            if len(analyzed_key) != 1:
                raise ValueError(f"synonym key must be a single content word: {key!r}")
            expanded = tuple(t for v in values for t in analyze(v))
            self.synonyms[analyzed_key[0]] = expanded

    def normalize(self, query: str) -> NormalizedQuery:
        text = normalize(query)
        terms = tuple(analyze(query))
        expanded: list[str] = list(dict.fromkeys(terms))
        extra: list[str] = []
        groups: list[tuple[str, ...]] = []
        for term in dict.fromkeys(terms):  # de-duplicated, order kept
            syns = self.synonyms.get(term, ())
            groups.append((term, *syns))
            for syn in syns:
                if syn not in expanded:
                    expanded.append(syn)
                    extra.append(syn)
        dense_text = f"{text} {' '.join(extra)}".strip()
        return NormalizedQuery(
            original=query, text=dense_text, terms=terms, expanded_terms=tuple(expanded), term_groups=tuple(groups)
        )


class DefaultFilterBuilder(MetadataFilterBuilder):
    """AND of every set filter. ``max_fee_pkr`` keeps chunks without a fee (policies, unpublished fees)."""

    def build(self, filters: SearchFilters | None) -> MetadataPredicate:
        if filters is None:
            return lambda _m: True
        doc_types = set(filters.doc_types) if filters.doc_types is not None else None
        slugs = {s.lower() for s in filters.course_slugs} if filters.course_slugs is not None else None
        category = filters.category.casefold() if filters.category else None
        mentor = filters.mentor.casefold() if filters.mentor else None

        def keep(m: ChunkMetadata) -> bool:
            if filters.max_fee_pkr is not None and m.fee_pkr is not None and m.fee_pkr > filters.max_fee_pkr:
                return False
            if doc_types is not None and m.doc_type not in doc_types:
                return False
            if slugs is not None and (m.course_slug or "").lower() not in slugs:
                return False
            if category is not None and (m.category or "").casefold() != category:
                return False
            if filters.level is not None and m.level != filters.level:
                return False
            return not (mentor is not None and (m.mentor or "").casefold() != mentor)

        return keep


class ReciprocalRankFusion(FusionStrategy):
    """score(d) = sum over lists of 1 / (k + rank), normalised by the best possible score.

    Normalising by ``n_lists / (k + 1)`` puts scores in [0, 1]: a chunk ranked first by every
    retriever scores 1.0; a chunk only one of two retrievers found scores at most 0.5.
    """

    def __init__(self, k: int = 60) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def fuse(self, ranked: Mapping[str, Sequence[Candidate]]) -> list[FusedCandidate]:
        if not ranked:
            return []
        totals: dict[str, float] = {}
        contributions: dict[str, dict[str, Candidate]] = {}
        for name, candidates in ranked.items():
            seen: set[str] = set()
            for cand in candidates:
                if cand.chunk_id in seen:  # a retriever must not vote twice for one chunk
                    continue
                seen.add(cand.chunk_id)
                totals[cand.chunk_id] = totals.get(cand.chunk_id, 0.0) + 1.0 / (self.k + cand.rank)
                contributions.setdefault(cand.chunk_id, {})[name] = cand
        best = len(ranked) / (self.k + 1)
        fused = [
            FusedCandidate(chunk_id=cid, score=min(1.0, total / best), contributions=contributions[cid])
            for cid, total in totals.items()
        ]
        # Deterministic: score desc, then number of retrievers agreeing, then id.
        return sorted(fused, key=lambda f: (-f.score, -len(f.contributions), f.chunk_id))


def ranked(candidates: Sequence[tuple[str, float]], *, min_score: float, k: int) -> list[Candidate]:
    """Sort (id, score) pairs, keep those with score > 0 and >= ``min_score``, cut to k, rank from 1.

    Ties break on chunk id so results are deterministic.
    """
    kept = sorted((c for c in candidates if c[1] > 0 and c[1] >= min_score), key=lambda c: (-c[1], c[0]))
    return [Candidate(chunk_id=cid, score=score, rank=i) for i, (cid, score) in enumerate(kept[:k], start=1)]
