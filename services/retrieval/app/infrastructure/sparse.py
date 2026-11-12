"""BM25 keyword retriever (rank_bm25), rebuilt whenever the corpus signature changes."""

from __future__ import annotations

import math
import threading
from collections.abc import Collection, Mapping, Sequence

from rank_bm25 import BM25Okapi

from app.domain.models import Candidate, Corpus
from app.domain.ports import SparseRetriever
from app.domain.ranking import ranked
from app.domain.text import analyze


class LuceneIdfBM25(BM25Okapi):  # type: ignore[misc]
    """BM25Okapi with Lucene's IDF: ln(1 + (N - n + 0.5) / (n + 0.5)), always > 0.

    rank_bm25's default IDF is <= 0 for terms in at least half the documents (exactly 0 at
    n = N/2), so common but meaningful words such as "fee" could silently stop matching as the
    corpus grows. Lucene/Elasticsearch use this variant for exactly that reason.
    """

    def _calc_idf(self, nd: Mapping[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


class Bm25SparseRetriever(SparseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._signature: str | None = None
        self._ids: list[str] = []
        self._index: LuceneIdfBM25 | None = None
        self._lock = threading.Lock()

    def _ensure(self, corpus: Corpus) -> LuceneIdfBM25 | None:
        with self._lock:
            if self._signature != corpus.signature:
                self._ids = [c.id for c in corpus.chunks]
                docs = [analyze(c.text) for c in corpus.chunks]
                # rank_bm25 divides by the average document length: guard the all-empty corpus.
                self._index = LuceneIdfBM25(docs, k1=self.k1, b=self.b) if any(docs) else None
                self._signature = corpus.signature
            return self._index

    def search(self, terms: Sequence[str], corpus: Corpus, allowed_ids: Collection[str], k: int) -> list[Candidate]:
        if not terms or not len(corpus):
            return []
        index = self._ensure(corpus)
        if index is None:
            return []
        scores = index.get_scores(list(terms))
        allowed = set(allowed_ids)
        pairs = [(cid, float(s)) for cid, s in zip(self._ids, scores, strict=True) if cid in allowed]
        return ranked(pairs, min_score=0.0, k=k)
