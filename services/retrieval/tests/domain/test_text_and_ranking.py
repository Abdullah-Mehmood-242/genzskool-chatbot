"""Pure domain logic: analysis, normalisation, coverage, filters, RRF, candidate ranking, config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import make_chunk

from app.domain.models import Candidate, Corpus, NormalizedQuery, RetrievalConfig
from app.domain.ranking import DefaultFilterBuilder, QueryNormalizer, ReciprocalRankFusion, ranked
from app.domain.text import STOPWORDS, analyze, normalize, tokenize
from common.contracts import DocType, Level, SearchFilters

# ---------------------------------------------------------------- text


@pytest.mark.parametrize(
    ("raw", "out"),
    [
        ("Rs. 30,000", "rs. 30000"),
        ("Rs 1,25,000", "rs 125000"),
        ("12,34,56,789", "123456789"),
        ("a,b 1,2", "a,b 1,2"),  # not thousands groups
        ("  Shopify STORE \n", "shopify store"),
        ("Ｓｈｏｐｉｆｙ", "shopify"),
        ("it’s", "it's"),
    ],
)
def test_normalize(raw: str, out: str) -> None:
    assert normalize(raw) == out


def test_tokenize_lemmas_possessives_and_dotted_words() -> None:
    assert tokenize("Courses' fees at Make.com, Genz Skool's mentors") == [
        "course",
        "fee",
        "at",
        "make.com",
        "genz",
        "skool",
        "mentor",
    ]
    assert tokenize("FEES 30,000") == ["fee", "30000"]


def test_analyze_drops_english_and_roman_urdu_stopwords() -> None:
    assert analyze("How much is the Shopify course?") == ["shopify", "course"]
    assert analyze("kitne paise hain ebay course ke") == ["kitne", "paise", "ebay", "course"]
    assert analyze("mujhe kya karna chahiye") == []
    assert analyze("hi hello thanks") == []


def test_analyze_ignores_non_latin_script() -> None:
    assert analyze("شاپیفائی کورس") == []


def test_stopwords_never_hide_course_names() -> None:
    for word in ("shopify", "tiktok", "ebay", "video", "editing", "marketing", "design", "automation", "fee", "mentor"):
        assert word not in STOPWORDS


# ---------------------------------------------------------------- normaliser & coverage


def test_normalizer_expands_synonyms_once_and_keeps_groups() -> None:
    n = QueryNormalizer({"price": ["fee"], "cost": ["fee"], "reel": ["video", "editing"]})
    q = n.normalize("Price and cost of reels? price!")
    assert q.terms == ("price", "cost", "reels", "price")
    assert q.expanded_terms == ("price", "cost", "reels", "fee")
    assert q.term_groups == (("price", "fee"), ("cost", "fee"), ("reels",))
    assert q.text == "price and cost of reels? price! fee"


def test_synonym_keys_are_analyzed_like_queries() -> None:
    n = QueryNormalizer({"Fees": ["price"]})  # 'Fees' lemmatises to 'fee'
    assert n.normalize("fees").expanded_terms == ("fee", "price")


@pytest.mark.parametrize("bad_key", ["the", "two words", ""])
def test_invalid_synonym_keys_rejected(bad_key: str) -> None:
    with pytest.raises(ValueError, match="single content word"):
        QueryNormalizer({bad_key: ["x"]})


def test_normalizer_without_synonyms() -> None:
    q = QueryNormalizer().normalize("Shopify course")
    assert q.expanded_terms == q.terms == ("shopify", "course")
    assert q.text == "shopify course"


def test_coverage_counts_user_terms_matched_directly_or_by_synonym() -> None:
    q = QueryNormalizer({"price": ["fee"]}).normalize("shopify price dropshipping")
    assert q.coverage({"shopify", "fee"}) == (2, pytest.approx(2 / 3))
    assert q.coverage({"unrelated"}) == (0, 0.0)
    assert NormalizedQuery("", "", (), ()).coverage({"x"}) == (0, 0.0)


@pytest.mark.parametrize(
    ("matched", "coverage", "passes"),
    [(2, 0.2, True), (1, 0.5, True), (1, 1.0, True), (1, 0.33, False), (0, 0.0, False), (0, 1.0, False)],
)
def test_passes_coverage(matched: int, coverage: float, passes: bool) -> None:
    assert RetrievalConfig().passes_coverage(matched, coverage) is passes


# ---------------------------------------------------------------- filters


CHUNKS = [
    make_chunk(
        "shopify", "x", fee=30000, slug="shopify", category="E-commerce", level=Level.BEGINNER, mentor="Hamza Ali"
    ),
    make_chunk(
        "dm",
        "x",
        fee=35000,
        slug="digital-marketing",
        category="Marketing",
        level=Level.BEGINNER,
        mentor="Mahnoor Raza",
    ),
    make_chunk(
        "ta",
        "x",
        fee=None,
        slug="tiktok-automation",
        category="AI Automation / Social",
        level=Level.INTERMEDIATE,
        mentor="Sana Iqbal",
    ),
    make_chunk("policy", "x", doc_type=DocType.POLICY),
    make_chunk("mentor", "x", doc_type=DocType.MENTOR, mentor="Hamza Ali"),
]


def _ids(filters: SearchFilters | None) -> set[str]:
    keep = DefaultFilterBuilder().build(filters)
    return {c.id for c in CHUNKS if keep(c.metadata)}


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (None, {"shopify", "dm", "ta", "policy", "mentor"}),
        (SearchFilters(), {"shopify", "dm", "ta", "policy", "mentor"}),
        (SearchFilters(max_fee_pkr=30000), {"shopify", "ta", "policy", "mentor"}),  # unknown fee kept, flagged later
        (SearchFilters(max_fee_pkr=29999), {"ta", "policy", "mentor"}),
        (SearchFilters(max_fee_pkr=0), {"ta", "policy", "mentor"}),
        (SearchFilters(doc_types=[DocType.COURSE]), {"shopify", "dm", "ta"}),
        (SearchFilters(doc_types=[DocType.POLICY, DocType.MENTOR]), {"policy", "mentor"}),
        (SearchFilters(doc_types=[]), set()),
        (SearchFilters(course_slugs=["Shopify", "ta-none"]), {"shopify"}),
        (SearchFilters(category="marketing"), {"dm"}),
        (SearchFilters(level=Level.INTERMEDIATE), {"ta"}),
        (SearchFilters(mentor="hamza ali"), {"shopify", "mentor"}),
        (SearchFilters(max_fee_pkr=40000, level=Level.BEGINNER, doc_types=[DocType.COURSE]), {"shopify", "dm"}),
    ],
)
def test_filter_builder(filters: SearchFilters | None, expected: set[str]) -> None:
    assert _ids(filters) == expected


# ---------------------------------------------------------------- RRF


def _c(cid: str, rank: int, score: float = 1.0) -> Candidate:
    return Candidate(chunk_id=cid, score=score, rank=rank)


def test_rrf_rank_one_everywhere_scores_one() -> None:
    fused = ReciprocalRankFusion(60).fuse({"dense": [_c("a", 1)], "sparse": [_c("a", 1)]})
    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(1.0)
    assert set(fused[0].contributions) == {"dense", "sparse"}


def test_rrf_single_list_caps_at_half() -> None:
    fused = ReciprocalRankFusion(60).fuse({"dense": [], "sparse": [_c("a", 1)]})
    assert fused[0].score == pytest.approx(0.5)


def test_rrf_agreement_beats_single_strong_vote() -> None:
    fused = ReciprocalRankFusion(60).fuse(
        {"dense": [_c("both", 3), _c("dense_only", 1)], "sparse": [_c("both", 3), _c("sparse_only", 1)]}
    )
    assert fused[0].chunk_id == "both"


def test_rrf_formula_and_normalisation() -> None:
    k = 10
    fused = {
        f.chunk_id: f.score for f in ReciprocalRankFusion(k).fuse({"x": [_c("a", 1), _c("b", 4)], "y": [_c("b", 2)]})
    }
    best = 2 / (k + 1)
    assert fused["a"] == pytest.approx((1 / (k + 1)) / best)
    assert fused["b"] == pytest.approx((1 / (k + 4) + 1 / (k + 2)) / best)


def test_rrf_ignores_duplicate_votes_from_one_list() -> None:
    fused = ReciprocalRankFusion(60).fuse({"x": [_c("a", 1), _c("a", 2)]})
    assert fused[0].score == pytest.approx(1.0)


def test_rrf_deterministic_tie_break() -> None:
    fused = ReciprocalRankFusion(60).fuse({"x": [_c("b", 1)], "y": [_c("a", 1)]})
    assert [f.chunk_id for f in fused] == ["a", "b"]


def test_rrf_empty_inputs() -> None:
    assert ReciprocalRankFusion().fuse({}) == []
    assert ReciprocalRankFusion().fuse({"x": [], "y": []}) == []


def test_rrf_scores_within_unit_interval() -> None:
    lists = {"x": [_c(str(i), i) for i in range(1, 30)], "y": [_c(str(i), 30 - i) for i in range(1, 30)]}
    assert all(0 < f.score <= 1 for f in ReciprocalRankFusion(5).fuse(lists))


def test_rrf_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        ReciprocalRankFusion(0)


# ---------------------------------------------------------------- ranked() and config


def test_ranked_filters_sorts_ties_and_cuts() -> None:
    out = ranked([("b", 0.5), ("a", 0.5), ("c", 0.9), ("z", 0.0), ("neg", -0.3), ("low", 0.1)], min_score=0.2, k=3)
    assert [(c.chunk_id, c.rank) for c in out] == [("c", 1), ("a", 2), ("b", 3)]


def test_ranked_min_zero_still_excludes_zero_scores() -> None:
    assert [c.chunk_id for c in ranked([("a", 0.0), ("b", 0.01)], min_score=0.0, k=5)] == ["b"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"top_k": 10, "candidate_k": 5},
        {"fusion_k": 0},
        {"score_threshold": 1.5},
        {"dense_min_score": -0.1},
        {"sparse_min_score": -1},
        {"min_term_matches": 0},
        {"min_term_coverage": 2},
    ],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"must|need"):
        RetrievalConfig(**kwargs)  # type: ignore[arg-type]


def test_corpus_by_id_and_len() -> None:
    corpus = Corpus(chunks=tuple(CHUNKS), versions=(("d", 1),), signature="s")
    assert len(corpus) == 5
    assert corpus.by_id["dm"].metadata.fee_pkr == 35000
    assert len(Corpus()) == 0
