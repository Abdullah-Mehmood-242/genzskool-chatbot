"""Deterministic recommender on the real catalogue: budget, level, sequencing, bundling, totals."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.domain.models import AdvisorRules, CourseOffer, Experience, LearnerProfile, Reason
from app.domain.recommend import Recommender, dedupe
from common.contracts import Level

ALL = [
    "shopify",
    "tiktok-shop",
    "ebay",
    "video-editing",
    "digital-marketing",
    "graphic-design",
    "ai-automation",
    "tiktok-automation",
]


@pytest.fixture
def rec(rules: AdvisorRules) -> Recommender:
    return Recommender(rules)


def plan_slugs(plan: object) -> list[str]:
    return [i.offer.slug for i in plan.items]  # type: ignore[attr-defined]


def test_catalogue_fixture_is_complete(offers: dict[str, CourseOffer]) -> None:
    assert set(offers) == set(ALL)
    assert offers["tiktok-automation"].fee_pkr is None
    assert offers["digital-marketing"].fee_pkr == 35000


# ---------------------------------------------------------------- spec example questions


def test_shopify_or_tiktok_shop_with_30k_budget(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    """'Shopify or TikTok Shop? Budget is Rs. 30,000.' Both cost 30,000: one now, one later."""
    plan = rec.plan(LearnerProfile(interests=("shopify", "tiktok-shop"), budget_pkr=30000), list(offers.values()))
    assert plan_slugs(plan) == ["shopify", "tiktok-shop"]
    first, second = plan.items
    assert first.now
    assert Reason.FITS_BUDGET in first.reasons
    assert not second.now
    assert Reason.LATER_BUDGET in second.reasons
    assert plan.total_now_pkr == 30000
    assert plan.total_all_pkr == 60000
    assert plan.over_budget == ()


def test_beginner_wanting_to_earn_online_gets_beginner_courses_only(
    rec: Recommender, offers: dict[str, CourseOffer]
) -> None:
    profile = LearnerProfile(goal="earn_online", experience_level=Experience.NONE)
    plan = rec.plan(profile, list(offers.values()))
    assert len(plan.items) == 3  # max_recommendations
    assert all(i.offer.level == Level.BEGINNER for i in plan.items)
    assert all(Reason.BEGINNER_FRIENDLY in i.reasons for i in plan.items)
    fees = [i.offer.fee_pkr for i in plan.items]
    assert fees == sorted(fees)  # cheapest first when nothing else decides


def test_reels_editor_gets_video_editing(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("video-editing",)), list(offers.values()))
    assert plan_slugs(plan) == ["video-editing"]
    assert plan.total_all_pkr == 30000
    assert Reason.MATCHES_INTEREST in plan.items[0].reasons


# ---------------------------------------------------------------- budget


def test_over_budget_courses_are_excluded_and_reported(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("digital-marketing", "ebay"), budget_pkr=30000), list(offers.values()))
    assert plan_slugs(plan) == ["ebay"]
    assert [o.slug for o in plan.over_budget] == ["digital-marketing"]
    assert plan.total_now_pkr == 28000


def test_nothing_affordable(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("digital-marketing",), budget_pkr=10000), list(offers.values()))
    assert plan.items == ()
    assert [o.slug for o in plan.over_budget] == ["digital-marketing"]
    assert plan.total_now_pkr is None


def test_unpublished_fee_is_kept_flagged_and_never_summed(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("tiktok-automation", "ebay"), budget_pkr=1000), list(offers.values()))
    assert plan_slugs(plan) == ["tiktok-automation"]
    item = plan.items[0]
    assert Reason.FEE_UNPUBLISHED in item.reasons
    assert item.now
    assert plan.total_now_pkr is None  # unknown fees are never treated as 0
    assert plan.unknown_fee == (offers["tiktok-automation"],)


def test_cumulative_budget_takes_what_fits_in_order(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    profile = LearnerProfile(interests=("ebay", "graphic-design", "shopify"), budget_pkr=60000)
    plan = rec.plan(profile, list(offers.values()))
    assert [(i.offer.slug, i.now) for i in plan.items] == [("ebay", True), ("graphic-design", True), ("shopify", False)]
    assert plan.total_now_pkr == 56000
    assert plan.total_all_pkr == 86000


def test_exact_budget_boundary_fits(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("digital-marketing",), budget_pkr=35000), list(offers.values()))
    assert plan_slugs(plan) == ["digital-marketing"]
    assert plan.items[0].now


# ---------------------------------------------------------------- sequencing, level, bundling


def test_prerequisite_comes_first_even_if_mentioned_second(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("shopify", "digital-marketing")), list(offers.values()))
    assert plan_slugs(plan) == ["digital-marketing", "shopify"]
    assert Reason.TAKE_FIRST in plan.items[0].reasons
    assert Reason.TAKE_FIRST not in plan.items[1].reasons


def test_chained_sequencing(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("tiktok-automation", "ai-automation")), list(offers.values()))
    assert plan_slugs(plan) == ["ai-automation", "tiktok-automation"]


def test_sequencing_cycle_in_config_does_not_hang(offers: dict[str, CourseOffer]) -> None:
    rules = AdvisorRules(sequencing=(("ebay", "shopify"), ("shopify", "ebay")))
    plan = Recommender(rules).plan(LearnerProfile(interests=("ebay", "shopify")), list(offers.values()))
    assert sorted(plan_slugs(plan)) == ["ebay", "shopify"]


def test_next_step_suggested_from_sequencing(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("digital-marketing",)), list(offers.values()))
    assert {o.slug for o in plan.next_steps} == {"shopify", "tiktok-shop"}


def test_next_steps_respect_budget(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("ai-automation",), budget_pkr=40000), list(offers.values()))
    assert [o.slug for o in plan.next_steps] == ["tiktok-automation"]  # unpublished fee: not excluded


def test_intermediate_course_flagged_for_beginners_who_ask_for_it(
    rec: Recommender, offers: dict[str, CourseOffer]
) -> None:
    profile = LearnerProfile(interests=("ai-automation", "ebay"), experience_level=Experience.NONE)
    plan = rec.plan(profile, list(offers.values()))
    ai = next(i for i in plan.items if i.offer.slug == "ai-automation")
    assert Reason.INTERMEDIATE in ai.reasons
    assert plan_slugs(plan)[0] == "ai-automation"  # explicit interest order wins over level


def test_experienced_learner_without_interests_sees_everything(
    rec: Recommender, offers: dict[str, CourseOffer]
) -> None:
    plan = rec.plan(LearnerProfile(experience_level=Experience.EXPERIENCED), [offers["ai-automation"], offers["ebay"]])
    assert set(plan_slugs(plan)) == {"ai-automation", "ebay"}


def test_mentor_bundle(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("shopify", "ebay")), list(offers.values()))
    assert plan.mentor_bundles == (("Hamza Ali", ("shopify", "ebay")),)  # in recommended order


def test_no_bundle_for_single_course_per_mentor(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("shopify", "video-editing")), list(offers.values()))
    assert plan.mentor_bundles == ()


# ---------------------------------------------------------------- limits and robustness


def test_max_recommendations_cap(offers: dict[str, CourseOffer]) -> None:
    plan = Recommender(AdvisorRules(max_recommendations=2)).plan(
        LearnerProfile(interests=tuple(ALL), experience_level=Experience.EXPERIENCED), list(offers.values())
    )
    assert len(plan.items) == 2
    assert [i.order for i in plan.items] == [1, 2]


def test_only_offers_from_evidence_are_considered(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("shopify", "ebay")), [offers["ebay"]])
    assert plan_slugs(plan) == ["ebay"]  # Shopify was not retrieved: never invented


def test_empty_evidence(rec: Recommender) -> None:
    plan = rec.plan(LearnerProfile(interests=("shopify",), budget_pkr=30000), [])
    assert plan.items == ()
    assert plan.over_budget == ()
    assert plan.total_now_pkr is None


def test_duplicate_offers_are_merged(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    dup = replace(offers["ebay"], chunk_id="other-chunk")
    assert dedupe([offers["ebay"], dup]) == [offers["ebay"]]
    assert plan_slugs(rec.plan(LearnerProfile(interests=("ebay",)), [offers["ebay"], dup])) == ["ebay"]


def test_computed_numbers_for_grounding(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    plan = rec.plan(LearnerProfile(interests=("ebay", "shopify"), budget_pkr=60000), list(offers.values()))
    assert plan.computed_numbers == {58000, 60000}


def test_deterministic(rec: Recommender, offers: dict[str, CourseOffer]) -> None:
    profile = LearnerProfile(goal="earn_online", budget_pkr=60000)
    shuffled = list(reversed(list(offers.values())))
    assert rec.plan(profile, list(offers.values())) == rec.plan(profile, shuffled)


@pytest.mark.parametrize(
    "kwargs", [{"max_clarifying_questions": 3}, {"max_recommendations": 0}, {"regenerate_on_ungrounded": -1}]
)
def test_rules_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be"):
        AdvisorRules(**kwargs)
