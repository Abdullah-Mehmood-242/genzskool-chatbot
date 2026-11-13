"""Deterministic recommendation rules (spec 6.3): eligibility, ordering and cost are decided here,
never by the LLM. Input is only what retrieval returned, so nothing can be invented."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.domain.models import (
    AdvisorRules,
    CourseOffer,
    LearnerProfile,
    Reason,
    RecommendationItem,
    RecommendationPlan,
)
from common.contracts import Level

_LEVEL_RANK = {Level.BEGINNER: 0, None: 1, Level.INTERMEDIATE: 2}


def dedupe(offers: Iterable[CourseOffer]) -> list[CourseOffer]:
    seen: dict[str, CourseOffer] = {}
    for offer in offers:
        seen.setdefault(offer.slug, offer)
    return list(seen.values())


class Recommender:
    def __init__(self, rules: AdvisorRules) -> None:
        self.rules = rules
        self._before: dict[str, set[str]] = {}  # slug -> slugs that should come before it
        for first, then in rules.sequencing:
            self._before.setdefault(then, set()).add(first)

    # ------------------------------------------------------------------ public

    def plan(self, profile: LearnerProfile, offers: Sequence[CourseOffer]) -> RecommendationPlan:
        catalogue = dedupe(offers)
        candidates = self._candidates(profile, catalogue)
        budget = profile.budget_pkr
        over_budget = tuple(
            o for o in candidates if budget is not None and o.fee_pkr is not None and o.fee_pkr > budget
        )
        eligible = [o for o in candidates if o not in over_budget]
        ordered = self._order(eligible, profile)[: self.rules.max_recommendations]
        items = self._items(ordered, profile)
        known_now = [i.offer.fee_pkr for i in items if i.now and i.offer.fee_pkr is not None]
        known_all = [i.offer.fee_pkr for i in items if i.offer.fee_pkr is not None]
        picked = {i.offer.slug for i in items}
        return RecommendationPlan(
            items=items,
            budget_pkr=budget,
            total_now_pkr=sum(known_now) if known_now else None,
            total_all_pkr=sum(known_all) if known_all else None,
            over_budget=over_budget,
            mentor_bundles=self._bundles(items),
            next_steps=tuple(
                o
                for o in catalogue
                if o.slug not in picked
                and any(o.slug in self._after(p) for p in picked)
                and (budget is None or o.fee_pkr is None or o.fee_pkr <= budget)
            ),
        )

    # ------------------------------------------------------------------ rules

    def _candidates(self, profile: LearnerProfile, catalogue: list[CourseOffer]) -> list[CourseOffer]:
        by_slug = {o.slug: o for o in catalogue}
        if profile.interests:
            return [by_slug[s] for s in profile.interests if s in by_slug]
        # No stated interest: beginners get beginner courses only; others get everything found.
        if profile.is_beginner:
            return [o for o in catalogue if o.level != Level.INTERMEDIATE]
        return list(catalogue)

    def _after(self, slug: str) -> set[str]:
        return {then for then, firsts in self._before.items() if slug in firsts}

    def _order(self, offers: list[CourseOffer], profile: LearnerProfile) -> list[CourseOffer]:
        interest_rank = {s: i for i, s in enumerate(profile.interests)}

        def key(o: CourseOffer) -> tuple[int, int, int, str]:
            return (
                interest_rank.get(o.slug, len(interest_rank)),
                _LEVEL_RANK[o.level] if profile.is_beginner else 0,
                o.fee_pkr if o.fee_pkr is not None else 10**9,  # unpublished fees sort last
                o.slug,
            )

        remaining = sorted(offers, key=key)
        ordered: list[CourseOffer] = []
        # Stable topological sort: take the first remaining course whose prerequisites (among the
        # remaining picks) are already placed. Cycles in config fall back to the key order.
        while remaining:
            placed = {o.slug for o in ordered}
            pending = {o.slug for o in remaining}
            ready = next(
                (o for o in remaining if not (self._before.get(o.slug, set()) & pending) - placed),
                remaining[0],
            )
            ordered.append(ready)
            remaining.remove(ready)
        return ordered

    def _items(self, ordered: list[CourseOffer], profile: LearnerProfile) -> tuple[RecommendationItem, ...]:
        budget = profile.budget_pkr
        spent = 0
        items: list[RecommendationItem] = []
        slugs = {o.slug for o in ordered}
        for n, offer in enumerate(ordered, start=1):
            reasons: list[Reason] = []
            if offer.slug in profile.interests:
                reasons.append(Reason.MATCHES_INTEREST)
            if offer.level == Level.BEGINNER and profile.is_beginner:
                reasons.append(Reason.BEGINNER_FRIENDLY)
            if offer.level == Level.INTERMEDIATE:
                reasons.append(Reason.INTERMEDIATE)
            if self._after(offer.slug) & slugs:
                reasons.append(Reason.TAKE_FIRST)
            now = True
            if offer.fee_pkr is None:
                reasons.append(Reason.FEE_UNPUBLISHED)
            elif budget is not None:
                if spent + offer.fee_pkr <= budget:
                    spent += offer.fee_pkr
                    reasons.append(Reason.FITS_BUDGET)
                else:
                    now = False
                    reasons.append(Reason.LATER_BUDGET)
            items.append(RecommendationItem(offer=offer, order=n, reasons=tuple(reasons), now=now))
        return tuple(items)

    @staticmethod
    def _bundles(items: tuple[RecommendationItem, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
        by_mentor: dict[str, list[str]] = {}
        for item in items:
            if item.offer.mentor:
                by_mentor.setdefault(item.offer.mentor, []).append(item.offer.slug)
        return tuple((m, tuple(s)) for m, s in sorted(by_mentor.items()) if len(s) >= 2)
