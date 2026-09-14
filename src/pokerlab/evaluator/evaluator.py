from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

from pokerlab.cards.card import Card

_WHEEL = (14, 5, 4, 3, 2)


class HandCategory(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


_CATEGORY_NAMES = {
    HandCategory.HIGH_CARD: "High Card",
    HandCategory.PAIR: "Pair",
    HandCategory.TWO_PAIR: "Two Pair",
    HandCategory.THREE_OF_A_KIND: "Three of a Kind",
    HandCategory.STRAIGHT: "Straight",
    HandCategory.FLUSH: "Flush",
    HandCategory.FULL_HOUSE: "Full House",
    HandCategory.FOUR_OF_A_KIND: "Four of a Kind",
    HandCategory.STRAIGHT_FLUSH: "Straight Flush",
}


@dataclass(frozen=True, order=True)
class HandRank:
    """Comparable strength of a 5-card poker hand.

    Comparison is purely by (category, tiebreakers), which is exactly the
    field order here, so plain dataclass ordering already implements correct
    poker hand comparison. `best_five` is excluded from comparison since it
    is only for display/logging.
    """

    category: HandCategory
    tiebreakers: tuple[int, ...]
    best_five: tuple[Card, ...] = field(compare=False)

    def describe(self) -> str:
        return f"{_CATEGORY_NAMES[self.category]} ({', '.join(str(c) for c in self.best_five)})"


def _straight_high(unique_ranks_desc: list[int]) -> int | None:
    """Given exactly 5 distinct ranks sorted descending, return the straight's
    high card, or None if they are not a straight. Handles the A-2-3-4-5
    wheel, whose high card is 5, not the ace."""
    if len(unique_ranks_desc) != 5:
        return None
    if unique_ranks_desc[0] - unique_ranks_desc[-1] == 4:
        return unique_ranks_desc[0]
    if tuple(unique_ranks_desc) == _WHEEL:
        return 5
    return None


def rank_five(cards: Sequence[Card]) -> HandRank:
    """Rank exactly 5 cards. Use `evaluate` for 5-7 card hands (e.g. hole + board)."""
    if len(cards) != 5:
        raise ValueError(f"rank_five requires exactly 5 cards, got {len(cards)}")

    cards = tuple(cards)
    ranks = [c.rank.value for c in cards]
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    unique_ranks_desc = sorted(set(ranks), reverse=True)
    straight_high = _straight_high(unique_ranks_desc)

    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Groups ordered by (count desc, rank desc): this single ordering is the
    # correct tiebreaker sequence for every non-straight category (quads,
    # full house, trips, two pair, pair, high card all fall out of it).
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    group_tiebreak = tuple(rank for rank, _count in groups)
    counts_desc = tuple(sorted(counts.values(), reverse=True))

    if is_flush and straight_high is not None:
        return HandRank(HandCategory.STRAIGHT_FLUSH, (straight_high,), cards)
    if counts_desc == (4, 1):
        return HandRank(HandCategory.FOUR_OF_A_KIND, group_tiebreak, cards)
    if counts_desc == (3, 2):
        return HandRank(HandCategory.FULL_HOUSE, group_tiebreak, cards)
    if is_flush:
        return HandRank(HandCategory.FLUSH, group_tiebreak, cards)
    if straight_high is not None:
        return HandRank(HandCategory.STRAIGHT, (straight_high,), cards)
    if counts_desc == (3, 1, 1):
        return HandRank(HandCategory.THREE_OF_A_KIND, group_tiebreak, cards)
    if counts_desc == (2, 2, 1):
        return HandRank(HandCategory.TWO_PAIR, group_tiebreak, cards)
    if counts_desc == (2, 1, 1, 1):
        return HandRank(HandCategory.PAIR, group_tiebreak, cards)
    return HandRank(HandCategory.HIGH_CARD, group_tiebreak, cards)


def evaluate(cards: Sequence[Card]) -> HandRank:
    """Best HandRank achievable from 5-7 cards (e.g. 2 hole + up to 5 board cards).

    Brute-forces every 5-card combination (at most C(7,5) = 21) rather than
    using lookup tables or bit tricks, trading a little speed for an
    implementation that is straightforward to verify by hand.
    """
    if len(cards) < 5:
        raise ValueError(f"evaluate requires at least 5 cards, got {len(cards)}")
    return max(rank_five(combo) for combo in itertools.combinations(cards, 5))


def compare(a: HandRank, b: HandRank) -> int:
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


class HandEvaluator:
    """Thin object-oriented facade over the module-level evaluation functions."""

    @staticmethod
    def evaluate(hole: tuple[Card, Card], board: Sequence[Card]) -> HandRank:
        return evaluate((*hole, *board))

    @staticmethod
    def evaluate_cards(cards: Sequence[Card]) -> HandRank:
        return evaluate(cards)

    @staticmethod
    def compare(a: HandRank, b: HandRank) -> int:
        return compare(a, b)
