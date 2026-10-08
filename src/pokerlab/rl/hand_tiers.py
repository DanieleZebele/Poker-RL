"""The 169 starting hands, ranked and cut into five bands.

Shared by the opponents' styles (`rl/styles.py`, which loosen and tighten mostly the
marginal hands) and the studies of how a model plays (`studies/agents`). The ranking is
Bill Chen's formula -- quick, well known, and good enough to tell the best 5% from the
worst 40% -- cut by share of the 1,326 combinations. Pure Python.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pokerlab.cards.card import Card

RANK_SYMBOLS = "23456789TJQKA"  # index + 2 is the rank


def rank_symbol(rank: int) -> str:
    return RANK_SYMBOLS[rank - 2]


def hand_class(hole: Sequence[Card]) -> str:
    """One of the 169 starting hands: "AA", "AKs", "T9o"."""
    high, low = sorted((int(card.rank) for card in hole), reverse=True)
    if high == low:
        return rank_symbol(high) * 2
    suited = hole[0].suit == hole[1].suit
    return f"{rank_symbol(high)}{rank_symbol(low)}{'s' if suited else 'o'}"


def _class_ranks(cls: str) -> tuple[int, int]:
    return RANK_SYMBOLS.index(cls[0]) + 2, RANK_SYMBOLS.index(cls[1]) + 2


def chen_score(cls: str) -> int:
    """Bill Chen's points for a starting hand: a quick, well-known ranking."""
    high, low = _class_ranks(cls)
    points = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}.get(high, high / 2)
    if high == low:
        return math.ceil(max(points * 2, 5))
    if cls.endswith("s"):
        points += 2
    gap = high - low - 1
    points -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and high < 12:
        points += 1
    return math.ceil(points)


def combos(cls: str) -> int:
    return 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12


ALL_CLASSES = tuple(
    rank_symbol(high) * 2 if high == low else f"{rank_symbol(high)}{rank_symbol(low)}{suffix}"
    for high in range(14, 1, -1)
    for low in range(high, 1, -1)
    for suffix in (("",) if high == low else ("s", "o"))
)
TIERS = ("top 5%", "5-15%", "15-35%", "35-60%", "ultimo 40%")
_TIER_EDGES = (0.05, 0.15, 0.35, 0.60, 1.01)


def _tier_table() -> dict[str, str]:
    """Each starting hand's band: ranked by Chen points (pairs, then the higher card, break
    ties), cut where the share of combinations seen so far crosses each edge."""
    ranked = sorted(
        ALL_CLASSES,
        key=lambda cls: (-chen_score(cls), len(cls) != 2, -_class_ranks(cls)[0], -_class_ranks(cls)[1]),
    )
    table: dict[str, str] = {}
    seen = 0
    for cls in ranked:
        share = seen / 1326
        table[cls] = next(name for name, edge in zip(TIERS, _TIER_EDGES) if share < edge)
        seen += combos(cls)
    return table


TIER_OF = _tier_table()
