from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    num_players: int
    starting_stack: int
    small_blind: int
    big_blind: int
    ante: int = 0  # reserved for future tournament-style play, unused for now

    def __post_init__(self) -> None:
        if not (2 <= self.num_players <= 9):
            raise ValueError("num_players must be between 2 and 9")
        if self.starting_stack <= 0:
            raise ValueError("starting_stack must be positive")
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("blinds must be positive")
        if self.small_blind >= self.big_blind:
            raise ValueError("small_blind must be less than big_blind")
        if self.ante < 0:
            raise ValueError("ante cannot be negative")


@dataclass(frozen=True)
class BlindSchedule:
    """The blinds go up by `factor` every `every` hands, rounded up.

    Level `k` (the `k`-th increase, reached after `k * every` hands) has blinds
    `ceil(initial * factor ** k)`, each blind from the *initial* one rather than from the
    last rounded one: rounding up at every step would compound, and small blinds
    (1/2 at a factor of 1.2) would otherwise race ahead of the factor asked for. With
    `factor >= 1` and `small < big` the two blinds stay in order at every level, since
    their difference grows with the factor and a difference of one chip or more cannot
    round away."""

    every: int
    factor: float

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError("the blinds go up every N hands, with N at least 1")
        if not self.factor >= 1.0:
            raise ValueError("the blinds can only go up: the factor must be at least 1")

    def level(self, hands_played: int) -> int:
        """How many increases have happened before the hand after `hands_played` hands."""
        return hands_played // self.every

    def blinds(self, small_blind: int, big_blind: int, hands_played: int) -> tuple[int, int]:
        """`(small, big)` for the next hand, `hands_played` hands into the session."""
        scale = self.factor ** self.level(hands_played)
        # round() first: 100 * 1.2 is 120.00000000000001 in floating point, and ceil of that is 121.
        return (math.ceil(round(small_blind * scale, 9)), math.ceil(round(big_blind * scale, 9)))
