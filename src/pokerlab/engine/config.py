from __future__ import annotations

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
