from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pokerlab.cards.card import Card
from pokerlab.cards.deck import Deck
from pokerlab.engine.actions import ActionType


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class PlayerStatus(Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"
    BUSTED = "busted"  # excluded before the hand even started (stack was already 0)


@dataclass
class PlayerState:
    """Per-hand state for one occupied seat. Table keeps persistent chip
    counts across hands; this is rebuilt fresh for every hand."""

    seat: int
    player_id: str
    name: str
    stack: int
    hole_cards: tuple[Card, Card] | None = None
    current_bet: int = 0  # chips committed on the current street
    total_committed: int = 0  # chips committed across the whole hand
    status: PlayerStatus = PlayerStatus.ACTIVE

    def commit(self, amount: int) -> None:
        """Move `amount` chips from this player's stack into the pot."""
        if amount < 0:
            raise ValueError("cannot commit a negative amount")
        if amount > self.stack:
            raise ValueError(f"seat {self.seat} cannot commit {amount}, only has {self.stack}")
        self.stack -= amount
        self.current_bet += amount
        self.total_committed += amount
        if self.stack == 0 and self.status == PlayerStatus.ACTIVE:
            self.status = PlayerStatus.ALL_IN


@dataclass
class ActionRecord:
    street: Street
    seat: int
    player_id: str
    action_type: ActionType
    amount: int
    stack_before: int
    stack_after: int
    pot_before: int
    timestamp: float


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible_seats: frozenset[int]


@dataclass
class HandState:
    hand_id: str
    button_seat: int
    seats: list[PlayerState]
    deck: Deck
    small_blind: int
    big_blind: int
    community_cards: list[Card] = field(default_factory=list)
    street: Street = Street.PREFLOP
    current_bet_to_match: int = 0
    min_raise: int = 0
    to_act: set[int] = field(default_factory=set)
    raise_barred: set[int] = field(default_factory=set)
    action_log: list[ActionRecord] = field(default_factory=list)

    def seat_state(self, seat: int) -> PlayerState:
        for ps in self.seats:
            if ps.seat == seat:
                return ps
        raise KeyError(f"no seat {seat} in this hand")

    def pot_total(self) -> int:
        return sum(ps.total_committed for ps in self.seats)

    def hand_active_seats(self) -> list[PlayerState]:
        """Non-folded seats: still eligible to win the pot at showdown."""
        return [ps for ps in self.seats if ps.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)]

    def actionable_seats(self) -> list[PlayerState]:
        """Seats that can still make a betting decision this round."""
        return [ps for ps in self.seats if ps.status == PlayerStatus.ACTIVE and ps.stack > 0]
