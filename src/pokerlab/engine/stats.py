"""Opponent statistics (a HUD): who plays how, over the last few hundred hands.

A `StatsTracker` watches finished hands and keeps, per player, a sliding window of
what they did and what chances they had to do it. `Table` can be given one; if it
is, every `Observation` carries each seat's statistics as a vector of up to
`STAT_SLOTS` numbers in [0, 1]. **Passing them is optional, seat by seat**: a seat
with no entry in `Observation.seat_stats` (no tracker, a player never seen, the
spot screen with no history) is encoded as zeros, which is also what an
all-unknown vector looks like, so there is no special case for "not given".

**What is counted.** Every statistic is a pair, *events* over *opportunities*,
because "3-bet 20%" over three chances and over three hundred are different facts:

    VPIP       put chips in preflop voluntarily      / every hand dealt in
    PFR        raised preflop                        / every hand dealt in
    3BET       re-raised an open                     / faced exactly one raise
    F3BET      folded to a re-raise of their open    / opened and were re-raised
    STEAL      raised first in from CO/BTN/SB        / first in from CO/BTN/SB
    AGG        bet or raised                         / bet, raised or called postflop
    CBET       bet the flop as the last preflop raiser / had the chance to
    FCBET      folded to that flop bet               / faced it
    WTSD       reached showdown                      / saw the flop

A statistic is counted **once per hand per player**, at their first chance (3BET,
F3BET, STEAL, CBET, FCBET); AGG counts every postflop decision that puts chips in.
An all-in for no more than what is already on the table is a call, not a raise,
which is why the amounts of the engine's own `ActionRecord`s decide it: they hold
the street total after the action.

**The vector** (`stat_vector`, `USED_SLOTS` = `STAT_SLOTS` = 20 numbers, all of them
what the model's input holds for a seat):

    0       1.0, "statistics supplied"
    1       hands in the window, log-scaled
    2k+2    rate of statistic k (0 when it had no opportunity)
    2k+3    its opportunities, log-scaled      (k = 0..8, in the order above)

Rate and count travel together so a model can weigh a 60% over 4 chances differently
from a 60% over 80, and zero opportunities reads as unknown with no special case.
**Changing what any slot means is a change of `FEATURE_VERSION`** (`rl/features.py`).

**The window is per player identity**, `player_id`, and slides over the last
`WINDOW` hands that player was dealt into. The tracker only means something where
identity is stable across hands -- a session at one table, a real room. A player
whose seat is filled by a different model every hand has statistics describing
nobody, which is the reason the training collector does not use it yet.

Pure Python, imports nothing from `players/` or `rl/`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pokerlab.engine.actions import ActionType
from pokerlab.engine.state import ActionRecord, Street

# What the model's input holds per player: exactly the vector this module fills. (There used
# to be 100 slots a seat, 80 of them reserved for later and always zero; they were removed
# because a statistic added later changes what the input means anyway, which is a new
# `FEATURE_VERSION` whatever the room.)
USED_SLOTS = 20
STAT_SLOTS = USED_SLOTS

WINDOW = 200

STATS = (
    "vpip",
    "pfr",
    "three_bet",
    "fold_to_three_bet",
    "steal",
    "aggression",
    "cbet",
    "fold_to_cbet",
    "wtsd",
)
STAT_COUNT = len(STATS)
assert 2 + 2 * STAT_COUNT == USED_SLOTS

_LOG_WINDOW = math.log1p(WINDOW)
_VOLUNTARY = (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)


def _scaled(count: int) -> float:
    return min(math.log1p(count) / _LOG_WINDOW, 1.0)


@dataclass(frozen=True)
class HandCounts:
    """What one player did in one hand: `(events, opportunities)` per statistic."""

    events: tuple[int, ...]
    opportunities: tuple[int, ...]


def _steal_seats(dealt: Sequence[int], button_seat: int) -> set[int]:
    """Cutoff, button and small blind: where a first-in raise is a steal."""
    ordered = sorted(dealt)
    start = ordered.index(button_seat)
    clockwise = ordered[start:] + ordered[:start]  # button first
    seats = {button_seat}
    if len(clockwise) >= 3:
        seats.add(clockwise[1])  # small blind
    if len(clockwise) >= 4:
        seats.add(clockwise[-1])  # cutoff
    return seats


def analyse_hand(
    actions: Sequence[ActionRecord],
    *,
    dealt: Sequence[int],
    button_seat: int,
    board_cards: int,
    player_ids: Mapping[int, str],
) -> dict[str, HandCounts]:
    """Every dealt-in player's counts for one finished hand, by `player_id`.

    `actions` is the hand's full action log in order, `dealt` the seats that were
    dealt cards, `board_cards` how many community cards were dealt (so a flop that
    was never reached is known), `player_ids` the identity of every dealt seat.
    """
    # An ante is no decision and no bet (its record's amount is 0): left in, it would read
    # as a first action and open a steal or 3-bet chance for whoever paid it.
    actions = [record for record in actions if record.action_type is not ActionType.POST_ANTE]
    events = {seat: [0] * STAT_COUNT for seat in dealt}
    chances = {seat: [0] * STAT_COUNT for seat in dealt}
    index = {name: i for i, name in enumerate(STATS)}

    def count(seat: int, stat: str, happened: bool) -> None:
        chances[seat][index[stat]] += 1
        if happened:
            events[seat][index[stat]] += 1

    steal_seats = _steal_seats(dealt, button_seat)
    folded_preflop: set[int] = set()
    folded: set[int] = set()
    for record in actions:
        if record.action_type is ActionType.FOLD:
            folded.add(record.seat)
            if record.street is Street.PREFLOP:
                folded_preflop.add(record.seat)

    # ---- preflop ----------------------------------------------------------
    level = 0
    raises = 0
    voluntary_before = 0
    opener: int | None = None
    last_aggressor: int | None = None
    voluntary = set()
    raised = set()
    seen_three_bet = set()
    seen_fold_to_three_bet = set()
    seen_steal = set()
    for record in actions:
        if record.street is not Street.PREFLOP:
            continue
        seat = record.seat
        kind = record.action_type
        if kind is ActionType.POST_BLIND:
            level = max(level, record.amount)
            continue
        if kind is ActionType.CHECK:
            continue
        aggressive = kind in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN) and (
            record.amount > level
        )
        if raises == 1 and seat not in seen_three_bet:
            seen_three_bet.add(seat)
            count(seat, "three_bet", aggressive)
        if seat == opener and raises >= 2 and seat not in seen_fold_to_three_bet:
            seen_fold_to_three_bet.add(seat)
            count(seat, "fold_to_three_bet", kind is ActionType.FOLD)
        if raises == 0 and voluntary_before == 0 and seat in steal_seats and seat not in seen_steal:
            seen_steal.add(seat)
            count(seat, "steal", aggressive)
        if kind in _VOLUNTARY:
            voluntary.add(seat)
            voluntary_before += 1
        if aggressive:
            raised.add(seat)
            raises += 1
            level = record.amount
            last_aggressor = seat
            if opener is None:
                opener = seat
    for seat in dealt:
        count(seat, "vpip", seat in voluntary)
        count(seat, "pfr", seat in raised)

    # ---- flop to river ---------------------------------------------------
    saw_flop = {seat for seat in dealt if seat not in folded_preflop} if board_cards >= 3 else set()
    seen_fold_to_cbet: set[int] = set()
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        level = 0
        bets = 0
        acted: set[int] = set()
        cbet_seat: int | None = None
        for record in actions:
            if record.street is not street or record.action_type is ActionType.POST_BLIND:
                continue
            seat = record.seat
            kind = record.action_type
            aggressive = kind in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN) and (
                record.amount > level
            )
            if street is Street.FLOP:
                if seat == last_aggressor and seat not in acted and level == 0:
                    count(seat, "cbet", aggressive)
                    if aggressive:
                        cbet_seat = seat
                elif (
                    cbet_seat is not None
                    and seat != cbet_seat
                    and bets == 1
                    and seat not in seen_fold_to_cbet
                ):
                    seen_fold_to_cbet.add(seat)
                    count(seat, "fold_to_cbet", kind is ActionType.FOLD)
            if kind in _VOLUNTARY:
                count(seat, "aggression", aggressive)
            if aggressive:
                bets += 1
                level = record.amount
            acted.add(seat)

    # ---- showdown ---------------------------------------------------------
    standing = [seat for seat in dealt if seat not in folded]
    for seat in saw_flop:
        count(seat, "wtsd", seat in standing and len(standing) >= 2)

    return {
        player_ids[seat]: HandCounts(tuple(events[seat]), tuple(chances[seat])) for seat in dealt
    }


class StatsTracker:
    """A sliding window of `WINDOW` hands per player."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = window
        self._hands: dict[str, deque[HandCounts]] = {}
        self._events: dict[str, list[int]] = {}
        self._chances: dict[str, list[int]] = {}

    def record_hand(
        self,
        actions: Sequence[ActionRecord],
        *,
        dealt: Sequence[int],
        button_seat: int,
        board_cards: int,
        player_ids: Mapping[int, str],
    ) -> None:
        counts = analyse_hand(
            actions,
            dealt=dealt,
            button_seat=button_seat,
            board_cards=board_cards,
            player_ids=player_ids,
        )
        for player_id, hand in counts.items():
            self.add(player_id, hand)

    def add(self, player_id: str, hand: HandCounts) -> None:
        """One hand's counts for one player. Public so several seats can be pooled
        under one identity -- the training collector pools every seat the learner
        sat in, to describe the *model* rather than a seat."""
        window = self._hands.setdefault(player_id, deque())
        events = self._events.setdefault(player_id, [0] * STAT_COUNT)
        chances = self._chances.setdefault(player_id, [0] * STAT_COUNT)
        window.append(hand)
        for i in range(STAT_COUNT):
            events[i] += hand.events[i]
            chances[i] += hand.opportunities[i]
        if len(window) > self._window:
            old = window.popleft()
            for i in range(STAT_COUNT):
                events[i] -= old.events[i]
                chances[i] -= old.opportunities[i]

    def forget(self, player_id: str) -> None:
        """Drop everything known about a player: whoever sat in a seat has left."""
        self._hands.pop(player_id, None)
        self._events.pop(player_id, None)
        self._chances.pop(player_id, None)

    def hands(self, player_id: str) -> int:
        return len(self._hands.get(player_id, ()))

    def rates(self, player_id: str) -> dict[str, tuple[int, int]]:
        """`(events, opportunities)` per statistic over the window."""
        events = self._events.get(player_id, [0] * STAT_COUNT)
        chances = self._chances.get(player_id, [0] * STAT_COUNT)
        return {name: (events[i], chances[i]) for i, name in enumerate(STATS)}

    def vector(self, player_id: str) -> tuple[float, ...] | None:
        """The player's `USED_SLOTS` numbers, or None if they were never seen."""
        hands = self.hands(player_id)
        if hands == 0:
            return None
        slots = [1.0, _scaled(hands)]
        for events, chances in self.rates(player_id).values():
            slots.append(events / chances if chances > 0 else 0.0)
            slots.append(_scaled(chances))
        return tuple(slots)

    def vectors(self, player_ids: Mapping[int, str]) -> dict[int, tuple[float, ...]]:
        """The vector of every seat whose player has been seen, by seat."""
        found: dict[int, tuple[float, ...]] = {}
        for seat, player_id in player_ids.items():
            vector = self.vector(player_id)
            if vector is not None:
                found[seat] = vector
        return found
