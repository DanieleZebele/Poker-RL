"""Encoding of an `Observation` into a flat, fixed-size feature vector.

Pure Python on purpose (`list[float]`, no numpy, no torch): this is the
highest-bug-density code in the RL section, and keeping it dependency-free puts
it inside the existing test suite. Conversion to a batched array happens once,
at the boundary in `rollout.py`.

The street is encoded as a one-hot input rather than selecting between separate
per-street networks. Pot odds, stack-to-pot ratio, position and field size are
the same concepts on every street, so a shared trunk learns them once instead of
four times on a quarter of the data each; and with a terminal-only reward, one
critic keeps a consistent value scale for GAE to bootstrap across street
boundaries. The board's 0/3/4/5 cards are handled by the card planes: a street
that has not been dealt is an all-zero plane, naturally distinct from any card.

**The encoding is versioned** (`FEATURE_VERSION`, the one new models are built at;
`SUPPORTED_VERSIONS`, everything this build can still serve). `encode_observation(..., version=)`
writes either, and a model is fed the version it was trained on, so models of both versions play
at one table. New models are built at version 3 again (2026-10-19, the user's choice): version 4
is still served, for the models of it in the store, and nothing builds it. It changes two things
from version 3:

- **One scale for every amount.** Every quantity in chips is `_amount`: big blinds through
  `log1p`, over a fixed 1000 bb, so a stack, a pot, a bet and a min-raise are in one unit with
  one resolution (a 1 bb stack is 0.10, a 100 bb stack 0.67). Version 3 mixed three units
  (the starting stack, the big blind and the pot) with a clip at 1. What stays apart are the
  few quantities that have no unit at all -- pot odds, stack-to-pot ratios -- compressed with
  `_squash` (`r / (r + scale)`), which does not saturate: a 3x-pot bet is distinct from a 1x.
- **The actions are a sequence.** The last `SEQUENCE_LENGTH` actions of the hand, one row each
  (who, what, on which street, for how much, against what, how deep the actor is left), oldest
  first and right-aligned so the most recent action is always in the same place. The per-street
  aggregates stay, for what falls out of the window. Every seat also says whether it still has
  to respond to the street's bet (`_seats_to_act`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pokerlab.cards.card import Card, Suit
from pokerlab.engine.actions import FORCED_ACTIONS, ActionType
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.engine.stats import STAT_SLOTS
from pokerlab.rl.action_space import ACTION_DIM

if TYPE_CHECKING:
    from pokerlab.players.base import Observation

MAX_SEATS = 9

# Bump whenever the *meaning* of any feature changes, even if OBS_DIM does not, and add the new
# version to `SUPPORTED_VERSIONS` while keeping the old ones served. Checkpoints record it, so
# a model trained to read slot 340 as one thing is not silently reused once that slot means
# another. Comparing dimensions alone would not catch it: the position fix that reordered
# seats kept OBS_DIM identical.
FEATURE_VERSION = 3
SUPPORTED_VERSIONS = (3, 4)

CARD_PLANES = 6
CARDS_DIM = CARD_PLANES * 52
STREET_DIM = 4
FIELD_SCALARS_DIM = 8
HISTORY_STREETS = 4
HISTORY_FEATURES = 10
HISTORY_DIM = HISTORY_STREETS * HISTORY_FEATURES
MASK_DIM = ACTION_DIM

# Version 3, as it was. Opponent statistics per seat (`engine/stats.py`) come after the base
# features every seat has; optional: a seat with no statistics supplied is all zeros there.
POT_SCALARS_DIM_V3 = 24
SEAT_BASE_FEATURES_V3 = 9
SEAT_FEATURES_V3 = SEAT_BASE_FEATURES_V3 + STAT_SLOTS
SEATS_DIM_V3 = MAX_SEATS * SEAT_FEATURES_V3
OBS_DIM_V3 = (
    CARDS_DIM
    + STREET_DIM
    + POT_SCALARS_DIM_V3
    + FIELD_SCALARS_DIM
    + SEATS_DIM_V3
    + HISTORY_DIM
    + MASK_DIM
)

# Version 4, served but no longer built.
POT_SCALARS_DIM_V4 = 17
SEAT_BASE_FEATURES_V4 = 10
SEAT_FEATURES_V4 = SEAT_BASE_FEATURES_V4 + STAT_SLOTS
SEATS_DIM_V4 = MAX_SEATS * SEAT_FEATURES_V4
SEQUENCE_LENGTH = 20
ACTION_FEATURES = 28
SEQUENCE_DIM = SEQUENCE_LENGTH * ACTION_FEATURES

OBS_DIM_V4 = (
    CARDS_DIM
    + STREET_DIM
    + POT_SCALARS_DIM_V4
    + FIELD_SCALARS_DIM
    + SEATS_DIM_V4
    + HISTORY_DIM
    + SEQUENCE_DIM
    + MASK_DIM
)

_OBS_DIM_BY_VERSION = {3: OBS_DIM_V3, 4: OBS_DIM_V4}
assert set(_OBS_DIM_BY_VERSION) == set(SUPPORTED_VERSIONS)
# The names without a suffix are the layout of the version new models are built at.
OBS_DIM = _OBS_DIM_BY_VERSION[FEATURE_VERSION]
POT_SCALARS_DIM = POT_SCALARS_DIM_V3
SEAT_BASE_FEATURES = SEAT_BASE_FEATURES_V3
SEAT_FEATURES = SEAT_FEATURES_V3
SEATS_DIM = SEATS_DIM_V3
assert OBS_DIM == OBS_DIM_V3, "the aliases above name version 3's layout"


def observation_dim(version: int) -> int:
    """How many numbers `encode_observation(..., version=version)` writes."""
    try:
        return _OBS_DIM_BY_VERSION[version]
    except KeyError:
        raise ValueError(
            f"feature version {version} is not one this build serves {SUPPORTED_VERSIONS}"
        ) from None


_SUIT_INDEX = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
_STREET_INDEX = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
_BETTING_STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
_LOG_SCALE = math.log1p(2000.0)


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _ratio(numerator: float, denominator: float) -> float:
    return _clip01(numerator / denominator) if denominator > 0 else 0.0


def _log_ratio(value: float) -> float:
    """Compress an unbounded blind-denominated quantity into [0, 1]."""
    return _clip01(math.log1p(max(value, 0.0)) / _LOG_SCALE)


# Version 4: one unit for every amount, and a compression for what has no unit.
_AMOUNT_SCALE = math.log1p(1000.0)  # 1000 big blinds: a 100 bb stack is 0.67, a 900 bb pot 0.98
SPR_SCALE = 4.0  # a stack-to-pot ratio of 4 reads 0.5


def _amount(chips: float, big_blind: int) -> float:
    """Any quantity in chips, in [0, 1]: big blinds through `log1p` over 1000 bb."""
    return _clip01(math.log1p(max(chips, 0.0) / big_blind) / _AMOUNT_SCALE)


def _squash(numerator: float, denominator: float, scale: float = 1.0) -> float:
    """`r / (r + scale)` for `r = numerator / denominator`: [0, inf) into [0, 1) without a clip.

    A bet of 3x the pot is told from 1x (0.75 against 0.5), where a clip at 1 made them the
    same number. With no denominator the ratio is as large as it can be (1.0) unless there
    is nothing to divide either (0.0)."""
    if denominator <= 0:
        return 1.0 if numerator > 0 else 0.0
    ratio = max(numerator, 0.0) / denominator
    return ratio / (ratio + scale)


def _card_index(card: Card) -> int:
    return _SUIT_INDEX[card.suit] * 13 + (int(card.rank) - 2)


def card_index(card: Card) -> int:
    """Where a card sits in a 52-long plane: `suit * 13 + rank - 2` (clubs, diamonds, hearts, spades)."""
    return _card_index(card)


def committed_by_seat(observation: Observation) -> dict[int, int]:
    """Chips each seat has put in over the whole hand.

    `SeatPublicInfo` only carries the current street's bet, but every
    `ActionRecord` stores the acting seat's running per-street total (the engine
    logs `ps.current_bet` *after* applying the action), so the max per
    (seat, street) summed over streets recovers the hand total. An ante is the
    exception: it is in the pot but not in the street's bet (its record's amount
    is 0), so it is added from the chips it took off the stack.
    """
    per_street: dict[tuple[int, Street], int] = {}
    totals: dict[int, int] = {}
    for record in observation.action_history:
        if record.action_type is ActionType.POST_ANTE:
            totals[record.seat] = totals.get(record.seat, 0) + record.stack_before - record.stack_after
            continue
        key = (record.seat, record.street)
        per_street[key] = max(per_street.get(key, 0), record.amount)
    for (seat, _street), amount in per_street.items():
        totals[seat] = totals.get(seat, 0) + amount
    return totals


def chips_at_stake(observation: Observation) -> int:
    """The most this seat can still win or lose from one opponent this hand, in chips:
    its own chips for the hand (stack plus what it has put in) against the deepest
    opponent still in it, measured the same way.

    Not the effective stack the features carry, which is what is left *behind* and
    reads 0 as soon as one opponent is all-in: this counts the chips already in too,
    so it is what the size of the hand's result goes with. A function of the
    Observation alone, which is what lets the critic's loss weigh a decision by it
    (`rl/ppo.py`)."""
    committed = committed_by_seat(observation)
    mine = observation.my_stack + committed.get(observation.my_seat, 0)
    opponents = [
        seat.stack + committed.get(seat.seat, 0)
        for seat in observation.seats
        if seat.seat != observation.my_seat and seat.status is not PlayerStatus.FOLDED
    ]
    return min(mine, max(opponents)) if opponents else mine


def _card_planes(observation: Observation) -> list[float]:
    planes = [0.0] * CARDS_DIM
    board = observation.community_cards
    groups = (
        observation.hole_cards,
        board[0:3],
        board[3:4],
        board[4:5],
        board,
        (*observation.hole_cards, *board),
    )
    for plane, cards in enumerate(groups):
        offset = plane * 52
        for card in cards:
            planes[offset + _card_index(card)] = 1.0
    return planes


def _pot_scalars_v3(observation: Observation, big_blind: int, starting_stack: int) -> list[float]:
    to_call = max(0, observation.current_bet_to_match - observation.my_current_bet)
    pot = observation.pot_size
    pot_after_call = pot + to_call
    my_stack = observation.my_stack
    committed = committed_by_seat(observation)
    my_committed = committed.get(observation.my_seat, 0)

    live_opponent_stacks = [
        seat.stack
        for seat in observation.seats
        if seat.seat != observation.my_seat and seat.status is not PlayerStatus.FOLDED
    ]
    effective_stack = min([my_stack, *live_opponent_stacks]) if live_opponent_stacks else my_stack
    total_chips = sum(seat.stack for seat in observation.seats) + pot

    spr = my_stack / pot if pot > 0 else 20.0
    effective_spr = effective_stack / pot if pot > 0 else 20.0

    return [
        _ratio(to_call, pot_after_call),
        _ratio(to_call, my_stack),
        1.0 if to_call >= my_stack else 0.0,
        1.0 if to_call > 0 else 0.0,
        _ratio(pot, total_chips),
        _ratio(my_committed, pot),
        _clip01(spr / 20.0),
        _clip01(math.log1p(spr) / math.log1p(20.0)),
        _clip01(effective_spr / 20.0),
        _clip01(math.log1p(effective_spr) / math.log1p(20.0)),
        _ratio(observation.my_current_bet, pot),
        _ratio(observation.current_bet_to_match, pot),
        _log_ratio(pot / big_blind),
        _log_ratio(my_stack / big_blind),
        _log_ratio(to_call / big_blind),
        _log_ratio(observation.min_raise / big_blind),
        _log_ratio(effective_stack / big_blind),
        _log_ratio(observation.current_bet_to_match / big_blind),
        _ratio(my_stack, starting_stack),
        _ratio(effective_stack, starting_stack),
        _ratio(pot, starting_stack),
        _ratio(to_call, starting_stack),
        _ratio(observation.min_raise, pot_after_call),
        _ratio(my_committed, starting_stack),
    ]


def _field_scalars(observation: Observation, seats_from_button: list[int]) -> list[float]:
    blind_seats = [
        record.seat
        for record in observation.action_history
        if record.action_type is ActionType.POST_BLIND
    ]
    my_seat = observation.my_seat
    num_seats = len(observation.seats)
    # Positional value follows the postflop action order, which is NOT the
    # seating order from the button: action opens on the small blind and closes
    # on the button, so the seat *before* the button is second-best while the
    # small blind is worst. Rank by how many seats act after me, not by distance
    # clockwise from the button.
    seats_acting_after_me = (num_seats - seats_from_button.index(my_seat)) % num_seats

    live_opponents = 0
    actionable_opponents = 0
    all_in_opponents = 0
    for seat in observation.seats:
        if seat.seat == my_seat:
            continue
        if seat.status is PlayerStatus.FOLDED:
            continue
        live_opponents += 1
        if seat.status is PlayerStatus.ALL_IN:
            all_in_opponents += 1
        elif seat.stack > 0:
            actionable_opponents += 1

    return [
        1.0 - seats_acting_after_me / max(num_seats - 1, 1),  # 1.0 button, 0.0 small blind
        1.0 if observation.button_seat == my_seat else 0.0,
        1.0 if blind_seats[:1] == [my_seat] else 0.0,
        1.0 if blind_seats[1:2] == [my_seat] else 0.0,
        _ratio(live_opponents, MAX_SEATS - 1),
        _ratio(actionable_opponents, MAX_SEATS - 1),
        _ratio(all_in_opponents, MAX_SEATS - 1),
        _ratio(num_seats, MAX_SEATS),
    ]


def _seat_slots_v3(observation: Observation, rotated_seats: list, starting_stack: int) -> list[float]:
    """Seats indexed relative to me (slot 0 is always me), so the encoding is
    invariant to absolute seat numbering."""
    slots = [0.0] * SEATS_DIM_V3
    pot = observation.pot_size
    committed = committed_by_seat(observation)
    for slot, seat in enumerate(rotated_seats):
        offset = slot * SEAT_FEATURES_V3
        slots[offset : offset + SEAT_BASE_FEATURES_V3] = [
            1.0,
            1.0 if seat.seat == observation.my_seat else 0.0,
            1.0 if seat.status is PlayerStatus.ACTIVE else 0.0,
            1.0 if seat.status is PlayerStatus.FOLDED else 0.0,
            1.0 if seat.status is PlayerStatus.ALL_IN else 0.0,
            _ratio(seat.stack, starting_stack),
            _ratio(seat.current_bet, pot),
            _ratio(committed.get(seat.seat, 0), pot),
            1.0 if seat.is_button else 0.0,
        ]
        # Optional: a seat with no statistics supplied keeps its stat slots at zero.
        supplied = observation.seat_stats.get(seat.seat)
        if supplied:
            if len(supplied) > STAT_SLOTS:
                raise ValueError(
                    f"seat {seat.seat} has {len(supplied)} statistics, the encoding has "
                    f"room for {STAT_SLOTS}: truncating would silently drop some"
                )
            start = offset + SEAT_BASE_FEATURES_V3
            slots[start : start + len(supplied)] = [_clip01(value) for value in supplied]
    return slots


def _history_aggregates(observation: Observation, slot_of_seat: dict[int, int]) -> list[float]:
    features: list[float] = []
    pot = observation.pot_size
    current_street = _STREET_INDEX[observation.street]

    for index, street in enumerate(_BETTING_STREETS):
        records = [r for r in observation.action_history if r.street is street]
        raises = [r for r in records if r.action_type in (ActionType.BET, ActionType.RAISE)]
        aggressors = [r.seat for r in raises]
        features.extend(
            [
                1.0 if index <= current_street else 0.0,
                _ratio(len(raises), 4),
                _ratio(sum(1 for r in records if r.action_type is ActionType.CALL), 8),
                _ratio(sum(1 for r in records if r.action_type is ActionType.CHECK), 8),
                _ratio(sum(1 for r in records if r.action_type is ActionType.FOLD), 8),
                _ratio(sum(1 for r in records if r.action_type is ActionType.ALL_IN), 4),
                _ratio(max((r.amount for r in records), default=0), pot),
                1.0 if any(r.seat == observation.my_seat for r in raises) else 0.0,
                1.0
                if any(
                    r.seat == observation.my_seat and r.action_type is ActionType.CALL
                    for r in records
                )
                else 0.0,
                _ratio(slot_of_seat[aggressors[-1]] + 1, MAX_SEATS) if aggressors else 0.0,
            ]
        )
    return features


def _seats_to_act(observation: Observation) -> set[int]:
    """The seats that still have to respond to this street's bet level.

    Read off the street's records: a bet or raise (any action whose street total goes above
    the level so far, a shove included) leaves only its author as having responded; anyone
    else who acts (call, check, fold, a short all-in) has responded; the blinds set the level
    without responding, so the big blind still has its option. Only seats that can still act
    count (`ACTIVE`: not folded, not all-in). An approximation of the engine's own `to_act`
    where a short all-in raise bars a re-raise -- the seat still has to respond, which is what
    this says."""
    level = 0
    responded: set[int] = set()
    for record in observation.action_history:
        if record.street is not observation.street:
            continue
        if record.action_type in FORCED_ACTIONS:
            level = max(level, record.amount)
        elif record.amount > level:
            level = record.amount
            responded = {record.seat}
        else:
            responded.add(record.seat)
    return {
        seat.seat
        for seat in observation.seats
        if seat.status is PlayerStatus.ACTIVE and seat.seat not in responded
    }


def _pot_scalars_v4(
    observation: Observation, big_blind: int, committed: dict[int, int]
) -> list[float]:
    to_call = max(0, observation.current_bet_to_match - observation.my_current_bet)
    pot = observation.pot_size
    pot_after_call = pot + to_call
    my_stack = observation.my_stack
    my_committed = committed.get(observation.my_seat, 0)

    live_opponent_stacks = [
        seat.stack
        for seat in observation.seats
        if seat.seat != observation.my_seat and seat.status is not PlayerStatus.FOLDED
    ]
    # Against the shortest live opponent (side pots: 0 as soon as one is all-in) and against
    # the deepest (what I can still win from someone).
    effective_shortest = min([my_stack, *live_opponent_stacks])
    effective_deepest = min(my_stack, max(live_opponent_stacks)) if live_opponent_stacks else my_stack

    return [
        # amounts, one scale
        _amount(pot, big_blind),
        _amount(my_stack, big_blind),
        _amount(effective_shortest, big_blind),
        _amount(effective_deepest, big_blind),
        _amount(to_call, big_blind),
        _amount(observation.current_bet_to_match, big_blind),
        _amount(observation.my_current_bet, big_blind),
        _amount(observation.min_raise, big_blind),
        _amount(my_committed, big_blind),
        # no unit
        _ratio(to_call, pot_after_call),  # pot odds
        _ratio(to_call, my_stack),
        _squash(my_stack, pot, SPR_SCALE),
        _squash(effective_deepest, pot, SPR_SCALE),
        _squash(observation.min_raise, pot_after_call),
        _ratio(my_committed, pot),
        # flags
        1.0 if to_call >= my_stack else 0.0,
        1.0 if to_call > 0 else 0.0,
    ]


def _seat_slots_v4(
    observation: Observation,
    rotated_seats: list,
    big_blind: int,
    committed: dict[int, int],
    to_act: set[int],
) -> list[float]:
    """Seats indexed relative to me (slot 0 is always me), so the encoding is
    invariant to absolute seat numbering."""
    slots = [0.0] * SEATS_DIM_V4
    for slot, seat in enumerate(rotated_seats):
        offset = slot * SEAT_FEATURES_V4
        slots[offset : offset + SEAT_BASE_FEATURES_V4] = [
            1.0,
            1.0 if seat.seat == observation.my_seat else 0.0,
            1.0 if seat.status is PlayerStatus.ACTIVE else 0.0,
            1.0 if seat.status is PlayerStatus.FOLDED else 0.0,
            1.0 if seat.status is PlayerStatus.ALL_IN else 0.0,
            _amount(seat.stack, big_blind),
            _amount(seat.current_bet, big_blind),
            _amount(committed.get(seat.seat, 0), big_blind),
            1.0 if seat.is_button else 0.0,
            1.0 if seat.seat in to_act else 0.0,
        ]
        supplied = observation.seat_stats.get(seat.seat)
        if supplied:
            if len(supplied) > STAT_SLOTS:
                raise ValueError(
                    f"seat {seat.seat} has {len(supplied)} statistics, the encoding has "
                    f"room for {STAT_SLOTS}: truncating would silently drop some"
                )
            start = offset + SEAT_BASE_FEATURES_V4
            slots[start : start + len(supplied)] = [_clip01(value) for value in supplied]
    return slots


# What a row of the action sequence says, in order. Forced posts (blind, ante) share one type.
_SEQUENCE_TYPES = (
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,
    ActionType.RAISE,
    ActionType.ALL_IN,
)
_TYPE_SLOT = {**{kind: i for i, kind in enumerate(_SEQUENCE_TYPES)}} | {
    ActionType.POST_BLIND: len(_SEQUENCE_TYPES),
    ActionType.POST_ANTE: len(_SEQUENCE_TYPES),
}
_ROW_ACTOR = len(_SEQUENCE_TYPES) + 1  # 7: where the actor's one-hot starts
_ROW_STREET = _ROW_ACTOR + MAX_SEATS  # 16
_ROW_AMOUNTS = _ROW_STREET + STREET_DIM  # 20
assert _ROW_AMOUNTS + 8 == ACTION_FEATURES


def _action_sequence(
    observation: Observation, slot_of_seat: dict[int, int], big_blind: int
) -> list[float]:
    """The last `SEQUENCE_LENGTH` actions of the hand, `ACTION_FEATURES` numbers each, oldest
    first and right-aligned (padding rows are all zero, so the most recent action is always
    the last row). Blinds and antes are rows like any other.

    A row says who acted (relative to me), what, on which street, and:
    - the chips it put in (`_amount`, and as a share of the pot before it);
    - the street total it took the actor to, for a bet, raise, call or all-in (`_amount`, and as
      a share of the pot before it: "raise to 3x");
    - what it faced: the bet it had to meet, as a share of the pot before it;
    - whether it left the actor all-in, and how deep it left them (stack over the pot after);
    - that the row exists (the padding is told from a real action by it).
    Walking the whole history is what tells what each action faced, so rows outside the
    window are tracked and dropped."""
    records = observation.action_history
    first = max(0, len(records) - SEQUENCE_LENGTH)
    level: dict[Street, int] = {}
    bet_of: dict[tuple[int, Street], int] = {}
    rows: list[float] = []
    for index, record in enumerate(records):
        level_before = level.get(record.street, 0)
        faced = max(level_before - bet_of.get((record.seat, record.street), 0), 0)
        level[record.street] = max(level_before, record.amount)
        bet_of[(record.seat, record.street)] = record.amount
        if index < first:
            continue

        put_in = max(record.stack_before - record.stack_after, 0)
        pot_before = record.pot_before
        total = (
            0
            if record.action_type in (ActionType.FOLD, ActionType.CHECK, ActionType.POST_ANTE)
            else record.amount
        )
        row = [0.0] * ACTION_FEATURES
        row[_TYPE_SLOT[record.action_type]] = 1.0
        row[_ROW_ACTOR + slot_of_seat[record.seat]] = 1.0
        row[_ROW_STREET + _STREET_INDEX[record.street]] = 1.0
        row[_ROW_AMOUNTS : _ROW_AMOUNTS + 8] = [
            _amount(put_in, big_blind),
            _squash(put_in, pot_before),
            _amount(total, big_blind),
            _squash(total, pot_before),
            _squash(faced, pot_before),
            1.0 if record.stack_after <= 0 else 0.0,
            _squash(record.stack_after, pot_before + put_in, SPR_SCALE),
            1.0,
        ]
        rows.extend(row)
    return [0.0] * (SEQUENCE_DIM - len(rows)) + rows


def encode_observation(
    observation: Observation,
    *,
    big_blind: int,
    starting_stack: int,
    legal_mask: Sequence[bool],
    version: int = FEATURE_VERSION,
) -> list[float]:
    """Flatten an `Observation` into `observation_dim(version)` features in [0, 1].

    `big_blind` and `starting_stack` are passed in rather than read off the
    observation: `Observation` deliberately carries no table configuration, and
    deriving the blind from the POST_BLIND records would be wrong for a short
    blind. `legal_mask` is required because echoing it into the input is what
    spares the value head from re-deriving which actions even exist.

    `version` is the encoding the model being fed was trained on. `starting_stack` is read
    only by version 3: version 4 has one scale in big blinds and no stack constant at all.
    """
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"feature version {version} is not one this build serves {SUPPORTED_VERSIONS}")
    seats = list(observation.seats)
    my_index = next(i for i, seat in enumerate(seats) if seat.seat == observation.my_seat)
    rotated = seats[my_index:] + seats[:my_index]
    slot_of_seat = {seat.seat: slot for slot, seat in enumerate(rotated)}

    button_index = next(i for i, seat in enumerate(seats) if seat.is_button)
    seats_from_button = [seat.seat for seat in seats[button_index:] + seats[:button_index]]

    street_one_hot = [0.0] * STREET_DIM
    street_one_hot[_STREET_INDEX[observation.street]] = 1.0
    mask = [1.0 if allowed else 0.0 for allowed in legal_mask]

    if version == 3:
        return [
            *_card_planes(observation),
            *street_one_hot,
            *_pot_scalars_v3(observation, big_blind, starting_stack),
            *_field_scalars(observation, seats_from_button),
            *_seat_slots_v3(observation, rotated, starting_stack),
            *_history_aggregates(observation, slot_of_seat),
            *mask,
        ]

    committed = committed_by_seat(observation)
    return [
        *_card_planes(observation),
        *street_one_hot,
        *_pot_scalars_v4(observation, big_blind, committed),
        *_field_scalars(observation, seats_from_button),
        *_seat_slots_v4(observation, rotated, big_blind, committed, _seats_to_act(observation)),
        *_history_aggregates(observation, slot_of_seat),
        *_action_sequence(observation, slot_of_seat, big_blind),
        *mask,
    ]
