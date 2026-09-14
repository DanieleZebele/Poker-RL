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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pokerlab.cards.card import Card, Suit
from pokerlab.engine.actions import ActionType
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.rl.action_space import ACTION_DIM

if TYPE_CHECKING:
    from pokerlab.players.base import Observation

MAX_SEATS = 9

# Bump whenever the *meaning* of any feature changes, even if OBS_DIM does not.
# Checkpoints record it, so a model trained to read slot 340 as one thing is not
# silently reused once that slot means another. Comparing dimensions alone would
# not catch it: the position fix that reordered seats kept OBS_DIM identical.
FEATURE_VERSION = 1

CARD_PLANES = 6
CARDS_DIM = CARD_PLANES * 52
STREET_DIM = 4
POT_SCALARS_DIM = 24
FIELD_SCALARS_DIM = 8
SEAT_FEATURES = 9
SEATS_DIM = MAX_SEATS * SEAT_FEATURES
HISTORY_STREETS = 4
HISTORY_FEATURES = 10
HISTORY_DIM = HISTORY_STREETS * HISTORY_FEATURES
MASK_DIM = ACTION_DIM

OBS_DIM = (
    CARDS_DIM
    + STREET_DIM
    + POT_SCALARS_DIM
    + FIELD_SCALARS_DIM
    + SEATS_DIM
    + HISTORY_DIM
    + MASK_DIM
)

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


def _card_index(card: Card) -> int:
    return _SUIT_INDEX[card.suit] * 13 + (int(card.rank) - 2)


def committed_by_seat(observation: Observation) -> dict[int, int]:
    """Chips each seat has put in over the whole hand.

    `SeatPublicInfo` only carries the current street's bet, but every
    `ActionRecord` stores the acting seat's running per-street total (the engine
    logs `ps.current_bet` *after* applying the action), so the max per
    (seat, street) summed over streets recovers the hand total.
    """
    per_street: dict[tuple[int, Street], int] = {}
    for record in observation.action_history:
        key = (record.seat, record.street)
        per_street[key] = max(per_street.get(key, 0), record.amount)
    totals: dict[int, int] = {}
    for (seat, _street), amount in per_street.items():
        totals[seat] = totals.get(seat, 0) + amount
    return totals


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


def _pot_scalars(observation: Observation, big_blind: int, starting_stack: int) -> list[float]:
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


def _seat_slots(observation: Observation, rotated_seats: list, starting_stack: int) -> list[float]:
    """Seats indexed relative to me (slot 0 is always me), so the encoding is
    invariant to absolute seat numbering."""
    slots = [0.0] * SEATS_DIM
    pot = observation.pot_size
    committed = committed_by_seat(observation)
    for slot, seat in enumerate(rotated_seats):
        offset = slot * SEAT_FEATURES
        slots[offset : offset + SEAT_FEATURES] = [
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


def encode_observation(
    observation: Observation,
    *,
    big_blind: int,
    starting_stack: int,
    legal_mask: Sequence[bool],
) -> list[float]:
    """Flatten an `Observation` into `OBS_DIM` features in [0, 1].

    `big_blind` and `starting_stack` are passed in rather than read off the
    observation: `Observation` deliberately carries no table configuration, and
    deriving the blind from the POST_BLIND records would be wrong for a short
    blind. `legal_mask` is required because echoing it into the input is what
    spares the value head from re-deriving which actions even exist.
    """
    seats = list(observation.seats)
    my_index = next(i for i, seat in enumerate(seats) if seat.seat == observation.my_seat)
    rotated = seats[my_index:] + seats[:my_index]
    slot_of_seat = {seat.seat: slot for slot, seat in enumerate(rotated)}

    button_index = next(i for i, seat in enumerate(seats) if seat.is_button)
    seats_from_button = [seat.seat for seat in seats[button_index:] + seats[:button_index]]

    street_one_hot = [0.0] * STREET_DIM
    street_one_hot[_STREET_INDEX[observation.street]] = 1.0

    return [
        *_card_planes(observation),
        *street_one_hot,
        *_pot_scalars(observation, big_blind, starting_stack),
        *_field_scalars(observation, seats_from_button),
        *_seat_slots(observation, rotated, starting_stack),
        *_history_aggregates(observation, slot_of_seat),
        *(1.0 if allowed else 0.0 for allowed in legal_mask),
    ]
