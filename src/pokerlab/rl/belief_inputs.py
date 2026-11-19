"""What the belief network reads, as plain numbers, and the table's memory of every player.

The belief network (`rl/belief.py`, feature version 5) estimates, for every opponent still in a
hand, the range of hole cards it holds, from public information only: the hand's actions so far,
the opponent's position and stack, the board, and a memory of how that player played the earlier
hands at this table. This module is the torch-free half of it, in the ordinary test suite:

- **tokens** (`action_tokens`): one per action record, blinds included, `TOKEN_DIM` numbers: the
  street, the actor's position, what it did (`ACTION_CLASSES`) and eight numbers on the spot it
  did it in. The actor's seat is kept apart (`HandTokens.actors`) so the network can mark which
  tokens are its target's own. The same function reads a finished `HandHistory` (the memory, the
  study that trained the network) and the prefix an `Observation` holds (a decision).
- **a target's statics** (`target_static`): its position, the table's size, its stack at the start
  of the hand, the street, how many are still in.
- **a finished hand as the memory reads it** (`memory_hand`): every token, and per player its
  result and its cards *only if it showed them* (still in at a showdown, which the engine reveals:
  the only private cards anyone at the table ever sees).
- **`MemoryTracker`**, which a `Table` is given like a `StatsTracker`: it keeps the table's
  finished hands and hands each `Observation` a `MemoryView`, through which a model's belief
  network brings its own memory of every player up to date -- lazily, when it first asks, so a
  table with no model of version 5 at it pays nothing but a list append per hand.

Cards are indices 0..51, `suit * 13 + rank - 2` (`features.card_index`); a **combo** is one of the
1326 pairs of cards, lower card first (`COMBO_CARDS`).

The study that trained the network (`studies/belief`) imports these functions, so the network is
fed in play exactly what it was trained on.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pokerlab.engine.actions import ActionType
from pokerlab.engine.history import HandHistory
from pokerlab.engine.state import ActionRecord, PlayerStatus, Street
from pokerlab.rl.features import card_index, committed_by_seat

STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
BOARD_AT = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
POSITIONS = ("BTN", "SB", "BB", "UTG", "MP", "HJ", "CO")
ACTION_CLASSES = ("blind", "fold", "check", "call", "raise", "shove")
NUM_CARDS = 52
NUM_COMBOS = 1326
# How oracle guiding gives the learner the opponents' true cards (`rl/belief.py`, `--belief-oracle-mode`):
# `mix` every vector `(1 - s) * range + s * true hand`; `swap` with probability `s` per opponent and hand
# the true hand's vector, else the range's; `near` with probability `s` the range kept to the
# `NEAR_COMBOS` combos nearest the true hand (in the encoder's space), else the whole range.
ORACLE_MODES = ("mix", "swap", "near")
NEAR_COMBOS = 64  # the default of `--belief-oracle-near`


def near_combos(near: tuple[int, ...], board_cards: int) -> int:
    """The combos `near` keeps on this street: one value for every street, or one per street (preflop,
    flop, turn, river), the street read off the board's size."""
    if len(near) == 1:
        return near[0]
    return near[{0: 0, 3: 1, 4: 2, 5: 3}[board_cards]]
MAX_SEATS = 9

# street (4) + actor position (7) + action class (6) + eight numbers on the spot.
SPOT_NUMBERS = 8
TOKEN_DIM = len(STREETS) + len(POSITIONS) + len(ACTION_CLASSES) + SPOT_NUMBERS
FOLD_COLUMN = len(STREETS) + len(POSITIONS) + ACTION_CLASSES.index("fold")
# position (7), table size, stack, street (4), players still in, and the condition: 1 when a range is
# asked of cards known because they were shown at a showdown (how the real hands of the training data
# were labelled, `studies/belief/belief_unified.py`); play always asks with 0, the range as it is.
STATIC_DIM = len(POSITIONS) + 1 + 1 + len(STREETS) + 1 + 1

STACK_SCALE = math.log1p(100.0)  # a 100 bb stack (the deepest of the mixture) reads 1
AMOUNT_SCALE = math.log1p(1000.0)

COMBO_CARDS: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in range(NUM_CARDS) for b in range(a + 1, NUM_CARDS)
)
_COMBO_OF_PAIR = {pair: index for index, pair in enumerate(COMBO_CARDS)}


def combo_index(a: int, b: int) -> int:
    """The combo of two different cards, in either order."""
    return _COMBO_OF_PAIR[(a, b) if a < b else (b, a)]


def positions(dealt: Sequence[int], button_seat: int) -> dict[int, str]:
    """Every dealt seat's position: BTN, SB, BB, then UTG, MP... up to HJ and CO.
    Heads-up the button is the small blind and is called BTN."""
    seats = sorted(dealt)
    start = seats.index(button_seat)
    order = seats[start:] + seats[:start]
    if len(order) == 2:
        return {order[0]: "BTN", order[1]: "BB"}
    names = {order[0]: "BTN", order[1]: "SB", order[2]: "BB"}
    rest = order[3:]
    late = ["CO", "HJ"]
    for index, seat in enumerate(reversed(rest)):
        names[seat] = late[index] if index < len(late) else ("UTG" if seat == rest[0] else "MP")
    return names


def _squash(ratio: float) -> float:
    return ratio / (ratio + 1.0) if ratio > 0 else 0.0


@dataclass
class HandTokens:
    """A hand read as tokens, one per action record, and what each token's decision faced."""

    tokens: list[list[float]]
    actors: list[int]  # the seat of each token's actor
    streets: list[int]  # the index in STREETS of each token's street
    voluntary: list[bool]  # False for the blinds (nobody chose them)
    live_before: list[frozenset[int]]  # the seats still in the hand just before each token


def action_tokens(
    records: Sequence[ActionRecord], dealt: Sequence[int], button_seat: int, big_blind: int
) -> HandTokens:
    """Every action record as a token (`TOKEN_DIM` numbers).

    An all-in for no more than the bet is a call (with its all-in flag), as in `engine/stats.py`;
    one above it is a `shove`. Antes are not read (training and every rated session play
    without)."""
    names = positions(dealt, button_seat)
    live = set(dealt)
    street: Street | None = None
    level, put, raises = 0, {}, 0
    found = HandTokens([], [], [], [], [])
    for record in records:
        if record.action_type == ActionType.POST_ANTE:
            continue
        if record.street != street:
            street = record.street
            level = big_blind if street == Street.PREFLOP else 0
            put, raises = {}, 0
        to_call = max(level - put.get(record.seat, 0), 0)
        kind = record.action_type
        if kind == ActionType.POST_BLIND:
            label = "blind"
        elif kind == ActionType.FOLD:
            label = "fold"
        elif kind == ActionType.CHECK:
            label = "check"
        elif kind == ActionType.CALL:
            label = "call"
        elif kind in (ActionType.BET, ActionType.RAISE):
            label = "raise"
        else:  # all-in
            label = "shove" if record.amount > level else "call"
        pot = max(record.pot_before, 1)
        raise_size = (record.amount - level) / (pot + to_call) if label in ("raise", "shove") else 0.0
        stack_bb = record.stack_before / big_blind
        numbers = [
            to_call / (pot + to_call),  # the price of a call
            _squash(raise_size),
            math.log1p(record.amount / big_blind) / AMOUNT_SCALE if label != "fold" else 0.0,
            math.log1p(stack_bb) / STACK_SCALE,
            _squash(record.stack_before / pot),  # how deep it was against the pot
            min(raises, 4) / 4.0,
            len(live) / MAX_SEATS,
            1.0 if record.stack_after == 0 and label != "fold" else 0.0,
        ]
        token = [0.0] * TOKEN_DIM
        token[STREETS.index(street)] = 1.0
        token[len(STREETS) + POSITIONS.index(names[record.seat])] = 1.0
        token[len(STREETS) + len(POSITIONS) + ACTION_CLASSES.index(label)] = 1.0
        token[len(STREETS) + len(POSITIONS) + len(ACTION_CLASSES) :] = numbers
        found.tokens.append(token)
        found.actors.append(record.seat)
        found.streets.append(STREETS.index(street))
        found.voluntary.append(label != "blind")
        found.live_before.append(frozenset(live))
        if label == "fold":
            live.discard(record.seat)
        if label == "blind":
            put[record.seat] = record.amount
            level = max(level, record.amount)
        elif label in ("call", "raise", "shove"):
            put[record.seat] = record.amount
            if record.amount > level:
                if label in ("raise", "shove"):
                    raises += 1
                level = record.amount
    return found


def target_static(position: str, players: int, stack_bb: float, street: int, live: int) -> list[float]:
    """What the network is told about its target besides the tokens (`STATIC_DIM` numbers):
    its position, the table's size, its stack at the start of the hand in big blinds, the street
    (index in `STREETS`), how many players are still in, and the condition, 0 (no condition)."""
    found = [0.0] * STATIC_DIM
    found[POSITIONS.index(position)] = 1.0
    found[len(POSITIONS)] = players / MAX_SEATS
    found[len(POSITIONS) + 1] = math.log1p(max(stack_bb, 0.0)) / STACK_SCALE
    found[len(POSITIONS) + 2 + street] = 1.0
    found[-2] = live / MAX_SEATS
    return found


def decision_question(observation: Any, big_blind: int) -> dict[str, Any] | None:
    """What the belief network is asked at a decision, in plain numbers (None: no opponent is
    still in): the live opponents' seats (`targets`), the hand's tokens so far and their actors,
    each target's statics, the visible board (sorted card indices) and every card the observer
    sees (`known`: the board and its own cards, which no opponent can hold)."""
    me = observation.my_seat
    live = [seat for seat in observation.seats if seat.status is not PlayerStatus.FOLDED]
    targets = [seat for seat in live if seat.seat != me]
    if not targets:
        return None
    dealt = [seat.seat for seat in observation.seats]
    read = action_tokens(observation.action_history, dealt, observation.button_seat, big_blind)
    names = positions(dealt, observation.button_seat)
    committed = committed_by_seat(observation)
    street = STREETS.index(observation.street)
    board = tuple(sorted(card_index(card) for card in observation.community_cards))
    static = [
        target_static(names[t.seat], len(dealt), (t.stack + committed.get(t.seat, 0)) / big_blind,
                      street, len(live))
        for t in targets
    ]
    known = set(board) | {card_index(card) for card in observation.hole_cards}
    return {"targets": [t.seat for t in targets], "tokens": read.tokens, "actors": read.actors,
            "static": static, "board": board, "known": known}


@dataclass(frozen=True)
class MemoryPlayer:
    """One player of a finished hand, as the memory reads it."""

    player_id: str
    seat: int
    shown: bool  # its cards were revealed at a showdown
    cards: tuple[int, int]  # (-1, -1) unless shown
    result_bb: float  # what its stack did


@dataclass(frozen=True)
class MemoryHand:
    """A finished hand as the memory reads it: every token, the final board (-1 for a card not
    dealt) and every player dealt in."""

    tokens: list[list[float]]
    actors: list[int]
    board: tuple[int, ...]
    players: tuple[MemoryPlayer, ...]


def memory_hand(hand: HandHistory, player_ids: Mapping[int, str]) -> MemoryHand:
    """`hand` as the memory reads it. A player's cards are read only if they were shown: still in
    the hand, with someone else still in, when it ended."""
    dealt = sorted(hand.starting_stacks)
    read = action_tokens(hand.actions, dealt, hand.button_seat, hand.big_blind)
    still_in = set(dealt)
    for actor, token in zip(read.actors, read.tokens, strict=True):
        if token[FOLD_COLUMN] == 1.0:
            still_in.discard(actor)
    showdown = len(still_in) >= 2
    board = [card_index(card) for card in hand.community_cards]
    players = []
    for seat in dealt:
        shown = showdown and seat in still_in
        pair = hand.hole_cards[seat]
        players.append(MemoryPlayer(
            player_id=player_ids[seat],
            seat=seat,
            shown=shown,
            cards=(card_index(pair[0]), card_index(pair[1])) if shown else (-1, -1),
            result_bb=(hand.final_stacks[seat] - hand.starting_stacks[seat]) / hand.big_blind,
        ))
    return MemoryHand(read.tokens, read.actors, tuple(board + [-1] * (5 - len(board))), tuple(players))


# What a reader of the memory passes to bring its own states up to date: the hands it has not
# read yet, in order, and its states by player id (missing: never seen), which it updates.
MemoryUpdate = Callable[[Sequence[MemoryHand], dict[str, Any]], None]


# The last hands a tracker always keeps, read or not: a model of version 5 asks in the first hand
# or two it sits in (every player but the big blind acts preflop), so a reader that first asks a
# little after another still finds every hand, and a table with no such model holds a short list.
KEEP_WITHOUT_READERS = 20


class MemoryTracker:
    """A table's memory of its players, for the belief networks of the models sitting there.

    `Table(memory_tracker=)` calls `record(hand, player_ids)` after every hand and `view(player_ids)`
    before every hand; the view rides in each `Observation` (`seat_memory`). What a player has done
    is kept as the table's finished hands, not as numbers: each belief network has its own memory
    network, so each *reader* (a key, the fingerprint of its network) keeps its own states and
    reads the hands it has not read yet when it next asks (`MemoryView.states`). Nothing is
    computed for a table where nobody asks, and nothing here imports torch.

    The last `KEEP_WITHOUT_READERS` hands are always kept; an older one is dropped once every reader
    has read it (at once, with none), so a table of version-3 models holds a short list.

    Identity is the `player_id`, as for the statistics, and the tracker is replaced when the
    players change (`TableBank.reset_stats`): what it remembers describes nobody new."""

    def __init__(self) -> None:
        self._hands: list[tuple[HandHistory, dict[int, str]] | MemoryHand] = []
        self._first = 0  # the number of the first hand still kept
        self._count = 0
        self._readers: dict[str, tuple[int, dict[str, Any]]] = {}

    def record(self, hand: HandHistory, player_ids: Mapping[int, str]) -> None:
        self._hands.append((hand, dict(player_ids)))
        self._count += 1
        self._drop_read()

    def _drop_read(self) -> None:
        limit = self._count - KEEP_WITHOUT_READERS
        if self._readers:
            limit = min(limit, min(done for done, _ in self._readers.values()))
        if limit > self._first:
            del self._hands[: limit - self._first]
            self._first = limit

    def view(self, player_ids: Mapping[int, str]) -> MemoryView:
        return MemoryView(self, dict(player_ids), self._count)

    @property
    def hands(self) -> int:
        return self._count

    def _states(self, key: str, update: MemoryUpdate, upto: int) -> dict[str, Any]:
        done, states = self._readers.get(key, (self._first, {}))
        if done > upto:
            raise RuntimeError(
                f"the memory was asked about the table after {upto} hands, but reader {key} has read "
                f"{done}: a view is the memory of the hand it was handed with, and that hand is over"
            )
        if done < upto:
            hands = []
            for number in range(done, upto):
                kept = self._hands[number - self._first]
                if not isinstance(kept, MemoryHand):
                    kept = self._hands[number - self._first] = memory_hand(*kept)
                hands.append(kept)
            update(hands, states)
        self._readers[key] = (upto, states)
        self._drop_read()
        return states


class MemoryView:
    """What an `Observation` carries of the table's memory: the tracker, who sits where in this
    hand and how many hands came before it. `states(key, update)` is every seat's state for the
    reader `key` after those hands (None for a player the table has not seen play one); asked
    once a later hand has been read it raises, since what it describes is gone."""

    def __init__(self, tracker: MemoryTracker, player_ids: dict[int, str], hands: int) -> None:
        self._tracker = tracker
        self.player_ids = player_ids
        self.hands = hands

    def states(self, key: str, update: MemoryUpdate) -> dict[int, Any]:
        found = self._tracker._states(key, update, self.hands)
        return {seat: found.get(player_id) for seat, player_id in self.player_ids.items()}
