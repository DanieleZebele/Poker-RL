"""How does one agent play? A self-play study of a single model.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/agent_study.py --model '#1'
    ... --model <label | path/to/agent.pt | #N> --hands 40000 --jobs 16
    ... --players 6 --stack-bb 100          # one table size, everyone 100 bb deep
    ... --save hands.jsonl / --load hands.jsonl   # play once, read the report many times
    ... --made-hands [--per-class 400]    # not self-play: does it recognise straights and flushes?

The model sits in every seat (it plays against itself) on the training mixture of tables
and stacks unless `--players`/`--stack-bb` fix them, through `TableBank.play_session`, so
it reads the opponent statistics as it does in a rated session. Every hand is then read
back from its `HandHistory` -- every action with its street total, every player's cards and
the board -- and the report answers:

1. the usual statistics (VPIP, PFR, 3-bet, ...), from `engine/stats.py`;
2. preflop, by strength of the starting hand (five bands of the 169 hands, ranked by the
   Chen formula and cut by share of the 1,326 combinations: the best 5%, 5-15%, 15-35%,
   35-60%, the worst 40%): what it does when nobody has raised yet, facing a raise,
   facing a re-raise; and the same by position;
3. two 13x13 grids: how often each hand raises when nobody has raised yet, and how often
   it enters the pot at all;
4. after the flop, by what it holds (nothing, a draw, a pair, top pair or better, two
   pair, trips/a set, a straight or better): checked to, and facing a bet;
5. bluffs: what it bets the river with, how often a river bet is behind a player still in
   the hand (the cards are known: it is self-play), how often it bets with nothing when
   checked to, and how often that works;
6. strong hands it does not raise: the slowplays (checking or just calling a monster,
   limping the best 5%) and the strong hands it folds;
7. what each preflop band wins, in bb a hand (self-play: it sums to zero over the bands);
8. a few example hands of each kind.

`--made-hands` replaces all of this with a different test, in `made_hands.py`: fixed
heads-up spots with straights, the wheel and flushes, their twins with the combination broken,
and what the model does leading and facing a bet.

The analysis half (everything that reads a `HandHistory`) is plain Python and torch-free,
so it is tested in the ordinary suite; torch is imported only to play.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from pokerlab.cards.card import Card
from pokerlab.engine.actions import ActionType
from pokerlab.engine.history import HandHistory, HandHistoryReader, HandHistoryWriter
from pokerlab.engine.state import Street
from pokerlab.engine.stats import STATS, analyse_hand
from pokerlab.evaluator.evaluator import HandCategory, evaluate
from pokerlab.rl.hand_tiers import (
    ALL_CLASSES,
    RANK_SYMBOLS,
    TIER_OF,
    TIERS,
    hand_class,
    rank_symbol,
)
from pokerlab.rl.style_log import size_group
from pokerlab.rl.value_diagnostics import STACK_LABELS, stack_label

# ---- the starting hands: `rl/hand_tiers.py`, plus the two groups of bands read here ----

STRONG_TIERS = TIERS[:2]  # the best 15%
WEAK_TIERS = TIERS[-1:]  # the worst 40%

# ---- what a hand holds after the flop ---------------------------------------

MADE_CLASSES = ("nulla", "progetto", "coppia", "top pair+", "due coppie", "tris", "mostro")
STRONG_MADE = ("due coppie", "tris", "mostro")


def _has_draw(hole: Sequence[Card], board: Sequence[Card]) -> bool:
    """A flush draw or an open-ended straight draw that uses one of its own cards."""
    cards = [*hole, *board]
    for suit in {card.suit for card in hole}:
        if sum(card.suit == suit for card in cards) == 4:
            return True
    ranks = {int(card.rank) for card in cards}
    own = {int(card.rank) for card in hole}
    if 14 in ranks:
        ranks.add(1)
    if 14 in own:
        own.add(1)
    for low in range(2, 11):  # both ends open: low - 1 and low + 4 exist
        window = set(range(low, low + 4))
        if window <= ranks and window & own:
            return True
    return False


def made_class(hole: Sequence[Card], board: Sequence[Card]) -> str:
    """What a hand holds after the flop, judged by what its own cards add to the board."""
    full = evaluate([*hole, *board])
    if full.category >= HandCategory.STRAIGHT:
        if len(board) == 5 and evaluate(list(board)) >= full:
            return "nulla"  # it plays the board
        return "mostro"
    board_counts = Counter(int(card.rank) for card in board)
    first, second = (int(card.rank) for card in hole)
    if first == second and board_counts[first]:
        return "tris"  # a set
    hits = {rank for rank in (first, second) if board_counts[rank]}
    if any(board_counts[rank] == 2 for rank in hits):
        return "tris"
    if first != second and len(hits) == 2:
        return "due coppie"
    top = max(board_counts)
    if first == second:
        return "top pair+" if first > top else "coppia"
    if hits:
        return "top pair+" if hits.pop() == top else "coppia"
    if len(board) < 5 and _has_draw(hole, board):
        return "progetto"
    return "nulla"


# ---- positions ----------------------------------------------------------------

POSITION_GROUPS = ("iniziale/media", "tardiva", "bui")
POSITION_ORDER = ("UTG", "MP", "HJ", "CO", "BTN", "SB", "BB")  # the preflop order of play


def positions(hand: HandHistory) -> dict[int, str]:
    """Every dealt seat's position name: BTN, SB, BB, then UTG... up to HJ and CO."""
    seats = sorted(hand.starting_stacks)
    start = seats.index(hand.button_seat)
    order = seats[start:] + seats[:start]  # the button first
    if len(order) == 2:
        return {order[0]: "BTN", order[1]: "BB"}  # heads-up the button posts the small blind
    names = {order[0]: "BTN", order[1]: "SB", order[2]: "BB"}
    rest = order[3:]
    late = ["CO", "HJ"]
    for index, seat in enumerate(reversed(rest)):
        names[seat] = late[index] if index < len(late) else ("UTG" if seat == rest[0] else "MP")
    return names


def position_group(name: str) -> str:
    if name in ("SB", "BB"):
        return "bui"
    if name in ("BTN", "CO"):
        return "tardiva"
    return "iniziale/media"


# ---- decisions read back from a hand ----------------------------------------

_BOARD_AT = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
KINDS = ("fold", "check", "call", "raise")


@dataclass(frozen=True)
class Decision:
    hand: int  # index of the hand in the study
    seat: int
    street: Street
    position: str
    players: int  # dealt in
    live: int  # still in the hand when it acted, itself included
    hole: tuple[Card, Card]
    board: tuple[Card, ...]
    to_call: int
    raises_before: int  # bets and raises already made on this street (blinds not counted)
    kind: str  # fold, check, call, raise (a bet, a raise or an all-in above the bet)
    amount: int  # its street total after the action
    level_before: int  # the bet to match before it acted
    pot_before: int
    stack_before: int
    all_in: bool
    big_blind: int
    # The last bet or raise of the street was an all-in: usually no raise left to make (one
    # may still exist when a third player with chips is in, which this does not tell apart).
    facing_all_in: bool = False

    # Cached: a decision is read by the whole study and by its group's.
    @cached_property
    def cls(self) -> str:
        return hand_class(self.hole)

    @cached_property
    def tier(self) -> str:
        return TIER_OF[self.cls]

    @cached_property
    def made(self) -> str:
        return made_class(self.hole, self.board)


def decisions_of(hand: HandHistory, index: int = 0) -> list[Decision]:
    """Every voluntary action of a hand, with what the player faced when it took it.

    An all-in for no more than the bet is a call, as in `engine/stats.py`."""
    names = positions(hand)
    players = len(hand.starting_stacks)
    folded: set[int] = set()
    street: Street | None = None
    level, put, raises, shoved = 0, {}, 0, False
    found: list[Decision] = []
    for record in hand.actions:
        if record.street != street:
            street = record.street
            level = hand.big_blind if street == Street.PREFLOP else 0
            put, raises, shoved = {}, 0, False
        if record.action_type == ActionType.POST_BLIND:
            put[record.seat] = record.amount
            continue
        to_call = max(level - put.get(record.seat, 0), 0)
        kind = {
            ActionType.FOLD: "fold",
            ActionType.CHECK: "check",
            ActionType.CALL: "call",
            ActionType.BET: "raise",
            ActionType.RAISE: "raise",
        }.get(record.action_type)
        if kind is None:  # all-in
            kind = "raise" if record.amount > level else "call"
        found.append(
            Decision(
                hand=index,
                seat=record.seat,
                street=street,
                position=names[record.seat],
                players=players,
                live=players - len(folded),
                hole=hand.hole_cards[record.seat],
                board=tuple(hand.community_cards[: _BOARD_AT[street]]),
                to_call=to_call,
                raises_before=raises,
                kind=kind,
                amount=record.amount,
                level_before=level,
                pot_before=record.pot_before,
                stack_before=record.stack_before,
                all_in=record.stack_after == 0,
                big_blind=hand.big_blind,
                facing_all_in=shoved and to_call > 0,
            )
        )
        if kind == "raise":
            raises += 1
            level = record.amount
            shoved = record.stack_after == 0
        if kind == "fold":
            folded.add(record.seat)
        put[record.seat] = record.amount
    return found


def river_standing(hand: HandHistory, decision: Decision, still_in: Iterable[int]) -> str:
    """On the river, against the cards of the players still in: davanti, pari or dietro."""
    board = list(hand.community_cards[:5])
    mine = evaluate([*decision.hole, *board])
    others = [evaluate([*hand.hole_cards[seat], *board]) for seat in still_in if seat != decision.seat]
    if not others:
        return "davanti"
    best_other = max(others)
    if mine > best_other:
        return "davanti"
    return "pari" if mine == best_other else "dietro"


def last_standing(hand: HandHistory) -> int | None:
    """The seat that won without a showdown, everyone else having folded; else None."""
    folded = {record.seat for record in hand.actions if record.action_type == ActionType.FOLD}
    left = [seat for seat in hand.starting_stacks if seat not in folded]
    return left[0] if len(left) == 1 else None


# ---- the tallies ----------------------------------------------------------------


@dataclass
class Tally:
    """Counts of what was done, per key."""

    counts: dict[object, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def add(self, key: object, what: str) -> None:
        self.counts[key][what] += 1

    def total(self, key: object) -> int:
        return sum(self.counts[key].values())

    def share(self, key: object, what: str) -> float | None:
        total = self.total(key)
        return self.counts[key][what] / total if total else None


@dataclass
class Study:
    hands: int = 0
    seat_hands: int = 0
    tables: Counter = field(default_factory=Counter)
    stats_events: list[int] = field(default_factory=lambda: [0] * len(STATS))
    stats_chances: list[int] = field(default_factory=lambda: [0] * len(STATS))
    preflop: Tally = field(default_factory=Tally)  # (situation, tier)
    preflop_position: Tally = field(default_factory=Tally)  # (position group, tier group), first in
    open_sizes: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))  # tier -> bb
    open_all_in: Counter = field(default_factory=Counter)  # tier -> first-in raises that were all-in
    raise_first: Tally = field(default_factory=Tally)  # class -> raise or not, first in
    vpip: Tally = field(default_factory=Tally)  # class -> played or not
    postflop: Tally = field(default_factory=Tally)  # (street, facing, made)
    bet_sizes: dict[tuple[str, str], list[float]] = field(default_factory=lambda: defaultdict(list))
    river_bets: Tally = field(default_factory=Tally)  # "classe"/"contro le carte" -> ...
    bluffs: Tally = field(default_factory=Tally)  # (street, made) when checked to -> bet or not
    bluff_outcome: Counter = field(default_factory=Counter)  # river bets with nothing: worked or not
    slowplay: Tally = field(default_factory=Tally)
    strong_folds: Counter = field(default_factory=Counter)  # (street, made) -> folds
    results: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))  # tier -> bb
    results_position: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    # kind -> (hand, the seat the example is about, the street it did it on)
    examples: dict[str, list[tuple[HandHistory, int, Street]]] = field(default_factory=lambda: defaultdict(list))
    # dimension ("giocatori", "stack") -> group label -> the same study over that group's seats
    groups: dict[str, dict[str, Study]] = field(default_factory=dict)


def _situation(decision: Decision) -> str:
    if decision.raises_before == 0:
        return "nessun rilancio"
    return "contro un rilancio" if decision.raises_before == 1 else "contro 3-bet o piu'"


def _example(study: Study, kind: str, hand: HandHistory, decision: Decision, limit: int) -> None:
    kept = study.examples[kind]
    if len(kept) < limit and (not kept or kept[-1][0] is not hand):
        kept.append((hand, decision.seat, decision.street))


def effective_stack_bb(hand: HandHistory, seat: int) -> float:
    """What a seat could win or lose: its stack or the deepest other one, the smaller."""
    deepest = max(stack for other, stack in hand.starting_stacks.items() if other != seat)
    return min(hand.starting_stacks[seat], deepest) / hand.big_blind


# How the seats are split: a seat-hand belongs to one group of each dimension. The bands
# are the training's own (`style_log.SIZE_GROUPS`, `value_diagnostics.STACK_EDGES_BB`).
GROUPINGS = {
    "giocatori": lambda hand, seat: size_group(len(hand.starting_stacks)),
    "stack": lambda hand, seat: stack_label(effective_stack_bb(hand, seat)),
}
GROUP_ORDER = {"giocatori": ("2-3", "4-6", "7-9"), "stack": STACK_LABELS}
GROUP_UNIT = {"giocatori": "giocatori", "stack": "bb di stack effettivo"}


@dataclass
class _HandView:
    """What every study a hand is added to reads, worked out once."""

    hand: HandHistory
    dealt: list[int]
    names: dict[int, str]
    counts: dict[str, object]  # player id -> HandCounts
    decisions: list[Decision]
    standings: dict[int, str]  # index of a river bet or raise -> davanti, pari or dietro
    winner: int | None


def _view(hand: HandHistory, index: int) -> _HandView:
    dealt = sorted(hand.starting_stacks)
    decisions = decisions_of(hand, index)
    standings: dict[int, str] = {}
    folded: set[int] = set()
    for position, decision in enumerate(decisions):
        if decision.street == Street.RIVER and decision.kind == "raise":
            standings[position] = river_standing(hand, decision, [seat for seat in dealt if seat not in folded])
        if decision.kind == "fold":
            folded.add(decision.seat)
    counts = analyse_hand(
        hand.actions, dealt=dealt, button_seat=hand.button_seat,
        board_cards=len(hand.community_cards), player_ids={seat: f"s{seat}" for seat in dealt},
    )
    return _HandView(hand, dealt, positions(hand), counts, decisions, standings, last_standing(hand))


def _add_hand(study: Study, view: _HandView, seats: set[int], examples: int) -> None:
    """Add what `seats` did in one hand to `study` (every other seat is only the table)."""
    hand = view.hand
    study.hands += 1
    study.tables[len(view.dealt)] += 1
    study.seat_hands += len(seats)
    for seat in seats:
        seat_counts = view.counts[f"s{seat}"]
        for slot, (events, chances) in enumerate(zip(seat_counts.events, seat_counts.opportunities)):
            study.stats_events[slot] += events
            study.stats_chances[slot] += chances

    played: set[int] = set()
    acted: set[int] = set()  # had a preflop decision: a big blind's walk is not a hand it chose
    first_in_seen: set[int] = set()
    for position, decision in enumerate(view.decisions):
        if decision.seat not in seats:
            continue
        if decision.street == Street.PREFLOP:
            _preflop(study, hand, decision, first_in_seen, examples)
            acted.add(decision.seat)
            if decision.kind in ("call", "raise"):
                played.add(decision.seat)
        else:
            _postflop(study, hand, decision, view.standings.get(position), view.winner, examples)

    for seat in seats:
        cls = hand_class(hand.hole_cards[seat])
        if seat in acted:
            study.vpip.add(cls, "si" if seat in played else "no")
        won = (hand.final_stacks[seat] - hand.starting_stacks[seat]) / hand.big_blind
        study.results[TIER_OF[cls]].append(won)
        study.results_position[view.names[seat]].append(won)


def analyse(hands: Iterable[HandHistory], *, examples: int = 3, group_examples: int = 0) -> Study:
    """The whole study, and the same study per group of table sizes and of effective stack
    (`Study.groups`), in one pass: each seat-hand goes to the whole and to its two groups."""
    study = Study()
    study.groups = {dimension: {} for dimension in GROUPINGS}
    for index, hand in enumerate(hands):
        view = _view(hand, index)
        _add_hand(study, view, set(view.dealt), examples)
        for dimension, group_of in GROUPINGS.items():
            split: dict[str, set[int]] = defaultdict(set)
            for seat in view.dealt:
                split[group_of(hand, seat)].add(seat)
            for label, seats in split.items():
                group = study.groups[dimension].setdefault(label, Study())
                _add_hand(group, view, seats, group_examples)
    return study


def _preflop(study: Study, hand: HandHistory, decision: Decision, first_in_seen: set[int], examples: int) -> None:
    tier = decision.tier
    situation = _situation(decision)
    what = {"check": "limp/check", "call": "call" if decision.raises_before else "limp/check"}.get(
        decision.kind, decision.kind
    )
    study.preflop.add((situation, tier), what)
    first_in = decision.raises_before == 0 and decision.seat not in first_in_seen
    if decision.raises_before == 0:
        first_in_seen.add(decision.seat)
    if not first_in:
        # Not a short all-in: with no more chips than the bet it could only call.
        if (
            decision.raises_before >= 1 and tier == TIERS[0] and decision.kind == "call"
            and not decision.all_in and not decision.facing_all_in
        ):
            _example(study, "mano top 5% che chiama un rilancio invece di rilanciare", hand, decision, examples)
        return
    study.raise_first.add(decision.cls, "raise" if decision.kind == "raise" else "no")
    group = "forti (top 15%)" if tier in STRONG_TIERS else "scarse (ultimo 40%)" if tier in WEAK_TIERS else "medie"
    study.preflop_position.add((position_group(decision.position), group), decision.kind if decision.kind != "check" else "call")
    if decision.kind == "raise":
        study.open_sizes[tier].append(decision.amount / decision.big_blind)
        if decision.all_in:
            study.open_all_in[tier] += 1
    if tier == TIERS[0] and decision.kind in ("call", "check") and decision.position != "BB":
        _example(study, "limp con una mano top 5%", hand, decision, examples)
    if tier in WEAK_TIERS and decision.kind == "raise":
        _example(study, "rilancio con una mano dell'ultimo 40%", hand, decision, examples)


def _postflop(
    study: Study, hand: HandHistory, decision: Decision, standing: str | None, winner: int | None, examples: int
) -> None:
    street = decision.street.value
    made = decision.made
    facing = decision.to_call > 0
    what = decision.kind if not (decision.kind == "raise" and not facing) else "bet"
    study.postflop.add((street, "contro una puntata" if facing else "nessuna puntata", made), what)
    if what == "bet" and decision.pot_before > 0:
        study.bet_sizes[(street, made)].append(decision.amount / decision.pot_before)
    if not facing:
        study.bluffs.add((street, made), "bet" if what == "bet" else "check")
        if made in STRONG_MADE:
            study.slowplay.add(("check su mano forte", street), "check" if what == "check" else "bet")
            if what == "check" and made == "mostro":
                _example(study, "check con un mostro", hand, decision, examples)
    elif made in STRONG_MADE:
        # Against an all-in there is no raise left to make, so a call there is no slowplay.
        if not decision.facing_all_in:
            study.slowplay.add(("contro una puntata con mano forte", street), decision.kind)
        if decision.kind == "fold":
            study.strong_folds[(street, made)] += 1
            _example(study, "fold con mano forte", hand, decision, examples)
    if decision.street == Street.RIVER and decision.kind == "raise":
        study.river_bets.add("classe", made)
        study.river_bets.add("contro le carte", standing)
        if made == "nulla":
            study.bluff_outcome["riuscito" if winner == decision.seat else "chiamato"] += 1
            _example(study, "bluff al river (puntata con niente)", hand, decision, examples)


# ---- the report ------------------------------------------------------------------


def _pct(value: float | None, width: int = 5) -> str:
    return f"{100 * value:{width}.0f}%" if value is not None else " " * width + "-"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def format_summary(study: Study) -> list[str]:
    tables = ", ".join(f"{size}: {100 * n / study.hands:.0f}%" for size, n in sorted(study.tables.items()))
    lines = [f"mani giocate {study.hands:,} ({study.seat_hands:,} mani-posto), tavoli {tables}", ""]
    labels = {
        "vpip": "VPIP", "pfr": "PFR", "three_bet": "3-bet", "fold_to_three_bet": "fold al 3-bet",
        "steal": "steal", "aggression": "aggressivita' postflop", "cbet": "c-bet",
        "fold_to_cbet": "fold alla c-bet", "wtsd": "WTSD (va allo showdown)",
    }
    for slot, name in enumerate(STATS):
        chances = study.stats_chances[slot]
        rate = study.stats_events[slot] / chances if chances else None
        lines.append(f"  {labels[name]:<26} {_pct(rate)}   su {chances:,} occasioni")
    return lines


def format_preflop(study: Study, *, min_count: int) -> list[str]:
    lines = ["Fasce delle mani di partenza (formula di Chen, quota delle 1.326 combinazioni):"]
    for tier in TIERS:
        members = [cls for cls in ALL_CLASSES if TIER_OF[cls] == tier]
        shown = " ".join(members[:12]) + (" ..." if len(members) > 12 else "")
        lines.append(f"  {tier:<11} {shown}")
    columns = {
        "nessun rilancio": ("fold", "limp/check", "raise"),
        "contro un rilancio": ("fold", "call", "raise"),
        "contro 3-bet o piu'": ("fold", "call", "raise"),
    }
    for situation, kinds in columns.items():
        lines += ["", f"{situation}:", f"  {'fascia':<11} {'n':>8}  " + "  ".join(f"{k:>10}" for k in kinds)
                  + ("   rilancio mediano   all-in" if situation == "nessun rilancio" else "")]
        for tier in TIERS:
            key = (situation, tier)
            total = study.preflop.total(key)
            if total < min_count:
                lines.append(f"  {tier:<11} {total:>8}  (troppo poche)")
                continue
            row = f"  {tier:<11} {total:>8}  " + "  ".join(f"{_pct(study.preflop.share(key, k), 9)}" for k in kinds)
            if situation == "nessun rilancio":
                sizes = study.open_sizes[tier]
                median = _median(sizes)
                all_in = study.open_all_in[tier] / len(sizes) if sizes else None
                row += f"   {median:8.1f} bb" if median is not None else "          -"
                row += f"   {_pct(all_in)}"
            lines.append(row)
    lines += ["", "primo a entrare (nessun rilancio prima), per posizione: quota di rilanci",
              f"  {'posizione':<16}" + "".join(f"{g:>22}" for g in ("forti (top 15%)", "medie", "scarse (ultimo 40%)"))]
    for group in POSITION_GROUPS:
        cells = []
        for hands in ("forti (top 15%)", "medie", "scarse (ultimo 40%)"):
            key = (group, hands)
            total = study.preflop_position.total(key)
            cells.append(f"{_pct(study.preflop_position.share(key, 'raise'))} ({total:>6})" if total >= min_count else f"{'-':>14}")
        lines.append(f"  {group:<16}" + "".join(f"{cell:>22}" for cell in cells))
    return lines


def format_grid(tally: Tally, what: str, *, min_count: int) -> list[str]:
    """13x13: pairs on the diagonal, suited above it, offsuit below; '.' = too few hands."""
    header = "      " + "".join(f"{symbol:>4}" for symbol in reversed(RANK_SYMBOLS))
    lines = [header]
    for row in range(14, 1, -1):
        cells = []
        for col in range(14, 1, -1):
            if row == col:
                cls = rank_symbol(row) * 2
            elif row > col:
                cls = f"{rank_symbol(row)}{rank_symbol(col)}s"
            else:
                cls = f"{rank_symbol(col)}{rank_symbol(row)}o"
            total = tally.total(cls)
            share = tally.share(cls, what)
            cells.append(f"{100 * share:4.0f}" if total >= min_count and share is not None else "   .")
        lines.append(f"  {rank_symbol(row):>2}  " + "".join(cells))
    lines.append("  (sopra la diagonale suited, sotto offsuit; riga = carta piu' alta per le suited)")
    return lines


def format_postflop(study: Study, *, min_count: int) -> list[str]:
    lines = []
    for street in ("flop", "turn", "river"):
        lines += [f"{street}:", f"  {'mano':<12} {'nessuna puntata':>34}   {'contro una puntata':>36}",
                  f"  {'':<12} {'n':>7} {'check':>7} {'bet':>7} {'puntata':>10}   {'n':>7} {'fold':>7} {'call':>7} {'raise':>7}"]
        for made in MADE_CLASSES:
            open_key = (street, "nessuna puntata", made)
            facing_key = (street, "contro una puntata", made)
            n_open, n_facing = study.postflop.total(open_key), study.postflop.total(facing_key)
            left = f"{n_open:>7} {_pct(study.postflop.share(open_key, 'check'), 6)} {_pct(study.postflop.share(open_key, 'bet'), 6)}"
            size = _median(study.bet_sizes[(street, made)])
            left += f" {100 * size:8.0f}%p" if size is not None else f" {'-':>9}"
            if n_open < min_count:
                left = f"{n_open:>7} {'(poche)':>27}"
            right = f"{n_facing:>7} " + " ".join(_pct(study.postflop.share(facing_key, k), 6) for k in ("fold", "call", "raise"))
            if n_facing < min_count:
                right = f"{n_facing:>7} {'(poche)':>23}"
            lines.append(f"  {made:<12} {left}   {right}")
        lines.append("")
    lines.append("  puntata = puntata mediana in % del piatto. mano: cosa aggiungono le sue carte al board")
    lines.append("  (progetto = colore o scala aperta senza coppia; mostro = scala o meglio).")
    return lines


def format_bluffs(study: Study) -> list[str]:
    lines = ["con che cosa punta o rilancia al river:"]
    total = study.river_bets.total("classe")
    for made in MADE_CLASSES:
        share = study.river_bets.share("classe", made)
        if share:
            lines.append(f"  {made:<12} {_pct(share)}")
    lines.append(f"  (su {total:,} puntate/rilanci al river)")
    lines += ["", "le stesse puntate contro le carte di chi era ancora in mano:"]
    for standing in ("davanti", "pari", "dietro"):
        lines.append(f"  {standing:<12} {_pct(study.river_bets.share('contro le carte', standing))}")
    lines += ["", "quando nessuno ha puntato, quanto spesso punta:"]
    lines.append(f"  {'':<12} {'flop':>14} {'turn':>14} {'river':>14}")
    for made in ("nulla", "progetto", "coppia"):
        cells = []
        for street in ("flop", "turn", "river"):
            key = (street, made)
            total_key = study.bluffs.total(key)
            cells.append(f"{_pct(study.bluffs.share(key, 'bet'))} ({total_key:>5})" if total_key else f"{'-':>14}")
        lines.append(f"  {made:<12} " + " ".join(f"{cell:>14}" for cell in cells))
    tried = sum(study.bluff_outcome.values())
    if tried:
        worked = _pct(study.bluff_outcome["riuscito"] / tried)
        lines += ["", f"bluff al river con niente: {tried:,}, riusciti (tutti foldano) {worked}"]
    return lines


def format_slowplay(study: Study) -> list[str]:
    lines = ["mani forti (due coppie, tris, mostro) quando nessuno ha puntato: check invece di puntare"]
    for street in ("flop", "turn", "river"):
        key = ("check su mano forte", street)
        lines.append(f"  {street:<6} check {_pct(study.slowplay.share(key, 'check'))}  su {study.slowplay.total(key):,}")
    lines += ["", "mani forti contro una puntata (non all-in):"]
    lines.append(f"  {'':<6} {'fold':>6} {'call':>6} {'raise':>6}")
    for street in ("flop", "turn", "river"):
        key = ("contro una puntata con mano forte", street)
        lines.append(f"  {street:<6} " + " ".join(_pct(study.slowplay.share(key, k)) for k in ("fold", "call", "raise"))
                     + f"   su {study.slowplay.total(key):,}")
    if study.strong_folds:
        lines += ["", "fold con mano forte, per tipo:"]
        for (street, made), count in sorted(study.strong_folds.items(), key=lambda item: -item[1]):
            lines.append(f"  {street:<6} {made:<12} {count:,}")
    return lines


def format_results(study: Study) -> list[str]:
    lines = [f"  {'fascia':<11} {'mani':>9} {'bb/mano':>9}"]
    for tier in TIERS:
        values = study.results[tier]
        mean = _mean(values)
        lines.append(f"  {tier:<11} {len(values):>9,} {mean:>+9.2f}" if mean is not None else f"  {tier:<11} {0:>9}")
    lines += ["", f"  {'posizione':<11} {'mani':>9} {'bb/mano':>9}"]
    for name in POSITION_ORDER:
        values = study.results_position.get(name, [])
        if values:
            lines.append(f"  {name:<11} {len(values):>9,} {_mean(values):>+9.2f}")
    return lines


def format_hand(hand: HandHistory, hero: int | None = None, at: Street | None = None) -> list[str]:
    """One hand in a few lines: seats, cards, the actions street by street; `hero`, the seat
    the example is about, is starred."""
    names = positions(hand)
    bb = hand.big_blind
    seats = " ".join(
        f"{'*' if seat == hero else ''}{names[seat]}[{' '.join(str(c) for c in hand.hole_cards[seat])}] {hand.starting_stacks[seat] / bb:.0f}bb"
        for seat in sorted(hand.starting_stacks, key=lambda s: POSITION_ORDER.index(names[s]))
    )
    lines = [f"    {len(hand.starting_stacks)} giocatori: {seats}"]
    if hero is not None:
        lines[0] += f"   <- {names[hero]}" + (f" al {at.value}" if at is not None else "")
    by_street: dict[Street, list[str]] = defaultdict(list)
    for record in hand.actions:
        if record.action_type == ActionType.POST_BLIND:
            continue
        amount = f" {record.amount / bb:g}" if record.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN) else ""
        by_street[record.street].append(f"{names[record.seat]} {record.action_type.value}{amount}")
    for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
        if street in by_street:
            board = " ".join(str(c) for c in hand.community_cards[: _BOARD_AT[street]])
            lines.append(f"    {street.value:<7} {('[' + board + '] ') if board else ''}{', '.join(by_street[street])}")
    result = ", ".join(
        f"{names[seat]} {(hand.final_stacks[seat] - hand.starting_stacks[seat]) / bb:+g}"
        for seat in hand.starting_stacks
        if hand.final_stacks[seat] != hand.starting_stacks[seat]
    )
    lines.append(f"    risultato (bb): {result}")
    return lines


def _rate(events: int, chances: int, min_count: int) -> str:
    return _pct(events / chances) if chances >= min_count else f"{'-':>6}"


def _tally_rate(tally: Tally, key: object, what: str, min_count: int) -> str:
    return _rate(tally.counts[key][what], tally.total(key), min_count)


def _mean_bb(values: Sequence[float], min_count: int) -> str:
    return f"{sum(values) / len(values):+6.2f}" if len(values) >= min_count else f"{'-':>6}"


def comparison_rows(study: Study, *, min_count: int) -> list[tuple[str, str]]:
    """The study's main numbers, one per row, for a table that puts groups side by side."""
    rows: list[tuple[str, str]] = [("mani-posto", f"{study.seat_hands:>6,}")]
    names = {
        "vpip": "VPIP", "pfr": "PFR", "three_bet": "3-bet", "fold_to_three_bet": "fold al 3-bet",
        "steal": "steal", "aggression": "aggressivita' postflop", "cbet": "c-bet",
        "fold_to_cbet": "fold alla c-bet", "wtsd": "WTSD",
    }
    for slot, name in enumerate(STATS):
        rows.append((names[name], _rate(study.stats_events[slot], study.stats_chances[slot], min_count)))
    for tier in (TIERS[0], TIERS[2], TIERS[4]):
        rows.append((f"apre (nessun rilancio), {tier}", _tally_rate(study.preflop, ("nessun rilancio", tier), "raise", min_count)))
    for tier in (TIERS[0], TIERS[2]):
        rows.append((f"rilancia contro un rilancio, {tier}", _tally_rate(study.preflop, ("contro un rilancio", tier), "raise", min_count)))
    opens = [size for sizes in study.open_sizes.values() for size in sizes]
    median = _median(opens)
    rows.append(("rilancio d'apertura mediano", f"{median:5.1f}bb" if median is not None and len(opens) >= min_count else f"{'-':>6}"))
    rows.append(("aperture all-in", _rate(sum(study.open_all_in.values()), len(opens), min_count)))
    for street in ("flop", "river"):
        rows.append((f"{street}: punta con niente se nessuno punta", _tally_rate(study.bluffs, (street, "nulla"), "bet", min_count)))
    rows.append(("puntate al river con niente", _tally_rate(study.river_bets, "classe", "nulla", min_count)))
    rows.append(("puntate al river dietro alle carte", _tally_rate(study.river_bets, "contro le carte", "dietro", min_count)))
    tried = sum(study.bluff_outcome.values())
    rows.append(("bluff al river riusciti", _rate(study.bluff_outcome["riuscito"], tried, min_count)))
    checks = sum(study.slowplay.counts[("check su mano forte", street)]["check"] for street in ("flop", "turn", "river"))
    strong = sum(study.slowplay.total(("check su mano forte", street)) for street in ("flop", "turn", "river"))
    rows.append(("check con mano forte", _rate(checks, strong, min_count)))
    calls = sum(study.slowplay.counts[("contro una puntata con mano forte", street)]["call"] for street in ("flop", "turn", "river"))
    faced = sum(study.slowplay.total(("contro una puntata con mano forte", street)) for street in ("flop", "turn", "river"))
    rows.append(("solo call con mano forte contro puntata", _rate(calls, faced, min_count)))
    rows.append(("fold con mano forte", f"{sum(study.strong_folds.values()):>6}"))
    for tier in (TIERS[0], TIERS[4]):
        rows.append((f"bb/mano, {tier}", _mean_bb(study.results[tier], min_count)))
    every = [won for values in study.results.values() for won in values]
    rows.append(("bb/mano, tutte", _mean_bb(every, min_count)))
    return rows


def format_comparison(study: Study, dimension: str, *, min_count: int) -> list[str]:
    """The groups of one dimension side by side."""
    groups = study.groups.get(dimension, {})
    labels = [label for label in GROUP_ORDER[dimension] if label in groups]
    if not labels:
        return ["  (nessun dato)"]
    columns = [dict(comparison_rows(groups[label], min_count=min_count)) for label in labels]
    names = [name for name, _value in comparison_rows(groups[labels[0]], min_count=min_count)]
    width = max(len(name) for name in names)
    lines = [f"  {'':<{width}}" + "".join(f"{label:>10}" for label in labels) + f"   ({GROUP_UNIT[dimension]})"]
    for name in names:
        lines.append(f"  {name:<{width}}" + "".join(f"{column[name]:>10}" for column in columns))
    lines.append(f"  '-' = meno di {min_count} casi. Lo stack effettivo e' il proprio o quello del piu' profondo")
    lines.append("  degli altri, il minore; per stack i bb/mano non sommano a zero (corti contro profondi).")
    return lines


def format_report(study: Study, *, label: str, min_count: int, title: str | None = None) -> list[str]:
    def section(title: str) -> list[str]:
        return ["", f"== {title} " + "=" * max(0, 74 - len(title))]

    lines = [title or f"=== come gioca {label} (contro se stesso) ===", *format_summary(study)]
    for dimension, heading in (("giocatori", "PER NUMERO DI GIOCATORI"), ("stack", "PER STACK EFFETTIVO")):
        if study.groups.get(dimension):
            lines += section(heading) + format_comparison(study, dimension, min_count=min_count)
    lines += section("PREFLOP") + format_preflop(study, min_count=min_count)
    lines += section("PREFLOP: quota di rilanci da primo a entrare, mano per mano")
    lines += format_grid(study.raise_first, "raise", min_count=min_count)
    lines += section("PREFLOP: quota di mani giocate (VPIP), mano per mano")
    lines += format_grid(study.vpip, "si", min_count=min_count)
    lines += section("POSTFLOP, per cosa ha in mano") + format_postflop(study, min_count=min_count)
    lines += section("BLUFF") + format_bluffs(study)
    lines += section("MANI FORTI GIOCATE PIANO (slowplay) E FOLD DI MANI FORTI") + format_slowplay(study)
    lines += section("RISULTATI (bb a mano; contro se stesso sommano a zero)") + format_results(study)
    if study.examples:
        lines += section("ESEMPI")
        for kind, hands in study.examples.items():
            lines.append(f"  {kind}:")
            for hand, hero, street in hands:
                lines += format_hand(hand, hero, street)
                lines.append("")
    return lines


# ---- playing --------------------------------------------------------------------
# torch is imported lazily below, never at module import time.


@dataclass(frozen=True)
class PlayJob:
    """One worker's share of the hands, in picklable primitives (the model is loaded in
    the worker: `make_policy_fn` returns a closure, which does not cross a process)."""

    seed: int
    hands: int
    path: str
    weights: tuple[float, ...]
    stack_min_bb: float
    stack_max_bb: float
    small_blind: int
    big_blind: int
    session_hands: int
    device: str


def play_job(job: PlayJob) -> list[HandHistory]:
    """`job.hands` hands with the model in every seat, in sessions at drawn table sizes."""
    import torch

    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.table_mix import TableMix

    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    model, _checkpoint = build_model_from_checkpoint(job.path, device=job.device)
    policy = make_policy_fn(model, device=job.device)
    mix = TableMix(
        weights=job.weights, stack_min_bb=job.stack_min_bb, stack_max_bb=job.stack_max_bb,
        small_blind=job.small_blind, big_blind=job.big_blind,
    )
    rng = random.Random(job.seed)
    bank = TableBank(mix, rng)
    hands: list[HandHistory] = []
    while len(hands) < job.hands:
        size = mix.draw_size(rng)
        for proxy in bank.seats(size):
            proxy.inner = RLAgentPlayer(
                proxy.player_id, proxy.name, policy_fn=policy,
                big_blind=mix.big_blind, starting_stack=mix.starting_stack,
            )
        bank.play_session(size, min(job.session_hands, job.hands - len(hands)), on_hand=hands.append)
    return hands


def play(
    path: Path, *, hands: int, jobs: int, seed: int, weights: Sequence[float], stack_min_bb: float,
    stack_max_bb: float, small_blind: int, big_blind: int, session_hands: int, device: str,
) -> list[HandHistory]:
    workers = max(1, min(jobs, hands))
    rng = random.Random(seed)
    shares = [hands // workers + (1 if index < hands % workers else 0) for index in range(workers)]
    plan = [
        PlayJob(rng.randrange(2**31), share, str(path), tuple(weights), stack_min_bb, stack_max_bb,
                small_blind, big_blind, session_hands, device)
        for share in shares
    ]
    if workers == 1:
        return play_job(plan[0])
    played: list[HandHistory] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for part in executor.map(play_job, plan):
            played += part
    return played


def main(argv: Sequence[str] | None = None) -> int:
    from duel_power import resolve_model

    from pokerlab.rl.table_mix import (
        DEFAULT_BIG_BLIND,
        DEFAULT_SESSION_HANDS,
        DEFAULT_SMALL_BLIND,
        DEFAULT_STACK_MAX_BB,
        DEFAULT_STACK_MIN_BB,
        DEFAULT_TABLE_WEIGHTS,
        SIZES,
    )

    parser = argparse.ArgumentParser(description="come gioca un agente: studio in self-play")
    parser.add_argument("--model", default="#1", help="etichetta, percorso del checkpoint o #N (rango globale)")
    parser.add_argument("--hands", type=int, default=20000)
    parser.add_argument("--jobs", type=int, default=8, help="processi in parallelo, uno per core")
    parser.add_argument("--players", type=int, default=None, help="un solo numero di giocatori (default: la miscela del training)")
    parser.add_argument("--stack-bb", type=float, default=None, help="stessi stack per tutti (default: 1-100 bb a caso)")
    parser.add_argument("--sb", type=int, default=DEFAULT_SMALL_BLIND)
    parser.add_argument("--bb", type=int, default=DEFAULT_BIG_BLIND)
    parser.add_argument("--session-hands", type=int, default=DEFAULT_SESSION_HANDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=Path("checkpoints"), help="la cartella dei checkpoint (per cercare un modello per etichetta)")
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    parser.add_argument("--min-count", type=int, default=30, help="sotto questo campione una cella non e' mostrata")
    parser.add_argument("--examples", type=int, default=2, help="mani d'esempio per tipo")
    parser.add_argument("--by", choices=tuple(GROUPINGS), default=None,
                        help="aggiunge il report completo per ogni gruppo: di numero di giocatori o di stack effettivo")
    parser.add_argument("--save", type=Path, default=None, help="scrive le mani giocate (JSON lines)")
    parser.add_argument("--load", type=Path, default=None, help="rilegge mani salvate invece di giocare")
    parser.add_argument("--made-hands", action="store_true",
                        help="invece del self-play: riconosce scale (anche A2345) e colori? (made_hands.py)")
    parser.add_argument("--per-class", type=int, default=300, help="--made-hands: situazioni per classe e per strada")
    parser.add_argument("--runouts", type=int, default=100, help="--made-hands: avversari/board per la equity vera")
    parser.add_argument("--folds", type=int, default=2,
                        help="--made-hands: mani foldate da mostrare con la cronologia, per strada e per classe")
    parser.add_argument("--equity-model", type=Path, default=None,
                        help="--made-hands: la rete di equity (default: quella di config.toml)")
    args = parser.parse_args(argv)

    if args.made_hands:
        from made_hands import default_equity_model, study

        label, path, rating = resolve_model(args.model, global_dir=args.global_dir, root=args.root)
        if rating is not None:
            label = f"{label} (rating {rating:.0f})"
        equity_model = args.equity_model or default_equity_model()
        for line in study(path, equity_model, label=label, per_class=args.per_class, runouts=args.runouts,
                          jobs=args.jobs, seed=args.seed, min_count=args.min_count, folds=args.folds):
            print(line)
        return 0

    if args.load is not None:
        label = args.load.stem
        hands = list(HandHistoryReader(args.load).iter_hands())
    else:
        label, path, rating = resolve_model(args.model, global_dir=args.global_dir, root=args.root)
        if rating is not None:
            label = f"{label} (rating {rating:.0f})"
        weights = list(DEFAULT_TABLE_WEIGHTS)
        if args.players is not None:
            if args.players not in SIZES:
                raise SystemExit(f"--players deve essere tra {SIZES[0]} e {SIZES[-1]}")
            weights = [1.0 if size == args.players else 0.0 for size in SIZES]
        low, high = (args.stack_bb, args.stack_bb) if args.stack_bb else (DEFAULT_STACK_MIN_BB, DEFAULT_STACK_MAX_BB)
        print(f"gioco {args.hands:,} mani di {label} contro se stesso su {args.jobs} processi...", file=sys.stderr, flush=True)
        hands = play(
            path, hands=args.hands, jobs=args.jobs, seed=args.seed, weights=weights, stack_min_bb=low,
            stack_max_bb=high, small_blind=args.sb, big_blind=args.bb, session_hands=args.session_hands,
            device=args.device,
        )
        if args.save is not None:
            with HandHistoryWriter(args.save) as writer:
                for hand in hands:
                    writer.append(hand)
    study = analyse(hands, examples=args.examples, group_examples=args.examples if args.by else 0)
    for line in format_report(study, label=label, min_count=args.min_count):
        print(line)
    if args.by:
        groups = study.groups[args.by]
        for group in GROUP_ORDER[args.by]:
            if group in groups:
                title = f"\n\n=== {label}: solo {group} {GROUP_UNIT[args.by]} ==="
                for line in format_report(groups[group], label=label, min_count=args.min_count, title=title):
                    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
