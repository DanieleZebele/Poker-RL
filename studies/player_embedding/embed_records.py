"""What an observer sees of one player in one hand, as numbers: the input of the player embedder.

A played hand (`HandHistory`) becomes, for every seat, a `SeatHand`:

- one **decision vector** per voluntary action the seat took (`DECISION_DIM` numbers): what it
  faced when it acted (`STATE_DIM`: street, position, the bet to call against the pot, the stack,
  the raises before it, how many were still in...) followed by what it did (`ACTION_DIM`: which of
  `ACTION_CLASSES` and, for a raise, its size against the pot after the call). Nothing in it is
  private: it is what anyone at the table saw;
- the **showdown it was seen to take**: nothing (`SHOW_NONE`), its cards were shown
  (`SHOW_SHOWED`, with their strength) or it lost and did not show them (`SHOW_MUCKED`), and
  whether it took the pot (`won`);
- what only the training knows and an observer never does, kept apart as targets: the class of
  each action and **the strength of the seat's own cards** at that decision (`strengths`).

**The muck is simulated here, not in the engine.** The engine reveals every player that reaches the
showdown. `showdown_view` keeps that for the winners (a pot is taken by showing) and lets each
loser show with the probability of its own (`p_show`, drawn per player and session by the
generator, from "never" to "always"). So an embedder trained on this reads a shown hand as a
*selected* sample, which is what a real table gives.

Pure Python, no torch and no numpy: the featurisation is what decides what the embedder can learn,
so it is tested in the ordinary suite (`tests/unit/test_study_player_embedding.py`). It reads a hand
through `agent_study.decisions_of`, the same reader the study of one model uses.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_study import Decision, decisions_of, made_class

from pokerlab.cards.card import Card
from pokerlab.engine.actions import ActionType
from pokerlab.engine.history import HandHistory
from pokerlab.engine.state import Street
from pokerlab.rl.hand_tiers import chen_score

STREET_ORDER = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
POSITIONS = ("UTG", "MP", "HJ", "CO", "BTN", "SB", "BB")
ACTION_CLASSES = ("fold", "check", "call", "raise_small", "raise_medium", "raise_large", "all_in")
NUM_ACTIONS = len(ACTION_CLASSES)

# street (4) + position (7) + nine numbers on the spot the player is in.
STATE_DIM = len(STREET_ORDER) + len(POSITIONS) + 9
# the action class one-hot and, for a raise, its size against the pot after the call.
ACTION_DIM = NUM_ACTIONS + 1
DECISION_DIM = STATE_DIM + ACTION_DIM

SHOW_NONE, SHOW_SHOWED, SHOW_MUCKED = 0, 1, 2
# the three states one-hot, "took the pot", and the strength of the cards shown.
SHOW_DIM = 5

# The most decisions kept for one player in one hand (a few streets of re-raises); the rare
# hand with more is cut, and `SeatHand.truncated` says how many were dropped.
MAX_DECISIONS = 8

# Ordered by how much the hand holds; `agent_study.made_class` names them.
MADE_ORDER = ("nulla", "progetto", "coppia", "top pair+", "due coppie", "tris", "mostro")


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def raise_fraction(decision: Decision) -> float:
    """A bet or raise against the pot as it stands after the player has called."""
    return (decision.amount - decision.level_before) / max(decision.pot_before + decision.to_call, 1)


def action_class(decision: Decision) -> int:
    """Index in `ACTION_CLASSES`: fold, check, call, or a raise by its size (all-in apart)."""
    if decision.kind in ("fold", "check", "call"):
        return ACTION_CLASSES.index(decision.kind)
    if decision.all_in:
        return ACTION_CLASSES.index("all_in")
    size = raise_fraction(decision)
    if size < 0.5:
        return ACTION_CLASSES.index("raise_small")
    return ACTION_CLASSES.index("raise_medium" if size <= 1.0 else "raise_large")


def state_features(decision: Decision) -> list[float]:
    """What the player faced, all in [0, 1]. No hole card is in it."""
    bb = max(decision.big_blind, 1)
    pot = max(decision.pot_before, 1)
    street = [1.0 if decision.street == s else 0.0 for s in STREET_ORDER]
    position = [1.0 if decision.position == name else 0.0 for name in POSITIONS]
    spot = [
        _clip01(decision.to_call / pot / 3.0),
        _clip01(math.log1p(decision.to_call / bb) / math.log1p(100.0)),
        _clip01(math.log1p(decision.pot_before / bb) / math.log1p(900.0)),
        _clip01(decision.stack_before / bb / 100.0),
        _clip01(decision.stack_before / pot / 10.0),
        _clip01(decision.raises_before / 4.0),
        decision.live / 9.0,
        decision.players / 9.0,
        1.0 if decision.facing_all_in else 0.0,
    ]
    return street + position + spot


def action_features(decision: Decision) -> list[float]:
    cls = action_class(decision)
    onehot = [1.0 if index == cls else 0.0 for index in range(NUM_ACTIONS)]
    raised = decision.kind == "raise"
    return onehot + [_clip01(raise_fraction(decision) / 3.0) if raised else 0.0]


def decision_vector(decision: Decision) -> list[float]:
    return state_features(decision) + action_features(decision)


def strength_of_cards(hole: tuple[Card, Card], board: tuple[Card, ...]) -> float:
    """How good the cards are at that moment, in [0, 1]: Bill Chen's points before the flop,
    what the hand holds after it (`MADE_ORDER`). A teacher for the embedder, never an input."""
    if not board:
        from pokerlab.rl.hand_tiers import hand_class

        return _clip01((chen_score(hand_class(hole)) + 1) / 21.0)
    return MADE_ORDER.index(made_class(hole, board)) / (len(MADE_ORDER) - 1)


def strength_of(decision: Decision) -> float:
    return strength_of_cards(decision.hole, decision.board)


@dataclass(frozen=True)
class SeatHand:
    """One seat's hand as an observer reads it, plus the targets the training adds."""

    vectors: list[list[float]]  # DECISION_DIM numbers each
    classes: list[int]  # the action class of each decision
    strengths: list[float]  # the strength of the seat's own cards at each decision (target only)
    show: int = SHOW_NONE
    won: int = 0
    shown_strength: float = 0.0
    truncated: int = 0
    seat: int = field(default=-1, compare=False)


def showdown_view(
    hand: HandHistory, p_show: Mapping[int, float], rng: random.Random
) -> dict[int, tuple[int, int, float]]:
    """For every dealt seat: `(show state, took the pot, strength shown)`.

    A showdown needs two players still in at the end. The winners show (a chopped pot too: a
    seat that gained nothing but lost nothing is a chop); each loser shows with its own
    `p_show[seat]` and otherwise mucks. A seat that folded, or any seat of a hand that ended
    with everyone else folding, shows nothing."""
    folded = {record.seat for record in hand.actions if record.action_type == ActionType.FOLD}
    dealt = list(hand.starting_stacks)
    contenders = [seat for seat in dealt if seat not in folded]
    showdown = len(contenders) >= 2
    board = tuple(hand.community_cards[:5])
    view: dict[int, tuple[int, int, float]] = {}
    for seat in dealt:
        net = hand.final_stacks.get(seat, 0) - hand.starting_stacks[seat]
        won = 1 if net > 0 else 0
        state, shown = SHOW_NONE, 0.0
        if showdown and seat in contenders:
            if net >= 0 or rng.random() < p_show.get(seat, 1.0):
                state = SHOW_SHOWED
                shown = strength_of_cards(hand.hole_cards[seat], board)
            else:
                state = SHOW_MUCKED
        view[seat] = (state, won, shown)
    return view


def seat_hands(
    hand: HandHistory, p_show: Mapping[int, float], rng: random.Random
) -> dict[int, SeatHand]:
    """Every dealt seat's `SeatHand` for this hand (a seat with no decision gets an empty one:
    the blinds it posted and the hand it sat out of are an observation too)."""
    decisions: dict[int, list[Decision]] = {seat: [] for seat in hand.starting_stacks}
    for decision in decisions_of(hand):
        decisions[decision.seat].append(decision)
    view = showdown_view(hand, p_show, rng)
    out: dict[int, SeatHand] = {}
    for seat, mine in decisions.items():
        kept = mine[:MAX_DECISIONS]
        show, won, shown = view[seat]
        out[seat] = SeatHand(
            vectors=[decision_vector(d) for d in kept],
            classes=[action_class(d) for d in kept],
            strengths=[strength_of(d) for d in kept],
            show=show,
            won=won,
            shown_strength=shown,
            truncated=len(mine) - len(kept),
            seat=seat,
        )
    return out
