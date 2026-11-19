"""What anyone at the table saw of a hand, as numbers: the input of the belief network.

The belief network estimates, for one player (the *target*), the range of hole cards it holds at a
moment of the hand, from public information only. This module turns a played hand
(`HandHistory`) into that information:

- one **token** per action record, blinds included (`TOKEN_DIM` numbers): the street, the actor's
  position, what it did (`ACTION_CLASSES`) and the numbers of the spot it did it in (the bet it
  faced over the pot, its raise over the pot, its stack). Nothing private: the actor's seat is kept
  apart (`HandTokens.actors`) so the network can mark which tokens are the target's own;
- the **opponent statistics** every seat had at the start of the hand (`hud_before_hands`): a
  session replayed through a fresh `StatsTracker`, exactly as the table that played it fed its
  own, so the vectors are the ones the models read while playing;
- the **samples** (`samples_of`): at every voluntary action, for every *other* player still in the
  hand, one question -- "what does this player hold, given everything before this action?". The
  player about to act is the *observer*: its cards are what the network may rule out (card
  removal), never what it reads.

The tokens, the positions and the combos are `src/pokerlab/rl/belief_inputs.py`, imported here: a
model of feature version 5 is fed them at the table, so the study and play cannot drift apart.

Cards are indices 0..51 as everywhere in the project (`card_index`: `suit * 13 + rank - 2`, suits
c d h s, the index of `rl/features.py` and `rl/equity_net.py`) and a pair of cards one of the 1326
**combos** (`combo_index`, lower card first). Pure Python, no torch and no numpy: tested in
the ordinary suite (`tests/unit/test_study_belief.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pokerlab.cards.card import Card, Rank, Suit
from pokerlab.engine.history import HandHistory
from pokerlab.engine.stats import USED_SLOTS, StatsTracker
from pokerlab.rl.belief_inputs import (
    ACTION_CLASSES,
    AMOUNT_SCALE,
    BOARD_AT,
    COMBO_CARDS,
    MAX_SEATS,
    NUM_CARDS,
    NUM_COMBOS,
    POSITIONS,
    STACK_SCALE,
    STREETS,
    TOKEN_DIM,
    HandTokens,
    action_tokens,
    combo_index,
)
from pokerlab.rl.belief_inputs import positions as seat_positions
from pokerlab.rl.features import card_index

__all__ = [
    "ACTION_CLASSES", "AMOUNT_SCALE", "BOARD_AT", "COMBO_CARDS", "HUD_DIM", "MAX_SEATS", "NUM_CARDS",
    "NUM_COMBOS", "POSITIONS", "STACK_SCALE", "STREETS", "SUITS", "TOKEN_DIM", "HandTokens", "Sample",
    "blocked_combos", "card_index", "card_of", "combo_index", "hand_tokens", "hud_before_hands",
    "positions", "samples_of",
]

SUITS = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
HUD_DIM = USED_SLOTS


def card_of(index: int) -> Card:
    return Card(Rank(index % 13 + 2), SUITS[index // 13])


def blocked_combos(cards: Iterable[int]) -> list[int]:
    """Every combo holding one of `cards`: what a player who sees them rules out."""
    seen = set(cards)
    return [i for i, (a, b) in enumerate(COMBO_CARDS) if a in seen or b in seen]


def positions(hand: HandHistory) -> dict[int, str]:
    """Every dealt seat's position (`belief_inputs.positions`)."""
    return seat_positions(list(hand.starting_stacks), hand.button_seat)


def hand_tokens(hand: HandHistory) -> HandTokens:
    """Every action record of `hand` as a token: `belief_inputs.action_tokens`, which is also what a
    model of feature version 5 reads at the table."""
    return action_tokens(hand.actions, list(hand.starting_stacks), hand.button_seat, hand.big_blind)


@dataclass(frozen=True)
class Sample:
    """One question: what does `target` hold just before token `prefix` of the hand?"""

    prefix: int  # the tokens before it are what is known
    target: int  # seat
    observer: int  # seat about to act (its cards are ruled out, never read)
    street: int  # index in STREETS
    live: int  # players still in the hand


def samples_of(tokens: HandTokens) -> list[Sample]:
    """At every voluntary action, one sample per other player still in the hand."""
    found = []
    for t, voluntary in enumerate(tokens.voluntary):
        if not voluntary:
            continue
        observer = tokens.actors[t]
        live = tokens.live_before[t]
        for target in sorted(live):
            if target != observer:
                found.append(Sample(t, target, observer, tokens.streets[t], len(live)))
    return found


def hud_before_hands(hands: Sequence[HandHistory]) -> list[dict[int, tuple[float, ...]]]:
    """For a session's hands in order, every seat's statistics vector at the start of each hand.

    The session is replayed through a fresh `StatsTracker` the way `Table` feeds one: the seats
    keep their players for the whole session, so a seat is its player's identity. A seat not yet
    seen has no entry (it reads as zeros)."""
    tracker = StatsTracker()
    found = []
    for hand in hands:
        ids = {seat: f"s{seat}" for seat in hand.starting_stacks}
        found.append(tracker.vectors(ids))
        tracker.record_hand(
            hand.actions,
            dealt=sorted(hand.starting_stacks),
            button_seat=hand.button_seat,
            board_cards=len(hand.community_cards),
            player_ids=ids,
        )
    return found
