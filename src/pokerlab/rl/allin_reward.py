"""The training reward of a hand whose betting closed before the river: its expected result.

Once nobody can bet any more (everyone still in the hand is all-in, or only one of them
has chips left) the rest of the board is pure luck: no decision follows, so the
expectation of the result over the cards still to come is an unbiased stand-in for the
result itself, with the luck of the runout averaged out. Measured on 80,000 hands of the
gen-3 models (`config.toml` mixture, 1-100 bb stacks): 31% of hands close before the
river (23.5% preflop), and on the learner's preflop decisions those hands carry 96% of
the value target's variance; the expectation removes 74% of it (sd 19.2 -> 9.7 bb), 43%
on the flop and 26% on the turn. That luck is what a critic cannot predict and what kept
its explained variance on the preflop near zero.

Training only: `HandTrajectory.reward`, train/100 and every rated result stay on the
chips that actually moved.

The expectation is estimated over `runouts` boards drawn at random (a turn all-in is
enumerated exactly when its rivers are no more than that), every dealt hole card out of
the deck -- a folded hand's cards are out of the real deck too -- and settled by the
engine's own pot code, side pots and split pots included. Fewer runouts leave some luck
in: the target's variance is about `(the expectation's) + (the luck's) / runouts`, so on
the preflop 10 runouts remove ~68% of it, 20 ~71%, 50 ~73%. What it costs is the
evaluator, ~150 us a hand: measured on the same setup, collecting costs +5% CPU at 10,
+12% at 20 and +31% at 50.

Pure Python, no torch.
"""

from __future__ import annotations

import random
from itertools import combinations

from pokerlab.cards.card import Card, Rank, Suit
from pokerlab.engine.actions import ActionType
from pokerlab.engine.history import HandHistory
from pokerlab.engine.pots import compute_pots, distribute_pots
from pokerlab.engine.state import PlayerState, PlayerStatus, Street
from pokerlab.evaluator.evaluator import evaluate

DEFAULT_ALLIN_RUNOUTS = 20

_STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
_BOARD_AT = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4}
_DECK = tuple(Card(rank, suit) for suit in Suit for rank in Rank)


def closing_street(hand: HandHistory) -> Street | None:
    """The street on which betting closed for good, if that was before the river with
    two or more players still in the hand; None otherwise (a hand won by a fold, or one
    whose river was bet). The last action that was not a blind tells the street; a hand
    with none (every seat all-in on its blind) closed preflop."""
    folded = {a.seat for a in hand.actions if a.action_type == ActionType.FOLD}
    if len(hand.starting_stacks) - len(folded & hand.starting_stacks.keys()) < 2:
        return None
    if len(hand.community_cards) < 5:
        return None
    streets = [_STREETS.index(a.street) for a in hand.actions if a.action_type != ActionType.POST_BLIND]
    closed = _STREETS[max(streets, default=0)]
    return None if closed == Street.RIVER else closed


def expected_deltas(hand: HandHistory, rng: random.Random, runouts: int) -> dict[int, float] | None:
    """Every seat's expected chip delta over the cards still to come when the betting
    closed, or None when it did not close before the river (`closing_street`)."""
    closed = closing_street(hand)
    if closed is None or runouts < 1:
        return None
    folded = {a.seat for a in hand.actions if a.action_type == ActionType.FOLD}
    # What each seat put in, after the uncalled part was handed back: what it lost to the
    # pot, plus what the pot paid it.
    committed = {
        seat: hand.starting_stacks[seat] - hand.final_stacks[seat] + hand.payouts.get(seat, 0)
        for seat in hand.starting_stacks
    }
    seats = [
        PlayerState(
            seat=seat,
            player_id="",
            name="",
            stack=0,
            hole_cards=hand.hole_cards[seat],
            total_committed=committed[seat],
            status=PlayerStatus.FOLDED if seat in folded else PlayerStatus.ALL_IN,
        )
        for seat in hand.starting_stacks
    ]
    pots = compute_pots(seats)
    known = list(hand.community_cards[: _BOARD_AT[closed]])
    dead = {card for cards in hand.hole_cards.values() for card in cards} | set(known)
    deck = [card for card in _DECK if card not in dead]
    missing = 5 - len(known)

    boards: list[tuple[Card, ...]] = list(combinations(deck, missing)) if missing == 1 else []
    if not boards or len(boards) > runouts:
        boards = [tuple(rng.sample(deck, missing)) for _ in range(runouts)]

    totals = dict.fromkeys(hand.starting_stacks, 0.0)
    for drawn in boards:
        board = known + list(drawn)
        ranks: dict[tuple[Card, ...], object] = {}

        def rank(cards, ranks=ranks):
            # A seat in several side pots is ranked once per board.
            key = tuple(cards)
            if key not in ranks:
                ranks[key] = evaluate(cards)
            return ranks[key]

        # `distribute_pots` adds the winnings to these throwaway states' stacks; only
        # the payouts it returns are read.
        payouts = distribute_pots(pots, seats, board, hand.button_seat, evaluate_fn=rank)
        for seat in totals:
            totals[seat] += payouts.get(seat, 0) - committed[seat]
    return {seat: total / len(boards) for seat, total in totals.items()}
