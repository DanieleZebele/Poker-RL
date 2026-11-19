"""The pure parts of `studies/agents/board_danger.py`: board and hand classes, the exact equity."""

import random

import pytest
from board_danger import (
    DealJob,
    board_class,
    equity,
    flush_need,
    hand_class,
    make_deals,
    range_value,
    showdowns,
    straight_need,
)

from pokerlab.cards.card import Card


def cards(text: str) -> list[Card]:
    return [Card.parse(card) for card in text.split()]


@pytest.mark.parametrize(
    ("board", "need"),
    [("Ks Qh Jd Tc 2s", 1), ("Ks 8h 3d 2c 5s", 2), ("As 2h 3d 4c 9s", 1), ("Ks 8h 2d", 4), ("9s 8h 7d 6c 5s", 0)],
)
def test_straight_need_counts_the_cards_a_hand_must_add_the_ace_low_included(board, need):
    assert straight_need(cards(board)) == need


def test_flush_need_counts_the_cards_of_the_longest_suit():
    assert flush_need(cards("Ks 8s 3s 2s 5d")) == 1
    assert flush_need(cards("Ks 8h 3d 2c")) == 4


@pytest.mark.parametrize(
    ("board", "kind"),
    [
        ("Ks Qh Jd Tc 2s", "4 alla scala"),
        ("Ks 8h 3d 2c 5s", "3 alla scala"),
        ("Ks 8s 3s 2s 5d", "4 a colore"),
        ("Ks 8h 2d 2c", None),  # paired
        ("Ks Qs Js Ts 2d", None),  # two dangers at once
        ("9s 8h 7d 6c 5s", None),  # the board is the straight
        ("Ks 8h 2d", "secco"),
    ],
)
def test_a_board_gets_one_danger_at_a_time(board, kind):
    assert board_class(cards(board)) == kind


@pytest.mark.parametrize(
    ("hole", "board", "cls"),
    [
        ("Kc 7d", "Ks Qh Jd Tc 2s", "top pair"),
        ("Qc Qd", "Js 8h 3d 2c 5s", "overpair"),
        ("Ac 9d", "Ks Qh Jd Tc 2s", "scala"),
        ("Kc 8d", "Ks 8h 3d 2c 5s", "doppia coppia"),
        ("5c 5d", "5s 9h 2d Kc 7s", "tris"),
        ("Ac 2s", "Ks 8s 3s 4s 9d", "colore"),
        ("8c 7d", "Ks 8h 3d 2c", None),  # a pair, but not the top one
    ],
)
def test_the_hand_class_is_what_the_hand_makes_with_its_own_cards(hole, board, cls):
    assert hand_class(cards(hole), cards(board)) == cls


def test_the_river_equity_is_exact_and_falls_on_a_dangerous_board():
    rng = random.Random(0)
    wet = equity(rng, cards("Kc 7d"), cards("Ks Qh Jd Tc 2s"), 0)
    dry = equity(rng, cards("Kc 7d"), cards("Ks 8h 3d 2c 5s"), 0)
    assert dry > wet + 0.2
    assert equity(rng, cards("As Ah"), cards("Ad Ac Ks Kh 2c"), 0) == 1.0


def test_the_deals_fall_in_their_cells_and_never_overfill_one():
    deals = make_deals(DealJob(seed=1, street="turn", per_cell=1, runouts=10, max_draws=20_000))
    seen = [(d.board_kind, d.hand) for d in deals]
    assert len(seen) == len(set(seen))
    for deal in deals:
        assert board_class(deal.board) == deal.board_kind
        assert hand_class(deal.hole, deal.board) == deal.hand


def test_showdowns_enumerate_every_opponent_hand_on_the_river():
    hands, results = showdowns(cards("Kc 7d"), cards("Ks Qh Jd Tc 2s"))
    assert len(hands) == 990
    beaten = sum(1 for (a, b), r in zip(hands, results, strict=True) if r == 0.0 and "A" in (a.rank.symbol, b.rank.symbol))
    assert beaten > 0
    assert sum(results) / len(results) == equity(random.Random(0), cards("Kc 7d"), cards("Ks Qh Jd Tc 2s"), 0)


def test_a_bet_everyone_folds_to_wins_the_pot_and_one_only_called_by_better_loses_the_bet():
    everyone_folds = range_value([0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    assert everyone_folds.ev_gain == pytest.approx(1.0 - 0.5)
    only_better_call = range_value([0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], bet=0.75)
    assert only_better_call.called == 0.0
    # the worse hand folds (+1 pot, half the time), the better calls (-0.75): against a check worth 0.5
    assert only_better_call.ev_gain == pytest.approx(0.5 * 1.0 + 0.5 * -0.75 - 0.5)
