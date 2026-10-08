"""`studies/agents/made_hands.py`: the classes, the deals, the spots and the report are
torch-free; one test asks a tiny network."""

from __future__ import annotations

import random

import pytest
from made_hands import (
    BASE,
    ROWS,
    STREETS,
    TARGETS,
    TWIN,
    Deal,
    DealJob,
    ExampleRow,
    Reading,
    classify,
    folded_hands,
    format_report,
    history_lines,
    make_deals,
    planted,
    script,
    spot_of,
    true_equity,
    twin_hole,
)

from pokerlab.cards.card import Card
from pokerlab.engine.actions import ActionType
from pokerlab.engine.state import Street


def cards(text: str) -> list[Card]:
    return [Card.parse(card) for card in text.split()]


@pytest.mark.parametrize(
    ("hole", "board", "expected"),
    [
        ("As 2d", "5c 4d 3h Ks 9s", "scala A2345"),
        ("As 5d", "4c 3d 2h", "scala A2345"),
        ("6s 2c", "5c 4d 3h Ks 9s", "scala"),
        ("Ac Kd", "Th Jc Qd", "scala"),
        ("Ah Kh", "8h 4h 2h", "colore"),
        ("Qd 7d", "Ah 8h 4h Kc 2s", "niente"),
        ("Ad Kc", "5c 4d 3h Ks 9s", "coppia"),
        ("Kc 9d", "9h 8c 7d 2s Kh", "doppia coppia"),
        ("4s 4d", "4c 3d 2h", "tris"),
        ("2s 2d", "2c 2h 9d", "full o meglio"),
    ],
)
def test_the_class_of_a_hand_is_what_its_own_cards_add(hole, board, expected):
    assert classify(cards(hole), cards(board)) == expected


def test_a_board_that_plays_is_its_own_class_and_a_pair_the_board_made_is_nobodys():
    assert classify(cards("Jc Tc"), cards("5c 4d 3h 2s As")) == "scala del board"
    assert classify(cards("Kd 2d"), cards("Ah 8h 4h Kh 2h")) == "colore del board"
    # the board's own pair, the hand adding only a kicker to it: no class
    assert classify(cards("Ac 7d"), cards("9h 9c 5d 3s 2c")) is None
    assert classify(cards("Qc 5h"), cards("9h 9c 5d 3s 2c")) == "doppia coppia"


def test_planted_deals_hold_their_combination_and_never_repeat_a_card():
    rng = random.Random(3)
    for target in (*TARGETS, "scala del board", "colore del board"):
        for board_cards in (3, 4, 5):
            if target.endswith("del board") and board_cards != 5:
                continue
            kept = 0
            for _ in range(200):
                made = planted(rng, target, board_cards)
                if made is None:
                    continue
                hole, board = made
                assert len({*hole, *board}) == 2 + board_cards
                kept += classify(hole, board) == target
            assert kept > 0, (target, board_cards)


def test_a_twin_has_the_same_board_and_nothing():
    rng = random.Random(5)
    done = 0
    for _ in range(100):
        made = planted(rng, "scala", 5)
        if made is None or classify(*made) != "scala":
            continue
        hole, board = made
        other = twin_hole(rng, hole, board)
        if other is None:
            continue
        assert classify(other, board) == "niente"
        assert sum(a != b for a, b in zip(hole, other, strict=True)) == 1
        done += 1
    assert done > 20


def test_true_equity_is_sensible():
    rng = random.Random(0)
    nuts, _ = true_equity(rng, cards("As Ks"), cards("Qs Js Ts"), 200)
    air, _ = true_equity(rng, cards("2c 7d"), cards("Ah Kh Qh"), 200)
    assert nuts == 1.0  # a royal flush on the flop cannot lose
    assert air < 0.4


def test_the_deals_of_a_street_cover_every_class_with_their_twins_in_place():
    deals = make_deals(DealJob(seed=1, street="river", per_class=3, runouts=10))
    assert {d.cls for d in deals if not d.cls.startswith(TWIN)} == {*BASE, *TARGETS, "scala del board", "colore del board"}
    for index, deal in enumerate(deals):
        assert len(deal.board) == 5 and len({*deal.hole, *deal.board}) == 7
        if deal.twin_of is not None:
            original = deals[deal.twin_of]
            assert deal.cls == TWIN + original.cls and deal.board == original.board
            assert deal.twin_of < index
    flop = make_deals(DealJob(seed=1, street="flop", per_class=2, runouts=10))
    assert all(len(d.board) == 3 for d in flop) and not any(d.cls.endswith("del board") for d in flop)


@pytest.mark.parametrize("street", list(STREETS))
def test_the_spots_put_the_hero_on_the_street_and_in_the_seat_asked_for(street):
    hole, board = cards("As 2d"), cards("5c 4d 3h Ks 9s")[: STREETS[street]]
    leading, legal = spot_of(hole, board, street, facing=False)
    assert leading.street == Street(street) and leading.my_seat == 1
    assert {a.action_type for a in legal} >= {ActionType.CHECK, ActionType.BET}
    assert list(leading.hole_cards) == hole
    facing, legal = spot_of(hole, board, street, facing=True)
    assert facing.my_seat == 0 and facing.street == Street(street)
    assert {a.action_type for a in legal} >= {ActionType.FOLD, ActionType.CALL}
    assert len(script(street, True)) == len(script(street, False)) + 1


def test_the_report_holds_every_section():
    deals = []
    for street in STREETS:
        deals += make_deals(DealJob(seed=2, street=street, per_class=2, runouts=5))
    readings = [Reading(0.5, (0.2, 0.3, 0.5), (0.1, 0.2, 0.7)) for _ in deals]
    probe = {("flop", "scala"): (1.0, 40)}
    examples = [ExampleRow("flop", "4c 3d 2h", "As 5d", "scala A2345", 0.9, (0.0, 0.5, 0.5), (0.1, 0.1, 0.8))]
    text = "\n".join(format_report(deals, readings, probe, examples, label="bot", min_count=1))
    for title in ("FLOP", "TURN", "RIVER", "GEMELLI", "SONDA", "ESEMPI"):
        assert title in text
    assert all(cls in text for cls in ROWS if any(d.cls == cls for d in deals))


def test_a_tiny_network_is_asked_in_every_spot(tmp_path):
    pytest.importorskip("torch")
    from made_hands import _ask
    from support import tiny_model

    from pokerlab.rl.ppo import build_model_from_checkpoint, save_checkpoint

    path = tmp_path / "agent.pt"
    save_checkpoint(path, tiny_model())
    model, _ = build_model_from_checkpoint(path)
    deals = make_deals(DealJob(seed=4, street="turn", per_class=1, runouts=5))
    for facing in (False, True):
        rows = _ask(model, [spot_of(d.hole, d.board, d.street, facing) for d in deals])
        assert len(rows) == len(deals)
        assert all(sum(row) == pytest.approx(1.0, abs=1e-5) for row in rows)
    assert isinstance(deals[0], Deal)


def test_the_history_shows_every_action_the_hero_faces_in_big_blinds():
    board = cards("5c 4d 3h Ks 9s")
    observation, _legal = spot_of(cards("As 2d"), board, "river", facing=True)
    assert history_lines(observation, board) == [
        "      preflop *SB raise 2.5, BB call",
        "      flop    [5c 4d 3h] BB check, *SB check",
        "      turn    [5c 4d 3h Ks] BB check, *SB check",
        "      river   [5c 4d 3h Ks 9s] BB bet 3.75",
    ]


def test_folded_hands_are_those_whose_likeliest_answer_is_a_fold():
    deals = []
    for street in STREETS:
        deals += make_deals(DealJob(seed=6, street=street, per_class=4, runouts=5))
    spots = [spot_of(d.hole, d.board, d.street, True) for d in deals]
    # fold on the even deals, call on the others
    readings = [Reading(0.5, (0.3, 0.3, 0.4), (0.7, 0.2, 0.1) if i % 2 == 0 else (0.1, 0.8, 0.1)) for i in range(len(deals))]
    shown, counts = folded_hands(deals, readings, spots, per_group=2, seed=0)
    assert shown and all(hand.facing[0] == 0.7 for hand in shown)
    assert all(len(hand.history) >= 2 and hand.history[-1].endswith("BB bet 3.75") for hand in shown)
    for (street, cls), (folded, total) in counts.items():
        assert 0 <= folded <= total
        assert len([h for h in shown if (h.street, h.cls) == (street, cls)]) == min(2, folded)
    text = "\n".join(format_report(deals, readings, {}, [], label="bot", min_count=1, folds=shown, fold_counts=counts))
    assert "MANI FOLDATE" in text and "<- il modello" in text
