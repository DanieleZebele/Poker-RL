"""The human studies: profiles and style fitting (`studies/agents/human_imitation.py`) and the
replay of real hands into model inputs (`studies/human_clone/clone_generate.py`)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokerlab.engine.stats import STATS, StatsTracker


def test_kmeans_separates_clear_clusters():
    from human_imitation import kmeans

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (30, 2))])
    labels = kmeans(x, 2, rng)
    assert len(set(labels[:50])) == 1 and len(set(labels[50:])) == 1 and labels[0] != labels[-1]


def test_the_regression_finds_the_rate_and_how_each_parameter_moves_it():
    from human_imitation import PARAMS, regress

    rng = np.random.default_rng(1)
    true_rate, slope = 0.3, 0.2  # the first statistic rises 0.2 per unit of the first parameter
    rows = []
    for _ in range(4000):
        delta = list(rng.normal(0, 0.25, PARAMS))
        chances = [100] * len(STATS)
        events = [int(rng.binomial(100, true_rate + slope * delta[0]))] + [50] * (len(STATS) - 1)
        rows.append({"profile": 0, "delta": delta, "events": events, "chances": chances})
    found = regress(rows, 1)[0]
    assert found["rate"][0] == pytest.approx(true_rate, abs=0.01)
    assert found["grad"][0][0] == pytest.approx(slope, abs=0.03)
    assert abs(found["grad"][0][1]) < 0.03
    assert found["rate"][1] == pytest.approx(0.5)


def test_a_step_moves_the_style_towards_the_target():
    from human_imitation import PARAMS, step

    grad = np.zeros((len(STATS), PARAMS))
    grad[0, 0] = 0.2
    measured = {"rate": [0.3] * len(STATS), "grad": grad.tolist()}
    target = np.array([0.4] + [0.3] * (len(STATS) - 1))
    new = step(np.zeros(PARAMS), measured, target, np.full(len(STATS), 0.1), ridge=1e-9, max_step=5.0, max_axis=1.5)
    assert new[0] == pytest.approx(0.5)  # 0.1 more rate at 0.2 a unit
    assert np.allclose(new[1:], 0.0)
    capped = step(np.zeros(PARAMS), measured, target, np.full(len(STATS), 0.1), ridge=1e-9, max_step=0.2, max_axis=1.5)
    assert capped[0] == pytest.approx(0.2)


def test_a_real_hand_is_replayed_into_one_decision_per_voluntary_action():
    from clone_generate import replay

    hand = {
        "schema_version": 1, "hand_id": "t:1", "started_at": 0.0, "num_players": 3, "small_blind": 50,
        "big_blind": 100, "button_seat": 2,
        "starting_stacks": {"0": 10000, "1": 10000, "2": 10000},
        "seat_names": {"0": "a", "1": "b", "2": "c"},
        "community_cards": ["2c", "7d", "9h", "Js", "Qd"],
        "actions": [
            {"street": "preflop", "seat": 0, "player_id": "a", "action_type": "post_blind", "amount": 50, "stack_before": 10000, "stack_after": 9950, "pot_before": 0},
            {"street": "preflop", "seat": 1, "player_id": "b", "action_type": "post_blind", "amount": 100, "stack_before": 10000, "stack_after": 9900, "pot_before": 50},
            {"street": "preflop", "seat": 2, "player_id": "c", "action_type": "raise", "amount": 300, "stack_before": 10000, "stack_after": 9700, "pot_before": 150},
            {"street": "preflop", "seat": 0, "player_id": "a", "action_type": "fold", "amount": 0, "stack_before": 9950, "stack_after": 9950, "pot_before": 450},
            {"street": "preflop", "seat": 1, "player_id": "b", "action_type": "call", "amount": 300, "stack_before": 9900, "stack_after": 9700, "pot_before": 450},
            {"street": "flop", "seat": 1, "player_id": "b", "action_type": "check", "amount": 0, "stack_before": 9700, "stack_after": 9700, "pot_before": 650},
            {"street": "flop", "seat": 2, "player_id": "c", "action_type": "bet", "amount": 400, "stack_before": 9700, "stack_after": 9300, "pot_before": 650},
            {"street": "flop", "seat": 1, "player_id": "b", "action_type": "fold", "amount": 0, "stack_before": 9700, "stack_after": 9700, "pot_before": 1050},
        ],
        "hole_cards": {"2": ["As", "Ah"]}, "known_by": {}, "payouts": {}, "final_stacks": {},
    }
    for record in hand["actions"]:
        record["timestamp"] = 0.0
    tracker = StatsTracker()
    order, decisions, shown, board = replay(json.loads(json.dumps(hand)), tracker)
    assert order == [2, 0, 1]  # the button first
    # sim seats: c 0, a 1, b 2 -- c raises, a folds, b calls, b checks, c bets, b folds
    assert [seat for seat, *_ in decisions] == [0, 1, 2, 2, 0, 2]
    assert all(index is not None and mask[index] for _seat, _f, mask, index, _street in decisions)
    assert [street for *_, street in decisions] == [0, 0, 0, 1, 1, 1]
    assert {seat: [c.__str__() for c in cards] for seat, cards in shown.items()} == {0: ["As", "Ah"]}
    assert [tracker.hands(name) for name in "abc"] == [1, 1, 1]  # the session's tracker saw the hand
