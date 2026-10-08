"""`studies/agents/archetype_study.py`: the push on the logits and the paired difference are
plain arithmetic; one test plays both arms with a tiny network and checks they are paired."""

from __future__ import annotations

import pytest
from archetype_study import (
    ARCHETYPE_BIASES,
    EXPECTED,
    ArchetypeJob,
    bias_vector,
    paired_difference,
    situation,
)
from hud_study import BEHAVIOURS
from test_study_hud_study import played_states

from pokerlab.engine.state import Street
from pokerlab.rl.action_space import ACTION_DIM, ALL_IN_BIN, CHECK_CALL_BIN, FOLD_BIN


def test_every_archetype_pushes_every_situation():
    for pushes in ARCHETYPE_BIASES.values():
        assert set(pushes) == {"preflop", "checked", "facing"}


def test_the_push_lands_on_the_bins_of_its_group():
    observation = next(o for o, _legal in played_states(40) if o.street == Street.PREFLOP)
    nit = bias_vector("nit", observation, 2.0)
    assert len(nit) == ACTION_DIM
    assert nit[FOLD_BIN] == 2.0 and nit[CHECK_CALL_BIN] == 0.0 and nit[ALL_IN_BIN] == -2.0
    station = bias_vector("calling station", observation, 1.0)
    assert station[CHECK_CALL_BIN] == 1.0 and station[FOLD_BIN] == -1.0


def test_the_situation_is_read_from_the_bet_to_match():
    for observation, _legal in played_states(80):
        found = situation(observation)
        if observation.street == Street.PREFLOP:
            assert found == "preflop"
        else:
            facing = observation.current_bet_to_match > observation.my_current_bet
            assert found == ("facing" if facing else "checked")


def test_the_expected_directions_name_real_behaviours():
    labels = {label for label, _picks, _groups in BEHAVIOURS}
    for directions in EXPECTED.values():
        assert set(directions) <= labels
        assert set(directions.values()) <= {"+", "-"}


def test_the_paired_difference_is_in_bb_per_100_with_its_interval():
    mean, half = paired_difference([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0])
    assert mean == pytest.approx(100.0)
    assert half == pytest.approx(0.0)
    _mean, half = paired_difference([1.0, -1.0] * 50, [0.0] * 100)
    assert half > 0


def test_both_arms_play_the_same_hands_and_only_the_real_one_keeps_states(tmp_path):
    pytest.importorskip("torch")
    from archetype_study import run_job
    from support import tiny_model

    from pokerlab.rl.ppo import save_checkpoint

    path = tmp_path / "agent.pt"
    save_checkpoint(path, tiny_model())
    common = {
        "seed": 3, "hands": 30, "hero_path": str(path), "opponent_path": str(path), "archetype": "maniaco",
        "strength": 2.0, "population_hud": (0.0,) * 20, "weights": (0, 1, 0, 0, 0, 0, 0, 0),
        "stack_min_bb": 20.0, "stack_max_bb": 50.0, "small_blind": 50, "big_blind": 100, "session_hands": 10,
    }
    real = run_job(ArchetypeJob(real_hud=True, keep=20, **common))
    blind = run_job(ArchetypeJob(real_hud=False, keep=0, **common))
    assert len(real.hero_bb) == len(blind.hero_bb) == 30
    assert 0 < len(real.states) <= 20 and blind.states == []
    assert sum(real.opponent_chances) > 0
    # Seeded throughout: the same arm played again is the same hands, chip for chip.
    assert run_job(ArchetypeJob(real_hud=False, keep=0, **common)).hero_bb == blind.hero_bb
