"""The pure parts of studies/critic_loss/q_values.py: how an agent chooses from the Q's, and the
paired comparison."""

import math

import pytest
from q_values import agent_distribution, arms_of, paired

MASK = [False, True, True, True]
PROBS = [0.0, 0.7, 0.28, 0.02]
Q = [9.0, 0.1, 0.2, 0.5]


def test_greedy_takes_the_best_legal_q_whatever_the_policy_thinks():
    assert agent_distribution("greedy", Q, PROBS, MASK) == [0.0, 0.0, 0.0, 1.0]


def test_supported_ignores_actions_the_policy_hardly_plays():
    assert agent_distribution("supported", Q, PROBS, MASK, 0.05) == [0.0, 0.0, 1.0, 0.0]


def test_supported_falls_back_to_the_policys_likeliest_when_nothing_is_supported():
    assert agent_distribution("supported", Q, PROBS, MASK, 0.9) == [0.0, 1.0, 0.0, 0.0]


def test_improve_leaves_the_policy_alone_where_the_qs_are_equal():
    dist = agent_distribution("improve", [0.0, 0.3, 0.3, 0.3], PROBS, MASK, 0.01)
    assert dist == pytest.approx(PROBS)


def test_improve_moves_mass_toward_the_better_q_and_never_onto_an_illegal_action():
    dist = agent_distribution("improve", Q, PROBS, MASK, 0.1)
    assert dist[0] == 0.0
    assert sum(dist) == pytest.approx(1.0)
    assert dist[3] > PROBS[3] and dist[1] < PROBS[1]
    assert dist[3] / dist[2] == pytest.approx(PROBS[3] / PROBS[2] * math.exp(3.0))


def test_the_paired_difference_is_in_bb_per_100_over_the_tables_both_played():
    base = {0: [1.0, -1.0], 1: [0.0, 0.0], 2: [5.0, 5.0]}
    other = {0: [2.0, 0.0], 1: [1.0, 1.0]}
    mean, half = paired(base, other)
    assert mean == pytest.approx(100.0)
    assert half == pytest.approx(0.0)
    assert math.isnan(paired({}, {})[0])


def test_every_q_network_gets_the_same_agents_after_the_policy():
    arms = arms_of(["a", "b"])
    assert arms[0] == ("policy", "", "policy", 0.0)
    assert len(arms) == 1 + 2 * 4
    assert [a[2] for a in arms[1:5]] == ["greedy", "supported", "improve", "improve"]
