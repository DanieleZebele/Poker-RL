"""A drawn raise sized by every raise bin's probability (`action_space.weighted_raise_to`)."""

from __future__ import annotations

import random

import pytest
from support import fixed_mix, make_random_legal_bot

from pokerlab.engine.actions import ActionType
from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.base import Player
from pokerlab.players.rl_agent import PolicyDecision, RLAgentPlayer
from pokerlab.rl import action_space
from pokerlab.rl.action_space import (
    ACTION_DIM,
    ALL_IN_BIN,
    RAISE_BINS,
    _aggressive_legal,
    _raise_levels,
    action_index_to_action,
    legal_action_mask,
    weighted_raise_to,
)


def _decisions(count: int = 60, seed: int = 0) -> list:
    """Decisions where a raise is possible, from hands of random players."""
    kept = []

    class Keep(Player):
        def __init__(self, seat):
            super().__init__(f"p{seat}", f"p{seat}")
            self.bot = make_random_legal_bot(f"p{seat}", rng=random.Random(seed + seat))

        def act(self, observation, legal):
            if _aggressive_legal(legal) is not None:
                kept.append((observation, legal))
            return self.bot.act(observation, legal)

    table = Table(GameConfig(4, 10_000, 50, 100), [Keep(s) for s in range(4)], rng=random.Random(seed))
    while len(kept) < count:
        table.stacks = [10_000] * 4
        table.play_hand()
    return kept[:count]


def test_the_weighted_raise_is_the_probability_weighted_mean_of_the_open_raise_levels():
    rng = random.Random(1)
    for observation, legal in _decisions():
        mask = legal_action_mask(observation, legal)
        probabilities = [rng.random() for _ in range(ACTION_DIM)]
        levels = dict(_raise_levels(observation, _aggressive_legal(legal)))
        open_bins = [i for i in RAISE_BINS if mask[i]]
        if not open_bins:
            continue
        expected = sum(probabilities[i] * levels[i] for i in open_bins) / sum(probabilities[i] for i in open_bins)
        found = weighted_raise_to(probabilities, observation, legal, mask)
        assert found == round(expected)
        aggressive = _aggressive_legal(legal)
        assert aggressive.min_amount <= found <= aggressive.max_amount


def test_one_raise_bin_with_all_the_probability_raises_to_its_own_level():
    for observation, legal in _decisions(20):
        mask = legal_action_mask(observation, legal)
        for index in (i for i in RAISE_BINS if mask[i]):
            probabilities = [0.0] * ACTION_DIM
            probabilities[index] = 1.0
            weighted = action_index_to_action(index, observation, legal, weights=(probabilities, mask))
            assert weighted == action_index_to_action(index, observation, legal)


def test_the_all_in_and_the_passive_bins_are_never_reweighted():
    observation, legal = _decisions(1)[0]
    mask = legal_action_mask(observation, legal)
    weights = ([1.0 / ACTION_DIM] * ACTION_DIM, mask)
    for index in (0, 1, ALL_IN_BIN):
        if mask[index]:
            assert action_index_to_action(index, observation, legal, weights=weights) == \
                action_index_to_action(index, observation, legal)


def _fixed_policy(index: int, sizing: str):
    probabilities = [0.0] * ACTION_DIM
    for i in RAISE_BINS:
        probabilities[i] = 1.0 / len(RAISE_BINS)

    def policy(features, mask, *style):
        open_bins = [i for i in RAISE_BINS if mask[i]]
        chosen = index if mask[index] else (open_bins[0] if open_bins else 1)
        return PolicyDecision(action_index=chosen, probabilities=list(probabilities))

    policy.raise_sizing = sizing
    return policy


@pytest.mark.parametrize("sizing", ["bin", "weighted"])
def test_a_player_sizes_its_raise_as_its_policy_says(sizing):
    observation, legal = next((o, lg) for o, lg in _decisions(40) if legal_action_mask(o, lg)[RAISE_BINS[0]])
    mix = fixed_mix(4)
    player = RLAgentPlayer("x", "x", policy_fn=_fixed_policy(RAISE_BINS[0], sizing),
                           big_blind=mix.big_blind, starting_stack=mix.starting_stack)
    action = player.act(observation, legal)
    assert action.action_type in (ActionType.BET, ActionType.RAISE)
    mask = legal_action_mask(observation, legal)
    if sizing == "bin":
        assert action == action_index_to_action(RAISE_BINS[0], observation, legal)
    else:
        weights = ([1.0 / len(RAISE_BINS) if i in RAISE_BINS else 0.0 for i in range(ACTION_DIM)], mask)
        assert action.amount == weighted_raise_to(weights[0], observation, legal, mask)


def test_the_process_setting_refuses_what_it_does_not_know_and_defaults_to_bins():
    assert action_space.raise_sizing() == "bin"
    with pytest.raises(ValueError):
        action_space.set_raise_sizing("average")


def test_a_model_plays_by_the_process_setting_and_the_learner_always_by_bins(monkeypatch):
    pytest.importorskip("torch")
    from support import fake_collector, tiny_model

    from pokerlab.rl.policy import make_policy_fn

    model = tiny_model(hidden=8, num_layers=1)
    monkeypatch.setattr(action_space, "_RAISE_SIZING", "weighted")
    policy = make_policy_fn(model)
    assert policy.raise_sizing == "weighted"
    observation, legal = _decisions(1)[0]
    player = RLAgentPlayer("x", "x", policy_fn=policy, big_blind=100, starting_stack=10_000)
    features, mask = player.features_for(observation, legal)
    decision = policy(features, mask)
    assert decision.probabilities is not None and sum(decision.probabilities) == pytest.approx(1.0)
    assert make_policy_fn(model, raise_sizing="bin").raise_sizing == "bin"
    # the trainer builds its learner's policy with bins, whatever the process says
    from pokerlab.rl import train

    source = __import__("inspect").getsource(train.SelfPlayTrainer.__init__)
    assert 'make_policy_fn(self._play_model, device=device, raise_sizing="bin")' in source
    collector = fake_collector(fixed_mix(3), make_policy_fn(model, raise_sizing="bin"), rng=random.Random(0))
    assert all(p._raise_sizing == "bin" for p in collector._learners)
