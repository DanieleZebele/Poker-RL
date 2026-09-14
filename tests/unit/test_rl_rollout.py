from __future__ import annotations

import random

import pytest

from pokerlab.engine.config import GameConfig
from pokerlab.players.rl_agent import DecisionRecord, PolicyDecision
from pokerlab.rl.rollout import HandTrajectory, SelfPlayCollector, compute_gae

BIG_BLIND = 2
STARTING_STACK = 200


def uniform_masked_policy(rng: random.Random):
    def policy_fn(features: list[float], mask: list[bool]) -> PolicyDecision:
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    return policy_fn


def make_trajectory(values: list[float], reward: float) -> HandTrajectory:
    decisions = [
        DecisionRecord(
            player_id="p0",
            seat=0,
            features=[],
            legal_mask=[],
            action_index=0,
            log_prob=0.0,
            value=value,
        )
        for value in values
    ]
    decisions[-1].reward = reward
    return HandTrajectory(seat=0, player_id="p0", decisions=decisions, reward=reward)


def make_collector(num_players: int = 4, seed: int = 11, rebuy: bool = True) -> SelfPlayCollector:
    config = GameConfig(
        num_players=num_players,
        starting_stack=STARTING_STACK,
        small_blind=1,
        big_blind=BIG_BLIND,
    )
    return SelfPlayCollector(
        config,
        uniform_masked_policy(random.Random(seed)),
        rng=random.Random(seed),
        rebuy=rebuy,
    )


def test_undiscounted_gae_returns_the_terminal_reward_at_every_step():
    """With gamma = lam = 1 the return is the realised chip delta, whatever the
    critic said -- the sanity check that the recursion is wired correctly."""
    trajectory = make_trajectory([1.0, 2.0, 3.0], reward=5.0)
    advantages, returns = compute_gae(trajectory, gamma=1.0, lam=1.0)
    assert returns == pytest.approx([5.0, 5.0, 5.0])
    assert advantages == pytest.approx([4.0, 3.0, 2.0])


def test_gae_with_a_perfect_critic_gives_zero_advantage():
    trajectory = make_trajectory([7.0, 7.0, 7.0], reward=7.0)
    advantages, _ = compute_gae(trajectory, gamma=1.0, lam=1.0)
    assert advantages == pytest.approx([0.0, 0.0, 0.0])


def test_lambda_discounts_advantage_towards_the_start_of_the_hand():
    trajectory = make_trajectory([0.0, 0.0, 0.0], reward=5.0)
    advantages, _ = compute_gae(trajectory, gamma=1.0, lam=0.95)
    assert advantages == pytest.approx([4.5125, 4.75, 5.0])


def test_collect_produces_one_trajectory_per_seat_that_acted():
    trajectories = make_collector().collect(20)
    assert trajectories
    for trajectory in trajectories:
        assert trajectory.decisions
        assert len(trajectory.advantages) == len(trajectory.decisions)
        assert len(trajectory.returns) == len(trajectory.decisions)
        # The reward is terminal: only the last decision of the hand carries it.
        assert [d.reward for d in trajectory.decisions[:-1]] == [0.0] * (
            len(trajectory.decisions) - 1
        )
        assert trajectory.decisions[-1].reward == trajectory.reward


def test_reward_scale_separates_the_training_signal_from_the_reported_big_blinds():
    """The critic predicts scaled rewards, but the trajectory keeps reporting
    big blinds -- otherwise value targets of +/-100 swamp the policy loss."""
    config = GameConfig(
        num_players=4, starting_stack=STARTING_STACK, small_blind=1, big_blind=BIG_BLIND
    )
    collector = SelfPlayCollector(
        config,
        uniform_masked_policy(random.Random(3)),
        rng=random.Random(3),
        reward_scale=0.01,
    )
    trajectories = [t for t in collector.collect(40) if t.reward != 0.0]

    assert trajectories, "no hand moved chips"
    for trajectory in trajectories:
        assert trajectory.decisions[-1].reward == pytest.approx(trajectory.reward * 0.01)
        assert abs(trajectory.decisions[-1].reward) < abs(trajectory.reward)


def test_rebuy_keeps_the_session_alive_while_busting_out_ends_it():
    with_rebuy = make_collector(num_players=3, seed=8, rebuy=True).collect(200)
    without_rebuy = make_collector(num_players=3, seed=8, rebuy=False).collect(200)
    assert len(with_rebuy) > len(without_rebuy)


def test_every_seat_gets_to_play_every_position_over_a_session():
    """The button rotates across rebuys, so training data is not skewed to one
    seat's position."""
    trajectories = make_collector(num_players=4, seed=2).collect(40)
    assert {t.seat for t in trajectories} == {0, 1, 2, 3}
