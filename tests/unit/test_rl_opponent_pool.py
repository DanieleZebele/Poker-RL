from __future__ import annotations

import random

from pokerlab.engine.config import GameConfig
from pokerlab.players.rl_agent import PolicyDecision
from pokerlab.rl.rollout import (
    Opponent,
    OpponentPool,
    SelfPlayCollector,
    policy_opponent,
    scripted_opponent,
)

BIG_BLIND = 2
STARTING_STACK = 200
CONFIG = GameConfig(
    num_players=6, starting_stack=STARTING_STACK, small_blind=1, big_blind=BIG_BLIND
)


def uniform_policy(rng: random.Random):
    def policy_fn(features: list[float], mask: list[bool]) -> PolicyDecision:
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    return policy_fn


def make_collector(pool: OpponentPool | None, opponent_probability: float = 1.0, seed: int = 5):
    return SelfPlayCollector(
        CONFIG,
        uniform_policy(random.Random(seed)),
        rng=random.Random(seed),
        opponent_pool=pool,
        opponent_probability=opponent_probability,
    )


def test_scripted_opponents_come_from_the_bot_catalog():
    opponent = scripted_opponent("shark")
    assert opponent.label == "Shark"
    player = opponent.factory("s1", "Shark", random.Random(0))
    assert player.player_id == "s1"


def test_pool_without_members_falls_back_to_self_play():
    assert OpponentPool().sample(random.Random(0)) is None


def test_snapshots_evict_the_oldest_beyond_the_cap():
    pool = OpponentPool(max_snapshots=2)
    for i in range(4):
        pool.add_snapshot(policy_opponent(f"snap{i}", uniform_policy(random.Random(i)), CONFIG))
    assert len(pool) == 2
    labels = {pool.sample(random.Random(seed)).label for seed in range(30)}
    assert labels <= {"snap2", "snap3"}


def test_a_full_opponent_table_leaves_exactly_one_learner_seat():
    """With opponent_probability = 1.0 every other seat is a bot, so a hand can
    never yield more than one trajectory."""
    pool = OpponentPool([scripted_opponent("calling_station")])
    trajectories = make_collector(pool, opponent_probability=1.0).collect(40)
    assert trajectories
    assert len(trajectories) <= 40


def test_pure_self_play_collects_from_several_seats_per_hand():
    trajectories = make_collector(pool=None).collect(40)
    assert len(trajectories) > 40  # more than one seat per hand contributes


def test_the_learner_seat_moves_around_the_table():
    pool = OpponentPool([scripted_opponent("rock")])
    trajectories = make_collector(pool, opponent_probability=1.0, seed=9).collect(120)
    assert len({t.seat for t in trajectories}) > 1


def test_opponent_seats_never_contribute_training_data():
    """A bot's decisions must not reach the buffer -- PPO's ratio is only valid
    for actions its own policy produced."""
    seen: list[str] = []

    def spy_factory(player_id: str, name: str, rng: random.Random):
        seen.append(player_id)
        return scripted_opponent("random").factory(player_id, name, rng)

    pool = OpponentPool([Opponent(label="Spy", factory=spy_factory)])
    collector = make_collector(pool, opponent_probability=1.0, seed=3)
    trajectories = collector.collect(30)

    assert seen, "no opponent was ever seated"
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            assert decision.legal_mask[decision.action_index]
    # One learner per hand means at most one distinct seat per hand's worth of data.
    assert len(trajectories) <= 30
