from __future__ import annotations

import random

import pytest
from support import make_random_legal_bot

from pokerlab.engine.config import GameConfig
from pokerlab.players.rl_agent import PolicyDecision
from pokerlab.rl.rollout import Opponent, OpponentPool, SelfPlayCollector

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


def fixed_opponent(label: str) -> Opponent:
    """A stand-in fixed-pool `Opponent` for tests that only care about pool
    composition/sampling mechanics, not about what a seat actually plays."""
    return Opponent(label=label, factory=lambda pid, name, rng: make_random_legal_bot(pid, name, rng=rng))


def test_pool_without_members_falls_back_to_self_play():
    assert OpponentPool().sample(random.Random(0)) is None


def test_the_pool_holds_nothing_but_the_models_it_was_given():
    """No past self of the learner can ever be seated: the snapshot mechanism
    (`add_snapshot`, `max_snapshots`, `snapshot_share`) was removed at the
    user's request, so the pool is exactly the previously trained models drawn
    from the store and nothing else. A snapshot is a copy of the network being
    trained, so it drifts with it and anchors nothing -- which is what the fixed
    pool is there to do -- and every seat it took was a seat not facing an
    independently trained model."""
    pool = OpponentPool([fixed_opponent("Rock"), fixed_opponent("Shark")])
    assert len(pool) == 2
    assert {pool.sample(random.Random(seed)).label for seed in range(40)} == {"Rock", "Shark"}
    for attribute in ("add_snapshot", "snapshot"):
        assert not hasattr(pool, attribute)


def test_a_full_opponent_table_leaves_exactly_one_learner_seat():
    """With opponent_probability = 1.0 every other seat is a bot, so a hand can
    never yield more than one trajectory."""
    pool = OpponentPool([fixed_opponent("CallingStation")])
    trajectories = make_collector(pool, opponent_probability=1.0).collect(40)
    assert trajectories
    assert len(trajectories) <= 40


def test_pure_self_play_collects_from_several_seats_per_hand():
    trajectories = make_collector(pool=None).collect(40)
    assert len(trajectories) > 40  # more than one seat per hand contributes


def test_the_learner_seat_moves_around_the_table():
    pool = OpponentPool([fixed_opponent("Rock")])
    trajectories = make_collector(pool, opponent_probability=1.0, seed=9).collect(120)
    assert len({t.seat for t in trajectories}) > 1


def test_opponent_seats_never_contribute_training_data():
    """A bot's decisions must not reach the buffer -- PPO's ratio is only valid
    for actions its own policy produced."""
    seen: list[str] = []

    def spy_factory(player_id: str, name: str, rng: random.Random):
        seen.append(player_id)
        return make_random_legal_bot(player_id, name, rng=rng)

    pool = OpponentPool([Opponent(label="Spy", factory=spy_factory)])
    collector = make_collector(pool, opponent_probability=1.0, seed=3)
    trajectories = collector.collect(30)

    assert seen, "no opponent was ever seated"
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            assert decision.legal_mask[decision.action_index]
    # One learner per hand means at most one distinct seat per hand's worth of data.
    assert len(trajectories) <= 30


# ---- how much of the table is the learner itself ---------------------------


def test_the_opponent_probability_decides_how_many_seats_face_a_real_model():
    """The single most consequential number in the training field: at 6-max the
    default 0.5 leaves ~3.5 of 6 seats to copies of the learner. Every one of
    the remaining 2.5 now faces an externally trained model -- it used to be
    barely 1.5, because `snapshot_share` took 40% of them for the run's own
    frozen snapshots, and that mechanism is gone."""
    from pokerlab.rl.train import TrainConfig

    assert TrainConfig().opponent_probability == 0.5
    assert not hasattr(TrainConfig(), "snapshot_share")

    def external_seats(seats, opponent_probability):
        return (seats - 1) * opponent_probability

    assert external_seats(6, 0.5) == pytest.approx(2.5)
    assert external_seats(6, 1.0) == pytest.approx(5.0)


def test_a_probability_of_one_gives_every_other_seat_an_opponent():
    """With it at 1.0 the learner holds exactly one seat, so a hand yields far
    fewer of its own decisions -- the cost that has to be paid for facing a
    real field, and the reason an experiment comparing the two must equalise
    hands per iteration before it compares anything else."""
    pool = OpponentPool([Opponent(label="x", factory=lambda *a: None)])
    collector_rng = random.Random(0)
    seats = 6
    taken = sum(
        1
        for _ in range(2000)
        for seat in range(seats - 1)
        if collector_rng.random() < 1.0
    )
    assert taken == 2000 * (seats - 1), "every non-learner seat is claimed"
    assert pool.sample(random.Random(0)) is not None
