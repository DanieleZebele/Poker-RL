import random
import threading

import pytest

from pokerlab.engine.actions import Action, ActionType, IllegalActionError
from pokerlab.engine.config import GameConfig
from pokerlab.rl.env import TablePokerEnv
from pokerlab.rl.features import OBS_DIM

BIG_BLIND = 2
STARTING_STACK = 200


def make_env(num_players: int = 4, seed: int = 5) -> TablePokerEnv:
    config = GameConfig(
        num_players=num_players,
        starting_stack=STARTING_STACK,
        small_blind=1,
        big_blind=BIG_BLIND,
    )
    return TablePokerEnv(config, rng=random.Random(seed))


def play_one_hand(env: TablePokerEnv, rng: random.Random) -> dict[str, float]:
    observations = env.reset()
    while observations:
        agent_id, observation = next(iter(observations.items()))
        mask = env.legal_action_mask(agent_id)
        assert len(env.encode_observation(observation)) == OBS_DIM
        index = rng.choice([i for i, allowed in enumerate(mask) if allowed])
        observations, rewards, dones, _infos = env.step(
            {agent_id: env.action_from_index(agent_id, index)}
        )
    assert all(dones.values())
    return rewards


def test_reset_hands_control_to_exactly_one_agent():
    with make_env() as env:
        observations = env.reset(seed=1)
        assert len(observations) == 1
        agent_id, observation = next(iter(observations.items()))
        assert agent_id == f"p{observation.my_seat}"


def test_a_full_hand_ends_with_zero_sum_terminal_rewards():
    rng = random.Random(0)
    with make_env() as env:
        for _ in range(10):
            rewards = play_one_hand(env, rng)
            assert set(rewards) == {f"p{i}" for i in range(4)}
            assert sum(rewards.values()) == pytest.approx(0.0)


def test_seeding_reset_makes_the_hand_reproducible():
    with make_env() as first, make_env() as second:
        first_rewards = play_one_hand_seeded(first, seed=42)
        second_rewards = play_one_hand_seeded(second, seed=42)
        assert first_rewards == second_rewards


def play_one_hand_seeded(env: TablePokerEnv, seed: int) -> dict[str, float]:
    observations = env.reset(seed=seed)
    rng = random.Random(seed)
    while observations:
        agent_id = next(iter(observations))
        mask = env.legal_action_mask(agent_id)
        index = rng.choice([i for i, allowed in enumerate(mask) if allowed])
        observations, rewards, _dones, _infos = env.step(
            {agent_id: env.action_from_index(agent_id, index)}
        )
    return rewards


def test_repeated_resets_do_not_leak_worker_threads():
    """The worker parks inside Player.act(), so a reset that forgot to abort the
    previous hand would strand a thread per episode."""
    before = threading.active_count()
    with make_env() as env:
        for _ in range(15):
            env.reset(seed=3)  # abandons the previous hand mid-decision
        env.close()
        assert threading.active_count() == before
    assert threading.active_count() == before


def test_an_engine_error_in_the_worker_surfaces_to_the_caller():
    with make_env() as env:
        observations = env.reset(seed=2)
        agent_id = next(iter(observations))
        # Preflop facing the big blind, checking is not legal.
        with pytest.raises(IllegalActionError):
            env.step({agent_id: Action(ActionType.CHECK)})


def test_stepping_before_reset_is_rejected():
    env = make_env()
    with pytest.raises(RuntimeError, match="call reset"):
        env.step({"p0": Action(ActionType.FOLD)})


def test_only_the_agent_to_act_may_be_queried_or_stepped():
    with make_env() as env:
        observations = env.reset(seed=9)
        agent_id = next(iter(observations))
        other = next(a for a in (f"p{i}" for i in range(4)) if a != agent_id)
        with pytest.raises(KeyError):
            env.legal_action_mask(other)
        with pytest.raises(KeyError):
            env.step({other: Action(ActionType.FOLD)})
