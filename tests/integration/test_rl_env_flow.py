import random

import pytest

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import DecisionRecord, PolicyDecision, RLAgentPlayer
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import OBS_DIM

BIG_BLIND = 2
STARTING_STACK = 500


def uniform_masked_policy(rng: random.Random):
    """Samples uniformly over whatever the mask allows -- the same contract a
    trained network has to satisfy, with none of its machinery."""

    def policy_fn(features: list[float], mask: list[bool]) -> PolicyDecision:
        assert len(features) == OBS_DIM
        assert len(mask) == ACTION_DIM
        allowed = [i for i, ok in enumerate(mask) if ok]
        return PolicyDecision(action_index=rng.choice(allowed))

    return policy_fn


def build_rl_table(num_players: int, seed: int, sink: list[DecisionRecord] | None = None) -> Table:
    rng = random.Random(seed)
    config = GameConfig(
        num_players=num_players,
        starting_stack=STARTING_STACK,
        small_blind=1,
        big_blind=BIG_BLIND,
    )
    players = [
        RLAgentPlayer(
            f"p{i}",
            f"RL{i}",
            policy_fn=uniform_masked_policy(random.Random(seed * 131 + i)),
            big_blind=BIG_BLIND,
            starting_stack=STARTING_STACK,
            on_decision=None if sink is None else sink.append,
        )
        for i in range(num_players)
    ]
    return Table(config, players, rng=rng)


@pytest.mark.parametrize("num_players", range(2, 10))
@pytest.mark.parametrize("seed", range(5))
def test_a_full_rl_table_never_produces_an_illegal_action(num_players, seed):
    """The action-mapping analogue of chip conservation: if any bin the mask
    leaves open maps to an action the engine rejects, this raises."""
    table = build_rl_table(num_players, seed)
    for _ in range(100):
        if sum(1 for s in table.stacks if s > 0) < 2:
            break
        total_before = sum(table.stacks)
        table.play_hand()
        assert sum(table.stacks) == total_before, "chip conservation invariant violated"


@pytest.mark.parametrize("num_players", range(2, 10))
def test_per_hand_rewards_are_zero_sum_in_big_blinds(num_players):
    table = build_rl_table(num_players, seed=7)
    any_hand_moved_chips = False
    for _ in range(50):
        if sum(1 for s in table.stacks if s > 0) < 2:
            break
        before = list(table.stacks)
        table.play_hand()
        rewards = [(after - start) / BIG_BLIND for start, after in zip(before, table.stacks)]
        # An evenly chopped pot legitimately leaves every stack untouched, so
        # this is a session-level check, not a per-hand one.
        any_hand_moved_chips |= any(r != 0.0 for r in rewards)
        assert sum(rewards) == pytest.approx(0.0)
    assert any_hand_moved_chips


def test_decisions_are_collected_with_features_and_mask():
    sink: list[DecisionRecord] = []
    table = build_rl_table(num_players=6, seed=3, sink=sink)
    table.play_session(20)

    assert sink, "the on_decision hook collected nothing"
    for record in sink:
        assert len(record.features) == OBS_DIM
        assert len(record.legal_mask) == ACTION_DIM
        assert record.legal_mask[record.action_index], "a masked bin was recorded as chosen"
        assert record.reward == 0.0  # filled in at hand end, not at decision time
