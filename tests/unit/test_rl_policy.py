from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from pokerlab.engine.config import GameConfig
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import OBS_DIM
from pokerlab.rl.policy import PokerActorCritic, make_policy_fn
from pokerlab.rl.rollout import SelfPlayCollector


@pytest.fixture
def model() -> PokerActorCritic:
    torch.manual_seed(0)
    return PokerActorCritic(hidden=64, num_layers=2)


def random_masks(batch: int, rng: random.Random) -> torch.Tensor:
    mask = torch.zeros(batch, ACTION_DIM, dtype=torch.bool)
    for row in range(batch):
        for index in rng.sample(range(ACTION_DIM), rng.randint(1, ACTION_DIM)):
            mask[row, index] = True
    return mask


def test_forward_returns_policy_logits_and_a_scalar_value(model):
    features = torch.randn(5, OBS_DIM)
    mask = torch.ones(5, ACTION_DIM, dtype=torch.bool)
    logits, values = model(features, mask)
    assert logits.shape == (5, ACTION_DIM)
    assert values.shape == (5,)


def test_masked_actions_get_exactly_zero_probability(model):
    rng = random.Random(3)
    features = torch.randn(16, OBS_DIM)
    mask = random_masks(16, rng)
    logits, _ = model(features, mask)
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.all(probabilities[~mask] == 0.0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(16))


def test_sampling_never_returns_a_masked_action(model):
    rng = random.Random(7)
    torch.manual_seed(7)
    features = torch.randn(64, OBS_DIM)
    mask = random_masks(64, rng)
    logits, _ = model(features, mask)
    distribution = torch.distributions.Categorical(logits=logits)
    for _ in range(50):
        sampled = distribution.sample()
        assert torch.all(mask.gather(1, sampled.unsqueeze(1)))


def test_a_single_legal_action_keeps_entropy_and_log_prob_finite(model):
    """Masking with -inf instead of a large finite value makes log_softmax
    return -inf and Categorical.entropy() NaN. One legal action is routine in
    poker, so this must hold."""
    features = torch.randn(4, OBS_DIM)
    mask = torch.zeros(4, ACTION_DIM, dtype=torch.bool)
    mask[:, 0] = True
    logits, _ = model(features, mask)
    distribution = torch.distributions.Categorical(logits=logits)

    assert torch.all(torch.isfinite(distribution.entropy()))
    assert torch.all(distribution.entropy() >= 0.0)
    sampled = distribution.sample()
    assert torch.all(sampled == 0)
    assert torch.all(torch.isfinite(distribution.log_prob(sampled)))


def test_gradients_flow_to_both_heads(model):
    features = torch.randn(8, OBS_DIM)
    mask = torch.ones(8, ACTION_DIM, dtype=torch.bool)
    logits, values = model(features, mask)
    (logits.sum() + values.sum()).backward()
    assert model.policy_head.weight.grad is not None
    assert model.value_head.weight.grad is not None
    assert torch.any(model.policy_head.weight.grad != 0.0)


def test_a_network_policy_plays_a_real_session_and_produces_trajectories(model):
    """The end-to-end bridge: torch tensors on one side, the plain-Python
    engine on the other, with no engine changes in between."""
    torch.manual_seed(1)
    config = GameConfig(num_players=4, starting_stack=200, small_blind=1, big_blind=2)
    collector = SelfPlayCollector(config, make_policy_fn(model), rng=random.Random(5))
    trajectories = collector.collect(15)

    assert trajectories
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            assert decision.legal_mask[decision.action_index]
            assert decision.log_prob <= 0.0
        assert len(trajectory.advantages) == len(trajectory.decisions)
