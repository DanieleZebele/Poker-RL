from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from support import fake_collector, fixed_mix, tiny_model

from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import OBS_DIM
from pokerlab.rl.policy import EQUITY_SLOTS, PokerActorCritic, make_policy_fn


@pytest.fixture
def model() -> PokerActorCritic:
    torch.manual_seed(0)
    return tiny_model(hidden=64, num_layers=2)


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
    logits, values = model(features, mask, torch.rand(8, EQUITY_SLOTS))
    (logits.sum() + values.sum()).backward()
    assert model.policy_head[-1].weight.grad is not None
    assert model.value_head[-1].weight.grad is not None
    assert torch.any(model.policy_head[-1].weight.grad != 0.0)


def test_a_network_policy_plays_a_real_session_and_produces_trajectories(model):
    """The end-to-end bridge: torch tensors on one side, the plain-Python
    engine on the other, with no engine changes in between."""
    torch.manual_seed(1)
    collector = fake_collector(fixed_mix(4, stack_bb=100), make_policy_fn(model), rng=random.Random(5))
    trajectories = collector.collect(15)

    assert trajectories
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            assert decision.legal_mask[decision.action_index]
            assert decision.log_prob <= 0.0
        assert len(trajectory.advantages) == len(trajectory.decisions)


def test_a_style_push_moves_the_choice_and_never_unmasks(model):
    """An opponent's style (`rl/styles.py`): a push on one bin makes it the choice, a masked
    bin stays out whatever the push, and a learner (no push) is untouched."""
    torch.manual_seed(0)
    policy = make_policy_fn(model)
    features = [0.0] * OBS_DIM
    mask = [True] * ACTION_DIM
    mask[3] = False
    bias = [0.0] * ACTION_DIM
    bias[1] = 50.0  # call, overwhelmingly
    bias[3] = 1e6  # masked: must stay impossible
    picks = {policy(features, mask, bias, 1.0).action_index for _ in range(50)}
    assert picks == {1}
    plain = {policy(features, mask).action_index for _ in range(200)}
    assert 3 not in plain and len(plain) > 1


def test_a_low_temperature_makes_a_style_mechanical(model):
    torch.manual_seed(1)
    policy = make_policy_fn(model)
    features = [0.0] * OBS_DIM
    mask = [True] * ACTION_DIM
    bias = [0.0] * ACTION_DIM
    bias[2] = 1.0
    cold = {policy(features, mask, bias, 0.01).action_index for _ in range(50)}
    assert len(cold) == 1


def test_the_distribution_is_what_the_policy_samples_from(model):
    """`policy_fn.distribution` is the softmax of the masked logits (with an opponent's style, if
    any): it sums to one, gives a masked bin exactly zero, and matches how often the policy
    really picks each bin."""
    policy = make_policy_fn(model)
    features = [0.0] * OBS_DIM
    mask = [True] * ACTION_DIM
    mask[3] = mask[5] = False
    probabilities = policy.distribution(features, mask)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
    assert probabilities[3] == probabilities[5] == 0.0

    torch.manual_seed(0)
    picks = [policy(features, mask).action_index for _ in range(3000)]
    for index, probability in enumerate(probabilities):
        assert picks.count(index) / 3000 == pytest.approx(probability, abs=0.04)

    bias = [0.0] * ACTION_DIM
    bias[1] = 5.0
    assert policy.distribution(features, mask, bias, 1.0)[1] > probabilities[1]
    flat = policy.distribution(features, mask, [0.0] * ACTION_DIM, 100.0)  # very hot: nearly uniform
    legal = [p for p, ok in zip(flat, mask) if ok]
    assert max(legal) - min(legal) < 0.01


def test_asking_for_the_distribution_does_not_move_the_random_stream(model):
    policy = make_policy_fn(model)
    features, mask = [0.0] * OBS_DIM, [True] * ACTION_DIM
    torch.manual_seed(3)
    expected = [policy(features, mask).action_index for _ in range(20)]
    torch.manual_seed(3)
    got = []
    for _ in range(20):
        policy.distribution(features, mask)
        got.append(policy(features, mask).action_index)
    assert got == expected
