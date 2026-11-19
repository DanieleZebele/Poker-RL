"""The critic as a classifier (HL-Gauss): bins uniform in symlog of the value, the loss the
cross-entropy, the value the softmax's mean; and how a checkpoint of one kind of critic is
resumed into a network of the other."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from support import fixed_mix, tiny_model

from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import OBS_DIM
from pokerlab.rl.policy import EQUITY_SLOTS, critic_centres, critic_edges, hl_gauss_targets, symlog
from pokerlab.rl.ppo import (
    PPOConfig,
    TrainingBatch,
    build_model_from_checkpoint,
    critic_output_keys,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from pokerlab.rl.train import build_parser, check_network_arguments, critic_support

CRITIC = {"bins": 51, "low": -1.05, "high": 8.05, "unit": 0.01}


def inputs(count: int = 64, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(count, OBS_DIM, generator=generator),
        torch.ones(count, ACTION_DIM, dtype=torch.bool),
        torch.rand(count, EQUITY_SLOTS, generator=generator),
    )


def batch(count: int = 64, seed: int = 0) -> TrainingBatch:
    features, masks, extra = inputs(count, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    return TrainingBatch(
        features=features,
        masks=masks,
        actions=torch.randint(0, ACTION_DIM, (count,), generator=generator),
        old_log_probs=torch.full((count,), -2.4),
        advantages=torch.randn(count, generator=generator),
        returns=torch.randn(count, generator=generator) * 0.3,
        critic_extra=extra,
    )


# ---- the support ----------------------------------------------------------------------------


def test_the_support_spans_a_lost_stack_to_every_other_stack_won_in_big_blinds():
    mix = fixed_mix(6, stack_bb=100.0, small_blind=50, big_blind=100)
    assert critic_support(mix, 0) is None
    support = critic_support(mix, 101)
    assert support == {"bins": 101, "low": -1.05, "high": pytest.approx(5.05), "unit": pytest.approx(0.01)}


def test_the_bins_are_uniform_in_symlog_and_their_centres_are_values():
    edges = critic_edges(CRITIC)
    assert len(edges) == CRITIC["bins"] + 1
    widths = edges[1:] - edges[:-1]
    assert torch.allclose(widths, widths[0].expand_as(widths), atol=1e-5)
    assert float(edges[0]) == pytest.approx(float(symlog(torch.tensor(-1.05), 0.01)), abs=1e-5)
    centres = critic_centres(CRITIC)
    assert torch.all(centres[1:] > centres[:-1])
    middle = (edges[1:] + edges[:-1]) / 2
    assert torch.allclose(symlog(centres, 0.01), middle, atol=1e-4)


def test_the_target_is_a_distribution_centred_on_the_return_and_clipped_to_the_support():
    edges = critic_edges(CRITIC)
    returns = torch.tensor([0.0, 0.05, -0.4, 2.0, 50.0, -50.0])
    target = hl_gauss_targets(symlog(returns, 0.01), edges)
    assert torch.allclose(target.sum(dim=-1), torch.ones(len(returns)))
    middle = (edges[1:] + edges[:-1]) / 2
    mean = target @ middle
    assert torch.allclose(mean[:4], symlog(returns[:4], 0.01), atol=0.02)
    assert int(target[4].argmax()) == CRITIC["bins"] - 1 and int(target[5].argmax()) == 0


# ---- the model ------------------------------------------------------------------------------


def test_the_value_is_the_mean_of_the_bins_and_the_policy_does_not_change():
    torch.manual_seed(0)
    scalar = tiny_model(hidden=16, num_layers=1)
    torch.manual_seed(0)
    classifier = tiny_model(hidden=16, num_layers=1, critic=CRITIC)
    features, masks, extra = inputs()
    assert classifier.value_head[-1].out_features == CRITIC["bins"]
    assert torch.equal(scalar.policy_logits(features, masks), classifier.policy_logits(features, masks))
    logits = classifier.value_head(classifier.value_trunk(torch.cat([features[:, 312:], extra], dim=-1)))
    expected = torch.softmax(logits, dim=-1) @ critic_centres(CRITIC)
    assert torch.allclose(classifier.critic_value(features, extra), expected)
    assert torch.equal(classifier(features, masks)[1], torch.zeros(len(features)))


def test_a_fresh_classifier_says_every_state_is_worth_about_zero():
    """Not the mean of uniform bins: with bins uniform in symlog that is about a deepest stack
    won. A broad bump at 0 starts it at about one big blind (0.01 in the value's unit)."""
    model = tiny_model(hidden=16, num_layers=1, critic={**CRITIC, "bins": 101})
    features, _, extra = inputs()
    assert model.critic_value(features, extra).abs().max() < 0.03


def test_a_fresh_classifier_learns_at_a_training_rate():
    """At the rate PPO trains it, not only at a large one: the start whose logits sat at -13.8
    (weights 0, the log of the narrow target of 0) explained nothing in this many steps."""
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1, critic={**CRITIC, "bins": 101})
    features, _, extra = inputs(256)
    returns = torch.where(extra[:, 0] > 0.5, torch.tensor(0.3), torch.tensor(-0.2))
    optimizer = torch.optim.Adam(model.critic_parameters(), lr=1e-3)
    for _ in range(600):   # the old start explained 0.05 here, this one 0.98
        _, losses = model.critic_terms(features, extra, returns)
        optimizer.zero_grad()
        losses.mean().backward()
        optimizer.step()
    with torch.no_grad():
        values = model.critic_value(features, extra)
    explained = 1 - float((returns - values).var() / returns.var())
    assert explained > 0.8


def test_the_classifier_learns_a_return_by_its_cross_entropy():
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1, critic=CRITIC)
    features, _, extra = inputs(32)
    returns = torch.where(extra[:, 0] > 0.5, torch.tensor(0.3), torch.tensor(-0.2))
    optimizer = torch.optim.Adam(model.critic_parameters(), lr=1e-2)
    for _ in range(300):
        _, losses = model.critic_terms(features, extra, returns)
        optimizer.zero_grad()
        losses.mean().backward()
        optimizer.step()
    with torch.no_grad():
        assert torch.allclose(model.critic_value(features, extra), returns, atol=0.03)


def test_a_ppo_update_trains_a_classifier():
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1, critic=CRITIC)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.value_head.parameters()]
    stats = ppo_update(model, optimizer, batch(), PPOConfig(epochs=2, minibatch_size=16))
    assert not any(math.isnan(value) for value in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.value_head.parameters()))


# ---- checkpoints ----------------------------------------------------------------------------


def test_a_classifier_checkpoint_rebuilds_with_its_support_and_values(tmp_path):
    model = tiny_model(hidden=16, num_layers=1, critic=CRITIC)
    save_checkpoint(tmp_path / "m.pt", model)
    rebuilt, checkpoint = build_model_from_checkpoint(tmp_path / "m.pt")
    assert rebuilt.critic == CRITIC == checkpoint["critic"]
    features, _, extra = inputs()
    assert torch.allclose(rebuilt.critic_value(features, extra), model.critic_value(features, extra))


def test_a_checkpoint_saved_before_the_classifier_is_a_scalar_critic(tmp_path):
    model = tiny_model(hidden=16, num_layers=1)
    save_checkpoint(tmp_path / "m.pt", model)
    saved = torch.load(tmp_path / "m.pt", weights_only=True)
    del saved["critic"]
    torch.save(saved, tmp_path / "old.pt")
    rebuilt, _ = build_model_from_checkpoint(tmp_path / "old.pt")
    assert rebuilt.critic is None
    loaded = load_checkpoint(tmp_path / "old.pt", tiny_model(hidden=16, num_layers=1))
    assert "critic_head_reset" not in loaded


@pytest.mark.parametrize(("parent", "child"), [(None, CRITIC), (CRITIC, None), (CRITIC, {**CRITIC, "bins": 21})])
def test_a_parent_with_another_critic_passes_on_everything_but_the_critics_output_layer(tmp_path, parent, child):
    torch.manual_seed(1)
    source = tiny_model(hidden=16, num_layers=1, head_layers=1, critic=parent)
    optimizer = torch.optim.Adam(source.parameters())
    save_checkpoint(tmp_path / "p.pt", source, optimizer)
    torch.manual_seed(2)
    target = tiny_model(hidden=16, num_layers=1, head_layers=1, critic=child)
    fresh = {key: value.clone() for key, value in target.state_dict().items()}
    target_optimizer = torch.optim.Adam(target.parameters())
    loaded = load_checkpoint(tmp_path / "p.pt", target, target_optimizer)
    assert loaded["critic_head_reset"] is True
    output = critic_output_keys(target)
    assert output == {"value_head.3.weight", "value_head.3.bias"}
    for key, value in target.state_dict().items():
        expected = fresh[key] if key in output else source.state_dict()[key]
        assert torch.equal(value, expected), key
    assert target_optimizer.state_dict()["state"] == {}
    assert target.critic == child


def test_a_single_bin_is_refused_naming_the_flag():
    args = build_parser().parse_args(["--config", "", "--critic-bins", "1"])
    with pytest.raises(ValueError, match="critic_bins"):
        check_network_arguments(args)


def test_a_classifier_saved_before_the_working_start_gets_its_output_layer_afresh(tmp_path):
    """A classifier checkpoint without `critic_output_init` began where it barely learns (the
    fleet published some): its child starts that layer afresh and keeps everything else; one
    saved since loads whole."""
    torch.manual_seed(1)
    source = tiny_model(hidden=16, num_layers=1, head_layers=1, critic=CRITIC)
    save_checkpoint(tmp_path / "now.pt", source)
    saved = torch.load(tmp_path / "now.pt", weights_only=True)
    assert saved["critic_output_init"] == 2
    del saved["critic_output_init"]
    torch.save(saved, tmp_path / "stuck.pt")

    whole = load_checkpoint(tmp_path / "now.pt", tiny_model(hidden=16, num_layers=1, head_layers=1, critic=CRITIC))
    assert "critic_head_reset" not in whole
    torch.manual_seed(2)
    target = tiny_model(hidden=16, num_layers=1, head_layers=1, critic=CRITIC)
    fresh = {key: value.clone() for key, value in target.state_dict().items()}
    assert load_checkpoint(tmp_path / "stuck.pt", target)["critic_head_reset"] is True
    output = critic_output_keys(target)
    for key, value in target.state_dict().items():
        assert torch.equal(value, fresh[key] if key in output else source.state_dict()[key]), key
