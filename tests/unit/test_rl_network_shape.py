from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from support import TINY_EQUITY, tiny_model
from torch import nn

from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import CARDS_DIM, OBS_DIM
from pokerlab.rl.policy import EQUITY_SLOTS, MASK_FILL, SHAPE_KEYS, PokerActorCritic
from pokerlab.rl.ppo import (
    IncompatibleCheckpointError,
    build_model_from_checkpoint,
    check_compatible,
    checkpoint_shape,
    save_checkpoint,
)
from pokerlab.rl.train import parent_shape_mismatch


def build(**shape) -> PokerActorCritic:
    torch.manual_seed(0)
    return tiny_model(**{"hidden": 24, "num_layers": 2, "head_hidden": 12, "head_layers": 2, **shape})


def parameters(hidden: int, layers: int, head_hidden: int, head_layers: int) -> int:
    """Counted by hand: the equity encoder, then two trunks (the policy's reads the encoder's
    output, the critic's the equities) and two heads; a block is a Linear, then a LayerNorm."""

    def block(i: int, o: int) -> int:
        return i * o + o + 2 * o

    def trunk(inputs: int) -> int:
        return block(inputs, hidden) + (layers - 1) * block(hidden, hidden)

    latent, inner = TINY_EQUITY["latent"], TINY_EQUITY["hidden"]
    encoder = (104 * latent + latent) + TINY_EQUITY["blocks"] * (
        2 * latent + (latent * inner + inner) + (inner * latent + latent)
    )
    rest = OBS_DIM - CARDS_DIM
    total = encoder + trunk(rest + latent) + trunk(rest + EQUITY_SLOTS)

    def head(out: int) -> int:
        n, width = 0, hidden
        for _ in range(head_layers):
            n += block(width, head_hidden)
            width = head_hidden
        return n + width * out + out

    return total + head(ACTION_DIM) + head(1)


@pytest.mark.parametrize("head_layers", [0, 1, 2, 3])
def test_a_head_is_its_hidden_blocks_then_the_output_layer(head_layers):
    model = build(head_layers=head_layers)
    for head, out in ((model.policy_head, ACTION_DIM), (model.value_head, 1)):
        linears = [m for m in head if isinstance(m, nn.Linear)]
        assert len(linears) == head_layers + 1
        assert linears[-1].out_features == out
        assert len([m for m in head if isinstance(m, nn.LayerNorm)]) == head_layers
        if head_layers:
            assert linears[0].in_features == 24 and linears[0].out_features == 12
    assert sum(p.numel() for p in model.parameters()) == parameters(24, 2, 12, head_layers)


def test_the_default_network_has_plain_heads():
    model = tiny_model()
    assert model.shape == {"hidden": 512, "num_layers": 3, "head_hidden": 256, "head_layers": 0}
    assert sum(p.numel() for p in model.parameters()) == parameters(512, 3, 256, 0)


def test_the_forward_pass_keeps_its_shapes_and_masks_whatever_the_heads():
    model = build()
    features = torch.randn(5, OBS_DIM)
    mask = torch.zeros(5, ACTION_DIM, dtype=torch.bool)
    mask[:, 3] = True
    logits, values = model(features, mask)
    assert logits.shape == (5, ACTION_DIM) and values.shape == (5,)
    assert torch.all(logits[:, 0] == MASK_FILL) and torch.all(logits[:, 3] != MASK_FILL)
    # One legal action: the entropy must stay finite, not NaN.
    assert torch.isfinite(torch.distributions.Categorical(logits=logits).entropy()).all()


def test_each_head_has_weights_of_its_own():
    """The critic's gradient must reach its own trunk and head, and not the policy's:
    separate networks are the whole reason to give each room."""
    model = build()
    features = torch.randn(4, OBS_DIM)
    mask = torch.ones(4, ACTION_DIM, dtype=torch.bool)
    _, values = model(features, mask, torch.rand(4, EQUITY_SLOTS))
    values.sum().backward()
    assert all(p.grad is not None and torch.any(p.grad != 0) for p in model.value_head[0].parameters()
               if p.dim() > 1)
    assert torch.any(model.value_trunk[0].weight.grad != 0)
    assert all(p.grad is None for p in model.trunk.parameters())
    assert all(p.grad is None for p in model.policy_head.parameters())


@pytest.mark.parametrize(
    "bad", [{"hidden": 0}, {"num_layers": 0}, {"head_hidden": 0}, {"head_layers": -1}]
)
def test_a_shape_that_cannot_be_built_is_refused(bad):
    with pytest.raises(ValueError, match="needs hidden"):
        build(**bad)


# ---- checkpoints ---------------------------------------------------------------


def test_a_checkpoint_records_the_whole_shape_and_rebuilds_the_same_network(tmp_path):
    model = build(head_layers=3)
    save_checkpoint(tmp_path / "m.pt", model)
    rebuilt, checkpoint = build_model_from_checkpoint(tmp_path / "m.pt")
    assert checkpoint_shape(checkpoint) == model.shape and set(model.shape) == set(SHAPE_KEYS)
    assert rebuilt.shape == model.shape
    features = torch.randn(3, OBS_DIM)
    mask = torch.ones(3, ACTION_DIM, dtype=torch.bool)
    for got, want in zip(rebuilt.eval()(features, mask), model.eval()(features, mask)):
        assert torch.allclose(got, want)


def test_a_checkpoint_saved_before_the_shape_or_the_encoder_was_recorded_is_refused_clearly(tmp_path):
    save_checkpoint(tmp_path / "m.pt", build())
    for dropped in (("head_layers", "head_hidden"), ("equity",)):
        old = torch.load(tmp_path / "m.pt", weights_only=True)
        for key in dropped:
            del old[key]
        with pytest.raises(IncompatibleCheckpointError, match="equity encoder"):
            check_compatible(old, "old.pt")


def test_a_parent_of_another_shape_is_reported_and_one_of_the_same_shape_is_not(tmp_path):
    save_checkpoint(tmp_path / "p.pt", build(head_layers=1))
    equity = build().equity
    assert parent_shape_mismatch(tmp_path / "p.pt", build(head_layers=1).shape, equity) is None
    other = parent_shape_mismatch(tmp_path / "p.pt", build(head_layers=2).shape, equity)
    assert other == build(head_layers=1).shape


def test_a_broken_parent_is_an_error_not_a_different_shape(tmp_path):
    save_checkpoint(tmp_path / "p.pt", build())
    stale = torch.load(tmp_path / "p.pt", weights_only=True)
    stale["obs_dim"] = OBS_DIM + 1
    torch.save(stale, tmp_path / "stale.pt")
    with pytest.raises(IncompatibleCheckpointError):
        parent_shape_mismatch(tmp_path / "stale.pt", build().shape, build().equity)
