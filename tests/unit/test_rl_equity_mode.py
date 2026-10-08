"""The policy reads the equity network's per-player encoder in place of the card planes, and the
critic, a network of its own, reads every player's equity in place of them."""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from support import fake_critic, fixed_mix

from pokerlab.players.rl_agent import DecisionRecord, PolicyDecision
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.equity_net import (
    EquityEncoder,
    EquityNet,
    canonical_suits,
    encoder_config,
    equity_net_from_checkpoint,
    load_encoder_weights,
    load_equity_checkpoint,
)
from pokerlab.rl.features import CARDS_DIM, OBS_DIM
from pokerlab.rl.policy import EQUITY_SLOTS, PokerActorCritic, make_critic_fns
from pokerlab.rl.ppo import (
    IncompatibleCheckpointError,
    PPOConfig,
    build_batch,
    build_model_from_checkpoint,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from pokerlab.rl.rollout import CriticFns, HandTrajectory, SelfPlayCollector
from pokerlab.rl.train import parent_shape_mismatch

PLANE = 52


def uniform_masked_policy(rng: random.Random):
    def policy_fn(features: list[float], mask: list[bool]) -> PolicyDecision:
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    return policy_fn


@pytest.fixture
def equity_file(tmp_path):
    """A small, randomly initialised equity network saved the way `studies/equity_net` does."""
    torch.manual_seed(3)
    net = EquityNet(hidden=16, latent=8, arch="attn", blocks=1, layers=1, heads=1)
    path = tmp_path / "equity.pt"
    torch.save({"config": net.config(), "state": net.state_dict()}, path)
    return path


@pytest.fixture
def equity_model(equity_file):
    saved = load_equity_checkpoint(equity_file)
    torch.manual_seed(1)
    model = PokerActorCritic(hidden=32, num_layers=2, equity=encoder_config(saved))
    load_encoder_weights(model.equity_encoder, saved)
    return model, saved


def features_and_mask(batch: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, OBS_DIM, generator=generator), torch.ones(batch, ACTION_DIM, dtype=torch.bool)


# ---- the encoder ---------------------------------------------------------------------


def test_the_encoder_is_the_equity_networks_own_phi_on_the_players_cards_and_the_board(equity_file):
    saved = load_equity_checkpoint(equity_file)
    net = equity_net_from_checkpoint(saved)
    encoder = EquityEncoder(**encoder_config(saved))
    load_encoder_weights(encoder, saved)
    hole = torch.zeros(4, PLANE)
    board = torch.zeros(4, PLANE)
    for row in range(4):
        hole[row, [row, 20 + row]] = 1.0
        board[row, [5, 30, 44 - row]] = 1.0
    own, shared = canonical_suits(hole.unsqueeze(1), board)
    expected = net.phi(torch.cat([own.squeeze(1), shared], dim=-1))
    expected = torch.nn.functional.layer_norm(expected, (expected.shape[-1],))
    assert torch.allclose(encoder(hole, board), expected, atol=1e-6)


def test_the_encoder_does_not_change_when_the_suits_are_renamed(equity_file):
    saved = load_equity_checkpoint(equity_file)
    encoder = EquityEncoder(**encoder_config(saved))
    load_encoder_weights(encoder, saved)
    hole = torch.zeros(1, PLANE)
    board = torch.zeros(1, PLANE)
    hole[0, [3, 16]] = 1.0  # two suits
    board[0, [5, 18, 31]] = 1.0
    swap = torch.tensor([2, 0, 3, 1])  # a permutation of the four suits
    index = (swap[:, None] * 13 + torch.arange(13)[None]).reshape(-1)
    assert torch.allclose(encoder(hole, board), encoder(hole[:, index], board[:, index]), atol=1e-6)


def test_the_encoders_weights_are_frozen_and_stay_so(equity_model):
    model, _ = equity_model
    assert all(not p.requires_grad for p in model.equity_encoder.parameters())
    before = [p.detach().clone() for p in model.equity_encoder.parameters()]
    features, mask = features_and_mask()
    logits, _ = model(features, mask)
    logits.sum().backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.step()
    model.train()
    assert all(torch.equal(a, b) for a, b in zip(before, model.equity_encoder.parameters()))


# ---- what each half of the network reads -----------------------------------------------


def test_the_policy_reads_the_own_cards_and_the_board_and_no_other_card_plane(equity_model):
    model, _ = equity_model
    features, mask = features_and_mask(1)
    base, _ = model(features, mask)
    for plane in (1, 2, 3, 5):  # flop, turn, river, hole + board: replaced by the encoder
        changed = features.clone()
        changed[:, plane * PLANE : (plane + 1) * PLANE] = 1.0 - changed[:, plane * PLANE : (plane + 1) * PLANE]
        assert torch.equal(model(changed, mask)[0], base)
    for plane in (0, 4):  # own cards, the whole board: what the encoder reads
        changed = features.clone()
        changed[:, plane * PLANE : (plane + 1) * PLANE] = 0.0
        assert not torch.equal(model(changed, mask)[0], base)
    rest = features.clone()
    rest[:, CARDS_DIM + 10] += 0.5  # anything after the cards
    assert not torch.equal(model(rest, mask)[0], base)


def test_the_policy_never_sees_what_the_critic_is_given(equity_model):
    model, _ = equity_model
    features, mask = features_and_mask()
    alone, _ = model(features, mask)
    with_extra, _ = model(features, mask, critic_extra=torch.rand(6, EQUITY_SLOTS))
    assert torch.equal(alone, with_extra)


def test_without_the_critics_input_the_value_is_zero_and_nothing_of_the_critic_runs(equity_model):
    model, _ = equity_model
    features, mask = features_and_mask()
    _, value = model(features, mask)
    assert torch.equal(value, torch.zeros(6))
    extra = torch.rand(6, EQUITY_SLOTS)
    assert torch.equal(model(features, mask, extra)[1], model.critic_value(features, extra))


def test_the_critic_ignores_the_cards_and_reads_the_equities(equity_model):
    model, _ = equity_model
    features, _ = features_and_mask()
    extra = torch.rand(6, EQUITY_SLOTS)
    base = model.critic_value(features, extra)
    cards = features.clone()
    cards[:, :CARDS_DIM] = 1.0 - cards[:, :CARDS_DIM]
    assert torch.equal(model.critic_value(cards, extra), base)
    assert not torch.equal(model.critic_value(features, extra + 0.3), base)


def test_the_two_halves_are_trained_apart(equity_model):
    """The critic's loss moves the critic only, the policy's moves the policy only."""
    model, _ = equity_model
    features, mask = features_and_mask()
    logits, value = model(features, mask, critic_extra=torch.rand(6, EQUITY_SLOTS))
    value.pow(2).sum().backward()
    assert all(p.grad is not None and torch.any(p.grad != 0) for p in model.value_trunk[0].parameters()
               if p.requires_grad)
    assert all(p.grad is None for p in model.trunk.parameters())
    assert all(p.grad is None for p in model.policy_head.parameters())
    model.zero_grad(set_to_none=True)
    logits, _ = model(features, mask, critic_extra=torch.rand(6, EQUITY_SLOTS))
    logits.sum().backward()
    assert torch.any(model.trunk[0].weight.grad != 0)
    assert all(p.grad is None for p in model.value_trunk.parameters())
    assert all(p.grad is None for p in model.value_head.parameters())


# ---- the critic's input, from the table -------------------------------------------------


def played(num_players: int, hands: int, critic: CriticFns, seed: int = 4) -> list[HandTrajectory]:
    collector = SelfPlayCollector(
        fixed_mix(num_players), uniform_masked_policy(random.Random(seed)), rng=random.Random(seed), critic=critic
    )
    return collector.collect(hands)


def plane_indices(features: list[float], plane: int) -> set[int]:
    return {i for i, x in enumerate(features[plane * PLANE : (plane + 1) * PLANE]) if x}


def test_every_decision_carries_the_cards_of_the_table_in_the_order_of_its_features():
    trajectories = played(4, 12, fake_critic())
    assert trajectories
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            view = decision.critic_view
            assert view is not None and len(view.holes) == 4
            # slot 0 is the deciding player: the two cards its own features show
            assert set(view.holes[0]) == plane_indices(decision.features, 0)
            assert set(view.board) == plane_indices(decision.features, 4)


def test_a_seat_that_has_folded_has_no_cards_in_the_view():
    trajectories = played(6, 40, fake_critic())
    holes_seen = [view.holes for t in trajectories for d in t.decisions for view in [d.critic_view]]
    assert any(None in holes for holes in holes_seen)  # somebody folded before somebody else acted
    assert all(holes[0] is not None for holes in holes_seen)  # the player acting is in the hand


def test_the_collector_fills_in_the_critics_input_and_value_and_computes_gae_from_them():
    trajectories = played(3, 10, fake_critic())
    for trajectory in trajectories:
        assert len(trajectory.advantages) == len(trajectory.decisions)
        for decision in trajectory.decisions:
            assert decision.critic_extra == [0.25] * EQUITY_SLOTS
            assert decision.value == pytest.approx(0.25 * EQUITY_SLOTS)


def test_a_collector_cannot_be_built_without_a_critic():
    with pytest.raises(TypeError, match="critic"):
        SelfPlayCollector(fixed_mix(3), uniform_masked_policy(random.Random(2)), rng=random.Random(2))


def test_the_equities_of_a_view_follow_the_slots_of_its_seats(equity_file):
    from pokerlab.players.rl_agent import DealView

    saved = load_equity_checkpoint(equity_file)
    net = equity_net_from_checkpoint(saved)
    model = PokerActorCritic(hidden=8, num_layers=1, equity=encoder_config(saved))
    equity = make_critic_fns(model, net).equity
    view = DealView(holes=((0, 14), None, (30, 43), (9, 22)), board=(3, 17, 40))
    row = equity([view])[0]
    assert len(row) == EQUITY_SLOTS
    assert row[1] == 0.0 and all(x == 0.0 for x in row[4:])
    assert sum(row) == pytest.approx(1.0, abs=1e-5)
    # the same hands with the folded seat closed up: the shares of the others do not move
    closed = equity([DealView(holes=((0, 14), (30, 43), (9, 22)), board=(3, 17, 40))])[0]
    assert [row[0], row[2], row[3]] == pytest.approx(closed[:3], abs=1e-5)


# ---- training ---------------------------------------------------------------------------


def test_a_ppo_update_trains_the_critic_on_what_the_collector_gave_it(equity_model):
    model, saved = equity_model
    critic = make_critic_fns(model, equity_net_from_checkpoint(saved))
    trajectories = played(3, 16, critic)
    batch = build_batch(trajectories)
    assert batch.critic_extra is not None and batch.critic_extra.shape == (len(batch), EQUITY_SLOTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in model.value_head.parameters()]
    stats = ppo_update(model, optimizer, batch, PPOConfig(epochs=2, minibatch_size=16))
    assert not any(math.isnan(value) for value in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.value_head.parameters()))


def test_a_batch_cannot_hold_a_decision_without_the_critics_input():
    def decision(extra):
        return DecisionRecord(
            player_id="p", seat=0, features=[0.0] * OBS_DIM, legal_mask=[True] * ACTION_DIM,
            action_index=0, log_prob=0.0, value=0.0, stake_bb=50.0, critic_extra=extra,
        )

    trajectory = HandTrajectory(
        seat=0, player_id="p", decisions=[decision([0.0] * EQUITY_SLOTS), decision(None)], reward=0.0,
        advantages=[0.0, 0.0], returns=[0.0, 0.0],
    )
    with pytest.raises(ValueError, match="critic"):
        build_batch([trajectory])


# ---- checkpoints --------------------------------------------------------------------------


def test_a_checkpoint_rebuilds_the_model_with_its_encoder_and_the_same_outputs(tmp_path, equity_model):
    model, _ = equity_model
    save_checkpoint(tmp_path / "m.pt", model)
    rebuilt, _ = build_model_from_checkpoint(tmp_path / "m.pt")
    assert rebuilt.equity == model.equity
    features, mask = features_and_mask()
    assert torch.allclose(rebuilt(features, mask)[0], model.eval()(features, mask)[0])
    assert all(not p.requires_grad for p in rebuilt.parameters())


def test_models_built_on_different_encoders_cannot_exchange_weights(tmp_path, equity_model):
    model, _ = equity_model
    save_checkpoint(tmp_path / "p.pt", model)
    other = PokerActorCritic(hidden=32, num_layers=2, equity={**model.equity, "latent": 4})
    with pytest.raises(IncompatibleCheckpointError, match="encoder"):
        load_checkpoint(tmp_path / "p.pt", other)


def test_a_checkpoint_saved_without_an_encoder_is_refused(tmp_path, equity_model):
    model, _ = equity_model
    save_checkpoint(tmp_path / "old.pt", model)
    saved = torch.load(tmp_path / "old.pt", weights_only=True)
    del saved["equity"]
    torch.save(saved, tmp_path / "old.pt")
    with pytest.raises(IncompatibleCheckpointError, match="equity"):
        build_model_from_checkpoint(tmp_path / "old.pt")
    with pytest.raises(IncompatibleCheckpointError, match="equity"):
        load_checkpoint(tmp_path / "old.pt", model)


def test_a_model_without_an_encoder_cannot_be_built():
    with pytest.raises(TypeError, match="equity"):
        PokerActorCritic(hidden=32, num_layers=2)


def test_a_parent_built_with_another_encoder_is_reported_like_one_of_another_shape(tmp_path, equity_model):
    model, _ = equity_model
    save_checkpoint(tmp_path / "p.pt", model)
    assert parent_shape_mismatch(tmp_path / "p.pt", model.shape, model.equity) is None
    assert parent_shape_mismatch(tmp_path / "p.pt", model.shape, {**model.equity, "latent": 4}) is not None
