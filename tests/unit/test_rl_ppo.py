from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import DecisionRecord
from pokerlab.players.scripted import get_bot_profile
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import FEATURE_VERSION, OBS_DIM
from pokerlab.rl.policy import PokerActorCritic
from pokerlab.rl.ppo import (
    IncompatibleCheckpointError,
    PPOConfig,
    TrainingBatch,
    build_batch,
    build_model_from_checkpoint,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from pokerlab.rl.rollout import HandTrajectory
from pokerlab.rl.train import SelfPlayTrainer, TrainConfig, archived_opponents


@pytest.fixture
def model() -> PokerActorCritic:
    torch.manual_seed(0)
    return PokerActorCritic(hidden=64, num_layers=2)


def synthetic_trajectory(length: int, offset: float) -> HandTrajectory:
    decisions = [
        DecisionRecord(
            player_id="p0",
            seat=0,
            features=[offset + i] * OBS_DIM,
            legal_mask=[True] * ACTION_DIM,
            action_index=i % ACTION_DIM,
            log_prob=-float(i),
            value=0.0,
        )
        for i in range(length)
    ]
    trajectory = HandTrajectory(seat=0, player_id="p0", decisions=decisions, reward=1.0)
    trajectory.advantages = [offset + 100 + i for i in range(length)]
    trajectory.returns = [offset + 200 + i for i in range(length)]
    return trajectory


def constant_batch(model: PokerActorCritic, advantage: float, size: int = 32) -> TrainingBatch:
    torch.manual_seed(1)
    features = torch.randn(size, OBS_DIM)
    masks = torch.ones(size, ACTION_DIM, dtype=torch.bool)
    actions = torch.zeros(size, dtype=torch.int64)
    with torch.no_grad():
        logits, _ = model(features, masks)
        old_log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
    return TrainingBatch(
        features=features,
        masks=masks,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=torch.full((size,), advantage),
        returns=torch.zeros(size),
    )


def probability_of_bin_zero(model: PokerActorCritic, batch: TrainingBatch) -> float:
    with torch.no_grad():
        logits, _ = model(batch.features, batch.masks)
        return float(torch.softmax(logits, dim=-1)[:, 0].mean())


def test_build_batch_keeps_decisions_advantages_and_returns_aligned():
    trajectories = [synthetic_trajectory(3, 0.0), synthetic_trajectory(2, 50.0)]
    batch = build_batch(trajectories)

    assert len(batch) == 5
    assert batch.features.shape == (5, OBS_DIM)
    assert batch.masks.shape == (5, ACTION_DIM)
    assert batch.actions.tolist() == [0, 1, 2, 0, 1]
    # Row i must pair the i-th decision with the i-th advantage of the same hand.
    assert batch.advantages.tolist() == [100.0, 101.0, 102.0, 150.0, 151.0]
    assert batch.returns.tolist() == [200.0, 201.0, 202.0, 250.0, 251.0]


def test_update_returns_finite_diagnostics(model):
    batch = constant_batch(model, advantage=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = ppo_update(model, optimizer, batch, PPOConfig(epochs=2, minibatch_size=16))

    assert set(stats) == {
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"
    }
    for name, value in stats.items():
        assert math.isfinite(value), f"{name} was not finite"
    assert stats["approx_kl"] >= 0.0
    assert 0.0 <= stats["clip_fraction"] <= 1.0


def test_a_positive_advantage_makes_the_taken_action_more_likely(model):
    """The gradient-direction check: PPO must push probability toward actions
    that beat the critic's expectation."""
    batch = constant_batch(model, advantage=1.0)
    before = probability_of_bin_zero(model, batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Advantage normalisation would zero a constant advantage (std == 0), and
    # the value/entropy terms would muddy an isolated policy-direction check.
    config = PPOConfig(
        epochs=4,
        minibatch_size=32,
        normalize_advantages=False,
        value_coefficient=0.0,
        entropy_coefficient=0.0,
    )
    ppo_update(model, optimizer, batch, config)
    assert probability_of_bin_zero(model, batch) > before


def test_a_negative_advantage_makes_the_taken_action_less_likely(model):
    batch = constant_batch(model, advantage=-1.0)
    before = probability_of_bin_zero(model, batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = PPOConfig(
        epochs=4,
        minibatch_size=32,
        normalize_advantages=False,
        value_coefficient=0.0,
        entropy_coefficient=0.0,
    )
    ppo_update(model, optimizer, batch, config)
    assert probability_of_bin_zero(model, batch) < before


def test_masked_actions_stay_impossible_after_an_update(model):
    batch = constant_batch(model, advantage=1.0)
    batch.masks[:, 3:] = False
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ppo_update(model, optimizer, batch, PPOConfig(epochs=3, minibatch_size=16))

    with torch.no_grad():
        logits, _ = model(batch.features, batch.masks)
        probabilities = torch.softmax(logits, dim=-1)
    assert torch.all(probabilities[:, 3:] == 0.0)


def test_checkpoint_round_trips_weights_and_optimizer_state(model, tmp_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ppo_update(model, optimizer, constant_batch(model, 1.0), PPOConfig(epochs=1))
    path = tmp_path / "nested" / "agent.pt"
    save_checkpoint(path, model, optimizer, iteration=7, metadata={"note": "test"})

    restored = PokerActorCritic(hidden=64, num_layers=2)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(path, restored, restored_optimizer)

    assert checkpoint["iteration"] == 7
    assert checkpoint["metadata"]["note"] == "test"
    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)


def test_a_checkpoint_from_a_different_encoding_is_rejected(model, tmp_path):
    """OBS_DIM changing must fail loudly, not load a silently wrong network."""
    path = tmp_path / "stale.pt"
    save_checkpoint(path, model)
    stale = torch.load(path, weights_only=True)
    stale["obs_dim"] = OBS_DIM + 1
    torch.save(stale, path)

    with pytest.raises(ValueError, match="the encoding changed"):
        load_checkpoint(path, PokerActorCritic(hidden=64, num_layers=2))


def test_a_checkpoint_from_a_stale_feature_version_is_rejected(model, tmp_path):
    """OBS_DIM alone cannot catch this: the position fix reordered what a slot
    means while keeping the vector exactly the same length."""
    path = tmp_path / "old.pt"
    save_checkpoint(path, model)
    stale = torch.load(path, weights_only=True)
    stale["feature_version"] = FEATURE_VERSION + 1
    torch.save(stale, path)

    with pytest.raises(IncompatibleCheckpointError, match="different"):
        load_checkpoint(path, PokerActorCritic(hidden=64, num_layers=2))


def test_a_model_is_rebuilt_at_the_shape_it_was_trained_with(tmp_path):
    original = PokerActorCritic(hidden=48, num_layers=1)
    path = tmp_path / "odd_shape.pt"
    save_checkpoint(path, original)

    rebuilt, checkpoint = build_model_from_checkpoint(path)
    assert (rebuilt.hidden, rebuilt.num_layers) == (48, 1)
    assert checkpoint["feature_version"] == FEATURE_VERSION
    assert not any(p.requires_grad for p in rebuilt.parameters())
    for before, after in zip(original.parameters(), rebuilt.parameters()):
        assert torch.equal(before, after)


def test_archived_agents_are_loaded_as_opponents(tmp_path):
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    for i in range(3):
        save_checkpoint(tmp_path / f"agent-{i}.pt", PokerActorCritic(hidden=32, num_layers=1))

    opponents = archived_opponents(tmp_path, game, limit=5)
    assert len(opponents) == 3
    assert {o.label for o in opponents} == {"agent-0", "agent-1", "agent-2"}

    player = opponents[0].factory("s1", opponents[0].label, random.Random(0))
    assert player.player_id == "s1"


def test_archived_agents_respect_the_limit_and_a_missing_directory(tmp_path):
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    for i in range(4):
        save_checkpoint(tmp_path / f"agent-{i}.pt", PokerActorCritic(hidden=32, num_layers=1))

    assert len(archived_opponents(tmp_path, game, limit=2)) == 2
    assert archived_opponents(tmp_path / "nope", game) == []


def test_unreadable_or_stale_archives_are_skipped_not_fatal(tmp_path):
    """The pool directory is user-owned; one bad file must not stop training."""
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    save_checkpoint(tmp_path / "good.pt", PokerActorCritic(hidden=32, num_layers=1))
    (tmp_path / "garbage.pt").write_text("not a checkpoint at all")

    stale_path = tmp_path / "stale.pt"
    save_checkpoint(stale_path, PokerActorCritic(hidden=32, num_layers=1))
    stale = torch.load(stale_path, weights_only=True)
    stale["feature_version"] = FEATURE_VERSION + 1
    torch.save(stale, stale_path)

    skipped: list[str] = []
    opponents = archived_opponents(tmp_path, game, on_skip=lambda p, why: skipped.append(p.name))

    assert [o.label for o in opponents] == ["good"]
    assert sorted(skipped) == ["garbage.pt", "stale.pt"]


def test_archived_agents_join_the_trainer_pool(tmp_path):
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    save_checkpoint(tmp_path / "veteran.pt", PokerActorCritic(hidden=32, num_layers=1))

    config = TrainConfig(hands_per_iteration=10, bot_keys=("random",))
    baseline = SelfPlayTrainer(game, config, model=PokerActorCritic(hidden=32, num_layers=1))
    with_archive = SelfPlayTrainer(
        game,
        config,
        model=PokerActorCritic(hidden=32, num_layers=1),
        extra_opponents=archived_opponents(tmp_path, game),
    )
    assert len(with_archive._pool) == len(baseline._pool) + 1


def test_a_training_iteration_runs_end_to_end_and_updates_the_weights():
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    trainer = SelfPlayTrainer(
        game,
        TrainConfig(hands_per_iteration=20, bot_keys=("random",), snapshot_every=1),
        PPOConfig(epochs=2, minibatch_size=64),
        rng=random.Random(0),
        model=PokerActorCritic(hidden=32, num_layers=1),
    )
    before = [p.detach().clone() for p in trainer.model.parameters()]
    stats = trainer.train_iteration()

    assert stats["iteration"] == 1.0
    assert stats["decisions"] > 0
    assert math.isfinite(stats["reward_bb"])
    assert any(
        not torch.equal(old, new) for old, new in zip(before, trainer.model.parameters())
    )


def test_duplicate_evaluation_deals_every_rotation_the_same_cards():
    """The property the whole variance reduction rests on: replaying a deal must
    put identical hole cards and identical board in each seat, so that rotating
    the learner through the seats hands it every hand that was dealt."""
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    profile = get_bot_profile("rock")
    trainer = SelfPlayTrainer(
        game, TrainConfig(), rng=random.Random(0), model=PokerActorCritic(hidden=32, num_layers=1)
    )

    deals = []
    for learner_seat in range(game.num_players):
        seen: dict = {}
        table = Table(
            game,
            trainer._seat_players(profile, learner_seat, random.Random(99)),
            rng=random.Random(1234),
            on_hand_started=lambda info, sink=seen: sink.update(info),
        )
        table.play_hand()
        deals.append(seen)

    first = deals[0]
    for other in deals[1:]:
        assert other["hole_cards"] == first["hole_cards"], "seats got different cards"
        assert other["button_seat"] == first["button_seat"], "the layout moved between rotations"


def test_duplicate_evaluation_is_deterministic_and_finite():
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    trainer = SelfPlayTrainer(
        game, TrainConfig(), rng=random.Random(1), model=PokerActorCritic(hidden=32, num_layers=1)
    )
    torch.manual_seed(0)
    first = trainer.evaluate_duplicate("calling_station", deals=5, seed=3)
    torch.manual_seed(0)
    second = trainer.evaluate_duplicate("calling_station", deals=5, seed=3)

    assert math.isfinite(first)
    assert first == pytest.approx(second)


def test_evaluation_reports_a_finite_win_rate():
    game = GameConfig(num_players=2, starting_stack=100, small_blind=1, big_blind=2)
    trainer = SelfPlayTrainer(
        game, TrainConfig(), rng=random.Random(1), model=PokerActorCritic(hidden=32, num_layers=1)
    )
    assert math.isfinite(trainer.evaluate("calling_station", hands=30, seed=2))
