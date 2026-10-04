from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import DecisionRecord
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import FEATURE_VERSION, OBS_DIM
from pokerlab.rl.global_store import read_sidecar
from pokerlab.rl.policy import PokerActorCritic
from pokerlab.rl.pool_registry import (
    DEFAULT_K_FACTOR,
    DEFAULT_K_SCHEDULE,
    DEFAULT_RATING,
    PoolMember,
    PoolRegistry,
    k_for_games,
    pairwise_elo_delta,
)
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
from pokerlab.rl.train import (
    SelfPlayTrainer,
    TrainConfig,
    registry_opponents,
)


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


def test_a_stale_or_unreadable_checkpoint_is_skipped_not_fatal(tmp_path):
    """The store is shared and long-lived; one bad file must not stop training."""
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    save_checkpoint(tmp_path / "good.pt", PokerActorCritic(hidden=32, num_layers=1))
    (tmp_path / "garbage.pt").write_text("not a checkpoint at all")

    stale_path = tmp_path / "stale.pt"
    save_checkpoint(stale_path, PokerActorCritic(hidden=32, num_layers=1))
    stale = torch.load(stale_path, weights_only=True)
    stale["feature_version"] = FEATURE_VERSION + 1
    torch.save(stale, stale_path)

    registry = PoolRegistry(directory=tmp_path)
    for label in ("good", "garbage", "stale"):
        registry.members[label] = PoolMember(label=label, kind="model", ref=f"{label}.pt")

    skipped: list[str] = []
    opponents = registry_opponents(
        registry, game, count=3, on_skip=lambda p, why: skipped.append(p.name)
    )

    assert {o.label for o in opponents} == {"good"}
    assert sorted(skipped) == ["garbage.pt", "stale.pt"]


def test_drawn_opponents_join_the_trainer_pool(tmp_path):
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    save_checkpoint(tmp_path / "veteran.pt", PokerActorCritic(hidden=32, num_layers=1))
    registry = PoolRegistry(directory=tmp_path)
    registry.members["veteran"] = PoolMember(label="veteran", kind="model", ref="veteran.pt")

    config = TrainConfig(hands_per_iteration=10)
    baseline = SelfPlayTrainer(game, config, model=PokerActorCritic(hidden=32, num_layers=1))
    with_pool = SelfPlayTrainer(
        game,
        config,
        model=PokerActorCritic(hidden=32, num_layers=1),
        extra_opponents=registry_opponents(registry, game, count=1),
    )
    assert len(with_pool._pool) == len(baseline._pool) + 1


def test_a_training_iteration_runs_end_to_end_and_updates_the_weights():
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    trainer = SelfPlayTrainer(
        game,
        TrainConfig(hands_per_iteration=20),
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


# ---- the ranked pool -----------------------------------------------------


def small_game() -> GameConfig:
    return GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)


def registry_with_models(tmp_path, count: int, *, max_models: int = 20) -> PoolRegistry:
    registry = PoolRegistry(directory=tmp_path, max_models=max_models)
    for i in range(count):
        save_checkpoint(tmp_path / f"agent-{i}.pt", PokerActorCritic(hidden=32, num_layers=1))
        registry.members[f"agent-{i}"] = PoolMember(
            label=f"agent-{i}", kind="model", ref=f"agent-{i}.pt"
        )
    return registry


def test_registry_opponents_seat_the_best_models_first(tmp_path):
    registry = registry_with_models(tmp_path, 3)
    registry.members["agent-2"].rating = 1900.0
    registry.members["agent-0"].rating = 1100.0

    opponents = registry_opponents(registry, small_game(), count=3)

    assert [o.label for o in opponents][:1] == ["agent-2"]
    assert "agent-0" in {o.label for o in opponents}


def test_registry_opponents_pad_a_short_pool_up_to_the_requested_count(tmp_path):
    """The point of the request: a table always gets a full field, made up
    by re-seating models when the pool is short of the requested count."""
    registry = registry_with_models(tmp_path, 2)
    opponents = registry_opponents(registry, small_game(), count=20)
    assert len(opponents) == 20


def test_registry_opponents_reuse_one_load_for_a_duplicated_member(tmp_path):
    """A duplicate is the same model seated twice, not a second copy of the
    weights."""
    registry = registry_with_models(tmp_path, 1)
    opponents = registry_opponents(registry, small_game(), count=12)
    by_label = [o for o in opponents if o.label == "agent-0"]
    assert len(by_label) > 1
    assert all(o is by_label[0] for o in by_label)


def test_registry_opponents_is_empty_with_no_models(tmp_path):
    """There is no catalog to fall back to any more (see CLAUDE.md,
    "Heuristic bots, removed"): an empty registry seats nothing."""
    registry = PoolRegistry(directory=tmp_path)
    assert registry_opponents(registry, small_game(), count=6) == []


def test_a_broken_checkpoint_in_the_pool_is_skipped_not_fatal(tmp_path):
    registry = registry_with_models(tmp_path, 1)
    (tmp_path / "garbage.pt").write_text("not a checkpoint")
    registry.members["garbage"] = PoolMember(label="garbage", kind="model", ref="garbage.pt")

    skipped: list[str] = []
    opponents = registry_opponents(
        registry, small_game(), count=3, on_skip=lambda p, why: skipped.append(p.name)
    )

    assert skipped == ["garbage.pt"]
    assert opponents


def test_evaluating_against_the_pool_returns_a_finite_win_rate(tmp_path):
    registry = registry_with_models(tmp_path, 2)
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
        registry=registry,
    )
    assert math.isfinite(trainer.evaluate_against_pool(4, hands=10, seed=3))


def test_evaluating_against_the_pool_rates_every_participant(tmp_path):
    registry = registry_with_models(tmp_path, 2)
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
        registry=registry,
    )
    trainer.evaluate_against_pool(6, hands=10, seed=5)

    assert sum(m.games for m in registry.members.values()) > 0
    assert any(m.rating != DEFAULT_RATING for m in registry.members.values())


def test_the_learner_rating_moves_with_its_results(tmp_path):
    registry = registry_with_models(tmp_path, 2)
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
        registry=registry,
    )
    before = trainer.learner_rating
    trainer.evaluate_against_pool(6, hands=10, seed=11)
    assert trainer.learner_rating != before


def test_the_learner_is_rated_through_the_burn_in_schedule(tmp_path):
    """The rating a run reports has to be able to reach the pool it is playing.

    At the flat K it could not: a session moves a rating by at most `K * 0.5`, so
    at 8 the ten sessions of a first evaluation were worth 40 points against
    pools averaging ~1580, and every one of 17 live workers sat between 1469 and
    1534 whatever its bb/100. The learner is not a registry member, so it counts
    its own rated sessions and reads its K off `DEFAULT_K_SCHEDULE` like anyone
    else -- fast while the rating is unknown, settling as the sessions accrue.
    """
    registry = registry_with_models(tmp_path, 2)
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
        registry=registry,
    )
    assert trainer.learner_games == 0

    trainer.evaluate_against_pool(6, hands=10, seed=5)

    # One rated session per block, counted across evaluations rather than per
    # evaluation: the rating accumulates, so the K that shapes it has to fall as
    # the run goes on instead of re-burning at every reading.
    assert trainer.learner_games == 6
    trainer.evaluate_against_pool(4, hands=10, seed=7)
    assert trainer.learner_games == 10


def test_the_learner_burn_in_expires_within_a_run(tmp_path):
    """A burn-in that never ends is just a jumpy rating. The learner leaves the
    schedule's top tier once it has played `DEFAULT_K_SCHEDULE`'s first threshold
    of rated sessions, so a run's later readings settle rather than chasing the
    last one."""
    registry = registry_with_models(tmp_path, 2)
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
        registry=registry,
    )
    burn_in_games = DEFAULT_K_SCHEDULE[1][0]

    trainer.evaluate_against_pool(burn_in_games, hands=10, seed=13)

    assert trainer.learner_games == burn_in_games
    assert k_for_games(trainer.learner_games) < DEFAULT_K_SCHEDULE[0][1]


def test_the_learner_can_actually_reach_the_rating_of_the_pool_it_beats(tmp_path):
    """The failure this whole change is about: a learner that wins every single
    pairwise comparison against a field rated 80 points above it must be able to
    close that gap inside one run. At the flat K of 8 it could not -- ten
    sessions were worth at most 40 points, so the reported rating stayed near the
    1500 baseline and said nothing about the model."""
    registry = PoolRegistry(directory=tmp_path, k_schedule=DEFAULT_K_SCHEDULE)
    pool_rating = DEFAULT_RATING + 80.0
    for i in range(5):
        registry.members[f"m{i}"] = PoolMember(
            label=f"m{i}", kind="model", ref=f"m{i}.pt", rating=pool_rating, frozen=True
        )

    rating = DEFAULT_RATING
    for game_index in range(10):
        results = {"learner": 100.0} | {f"m{i}": -20.0 for i in range(5)}
        ratings = {"learner": rating} | {f"m{i}": pool_rating for i in range(5)}
        deltas = registry.record_session_with_ratings(
            results, ratings, k_factors={"learner": k_for_games(game_index)}
        )
        rating += deltas["learner"]

    assert rating > pool_rating
    # Same ten sessions at the flat K: nowhere near.
    flat = DEFAULT_RATING
    for _ in range(10):
        results = {"learner": 100.0} | {f"m{i}": -20.0 for i in range(5)}
        ratings = {"learner": flat} | {f"m{i}": pool_rating for i in range(5)}
        flat += pairwise_elo_delta(results, ratings, k_factor=DEFAULT_K_FACTOR)["learner"]
    assert flat < pool_rating


def test_evaluating_without_a_registry_is_a_clear_error():
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
    )
    with pytest.raises(ValueError, match="registry"):
        trainer.evaluate_against_pool(1, hands=10)


def test_archiving_saves_the_weights_with_the_learners_rating_beside_them(tmp_path):
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
    )
    trainer._learner_rating = 1712.0
    trainer.archive(tmp_path / "agent-new.pt", iteration=25)

    assert (tmp_path / "agent-new.pt").exists()
    assert read_sidecar(tmp_path / "agent-new.pt") == (1712.0, 25)


def test_archiving_touches_no_registry_and_moves_nothing(tmp_path):
    """The model is published once, at the end of the run: archiving mid-run
    must not edit any ranking or file, so a label never changes weights."""
    registry = registry_with_models(tmp_path, 2)
    before = {label: (m.rating, m.games) for label, m in registry.members.items()}
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
        registry=registry,
    )
    trainer.archive(tmp_path / "agent-new.pt", iteration=10)

    assert {label: (m.rating, m.games) for label, m in registry.members.items()} == before
    assert "agent-new" not in registry.members
    assert not (tmp_path / "registry.json").exists()
    assert not (tmp_path / "retired").exists()
    assert (tmp_path / "agent-0.pt").exists() and (tmp_path / "agent-1.pt").exists()


def test_re_archiving_the_same_path_overwrites_the_best_so_far_in_place(tmp_path):
    """`main()` reuses one path across a whole run to keep only its best model
    on disk: the second call must replace both the weights and the recorded rating."""
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
    )
    path = tmp_path / "agent-run.pt"
    trainer._learner_rating = 1550.0
    trainer.archive(path, iteration=10)
    trainer._learner_rating = 1620.0
    trainer.archive(path, iteration=20)

    assert read_sidecar(path) == (1620.0, 20)
    assert [p.name for p in tmp_path.glob("*.pt")] == ["agent-run.pt"]


def test_evaluating_against_frozen_opponents_moves_only_the_learner(tmp_path):
    """A training run scores its learner against reference points: the drawn
    opponents' ratings must not move, or one run would edit its opponents."""
    registry = registry_with_models(tmp_path, 2)
    for member in registry.members.values():
        member.frozen = True
        member.rating = 1600.0
    trainer = SelfPlayTrainer(
        small_game(),
        TrainConfig(hands_per_iteration=10),
        model=PokerActorCritic(hidden=32, num_layers=1),
        registry=registry,
    )

    trainer.evaluate_against_pool(6, hands=10, seed=5)

    assert all(m.rating == 1600.0 for m in registry.members.values())
    assert trainer.learner_rating != DEFAULT_RATING


# ---- seating a trained model as an opponent -------------------------------


def test_a_checkpoint_can_be_seated_by_path(tmp_path):
    """`model:<path>` is the only bot spec now (see CLAUDE.md, "Heuristic
    bots, removed"), given by path rather than a hardcoded catalog key: a key
    pointing at a checkpoint would break `--list-bots` the moment the file
    went missing."""
    from pokerlab.cli.play import build_players, validate_bot_key

    path = tmp_path / "agent.pt"
    save_checkpoint(path, PokerActorCritic(hidden=32, num_layers=1))
    game = small_game()
    spec = f"model:{path}"
    validate_bot_key(spec)

    players = build_players(3, 0, random.Random(0), [spec, spec], game=game)
    assert sum(1 for p in players if type(p).__name__ == "RLAgentPlayer") >= 1


def test_a_missing_checkpoint_is_rejected_with_a_clear_message(tmp_path):
    from pokerlab.cli.play import validate_bot_key

    with pytest.raises(ValueError, match="no checkpoint at"):
        validate_bot_key(f"model:{tmp_path / 'nope.pt'}")


def test_seating_a_model_without_a_game_config_is_a_clear_error(tmp_path):
    """RLAgentPlayer normalises features by big_blind and starting_stack, which
    an Observation deliberately does not carry."""
    from pokerlab.cli.play import build_players

    path = tmp_path / "agent.pt"
    save_checkpoint(path, PokerActorCritic(hidden=32, num_layers=1))
    with pytest.raises(ValueError, match="GameConfig"):
        build_players(3, 0, random.Random(0), [f"model:{path}"])


def test_a_table_of_trained_models_plays_without_errors(tmp_path):
    from pokerlab.cli.play import build_players

    for index in range(3):
        save_checkpoint(tmp_path / f"a{index}.pt", PokerActorCritic(hidden=32, num_layers=1))
    game = small_game()
    keys = [f"model:{tmp_path / f'a{i}.pt'}" for i in range(3)]

    table = Table(game, build_players(3, 0, random.Random(1), keys, game=game), rng=random.Random(1))
    total = game.starting_stack * game.num_players
    for _ in range(15):
        table.play_hand()
        # Checked before the rebuy, or conservation would be trivially true.
        assert sum(table.stacks) == total, "chip conservation"
        # Stacks are reset the way evaluation does it: without a rebuy the table
        # busts down below two live seats and the engine refuses to deal.
        table.stacks = [game.starting_stack] * game.num_players


def test_discovery_ranks_models_from_the_global_registry(tmp_path):
    """Every model is in the one shared store with one rating, so 'the best'
    means the same wherever you sit."""
    from pokerlab.cli.play import discover_trained_models
    from pokerlab.rl.pool_registry import PoolMember, PoolRegistry

    (tmp_path / "models").mkdir()
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    registry = PoolRegistry(directory=global_dir, max_models=10**9)
    for label, rating in (("vm-a-agent", 1500.0), ("vm-b-agent", 1900.0)):
        (tmp_path / "models" / f"{label}.pt").write_bytes(b"weights")
        registry.members[label] = PoolMember(
            label=label, kind="model", ref=str(tmp_path / "models" / f"{label}.pt"), rating=rating
        )
    registry.save()

    found = discover_trained_models(tmp_path, limit=10)
    assert [label for label, _p, _r in found] == ["vm-b-agent", "vm-a-agent"]


def test_discovery_skips_registry_entries_whose_file_is_gone(tmp_path):
    from pokerlab.cli.play import discover_trained_models
    from pokerlab.rl.pool_registry import PoolMember, PoolRegistry

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    registry = PoolRegistry(directory=global_dir, max_models=10**9)
    registry.members["ghost"] = PoolMember(label="ghost", kind="model", ref="ghost.pt")
    registry.save()
    assert discover_trained_models(tmp_path, fallback_dir=None) == []


def test_discovery_finds_a_model_whose_ref_is_stale_by_looking_it_up_by_label(tmp_path):
    """A `ref` left pointing at an old location must not hide a top model: the
    checkpoint is found in the shared store by its label instead."""
    from pokerlab.cli.play import discover_global_top_models
    from pokerlab.rl.pool_registry import PoolMember, PoolRegistry

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "moved.pt").write_bytes(b"weights")
    registry = PoolRegistry(directory=global_dir, max_models=10**9)
    registry.members["moved"] = PoolMember(
        label="moved", kind="model", ref="machines/old/pool/moved.pt", rating=1800.0
    )
    registry.save()

    found = discover_global_top_models(global_dir, limit=3)
    assert [(label, path) for label, path, _r in found] == [("moved", tmp_path / "models" / "moved.pt")]


def test_discover_global_top_models_reads_the_one_shared_scale(tmp_path):
    """Unlike discover_trained_models, this reads rl/global_arena.py's single
    cross-machine registry rather than approximating 'best' per machine."""
    from pokerlab.cli.play import discover_global_top_models
    from pokerlab.rl.pool_registry import PoolMember, PoolRegistry

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    elsewhere = tmp_path / "machines" / "vm-a" / "pool"
    elsewhere.mkdir(parents=True)
    registry = PoolRegistry(directory=global_dir, max_models=10**9)
    for label, rating in (("weak", 1400.0), ("strong", 1900.0), ("mid", 1600.0)):
        path = elsewhere / f"{label}.pt"
        path.write_bytes(b"weights")
        registry.members[label] = PoolMember(
            label=label, kind="model", ref=str(path), rating=rating
        )
    registry.save()

    found = discover_global_top_models(global_dir, limit=2)
    assert [label for label, _p, _r in found] == ["strong", "mid"]


def test_discover_global_top_models_skips_missing_files(tmp_path):
    """A global-registry `ref` can point anywhere on the volume; a file that
    has since moved or been deleted must not break the whole list."""
    from pokerlab.cli.play import discover_global_top_models
    from pokerlab.rl.pool_registry import PoolMember, PoolRegistry

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    present = tmp_path / "present.pt"
    present.write_bytes(b"weights")
    registry = PoolRegistry(directory=global_dir, max_models=10**9)
    registry.members["ghost"] = PoolMember(
        label="ghost", kind="model", ref=str(tmp_path / "gone.pt"), rating=2000.0
    )
    registry.members["present"] = PoolMember(
        label="present", kind="model", ref=str(present), rating=1000.0
    )
    registry.save()

    found = discover_global_top_models(global_dir, limit=5)
    assert [label for label, _p, _r in found] == ["present"]


def test_discover_global_top_models_on_a_missing_registry_is_empty(tmp_path):
    from pokerlab.cli.play import discover_global_top_models

    assert discover_global_top_models(tmp_path / "nope", fallback_dir=None) == []
    assert discover_global_top_models(tmp_path / "nope", fallback_dir=tmp_path / "also-nope") == []


def test_with_no_checkpoints_the_models_come_from_the_top_models_folder(tmp_path):
    import json

    from pokerlab.cli.play import discover_global_top_models, discover_trained_models

    top = tmp_path / "top_models"
    top.mkdir()
    for label in ("mid", "best", "unrated"):
        (top / f"{label}.pt").write_bytes(b"weights")
    (top / "ratings.json").write_text(
        json.dumps({"ratings": {"mid": 1820.0, "best": 1847.8, "gone": 1900.0}}), encoding="utf-8"
    )
    found = discover_global_top_models(tmp_path / "no-global", limit=5, fallback_dir=top)
    # rated best first; a rating with no file is skipped; an unrated file comes last
    assert [(label, rating) for label, _p, rating in found] == [
        ("best", 1847.8), ("mid", 1820.0), ("unrated", 1500.0)
    ]
    assert discover_trained_models(tmp_path, fallback_dir=top)[0][0] == "best"
    assert len(discover_global_top_models(tmp_path / "no-global", limit=2, fallback_dir=top)) == 2


def test_the_registry_wins_over_the_top_models_folder(tmp_path):
    from pokerlab.cli.play import discover_global_top_models
    from pokerlab.rl.pool_registry import PoolMember, PoolRegistry

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (tmp_path / "live.pt").write_bytes(b"weights")
    registry = PoolRegistry(directory=global_dir, max_models=10**9)
    registry.members["live"] = PoolMember(label="live", kind="model", ref=str(tmp_path / "live.pt"))
    registry.save()
    top = tmp_path / "top_models"
    top.mkdir()
    (top / "copy.pt").write_bytes(b"weights")
    assert [label for label, _p, _r in discover_global_top_models(global_dir, fallback_dir=top)] == ["live"]


def test_the_fallback_folder_is_the_one_push_top_models_writes():
    from pokerlab.cli import play
    from pokerlab.rl import push_top_models

    assert str(play.TOP_MODELS_DIR) == push_top_models.DEFAULT_DIR
    assert play.TOP_MODELS_RATINGS == push_top_models.RATINGS_FILE


# ---- what a published model remembers about the run that made it --------------


def _train_args(**overrides):
    from types import SimpleNamespace

    values = {
        "hp_arm": "sampled", "machine": "host-a", "seed": 7, "resume": False,
        "iterations": 1000, "hands": 512, "players": 6, "stack": 200, "sb": 1, "bb": 2,
        "lr": 3e-4, "ppo_epochs": 4, "clip_epsilon": 0.2, "entropy_coef": 0.0,
        "opponent_probability": 0.5,
        "minibatch_size": 1024, "gae_lambda": 0.95, "value_coef": 0.5, "max_grad_norm": 0.5,
        "pool_models": 20, "pool_top_share": 0.5, "pool_top_n": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_run_metadata_records_every_swept_axis():
    """Anything a worker's settings can differ by has to be in here, or the run
    it produced cannot be attributed to the settings that produced it -- which
    is the whole reason the metadata slot is being filled at all."""
    from pokerlab.rl.train import run_metadata

    recorded = run_metadata(_train_args(lr=7e-4, hands=1024, ppo_epochs=2))

    for axis in (
        "lr", "hands", "ppo_epochs", "clip_epsilon",
        "opponent_probability", "pool_top_share", "pool_top_n",
        "minibatch_size", "gae_lambda", "value_coef", "max_grad_norm",
    ):
        assert axis in recorded, axis
    assert (recorded["lr"], recorded["hands"], recorded["ppo_epochs"]) == (7e-4, 1024, 2)


def test_run_metadata_records_which_arm_the_worker_was_in():
    """Runs that inherited their settings have them correlated with their
    parent's quality, so a response curve can only be read off the sampled arm.
    Without this field the two are indistinguishable afterwards."""
    from pokerlab.rl.train import run_metadata

    assert run_metadata(_train_args(hp_arm="inherited"))["hp_arm"] == "inherited"
    assert run_metadata(_train_args())["hp_arm"] == "sampled"


def test_the_metadata_survives_the_round_trip_into_a_published_model(tmp_path):
    """It has to come back out of the checkpoint: the next generation reads a
    parent's settings from exactly here to inherit and perturb them."""
    from pokerlab.rl.ppo import save_checkpoint
    from pokerlab.rl.train import run_metadata

    path = tmp_path / "agent.pt"
    save_checkpoint(path, PokerActorCritic(), metadata=run_metadata(_train_args(lr=9e-4)))

    reloaded = torch.load(path, map_location="cpu", weights_only=True)

    assert reloaded["metadata"]["lr"] == 9e-4
    assert reloaded["metadata"]["schema"] >= 1


def test_a_schema_version_travels_with_it():
    """So an analysis can refuse a mixture of schemas instead of averaging
    across fields that changed meaning."""
    from pokerlab.rl.train import RUN_METADATA_VERSION, run_metadata

    assert run_metadata(_train_args())["schema"] == RUN_METADATA_VERSION


# ---- the Elo fill-in phase --------------------------------------------------


def _fill_args(tmp_path, **overrides):
    values = {
        "scratch_dir": tmp_path / "scratch",
        "fill_stop_file": str(tmp_path / "FILL_STOP"),
        "fill_min_sessions": 10,
        "fill_deadline_minutes": 60.0,
        "fill_games_per_model": 1,
        "global_dir": tmp_path / "global",
        "global_root": tmp_path,
        "global_sample": 50,
        "global_benchmark_sample": 5,
        "global_hands_per_game": 1000,
        "global_lock_seconds": 120,
        "machine": "host-a",
        "device": "cpu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeRound:
    """Stands in for `run_population_round`, recording how it was called."""

    def __init__(self, *, sessions_played=4, played=55):
        self.calls = []
        self._sessions = sessions_played
        self._played = played

    def __call__(self, **kwargs):
        from pokerlab.rl.global_arena import PopulationRoundReport

        self.calls.append(kwargs)
        return PopulationRoundReport(
            played=self._played,
            sessions_played=self._sessions,
            # Far larger, as on a real fleet: this merge folded in every other
            # machine's pending results too.
            sessions=self._sessions + 900,
        )


def _patch_round(monkeypatch, fake):
    import pokerlab.rl.global_arena as arena
    import pokerlab.rl.train as train_module

    monkeypatch.setattr(arena, "run_population_round", fake)
    # Only `.members` is read, to build the rating map the top bias needs.
    monkeypatch.setattr(
        train_module, "load_ranking", lambda _dir: SimpleNamespace(members={})
    )


def test_the_fill_in_stops_once_the_supervisor_says_so(tmp_path, monkeypatch):
    from pokerlab.rl.train import run_elo_fill_in

    fake = FakeRound(sessions_played=4)
    _patch_round(monkeypatch, fake)
    (tmp_path / "FILL_STOP").touch()  # the generation is already over

    rounds, sessions = run_elo_fill_in(_fill_args(tmp_path), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2))

    # The minimum is honoured first, and only then the flag: 3 rounds of 4.
    assert (rounds, sessions) == (3, 12)


def test_the_minimum_holds_even_when_every_worker_finishes_together(tmp_path, monkeypatch):
    """The case the user asked to protect: without a floor, a generation whose
    workers all drew the same `--hands` would do no rating at all."""
    from pokerlab.rl.train import run_elo_fill_in

    fake = FakeRound(sessions_played=1)
    _patch_round(monkeypatch, fake)
    (tmp_path / "FILL_STOP").touch()

    _rounds, sessions = run_elo_fill_in(
        _fill_args(tmp_path, fill_min_sessions=7), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2)
    )

    assert sessions >= 7


def test_the_deadline_is_the_only_cap_and_it_beats_the_minimum(tmp_path, monkeypatch):
    """A worker whose supervisor died must stop on its own, and it must not
    keep going forever chasing a minimum it cannot reach."""
    from pokerlab.rl.train import run_elo_fill_in

    fake = FakeRound(sessions_played=1)
    _patch_round(monkeypatch, fake)

    rounds, sessions = run_elo_fill_in(
        # No stop file will ever appear, and no time to play in.
        _fill_args(
            tmp_path, fill_stop_file="", fill_min_sessions=10_000,
            fill_deadline_minutes=0.0,
        ),
        GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2),
    )

    assert (rounds, sessions) == (0, 0)


def test_a_fill_in_round_can_never_prune(tmp_path, monkeypatch):
    """It draws with a top bias, and eligibility for deletion is a percentile of
    games -- so a pruning pass under this draw would eat the middle of the
    population instead of its bottom."""
    from pokerlab.rl.global_arena import NO_PRUNE_TRIGGER
    from pokerlab.rl.train import run_elo_fill_in

    fake = FakeRound()
    _patch_round(monkeypatch, fake)
    (tmp_path / "FILL_STOP").touch()

    run_elo_fill_in(_fill_args(tmp_path, fill_min_sessions=1), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2))

    assert fake.calls[0]["trigger_size"] == NO_PRUNE_TRIGGER
    assert fake.calls[0]["draw"] is not None
    assert fake.calls[0]["games_per_model"] == 1


def test_the_worker_counts_the_sessions_it_played_not_the_ones_it_merged(
    tmp_path, monkeypatch
):
    """`sessions` counts what the merge folded in from every machine, which on a
    busy fleet is thousands -- reading it would satisfy the minimum on the first
    round without this worker having played anything."""
    from pokerlab.rl.train import run_elo_fill_in

    fake = FakeRound(sessions_played=2)
    _patch_round(monkeypatch, fake)
    (tmp_path / "FILL_STOP").touch()

    rounds, sessions = run_elo_fill_in(
        _fill_args(tmp_path, fill_min_sessions=6), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2)
    )

    assert (rounds, sessions) == (3, 6)


def test_the_supervisor_is_told_this_worker_is_only_waiting(tmp_path, monkeypatch):
    """The marker is the worker's half of the handshake: without it the
    supervisor waits for a worker that is waiting for the supervisor."""
    from pokerlab.rl.phases import FILL_DRAINING_FILENAME
    from pokerlab.rl.train import run_elo_fill_in

    seen = []
    marker_path = tmp_path / "scratch" / FILL_DRAINING_FILENAME

    class WatchingRound(FakeRound):
        def __call__(self, **kwargs):
            seen.append(marker_path.exists())
            return super().__call__(**kwargs)

    _patch_round(monkeypatch, WatchingRound())
    (tmp_path / "FILL_STOP").touch()

    run_elo_fill_in(_fill_args(tmp_path, fill_min_sessions=1), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2))

    assert seen == [True]  # up before the very first round
    assert not marker_path.exists()  # and gone afterwards


def test_a_broken_round_costs_the_extra_rounds_and_nothing_else(tmp_path, monkeypatch):
    """By this point the run has trained, published and been rated; a failure in
    a bonus phase must not turn that into a failed process."""
    from pokerlab.rl.phases import FILL_DRAINING_FILENAME
    from pokerlab.rl.train import run_elo_fill_in

    def explode(**_kwargs):
        raise RuntimeError("il volume e' sparito")

    _patch_round(monkeypatch, explode)

    rounds, sessions = run_elo_fill_in(_fill_args(tmp_path), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2))

    assert (rounds, sessions) == (0, 0)
    # The marker is cleared on the error path too, or the next generation's
    # supervisor reads a stale directory as a worker already draining.
    assert not (tmp_path / "scratch" / FILL_DRAINING_FILENAME).exists()


def test_an_empty_store_ends_the_phase_instead_of_spinning(tmp_path, monkeypatch):
    from pokerlab.rl.train import run_elo_fill_in

    _patch_round(monkeypatch, FakeRound(played=0, sessions_played=0))

    rounds, sessions = run_elo_fill_in(_fill_args(tmp_path), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2))

    assert (rounds, sessions) == (0, 0)


def test_the_ranking_is_re_read_for_every_round(tmp_path, monkeypatch):
    """The previous round just moved the ratings the top bias is computed from,
    and so did every other machine."""
    import pokerlab.rl.train as train_module
    from pokerlab.rl.train import run_elo_fill_in

    reads = []
    _patch_round(monkeypatch, FakeRound(sessions_played=1))
    monkeypatch.setattr(
        train_module,
        "load_ranking",
        lambda _dir: reads.append(1) or SimpleNamespace(members={}),
    )
    (tmp_path / "FILL_STOP").touch()

    run_elo_fill_in(_fill_args(tmp_path, fill_min_sessions=3), GameConfig(num_players=6, starting_stack=200, small_blind=1, big_blind=2))

    assert len(reads) == 3
