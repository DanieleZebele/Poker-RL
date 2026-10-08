"""The continuous-training supervisor: inheritance, sweeping, and state.

Only the filesystem choreography is covered here, with no torch: it is the part
that can silently lose work (a sweep that deletes a live worker's directory, a
model that is never published), and it is testable without training anything.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

import pokerlab.rl.loop as loop_module
from pokerlab.rl.global_store import (
    ARENA_REQUEST_FILENAME,
    read_member,
    request_benchmark_arena,
    write_sidecar,
)
from pokerlab.rl.loop import (
    HP_ARM_FALLBACK,
    HP_ARM_INHERITED,
    HP_ARM_SAMPLED,
    HP_AXES,
    HP_MULTIPLIERS,
    REWARD_WINDOW_HANDS,
    STAGE_ERROR,
    STAGE_STARTING,
    STAGE_TRAINING,
    Hyperparameters,
    LoopState,
    format_finishing_line,
    format_leaderboard,
    format_sweep_line,
    format_worker_table,
    hyperparameter_plan,
    inheritance_plan,
    launch_worker,
    parse_worker_log,
    perturb_hyperparameters,
    stage_cell,
    stage_label,
    starting_hyperparameters,
    sweep_stale_work,
    wait_for_workers,
    worker_progress,
)
from pokerlab.rl.phases import (
    DONE,
    ELO_FILL,
    ELO_MERGE,
    ELO_PLAY,
    EVALUATION,
    PRUNING,
    SERIES,
    hyperparameters_marker,
    marker,
    parse_progress,
    progress_marker,
)
from pokerlab.rl.pool_registry import PoolMember, PoolRegistry


def test_loop_state_round_trips(tmp_path):
    path = tmp_path / "loop_state.json"
    state = LoopState(started="oggi", generation=3, phase="arena", workers=8)
    state.history.append({"generation": 3, "best_label": "x"})
    state.save(path)

    reloaded = LoopState.load(path)
    assert (reloaded.generation, reloaded.phase, reloaded.workers) == (3, "arena", 8)
    assert reloaded.history[-1]["best_label"] == "x"


def test_a_corrupt_state_file_does_not_stop_the_loop(tmp_path):
    path = tmp_path / "loop_state.json"
    path.write_text("{not json", encoding="utf-8")
    assert LoopState.load(path).generation == 0


def test_state_is_written_atomically(tmp_path):
    """A crash mid-write must not leave a half-file that the next status read
    silently treats as 'never ran'."""
    path = tmp_path / "loop_state.json"
    LoopState(generation=1).save(path)
    LoopState(generation=2).save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["generation"] == 2
    assert not path.with_suffix(".json.tmp").exists()


def test_the_leaderboard_ranks_by_rating(tmp_path):
    registry = PoolRegistry(directory=tmp_path, max_models=20)
    for label, rating in (("low", 1100.0), ("high", 1900.0), ("mid", 1500.0)):
        registry.members[label] = PoolMember(label=label, ref=f"{label}.pt", rating=rating)

    lines = format_leaderboard(registry).splitlines()
    assert "high" in lines[1] and "low" in lines[3]


def test_the_leaderboard_leaves_out_the_frozen_benchmark_anchors(tmp_path):
    registry = PoolRegistry(directory=tmp_path, max_models=20)
    registry.members["anchor"] = PoolMember(
        label="anchor", ref="anchor.pt", rating=2500.0, frozen=True
    )
    registry.members["model"] = PoolMember(label="model", ref="model.pt", rating=1600.0)

    text = format_leaderboard(registry)

    assert "model" in text and "anchor" not in text


# ---- live monitoring -----------------------------------------------------


def write_worker_log(
    log_dir,
    generation,
    worker,
    iterations,
    *,
    entropy="0.500",
    rating=None,
    reward="+1.00",
):
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"iter {i:4}  reward   {reward} bb  policy -0.0100  value    1.000  "
        f"entropy {entropy}  kl 0.0100  clip 0.100"
        for i in range(1, iterations + 1)
    ]
    if rating is not None:
        lines.append(f"        eval vs pool: +10.0 bb/100  rating {rating}")
    (log_dir / f"gen{generation:04d}-w{worker:02d}.log").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def test_worker_progress_reads_the_workers_own_logs(tmp_path):
    """The supervisor is blocked in wait() for the whole training phase, so the
    logs are the only thing that reports progress while it runs."""
    write_worker_log(tmp_path, 2, 0, 37, entropy="0.812", rating="1523")
    write_worker_log(tmp_path, 2, 1, 12)

    rows = worker_progress(tmp_path, 2)

    assert [(row.name, row.iterations) for row in rows] == [("w00", 37), ("w01", 12)]
    assert rows[0].entropy == "0.812"
    assert rows[0].rating == "1523"
    assert rows[1].rating == "-", "a worker that has not evaluated yet has no rating"


def test_worker_progress_ignores_other_generations(tmp_path):
    write_worker_log(tmp_path, 1, 0, 100)
    write_worker_log(tmp_path, 2, 0, 5)
    assert [row.name for row in worker_progress(tmp_path, 2)] == ["w00"]
    assert worker_progress(tmp_path, 2)[0].iterations == 5


def test_worker_progress_on_a_missing_directory_is_empty(tmp_path):
    assert worker_progress(tmp_path / "nope", 1) == []


def test_worker_progress_survives_a_partially_written_line(tmp_path):
    """Logs are read while the worker is still writing to them."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "gen0001-w00.log").write_text("iter    1  reward +1.00 bb  entro", encoding="utf-8")
    rows = worker_progress(tmp_path, 1)
    assert (rows[0].name, rows[0].iterations, rows[0].entropy) == ("w00", 1, "-")
    assert rows[0].train_bb100 == pytest.approx(100.0), (
        "the part of the line that is there still counts"
    )


def test_the_final_pool_summary_is_not_mistaken_for_a_rating(tmp_path):
    """A finished worker prints a leaderboard whose header reads
    "membro  rating  partite"; matching "rating " anywhere reads the column
    title 'partite' as the worker's rating."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "gen0001-w00.log").write_text(
        "iter    1  reward +1.00 bb  policy -0.01  value 1.0  entropy 0.300  kl 0.01  clip 0.1\n"
        "        eval vs pool: +10.0 bb/100  rating 1523\n"
        "\npool finale:\n"
        "membro                              rating  partite\n"
        "agent-x                               1540       12\n",
        encoding="utf-8",
    )
    row = worker_progress(tmp_path, 1)[0]
    assert (row.rating, row.eval, row.entropy) == ("1523", "+10.0", "0.300")


# ---- what stage a worker is in -------------------------------------------


def test_the_stage_is_whatever_the_log_last_said():
    """The watcher has no other channel: the supervisor is blocked in wait()."""
    assert parse_worker_log("w00", "").stage == STAGE_STARTING
    assert parse_worker_log("w00", "iter    1  reward +1.0 bb\n").stage == STAGE_TRAINING
    for stage in (EVALUATION, ELO_PLAY, ELO_MERGE, PRUNING, DONE):
        text = f"iter    1  reward +1.0 bb\n{marker(stage)}\n"
        assert parse_worker_log("w00", text).stage == stage


def test_a_result_line_puts_the_worker_back_into_training():
    """An evaluation ends by resuming training, so the stage must not stay
    stuck on it once its result has been printed."""
    text = (
        f"{marker(EVALUATION)}\n"
        "        eval vs pool: +10.0 bb/100  rating 1523\n"
    )
    assert parse_worker_log("w00", text).stage == STAGE_TRAINING


def test_the_elo_stages_follow_each_other():
    """The three stages of the population pass are distinct: playing thousands
    of hands, merging under per-model locks, and the rare pruning pass."""
    text = "\n".join(marker(s) for s in (ELO_PLAY, ELO_MERGE, PRUNING, ELO_MERGE, DONE))
    assert parse_worker_log("w00", text).stage == DONE
    assert parse_worker_log("w00", "\n".join(text.splitlines()[:3])).stage == PRUNING


def test_a_crashed_worker_is_reported_as_an_error_and_stays_that_way():
    """A traceback is sticky: whatever the log says afterwards, the run died."""
    text = (
        "iter    1  reward +1.0 bb  entropy 0.9\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "RuntimeError: boom\n"
    )
    assert parse_worker_log("w00", text).stage == STAGE_ERROR


def test_the_last_stretch_of_a_run_is_named_for_what_it_is():
    """At the target iteration count 'training' is really the wrap-up: saving,
    publishing and the pass against the frozen anchors. A stage the worker
    announces itself keeps its own name, whenever it happens."""
    done = parse_worker_log("w00", "iter 1\n" * 100)
    assert stage_label(done, 100) == "fine training"
    finishing = parse_worker_log("w00", "iter 1\n" * 100 + marker(SERIES))
    assert stage_label(finishing, 100) == "ancore"
    mid = parse_worker_log("w00", "iter 1\n" * 40)
    assert stage_label(mid, 100) == "training"


# ---- the reward a worker is getting ---------------------------------------


def header(hands):
    """The line every worker prints as it starts, which is what sizes the window."""
    return f"device cpu | 6 seats | {hands} hands/iteration"


def test_the_training_rate_averages_a_fixed_number_of_hands():
    """One iteration swings hugely, so the column averages a window -- and an
    early bad patch must not weigh on the current reading."""
    per_iteration = 1000
    fresh = REWARD_WINDOW_HANDS // per_iteration
    lines = [header(per_iteration)]
    lines += [f"iter {i:4}  reward   -9.00 bb  entropy 0.5" for i in range(200)]
    lines += [f"iter {i:4}  reward   +2.00 bb  entropy 0.5" for i in range(fresh)]

    row = parse_worker_log("w00", "\n".join(lines))

    assert row.train_bb100 == pytest.approx(200.0)
    assert row.train_hands == REWARD_WINDOW_HANDS


def test_the_window_is_the_same_number_of_hands_whatever_hands_was_drawn():
    """The point of the change: `--hands` is a swept axis now, so a window of N
    *iterations* covered 3,200 hands for one worker and 8,000 for another, and
    two rows of one table carried noise differing by 1.58x with nothing saying
    which was which."""
    widths = {}
    for per_iteration in (320, 512, 800):
        lines = [header(per_iteration)]
        lines += [f"iter {i:4}  reward   +1.00 bb  entropy 0.5" for i in range(500)]
        row = parse_worker_log("w00", "\n".join(lines))
        widths[per_iteration] = row.train_hands
        assert row.train_bb100 == pytest.approx(100.0)

    # Each covers at least the budget, and never more than one iteration past it.
    for per_iteration, hands in widths.items():
        assert REWARD_WINDOW_HANDS <= hands < REWARD_WINDOW_HANDS + per_iteration


def test_a_window_still_filling_is_shown_and_says_how_much_is_behind_it():
    """Blanking it would remove the "is this worker alive" signal for the first
    fifth of every run, which is what the column is read for minute to minute."""
    lines = [header(512)]
    lines += [f"iter {i:4}  reward   +1.00 bb  entropy 0.5" for i in range(10)]

    row = parse_worker_log("w00", "\n".join(lines))

    assert row.train_bb100 == pytest.approx(100.0)
    assert row.train_hands == 5_120  # well under the budget, and it says so


def test_a_log_with_no_header_averages_everything_and_claims_nothing():
    """Without the hand count there is no way to size a window in hands, so the
    honest answer is every iteration recorded -- and `train_hands` stays 0
    rather than asserting a precision it cannot back."""
    text = "\n".join(f"iter {i:4}  reward   +1.00 bb  entropy 0.5" for i in range(30))

    row = parse_worker_log("w00", text)

    assert row.train_bb100 == pytest.approx(100.0)
    assert row.train_hands == 0


def test_the_training_rate_is_converted_to_bb_per_100_hands():
    """`train.py` prints bb *per hand*; every win rate shown in this project is
    bb/100, so the conversion happens at the parse boundary. The log line's own
    unit is deliberately unchanged -- a worker launched before this change keeps
    writing the old lines, and they must not be read 100x wrong."""
    text = "\n".join(f"iter {i:4}  reward   +0.07 bb  entropy 0.5" for i in range(3))
    assert parse_worker_log("w00", text).train_bb100 == pytest.approx(7.0)


def test_a_worker_that_has_not_finished_an_iteration_has_no_training_rate():
    assert parse_worker_log("w00", "").train_bb100 is None
    assert parse_worker_log("w00", marker(EVALUATION)).train_bb100 is None


def test_the_training_rate_reads_negative_values():
    text = "\n".join(f"iter {i:4}  reward   -1.50 bb  entropy 0.5" for i in range(3))
    assert parse_worker_log("w00", text).train_bb100 == pytest.approx(-150.0)


def test_the_evaluation_result_is_kept_next_to_the_rating():
    """Both halves of the eval line are shown: the bb/100 is noisy but it is
    what the run measured, and the rating is what the model is published with."""
    row = parse_worker_log("w00", "        eval vs pool: -12.5 bb/100  rating 1487\n")
    assert (row.eval, row.rating) == ("-12.5", "1487")


# ---- the status table ------------------------------------------------------


def test_the_table_summarises_stages_and_training_rates(tmp_path):
    write_worker_log(tmp_path, 1, 0, 100, rating="1523", reward="+2.00")
    write_worker_log(tmp_path, 1, 1, 40, reward="-1.00")
    rows = worker_progress(tmp_path, 1)

    text = "\n".join(format_worker_table(rows, 100))

    assert "iterazioni 140/200" in text
    assert "+50.00 bb/100 su 2 worker" in text  # (+2.00 + -1.00) / 2, per 100 hands
    assert "+10.00 bb/100 su 1" in text  # only the worker that has evaluated
    assert "1523" in text and "w00" in text and "w01" in text
    assert "bench" not in text, "the per-worker benchmark column was removed"


def test_the_table_warns_about_a_worker_that_stopped_writing(tmp_path):
    """A killed worker leaves its log behind; without the age column its bar
    sits at 40/100 forever and reads as a slow worker rather than a dead one."""
    write_worker_log(tmp_path, 1, 0, 40)
    write_worker_log(tmp_path, 1, 1, 40)
    rows = worker_progress(tmp_path, 1, now=os.path.getmtime(tmp_path / "gen0001-w00.log") + 3600)

    text = "\n".join(format_worker_table(rows, 100))

    assert "ATTENZIONE" in text and "w00" in text


def test_a_finished_worker_is_not_reported_as_stalled(tmp_path):
    """A run that is over stops writing by definition."""
    (tmp_path / "gen0001-w00.log").write_text(
        "iter    1  reward +1.0 bb  entropy 0.9\n" + marker(DONE) + "\n", encoding="utf-8"
    )
    rows = worker_progress(tmp_path, 1, now=os.path.getmtime(tmp_path / "gen0001-w00.log") + 86400)
    assert "ATTENZIONE" not in "\n".join(format_worker_table(rows, 100))



# ---- inheritance ---------------------------------------------------------


def store_with_ranking(tmp_path, count: int, *, games: int = 10):
    """A shared store of `count` models and the ranking that rates them."""
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    ranking = PoolRegistry(directory=tmp_path / "global", max_models=10**9)
    for index in range(count):
        label = f"m{index:02d}"
        (models / f"{label}.pt").write_bytes(b"weights")
        ranking.members[label] = PoolMember(
            label=label, ref=f"{label}.pt", rating=2000.0 - index, games=games
        )
    return models, ranking


def test_every_worker_inherits_when_the_store_has_parents(tmp_path):
    """Without inheritance the loop only ever produces models exactly
    --iterations deep, and from-scratch runs trained worse in every group."""
    models, ranking = store_with_ranking(tmp_path, 30)
    plan = inheritance_plan(ranking, models, 24, rng=random.Random(0))
    assert all(parent is not None for parent in plan)


def test_inheritors_get_distinct_parents_from_the_top_of_the_ranking(tmp_path):
    models, ranking = store_with_ranking(tmp_path, 30)
    plan = inheritance_plan(ranking, models, 4, rng=random.Random(1), tiers=(10,))
    parents = [p.stem for p in plan if p is not None]
    assert len(parents) == 4 and len(set(parents)) == 4, "every parent is different"
    assert all(int(label[1:]) < 10 for label in parents), "only the top_n are eligible"


def test_the_parents_differ_from_one_generation_to_the_next(tmp_path):
    models, ranking = store_with_ranking(tmp_path, 60)
    rng = random.Random(7)
    first = inheritance_plan(ranking, models, 8, rng=rng)
    second = inheritance_plan(ranking, models, 8, rng=rng)
    assert first != second


def test_parents_are_recycled_when_there_are_more_inheritors_than_models(tmp_path):
    models, ranking = store_with_ranking(tmp_path, 2)
    plan = inheritance_plan(ranking, models, 8, rng=random.Random(0))
    assert sorted({p.stem for p in plan}) == ["m00", "m01"]
    assert all(p is not None for p in plan)


def test_an_empty_store_means_everyone_starts_from_scratch(tmp_path):
    """Generation 1 has nothing to inherit from."""
    ranking = PoolRegistry(directory=tmp_path / "global", max_models=10**9)
    assert inheritance_plan(ranking, tmp_path / "models", 4, rng=random.Random(0)) == [None] * 4


def test_a_model_that_was_never_rated_is_not_a_parent(tmp_path):
    models, ranking = store_with_ranking(tmp_path, 3, games=0)
    assert inheritance_plan(ranking, models, 4, rng=random.Random(0)) == [None] * 4


# ---- launching a worker ------------------------------------------------------


def launch_args(tmp_path, **overrides):
    values = {
        "workers": 4, "seed_base": 1000, "iterations": 5, "hands": 8,
        "table_weights": [0.0, 1.0, 0, 0, 0, 0, 0, 0], "stack_min_bb": 50.0, "stack_max_bb": 50.0,
        "sb": 1, "bb": 2, "lr": 3e-4, "device": "cpu",
        "hidden": 512, "num_layers": 3, "head_hidden": 256, "head_layers": 1, "equity_model": "e.pt",
        "models_dir": tmp_path / "models",
        "machine": "host-a", "pool_models": 20, "table_hands": 200, "concurrent_tables": 8, "allin_runouts": 20, "style_share": 0.0, "style_spread": 0.5, "style_temperature_spread": 0.2, "style_jitter": 0.1, "style_scales": [1.0] * 5, "critic_stack_power": 1.4, "pool_top_share": 0.5, "pool_top_n": 100,
        "ppo_epochs": 4, "clip_epsilon": 0.2, "eval_every": 10,
        "minibatch_size": 1024, "gae_lambda": 0.95, "value_coef": 0.5, "policy_max_grad_norm": 0.5, "critic_max_grad_norm": 1.0,
        "entropy_coef": 0.0, "k_schedule": "0:16, 100:2", "draw_tiers": "10, 100, 1000, all",
        "eval_sessions": 2,
        "benchmark_dir": tmp_path / "benchmark",
        "global_dir": tmp_path / "global", "global_lock_seconds": 120, "global_elo": False,
        "benchmark_sessions": 10, "benchmark_resident": 10, "benchmark_rotate_every": 50,
        "opponent_probability": 0.5,
        "elo_fill_in": False, "fill_deadline_minutes": 150.0,
        "fill_sessions": 10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def capture_launch(tmp_path, monkeypatch, *, inherit_from=None, hp=None):
    captured = {}

    class FakePopen:
        def __init__(self, command, **_kwargs):
            captured["command"] = command

    monkeypatch.setattr(loop_module.subprocess, "Popen", FakePopen)
    worker_dir = tmp_path / "work" / "gen0007-w02"
    launch_worker(
        launch_args(tmp_path), 2, 7, worker_dir, tmp_path / "logs" / "w.log", inherit_from, hp
    )
    return captured["command"], worker_dir


def test_a_worker_is_told_where_the_store_is_and_where_to_keep_its_scratch(tmp_path, monkeypatch):
    command, worker_dir = capture_launch(tmp_path, monkeypatch)

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--models-dir") == str(tmp_path / "models")
    assert value("--scratch-dir") == str(worker_dir)
    assert value("--archive-prefix") == "gen0007-w02"
    assert value("--machine") == "host-a"
    assert value("--global-dir") == str(tmp_path / "global")
    assert worker_dir.is_dir()  # created for it
    # The old per-machine pool is gone from the interface entirely.
    assert "--pool-dir" not in command and "--pool-size" not in command
    assert "--no-global-elo" in command


def test_an_inheriting_worker_resumes_from_a_copy_of_its_parent(tmp_path, monkeypatch):
    parent = tmp_path / "models" / "parent.pt"
    parent.parent.mkdir()
    parent.write_bytes(b"parent weights")

    command, worker_dir = capture_launch(tmp_path, monkeypatch, inherit_from=parent)

    assert "--resume" in command
    assert (worker_dir.parent / "agent-gen0007-w02.pt").read_bytes() == b"parent weights"
    assert parent.read_bytes() == b"parent weights"  # the store's file is only read


def test_an_inheriting_worker_is_told_which_parent_it_came_from(tmp_path, monkeypatch):
    parent = tmp_path / "models" / "m-gen0003-w01-agent-20260101-000000.pt"
    parent.parent.mkdir()
    parent.write_bytes(b"w")

    inherited = loop_module.perturb_hyperparameters(a_parent(), random.Random(1))
    command, _ = capture_launch(tmp_path, monkeypatch, inherit_from=parent, hp=inherited)

    assert command[command.index("--parent-label") + 1] == parent.stem


def test_a_worker_that_runs_the_fleets_own_settings_took_no_step_from_its_parent(tmp_path, monkeypatch):
    """Its settings are not its parent's perturbed, so recording a step would put
    the parent's lineage drift into the optimizer's evidence as if it were an effect."""
    parent = tmp_path / "models" / "m-gen0003-w01-agent-20260101-000000.pt"
    parent.parent.mkdir()
    parent.write_bytes(b"w")

    command, _ = capture_launch(tmp_path, monkeypatch, inherit_from=parent)

    assert "--resume" in command and "--parent-label" not in command


def test_every_worker_is_handed_the_fleets_network_shape(tmp_path, monkeypatch):
    """The shape is the fleet's, resolved from `config.toml` when the generation
    started, so a worker never has to read the file to know what to build."""
    command, _ = capture_launch(tmp_path, monkeypatch)
    for flag, value in (("--hidden", "512"), ("--num-layers", "3"),
                        ("--head-hidden", "256"), ("--head-layers", "1")):
        assert command[command.index(flag) + 1] == value


def test_a_worker_without_a_parent_starts_fresh(tmp_path, monkeypatch):
    command, _ = capture_launch(tmp_path, monkeypatch)
    assert "--resume" not in command
    assert "--parent-label" not in command


def test_a_parent_that_vanished_leaves_the_worker_fresh(tmp_path, monkeypatch):
    """`--resume` reads whether the parent file is still there at launch time."""
    command, _ = capture_launch(tmp_path, monkeypatch, inherit_from=tmp_path / "models" / "gone.pt")

    assert "--resume" not in command


# ---- the per-worker hyperparameter sweep -----------------------------------------


def a_parent(**overrides):
    """A parent's run metadata, sitting on the fleet's own default settings."""
    values = {
        "schema": loop_module.RUN_METADATA_VERSION,
        "lr": 3.0e-4,
        "hands": 512,
        "opponent_probability": 0.50,
        "ppo_epochs": 4,
        "clip_epsilon": 0.20,
        "pool_top_share": 0.50,
        "pool_top_n": 100,
        "minibatch_size": 1024,
        "gae_lambda": 0.95,
        "value_coef": 0.5,
        "policy_max_grad_norm": 0.5, "critic_max_grad_norm": 1.0,
        "entropy_coef": 0.0,
    }
    values.update(overrides)
    return values


def a_start(**overrides):
    """The fleet's own settings, as a loop run with no flag and no file has them."""
    return starting_hyperparameters(
        SimpleNamespace(**{**{axis: a_parent()[axis] for axis in HP_AXES}, **overrides})
    )


def test_the_default_flags_are_what_the_fleet_runs_today():
    """A worker with no parent starts a step around these, so today's production
    values are what the loop's own flags have to say."""
    args = loop_module.build_parser().parse_args(["--config", ""])
    start = starting_hyperparameters(args)
    assert start.lr == 3.0e-4
    assert start.hands == 512
    assert start.opponent_probability == 0.50
    assert start.ppo_epochs == 4
    assert start.clip_epsilon == 0.20
    assert start.pool_top_share == 0.50
    assert start.pool_top_n == 100
    assert start.arm == HP_ARM_SAMPLED


def test_an_inheriting_worker_multiplies_every_axis_by_one_of_the_multipliers():
    """PBT's x{0.8, 1.0, 1.25} with the user's own 1.2, applied to the parent's
    value."""
    parent = a_parent()
    for seed in range(40):
        child = perturb_hyperparameters(parent, random.Random(seed))
        assert child is not None
        for axis in HP_AXES:
            got, was = getattr(child, axis), parent[axis]
            allowed = [
                round(was * m) if isinstance(got, int) else was * m
                for m in HP_MULTIPLIERS
            ]
            if axis in loop_module._HP_COMPLEMENT_AXES:
                allowed = [1 - (1 - was) * m for m in HP_MULTIPLIERS]
            if axis in loop_module._HP_PROBABILITY_AXES:
                allowed = [min(v, 1.0) for v in allowed]
            if was == 0:
                allowed.append(loop_module._HP_SEED_FROM_ZERO[axis])
            assert any(got == pytest.approx(value) for value in allowed), (axis, got, allowed)
        assert child.arm == HP_ARM_INHERITED


def test_the_multipliers_are_the_ones_that_were_asked_for():
    """Pinned because the step is the one number the whole search rests on."""
    assert HP_MULTIPLIERS == (0.8, 1.0, 1.2)


def test_a_lineage_is_never_snapped_back_to_any_grid():
    """Nothing but selection bounds a lineage: a parent far from the defaults
    keeps moving from where it is, and can go on past where it already is."""
    far = a_parent(lr=9.9e-4, hands=2000)
    children = [perturb_hyperparameters(far, random.Random(seed)) for seed in range(40)]
    assert any(child.lr > 9.9e-4 for child in children)
    assert any(child.hands > 2000 for child in children)
    assert all(
        any(child.lr == pytest.approx(9.9e-4 * m) for m in HP_MULTIPLIERS) for child in children
    )


def test_a_probability_axis_is_capped_at_one():
    """The one limit the user kept. Both of these reach code that would do
    something odd with a probability above 1."""
    high = a_parent(opponent_probability=0.95, pool_top_share=1.0)
    children = [perturb_hyperparameters(high, random.Random(seed)) for seed in range(60)]
    assert all(child.opponent_probability <= 1.0 for child in children)
    assert all(child.pool_top_share <= 1.0 for child in children)
    # Capped, not rejected: the x1.2 draw off 1.0 still lands, at exactly 1.0.
    assert any(child.pool_top_share == 1.0 for child in children)


def test_a_count_axis_that_varies_varies_by_at_least_one():
    """Plain rounding left 2 absorbing in *both* directions (round(2 x 1.2) =
    round(2 x 0.8) = 2), and 2 is an ordinary `ppo_epochs` --
    measured before the fix, 62% of lineages were stuck there after 20
    generations and 96% after 200. So when rounding would not move the value, it
    steps by exactly one instead."""
    for value in (1, 2, 3, 4, 6, 10):
        parent = a_parent(ppo_epochs=value)
        got = {
            perturb_hyperparameters(parent, random.Random(seed)).ppo_epochs
            for seed in range(60)
        }
        assert value in got, "the x1.0 draw has to leave it alone"
        moved = got - {value}
        assert moved, f"{value} is absorbing"
        assert all(abs(other - value) >= 1 for other in moved)
        # Exactly one where rounding stalls, the true x1.2/x0.8 where it does not.
        assert got == {max(1, value - 1), value, value + 1} or value >= 10


def test_a_count_axis_never_reaches_zero():
    """`ppo_epochs` 0 is not a smaller setting, it is a run that never updates its
    policy. 1 is the floor and it is reflecting, not absorbing."""
    bottom = a_parent(hands=1, ppo_epochs=1, pool_top_n=1)
    children = [perturb_hyperparameters(bottom, random.Random(s)) for s in range(80)]
    assert all(
        min(c.hands, c.ppo_epochs, c.pool_top_n) >= 1 for c in children
    )
    assert any(c.ppo_epochs == 2 for c in children), "1 must be able to climb back"

    # And a count stays a count on the way down from the fleet's own value.
    assert {
        perturb_hyperparameters(a_parent(), random.Random(seed)).ppo_epochs
        for seed in range(40)
    } == {3, 4, 5}


def test_a_parent_whose_settings_are_unknown_is_not_inherited_from():
    """Nothing published before the metadata existed carries it, which today is
    every model on the volume, so this is the normal case and not an error."""
    assert perturb_hyperparameters(None, random.Random(0)) is None
    assert perturb_hyperparameters({}, random.Random(0)) is None
    # A schema this build does not know: mixing two meanings of one field is
    # worse than sampling.
    assert perturb_hyperparameters(a_parent(schema=999), random.Random(0)) is None
    # A metadata dict that is missing an axis the sweep needs.
    incomplete = a_parent()
    del incomplete["clip_epsilon"]
    assert perturb_hyperparameters(incomplete, random.Random(0)) is None


def test_every_worker_inherits_and_none_is_sampled_by_choice():
    """A worker that *can* inherit always does, whatever its index."""
    plan = hyperparameter_plan(24, [a_parent()] * 24, a_start(), rng=random.Random(0))
    arms = [hp.arm for hp in plan]
    assert arms.count(HP_ARM_INHERITED) == 24
    assert arms.count(HP_ARM_SAMPLED) == 0
    assert arms.count(HP_ARM_FALLBACK) == 0


def test_the_supervisor_has_no_knob_left_for_a_sampled_share():
    """`--hp-inherit-share` decided how much of a generation drew its own
    settings. It is gone with the arm, not defaulted to 1.0, so nothing can put
    the fleet back into two arms by passing a flag."""
    import inspect

    with pytest.raises(AssertionError):
        _argparse_default("pokerlab.rl.loop", "--hp-inherit-share")
    assert "inherit_share" not in inspect.signature(hyperparameter_plan).parameters


def test_a_worker_with_nothing_to_inherit_from_starts_from_the_fleets_settings():
    """The only non-inherited case left, and it is not a choice: with no parent
    metadata there is nothing of a parent's to perturb, so the worker perturbs the
    starting point the flags (and `config.toml`) give, and says so in its arm."""
    start = a_start(lr=1.0e-3, hands=300)
    plan = hyperparameter_plan(8, [None] * 8, start, rng=random.Random(0))
    arms = [hp.arm for hp in plan]
    assert arms.count(HP_ARM_INHERITED) == 0
    assert arms.count(HP_ARM_FALLBACK) == 8
    assert arms.count(HP_ARM_SAMPLED) == 0
    for hp in plan:
        assert any(hp.lr == pytest.approx(1.0e-3 * m) for m in HP_MULTIPLIERS)
        assert any(hp.hands == round(300 * m) for m in HP_MULTIPLIERS)
    # Perturbed independently, not all moved together.
    assert len({hp.lr for hp in plan}) > 1


def test_hyperparameters_are_inherited_only_from_the_weights_parent():
    """Perturbing the settings of a model this worker is not resuming from would
    attribute a configuration to a run that never had it."""
    parents = [a_parent(lr=1.9e-4), None] * 4
    plan = hyperparameter_plan(8, parents, a_start(), rng=random.Random(3))
    for worker, hp in enumerate(plan):
        if parents[worker] is None:
            assert hp.arm == HP_ARM_FALLBACK
        else:
            assert hp.arm == HP_ARM_INHERITED
            assert any(
                hp.lr == pytest.approx(1.9e-4 * m) for m in HP_MULTIPLIERS
            )


def test_the_plan_is_reproducible_from_its_seed():
    parents = [a_parent(), None, a_parent(), None]
    first = hyperparameter_plan(4, parents, a_start(), rng=random.Random(7))
    second = hyperparameter_plan(4, parents, a_start(), rng=random.Random(7))
    assert first == second


def test_a_worker_is_launched_with_the_values_drawn_for_it(tmp_path, monkeypatch):
    """The whole sweep rests on this: a value drawn and not forwarded is a
    generation of runs recorded as something they were not."""
    captured = {}

    class FakePopen:
        def __init__(self, command, **_kwargs):
            captured["command"] = command

    monkeypatch.setattr(loop_module.subprocess, "Popen", FakePopen)
    parent = tmp_path / "models" / "parent.pt"
    parent.parent.mkdir()
    parent.write_bytes(b"parent weights")
    hp = Hyperparameters(
        lr=4.7e-4,
        hands=800,
        opponent_probability=0.32,
        ppo_epochs=6,
        clip_epsilon=0.13,
        pool_top_share=0.78,
        pool_top_n=195,
        arm=HP_ARM_INHERITED,
    )
    launch_worker(
        launch_args(tmp_path), 2, 7, tmp_path / "work" / "gen0007-w02",
        tmp_path / "logs" / "w.log", parent, hp,
    )
    command = captured["command"]

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--lr") == str(4.7e-4)
    assert value("--hands") == "800"
    assert value("--opponent-probability") == "0.32"
    assert value("--ppo-epochs") == "6"
    assert value("--pool-top-share") == "0.78"
    assert value("--pool-top-n") == "195"
    assert value("--clip-epsilon") == "0.13"
    assert value("--hp-arm") == HP_ARM_INHERITED


def test_a_worker_without_a_plan_runs_the_fleets_own_settings(tmp_path, monkeypatch):
    """`args` stays the single definition of every default, so a hand-driven
    `poker-loop` behaves exactly as it did before the sweep existed."""
    command, _ = capture_launch(tmp_path, monkeypatch)

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--hands") == "8"  # launch_args' own value, not a drawn one
    assert value("--ppo-epochs") == "4"
    assert value("--hp-arm") == HP_ARM_SAMPLED


def test_the_settings_are_read_back_out_of_a_real_checkpoint(tmp_path):
    """The one place the two halves meet: `train.py` writes the metadata into
    the archive and the supervisor reads it off the same file a generation
    later. A silent failure here does not break anything visibly -- it quietly
    turns every worker of the generation into a restart from the fleet's own settings,
    which is the opposite of the search the fleet is supposed to be running."""
    pytest.importorskip("torch")
    from support import tiny_model

    from pokerlab.rl.ppo import save_checkpoint
    from pokerlab.rl.train import run_metadata

    args = SimpleNamespace(
        hp_arm=HP_ARM_SAMPLED, parent_label="", hidden=512, num_layers=3, head_hidden=256, head_layers=1, equity_model="e.pt", machine="host-a", seed=7, resume=True, iterations=100,
        hands=640, table_weights=[25.0, 20.0, 15.0, 10.0, 10.0, 10.0, 5.0, 5.0], stack_min_bb=1.0, stack_max_bb=100.0, sb=50, bb=100, lr=3.8e-4, ppo_epochs=5,
        clip_epsilon=0.25, entropy_coef=0.0, opponent_probability=0.62,
        pool_models=20, table_hands=200, concurrent_tables=8, pool_top_share=0.5, pool_top_n=100,
        minibatch_size=1024, gae_lambda=0.95, value_coef=0.5, policy_max_grad_norm=0.5, critic_max_grad_norm=1.0,
    )
    path = tmp_path / "agent.pt"
    save_checkpoint(path, tiny_model(), iteration=100, metadata=run_metadata(args))

    recovered = loop_module.read_run_metadata(path)
    child = perturb_hyperparameters(recovered, random.Random(0))
    assert child is not None and child.arm == HP_ARM_INHERITED
    assert any(child.lr == pytest.approx(3.8e-4 * m) for m in HP_MULTIPLIERS)
    assert any(child.hands == round(640 * m) for m in HP_MULTIPLIERS)


def test_a_model_published_before_the_metadata_existed_reads_as_nothing(tmp_path):
    """Every model on the volume today is one of these, so this is the path the
    first generation actually takes."""
    pytest.importorskip("torch")
    from support import tiny_model

    from pokerlab.rl.ppo import save_checkpoint

    path = tmp_path / "old.pt"
    save_checkpoint(path, tiny_model(), iteration=100)

    assert loop_module.read_run_metadata(path) is None
    assert loop_module.read_run_metadata(tmp_path / "missing.pt") is None
    assert loop_module.read_run_metadata(None) is None


def test_an_unreadable_parent_costs_one_fallback_and_not_the_supervisor(tmp_path):
    """~10,000 files on an NFS mount: a truncated or half-copied checkpoint must
    not end a supervisor that has been running for weeks."""
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    assert loop_module.read_run_metadata(broken) is None


def test_inherited_workers_are_visible_in_the_status(tmp_path):
    """An inheriting worker and a fresh one are not comparable at the same
    iteration count, so the status has to distinguish them."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "gen0001-w00.log").write_text(
        "resumed from checkpoints/work/agent-gen0001-w00.pt\n"
        "iter    1  reward +1.0 bb  policy -0.01  value 1.0  entropy 0.900  kl 0.01  clip 0.1\n",
        encoding="utf-8",
    )
    (tmp_path / "gen0001-w01.log").write_text(
        "iter    1  reward +1.0 bb  policy -0.01  value 1.0  entropy 0.900  kl 0.01  clip 0.1\n",
        encoding="utf-8",
    )
    rows = worker_progress(tmp_path, 1)
    assert [row.inherited for row in rows] == [True, False]


# ---- clearing what an interrupted run left behind --------------------------------


def _abandoned_generation(work, generation=155, worker=7, *, archive=True, rating=1633.0):
    """A worker directory as a killed loop leaves it: the final archive it
    trained (with the sidecar recording its rating) and the live checkpoint beside it."""
    name = f"gen{generation:04d}-w{worker:02d}"
    worker_dir = work / name
    worker_dir.mkdir(parents=True)
    if archive:
        (worker_dir / "agent-20260923-070159.pt").write_bytes(b"the model this worker trained")
        write_sidecar(worker_dir / "agent-20260923-070159.pt", rating=rating)
    live = work / f"agent-{name}.pt"
    live.write_bytes(b"x" * 1000)
    return worker_dir, live


def _no_processes(monkeypatch, commands=()):
    monkeypatch.setattr(loop_module, "_live_commands", lambda: list(commands))


def sweep(tmp_path, **kwargs):
    return sweep_stale_work(
        tmp_path / "work", tmp_path / "models", tmp_path / "global",
        machine="host-a", scratch_dir=tmp_path / "tmp", **kwargs,
    )


def test_the_sweep_removes_abandoned_work_and_publishes_the_workers_own_archive(tmp_path, monkeypatch):
    _no_processes(monkeypatch)
    work = tmp_path / "work"
    worker_dir, live = _abandoned_generation(work)
    (work / "arena-gen0155").mkdir()

    report = sweep(tmp_path)

    assert not worker_dir.exists() and not live.exists() and not (work / "arena-gen0155").exists()
    assert (report.work_dirs, report.checkpoints, report.arena_dirs) == (1, 1, 1)
    assert report.freed_bytes > 0
    # The trained model survives, named as the worker would have named it...
    published = tmp_path / "models" / "host-a-gen0155-w07-agent-20260923-070159.pt"
    assert published.read_bytes() == b"the model this worker trained"
    assert report.salvaged == 1
    # ...and enters the ranking at the rating its sidecar recorded, not the baseline.
    member = read_member(tmp_path / "global", published.stem)
    assert member is not None and member.rating == 1633.0 and member.games == 0


def test_the_sweep_never_touches_what_a_running_process_still_names(tmp_path, monkeypatch):
    work = tmp_path / "work"
    worker_dir, live = _abandoned_generation(work, generation=155, worker=7)
    other_dir, other_live = _abandoned_generation(work, generation=155, worker=8)
    # gen0155-w07 is an orphan that outlived its supervisor and is still training.
    _no_processes(
        monkeypatch,
        [f"python poker-train --scratch-dir {worker_dir} --checkpoint {live}"],
    )

    report = sweep(tmp_path)

    assert worker_dir.exists() and live.exists()
    assert not other_dir.exists() and not other_live.exists()
    assert report.in_use == 2  # its directory and its live checkpoint
    assert report.work_dirs == 1
    assert not (tmp_path / "models" / "host-a-gen0155-w07-agent-20260923-070159.pt").exists()


def test_the_sweep_does_nothing_when_liveness_cannot_be_established(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "_live_commands", lambda: None)
    work = tmp_path / "work"
    worker_dir, live = _abandoned_generation(work)

    report = sweep(tmp_path)

    assert worker_dir.exists() and live.exists()
    assert report.total == 0


def test_a_directory_whose_archive_could_not_be_published_is_kept(tmp_path, monkeypatch):
    _no_processes(monkeypatch)
    work = tmp_path / "work"
    worker_dir, _live = _abandoned_generation(work)

    def broken_copy(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pokerlab.rl.global_store.shutil.copy2", broken_copy)
    report = sweep(tmp_path)

    assert (worker_dir / "agent-20260923-070159.pt").exists()  # the model is not lost
    assert report.work_dirs == 0


def test_an_archive_already_in_the_store_is_not_published_twice(tmp_path, monkeypatch):
    _no_processes(monkeypatch)
    work = tmp_path / "work"
    _abandoned_generation(work)
    models = tmp_path / "models"
    models.mkdir()
    existing = models / "host-a-gen0155-w07-agent-20260923-070159.pt"
    existing.write_bytes(b"already published by the worker itself")

    report = sweep(tmp_path)

    assert report.salvaged == 0
    assert existing.read_bytes() == b"already published by the worker itself"


def test_a_worker_dir_without_an_archive_is_simply_removed(tmp_path, monkeypatch):
    _no_processes(monkeypatch)
    worker_dir, _live = _abandoned_generation(tmp_path / "work", archive=False)

    report = sweep(tmp_path)

    assert not worker_dir.exists()
    assert report.salvaged == 0
    assert not (tmp_path / "models").exists() or not list((tmp_path / "models").glob("*.pt"))


def test_the_sweep_leaves_unrelated_files_alone(tmp_path, monkeypatch):
    _no_processes(monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes.txt").write_text("keep")
    (work / "keep-me").mkdir()
    (work / "gen12-wxx").mkdir()  # not the shape this loop creates

    report = sweep(tmp_path)

    assert report.total == 0
    assert sorted(p.name for p in work.iterdir()) == ["gen12-wxx", "keep-me", "notes.txt"]


def test_the_sweep_removes_only_this_machines_stale_publish_partials(tmp_path, monkeypatch):
    _no_processes(monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    stale_mine = models / ".host-a-gen1-w00-agent-x.pt.partial"
    fresh_mine = models / ".host-a-gen2-w00-agent-y.pt.partial"
    stale_theirs = models / ".host-b-gen1-w00-agent-x.pt.partial"
    published = models / "host-a-gen0-w00-agent-z.pt"
    for path in (stale_mine, fresh_mine, stale_theirs, published):
        path.write_bytes(b"x")
    old = stale_mine.stat().st_mtime - 7200
    for path in (stale_mine, stale_theirs):
        os.utime(path, (old, old))

    report = sweep(tmp_path)

    assert report.partials == 1
    assert not stale_mine.exists()
    assert fresh_mine.exists()  # might be mid-write
    assert stale_theirs.exists()  # another host's file is never ours to remove
    assert published.exists()


def test_the_sweep_removes_stale_shard_scratch_from_a_killed_global_elo(tmp_path, monkeypatch):
    _no_processes(monkeypatch)
    scratch = tmp_path / "tmp"
    stale = scratch / "global-arena-shard-abc123"
    fresh = scratch / "global-arena-shard-def456"
    for directory in (stale, fresh):
        directory.mkdir(parents=True)
        (directory / "candidates.json").write_text("[]")
    old = stale.stat().st_mtime - 7200
    os.utime(stale, (old, old))

    report = sweep(tmp_path)

    assert report.scratch_dirs == 1
    assert not stale.exists() and fresh.exists()


# ---- the population pass's session count ---------------------------------


def _argparse_default(module_name: str, flag: str) -> str:
    """How `module_name`'s parser spells the default for `flag`, read from its
    source. Reading the source rather than the parsed value is the point: the
    bug being guarded against is a *literal* creeping back in beside the
    constant, which a value comparison would only catch while they still agree.
    """
    import ast
    import importlib.util

    source = Path(importlib.util.find_spec(module_name).origin).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.unparse(keyword.value)
    raise AssertionError(f"{module_name} has no {flag}")


def test_the_population_passes_session_count_is_defined_in_exactly_one_place():
    """It was not: the constant said 500 while both CLIs and
    `run_population_sessions` carried a hard-coded 12, so the constant described
    nothing that ever ran and the real number was invisible from it."""
    import inspect

    from pokerlab.rl.global_arena import DEFAULT_GLOBAL_SESSIONS, run_population_sessions

    signature = inspect.signature(run_population_sessions)
    assert signature.parameters["sessions"].default == DEFAULT_GLOBAL_SESSIONS
    for module in ("pokerlab.rl.train", "pokerlab.rl.loop"):
        assert _argparse_default(module, "--global-sessions") == "DEFAULT_GLOBAL_SESSIONS"


# ---- the benchmark pass in the status --------------------------------------


def benchmark_log(bb100, *, rating="1530", sessions=500, anchors=42):
    return "\n".join([
        "iter    1  reward +1.0 bb  entropy 0.9",
        marker(SERIES),
        (
            f"        benchmark: {bb100} bb/100 su {sessions} sessioni contro "
            f"{anchors} ancore, rating {rating} (era 1587 contro il pool, 600 partite)"
        ),
    ]) + "\n"


def test_the_benchmark_pass_result_is_read_off_one_line():
    """The pass draws its opponents from the whole frozen set, so it reports one
    number."""
    row = parse_worker_log("w00", benchmark_log("-3.7"))
    assert row.benchmark_bb100 == pytest.approx(-3.7)
    assert row.benchmark_rating == "1530"
    assert row.benchmark_sessions == 500


def test_a_worker_that_has_not_reached_the_benchmark_pass_has_none():
    row = parse_worker_log("w00", "iter    1  reward +1.0 bb  entropy 0.9\n")
    assert row.benchmark_bb100 is None and row.benchmark_rating == "-"
    assert row.benchmark_sessions == 0


def test_the_benchmark_pass_shows_running_figures_until_it_concludes():
    """While the pass plays, the two cells carry the latest running bb/100 and
    rating off the `avanzamento` lines, flagged provisional; the final result
    line then replaces them and clears the flag."""
    from pokerlab.rl.phases import progress_marker

    running = "\n".join([
        "iter 1",
        marker(SERIES),
        progress_marker(SERIES, 1, 500, detail="rating 1512, +3.0 bb/100"),
        progress_marker(SERIES, 40, 500, detail="rating 1534, -7.5 bb/100", eta_seconds=600),
    ]) + "\n"
    row = parse_worker_log("w00", running)
    assert row.benchmark_live
    assert row.benchmark_bb100 == -7.5 and row.benchmark_rating == "1534"
    assert row.benchmark_sessions == 40

    done = running + benchmark_log("-12.4")
    final = parse_worker_log("w00", done)
    assert not final.benchmark_live and final.benchmark_bb100 == -12.4


def test_a_worker_reporting_only_the_rating_still_fills_its_column():
    from pokerlab.rl.phases import progress_marker

    text = "iter 1\n" + marker(SERIES) + "\n" + progress_marker(SERIES, 9, 500, detail="rating 1531") + "\n"
    row = parse_worker_log("w00", text)
    assert row.benchmark_rating == "1531" and row.benchmark_live
    assert row.benchmark_bb100 is None


def test_the_benchmark_pass_is_its_own_stage():
    assert parse_worker_log("w00", "iter 1\n" + marker(SERIES)).stage == SERIES
    assert stage_label(parse_worker_log("w00", "iter 1\n" + marker(SERIES)), 100) == "ancore"


def test_the_table_carries_the_benchmark_columns(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "gen0001-w00.log").write_text(benchmark_log("-12.4"), encoding="utf-8")
    (tmp_path / "gen0001-w01.log").write_text(
        benchmark_log("+30.0", rating="1602"), encoding="utf-8"
    )

    text = "\n".join(format_worker_table(worker_progress(tmp_path, 1), 100))

    assert "ancore/100" in text and "elo ancore" in text
    assert "-12.4" in text and "+30.0" in text
    assert "1530" in text and "1602" in text


def test_the_benchmark_cells_are_empty_until_the_pass_has_run(tmp_path):
    write_worker_log(tmp_path, 1, 0, 40)
    text = "\n".join(format_worker_table(worker_progress(tmp_path, 1), 100))
    assert "ancore/100" in text, "the column is always there"
    assert "bb/100 contro ogni serie congelata" not in text, "the matrix is gone for good"


def test_the_status_no_longer_prints_the_generation_table(tmp_path, capsys):
    """--status is about what is happening now, not the generation history."""
    LoopState(
        generation=1,
        history=[{"generation": 1, "workers": 2, "published_models": 2, "benchmark_bb100": 1.0}],
    ).save(tmp_path / "loop_state.json")
    args = SimpleNamespace(
        state_dir=tmp_path, global_dir=tmp_path / "g", models_dir=tmp_path / "m",
        benchmark_dir=tmp_path / "b", log_dir=tmp_path / "l", iterations=100, top=15,
    )
    loop_module.print_status(args)
    out = capsys.readouterr().out
    assert "ultime generazioni" not in out
    assert "benchmark del migliore" not in out


# ---- per-iteration history, for the dashboard's charts ----------------------


def test_thin_keeps_the_ends_and_the_budget():
    from pokerlab.rl.monitor import thin

    assert thin(5, 10) == [0, 1, 2, 3, 4]      # nothing to thin
    picked = thin(1000, 100)
    assert len(picked) <= 101
    assert picked[0] == 0 and picked[-1] == 999
    assert picked == sorted(set(picked))


def test_parse_worker_history_reads_all_six_metrics():
    from pokerlab.rl.monitor import parse_worker_history

    text = """resumed from checkpoints/x.pt
iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  entropy 0.427  kl 0.0242  clip 0.148
iter    2  reward   -0.15 bb  policy +0.0078  value    0.044  entropy 0.440  kl 0.0069  clip 0.063
        eval vs pool: +10.0 bb/100  rating 1519"""
    history = parse_worker_history("w01", text)
    assert history.iterations == [1, 2]
    assert history.train_bb100 == pytest.approx([54.0, -15.0])
    assert history.policy == [-0.0078, 0.0078]
    assert history.value == [0.039, 0.044]
    assert history.entropy == [0.427, 0.440]
    assert history.kl == [0.0242, 0.0069]
    assert history.clip == [0.148, 0.063]
    # Sparse readings are anchored to the iteration they were taken at.
    assert history.eval_bb100 == [[2, 10.0]]
    assert history.eval_rating == [[2, 1519.0]]


def test_parse_worker_history_reads_the_value_target_lines():
    from pokerlab.rl.monitor import parse_worker_history

    text = """iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  entropy 0.427  kl 0.0242  clip 0.148
valore per tavolo: 2 sd 0.270 ev -0.50 n 566 | 9 sd 0.674 ev -0.08 n 515  (spread 2.5x)
valore per stack (bb): <10 sd 0.083 ev -4.12 n 281 | 30+ sd 0.669 ev n/a n 2666  (spread 8.1x)
iter    2  reward   -0.15 bb  policy +0.0078  value    0.044  entropy 0.440  kl 0.0069  clip 0.063
iter   10  reward   +0.10 bb  policy +0.0078  value    0.044  entropy 0.440  kl 0.0069  clip 0.063
valore per tavolo: 2 sd 0.300 ev -0.10 n 500 | 9 sd 0.600 ev -0.05 n 400  (spread 2.0x)
valore per stack (bb): <10 sd 0.080 ev -2.00 n 280 | 30+ sd 0.600 ev +0.02 n 2600  (spread 7.5x)
valore per strada: preflop sd 0.500 ev +0.05 n 2000 | river sd 0.900 ev +0.60 n 300  (spread 1.8x)"""
    history = parse_worker_history("w01", text)
    # Each reading is anchored to the iteration printed just before it.
    assert history.value_sd_size == {"2": [[1, 0.27], [10, 0.3]], "9": [[1, 0.674], [10, 0.6]]}
    assert history.value_sd_stack["<10"] == [[1, 0.083], [10, 0.08]]
    # An n/a explained variance is left out of the series rather than plotted as 0.
    assert history.value_ev_stack == {"<10": [[1, -4.12], [10, -2.0]], "30+": [[10, 0.02]]}
    assert history.value_spread_size == [[1, 2.5], [10, 2.0]]
    assert history.value_spread_stack == [[1, 8.1], [10, 7.5]]
    # By street: sd and explained variance, and its spread is not mistaken for another kind's.
    assert history.value_sd_street == {"preflop": [[10, 0.5]], "river": [[10, 0.9]]}
    assert history.value_ev_street == {"preflop": [[10, 0.05]], "river": [[10, 0.6]]}


def test_parse_worker_history_reads_the_gradient_lines():
    from pokerlab.rl.monitor import parse_worker_history

    text = """iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  entropy 0.427  kl 0.0242  clip 0.148
gradienti: policy media 0.8000 sd 0.2000 tagliati 75.0% soglia 0.5 | critico media 0.1000 sd 0.0500 tagliati 0.0% soglia 1 [40 passi]
iter    2  reward   -0.15 bb  policy +0.0078  value    0.044  entropy 0.440  kl 0.0069  clip 0.063
iter    3  reward   -0.15 bb  policy +0.0078  value    0.044  entropy 0.440  kl 0.0069  clip 0.063
gradienti: policy media 0.3000 sd 0.1000 tagliati 10.0% soglia 0.65 | critico media 0.2000 sd 0.0400 tagliati 5.0% soglia 0.25 [40 passi]
gradienti: policy media 0.3000 sd 0.1000 tagliati 10.0% soglia 0.65 | critico media 0.2000 sd 0.04"""
    history = parse_worker_history("w01", text)
    # Anchored to the iteration printed just before; an iteration without a line has no
    # point, and a cut line is not read at all.
    assert history.grad_policy_mean == [[1, 0.8], [3, 0.3]]
    assert history.grad_policy_sd == [[1, 0.2], [3, 0.1]]
    assert history.grad_policy_clipped == [[1, 0.75], [3, 0.1]]
    assert history.grad_critic_mean == [[1, 0.1], [3, 0.2]]
    assert history.grad_critic_clipped == [[1, 0.0], [3, 0.05]]
    assert (history.grad_policy_threshold, history.grad_critic_threshold) == (0.65, 0.25)
    assert parse_worker_history("w01", text.splitlines()[0]).grad_policy_threshold is None


def test_parse_worker_history_reads_the_style_lines():
    from pokerlab.rl.monitor import parse_worker_history

    text = """iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  entropy 0.427  kl 0.0242  clip 0.148
stile (1100 mani): [vpip 440/1100 | pfr 220/1100 | three_bet 0/0]
iter   10  reward   +0.10 bb  policy +0.0078  value    0.044  entropy 0.440  kl 0.0069  clip 0.063
stile (9900 mani): [vpip 2970/9900 | pfr 1980/9900 | three_bet 50/500]
stile (9900 mani): [vpip 2970/99"""
    history = parse_worker_history("w01", text)
    # Each reading is anchored to the iteration printed just before it; a statistic
    # with no opportunity yet has no point, and a cut line is not read at all.
    assert history.style_rate["vpip"] == [[1, 0.4], [10, 0.3]]
    assert history.style_rate["pfr"] == [[1, 0.2], [10, 0.2]]
    assert history.style_rate["three_bet"] == [[10, 0.1]]
    assert history.style_latest["three_bet"] == [50, 500]
    assert history.style_hands == 9900


def test_parse_worker_history_reads_the_style_of_each_group_of_table_sizes():
    from pokerlab.rl.monitor import parse_worker_history

    text = """iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  entropy 0.427  kl 0.0242  clip 0.148
stile (1100 mani): [vpip 440/1100 | pfr 220/1100]
stile 2-3 (300 mani): [vpip 210/300 | pfr 150/300]
stile 7-9 (800 mani): [vpip 160/800 | pfr 0/0]
iter   10  reward   +0.10 bb  policy +0.0078  value    0.044  entropy 0.440  kl 0.0069  clip 0.063
stile 2-3 (900 mani): [vpip 540/900 | pfr 360/900]"""
    history = parse_worker_history("w01", text)
    # The pooled line is still the pooled style; each group has its own series.
    assert history.style_rate["vpip"] == [[1, 0.4]] and history.style_hands == 1100
    assert history.style_by_size["2-3"]["rate"]["vpip"] == [[1, 0.7], [10, 0.6]]
    assert history.style_by_size["2-3"]["latest"]["pfr"] == [360, 900]
    assert history.style_by_size["2-3"]["hands"] == 900
    assert history.style_by_size["7-9"]["rate"] == {"vpip": [[1, 0.2]]}  # no chance at pfr yet
    assert "4-6" not in history.style_by_size


def test_a_log_without_style_lines_has_no_style():
    from pokerlab.rl.monitor import parse_worker_history

    history = parse_worker_history(
        "w01", "iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  entropy 0.427  kl 0.0242  clip 0.148"
    )
    assert history.style_rate == {} and history.style_latest == {} and history.style_hands == 0
    assert history.style_by_size == {}


def test_a_log_without_value_lines_has_empty_value_series():
    from pokerlab.rl.monitor import parse_worker_history

    text = "iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  entropy 0.427  kl 0.0242  clip 0.148"
    history = parse_worker_history("w01", text)
    assert history.value_sd_size == {} and history.value_sd_stack == {}
    assert history.value_spread_size == [] and history.value_spread_stack == []


def test_parse_worker_history_drops_a_half_written_final_line():
    """The log is read while the worker is still appending to it."""
    from pokerlab.rl.monitor import parse_worker_history

    good = (
        "iter    1  reward   +0.54 bb  policy -0.0078  value    0.039  "
        "entropy 0.427  kl 0.0242  clip 0.148"
    )
    history = parse_worker_history("w01", good + "\niter    2  reward   +0.1 bb  poli")
    assert history.iterations == [1]
    assert history.total_iterations == 1



def test_every_cli_can_render_its_own_help(monkeypatch, capsys):
    """`--help` must not raise, which is not free: argparse runs every help
    string through `help % params`, so a literal `%` has to be doubled. A help
    text reading "a 95% interval" parses `% i` as an integer conversion and
    `--help` dies with a TypeError -- which happened, and no test caught it
    because nothing ever asked for the help.
    """
    import pokerlab.rl.benchmark_arena as benchmark_arena_module
    import pokerlab.rl.dashboard as dashboard_module
    import pokerlab.rl.killswitch as killswitch_module
    import pokerlab.rl.population_arena as population_arena_module
    import pokerlab.rl.train as train_module

    # train.main and loop.main read sys.argv; killswitch.main and dashboard.main
    # take argv. Both shapes are driven through sys.argv here.
    for module, script in (
        (train_module, "poker-train"),
        (loop_module, "poker-loop"),
        (killswitch_module, "poker-kill"),
        (dashboard_module, "poker-dashboard"),
        (population_arena_module, "poker-elo"),
        (benchmark_arena_module, "benchmark_arena"),
    ):
        monkeypatch.setattr("sys.argv", [script, "--help"])
        with pytest.raises(SystemExit) as exit_info:
            module.main()
        assert exit_info.value.code == 0, script
        assert capsys.readouterr().out.startswith("usage:"), script


def test_the_status_shows_the_rating_of_the_field_a_worker_drew():
    """`eval` and `rating` say how the learner did; this says against whom. Every
    worker draws its own pool, so without it two rows of the same table are not
    comparable."""
    from pokerlab.rl.monitor import format_worker_table, parse_worker_log

    row = parse_worker_log(
        "w01", "pool rating: media 1503 min 1421 max 1612\niter    1  reward   +0.1 bb"
    )
    assert row.pool_rating == "1503"
    assert "1503" in "\n".join(format_worker_table([row], 10))
    # A log from before the line existed, or a run against an empty store.
    assert parse_worker_log("w01", "iter    1  reward   +0.1 bb").pool_rating == "-"


# ---- progress inside the two long, silent stages ---------------------------
#
# The population pass and the benchmark pass take the better part of an hour
# each and print, between them, four lines. These tests pin the contract that
# lets `--status` and the dashboard say how far in a worker is.


def test_a_progress_line_round_trips():
    """One format, written in `phases.py` and parsed in `phases.py`: the worker
    that prints and the watcher that reads cannot drift apart."""
    line = progress_marker(ELO_PLAY, 1250, 2750, detail="258 sessioni", eta_seconds=2280)
    reported = parse_progress(line)
    assert (reported.stage, reported.done, reported.total) == (ELO_PLAY, 1250, 2750)
    assert (reported.detail, reported.eta_seconds) == ("258 sessioni", 2280.0)
    assert reported.percent == 45


def test_a_progress_line_without_an_eta_still_parses():
    """The first report of a stage has nothing to extrapolate from yet."""
    reported = parse_progress(progress_marker(SERIES, 0, 150))
    assert (reported.done, reported.total, reported.eta_seconds) == (0, 150, None)
    assert reported.percent == 0


def test_a_truncated_progress_line_is_not_a_progress_line():
    """The logs are read while the worker is appending to them, so the final
    line can be cut anywhere."""
    full = progress_marker(ELO_PLAY, 1250, 2750, detail="258 sessioni", eta_seconds=2280)
    assert parse_progress(full[:20]) is None
    assert parse_progress("iter    1  reward +1.0 bb") is None


def test_the_stage_cell_carries_the_progress_and_the_eta():
    text = (
        "iter    1  reward +1.0 bb  entropy 0.9\n"
        + marker(ELO_PLAY)
        + "\n"
        + progress_marker(ELO_PLAY, 1250, 2750, detail="258 sessioni", eta_seconds=2280)
        + "\n"
    )
    row = parse_worker_log("w00", text)
    assert row.progress.percent == 45
    # `stage_label` keeps answering "what is it doing"; the cell adds "how far".
    assert stage_label(row, 100) == "elo: gioco"
    assert stage_cell(row, 100) == "elo: gioco 45% ~38m"


def test_progress_is_dropped_when_the_next_stage_starts():
    """Otherwise the percentage of the stage just finished is shown against the
    stage just started, which is worse than showing nothing."""
    text = (
        marker(SERIES)
        + "\n"
        + progress_marker(SERIES, 150, 150, detail="serie 3/3")
        + "\n"
        + marker(ELO_PLAY)
        + "\n"
    )
    row = parse_worker_log("w00", text)
    assert row.stage == ELO_PLAY and row.progress is None
    assert stage_cell(row, 100) == "elo: gioco"


def test_a_worker_outside_those_stages_reports_no_progress():
    row = parse_worker_log("w00", "iter    1  reward +1.0 bb  entropy 0.9\n")
    assert row.progress is None
    assert stage_cell(row, 100) == stage_label(row, 100)


def test_the_finishing_line_reports_the_slowest_worker(tmp_path):
    """A generation ends when its slowest worker does, so the machine's own
    "how much longer" is the longest ETA, not the mean."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for worker, (done, eta) in enumerate([(1375, 1200), (550, 3600)]):
        (tmp_path / f"gen0001-w{worker:02d}.log").write_text(
            marker(ELO_PLAY)
            + "\n"
            + progress_marker(ELO_PLAY, done, 2750, eta_seconds=eta)
            + "\n",
            encoding="utf-8",
        )
    rows = worker_progress(tmp_path, 1)

    line = format_finishing_line(rows)[0]

    assert "2/2 worker nelle passate finali" in line
    assert "avanzamento medio 35%" in line  # (50 + 20) / 2
    assert "~1h00" in line  # the slower of the two, not the mean
    assert line in "\n".join(format_worker_table(rows, 100))


def test_there_is_no_finishing_line_before_anyone_is_finishing(tmp_path):
    """Most of a generation is spent training; an empty line every time would
    be noise in the one view that has to stay readable."""
    write_worker_log(tmp_path, 1, 0, 40)
    rows = worker_progress(tmp_path, 1)
    assert format_finishing_line(rows) == []
    assert "nelle passate finali" not in "\n".join(format_worker_table(rows, 100))


# ---- settling the anchors after one is added ---------------------------------


def arena_args(tmp_path):
    return SimpleNamespace(
        machine="host-a", workers=6, global_root=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs", config=None,
    )


def run_arena(tmp_path, monkeypatch, global_dir, *, exit_code=0, config=None):
    """Drive `run_requested_benchmark_arena` with the arena itself stubbed."""
    commands = []

    def fake_call(command, **kwargs):
        commands.append(command)
        return exit_code

    monkeypatch.setattr(loop_module.subprocess, "call", fake_call)
    args = arena_args(tmp_path)
    args.config = config
    ran = loop_module.run_requested_benchmark_arena(args, global_dir, 7)
    return ran, (commands[0] if commands else None)


def test_no_arena_runs_when_nothing_was_added(tmp_path, monkeypatch):
    ran, command = run_arena(tmp_path, monkeypatch, tmp_path / "global")
    assert ran is False and command is None


def test_a_request_runs_the_arena_and_leaves_its_parameters_to_the_config_file(
    tmp_path, monkeypatch
):
    """Convergence (in sessions) and session length are `config.toml`'s to decide: a flag
    here would beat the file, which is exactly what makes a file decoration."""
    global_dir = tmp_path / "global"
    request_benchmark_arena(global_dir, added=["m1"], machine="host-b")

    ran, command = run_arena(tmp_path, monkeypatch, global_dir)

    assert ran is True

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--workers") == "6"  # the supervisor's own worker count
    for flag in ("--arena-tolerance", "--arena-window", "--arena-min-sessions",
                 "--arena-max-sessions", "--session-hands", "--k"):
        assert flag not in command, flag
    assert "--config" not in command, "no file named: the arena finds the default one"
    assert not list(global_dir.glob(f"{ARENA_REQUEST_FILENAME}*"))


def test_the_arena_reads_the_same_file_as_the_supervisor(tmp_path, monkeypatch):
    global_dir = tmp_path / "global"
    request_benchmark_arena(global_dir, added=["m1"], machine="host-b")

    _, command = run_arena(tmp_path, monkeypatch, global_dir, config="/etc/other.toml")

    assert command[command.index("--config") + 1] == "/etc/other.toml"


def test_two_supervisors_cannot_both_claim_one_request(tmp_path, monkeypatch):
    global_dir = tmp_path / "global"
    request_benchmark_arena(global_dir, added=["m1"], machine="host-b")
    first, _ = run_arena(tmp_path, monkeypatch, global_dir)
    second, _ = run_arena(tmp_path, monkeypatch, global_dir)
    assert (first, second) == (True, False)


def test_a_failed_arena_is_not_retried_but_leaves_its_claim_behind(tmp_path, monkeypatch):
    global_dir = tmp_path / "global"
    request_benchmark_arena(global_dir, added=["m1"], machine="host-b")
    ran, _ = run_arena(tmp_path, monkeypatch, global_dir, exit_code=1)
    assert ran is True
    assert list(global_dir.glob(f"{ARENA_REQUEST_FILENAME}.claim-*.failed-*"))
    again, _ = run_arena(tmp_path, monkeypatch, global_dir)
    assert again is False


def test_run_sh_leaves_the_worker_count_to_the_loop():
    """`run.sh` no longer decides how many workers: it passes `--workers 0`
    (or `POKER_WORKERS`), and the fleet-wide ceiling comes from `config.toml`.
    A ceiling declared in the script again would be a second source of truth."""
    script = (Path(__file__).resolve().parents[2] / "run.sh").read_text(encoding="utf-8")
    assert "DEFAULT_WORKER_CEILING" not in script
    assert 'WORKERS="${POKER_WORKERS:-0}"' in script


def test_a_small_machine_is_held_by_cores_and_memory_not_the_ceiling(monkeypatch):
    auto = loop_module.auto_workers
    assert auto(25, cores=32, free_mb=100_000) == 25  # big box: the ceiling
    assert auto(25, cores=8, free_mb=100_000) == 6  # cores - 2
    assert auto(25, cores=32, free_mb=3000 + 700 * 4) == 4  # memory
    assert auto(25, cores=1, free_mb=100_000) == 1  # never below one
    assert auto(25, cores=32, free_mb=500) == 1
    # `free_mb=None` means "read it", so an unreadable memory is simulated at the source.
    monkeypatch.setattr(loop_module, "available_memory_mb", lambda: None)
    assert auto(25, cores=32) == 25  # memory unreadable: only cores and ceiling count


def test_workers_zero_resolves_through_the_ceiling(monkeypatch, tmp_path):
    monkeypatch.setattr(loop_module, "available_memory_mb", lambda: 100_000)
    monkeypatch.setattr(loop_module.os, "cpu_count", lambda: 32)
    (tmp_path / "equity.pt").write_bytes(b"")
    args, _ = loop_module.load_args(
        ["--workers", "0", "--config", "", "--equity-model", str(tmp_path / "equity.pt")]
    )
    assert loop_module.resolve_workers(args).workers == 25
    args.worker_ceiling = 10
    assert loop_module.resolve_workers(args).workers == 10
    args.workers = 7  # an explicit count is left alone
    assert loop_module.resolve_workers(args).workers == 7


# ---- the Elo fill-in handshake ---------------------------------------------


class FakeWorker:
    """A `Popen` as `wait_for_workers` uses it: something that can be polled."""

    def __init__(self, *, exits_after=0, code=0):
        self._left = exits_after
        self._code = code

    def poll(self):
        if self._left > 0:
            self._left -= 1
            return None
        return self._code


def fill_processes(tmp_path, count, **kwargs):
    processes = []
    for worker in range(count):
        worker_dir = tmp_path / f"gen0001-w{worker:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        processes.append((worker, worker_dir, FakeWorker(**kwargs)))
    return processes


def drain(worker_dir):
    (worker_dir / loop_module.FILL_DRAINING_FILENAME).touch()


def test_the_supervisor_releases_workers_that_are_only_filling_time(tmp_path):
    """The deadlock this exists to break: a filling worker waits for the
    supervisor to say the generation is over, and the supervisor would otherwise
    wait for that worker to exit."""
    processes = fill_processes(tmp_path, 3, exits_after=3)
    for _worker, worker_dir, _process in processes:
        drain(worker_dir)
    stop = tmp_path / "state" / "FILL_STOP"

    failures = wait_for_workers(processes, fill_stop=stop, sleep=lambda _s: None)

    assert failures == []
    assert stop.exists()


def test_the_flag_waits_for_the_last_worker_still_really_working(tmp_path):
    """One worker still training means the generation is not over, however many
    of the others are idling."""
    processes = fill_processes(tmp_path, 3, exits_after=2)
    drain(processes[0][1])
    drain(processes[1][1])
    seen = []
    stop = tmp_path / "state" / "FILL_STOP"

    wait_for_workers(processes, fill_stop=stop, sleep=lambda _s: seen.append(stop.exists()))

    # The flag went up only once the third worker had exited -- that is, never
    # while the supervisor was still waiting on real work.
    assert seen[0] is False
    assert not stop.exists()


def test_nothing_is_signalled_when_the_phase_is_off(tmp_path):
    processes = fill_processes(tmp_path, 2, exits_after=2)
    for _worker, worker_dir, _process in processes:
        drain(worker_dir)

    wait_for_workers(processes, fill_stop=None, sleep=lambda _s: None)

    assert not (tmp_path / "state" / "FILL_STOP").exists()


def test_a_worker_that_died_is_still_reported(tmp_path):
    """Polling replaced `process.wait()`, and the failure count is what the
    generation record and the fleet-outage diagnosis both read."""
    processes = fill_processes(tmp_path, 3, exits_after=1)
    processes[1] = (1, processes[1][1], FakeWorker(exits_after=1, code=2))

    failures = wait_for_workers(processes, fill_stop=None, sleep=lambda _s: None)

    assert failures == [(1, 2)]


def test_the_supervisor_forwards_the_stop_file_to_each_worker(tmp_path, monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, command, **_kwargs):
            captured["command"] = command

    monkeypatch.setattr(loop_module.subprocess, "Popen", FakePopen)
    stop = tmp_path / "state" / "FILL_STOP"
    launch_worker(
        launch_args(tmp_path, elo_fill_in=True), 2, 7, tmp_path / "work" / "gen0007-w02",
        tmp_path / "logs" / "w.log", None, None, stop,
    )
    command = captured["command"]

    assert "--elo-fill-in" in command
    assert command[command.index("--fill-stop-file") + 1] == str(stop)
    assert command[command.index("--fill-sessions") + 1] == "10"


def test_a_worker_is_told_nothing_about_filling_when_it_is_off(tmp_path, monkeypatch):
    command, _ = capture_launch(tmp_path, monkeypatch)
    assert "--elo-fill-in" not in command and "--fill-stop-file" not in command


def test_the_cap_is_only_ever_expressed_in_minutes():
    """At the user's decision: a maximum number of passes would cap the work,
    and what has to be bounded is how long a worker holds its core."""
    assert _argparse_default("pokerlab.rl.loop", "--fill-deadline-minutes") == (
        "DEFAULT_FILL_DEADLINE_MINUTES"
    )
    assert _argparse_default("pokerlab.rl.train", "--fill-deadline-minutes") == (
        "DEFAULT_FILL_DEADLINE_MINUTES"
    )
    source = Path(
        importlib.util.find_spec("pokerlab.rl.train").origin
    ).read_text(encoding="utf-8")
    assert "DEFAULT_FILL_DEADLINE_MINUTES = 150" in source
    assert "max-passes" not in source


def test_the_two_cli_defaults_for_the_phase_agree():
    """The supervisor forwards these, so a disagreement would be invisible: the
    worker would silently run the loop's number and the help would say another."""
    for flag in ("--fill-deadline-minutes", "--fill-sessions"):
        assert _argparse_default("pokerlab.rl.loop", flag) == _argparse_default(
            "pokerlab.rl.train", flag
        )


def test_only_the_supervisor_turns_the_phase_on():
    """A hand-run `poker-train` has nobody to wait for, so it must end when it
    ends rather than fill until its deadline."""
    assert _argparse_default("pokerlab.rl.train", "--elo-fill-in") == "False"
    assert _argparse_default("pokerlab.rl.loop", "--elo-fill-in") == "True"


def test_a_filling_worker_is_not_what_the_generation_is_waiting_for():
    """Its ETA is its own safety cap -- hours -- and being the longest it would
    become the answer to "how much longer" for the whole machine."""
    filling = parse_worker_log(
        "w00",
        "iter    1  reward +1.0 bb  entropy 0.9\n"
        + marker(ELO_FILL)
        + "\n"
        + progress_marker(ELO_FILL, 20, 50, detail="passata 2, 20 sessioni giocate")
        + "\n",
        age=5.0,
    )
    working = parse_worker_log(
        "w01",
        "iter    1  reward +1.0 bb  entropy 0.9\n"
        + marker(ELO_PLAY)
        + "\n"
        + progress_marker(ELO_PLAY, 100, 400, detail="", eta_seconds=1800)
        + "\n",
        age=5.0,
    )

    line = format_finishing_line([filling, working])[0]
    assert "1/2 worker nelle passate finali" in line
    assert "~30m" in line
    assert "1 in riempimento elo" in line


def test_with_everyone_filling_the_line_says_so_and_quotes_no_eta():
    filling = parse_worker_log(
        "w00", "iter 1  reward +1.0 bb\n" + marker(ELO_FILL) + "\n", age=5.0
    )
    assert format_finishing_line([filling]) == [
        "fine  : 1/1 worker in riempimento elo, in attesa degli altri"
    ]


# ---- the settings a worker was given, visible while it runs -----------------


def hp_log(**overrides):
    values = {
        "hp_arm": HP_ARM_SAMPLED, "lr": 3.8e-4, "hands": 640, "ppo_epochs": 5,
        "clip_epsilon": 0.25, "opponent_probability": 0.62,
    }
    values.update(overrides)
    return (
        "iter    1  reward +1.0 bb  entropy 0.9\n"
        + hyperparameters_marker(values)
        + "\n"
    )


def test_a_worker_reports_what_it_was_configured_with(tmp_path):
    """The values exist on the command line, in the supervisor's log and in the
    published checkpoint -- and a watcher reads none of those three. It reads
    the worker's log, so the worker has to say it there."""
    row = parse_worker_log("w00", hp_log(), age=1.0)

    assert row.hyperparameters["hp_arm"] == HP_ARM_SAMPLED
    assert row.hyperparameters["lr"] == "0.00038"
    assert row.hyperparameters["hands"] == "640"
    assert row.hyperparameters["clip_epsilon"] == "0.25"


def test_every_axis_of_the_sweep_is_one_the_worker_reports():
    """The guard that matters: add an axis to `HP_AXES` and this fails until
    the worker prints it, because a run whose configuration cannot be recovered
    from its own log is a run whose outcome cannot be attributed to anything."""
    from pokerlab.rl.train import REPORTED_AXES

    assert set(HP_AXES) <= set(REPORTED_AXES)
    assert "hp_arm" in REPORTED_AXES


def test_a_log_from_before_the_sweep_records_nothing_and_breaks_nothing():
    """Every log on the volume was one of these the day it landed, so "not
    recorded" has to read as normal rather than as an error."""
    row = parse_worker_log("w00", "iter    1  reward +1.0 bb  entropy 0.9\n", age=1.0)

    assert row.hyperparameters == {}
    assert format_sweep_line([row]) == []


def test_the_status_summarises_how_the_generation_is_split():
    """The per-worker table has no room for six more columns, so the terminal
    gets the summary and the dashboard's panel gets the detail."""
    rows = [
        parse_worker_log("w00", hp_log(lr=1.9e-4, hands=320), age=1.0),
        parse_worker_log("w01", hp_log(lr=4.7e-4, hands=800), age=1.0),
        parse_worker_log(
            "w02", hp_log(hp_arm=HP_ARM_INHERITED, lr=3.0e-4, hands=512), age=1.0
        ),
    ]

    line = format_sweep_line(rows)[0]

    assert "2 campionati" in line and "1 ereditati" in line
    assert "lr 0.00019-0.00047" in line
    assert "mani 320-800" in line


def test_the_ranges_are_ordered_by_value_not_as_text():
    """The values are kept as the strings the run used, and "1024" sorts before
    "320" as text."""
    rows = [
        parse_worker_log("w00", hp_log(hands=320), age=1.0),
        parse_worker_log("w01", hp_log(hands=1024), age=1.0),
    ]

    assert "mani 320-1024" in format_sweep_line(rows)[0]


def test_a_generation_half_of_which_predates_the_sweep_says_so():
    """Exactly what the fleet looks like for one generation after a rolling
    restart, so it must not silently report the sweep as smaller than it is."""
    rows = [
        parse_worker_log("w00", hp_log(), age=1.0),
        parse_worker_log("w01", "iter 1  reward +1.0 bb\n", age=1.0),
    ]

    assert "1 senza iperparametri registrati" in format_sweep_line(rows)[0]


def test_the_fallback_arm_is_named_apart_in_the_summary():
    """Its settings are an independent draw like any other, but the count is how
    often inheritance was available at all."""
    rows = [parse_worker_log("w00", hp_log(hp_arm=HP_ARM_FALLBACK), age=1.0)]

    assert "genitore senza metadati" in format_sweep_line(rows)[0]


def test_run_sh_does_not_override_the_axes_the_sweep_decides():
    """An explicit flag in `run.sh` beats a default in `loop.py`: a pinned
    `--hands 512` would silently override the value the sweep draws per worker.

    A default is only a default for the paths that do not override it, so the
    launcher has to leave every swept axis alone.
    """
    script = (Path(__file__).resolve().parents[2] / "run.sh").read_text(encoding="utf-8")
    # Anchored on the launch itself, not on the first mention of `poker-loop`:
    # an earlier `pgrep` guard mentions it too, and so does the comment above
    # the command explaining why these flags are absent.
    launch = script.index('setsid nohup "$VENV/bin/poker-loop"')
    start = script[launch:script.index("supervisor.log", launch)]
    for axis in ("--hands",
                 "--lr", "--opponent-probability", "--ppo-epochs", "--clip-epsilon"):
        assert axis not in start, f"run.sh pins {axis}, which the sweep decides per worker"
    # Same class of bug, different flags: the size and cadence of a validation
    # pass decide how many rated sessions are in the count the whole rating is
    # built on, so a pinned value would silently override `loop.py`'s default.
    for moved in ("--eval-every", "--eval-sessions"):
        assert moved not in start, f"run.sh pins {moved}, whose default moved in loop.py"


def test_the_chart_carries_the_same_rolling_mean_the_table_shows():
    """The curve's last point and the table's cell must be the same number, or
    the panel contradicts the row it hangs under."""
    from pokerlab.rl.monitor import parse_worker_history

    lines = [header(512)]
    for i in range(1, 301):
        lines.append(
            f"iter {i:4}  reward {(+1.0 if i % 2 else -0.5):+7.2f} bb  policy -0.01  "
            f"value 1.0  entropy 0.900  kl 0.01  clip 0.1"
        )
    text = "\n".join(lines)

    history = parse_worker_history("w00", text)
    row = parse_worker_log("w00", text)

    assert len(history.train_bb100_mean) == len(history.train_bb100)
    assert history.train_bb100_mean[-1] == pytest.approx(row.train_bb100)
    # And it is genuinely a mean: the raw series alternates between +100 and -50.
    assert history.train_bb100[-1] in (100.0, -50.0)
    assert 20.0 < history.train_bb100_mean[-1] < 30.0


def test_the_rolling_mean_is_computed_before_thinning():
    """`thin` subsamples without averaging -- deliberately, so a chart of a
    noisy metric keeps its outliers -- so a mean taken after it would be the
    mean of a sample rather than of the window."""
    from pokerlab.rl.monitor import parse_worker_history

    lines = [header(1000)]
    # 2,000 iterations against a 400-point budget: the series is thinned 5:1.
    lines += [
        f"iter {i:4}  reward   +1.00 bb  policy -0.01  value 1.0  entropy 0.9  "
        f"kl 0.01  clip 0.1"
        for i in range(1, 2001)
    ]

    history = parse_worker_history("w00", "\n".join(lines), max_points=400)

    # Thinned well below the 2,000 iterations played (the exact count depends on
    # `thin`, which also always keeps the last point) ...
    assert len(history.train_bb100) <= 401
    # ... and the mean series is thinned with the same indices, so the two stay
    # aligned with `iterations`.
    assert len(history.train_bb100_mean) == len(history.train_bb100)
    assert history.train_bb100_mean[-1] == pytest.approx(100.0)


def test_a_log_without_a_header_gets_no_mean_curve():
    """A curve drawn to a different rule than the cell it sits under would be
    worse than no curve."""
    from pokerlab.rl.monitor import parse_worker_history

    text = "\n".join(
        f"iter {i:4}  reward   +1.00 bb  policy -0.01  value 1.0  entropy 0.9  "
        f"kl 0.01  clip 0.1"
        for i in range(1, 30)
    )

    history = parse_worker_history("w00", text)

    assert history.train_bb100  # the raw series is still there
    assert history.train_bb100_mean == []


def test_a_parent_without_the_newest_axes_still_gets_perturbed():
    """Every model published before minibatch/lambda/value/grad-norm became axes
    lacks them; treating that as unusable would drop the fleet to sampling."""
    child = perturb_hyperparameters(a_parent(), random.Random(0))
    assert child is not None and child.arm == loop_module.HP_ARM_INHERITED
    assert child.minibatch_size in {round(1024 * m) for m in loop_module.HP_MULTIPLIERS}
    assert child.gae_lambda <= 1.0


def test_the_newest_axes_are_inherited_from_a_parent_that_has_them():
    parent = a_parent(minibatch_size=2000, gae_lambda=0.99, value_coef=1.0, policy_max_grad_norm=0.1)
    seen = {perturb_hyperparameters(parent, random.Random(s)).minibatch_size for s in range(50)}
    assert seen <= {1600, 2000, 2400}


def test_entropy_coefficient_can_leave_zero_but_only_upwards():
    """Zero is where the fleet lives and a multiplier alone never leaves it."""
    up = {perturb_hyperparameters(a_parent(entropy_coef=0.0), random.Random(s)).entropy_coef for s in range(60)}
    assert up == {0.0, 1e-3}
    down = {perturb_hyperparameters(a_parent(entropy_coef=1e-3), random.Random(s)).entropy_coef for s in range(60)}
    assert down == {0.8e-3, 1e-3, 1.2e-3}


def test_the_gae_lambda_moves_through_its_complement():
    """x0.8 / x1.2 on lambda would jump 0.95 to 0.76 or past 1; on 1 - lambda it
    stays near the parent and can never reach 1."""
    got = {round(perturb_hyperparameters(a_parent(gae_lambda=0.95), random.Random(s)).gae_lambda, 4)
           for s in range(60)}
    assert got == {0.96, 0.95, 0.94}
    top = {perturb_hyperparameters(a_parent(gae_lambda=1.0), random.Random(s)).gae_lambda for s in range(60)}
    assert max(top) < 1.0


# ---- config.toml -------------------------------------------------------------

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.toml"


def write_config(path, text):
    """A `config.toml` for a test: `text` plus an `equity_model`, which every run needs and
    which only has to exist as a file (nothing here loads it)."""
    equity = path.parent / "equity.pt"
    equity.write_bytes(b"")
    path.write_text(f'equity_model = "{equity}"\n' + text, encoding="utf-8")


def test_the_shipped_config_is_valid_for_every_cli():
    """A key that any CLI rejects would stop it at argparse the next time it starts:
    every supervisor and worker, but also `poker-elo`, `benchmark_arena` and the
    dashboard, all of which read the same file."""
    pytest.importorskip("torch")
    from importlib import import_module

    from pokerlab.config import parse_with_config
    from pokerlab.rl.siblings import TORCH_FREE, WITH_TORCH, sibling_parsers

    _args, report = loop_module.load_args(["--config", str(REPO_CONFIG)])
    assert report.applied  # the file says something
    for name in TORCH_FREE + WITH_TORCH:
        module = import_module(name)
        _args, report = parse_with_config(
            module.build_parser(),
            ["--config", str(REPO_CONFIG)],
            siblings=lambda name=name: sibling_parsers(name, with_torch=name in WITH_TORCH),
            lenient=name in TORCH_FREE,
        )
        assert report.path == REPO_CONFIG, name


def test_a_key_shared_with_an_arena_or_the_dashboard_means_the_same_everywhere():
    """One flat file, five programs: a key is checked against whichever parser
    owns it, so the same name with another type or default in another program
    would make one of them read a value meant for something else. This is why the
    arenas call a session's length `--session-hands`, as `poker-train`
    does, and not `--hands`, which there means hands per training iteration."""
    pytest.importorskip("torch")
    from importlib import import_module

    from pokerlab.config import _actions, _is_local
    from pokerlab.rl.siblings import TORCH_FREE, WITH_TORCH

    owners: dict[str, list[tuple[str, object, object]]] = {}
    for name in TORCH_FREE + WITH_TORCH:
        for dest, action in _actions(import_module(name).build_parser()).items():
            if not _is_local(dest, action):
                owners.setdefault(dest, []).append((name, action.type, action.default))
    for dest, entries in owners.items():
        if not any(name in TORCH_FREE for name, _, _ in entries):
            continue
        assert len({(kind, repr(default)) for _, kind, default in entries}) == 1, (dest, entries)


def test_run_sh_pins_nothing_the_config_file_sets():
    """An explicit flag beats the file, so a flag in `run.sh` for a key that
    `config.toml` sets would make the file a decoration."""
    from pokerlab.config import read_config

    script = (Path(__file__).resolve().parents[2] / "run.sh").read_text(encoding="utf-8")
    launch = script.index('setsid nohup "$VENV/bin/poker-loop"')
    start = script[launch:script.index("supervisor.log", launch)]
    for key in read_config(REPO_CONFIG):
        flag = "--" + key.replace("_", "-")
        assert flag + " " not in start.replace("\\\n", " "), f"run.sh pins {flag}"


def test_a_worker_is_told_not_to_read_the_config_again(tmp_path, monkeypatch):
    command, _ = capture_launch(tmp_path, monkeypatch)
    assert command[command.index("--config") + 1] == ""


def test_the_file_sets_where_a_worker_with_no_parent_starts(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "c.toml"
    write_config(path, "lr = 0.5\nhands = 9\nstack_max_bb = 80.0\n")
    args, _ = loop_module.load_args(["--config", str(path)])
    assert args.stack_max_bb == 80.0
    start = starting_hyperparameters(args)
    assert start.lr == 0.5 and start.hands == 9


def test_the_file_sets_the_shape_of_the_network(tmp_path):
    pytest.importorskip("torch")
    from pokerlab.rl.train import network_shape

    path = tmp_path / "c.toml"
    write_config(path, 
        "[network]\nhidden = 128\nnum_layers = 2\nhead_hidden = 64\nhead_layers = 3\n",
    )
    args, _ = loop_module.load_args(["--config", str(path)])
    assert network_shape(args) == {
        "hidden": 128, "num_layers": 2, "head_hidden": 64, "head_layers": 3,
    }


@pytest.mark.parametrize("key", ["hidden", "num_layers", "head_hidden"])
def test_a_network_that_cannot_be_built_is_a_config_error(tmp_path, key):
    pytest.importorskip("torch")
    path = tmp_path / "c.toml"
    write_config(path, f"{key} = 0\n")
    with pytest.raises(loop_module.ConfigError, match=key):
        loop_module.load_args(["--config", str(path)])


def test_zero_head_layers_is_a_network_and_a_negative_number_is_not(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "c.toml"
    write_config(path, "head_layers = 0\n")
    assert loop_module.load_args(["--config", str(path)])[0].head_layers == 0
    write_config(path, "head_layers = -1\n")
    with pytest.raises(loop_module.ConfigError, match="head_layers"):
        loop_module.load_args(["--config", str(path)])


def test_the_file_sets_how_far_a_worker_moves(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "c.toml"
    write_config(path, "hp_multipliers = [1.0]\n")
    args, _ = loop_module.load_args(["--config", str(path)])
    assert args.hp_multipliers == [1.0]

    start = starting_hyperparameters(args)
    plan = hyperparameter_plan(
        4, [a_parent(), None, a_parent(), None], start,
        rng=random.Random(0), multipliers=args.hp_multipliers,
    )
    # One factor, 1.0: nobody moves, whichever arm they are in.
    assert all(
        {axis: getattr(hp, axis) for axis in HP_AXES}
        == {axis: getattr(start, axis) for axis in HP_AXES}
        for hp in plan
    )


def test_a_multiplier_that_would_break_an_axis_is_a_config_error(tmp_path):
    pytest.importorskip("torch")
    from pokerlab.config import ConfigError

    for bad in ("[0.8, 0, 1.2]", "[-1.0]"):
        path = tmp_path / "c.toml"
        write_config(path, f"hp_multipliers = {bad}\n")
        with pytest.raises(ConfigError, match="hp_multipliers"):
            loop_module.load_args(["--config", str(path)])


def test_a_file_that_cannot_be_read_keeps_the_previous_values(tmp_path, capsys):
    pytest.importorskip("torch")
    path = tmp_path / "c.toml"
    write_config(path, "stack_max_bb = 50.0\n")
    argv = ["--config", str(path)]
    args, _ = loop_module.load_args(argv)
    assert args.stack_max_bb == 50.0

    write_config(path, "stack_max_bb = = 60\n")  # saved halfway
    kept = loop_module.refresh_args(args, lambda: loop_module.load_args(argv))
    assert kept is args and "valori precedenti" in capsys.readouterr().out

    write_config(path, "stack_max_bb = 60.0\n")
    fresh = loop_module.refresh_args(args, lambda: loop_module.load_args(argv))
    assert fresh.stack_max_bb == 60.0
    assert "stack_max_bb: 50.0 -> 60.0" in capsys.readouterr().out


def test_a_worker_is_handed_the_fleets_draw_tiers(tmp_path, monkeypatch):
    command, _ = capture_launch(tmp_path, monkeypatch)
    assert command[command.index("--draw-tiers") + 1] == "10, 100, 1000, all"


def test_the_file_sets_the_draw_tiers_and_a_bad_one_is_an_error(tmp_path):
    from pokerlab.config import ConfigError

    path = tmp_path / "c.toml"
    write_config(path, 'draw_tiers = "100, 1000, all"\n')
    args, _ = loop_module.load_args(["--config", str(path)])
    assert args.draw_tiers == "100, 1000, all"
    write_config(path, 'draw_tiers = "100, soon"\n')
    with pytest.raises(ConfigError):
        loop_module.load_args(["--config", str(path)])


def test_a_worker_is_handed_the_fleets_k_schedule(tmp_path, monkeypatch):
    command, _ = capture_launch(tmp_path, monkeypatch)
    assert command[command.index("--k-schedule") + 1] == "0:16, 100:2"


def test_the_file_sets_the_parent_bands_and_the_k_schedule(tmp_path):
    path = tmp_path / "c.toml"
    write_config(path, 'parent_tiers = "10, 10, all"\nk_schedule = "0:20, 50:5"\n')
    args, _ = loop_module.load_args(["--config", str(path)])
    assert args.parent_tiers == "10, 10, all"
    assert args.k_schedule == "0:20, 50:5"
    # And the same text is what poker-train reads for the same key.
    from pokerlab.rl.train import build_parser as build_train_parser

    assert "k_schedule" in {a.dest for a in build_train_parser()._actions}


@pytest.mark.parametrize(
    "line",
    ['parent_tiers = "10, soon"', 'parent_tiers = "0"', 'k_schedule = "5:16"',
     'k_schedule = "0:16, 0:11"', 'k_schedule = "0:-1"', "k_schedule = 3"],
)
def test_a_bad_parent_band_or_k_schedule_in_the_file_is_an_error(tmp_path, line):
    path = tmp_path / "c.toml"
    write_config(path, line + "\n")
    from pokerlab.config import ConfigError

    with pytest.raises(ConfigError):
        loop_module.load_args(["--config", str(path)])

