"""The real thing, small: `poker-loop` launching real `poker-train` workers.

This is the test that pins the contract between the two entry points -- every
flag the loop passes must be one train accepts -- and the whole life of a model:
drawn as an opponent, trained, published into the shared store at the end of its
run, registered in the ranking, and rated by the end-of-run pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from pokerlab.rl.sweep_log import read_observations

pytest.importorskip("torch")


def run_loop(tmp_path, *, generations, extra=()):
    import torch
    from support import TINY_EQUITY

    from pokerlab.rl.equity_net import EquityNet

    # Every network is built on an equity network, so every worker needs one to read.
    equity = EquityNet(**TINY_EQUITY, arch="sets")
    torch.save({"config": equity.config(), "state": equity.state_dict()}, tmp_path / "equity.pt")
    command = [
        sys.executable, "-m", "pokerlab.rl.loop",
        "--workers", "3", "--generations", str(generations),
        "--iterations", "2", "--hands", "8", "--table-weights", "1", "0", "0", "0", "0", "0", "0", "0",
        "--equity-model", str(tmp_path / "equity.pt"),
        "--stack-min-bb", "50", "--stack-max-bb", "50", "--sb", "1", "--bb", "2",
        "--eval-every", "2", "--eval-sessions", "1",
        # The benchmark directory this points at is empty, so the pass against
        # the anchors skips itself -- but say so explicitly rather than relying on
        # that: the production default is 500 sessions of 1000 hands, and a test
        # that grew a benchmark fixture would silently start playing half a
        # million hands per worker.
        "--benchmark-sessions", "2",
        "--global-sample", "6", "--global-benchmark-sample", "0",
        "--global-sessions", "3", "--session-hands", "2",
        "--machine", "host-a", "--seed-base", "5",
        "--state-dir", str(tmp_path / "state"),
        "--models-dir", str(tmp_path / "models"),
        "--work-dir", str(tmp_path / "work"),
        "--log-dir", str(tmp_path / "logs"),
        "--global-dir", str(tmp_path / "global"),
        "--global-root", str(tmp_path),
        "--benchmark-dir", str(tmp_path / "benchmark"),
        # Off unless a test is about it. The phase is on by default in
        # production and deliberately does not end until a floor of rated
        # sessions is met, which is minutes per worker -- paid once, in the one
        # test below that exercises it, rather than by every test here.
        "--no-elo-fill-in",
        # The repository's own config.toml sits in the cwd; a test sets what it
        # needs by flag and must not inherit the fleet's production values.
        "--config", "",
        *extra,
    ]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    return subprocess.run(
        command, capture_output=True, text=True, timeout=300, env=environment, check=False
    )


def _published(tmp_path):
    """Every model the run published: in the store, or moved into the benchmark. With
    this test's two-player tables a single anchor fills the benchmark, so the first
    model to be rated is promoted out of `models/` (see `add_benchmark_candidates`)."""
    return sorted(
        [*(tmp_path / "models").glob("*.pt"), *(tmp_path / "benchmark").rglob("*.pt")],
        key=lambda path: path.name,
    )


def test_two_generations_publish_rate_and_reuse_models(tmp_path):
    result = run_loop(tmp_path, generations=2)
    assert result.returncode == 0, result.stdout + result.stderr

    models = sorted(p.name for p in _published(tmp_path))
    # Every worker of every generation published exactly its own model,
    # named for the machine, generation and worker that trained it.
    assert len(models) == 6
    assert all(name.startswith("host-a-gen000") and "-w0" in name for name in models)
    assert {name.split("-agent-")[0] for name in models} == {
        f"host-a-gen000{g}-w0{w}" for g in (1, 2) for w in (0, 1, 2)
    }

    members = {p.stem for p in (tmp_path / "global" / "members").glob("*.json")}
    assert {name[:-3] for name in models} <= members  # all registered in the ranking
    rated = [
        json.loads(p.read_text())["games"] for p in (tmp_path / "global" / "members").glob("*.json")
    ]
    assert any(games > 0 for games in rated), "the end-of-run pass must have rated someone"

    # Nothing is left behind: no scratch, no live checkpoints, no locks, no partials.
    assert not list((tmp_path / "work").glob("gen*"))
    assert not list((tmp_path / "work").glob("agent-*.pt"))
    assert not list((tmp_path / "models").glob(".*"))
    assert not list((tmp_path / "global" / "locks").glob("*.lock"))

    state = json.loads((tmp_path / "state" / "loop_state.json").read_text())
    assert state["generation"] == 2 and len(state["history"]) == 2
    assert state["history"][0]["published_models"] == 3

    # The second generation drew its opponents from what the first one published.
    log = (tmp_path / "logs" / "gen0002-w00.log").read_text()
    assert "pool: drew" in log and "from" in log and "published host-a-gen0002-w00-agent-" in log

    # Every published model carries how it plays at the end of training, and the
    # worker's log shows it while it trains.
    import torch

    published = min((tmp_path / "models").glob("*.pt"))
    metadata = torch.load(published, map_location="cpu", weights_only=True)["metadata"]
    assert metadata["style_hands"] > 0
    events, chances = metadata["style"]["vpip"]
    assert 0 <= events <= chances
    assert any(line.startswith("stile (") for line in log.splitlines())

    # And its children left the evidence the sweep optimizer learns from: which
    # first-generation model each came from, and what the step cost and gained.
    seen = read_observations(tmp_path / "global")
    assert seen, "a worker that resumed from a parent must record its step"
    for observation in seen:
        assert observation.label.startswith("host-a-gen0002-")
        assert observation.parent.startswith("host-a-gen0001-")
        assert observation.cpu_seconds > 0
        assert observation.settings.keys() == observation.parent_settings.keys()


def test_without_inherited_hyperparameters_every_worker_steps_from_the_flags(tmp_path):
    """The weights still carry over; the settings do not. Every worker of every
    generation is one step -- x0.8, x1.0 or x1.2 -- from the fleet's own values, so
    nothing ever drifts off them, and none took a step from its parent, so there is
    nothing for the optimizer to record."""
    result = run_loop(
        tmp_path, generations=2, extra=("--no-inherit-hyperparameters", "--lr", "0.0005")
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "non ereditati" in result.stdout

    for generation in (1, 2):
        for worker in (0, 1, 2):
            log = (tmp_path / "logs" / f"gen000{generation}-w0{worker}.log").read_text()
            marker = next(line for line in log.splitlines() if line.startswith("iperparametri:"))
            assert "hp_arm=sampled " in marker
            lr = float(next(t for t in marker.split() if t.startswith("lr=")).split("=")[1])
            # Inherited, the second generation's lr would be a step from a step.
            assert any(lr == pytest.approx(0.0005 * m) for m in (0.8, 1.0, 1.2)), lr
    # The second generation still resumed from the first one's weights.
    assert "resumed from" in (tmp_path / "logs" / "gen0002-w00.log").read_text()
    assert read_observations(tmp_path / "global") == []


def test_the_network_shape_reaches_every_worker_and_survives_inheritance(tmp_path):
    """Heads with hidden layers, set from the loop's flags (as `config.toml` would):
    every published model records them, and the second generation, whose parents have
    the same shape, resumes from them instead of starting over."""
    import torch

    result = run_loop(
        tmp_path,
        generations=2,
        extra=("--hidden", "16", "--num-layers", "1", "--head-hidden", "8", "--head-layers", "2"),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    models = _published(tmp_path)
    assert len(models) == 6
    for path in models:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert (saved["hidden"], saved["num_layers"], saved["head_hidden"], saved["head_layers"]) == (
            16, 1, 8, 2,
        )
        assert saved["metadata"]["head_layers"] == 2
    second = (tmp_path / "logs" / "gen0002-w00.log").read_text()
    assert "resumed from" in second and "different network" not in second


def test_the_old_per_machine_layout_is_never_created(tmp_path):
    result = run_loop(tmp_path, generations=1, extra=("--no-global-elo",))
    assert result.returncode == 0, result.stdout + result.stderr

    created = {p.name for p in tmp_path.iterdir()}
    assert not ({"pool", "exchange", "retired", "checkpoints"} & created)
    assert not list(tmp_path.rglob("retired"))
    assert (tmp_path / "state" / "loop_state.json").exists()


def test_a_stop_file_ends_the_loop_after_the_generation_in_flight(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "STOP").write_text("")

    result = run_loop(tmp_path, generations=5, extra=("--no-global-elo",))

    assert result.returncode == 0
    assert "STOP" in result.stdout
    assert not list((tmp_path / "models").glob("*.pt"))


def test_a_run_publishes_its_last_checkpoint_not_its_best_rated(tmp_path):
    """The rule is deliberate and counter-intuitive, so it is pinned here.

    The in-run rating comes from a few thousand hands against the run's own drawn
    pool, which is a noisy measurement, while training improving the model is a
    reliable prior. Swap this only for a measurement good enough to beat the blind
    rule, not for that rating.
    """
    result = run_loop(
        tmp_path,
        generations=1,
        extra=("--no-global-elo", "--eval-every", "1"),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    log = (tmp_path / "logs" / "gen0001-w00.log").read_text()
    archived = [
        int(line.split("iterazione ")[1].split(",")[0])
        for line in log.splitlines()
        if "archived" in line and "iterazione" in line
    ]
    # One archive, written when training ends: the last iteration's weights,
    # whatever the rating did on the way (`--eval-every 1` rated every iteration).
    assert archived == [2]
    published = [line for line in log.splitlines() if line.startswith("published ")]
    assert published, "and the archive it leaves behind is what gets published"


def test_a_worker_that_finished_early_fills_the_wait_with_rating_passes(tmp_path):
    """The whole handshake, for real: a worker announces it is only killing
    time, the supervisor waits until every live worker says so, writes the flag,
    and they all exit. Getting either half wrong is a deadlock -- the worker
    waiting for the supervisor and the supervisor waiting for the worker -- and
    it would show up as a generation that never ends, so it is worth one real
    run rather than only fakes."""
    result = run_loop(
        tmp_path, generations=1, extra=("--elo-fill-in",)
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assert "dato il via libera" in result.stdout
    logs = sorted((tmp_path / "logs").glob("gen0001-w*.log"))
    assert logs
    filled = [path for path in logs if "phase: elo_fill" in path.read_text(encoding="utf-8")]
    assert filled, "nessun worker e' entrato nella fase di riempimento"
    text = filled[0].read_text(encoding="utf-8")
    assert "riempimento elo:" in text
    # And it cleaned up after itself: a marker left behind would have the next
    # generation's supervisor read a stale directory as a worker already
    # draining, and release that generation immediately.
    assert not list((tmp_path / "work").rglob("draining"))
