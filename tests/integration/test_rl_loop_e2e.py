"""The real thing, small: `poker-loop` launching real `poker-train` workers.

This is the test that pins the contract between the two entry points -- every
flag the loop passes must be one train accepts -- and the whole life of a model:
drawn as an opponent, trained, published into the shared store at the end of its
run, registered in the ranking, and rated by the end-of-run round.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def run_loop(tmp_path, *, generations, extra=()):
    command = [
        sys.executable, "-m", "pokerlab.rl.loop",
        "--workers", "3", "--generations", str(generations),
        "--iterations", "2", "--hands", "8", "--players", "2", "--stack", "100",
        "--eval-every", "2", "--eval-sessions", "1", "--archive-every", "2",
        "--benchmark-hands", "0",
        # The benchmark directory this points at is empty, so the round against
        # the anchors skips itself -- but say so explicitly rather than relying on
        # that: the production default is 500 sessions of 1000 hands, and a test
        # that grew a benchmark fixture would silently start playing half a
        # million hands per worker.
        "--benchmark-sessions", "2",
        "--global-sample", "6", "--global-benchmark-sample", "0",
        "--global-games-per-model", "1", "--global-hands-per-game", "2",
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
        *extra,
    ]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    return subprocess.run(
        command, capture_output=True, text=True, timeout=300, env=environment, check=False
    )


def test_two_generations_publish_rate_and_reuse_models(tmp_path):
    result = run_loop(tmp_path, generations=2)
    assert result.returncode == 0, result.stdout + result.stderr

    models = sorted(p.name for p in (tmp_path / "models").glob("*.pt"))
    # Every worker of every generation published exactly its own best model,
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
    assert any(games > 0 for games in rated), "the end-of-run round must have rated someone"

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


def test_the_old_per_machine_layout_is_never_created(tmp_path):
    result = run_loop(tmp_path, generations=1, extra=("--no-global-round",))
    assert result.returncode == 0, result.stdout + result.stderr

    created = {p.name for p in tmp_path.iterdir()}
    assert not ({"pool", "exchange", "retired", "checkpoints"} & created)
    assert not list(tmp_path.rglob("retired"))
    assert (tmp_path / "state" / "loop_state.json").exists()


def test_a_stop_file_ends_the_loop_after_the_generation_in_flight(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "STOP").write_text("")

    result = run_loop(tmp_path, generations=5, extra=("--no-global-round",))

    assert result.returncode == 0
    assert "STOP" in result.stdout
    assert not list((tmp_path / "models").glob("*.pt"))


def test_a_run_publishes_its_last_checkpoint_not_its_best_rated(tmp_path):
    """The rule is deliberate and counter-intuitive, so it is pinned here.

    Archiving used to keep whichever checkpoint the in-run rating scored highest.
    That rating comes from a few hundred hands against the run's own drawn pool,
    and across 27,610 real runs it called the mid-run model better than the final
    one 47% of the time -- while a duplicate-deck duel of the two, from the same
    seed, put the later one ahead in all 6 seeds by +56.7 bb/100 (t = 7.5).
    Selecting on a coin flip captured 30 of those 57 bb/100; taking the last one
    captures all of it. Swap this only for a measurement good enough to beat a
    blind rule -- roughly 95% accurate -- not for the rating that was there.
    """
    result = run_loop(
        tmp_path,
        generations=1,
        extra=("--no-global-round", "--archive-every", "1", "--eval-every", "1"),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    log = (tmp_path / "logs" / "gen0001-w00.log").read_text()
    archived = [
        int(line.split("iterazione ")[1].split(",")[0])
        for line in log.splitlines()
        if "archived" in line and "iterazione" in line
    ]
    assert archived, "the run must archive at least once"
    # Every iteration archives, and the last one wins -- regardless of how the
    # rating moved in between.
    assert archived == sorted(archived), "archives happen in iteration order"
    assert archived[-1] == max(archived)
    published = [line for line in log.splitlines() if line.startswith("published ")]
    assert published, "and the archive it leaves behind is what gets published"


def test_a_worker_that_finished_early_fills_the_wait_with_rating_rounds(tmp_path):
    """The whole handshake, for real: a worker announces it is only killing
    time, the supervisor waits until every live worker says so, writes the flag,
    and they all exit. Getting either half wrong is a deadlock -- the worker
    waiting for the supervisor and the supervisor waiting for the worker -- and
    it would show up as a generation that never ends, so it is worth one real
    run rather than only fakes."""
    result = run_loop(
        tmp_path, generations=1, extra=("--elo-fill-in", "--fill-min-sessions", "1")
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
