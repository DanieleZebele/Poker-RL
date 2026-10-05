"""Updating every model's Elo by hand: the end-of-training pass, on demand.

The Elo arithmetic is `test_rl_pool_registry.py` and the pass itself is
`test_rl_global_arena.py`. What matters here is the two things this tool does
differently from the automatic passes, because both are choices a future change
could quietly undo: it draws for *coverage* rather than uniformly, and it must
never delete a checkpoint as a side effect of refreshing a rating.

No torch: this module reaches it only through `global_arena`'s function-local
imports, which is what lets these tests run in the ordinary suite.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

import pokerlab.rl.population_arena as arena
from pokerlab.rl.global_arena import Candidate
from pokerlab.rl.global_store import write_member
from pokerlab.rl.pool_registry import MODEL, PoolMember


def test_the_module_does_not_import_torch():
    """Same requirement as the dashboard's, for the same reason: on the NFS server
    under load `import torch` was measured at over five minutes.

    In a subprocess, because `sys.modules` is process-global: run inside the suite,
    some earlier test file has already imported torch and the assertion would pass
    for the wrong reason (or fail for one).
    """
    import subprocess

    finished = subprocess.run(
        [sys.executable, "-c",
         "import sys, pokerlab.rl.population_arena; print('torch' in sys.modules)"],
        check=True, capture_output=True, text=True, timeout=600,
    )
    assert finished.stdout.strip() == "False", finished.stdout


def test_the_draw_reaches_the_least_played_first(tmp_path):
    """The automatic passes sample uniformly on purpose -- they can delete people.
    This one exists to reach models those passes have never seated."""
    population = [Candidate(label=f"m{i}", path=tmp_path / f"m{i}.pt") for i in range(10)]
    counts = {f"m{i}": {"games": i * 100} for i in range(10)}

    drawn = arena.least_played_draw(counts)(population, 4, random.Random(1))

    assert len(drawn) == 4
    # Half the slots go to the least played, deterministically.
    assert {c.label for c in drawn[:2]} == {"m0", "m1"}


def test_a_model_never_rated_is_drawn_before_a_veteran(tmp_path):
    population = [Candidate(label=n, path=tmp_path / f"{n}.pt") for n in ("vet", "new")]
    counts = {"vet": {"games": 900}}   # "new" is absent, i.e. zero games

    drawn = arena.least_played_draw(counts)(population, 1, random.Random(0))

    assert [c.label for c in drawn] == ["new"]


# The test store holds a handful of models: a six-handed table is the largest it seats.
SIX_SEATS = ["--table-weights", "0", "0", "0", "0", "1", "0", "0", "0"]


def run_with_stub(monkeypatch, argv):
    """Drive `main` with the pass itself stubbed out, and report its arguments."""
    calls: list[dict] = []

    from pokerlab.rl.global_arena import PopulationSessionsReport

    def fake_sessions(**kwargs):
        calls.append(kwargs)
        return PopulationSessionsReport(sessions_played=kwargs["sessions"])

    monkeypatch.setattr(arena, "run_population_sessions", fake_sessions)
    monkeypatch.setattr(arena, "write_snapshot", lambda *a, **k: None)
    assert arena.main(["--config", "", *SIX_SEATS, *argv]) == 0
    return calls


def store(tmp_path, models=8):
    for index in range(models):
        directory = tmp_path / "checkpoints" / "models"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"m{index}.pt").write_bytes(b"weights")
        write_member(
            tmp_path / "checkpoints" / "global",
            PoolMember(label=f"m{index}", kind=MODEL, ref="x", rating=1500.0, games=index),
        )
    return tmp_path / "checkpoints"


def test_pruning_is_off_unless_asked_for(tmp_path, monkeypatch):
    """Elimination deletes real files. A tool whose job is to refresh ratings must
    not delete anything just because it ran."""
    root = store(tmp_path)

    calls = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"), "--elo-sessions", "200",
    ])

    assert len(calls) == 2
    for call in calls:
        assert call["trigger_size"] == arena.NO_PRUNE_TRIGGER
        assert arena.NO_PRUNE_TRIGGER > 10**6, "must be out of reach of any real store"


def test_prune_restores_the_ordinary_trigger(tmp_path, monkeypatch):
    root = store(tmp_path)

    calls = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"),
        "--prune", "--global-trigger-size", "1234",
    ])

    assert calls[0]["trigger_size"] == 1234


def test_random_coverage_passes_no_draw_at_all(tmp_path, monkeypatch):
    """`--elo-coverage random` has to fall back to the pass's own uniform sample,
    not to a re-implementation of it."""
    root = store(tmp_path)

    calls = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"), "--elo-coverage", "random",
    ])

    assert calls[0]["draw"] is None
    played = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"), "--elo-coverage", "played",
    ])
    assert played[0]["draw"] is not None


def test_a_store_too_small_to_seat_a_table_is_reported_not_crashed(tmp_path):
    (tmp_path / "checkpoints" / "models").mkdir(parents=True)
    assert arena.main([
        "--config", "",
        "--root", str(tmp_path / "checkpoints"),
        "--global-dir", str(tmp_path / "checkpoints" / "global"),
    ]) == 1


def test_the_rating_summary_survives_an_empty_registry(tmp_path):
    assert "vuoto" in arena.rating_summary(Path(tmp_path / "nothing-here"))


def test_every_draw_refreshes_the_shared_snapshot(tmp_path, monkeypatch):
    """The merge inside the draw asks for a snapshot too, but unforced: it
    refreshes only if the file has gone stale and only if it wins the lock. A run
    here is minutes to hours, so everything that reads the snapshot would
    otherwise show ratings this run has already superseded."""
    from pokerlab.rl.global_arena import PopulationSessionsReport

    root = store(tmp_path)
    refreshes: list[dict] = []
    monkeypatch.setattr(
        arena, "run_population_sessions",
        lambda **k: PopulationSessionsReport(sessions_played=k["sessions"]),
    )
    monkeypatch.setattr(
        arena,
        "write_snapshot",
        lambda global_dir, **kwargs: bool(refreshes.append(kwargs)) or True,
    )

    assert arena.main([
        "--config", "", *SIX_SEATS,
        "--root", str(root), "--global-dir", str(root / "global"), "--elo-sessions", "300",
    ]) == 0

    assert len(refreshes) >= 3, "at least one refresh per draw, not just at the end"
    assert all(call["force"] for call in refreshes)


def test_the_config_file_sets_what_the_arena_plays(tmp_path, monkeypatch):
    """The point of sharing `config.toml`: the same keys `poker-train` reads --
    here the pass's sample, session count and length -- steer this tool too."""
    root = store(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(
        "global_sample = 7\nglobal_sessions = 3\nelo_sessions = 3\nsession_hands = 40\n"
        "global_trigger_size = 99\n# a key of poker-train: not this tool's business\nlr = 0.01\n",
        encoding="utf-8",
    )
    calls: list[dict] = []
    from pokerlab.rl.global_arena import PopulationSessionsReport

    monkeypatch.setattr(
        arena, "run_population_sessions", lambda **k: calls.append(k) or PopulationSessionsReport()
    )
    monkeypatch.setattr(arena, "write_snapshot", lambda *a, **k: None)

    assert arena.main([
        "--config", str(path), *SIX_SEATS, "--root", str(root),
        "--global-dir", str(root / "global"), "--prune",
    ]) == 0

    call = calls[0]
    assert (call["population_sample"], call["sessions"], call["session_hands"]) == (7, 3, 40)
    assert call["trigger_size"] == 99


def test_a_flag_beats_the_config_file(tmp_path, monkeypatch):
    root = store(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text("global_sessions = 3\n", encoding="utf-8")
    calls = run_with_stub(monkeypatch, [
        "--config", str(path), "--root", str(root), "--global-dir", str(root / "global"),
        "--global-sessions", "5",
    ])
    assert calls[0]["sessions"] == 5


def test_a_file_cannot_switch_pruning_on_for_the_whole_fleet(tmp_path):
    """`--prune` deletes checkpoints; it is a choice of one invocation."""
    root = store(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text("prune = true\n", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        arena.main(["--config", str(path), "--root", str(root)])
    assert raised.value.code == 2


def test_the_torch_free_clis_read_the_shipped_config_without_importing_torch():
    """The arenas and the dashboard cannot see `poker-train`'s parser, and must not
    import it to find out what its keys are: `import torch` on the NFS server under
    load was measured at over five minutes. In a subprocess, for the same reason as
    the test above."""
    import subprocess

    code = (
        "import sys\n"
        "from importlib import import_module\n"
        "from pokerlab.config import parse_with_config\n"
        "from pokerlab.rl.siblings import TORCH_FREE, sibling_parsers\n"
        "for name in TORCH_FREE:\n"
        "    parse_with_config(import_module(name).build_parser(), ['--config', 'config.toml'],\n"
        "        siblings=lambda name=name: sibling_parsers(name, with_torch=False), lenient=True)\n"
        "print('torch' in sys.modules)\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
        timeout=600, cwd=Path(__file__).resolve().parents[2],
    )
    assert finished.stdout.strip() == "False", finished.stdout


def test_the_k_schedule_of_the_config_file_reaches_the_merge(tmp_path, monkeypatch):
    root = store(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text('k_schedule = "0:4, 10:2"\n', encoding="utf-8")

    calls = run_with_stub(monkeypatch, [
        "--config", str(path), "--root", str(root), "--global-dir", str(root / "global"),
    ])

    assert tuple(calls[0]["k_schedule"]) == ((0, 4.0), (10, 2.0))


def test_the_run_plays_exactly_the_total_of_sessions_it_was_given(tmp_path, monkeypatch):
    """The length of a run is a total of sessions, not a number of passes: draws of
    `--global-sessions` of them, the last taking whatever is left."""
    root = store(tmp_path)

    calls = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"),
        "--elo-sessions", "10", "--global-sessions", "4",
    ])

    assert [call["sessions"] for call in calls] == [4, 4, 2]


def test_a_run_that_plays_nothing_stops_instead_of_spinning(tmp_path, monkeypatch):
    from pokerlab.rl.global_arena import PopulationSessionsReport

    root = store(tmp_path)
    calls: list[dict] = []
    monkeypatch.setattr(
        arena, "run_population_sessions", lambda **k: calls.append(k) or PopulationSessionsReport()
    )
    monkeypatch.setattr(arena, "write_snapshot", lambda *a, **k: None)

    assert arena.main([
        "--config", "", *SIX_SEATS, "--root", str(root), "--global-dir", str(root / "global"),
        "--elo-sessions", "1000",
    ]) == 0
    assert len(calls) == 1
