"""Updating every model's Elo by hand: the end-of-training round, on demand.

The Elo arithmetic is `test_rl_pool_registry.py` and the round itself is
`test_rl_global_arena.py`. What matters here is the two things this tool does
differently from the automatic rounds, because both are choices a future change
could quietly undo: it draws for *coverage* rather than uniformly, and it must
never delete a checkpoint as a side effect of refreshing a rating.

No torch: this module reaches it only through `global_arena`'s function-local
imports, which is what lets these tests run in the ordinary suite.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

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
    """The automatic rounds sample uniformly on purpose -- they can delete people.
    This one exists to reach models those rounds have never seated."""
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


def run_with_stub(monkeypatch, argv):
    """Drive `main` with the round itself stubbed out, and report its arguments."""
    calls: list[dict] = []

    from pokerlab.rl.global_arena import PopulationRoundReport

    def fake_round(**kwargs):
        calls.append(kwargs)
        return PopulationRoundReport()

    monkeypatch.setattr(arena, "run_population_round", fake_round)
    monkeypatch.setattr(arena, "write_snapshot", lambda *a, **k: None)
    assert arena.main(argv) == 0
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
        "--root", str(root), "--global-dir", str(root / "global"), "--rounds", "2",
    ])

    assert len(calls) == 2
    for call in calls:
        assert call["trigger_size"] == arena.NO_PRUNE_TRIGGER
        assert arena.NO_PRUNE_TRIGGER > 10**6, "must be out of reach of any real store"


def test_prune_restores_the_ordinary_trigger(tmp_path, monkeypatch):
    root = store(tmp_path)

    calls = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"),
        "--prune", "--trigger-size", "1234",
    ])

    assert calls[0]["trigger_size"] == 1234


def test_random_coverage_passes_no_draw_at_all(tmp_path, monkeypatch):
    """`--coverage random` has to fall back to the round's own uniform sample,
    not to a re-implementation of it."""
    root = store(tmp_path)

    calls = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"), "--coverage", "random",
    ])

    assert calls[0]["draw"] is None
    played = run_with_stub(monkeypatch, [
        "--root", str(root), "--global-dir", str(root / "global"), "--coverage", "played",
    ])
    assert played[0]["draw"] is not None


def test_a_store_too_small_to_seat_a_table_is_reported_not_crashed(tmp_path):
    (tmp_path / "checkpoints" / "models").mkdir(parents=True)
    assert arena.main([
        "--root", str(tmp_path / "checkpoints"),
        "--global-dir", str(tmp_path / "checkpoints" / "global"),
    ]) == 1


def test_the_rating_summary_survives_an_empty_registry(tmp_path):
    assert "vuoto" in arena.rating_summary(Path(tmp_path / "nothing-here"))


def test_every_round_refreshes_the_shared_snapshot(tmp_path, monkeypatch):
    """The merge inside the round asks for a snapshot too, but unforced: it
    refreshes only if the file has gone stale and only if it wins the lock. A
    round here is minutes to hours, so everything that reads the snapshot would
    otherwise show ratings this run has already superseded."""
    from pokerlab.rl.global_arena import PopulationRoundReport

    root = store(tmp_path)
    refreshes: list[dict] = []
    monkeypatch.setattr(arena, "run_population_round", lambda **k: PopulationRoundReport())
    monkeypatch.setattr(
        arena,
        "write_snapshot",
        lambda global_dir, **kwargs: bool(refreshes.append(kwargs)) or True,
    )

    assert arena.main([
        "--root", str(root), "--global-dir", str(root / "global"), "--rounds", "3",
    ]) == 0

    assert len(refreshes) >= 3, "at least one refresh per round, not just at the end"
    assert all(call["force"] for call in refreshes)
