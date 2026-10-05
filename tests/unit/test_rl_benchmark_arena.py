"""The one tool allowed to move a frozen anchor's rating.

What matters here is not the Elo arithmetic (that is `test_rl_pool_registry.py`)
but the guarantees around it: that the anchors' mean cannot drift, that nothing
but the rating is touched, that a run can be undone, and that this remains the
*only* path -- an ordinary session still refuses to move an anchor.
"""

from __future__ import annotations

import json

import pytest

from pokerlab.rl.benchmark_arena import (
    Anchor,
    apply_sessions,
    collect_anchors,
    format_table,
    restore,
    save_backup,
    write_ratings,
)
from pokerlab.rl.global_store import read_member, write_member
from pokerlab.rl.pool_registry import DEFAULT_K_SCHEDULE, MODEL, PoolMember, PoolRegistry


def anchor_store(tmp_path, count=6, series=2, rating=1500.0):
    """A benchmark directory and a registry holding its members, as on disk."""
    root = tmp_path / "checkpoints"
    global_dir = root / "global"
    per = count // series
    for s in range(series):
        directory = root / "benchmark" / f"benchmark_{s + 1}"
        directory.mkdir(parents=True, exist_ok=True)
        for n in range(per):
            label = f"s{s + 1}-m{n}"
            (directory / f"{label}.pt").write_bytes(b"weights")
            write_member(
                global_dir,
                PoolMember(
                    label=label, kind=MODEL, ref=str(directory / f"{label}.pt"),
                    rating=rating + n, games=7, frozen=True,
                ),
            )
    return root, global_dir


def test_anchors_are_collected_with_their_current_ratings(tmp_path):
    root, global_dir = anchor_store(tmp_path, count=6, series=2)

    anchors = collect_anchors(root, global_dir)

    assert len(anchors) == 6
    assert {a.series for a in anchors} == {"benchmark_1", "benchmark_2"}
    assert sorted(a.rating for a in anchors) == [1500.0, 1500.0, 1501.0, 1501.0, 1502.0, 1502.0]
    assert all(a.delta == 0.0 for a in anchors), "nothing has been played yet"


def test_an_unregistered_anchor_starts_at_the_default_rating(tmp_path):
    root, global_dir = anchor_store(tmp_path, count=2, series=1)
    directory = root / "benchmark" / "benchmark_1"
    (directory / "newcomer.pt").write_bytes(b"weights")

    anchors = {a.label: a for a in collect_anchors(root, global_dir)}

    assert anchors["newcomer"].rating == 1500.0


def test_each_anchor_moves_by_the_k_of_its_own_experience(tmp_path):
    """K always follows the hyperbolic staircase: a fresh anchor moves far more
    than one with tens of thousands of games, from the same session."""
    root, global_dir = anchor_store(tmp_path, count=2, series=1)
    anchors = {a.label: a for a in collect_anchors(root, global_dir)}
    anchors["s1-m0"].games = 0
    anchors["s1-m1"].games = 100_000

    assert apply_sessions(anchors, [{"s1-m0": 100.0, "s1-m1": -100.0}]) == 1

    fresh, veteran = anchors["s1-m0"], anchors["s1-m1"]
    assert fresh.delta > 0 > veteran.delta
    assert abs(fresh.delta) > 100 * abs(veteran.delta)
    assert (fresh.games, veteran.games) == (1, 100_001)


def test_the_k_schedule_of_the_config_file_is_the_one_the_anchors_are_rated_with(tmp_path):
    """One scale for the whole fleet: the staircase the file sets, not a private copy."""
    root, global_dir = anchor_store(tmp_path, count=2, series=1)
    anchors = {a.label: a for a in collect_anchors(root, global_dir)}

    apply_sessions(anchors, [{"s1-m0": 100.0, "s1-m1": -100.0}], ((0, 2.0),))

    # K = 2 for both, one pair, one opponent: a win from near-equal ratings (the
    # fixture gives them 1500 and 1501) moves each by about K / 2.
    assert anchors["s1-m0"].delta == pytest.approx(1.0, abs=0.01)
    assert anchors["s1-m1"].delta == pytest.approx(-1.0, abs=0.01)


def test_a_session_with_only_one_known_participant_is_ignored(tmp_path):
    root, global_dir = anchor_store(tmp_path, count=2, series=1)
    anchors = {a.label: a for a in collect_anchors(root, global_dir)}

    applied = apply_sessions(anchors, [{"s1-m0": 100.0, "someone-else": -100.0}])

    assert applied == 0
    assert all(a.delta == 0.0 for a in anchors.values())


def test_writing_changes_the_rating_and_nothing_else(tmp_path):
    root, global_dir = anchor_store(tmp_path, count=2, series=1)
    anchors = collect_anchors(root, global_dir)
    before = read_member(global_dir, anchors[0].label)
    anchors[0].rating += 42.0

    assert write_ratings(global_dir, anchors, machine="t", lock_ttl=5) == 2

    after = read_member(global_dir, anchors[0].label)
    assert after.rating == before.rating + 42.0
    assert (after.games, after.ref, after.kind) == (before.games, before.ref, before.kind)
    assert after.frozen, "an anchor must still be an anchor afterwards"


def test_an_anchor_stays_unreachable_to_everything_else(tmp_path):
    """The guarantee this module is the exception to: an ordinary rated session
    scores an anchor normally but never applies its delta."""
    _root, global_dir = anchor_store(tmp_path, count=2, series=1)
    members = {
        "a": PoolMember(label="a", kind=MODEL, ref="a.pt", rating=1500.0, frozen=True),
        "b": PoolMember(label="b", kind=MODEL, ref="b.pt", rating=1500.0),
    }
    registry = PoolRegistry(
        directory=global_dir, max_models=10, members=members, k_schedule=DEFAULT_K_SCHEDULE
    )

    registry.record_session({"a": 500.0, "b": -500.0})

    assert members["a"].rating == 1500.0, "frozen: unchanged even though it won"
    assert members["b"].rating < 1500.0
    assert members["a"].games == 1, "but it still counts as having played"


def test_a_run_can_be_undone_from_its_backup(tmp_path):
    root, global_dir = anchor_store(tmp_path, count=4, series=2)
    anchors = collect_anchors(root, global_dir)
    original = {a.label: a.rating for a in anchors}

    backup = save_backup(global_dir, anchors)
    for anchor in anchors:
        anchor.rating += 77.0
    write_ratings(global_dir, anchors, machine="t", lock_ttl=5)
    assert read_member(global_dir, anchors[0].label).rating == original[anchors[0].label] + 77.0

    assert restore(global_dir, backup, machine="t", lock_ttl=5) == 4

    for label, rating in original.items():
        assert read_member(global_dir, label).rating == rating
    assert json.loads(backup.read_text()) == original


def test_the_table_groups_by_folder_and_shows_every_model(tmp_path):
    """Asked for explicitly: every model in every benchmark folder, not a summary."""
    root, global_dir = anchor_store(tmp_path, count=6, series=2)
    anchors = collect_anchors(root, global_dir)
    anchors[0].rating += 30.0

    text = "\n".join(format_table(anchors))

    assert "benchmark_1" in text and "benchmark_2" in text
    for anchor in anchors:
        assert anchor.label in text
    assert "+30.0" in text


def test_a_restored_backup_does_not_need_the_checkpoints(tmp_path):
    """Restoring is pure bookkeeping: it must work even if the files moved."""
    root, global_dir = anchor_store(tmp_path, count=2, series=1)
    anchors = collect_anchors(root, global_dir)
    backup = save_backup(global_dir, anchors)
    write_ratings(
        global_dir,
        [Anchor(label=a.label, path=a.path, series=a.series, start=a.start, rating=9999.0)
         for a in anchors],
        machine="t", lock_ttl=5,
    )

    assert restore(global_dir, backup, machine="t", lock_ttl=5) == 2
    assert read_member(global_dir, anchors[0].label).rating == anchors[0].start


# ---- the convergence rule --------------------------------------------------


def test_drift_is_net_movement_not_the_distance_travelled():
    """A rating jittering around a settled value has not moved: with a flat K Elo
    never stops moving, so "stop when every delta is small" would never fire. Net
    drift ignores a model that went up three and back down three."""
    from pokerlab.rl.benchmark_arena import drift

    jitter = [(0, {"a": 1500.0}), (100, {"a": 1503.0}), (200, {"a": 1497.0}),
              (300, {"a": 1500.5})]
    climbing = [(0, {"a": 1500.0}), (100, {"a": 1510.0}), (200, {"a": 1520.0}),
                (300, {"a": 1530.0})]
    assert drift(jitter, window=300)[0] == pytest.approx(0.5)
    assert drift(climbing, window=300)[0] == pytest.approx(30.0)


def test_drift_is_infinite_until_a_checkpoint_a_window_old_exists():
    """There is nothing to compare with before the run is `window` sessions old."""
    from pokerlab.rl.benchmark_arena import drift

    assert drift([(0, {"a": 1500.0})], window=500) == (float("inf"), float("inf"))
    assert drift([(0, {"a": 1500.0}), (400, {"a": 1500.0})], window=500) == (
        float("inf"), float("inf"))
    assert drift([(0, {"a": 1500.0}), (500, {"a": 1500.0})], window=500) == (0.0, 0.0)


def test_drift_compares_with_the_latest_checkpoint_at_least_a_window_old():
    from pokerlab.rl.benchmark_arena import drift

    history = [(0, {"a": 1500.0}), (200, {"a": 1520.0}), (400, {"a": 1530.0}),
               (600, {"a": 1531.0})]
    # window 350: the reference is the checkpoint at 200 (600 - 350 = 250 >= 200).
    assert drift(history, window=350)[0] == pytest.approx(11.0)


def test_drift_reports_the_worst_anchor_not_the_average():
    """One anchor still moving means the order is not settled however quiet the
    others are."""
    from pokerlab.rl.benchmark_arena import drift

    history = [(0, {"a": 1500.0, "b": 1500.0, "c": 1500.0}),
               (300, {"a": 1500.0, "b": 1500.0, "c": 1512.0})]
    worst, average = drift(history, window=300)
    assert worst == pytest.approx(12.0)
    assert average == pytest.approx(4.0)


def test_drift_ignores_an_anchor_that_was_not_there_before():
    from pokerlab.rl.benchmark_arena import drift

    history = [(0, {"a": 1500.0}), (300, {"a": 1501.0, "new": 1600.0})]
    assert drift(history, window=300)[0] == pytest.approx(1.0)


def test_the_arena_stops_when_the_ratings_settle_but_not_before_the_minimum(
    tmp_path, monkeypatch, capsys
):
    """Convergence is judged in sessions: no anchor moved more than the tolerance
    over the window, and at least the minimum has been played."""
    import pokerlab.rl.benchmark_arena as module

    root, global_dir = anchor_store(tmp_path)
    played: list[int] = []
    monkeypatch.setattr(
        module,
        "play_global_sessions",
        lambda candidates, mix, *, sessions, **k: played.append(sessions) or [
            {candidates[0].label: 0.0, candidates[1].label: 0.0}
        ] * sessions,
    )
    monkeypatch.setattr(module, "write_snapshot", lambda *a, **k: True)

    assert module.main([
        "--config", "", *SIX_SEATS, "--root", str(root), "--global-dir", str(global_dir),
        "--arena-max-sessions", "100", "--arena-min-sessions", "8",
        "--arena-window", "4", "--arena-tolerance", "1000", "--dry-run",
        "--arena-checkpoint-sessions", "2",
    ]) == 0

    # Tolerance so loose that only the minimum holds it back: it stops at 8, not
    # at the window (4) and not at the cap (100).
    assert sum(played) == 8
    assert "CONVERGENZA" in capsys.readouterr().out


def test_hitting_the_cap_is_reported_as_not_converged(tmp_path, monkeypatch, capsys):
    import pokerlab.rl.benchmark_arena as module

    root, global_dir = anchor_store(tmp_path)
    monkeypatch.setattr(
        module,
        "play_global_sessions",
        lambda candidates, mix, *, sessions, **k: [
            {candidates[0].label: 100.0, candidates[1].label: -100.0}
        ] * sessions,
    )
    monkeypatch.setattr(module, "write_snapshot", lambda *a, **k: True)

    assert module.main([
        "--config", "", *SIX_SEATS, "--root", str(root), "--global-dir", str(global_dir),
        "--arena-max-sessions", "6", "--arena-min-sessions", "2",
        "--arena-window", "2", "--arena-tolerance", "0.0001", "--dry-run",
        "--arena-checkpoint-sessions", "2",
    ]) == 0
    assert "NON CONVERGE" in capsys.readouterr().out


# ---- keeping the shared snapshot current --------------------------------------


# The test store holds six anchors: a six-handed table is the largest it seats.
SIX_SEATS = ["--table-weights", "0", "0", "0", "0", "1", "0", "0", "0"]


def drive_checkpoints(monkeypatch, argv, *, checkpoint=2):
    """Drive `main` with the play stubbed out, reporting every snapshot refresh.

    Only the choreography is under test here -- how often the snapshot is
    rewritten -- so every session is a draw between two anchors and the ratings
    barely move. `checkpoint` shrinks the batch size so a handful of sessions
    spans several of them.
    """
    import pokerlab.rl.benchmark_arena as module

    refreshes: list[dict] = []
    monkeypatch.setattr(
        module,
        "play_global_sessions",
        lambda candidates, mix, *, sessions, **k: [
            {candidates[0].label: 0.0, candidates[1].label: 0.0}
        ] * sessions,
    )
    monkeypatch.setattr(
        module,
        "write_snapshot",
        lambda global_dir, **kwargs: bool(refreshes.append(kwargs)) or True,
    )
    assert module.main([
        "--config", "", *SIX_SEATS, "--arena-checkpoint-sessions", str(checkpoint), *argv,
    ]) == 0
    return refreshes


def test_every_checkpoint_refreshes_the_shared_snapshot(tmp_path, monkeypatch):
    """The member files are the truth, but nothing reads them one at a time:
    `--status`, the dashboard and the GUI all read `registry.json`. A long
    run can hold the anchors for days, so refreshing the snapshot only at the end
    would leave the whole fleet reporting the ratings the run started from."""
    root, global_dir = anchor_store(tmp_path)

    refreshes = drive_checkpoints(monkeypatch, [
        "--root", str(root), "--global-dir", str(global_dir),
        "--arena-max-sessions", "6", "--arena-min-sessions", "6",
    ])

    # Six sessions in batches of two: three checkpoints, plus the final write.
    assert len(refreshes) >= 4, "one refresh per checkpoint, not just at the end"
    # Forced: the staleness rule `write_snapshot` applies by default would skip
    # checkpoints that finish inside its window.
    assert all(call["force"] for call in refreshes)


def test_a_dry_run_refreshes_nothing(tmp_path, monkeypatch):
    """`--dry-run` plays and reports without writing, and the snapshot is a
    write like any other."""
    root, global_dir = anchor_store(tmp_path)

    refreshes = drive_checkpoints(monkeypatch, [
        "--root", str(root), "--global-dir", str(global_dir),
        "--arena-max-sessions", "6", "--arena-min-sessions", "6",
        "--dry-run",
    ])

    assert refreshes == []


def test_the_checkpoint_line_shows_each_series_span_and_shift(tmp_path):
    from pokerlab.rl.benchmark_arena import format_series_line

    root, global_dir = anchor_store(tmp_path, count=4, series=2)
    line = format_series_line(collect_anchors(root, global_dir))
    assert "b1" in line and "b2" in line


def test_the_config_file_sets_the_arena_and_a_flag_beats_it(tmp_path, monkeypatch):
    """Convergence (tolerance, window, minimum and maximum, all in sessions) and the
    session length come from `config.toml`, which is also how the arena the loop
    requests for a new anchor is parameterised."""
    import pokerlab.rl.benchmark_arena as module

    path = tmp_path / "config.toml"
    path.write_text(
        "arena_max_sessions = 700\narena_tolerance = 0.25\narena_window = 300\n"
        "arena_min_sessions = 500\narena_checkpoint_sessions = 50\nsession_hands = 40\nlr = 0.01\n",
        encoding="utf-8",
    )
    args = module.resolve_cli(module.build_parser(), ["--config", str(path)], lenient=True)
    assert (args.arena_max_sessions, args.arena_tolerance) == (700, 0.25)
    assert (args.arena_window, args.arena_min_sessions) == (300, 500)
    assert args.arena_checkpoint_sessions == 50
    assert args.session_hands == 40

    args = module.resolve_cli(
        module.build_parser(), ["--config", str(path), "--arena-max-sessions", "900"], lenient=True
    )
    assert args.arena_max_sessions == 900


def test_the_arena_never_plays_past_the_maximum(tmp_path, monkeypatch):
    """The cap is exact: the last batch takes whatever is left of it."""
    root, global_dir = anchor_store(tmp_path)
    played: list[int] = []

    import pokerlab.rl.benchmark_arena as module

    monkeypatch.setattr(
        module,
        "play_global_sessions",
        lambda candidates, mix, *, sessions, **k: played.append(sessions) or [
            {candidates[0].label: 0.0, candidates[1].label: 0.0}
        ] * sessions,
    )
    monkeypatch.setattr(module, "write_snapshot", lambda *a, **k: True)

    assert module.main([
        "--config", "", *SIX_SEATS, "--root", str(root), "--global-dir", str(global_dir),
        "--arena-max-sessions", "10", "--arena-min-sessions", "10",
        "--arena-tolerance", "-1", "--dry-run", "--arena-checkpoint-sessions", "4",
    ]) == 0

    assert played == [4, 4, 2]


def test_zero_sessions_only_refreshes_the_snapshot(tmp_path, monkeypatch):
    root, global_dir = anchor_store(tmp_path)
    import pokerlab.rl.benchmark_arena as module

    played: list[int] = []
    refreshes: list[dict] = []
    monkeypatch.setattr(module, "play_global_sessions", lambda *a, **k: played.append(1) or [])
    monkeypatch.setattr(
        module, "write_snapshot", lambda global_dir, **k: bool(refreshes.append(k)) or True
    )

    assert module.main([
        "--config", "", *SIX_SEATS, "--root", str(root), "--global-dir", str(global_dir),
        "--arena-max-sessions", "0",
    ]) == 0

    assert played == [] and len(refreshes) == 1


def test_a_checkpoint_must_hold_at_least_one_session(tmp_path):
    import pokerlab.rl.benchmark_arena as module

    root, global_dir = anchor_store(tmp_path)
    assert module.main([
        "--config", "", "--root", str(root), "--global-dir", str(global_dir),
        "--arena-checkpoint-sessions", "0",
    ]) == 2
