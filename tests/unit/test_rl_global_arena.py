"""Cross-machine population Elo: discovery, sampling, per-model locks, and the
end-of-run pass that plays it, applies it, and (rarely) prunes it.

The discovery/sampling/lock/promotion helpers are pure Python and need no
torch; only `play_global_sessions`/`run_population_sessions` load real models.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pokerlab.rl.global_arena import (
    Candidate,
    add_benchmark_candidates,
    apply_pending_population_sessions,
    delete_checkpoints,
    discover_all_copies,
    discover_benchmark_population,
    discover_population,
    draw_tiers_text,
    prune_ghost_members,
    repair_member_refs,
    sample_population,
    tiered_draw,
)
from pokerlab.rl.global_store import (
    acquire_locks,
    list_member_labels,
    load_global_registry,
    read_member,
    release_locks,
    write_member,
    write_snapshot,
)
from pokerlab.rl.pool_registry import PoolMember, PoolRegistry, parse_k_schedule


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real checkpoint")


# ---- discover_population --------------------------------------------------------


def test_discover_population_lists_every_model_in_the_shared_store(tmp_path):
    _touch(tmp_path / "models/host-a-gen0001-w00-agent-x.pt")
    _touch(tmp_path / "models/host-b-gen0001-w00-agent-x.pt")

    population = discover_population(tmp_path)

    # The machine prefix is part of the identity: two hosts' runs never collide.
    assert sorted(c.label for c in population) == [
        "host-a-gen0001-w00-agent-x",
        "host-b-gen0001-w00-agent-x",
    ]


def test_discover_population_ignores_in_flight_partial_files_and_other_directories(tmp_path):
    _touch(tmp_path / "models/.host-a-model.pt.partial")
    _touch(tmp_path / "models/.hidden.pt")
    _touch(tmp_path / "machines/host-a/pool/old.pt")  # the old layout is not the store
    _touch(tmp_path / "exchange/old.pt")
    _touch(tmp_path / "benchmark/anchor.pt")  # anchors are found separately

    assert discover_population(tmp_path) == []


def test_discover_population_is_empty_when_there_is_no_store(tmp_path):
    assert discover_population(tmp_path) == []


# ---- discover_all_copies -----------------------------------------------------


def test_discover_all_copies_maps_each_label_to_its_single_file(tmp_path):
    _touch(tmp_path / "models/one.pt")
    _touch(tmp_path / "models/two.pt")

    copies = discover_all_copies(tmp_path)

    assert copies == {"one": [tmp_path / "models/one.pt"], "two": [tmp_path / "models/two.pt"]}


# ---- delete_checkpoints -------------------------------------------------------


def test_delete_checkpoints_removes_every_listed_file(tmp_path):
    a = tmp_path / "models/doomed.pt"
    b = tmp_path / "models/doomed-extra.pt"
    for path in (a, b):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    deleted = delete_checkpoints(
        [PoolMember(label="doomed", kind="model", ref=str(a))], {"doomed": [a, b]}
    )

    assert deleted == 2
    assert not a.exists()
    assert not b.exists()


def test_delete_checkpoints_skips_a_label_with_no_known_copies(tmp_path):
    skipped = []
    deleted = delete_checkpoints(
        [PoolMember(label="gone", kind="model", ref="nowhere.pt")],
        {},
        on_skip=lambda label, why: skipped.append(label),
    )

    assert deleted == 0
    assert skipped == ["gone"]


# ---- prune_ghost_members -------------------------------------------------------


def test_prune_ghost_members_drops_members_missing_from_the_existing_labels(tmp_path):
    write_member(tmp_path, PoolMember(label="real", kind="model", ref="real.pt"))
    write_member(tmp_path, PoolMember(label="ghost", kind="model", ref="ghost.pt", frozen=True))

    dropped = prune_ghost_members(tmp_path, {"real"}, machine="a")

    assert dropped == 1
    assert list_member_labels(tmp_path) == {"real"}


def test_prune_ghost_members_reports_zero_when_nothing_is_stale(tmp_path):
    write_member(tmp_path, PoolMember(label="real", kind="model", ref="real.pt"))

    assert prune_ghost_members(tmp_path, {"real", "other"}, machine="a") == 0
    assert list_member_labels(tmp_path) == {"real"}


def _store_with(root, models=(), anchors=()):
    (root / "models").mkdir(parents=True, exist_ok=True)
    for label in models:
        (root / "models" / f"{label}.pt").write_bytes(b"weights")
    for label in anchors:
        directory = root / "benchmark" / "benchmark_1"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{label}.pt").write_bytes(b"weights")
    return root


def test_a_model_published_during_the_merge_is_not_a_ghost(tmp_path):
    """`existing_labels` is listed once and the merge then runs for tens of
    minutes while the fleet keeps publishing. Every model that arrives in that
    window is missing from the snapshot and would be deleted here: a live model,
    freshly rated, reset to the default rating with zero games. The
    re-check under the lock is what makes the snapshot merely a shortlist.
    """
    global_dir = tmp_path / "global"
    root = _store_with(tmp_path / "checkpoints", models=["old", "published_during_merge"])
    for label in ("old", "published_during_merge"):
        write_member(global_dir, PoolMember(label=label, kind="model", ref="x.pt"))

    # The snapshot predates the newcomer, exactly as a real merge's does.
    dropped = prune_ghost_members(global_dir, {"old"}, machine="a", root=root)

    assert dropped == 0
    assert list_member_labels(global_dir) == {"old", "published_during_merge"}


def test_an_anchor_is_found_in_its_series_not_only_in_the_store(tmp_path):
    """A promoted model lives in `benchmark/benchmark_<N>/`, so a check that
    only looked in `models/` would delete every anchor's rating -- and an
    anchor's rating is the fixed point the whole scale is measured against."""
    global_dir = tmp_path / "global"
    root = _store_with(tmp_path / "checkpoints", anchors=["anchor"])
    write_member(global_dir, PoolMember(label="anchor", kind="model", ref="x.pt", frozen=True))

    assert prune_ghost_members(global_dir, set(), machine="a", root=root) == 0
    assert prune_ghost_members(global_dir, {"unrelated"}, machine="a", root=root) == 0
    assert list_member_labels(global_dir) == {"anchor"}


def test_a_checkpoint_really_gone_is_still_dropped(tmp_path):
    """The re-check must not turn the sweep off: a member whose file is gone
    from both directories is still a ghost and still goes."""
    global_dir = tmp_path / "global"
    root = _store_with(tmp_path / "checkpoints", models=["alive"])
    for label in ("alive", "deleted"):
        write_member(global_dir, PoolMember(label=label, kind="model", ref="x.pt"))

    dropped = prune_ghost_members(global_dir, {"alive"}, machine="a", root=root)

    assert dropped == 1
    assert list_member_labels(global_dir) == {"alive"}


def test_prune_ghost_members_does_nothing_when_the_disk_shows_no_models_at_all(tmp_path):
    """An empty listing means the volume could not be read, not that every
    model was deleted: wiping the whole ledger over it would be catastrophic."""
    write_member(tmp_path, PoolMember(label="real", kind="model", ref="real.pt"))

    assert prune_ghost_members(tmp_path, set(), machine="a") == 0
    assert list_member_labels(tmp_path) == {"real"}


# ---- per-model member store ------------------------------------------------------


def test_member_store_round_trips_a_member(tmp_path):
    member = PoolMember(label="m1", kind="model", ref="x/m1.pt", rating=1612.5, games=7, frozen=True)
    write_member(tmp_path, member)

    assert read_member(tmp_path, "m1") == member
    assert read_member(tmp_path, "missing") is None
    assert load_global_registry(tmp_path).members == {"m1": member}


def test_snapshot_mirrors_the_member_files_and_is_not_rewritten_while_fresh(tmp_path):
    write_member(tmp_path, PoolMember(label="a", kind="model", ref="a.pt", rating=1600.0))

    assert write_snapshot(tmp_path, machine="m") is True
    assert PoolRegistry.load(tmp_path).members["a"].rating == 1600.0

    write_member(tmp_path, PoolMember(label="a", kind="model", ref="a.pt", rating=1650.0))
    assert write_snapshot(tmp_path, machine="m") is False  # still fresh
    assert PoolRegistry.load(tmp_path).members["a"].rating == 1600.0
    assert write_snapshot(tmp_path, machine="m", force=True) is True
    assert PoolRegistry.load(tmp_path).members["a"].rating == 1650.0


# ---- discover_benchmark_population / next_benchmark_dir ---------------------


def test_discover_benchmark_population_finds_loose_and_nested_series_files(tmp_path):
    _touch(tmp_path / "benchmark/bench-seed1.pt")
    _touch(tmp_path / "benchmark/benchmark_1/promoted-a.pt")
    _touch(tmp_path / "benchmark/benchmark_2/promoted-b.pt")

    labels = {c.label for c in discover_benchmark_population(tmp_path)}

    assert labels == {"bench-seed1", "promoted-a", "promoted-b"}


def test_discover_benchmark_population_is_empty_when_nothing_exists(tmp_path):
    assert discover_benchmark_population(tmp_path) == []


def test_discover_benchmark_population_ignores_anything_outside_the_benchmark_dir(tmp_path):
    """Only `checkpoints/benchmark/` holds anchors -- a same-named sibling
    directory (the old, pre-nesting layout) or a stray file is not one."""
    _touch(tmp_path / "benchmark_1/old-layout.pt")
    _touch(tmp_path / "benchmark_stray.pt")

    assert discover_benchmark_population(tmp_path) == []


# ---- sample_population --------------------------------------------------------


def test_sample_population_returns_everything_when_asked_for_more_than_it_has():
    population = [Candidate(label=f"m{i}", path=None) for i in range(3)]
    assert sample_population(population, {}, 10, random.Random(0)) == population


def test_sample_population_prefers_the_least_played_half():
    population = [Candidate(label=f"m{i}", path=None) for i in range(10)]
    ratings = {f"m{i}": {"games": i} for i in range(10)}

    chosen = {c.label for c in sample_population(population, ratings, 4, random.Random(0))}

    assert {"m0", "m1"} <= chosen


# ---- per-model locks ------------------------------------------------------------


def test_locks_are_exclusive_per_model_until_released(tmp_path):
    assert acquire_locks(tmp_path, ["m1"], machine="a") is True
    assert acquire_locks(tmp_path, ["m1"], machine="b") is False
    release_locks(tmp_path, ["m1"])
    assert acquire_locks(tmp_path, ["m1"], machine="b") is True


def test_locks_on_disjoint_models_never_block_each_other(tmp_path):
    assert acquire_locks(tmp_path, ["m1", "m2"], machine="a") is True
    assert acquire_locks(tmp_path, ["m3"], machine="b") is True
    release_locks(tmp_path, ["m1", "m2", "m3"])


def test_a_failed_multi_lock_acquire_holds_nothing(tmp_path):
    assert acquire_locks(tmp_path, ["m2"], machine="a") is True

    assert acquire_locks(tmp_path, ["m1", "m2", "m3"], machine="b") is False

    # m1 was taken before m2 failed: it must have been handed back.
    assert acquire_locks(tmp_path, ["m1", "m3"], machine="c") is True


def test_an_expired_lock_is_reclaimed(tmp_path):
    assert acquire_locks(tmp_path, ["m1"], machine="a", ttl=3600) is True
    # ttl=0 means any existing lock already counts as abandoned.
    assert acquire_locks(tmp_path, ["m1"], machine="b", ttl=0) is True


def test_there_is_no_registry_wide_lock_file(tmp_path):
    acquire_locks(tmp_path, ["m1"], machine="a")
    assert not (tmp_path / "ranking.lease").exists()
    assert sorted(p.name for p in (tmp_path / "locks").iterdir()) == ["m1.lock"]


def test_repair_member_refs_follows_files_into_the_store_and_benchmark(tmp_path):
    root = tmp_path / "checkpoints"
    _touch(root / "models/a.pt")
    _touch(root / "benchmark/benchmark_1/b.pt")
    _touch(root / "models/c.pt")
    for label, ref in (("a", "models/a.pt"), ("b", "exchange/b.pt"),
                       ("c", str(root / "models/c.pt")), ("gone", "x/gone.pt")):
        write_member(root / "global", PoolMember(label=label, kind="model", ref=ref))

    assert repair_member_refs(root / "global", root, machine="m") == 2

    assert read_member(root / "global", "a").ref == str(root / "models/a.pt")
    assert read_member(root / "global", "b").ref == str(root / "benchmark/benchmark_1/b.pt")
    assert read_member(root / "global", "gone").ref == "x/gone.pt"  # untouched


# ---- concurrent merging (no torch: hand-made pending files) ---------------------


def test_many_mergers_at_once_apply_every_session_exactly_once(tmp_path):
    """The point of per-model locks and claimed files: several processes folding
    results in at the same time neither lose a session nor apply one twice."""
    root = tmp_path / "checkpoints"
    labels = [f"m{i:02d}" for i in range(12)]
    for label in labels:
        _touch(root / f"models/{label}.pt")
    global_dir = root / "global"
    pending = global_dir / "pending"
    pending.mkdir(parents=True)
    rng = random.Random(0)
    expected: dict[str, int] = dict.fromkeys(labels, 0)
    for n in range(30):
        table = rng.sample(labels, 3)
        for label in table:
            expected[label] += 1
        (pending / f"host-b-20260101-000000-{n:04d}.json").write_text(json.dumps({
            "population_draw": [
                {"label": label, "path": str(root / f"models/{label}.pt")}
                for label in table
            ],
            "benchmark_draw": [],
            "sessions": [dict(zip(table, [30.0, -10.0, -20.0]))],
        }))

    def merge(machine: str):
        return apply_pending_population_sessions(
            global_dir=global_dir, root=root, machine=machine, trigger_size=10**9
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        reports = list(pool.map(merge, [f"host-{i}" for i in range(6)]))

    assert sum(r.sessions for r in reports) == 30
    assert sum(r.deferred_sessions for r in reports) == 0
    registry = load_global_registry(global_dir)
    assert {label: m.games for label, m in registry.members.items()} == {
        label: n for label, n in expected.items() if n
    }
    assert not [p for p in pending.iterdir() if p.suffix == ".json"]
    assert not list((global_dir / "locks").glob("*.lock"))


# ---- add_benchmark_candidates ------------------------------------------------


def _store(tmp_path, anchors, models):
    """`anchors` and `models` are lists of (rating, games); models are in the store."""
    root = tmp_path / "checkpoints"
    global_dir = root / "global"
    for kind, entries in (("anchor", anchors), ("model", models)):
        for i, (rating, games) in enumerate(entries):
            label = f"{kind}{i}"
            folder = "benchmark" if kind == "anchor" else "models"
            _touch(root / folder / f"{label}.pt")
            write_member(
                global_dir,
                PoolMember(
                    label=label, kind="model", ref=str(root / folder / f"{label}.pt"),
                    rating=rating, games=games, frozen=kind == "anchor",
                ),
            )
    return root, global_dir


def test_a_well_played_model_above_the_best_anchor_is_added(tmp_path):
    # games percentile 90 of [10, 20, 30, 40, 100] is 76: only 100 is above it.
    root, global_dir = _store(
        tmp_path,
        anchors=[(1500.0, 500)],
        models=[(1400.0, 10), (1450.0, 20), (1480.0, 30), (1490.0, 40), (1520.0, 100)],
    )

    added = add_benchmark_candidates(global_dir=global_dir, root=root, machine="m")

    assert [m.label for m in added] == ["model4"]
    # Adding an anchor asks for `benchmark_arena` to settle it.
    request = json.loads((global_dir / "benchmark_arena_request.json").read_text(encoding="utf-8"))
    assert request["added"] == ["model4"] and request["machine"] == "m"
    member = read_member(global_dir, "model4")
    assert member.frozen and member.rating == 1520.0
    assert member.ref == str(root / "benchmark" / "model4.pt")
    assert (root / "benchmark" / "model4.pt").is_file()
    assert not (root / "models" / "model4.pt").exists()


def test_a_model_within_the_margin_of_the_best_anchor_is_not_added(tmp_path):
    root, global_dir = _store(
        tmp_path, anchors=[(1500.0, 500)],
        models=[(1400.0, 10), (1450.0, 20), (1480.0, 30), (1490.0, 40), (1510.0, 100)],
    )
    assert add_benchmark_candidates(global_dir=global_dir, root=root, machine="m") == []
    assert read_member(global_dir, "model4").frozen is False
    assert not (global_dir / "benchmark_arena_request.json").exists()


def test_a_strong_model_with_few_games_is_not_added(tmp_path):
    root, global_dir = _store(
        tmp_path, anchors=[(1500.0, 500)],
        models=[(1400.0, 100), (1450.0, 200), (1480.0, 300), (1490.0, 400), (1600.0, 5)],
    )
    assert add_benchmark_candidates(global_dir=global_dir, root=root, machine="m") == []


def test_added_anchors_are_more_than_the_margin_apart(tmp_path):
    root, global_dir = _store(
        tmp_path, anchors=[(1500.0, 500)],
        models=[(1400.0, 10), (1450.0, 20), (1520.0, 100), (1525.0, 110), (1540.0, 120), (1560.0, 130)],
    )
    added = add_benchmark_candidates(
        global_dir=global_dir, root=root, machine="m", games_percentile=25.0
    )
    ratings = [m.rating for m in added]
    assert ratings == [1520.0, 1540.0, 1560.0]  # 1525 is within 10 of 1520


def test_nothing_is_added_without_any_anchor(tmp_path):
    root, global_dir = _store(tmp_path, anchors=[], models=[(1600.0, 100), (1500.0, 10)])
    assert add_benchmark_candidates(global_dir=global_dir, root=root, machine="m") == []


# ---- play_global_sessions / run_population_sessions (need real models) ------------

torch = pytest.importorskip("torch")

from support import fixed_mix

from pokerlab.rl.global_arena import play_global_sessions, run_population_sessions
from pokerlab.rl.policy import PokerActorCritic
from pokerlab.rl.ppo import save_checkpoint

GAME = fixed_mix(3)


def _make_checkpoint(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    save_checkpoint(path, PokerActorCritic(hidden=8, num_layers=1))


def _four_candidates(tmp_path):
    candidates = []
    for i in range(4):
        path = tmp_path / f"m{i}.pt"
        _make_checkpoint(path, seed=i)
        candidates.append(Candidate(label=f"m{i}", path=path))
    return candidates


def test_play_global_sessions_plays_exactly_the_sessions_asked_for(tmp_path):
    """A pass is `sessions` sessions, each seating `num_players` distinct models
    drawn at random: nobody is owed a number of games."""
    candidates = _four_candidates(tmp_path)

    sessions = play_global_sessions(candidates, GAME, sessions=7, session_hands=2, seed=1)

    assert len(sessions) == 7
    labels = {c.label for c in candidates}
    for session in sessions:
        assert len(session) == 3  # distinct models at one table
        assert set(session) <= labels


def test_play_global_sessions_draws_its_tables_at_random(tmp_path):
    """Over enough sessions every candidate is seated, and the draw differs with
    the seed -- not a fixed rotation."""
    candidates = _four_candidates(tmp_path)
    kwargs = {"sessions": 12, "session_hands": 1}

    first = play_global_sessions(candidates, GAME, seed=1, **kwargs)
    second = play_global_sessions(candidates, GAME, seed=2, **kwargs)

    assert {label for session in first for label in session} == {c.label for c in candidates}
    assert [sorted(s) for s in first] != [sorted(s) for s in second]


def test_play_global_sessions_reports_its_progress_in_sessions(tmp_path):
    """The pass is otherwise silent for ninety minutes. The total is exactly the
    number of sessions asked for, known up front, so the bar ends at 100% and can
    never overshoot its own denominator."""
    candidates = _four_candidates(tmp_path)
    seen: list[tuple[int, int, str]] = []

    sessions = play_global_sessions(
        candidates,
        GAME,
        sessions=5,
        session_hands=2,
        seed=1,
        on_progress=lambda done, total, detail: seen.append((done, total, detail)),
    )

    assert len(seen) == len(sessions) == 5  # one report per session, never batched
    assert all(total == 5 for _done, total, _detail in seen)
    assert [done for done, _t, _d in seen] == [1, 2, 3, 4, 5]
    assert seen[-1][2] == "5 sessioni"


@pytest.fixture
def small_population(tmp_path):
    """8 models in the shared store (half of them named "retired..."), plus one
    (never-before-registered) benchmark model."""
    root = tmp_path / "checkpoints"
    for i in range(4):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    for i in range(4):
        _make_checkpoint(root / f"models/retired{i}.pt", seed=10 + i)
    _make_checkpoint(root / "benchmark/bench0.pt", seed=20)
    return root


def test_run_population_sessions_rates_participants_from_the_whole_store(small_population):
    global_dir = small_population / "global"

    report = run_population_sessions(
        global_dir=global_dir,
        root=small_population,
        mix=GAME,
        machine="host-a",
        population_sample=20,  # more than the 8 available: draws all of them
        benchmark_sample=1,
        sessions=6,
        session_hands=2,
        seed=1,
        trigger_size=10**9,  # never prune in this test
    )

    assert report.sessions > 0
    registry = load_global_registry(global_dir)
    assert any(label.startswith("retired") for label in registry.members)
    assert sum(m.games for m in registry.members.values()) > 0


def test_run_population_sessions_never_moves_a_frozen_anchors_rating(small_population):
    global_dir = small_population / "global"
    kwargs = {
        "global_dir": global_dir,
        "root": small_population,
        "mix": GAME,
        "machine": "host-a",
        "population_sample": 20,
        "benchmark_sample": 1,
        "sessions": 6,
        "session_hands": 2,
        "trigger_size": 10**9,
    }

    run_population_sessions(seed=1, **kwargs)
    registry = load_global_registry(global_dir)
    anchor_rating_after_first = registry.members["bench0"].rating
    assert registry.members["bench0"].frozen

    run_population_sessions(seed=2, **kwargs)
    registry = load_global_registry(global_dir)

    assert registry.members["bench0"].rating == anchor_rating_after_first
    assert registry.members["bench0"].games > 0  # still counted, just not rated


def test_a_locked_participant_defers_its_session_instead_of_blocking_the_round(small_population):
    global_dir = small_population / "global"
    run_population_sessions(
        global_dir=global_dir, root=small_population, mix=GAME, machine="host-a",
        population_sample=20, benchmark_sample=1, sessions=6, session_hands=2,
        seed=1, trigger_size=10**9,
    )
    victim = next(iter(load_global_registry(global_dir).members))
    games_before = read_member(global_dir, victim).games
    assert acquire_locks(global_dir, [victim], machine="someone-else")

    # Every session includes the locked model with a single 3-seat table, so
    # each has to be put back rather than applied -- and nothing may hang.
    pending = global_dir / "pending"
    pending.mkdir(exist_ok=True)
    others = [m for m in load_global_registry(global_dir).members if m != victim][:2]
    (pending / "host-b-20260101-000000-abc.json").write_text(json.dumps({
        "population_draw": [], "benchmark_draw": [],
        "sessions": [{victim: 10.0, others[0]: -5.0, others[1]: -5.0}],
    }))
    report = apply_pending_population_sessions(
        global_dir=global_dir, root=small_population, machine="host-a",
        trigger_size=10**9, lock_wait=0.0,
    )

    assert report.deferred_sessions == 1
    assert report.sessions == 0
    assert read_member(global_dir, victim).games == games_before
    requeued = [p for p in pending.glob("*.json") if not p.name.startswith(".")]
    assert len(requeued) == 1

    release_locks(global_dir, [victim])
    report = apply_pending_population_sessions(
        global_dir=global_dir, root=small_population, machine="host-a", trigger_size=10**9
    )
    assert report.sessions == 1
    assert read_member(global_dir, victim).games == games_before + 1
    assert not [p for p in pending.glob("*.json") if not p.name.startswith(".")]


def test_a_pending_file_is_applied_exactly_once_however_many_mergers_run(small_population):
    global_dir = small_population / "global"
    run_population_sessions(
        global_dir=global_dir, root=small_population, mix=GAME, machine="host-a",
        population_sample=20, benchmark_sample=1, sessions=6, session_hands=2,
        seed=1, trigger_size=10**9,
    )
    games_after_first = sum(m.games for m in load_global_registry(global_dir).members.values())

    # A second merge with nothing pending must change nothing.
    report = apply_pending_population_sessions(
        global_dir=global_dir, root=small_population, machine="host-b", trigger_size=10**9
    )

    assert report.sessions == 0
    assert sum(m.games for m in load_global_registry(global_dir).members.values()) == games_after_first


def test_a_pass_refreshes_a_stale_ref_to_where_the_file_really_is(small_population):
    global_dir = small_population / "global"
    kwargs = {
        "global_dir": global_dir, "root": small_population, "mix": GAME, "machine": "host-a",
        "population_sample": 20, "benchmark_sample": 1, "sessions": 6,
        "session_hands": 2, "trigger_size": 10**9,
    }
    run_population_sessions(seed=1, **kwargs)
    label = "active0"
    member = read_member(global_dir, label)
    member.ref = "checkpoints/somewhere/that/no/longer/exists.pt"
    write_member(global_dir, member)

    run_population_sessions(seed=2, **kwargs)

    assert read_member(global_dir, label).ref == str(
        small_population / "models/active0.pt"
    )


def test_a_stale_claim_is_returned_to_the_queue(tmp_path):
    pending = tmp_path / "global" / "pending"
    pending.mkdir(parents=True)
    claim = pending / ".claim-deadbeef-host-a-20260101-000000-abc.json"
    claim.write_text(json.dumps({"sessions": []}))
    old = claim.stat().st_mtime - 7200
    os.utime(claim, (old, old))

    report = apply_pending_population_sessions(
        global_dir=tmp_path / "global", root=tmp_path / "empty", machine="m"
    )

    assert report.pending_merged == 1  # it was requeued, claimed again and merged
    assert not list(pending.glob("*.json"))


def test_run_population_sessions_skips_gracefully_with_no_population(tmp_path):
    global_dir = tmp_path / "global"
    empty_root = tmp_path / "empty"

    report = run_population_sessions(
        global_dir=global_dir,
        root=empty_root,
        mix=GAME,
        machine="host-a",
        seed=1,
    )

    assert report.sessions == 0
    assert not list((global_dir / "locks").glob("*.lock"))  # nothing left locked


def test_run_population_sessions_skips_an_unreadable_checkpoint(small_population):
    (small_population / "models/garbage.pt").write_text("not a checkpoint")
    global_dir = small_population / "global"
    skipped: list[str] = []

    report = run_population_sessions(
        global_dir=global_dir,
        root=small_population,
        mix=GAME,
        machine="host-a",
        population_sample=20,
        benchmark_sample=1,
        sessions=6,
        session_hands=2,
        seed=1,
        trigger_size=10**9,
        on_skip=lambda path, why: skipped.append(str(path)),
    )

    assert any("garbage" in path for path in skipped)
    assert report.sessions > 0  # the pass still completes with the rest


def test_run_population_sessions_prunes_and_deletes_every_doomed_model(tmp_path):
    root = tmp_path / "checkpoints"
    for i in range(8):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    global_dir = root / "global"

    report = run_population_sessions(
        global_dir=global_dir,
        root=root,
        mix=GAME,
        machine="host-a",
        population_sample=20,
        benchmark_sample=0,
        sessions=30,
        session_hands=2,
        seed=1,
        trigger_size=7,  # 8 tracked models >= 7 fires
        eliminate_fraction=0.5,  # doomed = round(eligible * 0.5)
        protect_percentile=0,
    )

    assert report.triggered_elimination
    assert report.eliminated == 4
    assert report.deleted_files == 4
    # Nothing is promoted: no anchor appears, the files are just gone.
    assert len(list((root / "models").glob("*.pt"))) == 4
    assert not (root / "benchmark").exists()
    registry = load_global_registry(global_dir)
    assert len(registry.members) == 4
    assert not any(m.frozen for m in registry.members.values())


def test_run_population_sessions_trigger_is_the_real_disk_population_not_the_ledger(tmp_path):
    """A big backlog of *unregistered* files (the normal state of the real
    volume: thousands of files no pass has sampled yet) must still trigger
    pruning -- the trigger cannot be keyed to how many models the ledger
    happens to already track."""
    root = tmp_path / "checkpoints"
    for i in range(3):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    global_dir = root / "global"
    # A near-empty ledger (well under any reasonable trigger) next to a
    # small *real* population -- trigger_size=3 must still fire because 3
    # real files exist, regardless of what the ledger currently tracks.
    empty_report = run_population_sessions(
        global_dir=global_dir,
        root=root,
        mix=GAME,
        machine="host-a",
        population_sample=3,
        benchmark_sample=0,
        sessions=3,
        session_hands=2,
        seed=1,
        trigger_size=3,
        eliminate_fraction=0.34,
        protect_percentile=0,
    )
    assert empty_report.triggered_elimination


def test_run_population_sessions_does_not_trigger_below_the_real_disk_population(tmp_path):
    """The inverse: a registry pre-seeded with many entries must NOT trigger
    pruning on its own if the real, current on-disk population is small --
    otherwise stale ledger bookkeeping (from models already deleted by a
    previous pass) could fire pruning against a population that has since
    shrunk back down."""
    root = tmp_path / "checkpoints"
    for i in range(3):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    global_dir = root / "global"
    registry = PoolRegistry(directory=global_dir, max_models=10**9)
    for i in range(50):
        registry.members[f"ghost{i:03d}"] = PoolMember(
            label=f"ghost{i:03d}", kind="model", ref=f"/nowhere/ghost{i:03d}.pt", rating=1000.0, games=10
        )
    registry.save()

    report = run_population_sessions(
        global_dir=global_dir,
        root=root,
        mix=GAME,
        machine="host-a",
        population_sample=3,
        benchmark_sample=0,
        sessions=3,
        session_hands=2,
        seed=1,
        trigger_size=10,  # only 3 real files exist, well under this
        eliminate_fraction=0.5,
        protect_percentile=0,
    )

    assert not report.triggered_elimination


def test_run_population_sessions_fires_again_when_the_population_crosses_the_trigger_again(tmp_path):
    """The threshold is now a simple gate: once the real on-disk population is
    above it, a prune attempt may fire again as soon as the backlog is above the
    same threshold again."""
    root = tmp_path / "checkpoints"
    for i in range(8):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    global_dir = root / "global"
    kwargs = {
        "global_dir": global_dir,
        "root": root,
        "mix": GAME,
        "machine": "host-a",
        "population_sample": 20,
        "benchmark_sample": 0,
        "sessions": 6,
        "session_hands": 2,
        "trigger_size": 7,
        "eliminate_fraction": 0.5,
        "protect_percentile": 0,
    }

    first = run_population_sessions(seed=1, **kwargs)
    assert first.triggered_elimination

    # Re-grow the backlog above the trigger; the new rule is simply "above
    # threshold -> prune attempt may fire".
    for i in range(8, 16):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    second = run_population_sessions(seed=2, **kwargs)
    assert second.triggered_elimination


def _doomed_set(tmp_path, ratings):
    doomed, paths = [], {}
    for rating in ratings:
        label = f"m{rating}"
        path = tmp_path / f"{label}.pt"
        path.write_bytes(b"weights")
        paths[label] = [path]
        doomed.append(PoolMember(label=label, kind="model", ref=str(path), rating=rating))
    return doomed, paths


# ---- splitting a pass across local processes --------------------------------


def test_shard_sessions_splits_exactly_and_never_overshoots():
    """The whole point: `ceil(games / workers)` per shard silently played more
    hands than were asked for whenever the division was uneven."""
    from pokerlab.rl.global_arena import shard_sessions

    for games in range(40):
        for workers in range(1, 20):
            per_shard = shard_sessions(games, workers)
            assert len(per_shard) == workers
            assert sum(per_shard) == games, (games, workers)
            # At most one game of imbalance, so no shard is a straggler.
            assert max(per_shard) - min(per_shard) <= 1


def test_shard_sessions_leaves_spare_shards_owing_nothing():
    from pokerlab.rl.global_arena import shard_sessions

    assert shard_sessions(3, 5) == [1, 1, 1, 0, 0]
    assert shard_sessions(20, 15) == [2] * 5 + [1] * 10
    assert shard_sessions(20, 10) == [2] * 10
    assert shard_sessions(0, 4) == [0, 0, 0, 0]


def test_the_module_is_runnable_as_a_shard():
    """`play_sharded` launches `python -m pokerlab.rl.global_arena`, so the module
    needs a `__main__` dispatch. It had none: every shard imported the module,
    did nothing, wrote no output, and was reported through `on_skip` as a missing
    file -- a sharded pass returned *zero* sessions. Nothing in production passes
    `workers > 1`, so it stayed dormant.

    Checked by running it with no arguments: `_shard_main` requires
    `--candidates`, so a missing-argument error proves the dispatch fires, while
    the silent success of a bare import would not.
    """
    import subprocess
    import sys

    finished = subprocess.run(
        [sys.executable, "-m", "pokerlab.rl.global_arena"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert finished.returncode == 2, finished.stdout + finished.stderr
    assert "--candidates" in finished.stderr


# ---- a draw that gives each rating band a quarter of the seats ------------------------------


def _population(n):
    return [Candidate(label=f"m{i:04d}", path=Path(f"/tmp/m{i:04d}.pt")) for i in range(n)]


def _ratings(n):
    # m0000 is the best, m{n-1} the worst.
    return {f"m{i:04d}": {"rating": 2000.0 - i} for i in range(n)}


def _band(label):
    rank = int(label[1:])
    return 0 if rank < 10 else 1 if rank < 100 else 2 if rank < 1000 else 3


def test_each_band_gets_a_quarter_of_the_seats():
    """`pick_parents` reads the top 100 and the ordering there was measured
    wrong, so ranks 1-10, 11-100, 101-1000 and the rest get equal shares."""
    population, ratings = _population(3000), _ratings(3000)

    drawn = tiered_draw(ratings)(population, 48, random.Random(0))

    assert len(drawn) == 48
    assert len({c.label for c in drawn}) == 48, "no model seated twice"
    counts = [sum(1 for c in drawn if _band(c.label) == band) for band in range(4)]
    # The top ten cannot fill 12 seats: they are all there, the shortfall spills.
    assert counts[0] == 10
    assert counts[1] >= 12 and counts[2] >= 12 and counts[3] >= 12


def test_the_leftover_seats_go_to_random_bands_so_the_average_is_exact():
    population, ratings = _population(3000), _ratings(3000)
    totals = [0, 0, 0, 0]
    passes = 400
    for seed in range(passes):
        for c in tiered_draw(ratings)(population, 50, random.Random(seed)):
            totals[_band(c.label)] += 1

    # 50 seats over 4 bands is 12.5 each. The top ten are capped at 10 and the
    # ~2.5 seats they cannot fill spill over the models not yet drawn.
    assert totals[0] / passes == 10
    assert 12.0 < totals[1] / passes < 14.0
    assert 12.0 < totals[2] / passes < 14.0
    assert sum(totals) == 50 * passes


def test_the_tail_is_still_reachable():
    population, ratings = _population(3000), _ratings(3000)
    seen = set()
    for seed in range(60):
        for c in tiered_draw(ratings)(population, 50, random.Random(seed)):
            seen.add(int(c.label[1:]))

    assert max(seen) > 2900, "the worst-rated end of the population is never drawn"


def test_an_unrated_model_is_reachable_through_the_last_band():
    """A model with no rating yet is untested, not excluded -- and
    `rate_against_benchmark` gives a freshly published one a rating immediately."""
    population = _population(2000)
    ratings = _ratings(1000)  # the other 1000 have never been rated

    seen = set()
    for seed in range(30):
        for c in tiered_draw(ratings)(population, 20, random.Random(seed)):
            seen.add(c.label)

    assert any(int(label[1:]) >= 1000 for label in seen)


def test_the_draw_fills_its_count_even_when_the_top_bands_are_short():
    """Whatever a band cannot fill spills to the models not yet drawn, so a
    pass is never short just because the store has few rated models."""
    population = _population(60)
    ratings = _ratings(3)

    drawn = tiered_draw(ratings)(population, 30, random.Random(1))

    assert len(drawn) == 30
    assert len({c.label for c in drawn}) == 30


def test_a_draw_larger_than_the_population_returns_everyone():
    population, ratings = _population(10), _ratings(10)

    drawn = tiered_draw(ratings)(population, 50, random.Random(0))

    assert {c.label for c in drawn} == {c.label for c in population}


def _share_by_tier(tiers, cutoffs, seats=60, passes=300, size=3000):
    """Mean number of seats per pass that fell in each band between `cutoffs`."""
    population, ratings = _population(size), _ratings(size)
    edges = [0, *cutoffs, size]
    totals = [0] * (len(edges) - 1)
    for seed in range(passes):
        for c in tiered_draw(ratings, tiers=tiers)(population, seats, random.Random(seed)):
            rank = int(c.label[1:])
            totals[next(i for i in range(len(totals)) if rank < edges[i + 1])] += 1
    return [total / passes for total in totals]


def test_three_tiers_give_a_third_of_the_seats_each():
    shares = _share_by_tier((100, 1000, None), (100, 1000))

    assert shares == pytest.approx([20.0, 20.0, 20.0], abs=0.01)


def test_a_tier_listed_twice_gets_twice_the_seats():
    shares = _share_by_tier((100, 100, None), (100,))

    assert shares == pytest.approx([40.0, 20.0], abs=0.01)


def test_seats_that_do_not_divide_evenly_are_still_exact_on_average():
    # 50 seats over three tiers is 16.67 each.
    shares = _share_by_tier((100, 1000, None), (100, 1000), seats=50, passes=600)

    assert shares == pytest.approx([50 / 3] * 3, abs=0.6)


def test_without_an_all_tier_the_tail_is_only_reached_by_the_spill():
    population, ratings = _population(3000), _ratings(3000)

    drawn = tiered_draw(ratings, tiers=(100, 1000))(population, 40, random.Random(0))

    assert len(drawn) == 40
    assert sum(int(c.label[1:]) >= 1000 for c in drawn) == 0


def test_the_draw_tiers_text_is_the_parent_tiers_syntax():
    assert draw_tiers_text("10, 100 ,ALL") == "10, 100, all"
    with pytest.raises(argparse.ArgumentTypeError):
        draw_tiers_text("10, soon")


def test_the_merge_rates_at_the_k_schedule_it_is_given(small_population):
    global_dir = small_population / "global"
    run_population_sessions(
        global_dir=global_dir, root=small_population, mix=GAME, machine="host-a",
        population_sample=20, benchmark_sample=1, sessions=2, session_hands=2,
        seed=1, trigger_size=10**9,
    )
    members = load_global_registry(global_dir).members
    table = [label for label, m in members.items() if not m.frozen][:3]
    pending = global_dir / "pending"
    pending.mkdir(exist_ok=True)

    def moved(k_schedule: str) -> float:
        before = read_member(global_dir, table[0]).rating
        (pending / "host-b-20260101-000000-k.json").write_text(json.dumps({
            "population_draw": [], "benchmark_draw": [],
            "sessions": [dict(zip(table, [30.0, -10.0, -20.0]))],
        }))
        report = apply_pending_population_sessions(
            global_dir=global_dir, root=small_population, machine="host-a",
            trigger_size=10**9, k_schedule=parse_k_schedule(k_schedule),
        )
        assert report.sessions == 1
        return read_member(global_dir, table[0]).rating - before

    # Both calls are for a model with some games behind it, so the tier read is
    # the single one each schedule has.
    small, large = moved("0:1"), moved("0:10")
    assert small > 0 and large > 0
    assert large > small * 5


def test_play_global_sessions_seats_one_table_size_per_session_from_the_mixture(tmp_path):
    from pokerlab.rl.table_mix import TableMix

    mix = TableMix(weights=(1, 1, 1, 0, 0, 0, 0, 0), stack_min_bb=5.0, stack_max_bb=50.0,
                   small_blind=1, big_blind=2)
    candidates = _four_candidates(tmp_path)

    sessions = play_global_sessions(candidates, mix, sessions=30, session_hands=2, seed=2)

    assert len(sessions) == 30
    assert {len(session) for session in sessions} == {2, 3, 4}


def test_play_global_sessions_needs_enough_models_for_the_largest_table(tmp_path):
    from pokerlab.rl.table_mix import TableMix

    mix = TableMix(weights=(1, 0, 0, 0, 0, 1, 0, 0), stack_min_bb=5.0, stack_max_bb=50.0,
                   small_blind=1, big_blind=2)  # 2- and 7-handed
    assert play_global_sessions(_four_candidates(tmp_path), mix, sessions=3, session_hands=1) == []
