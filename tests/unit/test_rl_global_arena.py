"""Cross-machine population Elo: discovery, sampling, per-model locks, and the
end-of-run round that plays it, applies it, and (rarely) prunes it.

The discovery/sampling/lock/promotion helpers are pure Python and need no
torch; only `play_global_round`/`run_population_round` load real models.
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pokerlab.rl.global_arena import (
    Candidate,
    add_benchmark_candidates,
    apply_pending_population_rounds,
    delete_checkpoints,
    discover_all_copies,
    discover_benchmark_population,
    discover_population,
    prune_ghost_members,
    repair_member_refs,
    sample_population,
    top_biased_draw,
)
from pokerlab.rl.global_store import (
    acquire_locks,
    list_member_labels,
    load_global_registry,
    migrate_legacy_registry,
    read_member,
    release_locks,
    write_member,
    write_snapshot,
)
from pokerlab.rl.pool_registry import PoolMember, PoolRegistry


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
    minutes -- 27 were measured on a 13,014-session round -- while the fleet
    keeps publishing. Every model that arrives in that window is missing from
    the snapshot and was being deleted here: a live model, freshly rated against
    the frozen series, reset to the default rating with zero games. The
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


def test_legacy_registry_json_is_split_into_member_files_once(tmp_path):
    legacy = PoolRegistry(directory=tmp_path, max_models=10**9)
    legacy.members["a"] = PoolMember(label="a", kind="model", ref="a.pt", rating=1550.0, games=4)
    legacy.members["b"] = PoolMember(label="b", kind="model", ref="b.pt", frozen=True)
    legacy.save()

    assert migrate_legacy_registry(tmp_path) == 2
    assert read_member(tmp_path, "a").rating == 1550.0
    assert read_member(tmp_path, "b").frozen

    # Already migrated: a second call must not overwrite what merges have written since.
    write_member(tmp_path, PoolMember(label="a", kind="model", ref="a.pt", rating=1700.0, games=9))
    assert migrate_legacy_registry(tmp_path) == 0
    assert read_member(tmp_path, "a").rating == 1700.0


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
        return apply_pending_population_rounds(
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
    # games percentile 75 of [10, 20, 30, 40, 100] is 40: only 100 is above it.
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


# ---- play_global_round / run_population_round (need real models) ------------

torch = pytest.importorskip("torch")

from pokerlab.engine.config import GameConfig
from pokerlab.rl.global_arena import play_global_round, run_population_round
from pokerlab.rl.policy import PokerActorCritic
from pokerlab.rl.ppo import save_checkpoint

GAME = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)


def _make_checkpoint(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    save_checkpoint(path, PokerActorCritic(hidden=8, num_layers=1))


def test_play_global_round_gives_every_candidate_its_games(tmp_path):
    candidates = []
    for i in range(4):
        path = tmp_path / f"m{i}.pt"
        _make_checkpoint(path, seed=i)
        candidates.append(Candidate(label=f"m{i}", path=path))

    sessions = play_global_round(candidates, GAME, games_per_model=3, hands_per_game=2, seed=1)

    due = dict.fromkeys([c.label for c in candidates], 0)
    for session in sessions:
        for label in session:
            due[label] += 1
    assert all(count >= 3 for count in due.values())


def test_play_global_round_reports_its_progress_in_owed_games(tmp_path):
    """The round is otherwise silent for ninety minutes. Progress counts owed
    games, not sessions: a session seats several models and decrements each
    one's debt, so the owed total is exact up front while the session count is
    not -- and a bar that can overshoot its own denominator is worse than none.
    """
    candidates = []
    for i in range(4):
        path = tmp_path / f"m{i}.pt"
        _make_checkpoint(path, seed=i)
        candidates.append(Candidate(label=f"m{i}", path=path))
    seen: list[tuple[int, int, str]] = []

    sessions = play_global_round(
        candidates,
        GAME,
        games_per_model=3,
        hands_per_game=2,
        seed=1,
        on_progress=lambda done, total, detail: seen.append((done, total, detail)),
    )

    assert len(seen) == len(sessions)  # one report per session, never batched
    assert all(total == 4 * 3 for _done, total, _detail in seen)
    assert [done for done, _t, _d in seen] == sorted(done for done, _t, _d in seen)
    assert seen[-1][0] == 4 * 3  # it ends at exactly 100%, not near it
    assert seen[-1][2] == f"{len(sessions)} sessioni"


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


def test_run_population_round_rates_participants_from_the_whole_store(small_population):
    global_dir = small_population / "global"

    report = run_population_round(
        global_dir=global_dir,
        root=small_population,
        game=GAME,
        machine="host-a",
        population_sample=20,  # more than the 8 available: draws all of them
        benchmark_sample=1,
        games_per_model=2,
        hands_per_game=2,
        seed=1,
        trigger_size=10**9,  # never prune in this test
    )

    assert report.sessions > 0
    registry = load_global_registry(global_dir)
    assert any(label.startswith("retired") for label in registry.members)
    assert sum(m.games for m in registry.members.values()) > 0


def test_run_population_round_never_moves_a_frozen_anchors_rating(small_population):
    global_dir = small_population / "global"
    kwargs = {
        "global_dir": global_dir,
        "root": small_population,
        "game": GAME,
        "machine": "host-a",
        "population_sample": 20,
        "benchmark_sample": 1,
        "games_per_model": 2,
        "hands_per_game": 2,
        "trigger_size": 10**9,
    }

    run_population_round(seed=1, **kwargs)
    registry = load_global_registry(global_dir)
    anchor_rating_after_first = registry.members["bench0"].rating
    assert registry.members["bench0"].frozen

    run_population_round(seed=2, **kwargs)
    registry = load_global_registry(global_dir)

    assert registry.members["bench0"].rating == anchor_rating_after_first
    assert registry.members["bench0"].games > 0  # still counted, just not rated


def test_a_locked_participant_defers_its_session_instead_of_blocking_the_round(small_population):
    global_dir = small_population / "global"
    run_population_round(
        global_dir=global_dir, root=small_population, game=GAME, machine="host-a",
        population_sample=20, benchmark_sample=1, games_per_model=2, hands_per_game=2,
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
    report = apply_pending_population_rounds(
        global_dir=global_dir, root=small_population, machine="host-a",
        trigger_size=10**9, lock_wait=0.0,
    )

    assert report.deferred_sessions == 1
    assert report.sessions == 0
    assert read_member(global_dir, victim).games == games_before
    requeued = [p for p in pending.glob("*.json") if not p.name.startswith(".")]
    assert len(requeued) == 1

    release_locks(global_dir, [victim])
    report = apply_pending_population_rounds(
        global_dir=global_dir, root=small_population, machine="host-a", trigger_size=10**9
    )
    assert report.sessions == 1
    assert read_member(global_dir, victim).games == games_before + 1
    assert not [p for p in pending.glob("*.json") if not p.name.startswith(".")]


def test_a_pending_file_is_applied_exactly_once_however_many_mergers_run(small_population):
    global_dir = small_population / "global"
    run_population_round(
        global_dir=global_dir, root=small_population, game=GAME, machine="host-a",
        population_sample=20, benchmark_sample=1, games_per_model=2, hands_per_game=2,
        seed=1, trigger_size=10**9,
    )
    games_after_first = sum(m.games for m in load_global_registry(global_dir).members.values())

    # A second merge with nothing pending must change nothing.
    report = apply_pending_population_rounds(
        global_dir=global_dir, root=small_population, machine="host-b", trigger_size=10**9
    )

    assert report.sessions == 0
    assert sum(m.games for m in load_global_registry(global_dir).members.values()) == games_after_first


def test_a_round_refreshes_a_stale_ref_to_where_the_file_really_is(small_population):
    global_dir = small_population / "global"
    kwargs = {
        "global_dir": global_dir, "root": small_population, "game": GAME, "machine": "host-a",
        "population_sample": 20, "benchmark_sample": 1, "games_per_model": 2,
        "hands_per_game": 2, "trigger_size": 10**9,
    }
    run_population_round(seed=1, **kwargs)
    label = "active0"
    member = read_member(global_dir, label)
    member.ref = "checkpoints/somewhere/that/no/longer/exists.pt"
    write_member(global_dir, member)

    run_population_round(seed=2, **kwargs)

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

    report = apply_pending_population_rounds(
        global_dir=tmp_path / "global", root=tmp_path / "empty", machine="m"
    )

    assert report.pending_merged == 1  # it was requeued, claimed again and merged
    assert not list(pending.glob("*.json"))


def test_run_population_round_skips_gracefully_with_no_population(tmp_path):
    global_dir = tmp_path / "global"
    empty_root = tmp_path / "empty"

    report = run_population_round(
        global_dir=global_dir,
        root=empty_root,
        game=GAME,
        machine="host-a",
        seed=1,
    )

    assert report.sessions == 0
    assert not list((global_dir / "locks").glob("*.lock"))  # nothing left locked


def test_run_population_round_skips_an_unreadable_checkpoint(small_population):
    (small_population / "models/garbage.pt").write_text("not a checkpoint")
    global_dir = small_population / "global"
    skipped: list[str] = []

    report = run_population_round(
        global_dir=global_dir,
        root=small_population,
        game=GAME,
        machine="host-a",
        population_sample=20,
        benchmark_sample=1,
        games_per_model=2,
        hands_per_game=2,
        seed=1,
        trigger_size=10**9,
        on_skip=lambda path, why: skipped.append(str(path)),
    )

    assert any("garbage" in path for path in skipped)
    assert report.sessions > 0  # the round still completes with the rest


def test_run_population_round_prunes_and_deletes_every_doomed_model(tmp_path):
    root = tmp_path / "checkpoints"
    for i in range(8):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    global_dir = root / "global"

    report = run_population_round(
        global_dir=global_dir,
        root=root,
        game=GAME,
        machine="host-a",
        population_sample=20,
        benchmark_sample=0,
        games_per_model=2,
        hands_per_game=2,
        seed=1,
        trigger_size=7,  # 8 tracked models >= 7 fires
        eliminate_fraction=0.5,  # doomed = round(eligible * 0.5)
        protect_percentile=0,
    )

    assert report.triggered_elimination
    assert report.eliminated == 4
    assert report.deleted_files == 4
    # Nothing is promoted any more: no anchor appears, the files are just gone.
    assert len(list((root / "models").glob("*.pt"))) == 4
    assert not (root / "benchmark").exists()
    registry = load_global_registry(global_dir)
    assert len(registry.members) == 4
    assert not any(m.frozen for m in registry.members.values())


def test_run_population_round_trigger_is_the_real_disk_population_not_the_ledger(tmp_path):
    """A big backlog of *unregistered* files (the normal state of the real
    volume: thousands of files no round has sampled yet) must still trigger
    pruning -- the trigger cannot be keyed to how many models the ledger
    happens to already track."""
    root = tmp_path / "checkpoints"
    for i in range(3):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    global_dir = root / "global"
    # A near-empty ledger (well under any reasonable trigger) next to a
    # small *real* population -- trigger_size=3 must still fire because 3
    # real files exist, regardless of what the ledger currently tracks.
    empty_report = run_population_round(
        global_dir=global_dir,
        root=root,
        game=GAME,
        machine="host-a",
        population_sample=3,
        benchmark_sample=0,
        games_per_model=1,
        hands_per_game=2,
        seed=1,
        trigger_size=3,
        eliminate_fraction=0.34,
        protect_percentile=0,
    )
    assert empty_report.triggered_elimination


def test_run_population_round_does_not_trigger_below_the_real_disk_population(tmp_path):
    """The inverse: a registry pre-seeded with many entries must NOT trigger
    pruning on its own if the real, current on-disk population is small --
    otherwise stale ledger bookkeeping (from models already deleted by a
    previous round) could fire pruning against a population that has since
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

    report = run_population_round(
        global_dir=global_dir,
        root=root,
        game=GAME,
        machine="host-a",
        population_sample=3,
        benchmark_sample=0,
        games_per_model=1,
        hands_per_game=2,
        seed=1,
        trigger_size=10,  # only 3 real files exist, well under this
        eliminate_fraction=0.5,
        protect_percentile=0,
    )

    assert not report.triggered_elimination


def test_run_population_round_fires_again_when_the_population_crosses_the_trigger_again(tmp_path):
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
        "game": GAME,
        "machine": "host-a",
        "population_sample": 20,
        "benchmark_sample": 0,
        "games_per_model": 2,
        "hands_per_game": 2,
        "trigger_size": 7,
        "eliminate_fraction": 0.5,
        "protect_percentile": 0,
    }

    first = run_population_round(seed=1, **kwargs)
    assert first.triggered_elimination

    # Re-grow the backlog above the trigger; the new rule is simply "above
    # threshold -> prune attempt may fire".
    for i in range(8, 16):
        _make_checkpoint(root / f"models/active{i}.pt", seed=i)
    second = run_population_round(seed=2, **kwargs)
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


# ---- splitting a round across local processes --------------------------------


def test_shard_games_splits_exactly_and_never_overshoots():
    """The whole point: `ceil(games / workers)` per shard silently played more
    hands than were asked for whenever the division was uneven."""
    from pokerlab.rl.global_arena import shard_games

    for games in range(40):
        for workers in range(1, 20):
            per_shard = shard_games(games, workers)
            assert len(per_shard) == workers
            assert sum(per_shard) == games, (games, workers)
            # At most one game of imbalance, so no shard is a straggler.
            assert max(per_shard) - min(per_shard) <= 1


def test_shard_games_leaves_spare_shards_owing_nothing():
    from pokerlab.rl.global_arena import shard_games

    assert shard_games(3, 5) == [1, 1, 1, 0, 0]
    assert shard_games(20, 15) == [2] * 5 + [1] * 10
    assert shard_games(20, 10) == [2] * 10
    assert shard_games(0, 4) == [0, 0, 0, 0]


def test_the_module_is_runnable_as_a_shard():
    """`play_sharded` launches `python -m pokerlab.rl.global_arena`, so the module
    needs a `__main__` dispatch. It had none: every shard imported the module,
    did nothing, wrote no output, and was reported through `on_skip` as a missing
    file -- a sharded round returned *zero* sessions. Nothing in production passes
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


# ---- a draw that spends half its seats on the top ------------------------------


def _population(n):
    return [Candidate(label=f"m{i:04d}", path=Path(f"/tmp/m{i:04d}.pt")) for i in range(n)]


def _ratings(n):
    # m0000 is the best, m{n-1} the worst.
    return {f"m{i:04d}": {"rating": 2000.0 - i} for i in range(n)}


def test_half_the_seats_go_to_the_top_band():
    """`pick_parents` reads the top 100 and the ordering there was measured
    wrong, while nothing reads model #5000's exact rating. Half the seats
    reserved for the top band buy five times the evidence where it is used."""
    population, ratings = _population(1000), _ratings(1000)

    drawn = top_biased_draw(ratings, top_n=100, share=0.5)(population, 50, random.Random(0))

    assert len(drawn) == 50
    assert len({c.label for c in drawn}) == 50, "no model seated twice"
    from_top = sum(1 for c in drawn if int(c.label[1:]) < 100)
    # 25 reserved, plus however many the uniform half happens to land there.
    assert from_top >= 25


def test_the_tail_is_still_reachable():
    """Biased, not restricted: the uniform half still reaches anyone, or the
    rest of the population would never be rated again at all."""
    population, ratings = _population(1000), _ratings(1000)
    seen = set()
    for seed in range(40):
        for c in top_biased_draw(ratings, top_n=100)(population, 50, random.Random(seed)):
            seen.add(int(c.label[1:]))

    assert max(seen) > 900, "the worst-rated end of the population is never drawn"


def test_an_unrated_model_is_reachable_through_the_uniform_half():
    """A model with no rating yet is untested, not excluded -- and
    `rate_against_benchmark` gives a freshly published one a rating immediately."""
    population = _population(200)
    ratings = _ratings(100)  # the other 100 have never been rated

    seen = set()
    for seed in range(30):
        for c in top_biased_draw(ratings, top_n=50)(population, 20, random.Random(seed)):
            seen.add(c.label)

    assert any(int(label[1:]) >= 100 for label in seen)


def test_the_draw_fills_its_count_even_when_the_top_band_is_short():
    """Whatever the top half cannot fill spills into the uniform half, so a
    round is never short just because the store has few rated models."""
    population = _population(60)
    ratings = _ratings(3)  # a top band of three

    drawn = top_biased_draw(ratings, top_n=1000, share=0.5)(population, 30, random.Random(1))

    assert len(drawn) == 30
    assert len({c.label for c in drawn}) == 30


def test_a_draw_larger_than_the_population_returns_everyone():
    population, ratings = _population(10), _ratings(10)

    drawn = top_biased_draw(ratings)(population, 50, random.Random(0))

    assert {c.label for c in drawn} == {c.label for c in population}
