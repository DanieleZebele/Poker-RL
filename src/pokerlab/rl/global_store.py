"""Storage and locking for the global registry: one file and one lock per model.

The global registry used to be a single `registry.json`, rewritten whole on
every save. That forces one lock for the entire ranking, because two writers
touching *different* models would still overwrite each other's file -- and a
single lock is exactly what made merging slow enough that only a handful of
games ever reached the ratings.

Here every model owns its own file (`members/<label>.json`) and its own lock
(`locks/<label>.lock`). A writer locks only the models it is about to change,
reads and writes only those files, and unlocks; writers on disjoint models never
wait for each other. Locks are an atomic `O_CREAT | O_EXCL` create (which NFS
provides) plus an expiry, so a machine that dies holding one does not block the
others forever.

`registry.json` still exists, but only as a read-only *snapshot* of the member
files for cheap consumers (the GUI's default table, `--status`). Nothing reads it
to decide anything, so it may lag by a few minutes.

Pure Python -- no torch -- like the rest of the ranking code.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from pokerlab.rl.pool_registry import (
    DEFAULT_RATING,
    MODEL,
    REGISTRY_FILENAME,
    PoolMember,
    PoolRegistry,
)

MEMBERS_DIRNAME = "members"
LOCKS_DIRNAME = "locks"
# Every trained model lives in this one directory, written once and never
# modified: an immutable file needs no lock to be read from any machine.
DEFAULT_MODELS_DIR = Path("checkpoints/models")

# A member lock is held for the few milliseconds it takes to read and rewrite a
# handful of small files, so a short expiry is enough -- and a crashed holder
# stops blocking anyone within two minutes.
DEFAULT_LOCK_SECONDS = 120
# The pruning pass is long (it can delete thousands of files), so its exclusive
# lock lives longer. It serialises *pruners* only, never ordinary merges.
PRUNE_LOCK = "__prune__"
PRUNE_LOCK_SECONDS = 1800
SNAPSHOT_LOCK = "__snapshot__"
SNAPSHOT_MAX_AGE_SECONDS = 300


# ---- locks ----------------------------------------------------------------


def _lock_file(global_dir: Path, label: str) -> Path:
    return global_dir / LOCKS_DIRNAME / f"{label}.lock"


def _is_stale(path: Path, ttl: float) -> bool:
    try:
        taken = float(json.loads(path.read_text(encoding="utf-8")).get("taken", 0))
    except (OSError, ValueError, AttributeError):
        # Created but not written yet (or unreadable): judge by mtime instead,
        # so a lock caught mid-write is not mistaken for an abandoned one.
        try:
            taken = path.stat().st_mtime
        except OSError:
            return False
    return time.time() - taken >= ttl


def _take(global_dir: Path, label: str, machine: str, ttl: float) -> bool:
    path = _lock_file(global_dir, label)
    for _attempt in range(2):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _attempt == 0 and _is_stale(path, ttl):
                path.unlink(missing_ok=True)
                continue
            return False
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump({"machine": machine, "label": label, "taken": time.time()}, out)
        return True
    return False


def acquire_locks(
    global_dir: str | Path,
    labels: list[str] | tuple[str, ...] | set[str],
    *,
    machine: str,
    ttl: float = DEFAULT_LOCK_SECONDS,
    timeout: float = 0.0,
) -> bool:
    """Lock every label in `labels`, or none of them.

    All-or-nothing with rollback, so two callers wanting overlapping sets can
    never each hold half of what the other needs (no deadlock). With a positive
    `timeout` a failed attempt is retried after a short random pause -- the locks
    are held for milliseconds, so waiting briefly almost always succeeds.
    """
    directory = Path(global_dir)
    (directory / LOCKS_DIRNAME).mkdir(parents=True, exist_ok=True)
    ordered = sorted(set(labels))
    deadline = time.monotonic() + timeout
    while True:
        taken: list[str] = []
        for label in ordered:
            if _take(directory, label, machine, ttl):
                taken.append(label)
            else:
                break
        else:
            return True
        release_locks(directory, taken)
        if time.monotonic() >= deadline:
            return False
        time.sleep(random.uniform(0.02, 0.1))


def release_locks(global_dir: str | Path, labels: list[str] | tuple[str, ...] | set[str]) -> None:
    for label in set(labels):
        _lock_file(Path(global_dir), label).unlink(missing_ok=True)


# ---- one file per member ----------------------------------------------------


def _member_file(global_dir: Path, label: str) -> Path:
    return global_dir / MEMBERS_DIRNAME / f"{label}.json"


def _read_member_file(path: Path) -> PoolMember | None:
    try:
        return PoolMember(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def read_member(global_dir: str | Path, label: str) -> PoolMember | None:
    return _read_member_file(_member_file(Path(global_dir), label))


def write_member(global_dir: str | Path, member: PoolMember) -> None:
    """Atomically replace one member's file. The caller holds that member's lock."""
    path = _member_file(Path(global_dir), member.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    staging.write_text(json.dumps(asdict(member)), encoding="utf-8")
    staging.replace(path)


def remove_member(global_dir: str | Path, label: str) -> None:
    _member_file(Path(global_dir), label).unlink(missing_ok=True)


def list_member_labels(global_dir: str | Path) -> set[str]:
    directory = Path(global_dir) / MEMBERS_DIRNAME
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.glob("*.json") if not p.name.startswith(".")}


def read_all_members(global_dir: str | Path, *, workers: int = 16) -> dict[str, PoolMember]:
    """Every member, read in parallel -- thousands of tiny files on NFS are
    latency-bound, not bandwidth-bound, so threads help a lot."""
    directory = Path(global_dir) / MEMBERS_DIRNAME
    if not directory.is_dir():
        return {}
    paths = [p for p in directory.glob("*.json") if not p.name.startswith(".")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(_read_member_file, paths))
    return {member.label: member for member in loaded if member is not None}


def migrate_legacy_registry(global_dir: str | Path) -> int:
    """Split an old single-file `registry.json` into per-member files, once.

    Runs only while `members/` does not exist yet. The files are written into a
    staging directory that is then renamed into place, so of several processes
    migrating at once exactly one wins and nobody ever sees a half-migrated store.
    """
    directory = Path(global_dir)
    members_dir = directory / MEMBERS_DIRNAME
    if members_dir.exists() or not (directory / REGISTRY_FILENAME).is_file():
        return 0
    legacy = PoolRegistry.load(directory, max_models=10**9)
    staging = directory / f".{MEMBERS_DIRNAME}-{uuid.uuid4().hex[:8]}.tmp"
    staging.mkdir(parents=True)
    for member in legacy.members.values():
        (staging / f"{member.label}.json").write_text(json.dumps(asdict(member)), encoding="utf-8")
    try:
        staging.rename(members_dir)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        return 0
    return len(legacy.members)


def load_global_registry(global_dir: str | Path) -> PoolRegistry:
    """The whole registry, assembled from the member files."""
    directory = Path(global_dir)
    migrate_legacy_registry(directory)
    registry = PoolRegistry(directory=directory, max_models=10**9)
    registry.members = read_all_members(directory)
    return registry


def load_ranking(global_dir: str | Path) -> PoolRegistry:
    """The ranking for *reading*: the `registry.json` snapshot when there is one
    (a single file, at most a few minutes behind), else the member files.

    Many processes start at the same moment (every worker of a generation draws
    its opponents on launch), and thousands of tiny reads apiece would hammer the
    shared volume. Anything that decides *writes* -- merging, pruning -- reads
    `load_global_registry` instead.
    """
    snapshot = PoolRegistry.load(global_dir, max_models=10**9)
    if snapshot.members:
        return snapshot
    return load_global_registry(global_dir)


def write_snapshot(
    global_dir: str | Path,
    *,
    machine: str,
    max_age: float = SNAPSHOT_MAX_AGE_SECONDS,
    force: bool = False,
) -> bool:
    """Refresh `registry.json` from the member files, if it is older than
    `max_age` seconds and nobody else is already doing so. Best effort: losing
    the race, or finding it fresh enough, just returns False."""
    directory = Path(global_dir)
    path = directory / REGISTRY_FILENAME
    if not force:
        try:
            if time.time() - path.stat().st_mtime < max_age:
                return False
        except OSError:
            pass
    if not acquire_locks(directory, [SNAPSHOT_LOCK], machine=machine, ttl=600):
        return False
    try:
        load_global_registry(directory).save()
        return True
    finally:
        release_locks(directory, [SNAPSHOT_LOCK])


# ---- publishing a trained model ------------------------------------------------


def sidecar_path(checkpoint: str | Path) -> Path:
    """The small JSON that travels with a not-yet-published checkpoint."""
    return Path(checkpoint).with_suffix(".json")


def write_sidecar(checkpoint: str | Path, *, rating: float, iteration: int) -> None:
    """Record what the run knew about a checkpoint when it saved it -- its
    learner rating and iteration -- so a *salvaged* model (published later by
    someone other than the run that trained it) still enters the ranking at the
    rating it earned rather than at the baseline."""
    path = sidecar_path(checkpoint)
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(json.dumps({"rating": rating, "iteration": iteration}), encoding="utf-8")
    staging.replace(path)


def read_sidecar(checkpoint: str | Path) -> tuple[float, int]:
    try:
        payload = json.loads(sidecar_path(checkpoint).read_text(encoding="utf-8"))
        return float(payload["rating"]), int(payload["iteration"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_RATING, 0


def publish_model(
    source: str | Path,
    *,
    models_dir: str | Path,
    global_dir: str | Path,
    name: str,
    rating: float,
    games: int = 0,
    iteration: int,
    machine: str,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
) -> PoolMember | None:
    """Put a finished model into the shared store and the global ranking.

    Write-once: an existing file of that name is never replaced (returns None),
    so a label always means the same weights. The copy goes to a dotted
    `.partial` beside the target and is renamed into place, so a reader on
    another machine never sees half a checkpoint.

    The member enters the registry at `rating` and `games` -- both earned by the
    run itself, against its pool and then against the frozen anchors. **`games`
    matters as much as the rating**: it is what `k_for_games` reads, so a model
    published with the ~600 rated sessions it really played is refined gently by
    later population rounds, while one published at zero would be treated as a
    newcomer and shoved around at the schedule's top tier on evidence it already
    has. It was hard-coded to 0 while the round's sessions were queued for the
    global merge instead, which applied the same evidence a second time.
    """
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)
    target = models / name
    if target.exists():
        return None
    staging = models / f".{name}.partial"
    try:
        shutil.copy2(source, staging)
        staging.replace(target)
    except OSError:
        staging.unlink(missing_ok=True)
        raise
    member = PoolMember(
        label=target.stem,
        kind=MODEL,
        ref=str(target),
        rating=rating,
        games=games,
        iteration=iteration,
    )
    if acquire_locks(global_dir, [member.label], machine=machine, ttl=lock_ttl, timeout=lock_ttl):
        try:
            if read_member(global_dir, member.label) is None:
                write_member(global_dir, member)
        finally:
            release_locks(global_dir, [member.label])
    return member


# ---- asking for a benchmark-arena run --------------------------------------
#
# A model added to the benchmark arrives carrying the rating it earned in the
# ordinary rounds, never measured against the anchors themselves. So adding one
# leaves a request here and a `poker-loop` supervisor runs `benchmark_arena`
# *between two generations*, when its workers have exited and the machine is
# free. It is deliberately not run by the training worker that added the model:
# the job is up to 200 rounds, and a worker that blocked on it would stall its
# whole generation.
#
# One request file, not a queue: the arena rates *every* anchor, so a second
# addition arriving before the run happens is covered by the same run and simply
# refreshes the request.

ARENA_REQUEST_FILENAME = "benchmark_arena_request.json"
# Longer than the longest run the request can ask for, so a claim is returned to
# the queue only when its owner is genuinely gone rather than merely slow. A
# supervisor that dies mid-run leaves the claim behind and the next one picks it
# up after this.
ARENA_CLAIM_TTL_SECONDS = 72 * 3600


def request_benchmark_arena(global_dir: str | Path, *, added: list[str], machine: str) -> Path:
    """Record that newly added anchors need to be settled against the others.

    Written whole through a dotted temporary and renamed into place, like every
    other shared write here, so a supervisor on another machine never reads a
    half-written request.
    """
    directory = Path(global_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ARENA_REQUEST_FILENAME
    payload = {"added": added, "machine": machine, "created": time.time()}
    temporary = directory / f".{ARENA_REQUEST_FILENAME}.partial"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def claim_benchmark_arena(global_dir: str | Path, *, machine: str) -> tuple[Path, dict] | None:
    """Take the pending request, if there is one, so exactly one machine runs it.

    Claimed by an atomic rename rather than by a lock: every supervisor in the
    fleet reaches this point on its own schedule, the rename has exactly one
    winner, and the claim then survives a run measured in days -- which no lock
    TTL sensibly could. A claim older than `ARENA_CLAIM_TTL_SECONDS` is returned
    to the queue, since its owner cannot still be working on it.

    Returns the claim path (to be passed back to `finish_benchmark_arena`) and
    the request payload, or None when there is nothing to do.
    """
    directory = Path(global_dir)
    for stale in directory.glob(f"{ARENA_REQUEST_FILENAME}.claim-*"):
        if ".failed-" in stale.name:
            continue  # kept for a human to look at, never re-queued
        try:
            if time.time() - stale.stat().st_mtime > ARENA_CLAIM_TTL_SECONDS:
                stale.replace(directory / ARENA_REQUEST_FILENAME)
        except OSError:
            continue
    path = directory / ARENA_REQUEST_FILENAME
    claim = directory / f"{ARENA_REQUEST_FILENAME}.claim-{machine}-{uuid.uuid4().hex[:8]}"
    try:
        path.rename(claim)
    except OSError:
        return None  # nothing pending, or another machine won the race
    try:
        return claim, json.loads(claim.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return claim, {}


def finish_benchmark_arena(claim: str | Path, *, failed: bool = False) -> None:
    """Retire a claim: removed when the run succeeded, kept for a human when it
    did not.

    A failed run is *not* returned to the queue. Whatever broke would break
    again on the next generation, and a request that re-runs a long job every few
    hours is worse than one that stops and leaves evidence.
    """
    claim = Path(claim)
    if failed:
        # Appended, not `with_suffix`, which would replace the `.claim-<machine>`
        # part and throw away who was running it.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        claim.replace(claim.with_name(f"{claim.name}.failed-{stamp}"))
    else:
        claim.unlink(missing_ok=True)
