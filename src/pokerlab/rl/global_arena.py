"""One Elo scale for every model on the shared volume.

Every model on the volume lives in one store, `checkpoints/models/`, and has one
rating on one scale. (Ratings used to be private to each machine, which made them
incomparable: Shark, identical code on every host, rated 1387 on one and 1487 on
another.) This module keeps that scale honest by letting models drawn at random
from the whole population play each other, so a model's rating is earned against
everyone rather than against the few it happened to meet locally.

The registry is stored one file per model (see `rl/global_store.py`) and every
update takes a lock on the individual models it touches -- never on the registry
as a whole -- so many machines and processes can merge results at the same time.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from pokerlab.rl.global_store import (
    DEFAULT_LOCK_SECONDS,
    PRUNE_LOCK,
    PRUNE_LOCK_SECONDS,
    acquire_locks,
    list_member_labels,
    load_global_registry,
    migrate_legacy_registry,
    read_member,
    release_locks,
    remove_member,
    request_benchmark_arena,
    write_member,
    write_snapshot,
)
from pokerlab.rl.phases import ELO_MERGE, ELO_PLAY, PRUNING
from pokerlab.rl.pool_registry import (
    DEFAULT_ELIMINATION_FRACTION,
    DEFAULT_K_SCHEDULE,
    DEFAULT_POPULATION_TRIGGER,
    DEFAULT_PROTECT_PERCENTILE,
    MODEL,
    PoolMember,
    PoolRegistry,
    interpolated_percentile,
)

DEFAULT_GLOBAL_DIR = Path("checkpoints/global")
# Pruning is disabled by lifting its trigger beyond any population this store
# could reach, rather than by branching around `_eliminate`: one pruning rule,
# one place. Two callers pass it -- `poker-elo`, whose job is to refresh ratings
# and which must not delete checkpoints as a side effect of being run, and the
# Elo fill-in phase, which uses `tiered_draw` and therefore *must* not prune
# (a biased draw makes the often-seated eligible sooner and the rarely-seated
# immune, so a pass would eat the middle of the population instead of its
# bottom -- see `tiered_draw`).
NO_PRUNE_TRIGGER = 10**9
# Every frozen Elo anchor lives under this one directory of the checkpoint
# root, all in one flat directory.
BENCHMARK_DIRNAME = "benchmark"
# The shared store of every trained model (see `rl/global_store.py`).
MODELS_DIRNAME = "models"

# ~50 population + ~5 benchmark per round is the composition asked for --
# see `run_population_round`. Both draws are plain uniform random: unlike
# `sample_population`'s least-played-first bias (still available, just not
# used here), a round's whole point now includes deciding who gets deleted,
# so a uniform draw is the more honest "any tracked model could come up"
# sampling this decision deserves.
DEFAULT_POPULATION_SAMPLE = 50
DEFAULT_BENCHMARK_SAMPLE = 5
# How a model becomes a frozen anchor (see `add_benchmark_candidates`): at the
# end of a run, after the population round, every model with more rated games
# than the `BENCHMARK_GAMES_PERCENTILE` of the population and a rating more than
# `BENCHMARK_MARGIN` points above the best anchor joins the frozen set. Pruning
# no longer promotes anything: every eliminated model is deleted.
BENCHMARK_GAMES_PERCENTILE = 90.0
BENCHMARK_MARGIN = 10.0
# A thousand hands per rated session, up from a hundred. Elo reads only the
# *sign* of each pair's chip delta, so the length of a session is what decides
# how often that sign is right -- and at 100 hands 6-max it very nearly is not.
# The spread of a 100-hand chip delta is of the order of +/-290 bb/100, while the
# real gap between two adjacent models is 10-40 bb/100, so the stronger of the
# two finishes ahead about 54% of the time. That does not merely add noise: Elo
# settles at the rating that reproduces the *observed* win frequency, so a 54%
# edge equilibrates ~28 points above instead of the ~150 it deserves, and the
# whole scale comes out compressed. Ten times the hands cuts the spread by ~3.2x
# and decompresses the gaps by the same factor, which is the only lever that
# does -- lowering K shrinks the jitter around the equilibrium but cannot move
# the equilibrium itself.
#
# `games` still counts sessions, so the K schedule is untouched: each rated
# result simply carries ten times the evidence at the same K.
#
# The cost is linear and it is real. A round is ~485 sessions with the usual 55
# models drawn, so 485,000 hands instead of 48,500 -- roughly 90 minutes of a
# worker's time per generation instead of 9. `--global-games-per-model` is the
# dial to turn down if that is too much; fewer, longer sessions is the better
# trade at a fixed hand budget, which is the same argument `DEFAULT_SERIES_HANDS`
# records.
DEFAULT_HANDS_PER_GAME = 1000

# Where a round's *played* results wait to be folded into the registry --
# see play_population_round_sharded and apply_pending_population_rounds.
PENDING_DIRNAME = "pending"
DEFAULT_GLOBAL_WORKERS = 8
# How long one session waits for the handful of member locks it needs before
# being put back for a later merge. Locks are held for milliseconds, so this is
# a ceiling that is essentially never reached.
DEFAULT_LOCK_WAIT_SECONDS = 20.0
# A pending file claimed by a merger that then died is handed back to the queue
# after this long.
STALE_CLAIM_SECONDS = 3600


# Rated sessions each drawn model owes per round. Every caller takes this
# default, so the number lives in exactly one place: it used to be 500 here
# while both CLIs and `run_population_round` carried a hard-coded 12, which
# meant the constant described nothing that ever ran.
#
# **12, at the user's decision, to cut the end-of-run Elo round by 75%.**
# Simulating the `due` queue above exactly, 55 drawn models give 481 sessions at
# 50 and 128 at 12, so 481,000 hands become 128,000: **~94 -> ~25 minutes on an
# idle core, ~160 -> ~43 minutes on a loaded fleet machine** (measured 85 and
# ~50 hands/s respectively). That is the single biggest block of wall time a
# worker spends after it stops training.
#
# **What it costs, stated plainly**: a round is the only thing that turns played
# hands into ratings, so each run now contributes ~3.8x less rated evidence to
# the global ranking -- and that ranking was already power-limited near the top,
# where a 1000-hand session picks the stronger of two adjacent models only ~58%
# of the time. A model also comes up in a round about 0.5% of the time out of
# ~9,600, so its rating converges slowly to begin with.
#
# **What makes the trade defensible is where the lost evidence comes back from.**
# The Elo fill-in phase (`train.run_elo_fill_in`) has workers that finish early
# play extra rating rounds while they wait for the slowest worker of their
# generation. Those rounds cost nothing -- the cores would be idle -- so this
# change moves rating work *off* the critical path and onto time that was being
# wasted, rather than simply deleting it. If the ratings visibly stop converging,
# the dial to raise is `--fill-games-per-model` before this one.
#
# History, because the number has moved four times and one of the moves was a
# panic. It was 500 here while both CLIs and `run_population_round` carried a
# hard-coded 12, so the constant described nothing that ever ran; then 12, then
# 50 for the 3.8x evidence; then briefly back to 12 when five of seven machines
# went unresponsive hours after 50 went live, every one with at least one worker
# in `elo_play` while the only still-running machine had none. **That revert was
# not justified and this change is not a repeat of it.** The correlation is
# confounded -- a machine early in its generation both has nobody in the round
# yet *and* has had less time to hit any problem -- and the memory theory does
# not survive measurement: the round's 55 loaded models cost 203 MB on top of a
# worker's ~514 MB, so even every worker being in the round at once adds ~6 GB to
# a 30-worker machine. The cause of that incident was never established, and no
# machine could be inspected (no SSH access from the NFS server).
DEFAULT_GAMES_PER_MODEL = 12

@dataclass(frozen=True)
class Candidate:
    """One network in the population, and where to read it from."""

    label: str
    path: Path


def discover_population(root: str | Path = Path("checkpoints")) -> list[Candidate]:
    """Every model in the shared store, `checkpoints/models/*.pt`.

    A model's label is its file name without `.pt`, and there is exactly one
    file per model: the store is write-once and machine-prefixed, so the copies
    that per-machine pools, the exchange and retired archives used to multiply
    -- and the deduplication they needed -- no longer exist.
    """
    directory = Path(root) / MODELS_DIRNAME
    return [
        Candidate(label=path.stem, path=path)
        for path in sorted(directory.glob("*.pt"))
        if not path.name.startswith(".")
    ]


def discover_all_copies(root: str | Path = Path("checkpoints")) -> dict[str, list[Path]]:
    """Every file of every network, keyed by label -- a single path each.

    Kept as a mapping to lists because `add_benchmark_candidates` and
    `delete_checkpoints` take that shape; with one file per model the lists just
    hold one entry.
    """
    return {c.label: [c.path] for c in discover_population(root)}


def checkpoint_on_disk(root: str | Path, label: str) -> bool:
    """Is this one label's checkpoint on disk *right now*?

    One label, asked at the moment it matters, as opposed to a whole-directory
    listing taken earlier -- which is the entire point (see
    `prune_ghost_members`). Two stats and a small rglob: the store is flat, so
    the model is one path, and the benchmark side is ~50 files across ten
    series.
    """
    root = Path(root)
    if (root / MODELS_DIRNAME / f"{label}.pt").exists():
        return True
    return any((root / BENCHMARK_DIRNAME).rglob(f"{label}.pt"))


def prune_ghost_members(
    global_dir: str | Path,
    existing_labels: Collection[str],
    *,
    machine: str,
    root: str | Path | None = None,
    ttl: float = DEFAULT_LOCK_SECONDS,
) -> int:
    """Drop every registry member whose checkpoint no longer exists anywhere.

    A member can become a ghost in ways this module does not fully control:
    someone deletes a file by hand, or a crash leaves a promotion half-done.
    A ghost is harmless to leave (it never gets sampled again) but it clutters
    the leaderboard with permanently-dead rows, so it is swept every merge.
    Compared against `existing_labels` -- the union of `discover_population`
    and `discover_benchmark_population`, supplied by the caller -- rather than
    against `member.ref`, so a stale `ref` never makes a live model look dead.

    An empty `existing_labels` means the disk could not be read (or there is
    nothing on it), not that every model is gone, so nothing is dropped then.

    **`existing_labels` is a snapshot, and a snapshot goes stale.** The caller
    lists the disk once and then merges pending sessions, which on a busy fleet
    takes tens of minutes -- a merge of 13,014 sessions was measured at 27 --
    while five machines keep publishing new models and registering them. Every
    one of those arrives in `list_member_labels` but not in the snapshot, and
    was being deleted here as a ghost: a live model, freshly rated against the
    frozen series, silently reset to the default rating with zero games. It
    showed up as bursts of "ghosts removed" landing exactly on the longest
    rounds, at a time when nothing was deleting checkpoints at all (pruning off,
    population under the trigger), so every one of them was a false positive.
    Pass `root` and each candidate is re-checked on disk **after** its lock is
    taken, immediately before removal, which is the only moment the answer is
    authoritative; the snapshot then merely narrows the candidates.
    """
    if not existing_labels:
        return 0
    existing = set(existing_labels)
    dropped = 0
    for label in sorted(list_member_labels(global_dir) - existing):
        if not acquire_locks(global_dir, [label], machine=machine, ttl=ttl):
            continue
        try:
            if root is not None and checkpoint_on_disk(root, label):
                continue  # published while we were merging: alive, not a ghost
            remove_member(global_dir, label)
            dropped += 1
        finally:
            release_locks(global_dir, [label])
    return dropped


def repair_member_refs(
    global_dir: str | Path,
    root: str | Path = Path("checkpoints"),
    *,
    machine: str,
    ttl: float = DEFAULT_LOCK_SECONDS,
) -> int:
    """Point every member's `ref` at where its checkpoint actually is now.

    Rounds refresh the `ref` of the models they play and promotion rewrites it,
    but a model that has not played since its file moved (a migration of the
    store, say) keeps the old path until its next round. This sweeps them all at
    once. Returns how many refs were corrected. A member with no checkpoint
    anywhere is left alone (`prune_ghost_members` handles it).
    """
    root = Path(root)
    path_by_label = {
        c.label: c.path
        for c in [*discover_population(root), *discover_benchmark_population(root)]
    }
    migrate_legacy_registry(global_dir)
    fixed = 0
    for label in sorted(list_member_labels(global_dir)):
        target = path_by_label.get(label)
        if target is None:
            continue
        if not acquire_locks(global_dir, [label], machine=machine, ttl=ttl):
            continue
        try:
            member = read_member(global_dir, label)
            if member is not None and member.ref != str(target):
                member.ref = str(target)
                write_member(global_dir, member)
                fixed += 1
        finally:
            release_locks(global_dir, [label])
    return fixed


def discover_benchmark_population(root: str | Path = Path("checkpoints")) -> list[Candidate]:
    """Every frozen Elo anchor: every checkpoint in `checkpoints/benchmark/`.

    Only ever *reads*; anchors are added by `add_benchmark_candidates`.
    """
    base = Path(root) / BENCHMARK_DIRNAME
    if not base.is_dir():
        return []
    found: dict[str, Candidate] = {}
    for path in sorted(base.rglob("*.pt")):
        if path.name.startswith("."):
            continue
        found.setdefault(path.stem, Candidate(label=path.stem, path=path))
    return list(found.values())


def sample_population(
    population: list[Candidate],
    ratings: dict[str, dict],
    count: int,
    rng: random.Random,
) -> list[Candidate]:
    """Pick who plays this round: the least-played first, then at random.

    Straight uniform sampling would leave a 23,000-model population with a
    handful of games each after many rounds, and a rating from a handful of
    games is not a rating. Taking the least-played half deterministically and
    filling the rest at random gives even coverage without freezing the draw
    into the same faces every round.
    """
    if count >= len(population):
        return list(population)
    played = {c.label: ratings.get(c.label, {}).get("games", 0) for c in population}
    ordered = sorted(population, key=lambda c: (played[c.label], c.label))
    half = count // 2
    chosen = ordered[:half]
    remainder = ordered[half:]
    rng.shuffle(remainder)
    return chosen + remainder[: count - half]


# The rating bands `tiered_draw` splits the ranking into, as `(first, last)`
# ranks (0-based, `last` exclusive, `None` = to the end) with an equal quarter
# of the seats each: the top 10, the rest of the top 100, the rest of the top
# 1,000, and everybody else.
DRAW_BANDS: tuple[tuple[int, int | None], ...] = ((0, 10), (10, 100), (100, 1000), (1000, None))


def tiered_draw(
    ratings: Mapping[str, dict],
    *,
    bands: tuple[tuple[int, int | None], ...] = DRAW_BANDS,
) -> Callable[[list[Candidate], int, random.Random], list[Candidate]]:
    """A draw that gives each rating band an equal share of the seats.

    **Why bias at all.** A round seats ~50 of ~9,600 models, so a given model
    comes up about 0.5% of the time and a rating can sit on the number its
    training run published for many generations. What is read is the top:
    `pick_parents` draws from the best 100, and the ordering there was measured
    wrong (see "The ranking is wrong at the top"). With a quarter of the seats on
    each of ranks 1-10, 11-100, 101-1,000 and the rest, the best ten are seated
    in nearly every round, the next ninety about every other round, and the
    long tail keeps a quarter of the seats rather than none.

    The bands are disjoint (a model belongs to one), seats are split by largest
    remainder with the leftover seats given to random bands so the average share
    is exact, and models inside a band are drawn uniformly without repetition.
    A band too small for its seats (the top ten against 12 or 13) hands the
    shortfall to the models not yet drawn, uniformly, so the round always comes
    out at `count` while the disk holds that many models. A model with no rating
    yet belongs to the last band: it is untested, not excluded.

    **Pruning and this draw do not go together.** Eligibility for deletion is a
    percentile of `games`, so seating some models far more often than others
    makes them eligible sooner and leaves the rarely-seated tail immune (see
    `DEFAULT_PROTECT_PERCENTILE`). Pass `trigger_size=NO_PRUNE_TRIGGER` unless
    that is wanted.
    """

    def draw(
        population: list[Candidate], count: int, rng: random.Random
    ) -> list[Candidate]:
        if count >= len(population):
            return list(population)
        rated = sorted(
            (c for c in population if c.label in ratings),
            key=lambda c: (-ratings[c.label].get("rating", 0.0), c.label),
        )
        unrated = [c for c in population if c.label not in ratings]
        groups: list[list[Candidate]] = []
        for index, (first, last) in enumerate(bands):
            group = rated[first:last]
            if index == len(bands) - 1:
                group = group + unrated
            groups.append(group)

        base, extra = divmod(count, len(bands))
        quotas = [base] * len(bands)
        for index in rng.sample(range(len(bands)), extra):
            quotas[index] += 1

        chosen: list[Candidate] = []
        for group, quota in zip(groups, quotas):
            chosen.extend(rng.sample(group, min(quota, len(group))))
        taken = {c.label for c in chosen}
        rest = [c for c in population if c.label not in taken]
        chosen.extend(rng.sample(rest, min(count - len(chosen), len(rest))))
        return chosen

    return draw


def play_global_round(
    candidates: list[Candidate],
    game,
    *,
    games_per_model: int = DEFAULT_GAMES_PER_MODEL,
    hands_per_game: int = DEFAULT_HANDS_PER_GAME,
    device: str = "cpu",
    seed: int = 0,
    on_skip=None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, float]]:
    """Play `games_per_model` sessions for each candidate, opponents at random.

    Tables are drawn at random from the sampled candidates rather than by
    rating, so a model from one machine routinely faces models from the others
    -- which is the entire point: a rating only means something across machines
    if the matches cross them.

    Returns the raw per-session chip deltas; the caller folds them into the
    ratings. Same split as the local arena, and for the same reason: playing
    parallelises, rating does not.

    `on_progress(done, total, detail)` fires after every session, counting
    **games owed**, not sessions: a session seats `num_players` models and
    decrements each one's debt, so the owed total is known exactly up front
    while the session count is not (it depends on how often the random seats
    land on a model that owes nothing). An exact denominator is the whole point
    -- a progress bar that can overshoot its own estimate is worse than none.
    """
    from pokerlab.engine.table import Table
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import SeatProxy, policy_opponent

    loaded = {}
    for candidate in candidates:
        try:
            model, _checkpoint = build_model_from_checkpoint(candidate.path, device=device)
        except Exception as exc:  # noqa: BLE001 - user-owned directories
            if on_skip is not None:
                on_skip(candidate.path, str(exc))
            continue
        loaded[candidate.label] = policy_opponent(
            candidate.label, make_policy_fn(model, device=device), game
        )
    labels = sorted(loaded)
    if len(labels) < game.num_players:
        return []

    rng = random.Random(seed)
    bot_rng = random.Random(rng.random())
    proxies = [SeatProxy(f"s{i}", f"S{i}") for i in range(game.num_players)]
    table = Table(game, list(proxies), rng=rng)

    # Every model owes the same number of games; the queue is what guarantees
    # it, while the other seats are filled at random around whoever is due.
    due = {label: games_per_model for label in labels}
    # The exact denominator for progress, fixed before the first session. Not to
    # be confused with the `owed` list rebuilt inside the loop, which is who
    # still has games to play.
    owed_total = len(labels) * games_per_model
    sessions: list[dict[str, float]] = []

    while any(count > 0 for count in due.values()):
        owed = [label for label, count in due.items() if count > 0]
        seat_labels = [max(owed, key=lambda label: (due[label], rng.random()))]
        pool = [label for label in labels if label != seat_labels[0]]
        seat_labels.extend(rng.sample(pool, game.num_players - 1))

        for seat, (proxy, label) in enumerate(zip(proxies, seat_labels)):
            opponent = loaded[label]
            proxy.inner = opponent.factory(proxy.player_id, label, bot_rng)
            proxy.name = label

        deltas = [0] * game.num_players
        for _ in range(hands_per_game):
            before = list(table.stacks)
            table.play_hand()
            for seat in range(game.num_players):
                deltas[seat] += table.stacks[seat] - before[seat]
            table.stacks = [game.starting_stack] * game.num_players

        results: dict[str, float] = {}
        for label, delta in zip(seat_labels, deltas):
            results[label] = results.get(label, 0.0) + delta
        sessions.append(results)
        for label in seat_labels:
            due[label] = max(0, due[label] - 1)
        if on_progress is not None:
            on_progress(
                owed_total - sum(due.values()), owed_total, f"{len(sessions)} sessioni"
            )

    return sessions


def _shard_main() -> None:
    """Play one shard of a population round and write its sessions to JSON.

    The supervisor (`play_population_round_sharded`) runs several of these at
    once, on one machine's cores: only *playing* parallelises, so each shard
    just plays its slice of `--games-per-model` for the same candidate set and
    reports back: no registry, no lock, no rating math here at all.
    """
    import argparse

    from pokerlab.engine.config import GameConfig

    parser = argparse.ArgumentParser(description="Play one shard of a global population round.")
    parser.add_argument("--candidates", type=Path, required=True, help="JSON [{label, path}, ...]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games-per-model", type=int, default=DEFAULT_GAMES_PER_MODEL)
    parser.add_argument("--hands-per-game", type=int, default=DEFAULT_HANDS_PER_GAME)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    game = GameConfig(
        num_players=args.players, starting_stack=args.stack, small_blind=args.sb, big_blind=args.bb
    )
    raw = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = [
        Candidate(label=entry["label"], path=Path(entry["path"]))
        for entry in raw
    ]

    sessions = play_global_round(
        candidates,
        game,
        games_per_model=args.games_per_model,
        hands_per_game=args.hands_per_game,
        device=args.device,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sessions), encoding="utf-8")


def add_benchmark_candidates(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    machine: str,
    games_percentile: float = BENCHMARK_GAMES_PERCENTILE,
    margin: float = BENCHMARK_MARGIN,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
    on_skip: Callable[[Path, str], None] | None = None,
) -> list[PoolMember]:
    """Freeze every model that is both well played and stronger than the whole
    benchmark: `games` above the `games_percentile` of the population's games
    and a rating more than `margin` above the best anchor.

    Called at the end of a training run, after the population round, so the
    ratings it reads already include that round. Candidates are taken from the
    lowest rating upwards and each must clear the previously added one by more
    than `margin` too, so the anchors stay at least `margin` apart rather than a
    whole cluster of near-identical models joining at once. Returns the models
    added (empty when there are no anchors yet, nothing qualifies, or the
    percentile is undefined).

    The population is the non-frozen members whose checkpoint is in the store.
    Each added model is *moved* into `checkpoints/benchmark/` (every other copy
    deleted) and marked `frozen=True` at the rating it holds now: it is already
    rated over many sessions, so no re-settling is needed. Serialised with
    pruning under `__prune__`, so a model cannot be deleted while being added.
    Whenever it adds anything it also requests a `benchmark_arena` run.
    """
    global_dir, root = Path(global_dir), Path(root)
    if not acquire_locks(global_dir, [PRUNE_LOCK], machine=machine, ttl=PRUNE_LOCK_SECONDS):
        return []
    added: list[PoolMember] = []
    try:
        registry = load_global_registry(global_dir)
        anchors = [m for m in registry.models() if m.frozen]
        on_disk = {c.label for c in discover_population(root)}
        population = [m for m in registry.models() if not m.frozen and m.label in on_disk]
        rated_games = [m.games for m in population if m.games > 0]
        if not anchors or not rated_games:
            return []
        threshold = interpolated_percentile(rated_games, games_percentile)
        floor = max(m.rating for m in anchors)
        candidates = sorted(
            (m for m in population if m.games > threshold),
            key=lambda m: (m.rating, m.label),
        )
        copies = discover_all_copies(root)
        target_dir = root / BENCHMARK_DIRNAME
        for candidate in candidates:
            if candidate.rating - floor <= margin:
                continue
            if not acquire_locks(global_dir, [candidate.label], machine=machine, ttl=lock_ttl):
                continue  # being rated right now: the next run will look again
            try:
                member = read_member(global_dir, candidate.label)
                sources = sorted(copies.get(candidate.label, []))
                if member is None or member.frozen or not sources:
                    continue
                primary, *extra_copies = sources
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / primary.name
                try:
                    if not target.exists():
                        staging = target_dir / f".{primary.name}.partial"
                        shutil.copy2(primary, staging)
                        staging.replace(target)
                    primary.unlink()
                except OSError as exc:
                    if on_skip is not None:
                        on_skip(primary, str(exc))
                    continue
                for duplicate in extra_copies:
                    try:
                        duplicate.unlink()
                    except OSError:
                        pass  # best-effort: the primary copy already landed safely
                member.ref = str(target)
                member.frozen = True
                write_member(global_dir, member)
                added.append(member)
                floor = member.rating
            finally:
                release_locks(global_dir, [candidate.label])
    finally:
        release_locks(global_dir, [PRUNE_LOCK])
    if added:
        # Each new anchor carries a rating earned against the ordinary field and
        # never measured against the other anchors: ask for `benchmark_arena` to
        # settle them. A `poker-loop` supervisor picks the request up between two
        # generations (see `global_store.request_benchmark_arena`).
        request_benchmark_arena(global_dir, added=[m.label for m in added], machine=machine)
    return added


def delete_checkpoints(
    members: Sequence[PoolMember],
    paths_by_label: dict[str, list[Path]],
    *,
    on_skip: Callable[[str, str], None] | None = None,
) -> int:
    """Permanently delete every copy of each member's checkpoint from disk.

    This is the genuinely destructive half of elimination (see
    `PoolRegistry.eliminate_lowest_rated`'s docstring): a member passed here
    is already gone from the registry, and after this call its file is gone
    too, for good -- there is no "retired" holding area.

    `paths_by_label` should come from `discover_all_copies()`. A file already
    gone (a previous interrupted round) is skipped via `on_skip` rather than
    treated as an error.

    Returns the number of *files* removed, which can exceed `len(members)`.
    """
    deleted = 0
    for member in members:
        sources = paths_by_label.get(member.label, [])
        if not sources:
            if on_skip is not None:
                on_skip(member.label, "checkpoint already gone, nothing to delete")
            continue
        for source in sources:
            try:
                source.unlink()
                deleted += 1
            except OSError as exc:
                if on_skip is not None:
                    on_skip(member.label, str(exc))
    return deleted


@dataclass(frozen=True)
class PlayedRound:
    """What the playing half of a round produced, before anything is merged.

    Two different counts, and conflating them is easy: `candidates` is how many
    models were drawn and seated, `sessions` how many rated games they played
    between them. A caller that has to stop after so many *games* -- the Elo
    fill-in phase does -- needs the second, and there is no way to derive it
    from the first (a session seats `num_players` models and decrements each
    one's debt, so how many it takes depends on where the random seats land).
    """

    candidates: int = 0
    sessions: int = 0


@dataclass(frozen=True)
class PopulationRoundReport:
    """What one call to `run_population_round` did, for the caller to log.

    `played` is how many candidates this call drew and played and
    `sessions_played` how many games it played with them; everything else
    describes the merge that followed, which folds in *every* pending result
    from every machine, not just this call's -- so `sessions` is routinely far
    larger than `sessions_played`, and it is the smaller number a worker
    counting its own contribution has to read. `deferred_sessions` are sessions
    that could not get their member locks in time and went back to the queue.
    """

    played: int = 0
    sessions_played: int = 0
    pending_merged: int = 0
    sessions: int = 0
    deferred_sessions: int = 0
    participants: int = 0
    ghosts_dropped: int = 0
    triggered_elimination: bool = False
    eliminated: int = 0
    deleted_files: int = 0


def play_population_round_sharded(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    game,
    machine: str,
    population_sample: int = DEFAULT_POPULATION_SAMPLE,
    benchmark_sample: int = DEFAULT_BENCHMARK_SAMPLE,
    games_per_model: int = DEFAULT_GAMES_PER_MODEL,
    hands_per_game: int = DEFAULT_HANDS_PER_GAME,
    device: str = "cpu",
    seed: int | None = None,
    workers: int = 1,
    draw: Callable[[list[Candidate], int, random.Random], list[Candidate]] | None = None,
    on_skip: Callable[[Path, str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> PlayedRound:
    """Draw a random sample and play it -- no lock held at all.

    This is the expensive half of a round, and it parallelises perfectly
    (each session is independent), unlike applying ratings, which must be
    folded in model by model -- see `apply_pending_population_rounds`.
    Splitting them is what lets every machine, and every core on it, evaluate
    at the same time; only the cheap bookkeeping step takes any locks, and
    only on the few models each session involves.

    `workers <= 1` (the default, and what every existing test uses) plays
    in-process via `play_global_round` directly -- no subprocess, so tests
    stay fast and simple. `workers > 1` shards across that many local
    subprocesses (`python -m pokerlab.rl.global_arena`, one per shard): the
    games are split exactly between them (see `play_sharded`), each plays its
    slice for the same candidates and reports its sessions back; a shard that
    crashes or writes unreadable JSON is skipped via `on_skip`, never fatal to
    the round.

    Results are written to one pending-result file under
    `<global_dir>/pending/`, named uniquely (machine, timestamp, a short
    random suffix) so concurrent writers -- other machines, or another round
    from this one -- never collide. Returns how many candidates were drawn and
    played (0 if the draw could not even seat one table -- not an error,
    just nothing to report).

    `on_progress` is reported only on the in-process path (`workers <= 1`),
    which is the one production uses: `train.py` leaves `workers` at its
    default, so every worker of every generation plays its round here. A
    sharded round (`rl/benchmark_arena.py`, run by hand) spreads its playing
    across subprocesses that write to their own scratch logs, and stitching
    those back together is not worth it for a command someone is watching
    directly.

    `seed` defaults to `None` (OS entropy), not the training run's own seed:
    reusing that would make every round of a repeated identical sweep replay
    the identical cross-machine pairings -- the opposite of what a shared
    population round is for, unlike `benchmark.py`, where determinism is the
    entire point.
    """
    global_dir = Path(global_dir)
    root = Path(root)
    rng = random.Random(seed)

    population = discover_population(root)
    benchmark_population = discover_benchmark_population(root)
    # Uniform by default, and deliberately: a round decides who might be *deleted*,
    # so an honest draw is the right fit (see the module header). `draw` is the hook
    # for a caller that wants coverage instead -- `rl/population_arena.py` passes a
    # least-played-first draw, because a hand-run rating sweep deletes nothing and
    # its whole job is to reach models the sampled rounds have never seated.
    if draw is None:
        population_draw = rng.sample(population, min(population_sample, len(population)))
    else:
        population_draw = draw(population, min(population_sample, len(population)), rng)
    benchmark_draw = rng.sample(
        benchmark_population, min(benchmark_sample, len(benchmark_population))
    )
    combined = population_draw + benchmark_draw
    if len(combined) < game.num_players:
        return PlayedRound()

    if workers <= 1:
        sessions = play_global_round(
            combined,
            game,
            games_per_model=games_per_model,
            hands_per_game=hands_per_game,
            device=device,
            seed=rng.randrange(2**31),
            on_skip=on_skip,
            on_progress=on_progress,
        )
    else:
        sessions = play_sharded(
            combined,
            game,
            games_per_model=games_per_model,
            hands_per_game=hands_per_game,
            device=device,
            rng=rng,
            workers=workers,
            on_skip=on_skip,
        )
    if not sessions:
        return PlayedRound()

    write_pending_sessions(
        global_dir,
        sessions,
        machine=machine,
        population={c.label: str(c.path) for c in population_draw},
        benchmark={c.label: str(c.path) for c in benchmark_draw},
    )
    return PlayedRound(candidates=len(combined), sessions=len(sessions))


def write_pending_sessions(
    global_dir: str | Path,
    sessions: Sequence[dict[str, float]],
    *,
    machine: str,
    population: Mapping[str, str],
    benchmark: Mapping[str, str],
) -> Path | None:
    """Queue played sessions for a later merge, and return the file written.

    The only way results reach the ratings: playing takes no lock and leaves its
    work here, merging claims the file and applies it (see
    `apply_pending_population_rounds`). The name carries the machine, the time
    and a random suffix, so concurrent writers -- other machines, or another
    round from this one -- never collide, and the file is written as a dotted
    `.partial` and renamed into place so a merger never claims a half-written
    one.

    `population` and `benchmark` map each participant's label to where its
    checkpoint was when it played; a label listed under `benchmark` is
    bootstrapped into the registry as a frozen anchor.
    """
    if not sessions:
        return None
    pending_dir = Path(global_dir) / PENDING_DIRNAME
    pending_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "population_draw": [{"label": k, "path": v} for k, v in population.items()],
        "benchmark_draw": [{"label": k, "path": v} for k, v in benchmark.items()],
        "sessions": list(sessions),
    }
    name = f"{machine}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
    partial = pending_dir / f".{name}.partial"
    partial.write_text(json.dumps(payload), encoding="utf-8")
    target = pending_dir / name
    partial.replace(target)
    return target


def shard_games(games_per_model: int, workers: int) -> list[int]:
    """How many games each shard owes, summing to exactly `games_per_model`.

    Shard `i` takes the floor plus one while the remainder lasts. It used to be
    `ceil(games_per_model / workers)` for every shard, which silently overshot on
    an uneven division -- 20 games across 15 shards played 30, half again the
    hands that were asked for. A shard owed 0 is not launched.
    """
    if workers < 1:
        return [games_per_model] if games_per_model else []
    base, remainder = divmod(games_per_model, workers)
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def play_sharded(
    combined: list[Candidate],
    game,
    *,
    games_per_model: int,
    hands_per_game: int,
    device: str,
    rng: random.Random,
    workers: int,
    on_skip: Callable[[Path, str], None] | None,
) -> list[dict[str, float]]:
    """Fan `combined` out to `workers` local subprocesses via `_shard_main` and
    collect their sessions.

    Public because it has two callers: the population round below, and
    `rl/benchmark_arena.py`, which is otherwise sequential and needs the same
    fan-out (its rounds are 200,000 hands each, over an hour on one core).

    **The games are split exactly**, shard `i` taking
    `games_per_model // workers` plus one while the remainder lasts, so the
    total played per model is `games_per_model` and not more. It used to be
    `ceil(games_per_model / workers)` for every shard, which silently overshot
    whenever the division was uneven -- 20 games across 15 shards played 30.
    A shard owed nothing is not launched.

    Scratch files live under a local temp dir (never the shared NFS volume --
    shard output is pure IPC between parent and child on one machine), removed
    unconditionally when done.

    **Children are given `OMP_NUM_THREADS=1`, and that is what makes sharding
    worth anything.** Left unset, torch sizes its intra-op pool from the cores it
    can see, so every shard would try to use the whole machine and N shards would
    oversubscribe it N-fold. Measured on an idle 32-core box, 6-max hands per
    second: one process at 15 threads reaches **80.1** and one process at 1 thread
    **72.1** -- so fifteen cores' worth of threads buys 1.11x -- while **fifteen
    processes at one thread each total 1,102.9**, i.e. 15.3x for the same fifteen
    cores. Threads are a 14x worse use of the machine, because the work is a
    batch-of-one forward pass (~334 us) inside a pure-Python engine that accounts
    for two thirds of a hand's 12.4 ms."""
    work_dir = Path(tempfile.mkdtemp(prefix="global-arena-shard-"))
    try:
        candidates_path = work_dir / "candidates.json"
        candidates_path.write_text(
            json.dumps([{"label": c.label, "path": str(c.path)} for c in combined]),
            encoding="utf-8",
        )
        per_shard = shard_games(games_per_model, workers)
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"

        processes = []
        for shard in range(workers):
            games_per_shard = per_shard[shard]
            if games_per_shard == 0:
                continue
            out = work_dir / f"shard{shard:02d}.json"
            command = [
                sys.executable, "-u", "-m", "pokerlab.rl.global_arena",
                "--candidates", str(candidates_path),
                "--out", str(out),
                "--games-per-model", str(games_per_shard),
                "--hands-per-game", str(hands_per_game),
                "--players", str(game.num_players),
                "--stack", str(game.starting_stack),
                "--sb", str(game.small_blind),
                "--bb", str(game.big_blind),
                "--seed", str(rng.randrange(2**31)),
                "--device", device,
            ]
            log = (work_dir / f"shard{shard:02d}.log").open("w", encoding="utf-8")
            processes.append(
                (out, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment))
            )

        sessions: list[dict[str, float]] = []
        for out, process in processes:
            if process.wait() != 0:
                if on_skip is not None:
                    on_skip(out, "shard di valutazione globale fallito")
                continue
            try:
                sessions.extend(json.loads(out.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                if on_skip is not None:
                    on_skip(out, str(exc))
        return sessions
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _claim_pending(pending_dir: Path) -> list[Path]:
    """Take exclusive ownership of every waiting pending file.

    Claiming is a rename to a dotted `.claim-<id>-<name>` in the same directory,
    which is atomic: of several mergers racing for one file exactly one succeeds.
    That is what stops two of them folding the same sessions in twice now that
    there is no registry-wide lock. A claim whose owner died is handed back to
    the queue after `STALE_CLAIM_SECONDS`.
    """
    if not pending_dir.is_dir():
        return []
    now = time.time()
    for stale in pending_dir.glob(".claim-*.json"):
        try:
            if now - stale.stat().st_mtime >= STALE_CLAIM_SECONDS:
                stale.rename(pending_dir / stale.name.split("-", 2)[2])
        except OSError:
            continue
    claimed: list[Path] = []
    for path in sorted(p for p in pending_dir.glob("*.json") if not p.name.startswith(".")):
        target = pending_dir / f".claim-{uuid.uuid4().hex[:8]}-{path.name}"
        try:
            path.rename(target)
            os.utime(target, None)  # the claim's age counts from now, not from when it was played
        except OSError:
            continue
        claimed.append(target)
    return claimed


def _apply_session(
    global_dir: Path,
    results: dict[str, float],
    *,
    draw_info: dict[str, tuple[str, bool]],
    path_by_label: dict[str, Path],
    machine: str,
    lock_ttl: float,
    lock_wait: float,
) -> bool:
    """Fold one session into the ratings under the locks of its participants.

    Only the models actually at this table are locked, and only while their
    small files are read, rated and rewritten, so sessions on different models
    never wait for one another. Returns False if the locks could not be had in
    time; the caller then queues the session for a later merge.
    """
    labels = list(results)
    if not acquire_locks(global_dir, labels, machine=machine, ttl=lock_ttl, timeout=lock_wait):
        return False
    try:
        members: dict[str, PoolMember] = {}
        for label in labels:
            member = read_member(global_dir, label)
            path, benchmark = draw_info.get(label, ("", False))
            if member is None:
                # Only a model that still has a checkpoint is (re)admitted: one
                # eliminated while its result waited in the queue must not be
                # resurrected as a ghost.
                if label not in path_by_label:
                    continue
                member = PoolMember(label=label, kind=MODEL, ref=path, frozen=benchmark)
            elif benchmark and not member.frozen:
                # First time seen under benchmark/: pinned from here on.
                member.frozen = True
            if label in path_by_label:
                member.ref = str(path_by_label[label])
            members[label] = member
        known = {label: value for label, value in results.items() if label in members}
        if len(known) >= 2:
            PoolRegistry(
                directory=global_dir,
                max_models=10**9,
                members=members,
                k_schedule=DEFAULT_K_SCHEDULE,
            ).record_session(known)
            for member in members.values():
                write_member(global_dir, member)
        return True
    finally:
        release_locks(global_dir, labels)


def _eliminate(
    *,
    global_dir: Path,
    root: Path,
    machine: str,
    lock_ttl: float,
    trigger_size: int,
    eliminate_fraction: float,
    protect_percentile: float,
    on_skip: Callable[[Path, str], None] | None,
    on_phase: Callable[[str], None] | None = None,
) -> tuple[int, int, int, str]:
    """The pruning pass: `(eliminated, deleted_files)`.

    Runs only when the real on-disk population has reached `trigger_size`, and
    under an exclusive `__prune__` lock so two machines never prune (or number a
    new benchmark series) at once. That lock serialises pruners only -- ordinary
    merges carry on around it, since each doomed model is locked individually
    and re-read fresh before anything is touched.
    """
    none = (0, 0)
    if not acquire_locks(global_dir, [PRUNE_LOCK], machine=machine, ttl=PRUNE_LOCK_SECONDS):
        return none
    try:
        # Recounted under the lock: another machine may have pruned since this
        # merge started, and a stale count would prune the same backlog twice.
        population = discover_population(root)
        if len(population) < trigger_size:
            return none
        # Only now is it certain this pass will prune: announced after the lock
        # and the recount, not when the merge merely noticed the trigger, so a
        # worker that lost the race to another pruner never reports "pruning".
        if on_phase is not None:
            on_phase(PRUNING)
        population_labels = {c.label for c in population}
        registry = load_global_registry(global_dir)
        rated_games = [
            m.games
            for m in registry.models()
            if not m.frozen and m.games > 0 and m.label in population_labels
        ]
        if not rated_games:
            return none
        games_threshold = interpolated_percentile(rated_games, protect_percentile)
        eligible_count = sum(
            1
            for m in registry.models()
            if not m.frozen and m.games >= games_threshold and m.label in population_labels
        )
        doomed = registry.eliminate_lowest_rated(
            count=round(eligible_count * eliminate_fraction),
            games_threshold=games_threshold,
            among=population_labels,
        )

        held: list[str] = []
        confirmed: list[PoolMember] = []
        try:
            for member in doomed:
                if not acquire_locks(global_dir, [member.label], machine=machine, ttl=lock_ttl):
                    continue  # being rated right now: leave it for the next pass
                held.append(member.label)
                fresh = read_member(global_dir, member.label)
                if fresh is not None and not fresh.frozen:
                    confirmed.append(fresh)
            if not confirmed:
                return none

            all_copies = discover_all_copies(root)
            deleted = delete_checkpoints(confirmed, all_copies, on_skip=on_skip)
            for member in confirmed:
                remove_member(global_dir, member.label)
            return len(confirmed), deleted
        finally:
            release_locks(global_dir, held)
    finally:
        release_locks(global_dir, [PRUNE_LOCK])


def apply_pending_population_rounds(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    machine: str,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
    lock_wait: float = DEFAULT_LOCK_WAIT_SECONDS,
    trigger_size: int = DEFAULT_POPULATION_TRIGGER,
    eliminate_fraction: float = DEFAULT_ELIMINATION_FRACTION,
    protect_percentile: float = DEFAULT_PROTECT_PERCENTILE,
    on_skip: Callable[[Path, str], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> PopulationRoundReport:
    """Fold every pending round -- this machine's and everyone else's -- into
    the ratings, then prune if the population has reached the trigger.

    There is no registry-wide lock. Each pending file is *claimed* by an atomic
    rename (so two mergers never apply the same sessions), and each session is
    then applied under locks on just its own few participants (`_apply_session`),
    which is what lets many machines merge at once.

    Steps:
      1. `discover_population(root)` and
         `discover_benchmark_population(root)`: the on-disk truth used to
         refresh every participant's `ref`, to admit new members only if their
         checkpoint exists, and to sweep ghosts.
      2. Claim the pending files, oldest name first; fold each session in.
         A candidate not yet in the registry is bootstrapped at the default
         rating (`frozen=True` if drawn from `benchmark/`). A session that cannot
         get its locks in `lock_wait` seconds is written back as a new pending
         file. A file that fails to parse is set aside as `<name>.bad` for a
         human to look at, never retried automatically.
      3. Drop ghost members (`prune_ghost_members`).
      4. Pruning, when the real on-disk population is at or above
         `trigger_size` -- the trigger is that simple, and fires again whenever
         the backlog is back above it. Eligibility requires `games` at or above
         the `protect_percentile` of games played among the models rated so
         far; `count` is `eliminate_fraction` of that eligible set. Doomed
         members are the lowest-rated among the eligible, and every one of
         them has its checkpoint -- every copy of it, via
         `discover_all_copies` -- permanently deleted.
      5. Refresh the read-only `registry.json` snapshot if it has gone stale.

    Never raises for a business-as-usual failure (nothing pending, an unreadable
    pending file or checkpoint): those go through `on_skip` or the report.

    `on_phase`, when given, is told which stage this call has reached
    (`ELO_MERGE`, then `PRUNING` if a pruning pass really starts, then
    `ELO_MERGE` again for the snapshot) -- see `rl/phases.py` for why.
    """
    global_dir = Path(global_dir)
    root = Path(root)
    pending_dir = global_dir / PENDING_DIRNAME
    if on_phase is not None:
        on_phase(ELO_MERGE)
    migrate_legacy_registry(global_dir)

    population = discover_population(root)
    benchmark_population = discover_benchmark_population(root)
    path_by_label = {c.label: c.path for c in population + benchmark_population}

    claimed = _claim_pending(pending_dir)
    sessions_applied = deferred = merged_files = 0
    participants: set[str] = set()
    for claim in claimed:
        try:
            payload = json.loads(claim.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if on_skip is not None:
                on_skip(claim, str(exc))
            claim.rename(pending_dir / f"{claim.name.split('-', 2)[2]}.bad")
            continue
        draw_info = {
            entry["label"]: (entry["path"], benchmark)
            for benchmark, key in ((False, "population_draw"), (True, "benchmark_draw"))
            for entry in payload.get(key, [])
        }
        leftover: list[dict[str, float]] = []
        for results in payload.get("sessions", []):
            if _apply_session(
                global_dir,
                results,
                draw_info=draw_info,
                path_by_label=path_by_label,
                machine=machine,
                lock_ttl=lock_ttl,
                lock_wait=lock_wait,
            ):
                sessions_applied += 1
                participants.update(results)
            else:
                leftover.append(results)
        if leftover:
            payload["sessions"] = leftover
            name = f"{machine}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
            partial = pending_dir / f".{name}.partial"
            partial.write_text(json.dumps(payload), encoding="utf-8")
            partial.replace(pending_dir / name)
            deferred += len(leftover)
        else:
            merged_files += 1
        claim.unlink(missing_ok=True)

    # Re-read the disk rather than reusing `path_by_label`, which was listed
    # before a merge that may have run for half an hour. The per-label re-check
    # below (`root=`) closes what is left of the window.
    ghosts_dropped = prune_ghost_members(
        global_dir,
        {
            c.label
            for c in [*discover_population(root), *discover_benchmark_population(root)]
        },
        machine=machine,
        root=root,
        ttl=lock_ttl,
    )

    eliminated = deleted_files = 0
    if len(population) >= trigger_size:
        eliminated, deleted_files = _eliminate(
            global_dir=global_dir,
            root=root,
            machine=machine,
            lock_ttl=lock_ttl,
            trigger_size=trigger_size,
            eliminate_fraction=eliminate_fraction,
            protect_percentile=protect_percentile,
                on_skip=on_skip,
            on_phase=on_phase,
        )
        if on_phase is not None:
            on_phase(ELO_MERGE)  # the snapshot below is bookkeeping, not pruning

    write_snapshot(global_dir, machine=machine)

    return PopulationRoundReport(
        pending_merged=merged_files,
        sessions=sessions_applied,
        deferred_sessions=deferred,
        participants=len(participants),
        ghosts_dropped=ghosts_dropped,
        triggered_elimination=eliminated > 0,
        eliminated=eliminated,
        deleted_files=deleted_files,
    )


def run_population_round(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    game,
    machine: str,
    population_sample: int = DEFAULT_POPULATION_SAMPLE,
    benchmark_sample: int = DEFAULT_BENCHMARK_SAMPLE,
    games_per_model: int = DEFAULT_GAMES_PER_MODEL,
    hands_per_game: int = DEFAULT_HANDS_PER_GAME,
    device: str = "cpu",
    seed: int | None = None,
    workers: int = 1,
    draw: Callable[[list[Candidate], int, random.Random], list[Candidate]] | None = None,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
    trigger_size: int = DEFAULT_POPULATION_TRIGGER,
    eliminate_fraction: float = DEFAULT_ELIMINATION_FRACTION,
    protect_percentile: float = DEFAULT_PROTECT_PERCENTILE,
    on_skip: Callable[[Path, str], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> PopulationRoundReport:
    """One end-of-training-run cross-machine Elo round, and (rarely) a prune
    that deletes real files. Called from `train.py::main()` at the end of
    every `poker-train` run.

    Two independent phases, deliberately: `play_population_round_sharded`
    (play -- no lock, sharded across `workers` local processes) always runs,
    then `apply_pending_population_rounds` (bookkeeping -- per-model locks only)
    always follows. Whatever this call played is durably in
    `<global_dir>/pending/` before the merge starts, so no played game is ever
    lost, whichever machine ends up folding it in.

    `on_phase` receives the stage names of `rl/phases.py` as the round moves
    through them (play, merge, pruning), which is what lets a watcher say which
    of the three a worker is in; `on_progress` reports how far into the playing
    half it is, which is what lets the watcher say how much of it is left.
    """
    if on_phase is not None:
        on_phase(ELO_PLAY)
    played = play_population_round_sharded(
        global_dir=global_dir,
        root=root,
        game=game,
        machine=machine,
        population_sample=population_sample,
        benchmark_sample=benchmark_sample,
        games_per_model=games_per_model,
        hands_per_game=hands_per_game,
        device=device,
        seed=seed,
        workers=workers,
        draw=draw,
        on_skip=on_skip,
        on_progress=on_progress,
    )
    merged = apply_pending_population_rounds(
        global_dir=global_dir,
        root=root,
        machine=machine,
        lock_ttl=lock_ttl,
        trigger_size=trigger_size,
        eliminate_fraction=eliminate_fraction,
        protect_percentile=protect_percentile,
        on_skip=on_skip,
        on_phase=on_phase,
    )
    return replace(merged, played=played.candidates, sessions_played=played.sessions)


if __name__ == "__main__":
    # **This block is what makes sharding work at all**, and its absence was a
    # silent bug: `play_sharded` launches `python -m pokerlab.rl.global_arena`,
    # which without a `__main__` guard merely imported the module, did nothing,
    # and wrote no output file -- so every shard was reported through `on_skip`
    # as a missing file and a sharded round returned *zero* sessions. It stayed
    # dormant because nothing in production passes `workers > 1`
    # (`train.py`'s `run_population_round` call leaves it at the default 1), so
    # the only path that exercised it was one nobody had run. Pinned by a test
    # that invokes the module as a subprocess and checks it plays.
    _shard_main()
