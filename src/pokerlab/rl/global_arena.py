"""One Elo scale for every model on the shared volume.

Every model on the volume lives in one store, `checkpoints/models/`, and has one
rating on one scale. This module keeps that scale honest by letting models drawn at random
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
from functools import partial
from pathlib import Path

from pokerlab.rl.device import resolve_device
from pokerlab.rl.global_store import (
    DEFAULT_LOCK_SECONDS,
    PRUNE_LOCK,
    PRUNE_LOCK_SECONDS,
    acquire_locks,
    list_member_labels,
    load_global_registry,
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
    PoolMember,
    PoolRegistry,
    interpolated_percentile,
)
from pokerlab.rl.style_log import StyleTally, merge_style
from pokerlab.rl.table_mix import DEFAULT_SESSION_HANDS, table_arguments
from pokerlab.rl.training_pool import parent_tiers_text

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

# ~50 population + ~5 benchmark per pass is the composition asked for --
# see `run_population_sessions`. Both draws are plain uniform random: unlike
# `sample_population`'s least-played-first bias (still available, just not
# used here), a pass's whole point now includes deciding who gets deleted,
# so a uniform draw is the more honest "any tracked model could come up"
# sampling this decision deserves.
DEFAULT_POPULATION_SAMPLE = 50
DEFAULT_BENCHMARK_SAMPLE = 5
# How a model becomes a frozen anchor (see `add_benchmark_candidates`): at the
# end of a run, after the population pass, every model with more rated games
# than the `BENCHMARK_GAMES_PERCENTILE` of the population and a rating more than
# `BENCHMARK_MARGIN` points above the best anchor joins the frozen set. Pruning
# promotes nothing: every eliminated model is deleted.
BENCHMARK_GAMES_PERCENTILE = 90.0
BENCHMARK_MARGIN = 10.0

# Where a pass's *played* results wait to be folded into the registry --
# see play_population_sessions_sharded and apply_pending_population_sessions.
PENDING_DIRNAME = "pending"
DEFAULT_GLOBAL_WORKERS = 8
# How long one session waits for the handful of member locks it needs before
# being put back for a later merge. Locks are held for milliseconds, so this is
# a ceiling that is essentially never reached.
DEFAULT_LOCK_WAIT_SECONDS = 20.0
# A pending file claimed by a merger that then died is handed back to the queue
# after this long.
STALE_CLAIM_SECONDS = 3600


# Rated sessions in a pass. Every caller takes this default, so the number lives
# in exactly one place. A pass is that many sessions of `DEFAULT_SESSION_HANDS`
# hands, each at a table size drawn from `TableMix` and seating that many models
# drawn at random from the sampled set: nobody is owed a number of games, so a model
# plays as often as the draw lands on it (~8 sessions each at ~55 drawn and the
# default mixture's 4.35 seats on average, with the spread chance gives).
#
# A pass is the only thing that turns played hands into ratings, so this is a
# trade between rating evidence per run and the wall time a worker spends after
# it stops training (the pass dominates it). The Elo fill-in phase
# (`train.run_elo_fill_in`) has workers that finish early play extra rating
# passes while they wait for the slowest worker of their generation; those cost
# nothing, since the cores would be idle. If the ratings visibly stop converging,
# the dial to raise is `--fill-sessions` before this one.
DEFAULT_GLOBAL_SESSIONS = 100

@dataclass(frozen=True)
class Candidate:
    """One network in the population, and where to read it from."""

    label: str
    path: Path


def discover_population(root: str | Path = Path("checkpoints")) -> list[Candidate]:
    """Every model in the shared store, `checkpoints/models/*.pt`.

    A model's label is its file name without `.pt`, and there is exactly one
    file per model: the store is write-once and machine-prefixed.
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
    can take tens of minutes, while other machines keep publishing new models and
    registering them. Every one of those arrives in `list_member_labels` but not
    in the snapshot, and would be deleted here as a ghost: a live model, freshly
    rated, silently reset to the default rating with zero games. With pruning off
    and the population under the trigger, nothing deletes checkpoints at all, so
    any ghost reported then is a false positive.
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


def backfill_member_styles(
    global_dir: str | Path,
    root: str | Path = Path("checkpoints"),
    *,
    machine: str,
    read_metadata: Callable[[Path], dict | None],
    skip: set[str],
    limit: int = 50,
    ttl: float = DEFAULT_LOCK_SECONDS,
) -> int:
    """Give a member that has no style the one its checkpoint was published with.

    A model published before members carried a style still holds it in its own
    metadata (`style`, `style_hands`) when the run that trained it measured one; this
    copies it across. A member that has played a pass already has a style and is left
    alone. `read_metadata` opens a checkpoint (the caller owns torch), `skip` is every
    label already tried, added to as it goes so a model whose checkpoint has none is not
    opened again, and `limit` bounds the checkpoints opened in one call (a store is
    thousands of files on a network mount). Returns how many members were filled.
    """
    root = Path(root)
    path_by_label = {
        c.label: c.path
        for c in [*discover_population(root), *discover_benchmark_population(root)]
    }
    filled = tried = 0
    for label in sorted(list_member_labels(global_dir)):
        if tried >= limit:
            break
        member = read_member(global_dir, label)
        if member is None or member.style_hands > 0 or label in skip or label not in path_by_label:
            continue
        skip.add(label)
        tried += 1
        metadata = read_metadata(path_by_label[label]) or {}
        style, hands = metadata.get("style"), metadata.get("style_hands")
        if not isinstance(style, dict) or not isinstance(hands, int) or hands <= 0:
            continue
        counts = {
            name: [int(pair[0]), int(pair[1])]
            for name, pair in style.items()
            if isinstance(pair, list | tuple) and len(pair) == 2
        }
        if not counts or not acquire_locks(global_dir, [label], machine=machine, ttl=ttl):
            continue
        try:
            member = read_member(global_dir, label)
            if member is not None and member.style_hands == 0:
                member.style, member.style_hands = counts, hands
                write_member(global_dir, member)
                filled += 1
        finally:
            release_locks(global_dir, [label])
    return filled


def repair_member_refs(
    global_dir: str | Path,
    root: str | Path = Path("checkpoints"),
    *,
    machine: str,
    ttl: float = DEFAULT_LOCK_SECONDS,
) -> int:
    """Point every member's `ref` at where its checkpoint actually is now.

    Passes refresh the `ref` of the models they play and promotion rewrites it,
    but a model that has not played since its file moved (a migration of the
    store, say) keeps the old path until its next pass. This sweeps them all at
    once. Returns how many refs were corrected. A member with no checkpoint
    anywhere is left alone (`prune_ghost_members` handles it).
    """
    root = Path(root)
    path_by_label = {
        c.label: c.path
        for c in [*discover_population(root), *discover_benchmark_population(root)]
    }
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
    """Pick who plays this pass: the least-played first, then at random.

    Straight uniform sampling would leave a 23,000-model population with a
    handful of games each after many passes, and a rating from a handful of
    games is not a rating. Taking the least-played half deterministically and
    filling the rest at random gives even coverage without freezing the draw
    into the same faces every pass.
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


# The tiers `tiered_draw` splits the ranking by, written like `--parent-tiers`:
# the best N rated models, or `None` for the whole store. Every tier gets an equal
# share of the seats, and a tier listed twice gets twice the share. The default is
# the top 10, the rest of the top 100, the rest of the top 1,000 and everybody else.
DRAW_TIERS: tuple[int | None, ...] = (10, 100, 1000, None)


def draw_tiers_text(text: str) -> str:
    """argparse `type=` for `--draw-tiers`: the `--parent-tiers` syntax, validated."""
    return parent_tiers_text(text)


def _tier_bands(
    tiers: Sequence[int | None],
) -> list[tuple[int, int | None, int]]:
    """`(first, last, weight)` per disjoint band of the ranking, 0-based, `last`
    exclusive and `None` to the end.

    The tiers are nested cutoffs (the top 10, the top 100, ...), so each band runs
    from the previous cutoff to its own; a cutoff listed `n` times is one band of
    weight `n`, and `all` is always the last.
    """
    weights: dict[int | None, int] = {}
    for tier in tiers:
        weights[tier] = weights.get(tier, 0) + 1
    cutoffs = sorted(c for c in weights if c is not None)
    if None in weights:
        cutoffs.append(None)
    bands: list[tuple[int, int | None, int]] = []
    first = 0
    for cutoff in cutoffs:
        bands.append((first, cutoff, weights[cutoff]))
        if cutoff is not None:
            first = cutoff
    return bands


def _split_seats(count: int, weights: Sequence[int], rng: random.Random) -> list[int]:
    """`count` seats split in proportion to `weights`, to whole seats.

    Each band gets the whole part of its share, and the leftover seats go to bands
    by systematic sampling on the fractional parts, so a band's expected number of
    seats is exactly its share even when the shares do not divide evenly.
    """
    total = sum(weights)
    shares = [count * weight / total for weight in weights]
    quotas = [int(share) for share in shares]
    leftover = count - sum(quotas)
    if leftover:
        point = rng.random()
        cumulative = 0.0
        for index, share in enumerate(shares):
            before = cumulative
            cumulative += share - quotas[index]
            quotas[index] += int(cumulative + point) - int(before + point)
    return quotas


def tiered_draw(
    ratings: Mapping[str, dict],
    *,
    tiers: Sequence[int | None] = DRAW_TIERS,
) -> Callable[[list[Candidate], int, random.Random], list[Candidate]]:
    """A draw that gives each rating tier an equal share of the seats.

    **Why bias at all.** A pass seats ~50 models out of the whole store, so a
    given model comes up rarely and a rating can sit on the number its training
    run published for many generations. What is read is the top: `pick_parents`
    draws from the best 100. With a quarter of the seats on each of ranks 1-10,
    11-100, 101-1,000 and the rest, the best ten are seated in nearly every
    pass, the next ninety about every other pass, and the long tail keeps a
    quarter of the seats rather than none.

    The tiers are the `--draw-tiers` cutoffs. They are turned into disjoint bands
    (a model belongs to one), seats are split between them by weight with the
    leftover seats given at random so the average share is exact, and models
    inside a band are drawn uniformly without repetition. A band too small for its
    seats (the top ten against 12 or 13) hands the shortfall to the models not yet
    drawn, uniformly, so the pass always comes out at `count` while the disk holds
    that many models. A model with no rating yet is untested, not excluded: it
    belongs to the open-ended band (`all`) and, without one, is reached only
    through that spill.

    **Pruning and this draw do not go together.** Eligibility for deletion is a
    percentile of `games`, so seating some models far more often than others
    makes them eligible sooner and leaves the rarely-seated tail immune (see
    `DEFAULT_PROTECT_PERCENTILE`). Pass `trigger_size=NO_PRUNE_TRIGGER` unless
    that is wanted.
    """
    bands = _tier_bands(tiers)
    weights = [weight for _, _, weight in bands]

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
        for first, last, _ in bands:
            group = rated[first:last]
            if last is None:
                group = group + unrated
            groups.append(group)

        quotas = _split_seats(count, weights, rng)
        chosen: list[Candidate] = []
        for group, quota in zip(groups, quotas):
            chosen.extend(rng.sample(group, min(quota, len(group))))
        taken = {c.label for c in chosen}
        rest = [c for c in population if c.label not in taken]
        chosen.extend(rng.sample(rest, min(count - len(chosen), len(rest))))
        return chosen

    return draw


def play_global_sessions(
    candidates: list[Candidate],
    mix,
    *,
    sessions: int = DEFAULT_GLOBAL_SESSIONS,
    session_hands: int = DEFAULT_SESSION_HANDS,
    device: str = "cpu",
    seed: int = 0,
    on_skip=None,
    on_progress: Callable[[int, int, str], None] | None = None,
    styles: list[dict[str, dict]] | None = None,
) -> list[dict[str, float]]:
    """Play `sessions` sessions, each seating candidates drawn at random.

    Tables are drawn at random from the sampled candidates rather than by
    rating, so a model from one machine routinely faces models from the others
    -- which is the entire point: a rating only means something across machines
    if the matches cross them. Nobody is owed a number of games: how often a model
    plays is whatever the draw gives it.

    Returns the raw per-session chip deltas; the caller folds them into the
    ratings. Same split as the local arena, and for the same reason: playing
    parallelises, rating does not.

    `on_progress(done, total, detail)` fires after every session, counting
    sessions: the total is exactly `sessions`, known up front, so the bar ends at
    100% and never overshoots.

    `styles`, when given, gets one entry per session, aligned with the returned list:
    how each model played in it (`StyleTally.export`). It is a list to fill rather
    than a second return value so the many callers that want only the chip deltas
    are untouched, and it costs nothing when left out.
    """
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank, policy_opponent

    loaded = {}
    for candidate in candidates:
        try:
            model, _checkpoint = build_model_from_checkpoint(candidate.path, device=device)
        except Exception as exc:  # noqa: BLE001 - user-owned directories
            if on_skip is not None:
                on_skip(candidate.path, str(exc))
            continue
        loaded[candidate.label] = policy_opponent(
            candidate.label, make_policy_fn(model, device=device), mix
        )
    labels = sorted(loaded)
    # The largest table that can be drawn decides how many models must be
    # available: a pass that could only seat the small ones would not be the
    # mixture the weights describe.
    if len(labels) < mix.max_players:
        return []

    rng = random.Random(seed)
    bot_rng = random.Random(rng.random())
    size_rng = random.Random(rng.random())
    bank = TableBank(mix, rng)

    played: list[dict[str, float]] = []

    for _ in range(sessions):
        # Drawn per session and held for all of it (see `table_mix`).
        num_players = mix.draw_size(size_rng)
        proxies = bank.seats(num_players)
        seat_labels = rng.sample(labels, num_players)

        for proxy, label in zip(proxies, seat_labels):
            opponent = loaded[label]
            proxy.inner = opponent.factory(proxy.player_id, label, bot_rng)
            proxy.name = label

        tally = StyleTally() if styles is not None else None
        deltas = bank.play_session(
            num_players,
            session_hands,
            on_hand=None
            if tally is None
            else partial(tally.add_hand, labels=dict(enumerate(seat_labels))),
        )
        if tally is not None:
            styles.append(tally.export())

        results: dict[str, float] = {}
        for label, delta in zip(seat_labels, deltas):
            results[label] = results.get(label, 0.0) + delta
        played.append(results)
        if on_progress is not None:
            on_progress(len(played), sessions, f"{len(played)} sessioni")

    return played


def _shard_main() -> None:
    """Play one shard of a population pass and write its sessions to JSON.

    The supervisor (`play_population_sessions_sharded`) runs several of these at
    once, on one machine's cores: only *playing* parallelises, so each shard
    just plays its slice of `--sessions` for the same candidate set and
    reports back: no registry, no lock, no rating math here at all.
    """
    import argparse

    from pokerlab.rl.table_mix import add_table_arguments, table_mix_from_args

    parser = argparse.ArgumentParser(description="Play one shard of a global population pass.")
    parser.add_argument("--candidates", type=Path, required=True, help="JSON [{label, path}, ...]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=DEFAULT_GLOBAL_SESSIONS)
    add_table_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="auto", help="cpu, cuda or auto (the default): the GPU if there is one"
    )
    args = parser.parse_args()

    mix = table_mix_from_args(args)
    raw = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = [
        Candidate(label=entry["label"], path=Path(entry["path"]))
        for entry in raw
    ]

    styles: list[dict[str, dict]] = []
    sessions = play_global_sessions(
        candidates,
        mix,
        sessions=args.sessions,
        session_hands=args.session_hands,
        device=resolve_device(args.device),
        seed=args.seed,
        styles=styles,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"sessions": sessions, "styles": styles}), encoding="utf-8")


def spread_pick(
    candidates: Sequence[PoolMember], fixed: Sequence[float], count: int
) -> list[PoolMember]:
    """Up to `count` of `candidates` whose ratings are as far apart as possible.

    Farthest-point selection: with nothing fixed yet it starts from the strongest and
    the weakest, then repeatedly takes the candidate whose rating is furthest from every
    one already chosen (`fixed` are ratings that are taken anyway, such as the existing
    anchors). More candidates than needed are therefore thinned to an even ladder, and
    ties go to the stronger model, then to the label, so the choice is reproducible.
    The result is in the order they were picked, which a caller that can only seat some
    of them reads as the order of preference.
    """
    pool = sorted(candidates, key=lambda m: (-m.rating, m.label))
    chosen: list[PoolMember] = []
    taken = list(fixed)
    while pool and len(chosen) < count:
        if not taken:
            pick = pool[0]  # the strongest first, then the weakest by distance
        else:
            pick = max(pool, key=lambda m: min(abs(m.rating - r) for r in taken))
        pool.remove(pick)
        chosen.append(pick)
        taken.append(pick.rating)
    return chosen


def add_benchmark_candidates(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    machine: str,
    games_percentile: float = BENCHMARK_GAMES_PERCENTILE,
    margin: float = BENCHMARK_MARGIN,
    minimum_anchors: int = 0,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
    on_skip: Callable[[Path, str], None] | None = None,
) -> list[PoolMember]:
    """Freeze every model that is both well played and stronger than the whole
    benchmark: `games` above the `games_percentile` of the population's games
    and a rating more than `margin` above the best anchor.

    Called at the end of a training run, after the population pass, so the
    ratings it reads already include that pass. Candidates are taken from the
    lowest rating upwards and each must clear the previously added one by more
    than `margin` too, so the anchors stay at least `margin` apart rather than a
    whole cluster of near-identical models joining at once. Returns the models
    added (empty when nothing qualifies or the percentile is undefined).

    **With fewer than `minimum_anchors` anchors it bootstraps instead.** The benchmark
    cannot start itself otherwise: no model can be stronger than a best anchor that
    does not exist, and the pass against the anchors cannot seat a table until it has
    one fewer than the largest table. So it fills the missing places with the models
    whose ratings are **as far apart as possible** (`spread_pick`), from the strongest
    to the weakest that have played, with no percentile and no margin: an anchor is a
    reference point, and a ladder with both ends and an even spread in between measures
    a new model better than eight near-identical strong ones. A rating from few
    sessions is a weak reason to freeze a model, but an empty benchmark is worse, and
    `benchmark_arena` settles the new anchors against one another. Models that have not
    played at all are used only if the ones that have run out. Once the count is met,
    the ordinary rule above applies.

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
    copies = discover_all_copies(root)
    target_dir = root / BENCHMARK_DIRNAME

    def promote(candidate: PoolMember) -> bool:
        """Move one model into the benchmark and freeze it. False if it was skipped."""
        if not acquire_locks(global_dir, [candidate.label], machine=machine, ttl=lock_ttl):
            return False  # being rated right now: the next run will look again
        try:
            member = read_member(global_dir, candidate.label)
            sources = sorted(copies.get(candidate.label, []))
            if member is None or member.frozen or not sources:
                return False
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
                return False
            for duplicate in extra_copies:
                try:
                    duplicate.unlink()
                except OSError:
                    pass  # best-effort: the primary copy already landed safely
            member.ref = str(target)
            member.frozen = True
            write_member(global_dir, member)
            added.append(member)
            return True
        finally:
            release_locks(global_dir, [candidate.label])

    try:
        registry = load_global_registry(global_dir)
        anchors = [m for m in registry.ranked() if m.frozen]
        on_disk = {c.label for c in discover_population(root)}
        population = [m for m in registry.ranked() if not m.frozen and m.label in on_disk]
        rated = [m for m in population if m.games > 0]
        if len(anchors) < minimum_anchors:
            missing = minimum_anchors - len(anchors)
            unplayed = [m for m in population if m.games == 0]
            for pool in (rated, unplayed):
                # Picked again whenever one could not be moved (it was being rated).
                for candidate in spread_pick(
                    pool, [m.rating for m in anchors + added], missing - len(added)
                ):
                    promote(candidate)
        elif anchors and rated:
            threshold = interpolated_percentile([m.games for m in rated], games_percentile)
            floor = max(m.rating for m in anchors)
            candidates = sorted(
                (m for m in population if m.games > threshold),
                key=lambda m: (m.rating, m.label),
            )
            for candidate in candidates:
                if candidate.rating - floor <= margin:
                    continue
                if promote(candidate):
                    floor = candidate.rating
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
    gone (a previous interrupted pass) is skipped via `on_skip` rather than
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
class PlayedSessions:
    """What the playing half of a pass produced, before anything is merged.

    Two different counts, and conflating them is easy: `candidates` is how many
    models were drawn and seated, `sessions` how many rated games they played
    between them. A caller that has to stop after so many *games* -- the Elo
    fill-in phase does -- needs the second, which is not derivable from the first
    (a session seats `num_players` of the `candidates` drawn).
    """

    candidates: int = 0
    sessions: int = 0


@dataclass(frozen=True)
class PopulationSessionsReport:
    """What one call to `run_population_sessions` did, for the caller to log.

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


def play_population_sessions_sharded(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    mix,
    machine: str,
    population_sample: int = DEFAULT_POPULATION_SAMPLE,
    benchmark_sample: int = DEFAULT_BENCHMARK_SAMPLE,
    sessions: int = DEFAULT_GLOBAL_SESSIONS,
    session_hands: int = DEFAULT_SESSION_HANDS,
    device: str = "cpu",
    seed: int | None = None,
    workers: int = 1,
    draw: Callable[[list[Candidate], int, random.Random], list[Candidate]] | None = None,
    on_skip: Callable[[Path, str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> PlayedSessions:
    """Draw a random sample and play it -- no lock held at all.

    This is the expensive half of a pass, and it parallelises perfectly
    (each session is independent), unlike applying ratings, which must be
    folded in model by model -- see `apply_pending_population_sessions`.
    Splitting them is what lets every machine, and every core on it, evaluate
    at the same time; only the cheap bookkeeping step takes any locks, and
    only on the few models each session involves.

    `workers <= 1` (the default, and what every existing test uses) plays
    in-process via `play_global_sessions` directly -- no subprocess, so tests
    stay fast and simple. `workers > 1` shards across that many local
    subprocesses (`python -m pokerlab.rl.global_arena`, one per shard): the
    games are split exactly between them (see `play_sharded`), each plays its
    slice for the same candidates and reports its sessions back; a shard that
    crashes or writes unreadable JSON is skipped via `on_skip`, never fatal to
    the pass.

    Results are written to one pending-result file under
    `<global_dir>/pending/`, named uniquely (machine, timestamp, a short
    random suffix) so concurrent writers -- other machines, or another pass
    from this one -- never collide. Returns how many candidates were drawn and
    played (0 if the draw could not even seat one table -- not an error,
    just nothing to report).

    `on_progress` is reported only on the in-process path (`workers <= 1`),
    which is the one production uses: `train.py` leaves `workers` at its
    default, so every worker of every generation plays its pass here. A
    sharded pass (`rl/benchmark_arena.py`, run by hand) spreads its playing
    across subprocesses that write to their own scratch logs, and stitching
    those back together is not worth it for a command someone is watching
    directly.

    `seed` defaults to `None` (OS entropy), not the training run's own seed:
    reusing that would make every pass of a repeated identical sweep replay
    the identical cross-machine pairings -- the opposite of what a shared
    population pass is for, unlike `benchmark.py`, where determinism is the
    entire point.
    """
    global_dir = Path(global_dir)
    root = Path(root)
    rng = random.Random(seed)

    population = discover_population(root)
    benchmark_population = discover_benchmark_population(root)
    # Uniform by default, and deliberately: a pass decides who might be *deleted*,
    # so an honest draw is the right fit (see the module header). `draw` is the hook
    # for a caller that wants coverage instead -- `rl/population_arena.py` passes a
    # least-played-first draw, because a hand-run rating sweep deletes nothing and
    # its whole job is to reach models the sampled passes have never seated.
    if draw is None:
        population_draw = rng.sample(population, min(population_sample, len(population)))
    else:
        population_draw = draw(population, min(population_sample, len(population)), rng)
    benchmark_draw = rng.sample(
        benchmark_population, min(benchmark_sample, len(benchmark_population))
    )
    combined = population_draw + benchmark_draw
    if len(combined) < mix.max_players:
        return PlayedSessions()

    styles: list[dict[str, dict]] = []
    if workers <= 1:
        played = play_global_sessions(
            combined,
            mix,
            sessions=sessions,
            session_hands=session_hands,
            device=device,
            seed=rng.randrange(2**31),
            on_skip=on_skip,
            on_progress=on_progress,
            styles=styles,
        )
    else:
        played = play_sharded(
            combined,
            mix,
            sessions=sessions,
            session_hands=session_hands,
            device=device,
            rng=rng,
            workers=workers,
            on_skip=on_skip,
            styles=styles,
        )
    if not played:
        return PlayedSessions()

    write_pending_sessions(
        global_dir,
        played,
        styles=styles,
        machine=machine,
        population={c.label: str(c.path) for c in population_draw},
        benchmark={c.label: str(c.path) for c in benchmark_draw},
    )
    return PlayedSessions(candidates=len(combined), sessions=len(played))


def write_pending_sessions(
    global_dir: str | Path,
    sessions: Sequence[dict[str, float]],
    *,
    machine: str,
    population: Mapping[str, str],
    benchmark: Mapping[str, str],
    styles: Sequence[dict[str, dict]] | None = None,
) -> Path | None:
    """Queue played sessions for a later merge, and return the file written.

    The only way results reach the ratings: playing takes no lock and leaves its
    work here, merging claims the file and applies it (see
    `apply_pending_population_sessions`). The name carries the machine, the time
    and a random suffix, so concurrent writers -- other machines, or another
    pass from this one -- never collide, and the file is written as a dotted
    `.partial` and renamed into place so a merger never claims a half-written
    one.

    `population` and `benchmark` map each participant's label to where its
    checkpoint was when it played; a label listed under `benchmark` is
    bootstrapped into the registry as a frozen anchor.

    `styles` is how each participant played, one entry per session in the same order
    (`StyleTally.export`); the merge folds it into the members' `style`.
    """
    if not sessions:
        return None
    pending_dir = Path(global_dir) / PENDING_DIRNAME
    pending_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "population_draw": [{"label": k, "path": v} for k, v in population.items()],
        "benchmark_draw": [{"label": k, "path": v} for k, v in benchmark.items()],
        "sessions": list(sessions),
        "styles": list(styles) if styles else [],
    }
    name = f"{machine}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
    partial = pending_dir / f".{name}.partial"
    partial.write_text(json.dumps(payload), encoding="utf-8")
    target = pending_dir / name
    partial.replace(target)
    return target


def shard_sessions(sessions: int, workers: int) -> list[int]:
    """How many sessions each shard plays, summing to exactly `sessions`.

    Shard `i` takes the floor plus one while the remainder lasts. (A flat
    `ceil(sessions / workers)` per shard would silently overshoot on an uneven
    division: 20 sessions across 15 shards would play 30.) A shard given 0 is
    not launched.
    """
    if workers < 1:
        return [sessions] if sessions else []
    base, remainder = divmod(sessions, workers)
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def play_sharded(
    combined: list[Candidate],
    mix,
    *,
    sessions: int,
    session_hands: int,
    device: str,
    rng: random.Random,
    workers: int,
    on_skip: Callable[[Path, str], None] | None,
    styles: list[dict[str, dict]] | None = None,
) -> list[dict[str, float]]:
    """Fan `combined` out to `workers` local subprocesses via `_shard_main` and
    collect their sessions.

    Public because it has two callers: the population pass below, and
    `rl/benchmark_arena.py`, which is otherwise sequential and needs the same
    fan-out (its passes are 200,000 hands each, over an hour on one core).

    **The sessions are split exactly**, shard `i` taking `sessions // workers`
    plus one while the remainder lasts, so the total played is `sessions` and not
    more. A shard given nothing is not launched.

    Scratch files live under a local temp dir (never the shared NFS volume --
    shard output is pure IPC between parent and child on one machine), removed
    unconditionally when done.

    **Children are given `OMP_NUM_THREADS=1`, and that is what makes sharding
    worth anything.** Left unset, torch sizes its intra-op pool from the cores it
    can see, so every shard would try to use the whole machine and N shards would
    oversubscribe it N-fold. The work is a batch-of-one forward pass inside a
    pure-Python engine, so threads inside one process buy almost nothing, while
    one process per core scales with the core count."""
    work_dir = Path(tempfile.mkdtemp(prefix="global-arena-shard-"))
    try:
        candidates_path = work_dir / "candidates.json"
        candidates_path.write_text(
            json.dumps([{"label": c.label, "path": str(c.path)} for c in combined]),
            encoding="utf-8",
        )
        per_shard = shard_sessions(sessions, workers)
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"

        processes = []
        for shard in range(workers):
            sessions_per_shard = per_shard[shard]
            if sessions_per_shard == 0:
                continue
            out = work_dir / f"shard{shard:02d}.json"
            command = [
                sys.executable, "-u", "-m", "pokerlab.rl.global_arena",
                "--candidates", str(candidates_path),
                "--out", str(out),
                "--sessions", str(sessions_per_shard),
                "--session-hands", str(session_hands),
                *table_arguments(mix),
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
                shard = json.loads(out.read_text(encoding="utf-8"))
                sessions.extend(shard["sessions"])
                if styles is not None:
                    styles.extend(shard["styles"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
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
    k_schedule: Sequence[tuple[int, float]] = DEFAULT_K_SCHEDULE,
    style: Mapping[str, dict] | None = None,
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
                member = PoolMember(label=label, ref=path, frozen=benchmark)
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
                k_schedule=tuple(k_schedule),
            ).record_session(known)
            for label, member in members.items():
                seen = (style or {}).get(label)
                if seen:
                    member.style, member.style_hands = merge_style(
                        member.style, member.style_hands, seen["style"], seen["hands"]
                    )
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
            for m in registry.ranked()
            if not m.frozen and m.games > 0 and m.label in population_labels
        ]
        if not rated_games:
            return none
        games_threshold = interpolated_percentile(rated_games, protect_percentile)
        eligible_count = sum(
            1
            for m in registry.ranked()
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


def apply_pending_population_sessions(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    machine: str,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
    lock_wait: float = DEFAULT_LOCK_WAIT_SECONDS,
    trigger_size: int = DEFAULT_POPULATION_TRIGGER,
    eliminate_fraction: float = DEFAULT_ELIMINATION_FRACTION,
    protect_percentile: float = DEFAULT_PROTECT_PERCENTILE,
    k_schedule: Sequence[tuple[int, float]] = DEFAULT_K_SCHEDULE,
    on_skip: Callable[[Path, str], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> PopulationSessionsReport:
    """Fold every pending pass -- this machine's and everyone else's -- into
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
        leftover_styles: list[dict[str, dict]] = []
        styles = payload.get("styles", [])
        for index, results in enumerate(payload.get("sessions", [])):
            style = styles[index] if index < len(styles) else None
            if _apply_session(
                global_dir,
                results,
                draw_info=draw_info,
                path_by_label=path_by_label,
                machine=machine,
                lock_ttl=lock_ttl,
                lock_wait=lock_wait,
                k_schedule=k_schedule,
                style=style,
            ):
                sessions_applied += 1
                participants.update(results)
            else:
                leftover.append(results)
                leftover_styles.append(style or {})
        if leftover:
            payload["sessions"] = leftover
            payload["styles"] = leftover_styles
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

    return PopulationSessionsReport(
        pending_merged=merged_files,
        sessions=sessions_applied,
        deferred_sessions=deferred,
        participants=len(participants),
        ghosts_dropped=ghosts_dropped,
        triggered_elimination=eliminated > 0,
        eliminated=eliminated,
        deleted_files=deleted_files,
    )


def run_population_sessions(
    *,
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    root: str | Path = Path("checkpoints"),
    mix,
    machine: str,
    population_sample: int = DEFAULT_POPULATION_SAMPLE,
    benchmark_sample: int = DEFAULT_BENCHMARK_SAMPLE,
    sessions: int = DEFAULT_GLOBAL_SESSIONS,
    session_hands: int = DEFAULT_SESSION_HANDS,
    device: str = "cpu",
    seed: int | None = None,
    workers: int = 1,
    draw: Callable[[list[Candidate], int, random.Random], list[Candidate]] | None = None,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
    trigger_size: int = DEFAULT_POPULATION_TRIGGER,
    eliminate_fraction: float = DEFAULT_ELIMINATION_FRACTION,
    protect_percentile: float = DEFAULT_PROTECT_PERCENTILE,
    k_schedule: Sequence[tuple[int, float]] = DEFAULT_K_SCHEDULE,
    on_skip: Callable[[Path, str], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> PopulationSessionsReport:
    """One end-of-training-run cross-machine Elo pass, and (rarely) a prune
    that deletes real files. Called from `train.py::main()` at the end of
    every `poker-train` run.

    Two independent phases, deliberately: `play_population_sessions_sharded`
    (play -- no lock, sharded across `workers` local processes) always runs,
    then `apply_pending_population_sessions` (bookkeeping -- per-model locks only)
    always follows. Whatever this call played is durably in
    `<global_dir>/pending/` before the merge starts, so no played game is ever
    lost, whichever machine ends up folding it in.

    `on_phase` receives the stage names of `rl/phases.py` as the pass moves
    through them (play, merge, pruning), which is what lets a watcher say which
    of the three a worker is in; `on_progress` reports how far into the playing
    half it is, which is what lets the watcher say how much of it is left.
    """
    if on_phase is not None:
        on_phase(ELO_PLAY)
    played = play_population_sessions_sharded(
        global_dir=global_dir,
        root=root,
        mix=mix,
        machine=machine,
        population_sample=population_sample,
        benchmark_sample=benchmark_sample,
        sessions=sessions,
        session_hands=session_hands,
        device=device,
        seed=seed,
        workers=workers,
        draw=draw,
        on_skip=on_skip,
        on_progress=on_progress,
    )
    merged = apply_pending_population_sessions(
        global_dir=global_dir,
        root=root,
        machine=machine,
        lock_ttl=lock_ttl,
        trigger_size=trigger_size,
        eliminate_fraction=eliminate_fraction,
        protect_percentile=protect_percentile,
        k_schedule=k_schedule,
        on_skip=on_skip,
        on_phase=on_phase,
    )
    return replace(merged, played=played.candidates, sessions_played=played.sessions)


if __name__ == "__main__":
    # **This block is what makes sharding work at all**, and its absence was a
    # silent bug: `play_sharded` launches `python -m pokerlab.rl.global_arena`,
    # which without a `__main__` guard merely imported the module, did nothing,
    # and wrote no output file -- so every shard was reported through `on_skip`
    # as a missing file and a sharded pass returned *zero* sessions. It stayed
    # dormant because nothing in production passes `workers > 1`
    # (`train.py`'s `run_population_sessions` call leaves it at the default 1), so
    # the only path that exercised it was one nobody had run. Pinned by a test
    # that invokes the module as a subprocess and checks it plays.
    _shard_main()
