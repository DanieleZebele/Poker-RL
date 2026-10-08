"""Rate the frozen benchmark anchors against each other. Run by hand, rarely.

The anchors in `checkpoints/benchmark/` are the fixed reference points the whole
global scale is measured against: `PoolMember.frozen` pins their ratings, and
`record_session_with_ratings` never applies a frozen member's delta, so no
training run, no population pass and no merge can move them. That is the point
-- a scale whose reference points drift measures nothing.

**This module is the single deliberate exception, and the only code in the
project that writes an anchor's rating.** It does not go through
`record_session_with_ratings` (which would refuse) but computes the deltas with
`pairwise_elo_delta` and writes the member files itself. Nothing imports it;
it exists to be run as a command, by a person who means it:

    python -m pokerlab.rl.benchmark_arena --workers 10

Why it is ever needed: an anchor is pinned at whatever rating it held the
instant it was promoted, which is one number from the ordinary population
passes, often earned against a field that no longer exists. The anchors are
never rated against *each other*, so nothing else checks that they sit correctly
relative to one another. This plays them among themselves and settles that.

Two properties are deliberate:

- **K always follows the hyperbolic staircase** (`--k-schedule`, from `config.toml`
  or `DEFAULT_K_SCHEDULE`, read from each anchor's own `games`, as everywhere else
  in the rating path); there is no flat-K option. The cost, accepted: with per-anchor Ks a
  session is no longer zero-sum in rating, so the anchors' *mean* can move,
  where a flat K kept it exactly fixed. Veteran anchors (tens of thousands of
  games) barely move at the bottom tier; a freshly added one moves most.
- **Only the ratings change.** Membership, `frozen`, `ref` and the checkpoint
  files are left exactly as they are, so afterwards the anchors are still frozen
  and still unreachable to everything else.

**How long it runs is measured in sessions, never in passes.** It plays until the
ratings settle -- no anchor has moved more than `--arena-tolerance` points over
the last `--arena-window` sessions -- but never stops before
`--arena-min-sessions` and never plays past `--arena-max-sessions`. It plays in
batches of `--arena-checkpoint-sessions` only so that what it has learned is on disk and
on screen at regular intervals, and the convergence test runs at those points: a
batch is a checkpoint, not a unit anything is counted in.

Every rating is written to a timestamped JSON backup before anything is touched,
so a run can be undone with `--restore`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pokerlab.config import add_config_arguments, resolve_cli
from pokerlab.rl.device import resolve_device
from pokerlab.rl.global_arena import (
    DEFAULT_GLOBAL_DIR,
    Candidate,
    discover_benchmark_population,
    play_global_sessions,
    play_sharded,
)
from pokerlab.rl.global_store import (
    DEFAULT_LOCK_SECONDS,
    acquire_locks,
    read_member,
    release_locks,
    write_member,
    write_snapshot,
)
from pokerlab.rl.pool_registry import (
    DEFAULT_K_SCHEDULE,
    DEFAULT_RATING,
    PoolMember,
    format_k_schedule,
    k_for_games,
    k_schedule_text,
    pairwise_elo_delta,
    parse_k_schedule,
)
from pokerlab.rl.siblings import sibling_parsers
from pokerlab.rl.table_mix import add_table_arguments, table_mix_from_args

# Convergence, and why it is measured as *net drift over a window* rather than
# per-session movement: an Elo rating never stops moving, it jitters around its
# equilibrium forever and the jitter is proportional to K. A rule like "stop when
# every delta is under a point" would never fire. Net drift filters the jitter out
# -- a model bouncing +3/-3 has barely moved over a window, while one still
# climbing has. All four numbers are in sessions, the unit a rating is earned in.
DEFAULT_TOLERANCE = 1.0          # rating points of net drift, per model, over the window
DEFAULT_WINDOW = 4_000           # sessions
DEFAULT_MIN_SESSIONS = 10_000    # never declare convergence before this many
DEFAULT_MAX_SESSIONS = 100_000   # safety cap, not a failure
# How many sessions are played between two writes of the ratings and two reports.
# It decides nothing about the ratings a run ends with, only how much of a killed
# run is lost (at most this many sessions), how often a person watching sees the
# anchors move, and how often the convergence test gets to run. The write is 60
# small files, milliseconds against minutes of play, so a small value is cheap --
# except with `--workers`, where every batch spawns its shards afresh and each
# loads every anchor, so a very small one pays that startup over and over.
DEFAULT_CHECKPOINT_SESSIONS = 200
BACKUP_DIRNAME = "benchmark_arena_backups"


@dataclass
class Anchor:
    """One benchmark model, where it lives and how its rating has moved."""

    label: str
    path: Path
    series: str
    start: float
    rating: float
    games: int = 0

    @property
    def delta(self) -> float:
        return self.rating - self.start


def collect_anchors(root: Path, global_dir: Path) -> list[Anchor]:
    """Every benchmark model, with the rating the global registry holds for it.

    A model with no member file yet has never played a rated game; it starts at
    the default rating, exactly as a population pass would bootstrap it.
    """
    anchors: list[Anchor] = []
    for candidate in discover_benchmark_population(root):
        member = read_member(global_dir, candidate.label)
        rating = member.rating if member is not None else DEFAULT_RATING
        games = member.games if member is not None else 0
        # The series is the directory the file sits in; loose files in the
        # benchmark root are their own group.
        series = candidate.path.parent.name
        anchors.append(
            Anchor(
                label=candidate.label,
                path=candidate.path,
                series=series,
                start=rating,
                rating=rating,
                games=games,
            )
        )
    anchors.sort(key=lambda a: (a.series, a.label))
    return anchors


def apply_sessions(
    anchors: dict[str, Anchor],
    sessions: list[dict[str, float]],
    k_schedule: Sequence[tuple[int, float]] = DEFAULT_K_SCHEDULE,
) -> int:
    """Fold played sessions into the in-memory ratings. Returns sessions applied.

    Ratings are read from `anchors` as they stand *before* each session and the
    deltas applied after, so the outcome does not depend on the order the
    participants are iterated in -- the same rule `pairwise_elo_delta` follows
    inside one session, applied between them.

    Each anchor's K comes from the staircase at the games it had when the session
    began, and the session then counts as one more game for it.
    """
    applied = 0
    for session in sessions:
        known = {label: value for label, value in session.items() if label in anchors}
        if len(known) < 2:
            continue
        ratings = {label: anchors[label].rating for label in known}
        k_factors = {label: k_for_games(anchors[label].games, k_schedule) for label in known}
        for label, delta in pairwise_elo_delta(known, ratings, k_factors=k_factors).items():
            anchors[label].rating += delta
            anchors[label].games += 1
        applied += 1
    return applied


def format_table(anchors: list[Anchor], *, width: int = 2) -> list[str]:
    """Every anchor, grouped by the directory it lives in, with its rating.

    All of them, every report: the whole point of the run is to watch the
    anchors sort themselves out, and a summary would hide which ones moved.
    """
    by_series: dict[str, list[Anchor]] = defaultdict(list)
    for anchor in anchors:
        by_series[anchor.series].append(anchor)

    lines: list[str] = []
    for series in sorted(by_series):
        group = by_series[series]
        mean = sum(a.rating for a in group) / len(group)
        lines.append(f"  {series}  ({len(group)} modelli, media {mean:.0f})")
        cells = [
            f"{a.label[:34]:<34}{a.rating:>7.0f}{a.delta:>+7.1f}"
            for a in sorted(group, key=lambda a: -a.rating)
        ]
        for index in range(0, len(cells), width):
            lines.append("    " + "   ".join(cells[index : index + width]))
    return lines


def format_series_line(anchors: list[Anchor]) -> str:
    """One line with each series' rating span, for printing at every checkpoint.

    `format_table` is 30 lines for 60 anchors; this is the same information at the
    resolution a quick look needs -- where each group sits and how far it has
    moved since the run started.
    """
    by_series: dict[str, list[Anchor]] = defaultdict(list)
    for anchor in anchors:
        by_series[anchor.series].append(anchor)
    parts = []
    for series in sorted(by_series):
        group = by_series[series]
        mean = sum(a.rating for a in group) / len(group)
        shift = sum(a.delta for a in group) / len(group)
        short = series.replace("benchmark_", "b")
        parts.append(
            f"{short} {mean:.0f} ({min(a.rating for a in group):.0f}-"
            f"{max(a.rating for a in group):.0f}, {shift:+.1f})"
        )
    return "         " + "  ".join(parts)


def drift(history: list[tuple[int, dict[str, float]]], window: int) -> tuple[float, float]:
    """`(max, mean)` net rating movement per model across the last `window` sessions.

    `history` is `(sessions played, ratings)` at every checkpoint, the first entry
    being the state before anything was played. Net, not cumulative: the difference
    between where each model sits now and where it sat at the latest checkpoint at
    least `window` sessions ago. A rating jittering around a settled value scores
    near zero here however far it travelled in between, which is exactly the
    distinction a stopping rule needs. Until a checkpoint that old exists there is
    nothing to compare with and the answer is infinite.
    """
    played, now = history[-1]
    older = [ratings for sessions, ratings in history if sessions <= played - window]
    if not older:
        return float("inf"), float("inf")
    before = older[-1]
    moves = [abs(now[label] - before[label]) for label in now if label in before]
    if not moves:
        return float("inf"), float("inf")
    return max(moves), sum(moves) / len(moves)


def save_backup(global_dir: Path, anchors: list[Anchor]) -> Path:
    """Every anchor's rating as it stands now, so a run can be undone."""
    directory = global_dir / BACKUP_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ratings-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(
        json.dumps({a.label: a.start for a in anchors}, indent=2), encoding="utf-8"
    )
    return path


def write_ratings(
    global_dir: Path,
    anchors: list[Anchor],
    *,
    machine: str,
    lock_ttl: float,
    on_skip=None,
    games: bool = True,
) -> int:
    """Persist the new ratings, one member at a time under its own lock.

    One lock per member rather than all of them at once: `acquire_locks` is
    all-or-nothing, and asking for 150 at a time would fail whenever any single
    merge held any one of them. Each member is re-read inside its lock so that
    whatever an ordinary pass changed meanwhile -- `ref`, `style` -- is kept, and
    only `rating` and `games` are overwritten: this is the one program that moves
    either for an anchor (an ordinary pass seats an anchor as a fixed yardstick and
    counts nothing for it, `PoolRegistry.record_session_with_ratings`), so the
    games written are the ones this run started from plus the sessions it played.
    `games=False` writes the ratings alone (what `--restore` puts back: a backup
    records ratings, and the games an anchor has played stay played).

    A member that is not registered yet is created, frozen, exactly as a
    population pass would have bootstrapped it.
    """
    written = 0
    for anchor in anchors:
        if not acquire_locks(
            global_dir, [anchor.label], machine=machine, ttl=lock_ttl, timeout=lock_ttl
        ):
            if on_skip is not None:
                on_skip(anchor.label, "lock non ottenuto")
            continue
        try:
            member = read_member(global_dir, anchor.label)
            if member is None:
                member = PoolMember(
                    label=anchor.label,
                    ref=str(anchor.path),
                    rating=anchor.rating,
                    games=anchor.games if games else 0,
                    frozen=True,
                )
            else:
                member.rating = anchor.rating
                if games:
                    member.games = anchor.games
                member.frozen = True  # stays an anchor: nothing else may move it
            write_member(global_dir, member)
            written += 1
        finally:
            release_locks(global_dir, [anchor.label])
    return written


def restore(global_dir: Path, backup: Path, *, machine: str, lock_ttl: float) -> int:
    """Put back the ratings a backup recorded."""
    saved = json.loads(Path(backup).read_text(encoding="utf-8"))
    anchors = [
        Anchor(label=label, path=Path(""), series="", start=rating, rating=rating)
        for label, rating in saved.items()
    ]
    return write_ratings(global_dir, anchors, machine=machine, lock_ttl=lock_ttl, games=False)


def build_parser() -> argparse.ArgumentParser:
    """The flags, named as `config.toml` names them (see `population_arena.build_parser`)."""
    parser = argparse.ArgumentParser(
        description="Play the frozen benchmark anchors against each other and update "
        "their Elo. The only tool that may change an anchor's rating.",
    )
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--arena-max-sessions", type=int, default=DEFAULT_MAX_SESSIONS,
                        help="safety cap: stop after this many rated sessions even if the "
                        "ratings are still moving (0 plays nothing and only refreshes the "
                        "shared snapshot)")
    parser.add_argument("--arena-tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="converged when no anchor has drifted more than this "
                        "many rating points over --arena-window sessions")
    parser.add_argument("--arena-window", type=int, default=DEFAULT_WINDOW,
                        help="how many sessions the drift is measured across")
    parser.add_argument("--arena-min-sessions", type=int, default=DEFAULT_MIN_SESSIONS,
                        help="never declare convergence before this many rated sessions")
    parser.add_argument("--arena-checkpoint-sessions", type=int,
                        default=DEFAULT_CHECKPOINT_SESSIONS,
                        help="sessions played between two writes of the ratings (and two "
                        "reports, and two convergence tests): at most this many are lost "
                        "if the run is killed. Does not change the ratings a run ends with")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="local processes to split each batch of sessions across (1 = play in "
        "process). Playing parallelises perfectly -- every session is independent "
        "-- and a run is millions of hands, so this is the only thing "
        "that makes it finish in hours rather than days. Do NOT "
        "raise OMP_NUM_THREADS instead: threads inside one process buy almost "
        "nothing for this workload, while one process per core scales with the "
        "core count. Shards are launched with OMP_NUM_THREADS=1",
    )
    add_table_arguments(parser)
    parser.add_argument(
        "--device", default="auto", help="cpu, cuda or auto (the default): the GPU if there is one"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="play and report, write nothing")
    parser.add_argument("--restore", type=Path, default=None,
                        help="put back the ratings from a backup file and exit")
    parser.add_argument("--machine", default="benchmark-arena")
    parser.add_argument("--global-lock-seconds", type=int, default=DEFAULT_LOCK_SECONDS)
    parser.add_argument(
        "--k-schedule", type=k_schedule_text, default=format_k_schedule(DEFAULT_K_SCHEDULE),
        help="the Elo K staircase as games:K pairs, e.g. '0:16, 20:11, 45:7.4': the K a "
             "model is rated at once it has played that many sessions. Same key as "
             "`poker-train`'s, so the file sets one scale for everyone",
    )
    add_config_arguments(parser)
    return parser


def _sibling_parsers() -> list[argparse.ArgumentParser]:
    return sibling_parsers("pokerlab.rl.benchmark_arena", with_torch=False)


def main(argv: list[str] | None = None) -> int:
    # Lenient: this program cannot see `poker-train`'s keys without importing torch.
    args = resolve_cli(build_parser(), argv, siblings=_sibling_parsers, lenient=True)
    if args is None:
        return 0

    if args.arena_checkpoint_sessions < 1:
        print(f"--arena-checkpoint-sessions deve essere almeno 1, non {args.arena_checkpoint_sessions}")
        return 2

    if args.restore is not None:
        count = restore(
            args.global_dir, args.restore, machine=args.machine, lock_ttl=args.global_lock_seconds
        )
        print(f"ripristinati {count} rating da {args.restore}")
        return 0

    k_schedule = parse_k_schedule(args.k_schedule)
    anchors_list = collect_anchors(args.root, args.global_dir)
    mix = table_mix_from_args(args)
    if len(anchors_list) < mix.max_players:
        print(
            f"solo {len(anchors_list)} modelli benchmark sotto {args.root}/benchmark, "
            f"un tavolo da {mix.max_players} non si siede: niente da fare"
        )
        return 1
    anchors = {a.label: a for a in anchors_list}
    series_count = len({a.series for a in anchors_list})

    # The projected cost, up front: it is linear in the sessions played and in their
    # length, and at --session-hands 1000 the default 10,000 sessions are ten
    # million hands. Printing the projection is what stops a run being launched
    # with a two-day budget by accident.
    workers = max(1, min(args.workers, max(1, args.arena_max_sessions)))

    def projected_hours(sessions: int) -> float:
        minutes = sessions * args.session_hands / 80.0 / 60.0 / workers   # ~80 hands/s per core
        # Sharding is not free, and leaving its cost out makes this projection
        # optimistic. Every batch spawns `workers` fresh processes, and each imports
        # torch and loads *every* anchor before playing a hand: that is
        # `workers x anchors` checkpoints pulled over the same NFS mount, so the cost
        # goes roughly with their product rather than with one shard's startup. The
        # 0.0039 min per checkpoint load is calibrated on a single point, so read the
        # projection as an order of magnitude.
        if workers > 1:
            minutes += -(-sessions // args.arena_checkpoint_sessions) * workers * len(anchors_list) * 0.0039
        return minutes / 60.0

    first_possible = max(args.arena_min_sessions, args.arena_window)
    print(f"{len(anchors_list)} modelli benchmark in {series_count} cartelle, "
          f"sessioni da {args.session_hands} mani, {workers} "
          f"{'processi' if workers > 1 else 'processo'}")
    print(f"minimo {args.arena_min_sessions:,} sessioni = ~{projected_hours(args.arena_min_sessions):.1f} h, "
          f"tetto {args.arena_max_sessions:,} = ~{projected_hours(args.arena_max_sessions):.1f} h; "
          f"prima convergenza possibile dopo {first_possible:,} sessioni "
          f"(rating scritti ogni {args.arena_checkpoint_sessions})")
    if workers == 1 and args.arena_max_sessions * args.session_hands > 50_000:
        print("   suggerimento: --workers N divide il gioco su N processi "
              "(NON alzare OMP_NUM_THREADS, rallenta)")
    print(f"gioco finche' l'Elo non converge: nessuna ancora spostata piu' di "
          f"{args.arena_tolerance:.1f} punti in {args.arena_window:,} sessioni")
    print("K: scalini per esperienza di ogni ancora (k_schedule)")
    if args.dry_run:
        print("DRY RUN: nessun rating verra' scritto")
    else:
        backup = save_backup(args.global_dir, anchors_list)
        print(f"backup dei rating attuali: {backup}")
        print("   (per annullare: --restore " + str(backup) + ")")
    print("\nsituazione iniziale:")
    for line in format_table(anchors_list):
        print(line)

    candidates = [Candidate(label=a.label, path=a.path) for a in anchors_list]
    rng = random.Random(args.seed)
    total_sessions = 0
    total_hands = 0
    converged = False
    history: list[tuple[int, dict[str, float]]] = [(0, {a.label: a.rating for a in anchors_list})]

    while total_sessions < args.arena_max_sessions:
        batch = min(args.arena_checkpoint_sessions, args.arena_max_sessions - total_sessions)
        started = time.time()
        if args.workers > 1:
            # Same fan-out the population pass uses, and for the same reason:
            # playing parallelises perfectly, rating does not. `play_sharded`
            # splits the games exactly and forces OMP_NUM_THREADS=1 in each
            # child. Rating still happens here, in one place, session by session.
            sessions = play_sharded(
                candidates,
                mix,
                sessions=batch,
                session_hands=args.session_hands,
                device=resolve_device(args.device),
                rng=rng,
                workers=min(args.workers, batch),
                on_skip=lambda path, why: print(f"  shard saltato {path}: {why}"),
            )
        else:
            sessions = play_global_sessions(
                candidates,
                mix,
                sessions=batch,
                session_hands=args.session_hands,
                device=resolve_device(args.device),
                seed=rng.randrange(2**31),
                on_skip=lambda path, why: print(f"  saltato {path}: {why}"),
            )
        applied = apply_sessions(anchors, sessions, k_schedule)
        if applied == 0:
            # Nothing was played (every shard failed, say): looping on would spin
            # forever against a total that can never be reached.
            print("  nessuna sessione giocata, mi fermo")
            break
        total_sessions += applied
        total_hands += applied * args.session_hands
        history.append((total_sessions, {a.label: a.rating for a in anchors_list}))
        worst, average = drift(history, args.arena_window)
        mean = sum(a.rating for a in anchors_list) / len(anchors_list)
        shown = "  -" if worst == float("inf") else f"{worst:5.2f}"
        print(
            f"{total_sessions:>8,} sessioni  {time.time() - started:4.0f}s  "
            f"media {mean:7.2f}  deriva su {args.arena_window:,} sessioni: max {shown} "
            f"media {'  -' if average == float('inf') else f'{average:5.2f}'}  "
            f"({total_hands:,} mani in totale)",
            flush=True,
        )
        # **Persist at every checkpoint, not only at the end.** The write is 60
        # small files, each under its own lock and re-read so only `rating` is
        # overwritten -- milliseconds against minutes of play. Writing once at
        # the end meant a run killed on its second day lost everything it had
        # learned, which for a run of this length is the likeliest way for it to
        # end. The ratings are a fixed point the whole store is measured against,
        # so each checkpoint leaving them in a consistent state is also the safer
        # thing: every checkpoint is a complete result, not a partial one. The
        # backup taken at startup still undoes the whole run.
        if not args.dry_run:
            written = write_ratings(
                args.global_dir,
                anchors_list,
                machine=args.machine,
                lock_ttl=args.global_lock_seconds,
                on_skip=lambda label, why: print(f"  non scritto {label}: {why}"),
            )
            if written != len(anchors_list):
                print(f"  attenzione: scritti {written}/{len(anchors_list)} rating")
            # And refresh the shared `registry.json` snapshot in the same breath.
            # The member files above are the truth, but nothing *reads* them one
            # by one: `--status`, the dashboard and the GUI all read the
            # snapshot, and `write_snapshot`'s own staleness rule only refreshes
            # it when some other writer happens to come along. A long run can
            # hold the anchors for days, so without this the whole fleet would
            # keep reporting the ratings this run started from. Forced, for the
            # same reason the ratings are persisted at every checkpoint. Costs
            # ~2 s (9,300 member files re-read into one 3 MB file) against
            # minutes of play, and a failure here is bookkeeping -- it must not
            # kill a multi-day run.
            try:
                write_snapshot(args.global_dir, machine=args.machine, force=True)
            except OSError as exc:
                print(f"  snapshot registry.json non aggiornato: {exc}")
        # At every checkpoint: where each band sits, its span, how far its mean
        # has moved since the start, and the full per-anchor table -- watching the
        # anchors sort themselves out is the whole point of the run.
        print(format_series_line(anchors_list), flush=True)
        for line in format_table(anchors_list):
            print(line)
        if total_sessions >= args.arena_min_sessions and worst <= args.arena_tolerance:
            converged = True
            print(f"\nCONVERGENZA: nessuna ancora si e' spostata piu' di "
                  f"{args.arena_tolerance:.1f} punti nelle ultime {args.arena_window:,} sessioni")
            break

    if not converged and args.arena_max_sessions > 0 and total_sessions >= args.arena_max_sessions:
        print(f"\nNON CONVERGE entro {args.arena_max_sessions:,} sessioni: i rating si muovono "
              f"ancora piu' di {args.arena_tolerance:.1f} punti. Alza --arena-max-sessions, "
              f"o --arena-tolerance se questa precisione non serve.")

    print(f"\n=== fine: {total_sessions:,} sessioni, {total_hands:,} mani ===")
    for line in format_table(anchors_list):
        print(line)

    if args.dry_run:
        print("\nDRY RUN: nessun rating scritto, nessuna cartella toccata.")
        return 0
    # The ratings are already on disk -- every checkpoint wrote them. This last pass
    # is belt and braces for the final batch plus the snapshot refresh.
    written = write_ratings(
        args.global_dir,
        anchors_list,
        machine=args.machine,
        lock_ttl=args.global_lock_seconds,
        on_skip=lambda label, why: print(f"  non scritto {label}: {why}"),
    )
    write_snapshot(args.global_dir, machine=args.machine, force=True)

    print(f"scritti {written}/{len(anchors_list)} rating; snapshot registry.json aggiornato")
    return 0


if __name__ == "__main__":
    sys.exit(main())
