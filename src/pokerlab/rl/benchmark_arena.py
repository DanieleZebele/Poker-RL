"""Rate the frozen benchmark anchors against each other. Run by hand, rarely.

The anchors in `checkpoints/benchmark/` are the fixed reference points the whole
global scale is measured against: `PoolMember.frozen` pins their ratings, and
`record_session_with_ratings` never applies a frozen member's delta, so no
training run, no population round and no merge can move them. That is the point
-- a scale whose reference points drift measures nothing.

**This module is the single deliberate exception, and the only code in the
project that writes an anchor's rating.** It does not go through
`record_session_with_ratings` (which would refuse) but computes the deltas with
`pairwise_elo_delta` and writes the member files itself. Nothing imports it;
it exists to be run as a command, by a person who means it:

    python -m pokerlab.rl.benchmark_arena --rounds 5

Why it is ever needed: an anchor is pinned at whatever rating it happened to
hold the instant it was eliminated and promoted, which is one number from the
ordinary population rounds, often earned against a field that no longer exists.
The anchors are never rated against *each other*, so nothing has ever checked
that `benchmark_1`'s models and `benchmark_14`'s sit correctly relative to one
another. This plays them among themselves and settles that.

Two properties are deliberate:

- **K always follows the hyperbolic staircase** (`DEFAULT_K_SCHEDULE`, read from
  each anchor's own `games`, as everywhere else in the rating path); there is no
  flat-K option. The cost, accepted at the user's decision: with per-anchor Ks a
  session is no longer zero-sum in rating, so the anchors' *mean* can move,
  where a flat K kept it exactly fixed. Veteran anchors (tens of thousands of
  games) barely move at the bottom tier; a freshly added one moves most.
- **Only the ratings change.** Membership, `frozen`, `ref` and the checkpoint
  files are left exactly as they are, so afterwards the anchors are still frozen
  and still unreachable to everything else.

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
from dataclasses import dataclass
from pathlib import Path

from pokerlab.engine.config import GameConfig
from pokerlab.rl.global_arena import (
    DEFAULT_GLOBAL_DIR,
    DEFAULT_HANDS_PER_GAME,
    Candidate,
    discover_benchmark_population,
    play_global_round,
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
    DEFAULT_RATING,
    MODEL,
    PoolMember,
    k_for_games,
    pairwise_elo_delta,
)

DEFAULT_MAX_ROUNDS = 500
DEFAULT_GAMES_PER_MODEL = 20
# Convergence, and why it is measured as *net drift over a window* rather than
# per-round movement: with a flat K an Elo rating never stops moving, it jitters
# around its equilibrium forever and the jitter is proportional to K. A rule like
# "stop when every delta is under a point" would never fire. Net drift filters
# the jitter out -- a model bouncing +3/-3 has barely moved over ten rounds,
# while one still climbing has.
DEFAULT_TOLERANCE = 1.0     # rating points of net drift, per model, over the window
DEFAULT_WINDOW = 10         # rounds
DEFAULT_MIN_ROUNDS = 20
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
    the default rating, exactly as a population round would bootstrap it.
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
    anchors: dict[str, Anchor], sessions: list[dict[str, float]]
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
        k_factors = {label: k_for_games(anchors[label].games) for label in known}
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
    """One line with each series' rating span, for printing every round.

    `format_table` is 30 lines for 60 anchors, which every round would bury the
    round line it is meant to accompany; this is the same information at the
    resolution a per-round view needs -- where each band sits and how far it has
    moved since the run started. After `regroup_benchmark` the series *are* the
    strength bands, so these numbers read as a ladder and should stay ordered.
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


def drift(history: list[dict[str, float]], window: int) -> tuple[float, float]:
    """`(max, mean)` net rating movement per model across the last `window` rounds.

    Net, not cumulative: the difference between where each model sits now and
    where it sat `window` rounds ago. A rating jittering around a settled value
    scores near zero here however far it travelled in between, which is exactly
    the distinction a stopping rule needs.
    """
    if len(history) <= window:
        return float("inf"), float("inf")
    before, now = history[-1 - window], history[-1]
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
) -> int:
    """Persist the new ratings, one member at a time under its own lock.

    One lock per member rather than all of them at once: `acquire_locks` is
    all-or-nothing, and asking for 150 at a time would fail whenever any single
    merge held any one of them. Each member is re-read inside its lock so that
    whatever an ordinary round changed meanwhile -- `games`, `ref` -- is kept,
    and only `rating` is overwritten.

    A member that is not registered yet is created, frozen, exactly as a
    population round would have bootstrapped it.
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
                    kind=MODEL,
                    ref=str(anchor.path),
                    rating=anchor.rating,
                    frozen=True,
                )
            else:
                member.rating = anchor.rating
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
    return write_ratings(global_dir, anchors, machine=machine, lock_ttl=lock_ttl)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Play the frozen benchmark anchors against each other and update "
        "their Elo. The only tool that may change an anchor's rating.",
    )
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                        help="safety cap: stop after this many rounds even if the "
                        "ratings are still moving")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="converged when no anchor has drifted more than this "
                        "many rating points over --window rounds")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="how many rounds the drift is measured across")
    parser.add_argument("--min-rounds", type=int, default=DEFAULT_MIN_ROUNDS,
                        help="never declare convergence before this many rounds")
    parser.add_argument("--games-per-model", type=int, default=DEFAULT_GAMES_PER_MODEL,
                        help="rated sessions each anchor owes per round")
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS_PER_GAME,
                        help="hands per session")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="local processes to split each round's sessions across (1 = play in "
        "process). Playing parallelises perfectly -- every session is independent "
        "-- and a round is tens of thousands of hands, so this is the only thing "
        "that makes a convergence run finish in hours rather than days. Do NOT "
        "raise OMP_NUM_THREADS instead: measured on an idle 32-core box, one "
        "process at 15 threads plays 80.1 hands/s against 72.1 at one thread -- "
        "1.11x for fifteen cores -- while fifteen one-thread processes total "
        "1,102.9, i.e. 15.3x. Shards are launched with OMP_NUM_THREADS=1",
    )
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=1,
                        help="print the full per-anchor table every N rounds (1 = every "
                        "round, the default; 0 = only at the end). At 60 anchors it is 30 "
                        "lines, but a round is minutes long, so seeing which anchors moved "
                        "is worth the space -- watching them sort themselves out is the "
                        "whole point of the run. Turn it down for a long unattended sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="play and report, write nothing")
    parser.add_argument("--restore", type=Path, default=None,
                        help="put back the ratings from a backup file and exit")
    parser.add_argument("--machine", default="benchmark-arena")
    parser.add_argument("--lock-seconds", type=float, default=DEFAULT_LOCK_SECONDS)
    args = parser.parse_args(argv)

    if args.restore is not None:
        count = restore(
            args.global_dir, args.restore, machine=args.machine, lock_ttl=args.lock_seconds
        )
        print(f"ripristinati {count} rating da {args.restore}")
        return 0

    anchors_list = collect_anchors(args.root, args.global_dir)
    if len(anchors_list) < args.players:
        print(
            f"solo {len(anchors_list)} modelli benchmark sotto {args.root}/benchmark, "
            f"un tavolo da {args.players} non si siede: niente da fare"
        )
        return 1
    anchors = {a.label: a for a in anchors_list}
    series_count = len({a.series for a in anchors_list})

    # The projected cost, up front: a round is quadratic in nothing but it *is*
    # linear in both dials, and at --hands 1000 with 60 anchors it is 200,000
    # hands. Printing the projection is what stops a run being launched with a
    # two-day round by accident.
    # Approximate: the `due` queue seats whoever is still owed games, so the real
    # count lands a little above this, and sharding raises it slightly again
    # (each shard runs its own queue). Close enough to decide on a budget.
    sessions_per_round = len(anchors_list) * args.games_per_model // args.players
    hands_per_round = sessions_per_round * args.hands
    workers = max(1, min(args.workers, args.games_per_model))
    minutes = hands_per_round / 80.0 / 60.0 / workers   # ~80 hands/s per core
    # Sharding is not free, and leaving its cost out made this projection 4x
    # optimistic -- which defeats the point of printing one. Every round spawns
    # `workers` fresh processes, and each imports torch and loads *every* anchor
    # before playing a hand: that is `workers x anchors` checkpoints pulled over
    # the same NFS mount, so the cost goes roughly with their product rather than
    # with one shard's startup. Measured once, 35 anchors at --workers 20 on the
    # 32-core box: 235 s a round against the 73 s the hands themselves account
    # for. One calibration point, so read it as an order of magnitude.
    if workers > 1:
        minutes += workers * len(anchors_list) * 0.0039
    print(f"{len(anchors_list)} modelli benchmark in {series_count} cartelle, "
          f"{args.games_per_model} sessioni da {args.hands} mani per round")
    print(f"round: {sessions_per_round} sessioni, {hands_per_round:,} mani, "
          f"{workers} {'processi' if workers > 1 else 'processo'} -> ~{minutes:.0f} min "
          f"(minimo {args.min_rounds} round = ~{minutes * args.min_rounds / 60:.1f} h, "
          f"prima convergenza possibile al round "
          f"{max(args.min_rounds, args.window)})")
    if workers == 1 and hands_per_round > 50_000:
        print("   suggerimento: --workers N divide il round su N processi "
              "(NON alzare OMP_NUM_THREADS, rallenta)")
    print(f"gioco finche' l'Elo non converge: nessuna ancora spostata piu' di "
          f"{args.tolerance:.1f} punti in {args.window} round "
          f"(minimo {args.min_rounds}, tetto {args.max_rounds})")
    print("K: scalini iperbolici per esperienza di ogni ancora (DEFAULT_K_SCHEDULE)")
    if args.dry_run:
        print("DRY RUN: nessun rating verra' scritto")
    else:
        backup = save_backup(args.global_dir, anchors_list)
        print(f"backup dei rating attuali: {backup}")
        print("   (per annullare: --restore " + str(backup) + ")")
    print("\nsituazione iniziale:")
    for line in format_table(anchors_list):
        print(line)

    game = GameConfig(
        num_players=args.players, starting_stack=args.stack,
        small_blind=args.sb, big_blind=args.bb,
    )
    candidates = [Candidate(label=a.label, path=a.path) for a in anchors_list]
    rng = random.Random(args.seed)
    total_sessions = 0

    history: list[dict[str, float]] = [{a.label: a.rating for a in anchors_list}]
    converged = False
    round_index = 0
    total_hands = 0

    while round_index < args.max_rounds:
        round_index += 1
        started = time.time()
        if args.workers > 1:
            # Same fan-out the population round uses, and for the same reason:
            # playing parallelises perfectly, rating does not. `play_sharded`
            # splits the games exactly and forces OMP_NUM_THREADS=1 in each
            # child. Rating still happens here, in one place, session by session.
            sessions = play_sharded(
                candidates,
                game,
                games_per_model=args.games_per_model,
                hands_per_game=args.hands,
                device=args.device,
                rng=rng,
                workers=min(args.workers, args.games_per_model),
                on_skip=lambda path, why: print(f"  shard saltato {path}: {why}"),
            )
        else:
            sessions = play_global_round(
                candidates,
                game,
                games_per_model=args.games_per_model,
                hands_per_game=args.hands,
                device=args.device,
                seed=rng.randrange(2**31),
                on_skip=lambda path, why: print(f"  saltato {path}: {why}"),
            )
        applied = apply_sessions(anchors, sessions)
        total_sessions += applied
        total_hands += applied * args.hands
        history.append({a.label: a.rating for a in anchors_list})
        worst, average = drift(history, args.window)
        mean = sum(a.rating for a in anchors_list) / len(anchors_list)
        shown = "  -" if worst == float("inf") else f"{worst:5.2f}"
        print(
            f"round {round_index:>4}  {applied:>4} sessioni  {time.time() - started:4.0f}s  "
            f"media {mean:7.2f}  deriva su {args.window} round: max {shown} "
            f"media {'  -' if average == float('inf') else f'{average:5.2f}'}  "
            f"({total_hands:,} mani in totale)",
            flush=True,
        )
        # **Persist after every round, not only at the end.** The write is 60
        # small files, each under its own lock and re-read so only `rating` is
        # overwritten -- milliseconds against an hour of play. Writing once at
        # the end meant a run killed on its second day lost everything it had
        # learned, which for a job that needs `--min-rounds` x ~an-hour is the
        # likeliest way for it to end. The ratings are a fixed point the whole
        # store is measured against, so each round leaving them in a consistent
        # state is also the safer thing: every round is a complete result, not a
        # partial one. The backup taken at startup still undoes the whole run.
        if not args.dry_run:
            written = write_ratings(
                args.global_dir,
                anchors_list,
                machine=args.machine,
                lock_ttl=args.lock_seconds,
                on_skip=lambda label, why: print(f"  non scritto {label}: {why}"),
            )
            if written != len(anchors_list):
                print(f"  attenzione: scritti {written}/{len(anchors_list)} rating")
            # And refresh the shared `registry.json` snapshot in the same breath.
            # The member files above are the truth, but nothing *reads* them one
            # by one: `--status`, the dashboard and the GUI all read the
            # snapshot, and `write_snapshot`'s own staleness rule only refreshes
            # it when some other writer happens to come along. A convergence run
            # can hold the anchors for days, so without this the whole fleet
            # would keep reporting the ratings this run started from. Forced, for
            # the same reason the ratings are persisted per round: every round is
            # a complete result. Costs ~2 s (9,300 member files re-read into one
            # 3 MB file) against a round measured in minutes, and a failure here
            # is bookkeeping -- it must not kill a multi-day run.
            try:
                write_snapshot(args.global_dir, machine=args.machine, force=True)
            except OSError as exc:
                print(f"  snapshot registry.json non aggiornato: {exc}")
        # Every round, compactly: where each band sits, its span, and how far its
        # mean has moved since the start. The full per-anchor table stays behind
        # --report-every, since at 60 anchors it is 30 lines.
        print(format_series_line(anchors_list), flush=True)
        if args.report_every and round_index % args.report_every == 0:
            for line in format_table(anchors_list):
                print(line)
        if round_index >= args.min_rounds and worst <= args.tolerance:
            converged = True
            print(f"\nCONVERGENZA: nessuna ancora si e' spostata piu' di "
                  f"{args.tolerance:.1f} punti negli ultimi {args.window} round")
            break

    if not converged:
        print(f"\nNON CONVERGE entro {args.max_rounds} round: i rating si muovono "
              f"ancora piu' di {args.tolerance:.1f} punti. Alza --max-rounds, "
              f"o --tolerance se questa precisione non serve.")

    print(f"\n=== fine: {round_index} round, {total_sessions} sessioni, "
          f"{total_hands:,} mani ===")
    for line in format_table(anchors_list):
        print(line)

    if args.dry_run:
        print("\nDRY RUN: nessun rating scritto, nessuna cartella toccata.")
        return 0
    # The ratings are already on disk -- every round wrote them. This last pass
    # is belt and braces for the final round plus the snapshot refresh.
    written = write_ratings(
        args.global_dir,
        anchors_list,
        machine=args.machine,
        lock_ttl=args.lock_seconds,
        on_skip=lambda label, why: print(f"  non scritto {label}: {why}"),
    )
    write_snapshot(args.global_dir, machine=args.machine, force=True)

    print(f"scritti {written}/{len(anchors_list)} rating; snapshot registry.json aggiornato")
    return 0


if __name__ == "__main__":
    sys.exit(main())
