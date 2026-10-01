"""Update every model's Elo by hand: the end-of-training-run round, on demand.

`train.py` plays one cross-population Elo round when a run finishes, which is
how ratings normally accumulate. That round **samples**: ~50 models out of a
store holding ~9,200, drawn uniformly. It is the right rule there -- the same
pass can decide who gets deleted, so an honest draw matters more than coverage --
but it means a given model waits a long time for its turn. At ~1,000 new models
a generation and 55 seated per round, a model has roughly a 0.5% chance of being
drawn in any one round, so a rating can sit on the number it was published with
for many generations.

This module is the manual counterpart: the identical machinery, run in a loop
until the population has been covered, and with the two differences that follow
from nobody being deleted.

- **Least-played first** (`--coverage played`, the default). `sample_population`
  takes the least-played half deterministically and fills the rest at random, so
  consecutive rounds reach different models instead of re-drawing the same faces.
  That function has existed and been tested all along, unused, precisely because
  the automatic rounds want a uniform draw; here coverage is the whole point.
  `--coverage random` restores the uniform draw.
- **Pruning off** (`--prune` to allow it). Elimination physically deletes
  checkpoints, and it fires when the on-disk population reaches
  `--global-trigger-size`; a tool whose job is to refresh ratings must not delete
  anything as a side effect of being run. Off is implemented by raising the
  trigger out of reach rather than by a separate code path, so there is only one
  pruning rule in the project.

**Freezes are respected because nothing here bypasses them.** The merge goes
through `apply_pending_population_rounds` exactly as a training run's does, and
`record_session_with_ratings` scores a frozen member normally against everyone
else, counts its `games`, and never applies its own delta. The benchmark anchors
therefore stay the fixed scale the rest of the population is measured against.
`rl/benchmark_arena.py` remains the only code in the project that moves an
anchor's rating, and this is not it.

Two things this shares with every other writer of the shared store: sessions are
written to `global/pending/` before any lock is taken, so nothing played is ever
lost; and the merge locks only the ~6 participants of one session at a time, so
it never blocks a training run's own merge.

    python -m pokerlab.rl.population_arena --rounds 20 --workers 10

A full sweep of ~9,200 models at 55 seated per round needs ~170 rounds, which the
startup projection spells out.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from pokerlab.engine.config import GameConfig
from pokerlab.rl.global_arena import (
    DEFAULT_BENCHMARK_SAMPLE,
    DEFAULT_GAMES_PER_MODEL,
    DEFAULT_GLOBAL_DIR,
    DEFAULT_HANDS_PER_GAME,
    DEFAULT_POPULATION_SAMPLE,
    NO_PRUNE_TRIGGER,
    Candidate,
    discover_population,
    run_population_round,
    sample_population,
)
from pokerlab.rl.global_store import (
    DEFAULT_LOCK_SECONDS,
    load_ranking,
    write_snapshot,
)
from pokerlab.rl.pool_registry import (
    DEFAULT_ELIMINATION_FRACTION,
    DEFAULT_POPULATION_TRIGGER,
    DEFAULT_PROTECT_PERCENTILE,
)


def least_played_draw(ratings: dict[str, dict]):
    """A `draw` callable for `run_population_round` biased to the least played."""

    def draw(population: list[Candidate], count: int, rng: random.Random):
        return sample_population(population, ratings, count, rng)

    return draw


def rating_summary(global_dir: Path) -> str:
    """The scale as it stands, for printing between rounds."""
    ranking = load_ranking(global_dir)
    members = list(ranking.members.values())
    if not members:
        return "registro vuoto"
    ratings = sorted((m.rating for m in members), reverse=True)
    games = [m.games for m in members]
    top = ratings[: max(1, len(ratings) // 100)]
    return (
        f"{len(members):,} membri  media {sum(ratings) / len(ratings):.0f}  "
        f"mediana {ratings[len(ratings) // 2]:.0f}  "
        f"1% migliore {sum(top) / len(top):.0f}  massimo {ratings[0]:.0f}  "
        f"partite: mediana {sorted(games)[len(games) // 2]}, "
        f"mai valutati {sum(1 for g in games if g == 0):,}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggiorna l'Elo di tutti i modelli: il giro di fine training, a mano.",
    )
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--machine", default="population-arena")
    parser.add_argument("--rounds", type=int, default=1, help="quanti giri giocare")
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_POPULATION_SAMPLE,
        help="modelli della popolazione seduti per giro",
    )
    parser.add_argument(
        "--benchmark-sample", type=int, default=DEFAULT_BENCHMARK_SAMPLE,
        help="ancore congelate sedute per giro: restano ferme, e sono la scala "
        "fissa contro cui gli altri vengono misurati",
    )
    parser.add_argument("--games-per-model", type=int, default=DEFAULT_GAMES_PER_MODEL)
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS_PER_GAME)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="processi locali su cui dividere le sessioni di un giro. Il "
        "parallelismo va preso qui, non alzando i thread di torch",
    )
    parser.add_argument(
        "--coverage", choices=("played", "random"), default="played",
        help="played: i meno giocati per primi, per coprire la popolazione. "
        "random: estrazione uniforme, come i giri automatici",
    )
    parser.add_argument(
        "--prune", action="store_true",
        help="permetti anche l'eliminazione dei modelli peggiori, che CANCELLA "
        "i file dal disco. Spento per default: questo strumento aggiorna i "
        "rating, non deve cancellare niente come effetto collaterale",
    )
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lock-seconds", type=float, default=DEFAULT_LOCK_SECONDS)
    parser.add_argument(
        "--trigger-size", type=int, default=DEFAULT_POPULATION_TRIGGER,
        help="usato solo con --prune",
    )
    parser.add_argument(
        "--eliminate-fraction", type=float, default=DEFAULT_ELIMINATION_FRACTION,
        help="usato solo con --prune",
    )
    parser.add_argument(
        "--protect-percentile", type=float, default=DEFAULT_PROTECT_PERCENTILE,
        help="usato solo con --prune",
    )
    args = parser.parse_args(argv)

    game = GameConfig(
        num_players=args.players, starting_stack=args.stack,
        small_blind=args.sb, big_blind=args.bb,
    )
    population = discover_population(args.root)
    if len(population) < args.players:
        print(f"solo {len(population)} modelli sotto {args.root}/models, "
              f"un tavolo da {args.players} non si siede")
        return 1

    seated = min(args.sample, len(population)) + args.benchmark_sample
    sessions_per_round = seated * args.games_per_model // args.players
    hands_per_round = sessions_per_round * args.hands
    workers = max(1, min(args.workers, args.games_per_model))
    minutes = hands_per_round / 80.0 / 60.0 / workers
    rounds_for_full_sweep = -(-len(population) // max(1, min(args.sample, len(population))))

    print(f"{len(population):,} modelli nello store, {seated} seduti per giro "
          f"({args.coverage})")
    print(f"giro: {sessions_per_round} sessioni da {args.hands} mani = "
          f"{hands_per_round:,} mani, {workers} "
          f"{'processi' if workers > 1 else 'processo'} -> ~{minutes:.0f} min")
    print(f"{args.rounds} giri richiesti -> ~{minutes * args.rounds / 60:.1f} h; "
          f"per coprire tutta la popolazione servirebbero ~{rounds_for_full_sweep} giri")
    print("pruning DISATTIVATO: nessun file verra' cancellato" if not args.prune
          else "PRUNING ATTIVO: i modelli peggiori possono essere CANCELLATI dal disco")
    print("le ancore congelate giocano e contano le partite, ma il loro rating non si muove")
    print(f"\nprima: {rating_summary(args.global_dir)}")

    rng = random.Random(args.seed)
    total_played = total_sessions = 0
    for index in range(1, args.rounds + 1):
        started = time.time()
        draw = None
        if args.coverage == "played":
            # Re-read the ranking each round: the previous round just changed the
            # game counts this draw is ordered by, and other machines are merging
            # into the same store meanwhile.
            ranking = load_ranking(args.global_dir)
            counts = {label: {"games": m.games} for label, m in ranking.members.items()}
            draw = least_played_draw(counts)
        report = run_population_round(
            global_dir=args.global_dir,
            root=args.root,
            game=game,
            machine=args.machine,
            population_sample=args.sample,
            benchmark_sample=args.benchmark_sample,
            games_per_model=args.games_per_model,
            hands_per_game=args.hands,
            device=args.device,
            seed=rng.randrange(2**31),
            workers=workers,
            draw=draw,
            lock_ttl=args.lock_seconds,
            trigger_size=args.trigger_size if args.prune else NO_PRUNE_TRIGGER,
            eliminate_fraction=args.eliminate_fraction,
            protect_percentile=args.protect_percentile,
            on_skip=lambda path, why: print(f"    saltato {path}: {why}"),
        )
        total_played += report.played
        total_sessions += report.sessions
        line = (
            f"giro {index:>4}/{args.rounds}  {report.played:>3} modelli  "
            f"{report.sessions:>4} sessioni valutate  {time.time() - started:4.0f}s"
        )
        if report.deferred_sessions:
            line += f"  {report.deferred_sessions} rinviate (lock occupati)"
        if report.eliminated:
            line += f"  ELIMINATI {report.eliminated}"
        if report.ghosts_dropped:
            line += f"  {report.ghosts_dropped} fantasmi rimossi"
        print(line, flush=True)
        # Refresh the shared `registry.json` snapshot after *every* round, not
        # just once at the end. The merge inside `run_population_round` already
        # asks for one, but unforced: it refreshes only if the snapshot has gone
        # stale and only if it wins the lock, so a long run can leave everything
        # that reads the snapshot -- `--status`, the dashboard, the GUI -- showing
        # ratings this run has already superseded. Same trade as in
        # `benchmark_arena`: ~2 s against a round measured in minutes to hours,
        # and a failure is bookkeeping, not a reason to abandon the loop.
        try:
            write_snapshot(args.global_dir, machine=args.machine, force=True)
        except OSError as exc:
            print(f"    snapshot registry.json non aggiornato: {exc}")

    write_snapshot(args.global_dir, machine=args.machine, force=True)
    print(f"\n=== fine: {args.rounds} giri, {total_played} posti, "
          f"{total_sessions} sessioni valutate ===")
    print(f"dopo:  {rating_summary(args.global_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
