"""Update every model's Elo by hand: the end-of-training-run rating, on demand.

`train.py` plays one cross-population Elo pass when a run finishes, which is
how ratings normally accumulate. That pass **samples**: ~50 models out of a
store holding ~9,200, drawn uniformly. It is the right rule there -- the same
pass can decide who gets deleted, so an honest draw matters more than coverage --
but it means a given model waits a long time for its turn. At ~1,000 new models
a generation and 55 seated per pass, a model has roughly a 0.5% chance of being
drawn in any one pass, so a rating can sit on the number it was published with
for many generations.

This module is the manual counterpart: the identical machinery, run until a
total number of sessions has been played, and with the two differences that
follow from nobody being deleted. **The length of a run is `--elo-sessions`, a
total of rated sessions, not a number of passes.** The sessions are played in
successive draws of `--global-sessions` of them (each draw is one sample of ~55
models held in memory, which is what makes a draw a unit of play), the last draw
taking whatever is left.

- **Least-played first** (`--elo-coverage played`, the default). `sample_population`
  takes the least-played half deterministically and fills the rest at random, so
  consecutive draws reach different models instead of re-drawing the same faces.
  That function has existed and been tested all along, unused, precisely because
  the automatic passes want a uniform draw; here coverage is the whole point.
  `--elo-coverage random` restores the uniform draw.
- **Pruning off** (`--prune` to allow it). Elimination physically deletes
  checkpoints, and it fires when the on-disk population reaches
  `--global-trigger-size`; a tool whose job is to refresh ratings must not delete
  anything as a side effect of being run. Off is implemented by raising the
  trigger out of reach rather than by a separate code path, so there is only one
  pruning rule in the project.

**Freezes are respected because nothing here bypasses them.** The merge goes
through `apply_pending_population_sessions` exactly as a training run's does, and
`record_session_with_ratings` scores a frozen member normally against everyone
else, counts its `games`, and never applies its own delta. The benchmark anchors
therefore stay the fixed scale the rest of the population is measured against.
`rl/benchmark_arena.py` remains the only code in the project that moves an
anchor's rating, and this is not it.

Two things this shares with every other writer of the shared store: sessions are
written to `global/pending/` before any lock is taken, so nothing played is ever
lost; and the merge locks only the ~6 participants of one session at a time, so
it never blocks a training run's own merge.

    python -m pokerlab.rl.population_arena --elo-sessions 2000 --workers 10

Every setting it shares with `poker-train` (the table, the sample, session
count and length, the lock, the pruning thresholds) is read from `config.toml`
under the same name, so the fleet and this tool cannot be tuned apart; a flag
still wins over the file. `--prune`, `--workers`, the directories and the seed
stay flags: they are choices of one invocation, not of the fleet.

A full sweep of the store at 55 seated per draw needs (store size / 55) draws,
which the startup projection spells out.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from pokerlab.config import add_config_arguments, resolve_cli
from pokerlab.rl.global_arena import (
    DEFAULT_BENCHMARK_SAMPLE,
    DEFAULT_GLOBAL_DIR,
    DEFAULT_GLOBAL_SESSIONS,
    DEFAULT_POPULATION_SAMPLE,
    NO_PRUNE_TRIGGER,
    Candidate,
    discover_population,
    run_population_sessions,
    sample_population,
)
from pokerlab.rl.global_store import (
    DEFAULT_LOCK_SECONDS,
    load_ranking,
    write_snapshot,
)
from pokerlab.rl.pool_registry import (
    DEFAULT_ELIMINATION_FRACTION,
    DEFAULT_K_SCHEDULE,
    DEFAULT_POPULATION_TRIGGER,
    DEFAULT_PROTECT_PERCENTILE,
    format_k_schedule,
    k_schedule_text,
    parse_k_schedule,
)
from pokerlab.rl.siblings import sibling_parsers
from pokerlab.rl.table_mix import add_table_arguments, table_mix_from_args


def least_played_draw(ratings: dict[str, dict]):
    """A `draw` callable for `run_population_sessions` biased to the least played."""

    def draw(population: list[Candidate], count: int, rng: random.Random):
        return sample_population(population, ratings, count, rng)

    return draw


def rating_summary(global_dir: Path) -> str:
    """The scale as it stands, for printing at the start and the end."""
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


def build_parser() -> argparse.ArgumentParser:
    """The flags, named as `config.toml` names them.

    A key of the shared file means one thing in every program, so a quantity
    `poker-train` also has keeps `poker-train`'s name (`--global-sample`,
    `--global-sessions`, ...) and reads the same value from the file.
    """
    parser = argparse.ArgumentParser(
        description="Aggiorna l'Elo di tutti i modelli: la passata di fine training, a mano.",
    )
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--machine", default="population-arena")
    parser.add_argument(
        "--elo-sessions", type=int, default=DEFAULT_GLOBAL_SESSIONS,
        help="sessioni valutate in tutto: e' l'unica misura di quanto dura la corsa. "
        "Si giocano a estrazioni di --global-sessions sessioni, ognuna con un campione "
        "nuovo di modelli",
    )
    parser.add_argument(
        "--global-sample", type=int, default=DEFAULT_POPULATION_SAMPLE,
        help="modelli della popolazione seduti per estrazione",
    )
    parser.add_argument(
        "--global-benchmark-sample", type=int, default=DEFAULT_BENCHMARK_SAMPLE,
        help="ancore congelate sedute per estrazione: restano ferme, e sono la scala "
        "fissa contro cui gli altri vengono misurati",
    )
    parser.add_argument(
        "--global-sessions", type=int, default=DEFAULT_GLOBAL_SESSIONS,
        help="sessioni giocate con uno stesso campione di modelli, ognuna con i "
        "giocatori estratti a caso",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="processi locali su cui dividere le sessioni di un'estrazione. Il "
        "parallelismo va preso qui, non alzando i thread di torch",
    )
    parser.add_argument(
        "--elo-coverage", choices=("played", "random"), default="played",
        help="played: i meno giocati per primi, per coprire la popolazione. "
        "random: estrazione uniforme, come quella automatica di fine training",
    )
    parser.add_argument(
        "--prune", action="store_true",
        help="permetti anche l'eliminazione dei modelli peggiori, che CANCELLA "
        "i file dal disco. Spento per default: questo strumento aggiorna i "
        "rating, non deve cancellare niente come effetto collaterale. Solo da "
        "riga di comando: un file condiviso non deve poterlo accendere per tutti",
    )
    add_table_arguments(parser)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--global-lock-seconds", type=int, default=DEFAULT_LOCK_SECONDS)
    parser.add_argument(
        "--global-trigger-size", type=int, default=DEFAULT_POPULATION_TRIGGER,
        help="usato solo con --prune",
    )
    parser.add_argument(
        "--global-eliminate-fraction", type=float, default=DEFAULT_ELIMINATION_FRACTION,
        help="usato solo con --prune",
    )
    parser.add_argument(
        "--global-protect-percentile", type=float, default=DEFAULT_PROTECT_PERCENTILE,
        help="usato solo con --prune",
    )
    parser.add_argument(
        "--k-schedule", type=k_schedule_text, default=format_k_schedule(DEFAULT_K_SCHEDULE),
        help="the Elo K staircase as games:K pairs, e.g. '0:16, 20:11, 45:7.4': the K a "
             "model is rated at once it has played that many sessions. Same key as "
             "`poker-train`'s, so the file sets one scale for everyone",
    )
    add_config_arguments(parser)
    return parser


def _sibling_parsers() -> list[argparse.ArgumentParser]:
    return sibling_parsers("pokerlab.rl.population_arena", with_torch=False)


def main(argv: list[str] | None = None) -> int:
    # Lenient: this program cannot see `poker-train`'s keys without importing torch.
    args = resolve_cli(build_parser(), argv, siblings=_sibling_parsers, lenient=True)
    if args is None:
        return 0

    mix = table_mix_from_args(args)
    population = discover_population(args.root)
    if len(population) < mix.max_players:
        print(f"solo {len(population)} modelli sotto {args.root}/models, "
              f"un tavolo da {mix.max_players} non si siede")
        return 1

    seated = min(args.global_sample, len(population)) + args.global_benchmark_sample
    batch_size = max(1, args.global_sessions)
    workers = max(1, min(args.workers, batch_size))
    hours = args.elo_sessions * args.session_hands / 80.0 / 60.0 / workers / 60.0
    draws_for_full_sweep = -(-len(population) // max(1, min(args.global_sample, len(population))))
    draws_planned = -(-args.elo_sessions // batch_size)

    print(f"{len(population):,} modelli nello store, {seated} seduti per estrazione "
          f"({args.elo_coverage})")
    print(f"{args.elo_sessions:,} sessioni da {args.session_hands} mani = "
          f"{args.elo_sessions * args.session_hands:,} mani, {workers} "
          f"{'processi' if workers > 1 else 'processo'} -> ~{hours:.1f} h")
    print(f"{draws_planned} estrazioni da {batch_size} sessioni; per coprire tutta la "
          f"popolazione ne servirebbero ~{draws_for_full_sweep}")
    print("pruning DISATTIVATO: nessun file verra' cancellato" if not args.prune
          else "PRUNING ATTIVO: i modelli peggiori possono essere CANCELLATI dal disco")
    print("le ancore congelate giocano e contano le partite, ma il loro rating non si muove")
    print(f"\nprima: {rating_summary(args.global_dir)}")

    rng = random.Random(args.seed)
    total_played = total_merged = played_sessions = 0
    while played_sessions < args.elo_sessions:
        started = time.time()
        batch = min(batch_size, args.elo_sessions - played_sessions)
        draw = None
        if args.elo_coverage == "played":
            # Re-read the ranking for every draw: the previous one just changed the
            # game counts this draw is ordered by, and other machines are merging
            # into the same store meanwhile.
            ranking = load_ranking(args.global_dir)
            counts = {label: {"games": m.games} for label, m in ranking.members.items()}
            draw = least_played_draw(counts)
        report = run_population_sessions(
            global_dir=args.global_dir,
            root=args.root,
            mix=mix,
            machine=args.machine,
            population_sample=args.global_sample,
            benchmark_sample=args.global_benchmark_sample,
            sessions=batch,
            session_hands=args.session_hands,
            device=args.device,
            seed=rng.randrange(2**31),
            workers=min(workers, batch),
            draw=draw,
            lock_ttl=args.global_lock_seconds,
            trigger_size=args.global_trigger_size if args.prune else NO_PRUNE_TRIGGER,
            eliminate_fraction=args.global_eliminate_fraction,
            protect_percentile=args.global_protect_percentile,
            k_schedule=parse_k_schedule(args.k_schedule),
            on_skip=lambda path, why: print(f"    saltato {path}: {why}"),
        )
        if report.sessions_played == 0:
            # Nothing came back (every shard failed, say): looping on would spin
            # forever against a total that can never be reached.
            print("    nessuna sessione giocata, mi fermo")
            break
        played_sessions += report.sessions_played
        total_played += report.played
        total_merged += report.sessions
        line = (
            f"{played_sessions:>8,}/{args.elo_sessions:,} sessioni  {report.played:>3} modelli  "
            f"{report.sessions:>4} sessioni valutate  {time.time() - started:4.0f}s"
        )
        if report.deferred_sessions:
            line += f"  {report.deferred_sessions} rinviate (lock occupati)"
        if report.eliminated:
            line += f"  ELIMINATI {report.eliminated}"
        if report.ghosts_dropped:
            line += f"  {report.ghosts_dropped} fantasmi rimossi"
        print(line, flush=True)
        # Refresh the shared `registry.json` snapshot after *every* draw, not
        # just once at the end. The merge inside `run_population_sessions` already
        # asks for one, but unforced: it refreshes only if the snapshot has gone
        # stale and only if it wins the lock, so a long run can leave everything
        # that reads the snapshot -- `--status`, the dashboard, the GUI -- showing
        # ratings this run has already superseded. Same trade as in
        # `benchmark_arena`: ~2 s against minutes of play, and a failure is
        # bookkeeping, not a reason to abandon the loop.
        try:
            write_snapshot(args.global_dir, machine=args.machine, force=True)
        except OSError as exc:
            print(f"    snapshot registry.json non aggiornato: {exc}")

    write_snapshot(args.global_dir, machine=args.machine, force=True)
    print(f"\n=== fine: {played_sessions:,} sessioni giocate, {total_played} posti, "
          f"{total_merged} sessioni valutate ===")
    print(f"dopo:  {rating_summary(args.global_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
