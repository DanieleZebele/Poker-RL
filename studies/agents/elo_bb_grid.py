"""How many bb/100 is an Elo gap worth? A study script. Run by hand, nothing imports it.

Plays every benchmark model against every other one -- and against itself -- and
fills a grid whose cell `[row, column]` is what the row model earns, in bb/100,
against the column model. Models are sorted by rating, so reading along a row
shows how the edge grows with the gap.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/elo_bb_grid.py --hands 10000 --jobs 20

- **Table shape.** Always 6 seats, 3 against 3: the row model sits in three
  seats, the column model in the other three, so there is no third party to
  muddy the comparison. Seats are redrawn per match from a seed.
- **Duplicate decks.** Each off-diagonal match is played twice from the same
  seed with the teams' seats swapped (`duel_power.run_stream`), which cancels
  the luck of the cards and is what makes a few bb/100 resolvable from 10,000
  hands. `--hands` is the length of each of the two arrangements.
- **What a cell is.** The row model's *own* bb/100 per seat, i.e. what one seat
  of it wins per 100 hands. `duel_power` reports the head-to-head margin of one
  seat over one seat, which is twice that: with no third party, what the row
  wins the column loses, so the margin is `2 x` the cell. Keep the factor in
  mind when comparing this grid with the figures quoted elsewhere.
- **Antisymmetric by construction.** `[i, j] = -[j, i]`, because the same
  played hands fill both: each unordered pair is played once.
- **The diagonal is the noise floor.** A model against itself has an expected
  edge of zero, so what the diagonal shows is how far a cell can stray from the
  truth through seating and card luck alone. It is played in one arrangement:
  mirroring two identical models cancels to exactly zero, which would hide the
  very thing the diagonal is there to show.
- **Resumable.** Every finished match is appended to `--out`; running again with
  the same file skips the matches already there (a changed `--hands` or `--seed`
  is refused rather than silently mixed).

Torch is imported only inside `duel_power.run_stream`, so everything else here
-- the pairing, the fill, the fit -- is plain Python and tested without it.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from duel_power import DuelMeasurement, StreamJob, build_jobs, run_stream

from pokerlab.rl.device import resolve_device

DEFAULT_BENCHMARK_DIR = Path("checkpoints/benchmark")
DEFAULT_GLOBAL_DIR = Path("checkpoints/global")
DEFAULT_OUT = Path("checkpoints/studies/elo_bb_grid.json")
DEFAULT_HANDS = 10_000
DEFAULT_CSV_EVERY_SECONDS = 300.0
PLAYERS = 6
SEATS_PER_TEAM = PLAYERS // 2
STACK, SMALL_BLIND, BIG_BLIND = 200, 1, 2


@dataclass(frozen=True)
class GridModel:
    label: str
    path: str
    rating: float


@dataclass
class Cell:
    """The result of one unordered pair, from the point of view of the lower index."""

    i: int
    j: int
    bb100: float     # model i's own bb/100 per seat against model j
    stderr: float    # standard error of that, in bb/100
    hands: int


def load_models(benchmark_dir: Path, global_dir: Path) -> list[GridModel]:
    """Every benchmark checkpoint with its global rating, weakest first.

    A model with no rating on record gets the default, so the grid still runs;
    it is simply placed where a newcomer would be.
    """
    from pokerlab.rl.global_store import read_member
    from pokerlab.rl.pool_registry import DEFAULT_RATING

    models: list[GridModel] = []
    for path in sorted(Path(benchmark_dir).rglob("*.pt")):
        if path.name.startswith("."):
            continue
        member = read_member(global_dir, path.stem)
        models.append(GridModel(path.stem, str(path), member.rating if member else DEFAULT_RATING))
    models.sort(key=lambda m: (m.rating, m.label))
    return models


def thin(models: Sequence[GridModel], count: int | None) -> list[GridModel]:
    """`count` models spread evenly over the rating range (all of them if None).

    The cost is quadratic in the number of models, so a quick look at the shape
    of the relation does not need all of them.
    """
    if count is None or count >= len(models):
        return list(models)
    if count < 2:
        raise ValueError("a grid needs at least two models")
    last = len(models) - 1
    picks = sorted({round(k * last / (count - 1)) for k in range(count)})
    return [models[k] for k in picks]


def pairs(count: int) -> list[tuple[int, int]]:
    """Every unordered pair including each model with itself: n(n+1)/2 matches."""
    return [(i, j) for i in range(count) for j in range(i, count)]


def pair_seed(base_seed: int, i: int, j: int) -> int:
    """A seed that depends only on the pair, so a resumed run replays the same deals."""
    return random.Random(f"{base_seed}:{i}:{j}").randrange(2**31)


def make_job(models: Sequence[GridModel], i: int, j: int, *, hands: int, seed: int, device: str) -> StreamJob:
    (job,) = build_jobs(
        path_a=Path(models[i].path),
        path_b=Path(models[j].path),
        field_paths=[],
        players=PLAYERS,
        seats_per_team=SEATS_PER_TEAM,
        stack=STACK,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
        streams=1,
        hands=hands,
        mirror=i != j,
        device=device,
        seed=pair_seed(seed, i, j),
    )
    return job


def cell_from_scores(i: int, j: int, normal: list[float], mirrored: list[float]) -> Cell:
    """Turn the per-hand scores of one match into a cell.

    The scores are `A's seats minus B's seats` in chips; the matches have no
    third party, so A's own total is half of that, and `to_bb100` already divides
    by the seats of a team.
    """
    measurement = DuelMeasurement(
        label_a=str(i), label_b=str(j), seats_per_team=SEATS_PER_TEAM, big_blind=BIG_BLIND,
        normal=[normal], mirrored=[mirrored] if mirrored else [],
    )
    mean, stderr = measurement.edge()
    return Cell(
        i=i, j=j,
        bb100=measurement.to_bb100(mean) / 2.0,
        stderr=measurement.to_bb100(stderr) / 2.0 if stderr != float("inf") else float("inf"),
        hands=measurement.hands_played,
    )


def play_pair(args: tuple[StreamJob, int, int]) -> Cell:
    job, i, j = args
    normal, mirrored = run_stream(job)
    return cell_from_scores(i, j, normal, mirrored)


def fill_grid(count: int, cells: Sequence[Cell]) -> list[list[float | None]]:
    """The full square grid: `[i][j]` is i's bb/100 against j, `None` where unplayed."""
    grid: list[list[float | None]] = [[None] * count for _ in range(count)]
    for cell in cells:
        grid[cell.i][cell.j] = cell.bb100
        if cell.i != cell.j:
            grid[cell.j][cell.i] = -cell.bb100
    return grid


def slope_through_origin(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares `bb/100 = slope * gap`, forced through zero, and its R^2.

    Through the origin because equal models must earn nothing -- the diagonal
    says so -- so an intercept would only soak up noise. R^2 is taken against the
    mean-zero total, which is the fair baseline for a no-intercept fit.
    Returns None when there is nothing to fit.
    """
    denominator = sum(gap * gap for gap, _ in points)
    if not points or denominator == 0:
        return None
    slope = sum(gap * value for gap, value in points) / denominator
    total = sum(value * value for _, value in points)
    residual = sum((value - slope * gap) ** 2 for gap, value in points)
    return slope, (1.0 - residual / total) if total else 0.0


def gap_points(models: Sequence[GridModel], cells: Sequence[Cell]) -> list[tuple[float, float]]:
    """`(rating gap, bb/100)` for every played off-diagonal pair, higher-rated side first."""
    points = []
    for cell in cells:
        if cell.i == cell.j:
            continue
        gap = models[cell.j].rating - models[cell.i].rating  # j is the stronger (sorted)
        points.append((gap, -cell.bb100))
    return points


# ---- saving, so a run can be resumed -----------------------------------------


def save_study(path: Path, models: Sequence[GridModel], cells: Sequence[Cell], *, hands: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hands": hands,
        "seed": seed,
        "models": [asdict(m) for m in models],
        "cells": [asdict(c) for c in cells],
    }
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(json.dumps(payload), encoding="utf-8")
    staging.replace(path)


def load_study(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resumable(saved: dict, models: Sequence[GridModel], *, hands: int, seed: int) -> list[Cell]:
    """The cells of an earlier run of *this* experiment, or an error saying why not."""
    if (saved["hands"], saved["seed"]) != (hands, seed):
        raise SystemExit(
            f"il file e' di un altro esperimento (mani {saved['hands']}, seed {saved['seed']}): "
            "usa un altro --out, oppure gli stessi --hands e --seed"
        )
    if [m["label"] for m in saved["models"]] != [m.label for m in models]:
        raise SystemExit("il file riguarda un altro insieme di modelli: usa un altro --out")
    return [Cell(**c) for c in saved["cells"]]


# ---- reporting ------------------------------------------------------------------


def format_models(models: Sequence[GridModel]) -> list[str]:
    return [f"  {k:>3}  {m.rating:7.1f}  {m.label}" for k, m in enumerate(models)]


def format_grid(grid: Sequence[Sequence[float | None]], *, width: int = 7) -> list[str]:
    count = len(grid)
    header = " " * 5 + "".join(f"{k:>{width}}" for k in range(count))
    lines = [header]
    for i, row in enumerate(grid):
        cells = "".join(
            f"{'.':>{width}}" if value is None else f"{value:>{width}.1f}" for value in row
        )
        lines.append(f"{i:>4} {cells}")
    return lines


def write_csv(path: Path, models: Sequence[GridModel], grid: Sequence[Sequence[float | None]]) -> None:
    lines = ["rating,label," + ",".join(m.label for m in models)]
    for model, row in zip(models, grid):
        lines.append(
            f"{model.rating:.1f},{model.label},"
            + ",".join("" if v is None else f"{v:.2f}" for v in row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_summary(models: Sequence[GridModel], cells: Sequence[Cell]) -> list[str]:
    lines: list[str] = []
    diagonal = [c.bb100 for c in cells if c.i == c.j]
    if diagonal:
        lines.append(
            f"diagonale (modello contro se stesso, atteso 0): media {statistics.fmean(diagonal):+.1f}, "
            f"scarto tipico {statistics.pstdev(diagonal):.1f} bb/100 su {len(diagonal)} celle"
        )
    points = gap_points(models, cells)
    fit = slope_through_origin(points)
    if fit is not None:
        slope, r2 = fit
        lines.append(
            f"adattamento bb/100 = pendenza x gap Elo, per l'origine: {slope:.3f} bb/100 per punto "
            f"(R^2 {r2:.2f}, {len(points)} coppie)"
        )
        lines.append(
            f"  quindi 100 punti Elo ~ {100 * slope:.1f} bb/100 (guadagno proprio per seggio; "
            f"il margine testa a testa e' il doppio)"
        )
    return lines


# ---- running ------------------------------------------------------------------------


def run_grid(
    models: Sequence[GridModel],
    done: Sequence[Cell],
    *,
    hands: int,
    seed: int,
    device: str,
    jobs: int,
    on_cell: Callable[[Cell, int, int], None] | None = None,
) -> list[Cell]:
    """Play every match not already in `done`; return all cells, old and new."""
    cells = list(done)
    have = {(c.i, c.j) for c in cells}
    todo = [(i, j) for i, j in pairs(len(models)) if (i, j) not in have]
    work = [(make_job(models, i, j, hands=hands, seed=seed, device=device), i, j) for i, j in todo]
    if jobs <= 1:
        results = map(play_pair, work)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        results = executor.map(play_pair, work)
    try:
        for finished, cell in enumerate(results, start=1):
            cells.append(cell)
            if on_cell is not None:
                on_cell(cell, finished, len(work))
    finally:
        if executor is not None:
            executor.shutdown()
    return cells


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Griglia bb/100 tra tutti i modelli del benchmark (3 contro 3 su tavolo da 6), "
        "per capire quanti bb/100 vale una differenza di Elo.",
    )
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS,
                        help=f"mani di ogni scontro, per ciascuna delle due disposizioni "
                        f"(default: {DEFAULT_HANDS})")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--max-models", type=int, default=None,
                        help="usa solo N modelli distribuiti sull'intero intervallo di rating "
                        "(il costo cresce col quadrato)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="scontri in parallelo, un processo ciascuno (~600 MB l'uno)")
    parser.add_argument(
        "--device", default="auto", help="cpu, cuda or auto (the default): the GPU if there is one"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="dove salvare i risultati; rilanciando col medesimo file si riprende")
    parser.add_argument("--csv", type=Path, default=None, help="scrive anche la griglia in CSV")
    parser.add_argument("--csv-every", type=float, default=DEFAULT_CSV_EVERY_SECONDS,
                        help="ogni quanti secondi riscrivere il CSV con i risultati finora "
                        f"(default: {DEFAULT_CSV_EVERY_SECONDS:.0f}, cioe' 5 minuti)")
    args = parser.parse_args(argv)

    models = thin(load_models(args.benchmark_dir, args.global_dir), args.max_models)
    if len(models) < 1:
        raise SystemExit(f"nessun modello in {args.benchmark_dir}")

    done: list[Cell] = []
    if args.out.is_file():
        done = resumable(load_study(args.out), models, hands=args.hands, seed=args.seed)

    matches = len(pairs(len(models)))
    off_diagonal = matches - len(models)
    total_hands = (off_diagonal * 2 + len(models)) * args.hands
    print(f"{len(models)} modelli, {matches} scontri ({len(done)} gia' fatti), "
          f"{args.hands:,} mani a disposizione, ~{total_hands:,} mani in tutto")
    for line in format_models(models):
        print(line)
    sys.stdout.flush()

    started = time.time()
    saved_cells = list(done)
    last_csv = started

    def on_cell(cell: Cell, finished: int, total: int) -> None:
        # Saved after every match: a study of this size runs for hours.
        saved_cells.append(cell)
        save_study(args.out, models, saved_cells, hands=args.hands, seed=args.seed)
        nonlocal last_csv
        if args.csv is not None and time.time() - last_csv >= args.csv_every:
            # Unplayed cells are left empty, so a half-finished grid is readable.
            write_csv(args.csv, models, fill_grid(len(models), saved_cells))
            last_csv = time.time()
        print(f"  {finished:>5}/{total}  {cell.i:>3} vs {cell.j:<3} {cell.bb100:+8.1f} bb/100 "
              f"(+/-{cell.stderr:.1f})  {time.time() - started:6.0f}s", flush=True)

    cells = run_grid(models, done, hands=args.hands, seed=args.seed, device=resolve_device(args.device),
                     jobs=args.jobs, on_cell=on_cell)

    grid = fill_grid(len(models), cells)
    print("\nbb/100 del modello di riga contro quello di colonna (indici = elenco sopra):")
    for line in format_grid(grid):
        print(line)
    print()
    for line in format_summary(models, cells):
        print(line)
    if args.csv is not None:
        write_csv(args.csv, models, grid)
        print(f"griglia in {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
