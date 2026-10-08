"""Reading a running loop's state and logs. **No torch, on purpose.**

Everything here is what a *watcher* needs: the supervisor's `loop_state.json`, the
workers' own logs, and the formatting of both. It lives apart from `rl/loop.py`
because `loop.py` imports `rl/ppo.py` for the benchmark, which imports torch --
and on the NFS server under load, importing torch can take **minutes**. A monitoring tool that takes five minutes to start is not a monitoring
tool, so `rl/dashboard.py` imports this and never touches `loop.py`.

`loop.py` re-exports these names, so `--status` and the tests keep importing them
from there; this module is the single definition.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pokerlab.rl.grad_log import GradientReading, parse_gradient_line
from pokerlab.rl.phases import (
    DONE,
    ELO_FILL,
    ELO_MERGE,
    ELO_PLAY,
    EVALUATION,
    HYPERPARAMETERS_PREFIX,
    PROGRESS_PREFIX,
    PRUNING,
    SERIES,
    Progress,
    parse_hyperparameters,
    parse_marker,
    parse_progress,
)
from pokerlab.rl.style_log import parse_style_line
from pokerlab.rl.value_diagnostics import SIZE_KIND, STACK_KIND, STREET_KIND, parse_value_line

STATE_FILENAME = "loop_state.json"
STOP_FILENAME = "STOP"


@dataclass
class LoopState:
    """What `--status` reads. Written after every phase, not just at the end,
    so a status check during a long generation still shows something true."""

    started: str = ""
    generation: int = 0
    phase: str = "idle"
    workers: int = 0
    central_pool: str = ""
    history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> LoopState:
        try:
            return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(path)


# What a worker can be doing, as far as its log shows. The stages a worker
# announces itself (`rl/phases.py`) plus the three the watcher infers.
STAGE_STARTING = "starting"  # no iteration finished yet
STAGE_TRAINING = "training"
STAGE_ERROR = "error"  # the log holds a traceback: the process died

STAGE_LABELS = {
    STAGE_STARTING: "avvio",
    STAGE_TRAINING: "training",
    EVALUATION: "valutazione",
    SERIES: "ancore",
    ELO_PLAY: "elo: gioco",
    ELO_MERGE: "elo: merge",
    PRUNING: "pruning",
    ELO_FILL: "elo: extra",
    DONE: "concluso",
    STAGE_ERROR: "ERRORE",
}
# The order the summary line lists them in: the order a run goes through them.
STAGE_ORDER = tuple(STAGE_LABELS)

# The training column averages the latest **100,000 hands**, not a fixed number
# of iterations: with `--hands` a swept axis, a window of 10 iterations would
# cover 3,200 hands for a worker that drew 320 and 8,000 for one that drew 800,
# so two rows of the same table would carry different noise and nothing would
# say which was which.
#
# Windows of a fixed number of hands put every worker's noise on the same footing
# whatever it drew. Measure that noise from first differences between consecutive
# iterations, not from the raw spread: a run improves over its life, and a raw
# standard deviation charges that trend as noise.
#
# The cost is that the window is *partial* early in a run (at 512 hands an
# iteration it fills at iteration 196). Partial is still shown rather than
# blanked, because "is this worker alive and not collapsed" is what the column
# is read for minute to minute, and `train_hands` says how much is behind it.
REWARD_WINDOW_HANDS = 100_000
# A log untouched this long, in a stage that should still be writing, means the
# process is gone or wedged. Generous on purpose: the population pass plays
# tens of thousands of hands between two lines.
STALE_LOG_SECONDS = 600

# `train.py` prints the reward in bb *per hand*; every column that reports a
# win rate in this project is bb/100, so the conversion happens here, at the
# parse boundary, and `train_bb100` is bb/100 from then on.
_ITER_REWARD = re.compile(r"^iter\s+\d+\s+reward\s+([+-]?\d+(?:\.\d+)?)\s+bb")
HANDS_PER_RATE = 100
# How many hands one iteration collected, off the header line every worker
# prints as it starts (`device cpu | tables 2:25% 3:20% ... | 512 hands/iteration`). Read from
# there rather than from the `iperparametri:` line because every log has it.
_HANDS_PER_ITERATION = re.compile(r"\|\s*(\d+)\s+hands/iteration")
_ITER_ENTROPY = re.compile(r"entropy\s+(\S+)")
_EVAL = re.compile(
    r"eval vs pool:\s*([+-]?\d+(?:\.\d+)?)\s*bb/100\s+rating\s+(-?\d+(?:\.\d+)?)"
)
# The rating of the field this worker drew, printed by `train.py` right after the
# draw. Without it the `eval` and `rating` columns cannot be read: they say how
# the learner did, this says against whom -- and every worker draws its own pool,
# so the same rating means different things in two rows of the same table.
_PARENT_RATING = re.compile(r"rating ereditato (-?\d+)")
_POOL_RATING = re.compile(r"^pool rating: media (-?\d+) min (-?\d+) max (-?\d+)\s*$")
# The one line `train.py` prints when the pass against the frozen anchors ends.
# It carries the bb/100, the rating the model is published with and the session
# count behind it -- which is the whole of what that pass concluded. A run
# that has not reached the pass leaves the two cells at "-".
_BENCHMARK_RATING = re.compile(
    r"benchmark: ([+-]?\d+(?:\.\d+)?) bb/100 su (\d+) sessioni contro (\d+) ancore, "
    r"rating (-?\d+(?:\.\d+)?)"
)
# The running figures `benchmark.rate_against_benchmark` puts in the detail of
# its `avanzamento series` lines. They fill the same two cells as the final
# result while the pass is still playing, flagged as provisional.
# The bb/100 is optional: a line may report the rating alone, and that is still
# worth putting in its column.
_SERIES_DETAIL = re.compile(
    r"rating (-?\d+(?:\.\d+)?)(?:, ([+-]?\d+(?:\.\d+)?) bb/100)?"
)


@dataclass(frozen=True)
class WorkerProgress:
    """One worker's state, read from its own log.

    `train_bb100` is the mean bb per 100 hands over the last `REWARD_WINDOW`
    iterations (what training optimises: the learner against the whole training
    field, copies of itself in the seats the pool did not fill included, so it is
    not the same measure as `eval`, which plays the drawn pool alone). `eval` and `rating`
    come from the last evaluation against the drawn pool, and each is "-" until
    the worker has produced one. `age` is how long ago the log was last written.
    """

    name: str
    iterations: int = 0
    inherited: bool = False
    # The parent's published rating, from the `resumed from` line; None when the
    # worker did not inherit or the log does not carry the figure.
    parent_rating: int | None = None
    stage: str = STAGE_STARTING
    train_bb100: float | None = None
    # How many hands `train_bb100` is averaged over. Below `REWARD_WINDOW_HANDS`
    # the window is still filling and the figure is correspondingly noisier; 0
    # means no iteration yet, or a log with no header line to size it from.
    train_hands: int = 0
    entropy: str = "-"
    eval: str = "-"
    rating: str = "-"
    # Mean rating of the models this worker drew for its training field, "-" until
    # the draw is logged (or for a run with an empty store).
    pool_rating: str = "-"
    # What this worker was configured with, as its own log reports it. Empty when
    # the log has no `iperparametri:` line yet, so every reader has to treat "not
    # recorded" as normal rather than as an error.
    hyperparameters: dict[str, str] = field(default_factory=dict)
    age: float | None = None
    # What the pass against the frozen anchors concluded: bb/100, the rating the
    # model is published with, and how many rated sessions were behind it. All
    # three stay empty until the run reaches that pass, which is the last thing
    # it does. The pass draws opponents from the whole frozen set, so there is
    # one number, not one per band.
    benchmark_bb100: float | None = None
    benchmark_rating: str = "-"
    benchmark_sessions: int = 0
    # True while the two figures above are the running values of a pass still
    # in progress, read off its `avanzamento` lines, not its conclusion.
    benchmark_live: bool = False
    # How far into the current stage the worker is, for the two stages that
    # report it (the benchmark pass and the population pass). None outside
    # them, and cleared the moment a new stage is announced, so a percentage
    # shown next to a stage always belongs to that stage.
    progress: Progress | None = None


def parse_worker_log(name: str, text: str, *, age: float | None = None) -> WorkerProgress:
    """Read a worker's progress out of the text of its log.

    The stage is whatever the *last* thing the log says implies: an iteration
    line means training, a `phase:` marker means that stage, and a result line
    (an evaluation) means the worker is back to training. A
    traceback is sticky -- nothing that follows it undoes a crash.

    The log is read while the worker is still appending to it, so the final line
    may be cut anywhere; every pattern below therefore tolerates a truncated
    line by simply not matching it.
    """
    iterations = 0
    inherited = False
    parent_rating: int | None = None
    stage = STAGE_STARTING
    rewards: list[float] = []
    entropy = eval_bb100 = rating = pool_rating = "-"
    hyperparameters: dict[str, str] = {}
    hands_per_iteration = 0
    benchmark_bb100: float | None = None
    benchmark_rating = "-"
    benchmark_sessions = 0
    benchmark_live = False
    progress: Progress | None = None
    for line in text.splitlines():
        if stage == STAGE_ERROR:
            break
        if line.startswith("resumed from "):
            inherited = True
            found = _PARENT_RATING.search(line)
            if found:
                parent_rating = int(found.group(1))
        elif line.startswith("device "):
            header = _HANDS_PER_ITERATION.search(line)
            if header:
                hands_per_iteration = int(header.group(1))
        elif line.startswith(HYPERPARAMETERS_PREFIX):
            parsed = parse_hyperparameters(line)
            if parsed:
                hyperparameters = parsed
        elif line.startswith("pool rating: "):
            matched = _POOL_RATING.match(line)
            if matched:
                pool_rating = matched.group(1)
        elif line.startswith("iter "):
            iterations += 1
            stage = STAGE_TRAINING
            matched = _ITER_REWARD.match(line)
            if matched:
                rewards.append(float(matched.group(1)))
            matched = _ITER_ENTROPY.search(line)
            if matched:
                entropy = matched.group(1)
        elif "eval vs pool:" in line:
            # Anchored on the eval line, not on "rating " anywhere: the final pool
            # summary a finished worker prints has a header reading "rating
            # partite", which a looser match reads as a rating of "partite".
            matched = _EVAL.search(line)
            if matched:
                eval_bb100, rating = matched.group(1), matched.group(2)
                stage = STAGE_TRAINING
        elif "benchmark:" in line:
            matched = _BENCHMARK_RATING.search(line)
            if matched:
                benchmark_bb100 = float(matched.group(1))
                benchmark_sessions = int(matched.group(2))
                benchmark_rating = matched.group(4)
                benchmark_live = False
        elif line.startswith(PROGRESS_PREFIX):
            reported = parse_progress(line)
            if reported is not None:
                progress = reported
                running = _SERIES_DETAIL.search(reported.detail)
                if reported.stage == SERIES and running and not (
                    benchmark_bb100 is not None and not benchmark_live
                ):
                    benchmark_rating = running.group(1).split(".")[0]
                    if running.group(2) is not None:
                        benchmark_bb100 = float(running.group(2))
                    benchmark_sessions = reported.done
                    benchmark_live = True
        elif line.startswith("Traceback (most recent call last)"):
            stage = STAGE_ERROR
        else:
            announced = parse_marker(line)
            if announced in STAGE_LABELS:
                stage = announced
                # A new stage starts at nothing done: without this, the
                # percentage of the stage just finished would be shown against
                # the stage just started.
                progress = None
    # As many of the latest iterations as it takes to cover the hand budget, or
    # all of them when the run has not produced that many yet. With the hand
    # count unknown -- no header line, which no real log lacks -- averaging
    # everything recorded is the best available answer, and `train_hands` says
    # 0 so nothing claims a precision it cannot back.
    if hands_per_iteration > 0:
        needed = max(1, math.ceil(REWARD_WINDOW_HANDS / hands_per_iteration))
        window = rewards[-needed:]
    else:
        window = list(rewards)
    return WorkerProgress(
        name=name,
        iterations=iterations,
        inherited=inherited,
        parent_rating=parent_rating,
        stage=stage,
        train_bb100=(
            sum(window) / len(window) * HANDS_PER_RATE if window else None
        ),
        train_hands=len(window) * hands_per_iteration,
        entropy=entropy,
        eval=eval_bb100,
        rating=rating,
        pool_rating=pool_rating,
        hyperparameters=hyperparameters,
        age=age,
        benchmark_bb100=benchmark_bb100,
        benchmark_rating=benchmark_rating,
        benchmark_sessions=benchmark_sessions,
        benchmark_live=benchmark_live,
        progress=progress if progress is not None and progress.stage == stage else None,
    )


def worker_progress(
    log_dir: Path, generation: int, *, now: float | None = None
) -> list[WorkerProgress]:
    """Live per-worker progress, read straight from the workers' own logs.

    The supervisor cannot report this: it is blocked in `process.wait()` for the
    whole training phase and learns nothing until a worker exits. The logs are
    the only thing that moves during those hours, so `--status` parses them
    rather than leaving the user staring at "phase: training".
    """
    current = time.time() if now is None else now
    rows: list[WorkerProgress] = []
    for path in sorted(Path(log_dir).glob(f"gen{generation:04d}-w*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            age = max(0.0, current - path.stat().st_mtime)
        except OSError:
            continue
        rows.append(parse_worker_log(path.stem.split("-", 1)[1], text, age=age))
    return rows


def stage_label(row: WorkerProgress, target_iterations: int) -> str:
    """The stage as shown, sharpened by how far through training the worker is."""
    # Past the last iteration the ordinary "training" is really the wrap-up:
    # saving, publishing, and the benchmark pass that rates the model.
    if target_iterations and row.iterations >= target_iterations and row.stage == STAGE_TRAINING:
        return "fine training"
    return STAGE_LABELS.get(row.stage, row.stage)


def _format_duration(seconds: float | None) -> str:
    """A duration in the shortest readable form. Used both for how long ago a
    log was written and for how much of a stage is left."""
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds // 3600:.0f}h{seconds % 3600 // 60:02.0f}"


def stage_cell(row: WorkerProgress, target_iterations: int) -> str:
    """The stage as the table shows it: its name, and -- for the two stages that
    report progress -- how far in it is and how much of it is left.

    Kept apart from `stage_label`, which answers "what is this worker doing" and
    is what the dashboard and the phase counts use; this answers "and how far
    along", which only a table cell wants.
    """
    cell = stage_label(row, target_iterations)
    if row.progress is None or row.progress.total <= 0:
        return cell
    cell += f" {row.progress.percent}%"
    if row.progress.eta_seconds is not None:
        cell += f" ~{_format_duration(row.progress.eta_seconds)}"
    return cell


def _is_stalled(row: WorkerProgress) -> bool:
    return (
        row.age is not None
        and row.age > STALE_LOG_SECONDS
        and row.stage not in (DONE, STAGE_ERROR)
    )


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):+.2f}" if values else "-"


def format_worker_table(rows: list[WorkerProgress], target_iterations: int) -> list[str]:
    """The per-worker block of the status: a summary, then one line per worker."""
    lines: list[str] = []
    inheriting = sum(1 for row in rows if row.inherited)
    total = sum(row.iterations for row in rows)
    lines.append(
        f"worker: {len(rows)}   iterazioni {total}/{len(rows) * target_iterations}   "
        f"{inheriting} ereditano i pesi (^), {len(rows) - inheriting} da zero (.)"
    )
    counts = {stage: sum(1 for row in rows if row.stage == stage) for stage in STAGE_ORDER}
    lines.append(
        "fasi  : "
        + ", ".join(f"{count} {STAGE_LABELS[stage]}" for stage, count in counts.items() if count)
    )
    lines.extend(format_finishing_line(rows))
    lines.extend(format_sweep_line(rows))

    rewards = [row.train_bb100 for row in rows if row.train_bb100 is not None]
    evals = [float(row.eval) for row in rows if row.eval != "-"]
    pools = [float(row.pool_rating) for row in rows if row.pool_rating != "-"]
    lines.append(
        f"medie : train {_mean(rewards)} bb/100 su {len(rewards)} worker, "
        f"eval {_mean(evals)} bb/100 su {len(evals)}, "
        f"rating del pool {_mean(pools)} su {len(pools)}"
    )
    stalled = [row.name for row in rows if _is_stalled(row)]
    if stalled:
        lines.append(
            f"ATTENZIONE: {len(stalled)} worker senza scrivere da oltre "
            f"{STALE_LOG_SECONDS // 60} minuti ({', '.join(stalled)}): processo morto o bloccato?"
        )

    lines.append("")
    lines.append(
        f"  {'':<8}{'progresso':<22}{'it':>8}  {'fase':<22}"
        f"{'train/100':>10}{'eval/100':>10}{'rating':>8}{'pool':>7}"
        f"{'ancore/100':>11}{'elo ancore':>11}{'entropia':>10}{'agg.':>7}"
    )
    for row in rows:
        filled = int(min(row.iterations / max(target_iterations, 1), 1.0) * 20)
        bar = "#" * filled + "." * (20 - filled)
        origin = "^" if row.inherited else "."
        train = f"{row.train_bb100:+.1f}" if row.train_bb100 is not None else "-"
        age = _format_duration(row.age) + ("!" if _is_stalled(row) else "")
        bench = f"{row.benchmark_bb100:+.1f}" if row.benchmark_bb100 is not None else "-"
        provisional = "~" if row.benchmark_live else ""
        bench = bench + provisional if bench != "-" else bench
        bench_rating = row.benchmark_rating + provisional if row.benchmark_rating != "-" else "-"
        lines.append(
            f"  {row.name:<5}{origin}  [{bar}]{row.iterations:4d}/{target_iterations:<4d} "
            f"{stage_cell(row, target_iterations):<22}"
            f"{train:>10}{row.eval:>10}{row.rating:>8}{row.pool_rating:>7}"
            f"{bench:>11}{bench_rating:>11}{row.entropy:>10}{age:>7}"
        )
    lines.append(
        f"  train/100 = bb ogni 100 mani nelle ultime {REWARD_WINDOW_HANDS:,} mani "
        "di training, contro pool e copie di se stesso (~+/-3 bb/100 a finestra piena; "
        "si riempie dopo ~1/5 della run); "
        "eval = ultima valutazione contro il pool; "
        "ancore = passata finale contro le ancore congelate (con ~ = valore provvisorio, "
        "la passata e' ancora in corso), elo ancore = rating con "
        "cui "
        "il modello entra nella classifica globale; pool = rating medio del campo "
        "che quel worker ha pescato, cioe' contro chi valgono eval e rating "
        "(entropia: tetto ln 11 = 2.40)"
    )
    return lines


# The arm names `loop.hyperparameter_plan` records, shortened for display. Kept
# here rather than imported from `loop.py`, which pulls torch in and which this
# module exists to stay independent of; an unknown arm falls back to its own
# name, so a new one shows up unabbreviated rather than disappearing.
ARM_LABELS = {
    "sampled": "campionati",
    "inherited": "ereditati",
    "sampled-fallback": "campionati (genitore senza metadati)",
}


def _as_number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("inf")


def format_sweep_line(rows: list[WorkerProgress]) -> list[str]:
    """One line saying what this generation is actually sweeping.

    Every worker now trains with its own settings, so "what is this machine
    running" no longer has one answer -- and the per-worker table has no room
    for six more columns. This is the summary that makes the sweep visible at
    all in the terminal: how the workers split between the arms, and the range
    the two widest axes were drawn over. The full set per worker is in the
    dashboard's expanded panel, and in each worker's own log.

    Absent entirely when no worker recorded any, which is every log written
    yet -- not an error, just nothing to say.
    """
    recorded = [row for row in rows if row.hyperparameters]
    if not recorded:
        return []
    arms: dict[str, int] = {}
    for row in recorded:
        # An absent key and an empty value mean the same thing: no arm was
        # assigned. `parse_hyperparameters` drops empty values, so both arrive
        # here as a missing key.
        arm = row.hyperparameters.get("hp_arm", "")
        arms[arm] = arms.get(arm, 0) + 1
    parts = [
        f"{count} {ARM_LABELS.get(arm, arm)}"
        for arm, count in sorted(arms.items(), key=lambda item: (-item[1], item[0]))
    ]
    line = f"sweep : {', '.join(parts)}"
    for axis, label in (("lr", "lr"), ("hands", "mani")):
        # Sorted numerically, not as text: the values are kept as strings so the
        # spelling the run used survives, and "1024" sorts before "320" as text.
        values = sorted(
            {row.hyperparameters[axis] for row in recorded if axis in row.hyperparameters},
            key=_as_number,
        )
        if values:
            line += f"   {label} {values[0]}" + (f"-{values[-1]}" if len(values) > 1 else "")
    if len(recorded) < len(rows):
        line += f"   ({len(rows) - len(recorded)} senza iperparametri registrati)"
    return [line]


def format_finishing_line(rows: list[WorkerProgress]) -> list[str]:
    """One line answering "how much longer", for the workers in a stage that
    reports it.

    The end of a generation is the part that looks stuck: the workers have
    finished training, their bars all read 100%, and they then spend an hour or
    more in the benchmark and population passes. The per-worker cells say how
    far each one is; this says when the machine as a whole should be free, which
    is the number a supervisor is waiting on -- a generation ends when its
    *slowest* worker does, so the line reports the longest ETA, not the mean.

    **The workers filling in are counted apart, and must be.** A worker in
    `ELO_FILL` has finished everything of its own and is playing extra rating
    passes precisely *because* it is waiting for the others; folding it in here
    would have it report an ETA of its own safety cap -- hours -- and, being the
    longest, that cap would become the generation's answer to "how much longer",
    which is exactly backwards. It is the one worker the generation is certainly
    not waiting for.
    """
    filling = [row for row in rows if row.stage == ELO_FILL]
    reporting = [
        row
        for row in rows
        if row.progress is not None and row.progress.total > 0 and row.stage != ELO_FILL
    ]
    if not reporting:
        if filling:
            return [
                (
                    f"fine  : {len(filling)}/{len(rows)} worker in riempimento elo, "
                    "in attesa degli altri"
                )
            ]
        return []
    mean_percent = sum(row.progress.percent for row in reporting) / len(reporting)
    etas = [
        row.progress.eta_seconds for row in reporting if row.progress.eta_seconds is not None
    ]
    line = (
        f"fine  : {len(reporting)}/{len(rows)} worker nelle passate finali "
        f"(elo e ancore), avanzamento medio {mean_percent:.0f}%"
    )
    if etas:
        line += f", il piu' lento ~{_format_duration(max(etas))}"
    if filling:
        line += f"; {len(filling)} in riempimento elo"
    return [line]



def benchmark_series(directory: str | Path) -> list[tuple[str, list[Path]]]:
    """The frozen set grouped the way it is stored: one entry per `benchmark_<N>/`
    directory, plus a `benchmark` entry for any loose checkpoints in the root.

    This is how the frozen set is *stored*, and the dashboard counts the groups
    for its "N serie" tile. Nothing scores a model against them separately any
    more: `benchmark.rate_against_benchmark` draws its opponents from the whole
    set, so a model's rating is one number rather than one per band. What a
    series still is, after `benchmark_arena`'s regroup, is a strength band -- a
    way of keeping the directories tidy, not a measurement.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    groups: list[tuple[str, list[Path]]] = []
    loose = sorted(directory.glob("*.pt"))
    if loose:
        groups.append((directory.name, loose))
    for child in sorted(p for p in directory.iterdir() if p.is_dir()):
        files = sorted(child.rglob("*.pt"))
        if files:
            groups.append((child.name, files))
    return groups


# Points per chart series. A run is 1000 iterations and a panel draws eight
# curves, so the whole history is thinned to this before it is sent to the page:
# ~10-20 KB a worker instead of megabytes, fetched only while a panel is open.
DEFAULT_MAX_POINTS = 400
# The whole of one `iter` line, all six metrics at once. `_ITER_REWARD` and
# `_ITER_ENTROPY` above read the two fields the status *table* needs from a
# partially written line; this one is for the charts, which want every field and
# can simply skip a line that does not match in full -- the logs are read while
# the workers are still appending to them, so the last line is routinely half
# written.
_ITER_FULL = re.compile(
    r"^iter\s+(\d+)\s+reward\s+([+-]?\d+(?:\.\d+)?)\s+bb\s+"
    r"policy\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"value\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"entropy\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"kl\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"clip\s+([+-]?\d+(?:\.\d+)?)\s*$"
)


def thin(count: int, max_points: int) -> list[int]:
    """Evenly spaced indices into a series of `count` points, last one included.

    Plain subsampling rather than averaging: these are already noisy per-
    iteration numbers and a chart of them is read for its trend and its
    outliers. Averaging would hide exactly the spike -- one iteration with a
    `kl` of 0.2 -- that the chart is there to reveal.
    """
    if count <= max_points or max_points <= 0:
        return list(range(count))
    step = count / max_points
    picked = sorted({min(count - 1, int(i * step)) for i in range(max_points)})
    if picked[-1] != count - 1:
        picked.append(count - 1)
    return picked


@dataclass(frozen=True)
class WorkerHistory:
    """One worker's whole run as series, ready to plot.

    The six per-iteration metrics are parallel lists, thinned together so they
    stay aligned with `iterations`. The two sparse readings are `[iteration,
    value]` pairs instead, because they are produced every `--eval-every`
    iterations rather than every one, and a chart has to place them at the
    iteration they belong to rather than spread them evenly.
    """

    name: str
    iterations: list[int] = field(default_factory=list)
    train_bb100: list[float] = field(default_factory=list)
    # The same rolling `REWARD_WINDOW_HANDS` mean the table's `train/100` shows,
    # one value per plotted iteration, so the chart's last point and the table
    # cell are the same number. Computed over *every* iteration and only then
    # thinned: `thin` subsamples without averaging, so a mean taken after it
    # would be a mean of a sample rather than of the window. Empty when the log
    # has no header line to size the window from.
    train_bb100_mean: list[float] = field(default_factory=list)
    policy: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    clip: list[float] = field(default_factory=list)
    eval_bb100: list[list[float]] = field(default_factory=list)
    eval_rating: list[list[float]] = field(default_factory=list)
    # The critic's target by table size, by effective stack and by street (`rl/value_diagnostics`),
    # one `[iteration, value]` pair per reading -- the first iteration and every
    # tenth -- like the two readings above, and for the same reason. The sd and
    # explained variance are keyed by group (`"2"`..`"9"`, `"<10"`..., `"preflop"`...), the spread
    # is the widest sd over the narrowest. Empty for a log that predates the lines.
    value_sd_size: dict[str, list[list[float]]] = field(default_factory=dict)
    value_sd_stack: dict[str, list[list[float]]] = field(default_factory=dict)
    value_ev_stack: dict[str, list[list[float]]] = field(default_factory=dict)
    value_sd_street: dict[str, list[list[float]]] = field(default_factory=dict)
    value_ev_street: dict[str, list[list[float]]] = field(default_factory=dict)
    value_spread_size: list[list[float]] = field(default_factory=list)
    value_spread_stack: list[list[float]] = field(default_factory=list)
    # The gradient-norm clips (`rl/grad_log.py`), one per network: the mean and sd of the
    # norm before the cut and the share of steps cut, one `[iteration, value]` pair per
    # kept iteration (thinned with the `iter` columns), and each threshold as of the
    # latest reading. Empty, and None, for a log without the line.
    grad_policy_mean: list[list[float]] = field(default_factory=list)
    grad_policy_sd: list[list[float]] = field(default_factory=list)
    grad_policy_clipped: list[list[float]] = field(default_factory=list)
    grad_policy_threshold: float | None = None
    grad_critic_mean: list[list[float]] = field(default_factory=list)
    grad_critic_sd: list[list[float]] = field(default_factory=list)
    grad_critic_clipped: list[list[float]] = field(default_factory=list)
    grad_critic_threshold: float | None = None
    # How the model plays (`rl/style_log.py`): each statistic's rate over its recent
    # training hands, one `[iteration, rate]` pair per reading (the first iteration
    # and every tenth), and the last reading's raw `[events, opportunities]` with the
    # seat-hands behind it. A statistic with no opportunity yet has no point.
    style_rate: dict[str, list[list[float]]] = field(default_factory=dict)
    style_latest: dict[str, list[int]] = field(default_factory=dict)
    style_hands: int = 0
    # The same per group of table sizes (`style_log.SIZE_GROUPS`):
    # `{"4-6": {"rate": {...}, "latest": {...}, "hands": n}}`. Empty for a log that
    # predates the per-size lines.
    style_by_size: dict[str, dict] = field(default_factory=dict)
    total_iterations: int = 0


def _at(values: list[float], keep: Sequence[int]) -> list[float]:
    """The kept indices of `values`, or nothing when there is nothing to keep."""
    return [values[k] for k in keep] if values else []


def rolling_reward(
    rows: Sequence[tuple[float, ...]], hands_per_iteration: int
) -> list[float]:
    """The trailing `REWARD_WINDOW_HANDS` mean at each iteration, in bb/100.

    The same window the table's `train/100` uses, so the chart's last point and
    the table cell agree by construction rather than by coincidence. Early
    iterations average everything available, exactly as the table does -- the
    curve therefore starts noisy and settles as the window fills, which is
    honest and is what `train_hands` reports alongside it.

    Empty when the hand count is unknown: there is then no way to size a window
    in hands, and a curve drawn to a different rule than the cell it sits under
    would be worse than no curve.
    """
    if hands_per_iteration <= 0 or not rows:
        return []
    needed = max(1, math.ceil(REWARD_WINDOW_HANDS / hands_per_iteration))
    prefix = [0.0]
    for row in rows:
        prefix.append(prefix[-1] + row[0])
    means = []
    for index in range(len(rows)):
        start = max(0, index - needed + 1)
        count = index + 1 - start
        means.append((prefix[index + 1] - prefix[start]) / count * HANDS_PER_RATE)
    return means


def parse_worker_history(
    name: str, text: str, *, max_points: int = DEFAULT_MAX_POINTS
) -> WorkerHistory:
    """Every per-iteration number in a worker's log, thinned for plotting.

    Tolerates a half-written final line the same way `parse_worker_log` does:
    the pattern is anchored at both ends, so a truncated line simply does not
    match and is dropped rather than parsed into a wrong point.
    """
    steps: list[int] = []
    rows: list[tuple[float, ...]] = []
    eval_bb: list[list[float]] = []
    eval_rating: list[list[float]] = []
    value_sd: dict[str, dict[str, list[list[float]]]] = {SIZE_KIND: {}, STACK_KIND: {}, STREET_KIND: {}}
    # Explained variance by stack and by street; by table size it is not drawn.
    value_ev: dict[str, dict[str, list[list[float]]]] = {STACK_KIND: {}, STREET_KIND: {}}
    value_spread: dict[str, list[list[float]]] = {SIZE_KIND: [], STACK_KIND: []}
    style_rate: dict[str, list[list[float]]] = {}
    style_latest: dict[str, list[int]] = {}
    style_hands = 0
    style_by_size: dict[str, dict] = {}
    hands_per_iteration = 0
    gradients: dict[int, GradientReading] = {}
    for line in text.splitlines():
        if line.startswith("device "):
            header = _HANDS_PER_ITERATION.search(line)
            if header:
                hands_per_iteration = int(header.group(1))
        elif line.startswith("iter "):
            matched = _ITER_FULL.match(line)
            if matched:
                steps.append(int(matched.group(1)))
                rows.append(tuple(float(matched.group(i)) for i in range(2, 8)))
        elif line.startswith("stile "):
            parsed_style = parse_style_line(line)
            if parsed_style is not None and steps:
                latest = {
                    stat: [events, chances]
                    for stat, (events, chances) in parsed_style.rates.items()
                }
                if parsed_style.group is None:
                    style_hands, style_latest, rates = parsed_style.hands, latest, style_rate
                else:
                    entry = style_by_size.setdefault(parsed_style.group, {"rate": {}})
                    entry["hands"], entry["latest"] = parsed_style.hands, latest
                    rates = entry["rate"]
                for stat, (events, chances) in parsed_style.rates.items():
                    if chances > 0:
                        rates.setdefault(stat, []).append([steps[-1], events / chances])
        elif line.startswith("gradienti:"):
            # Printed right after the `iter` line it belongs to.
            reading = parse_gradient_line(line)
            if reading is not None and steps:
                gradients[steps[-1]] = reading
        elif line.startswith("valore "):
            # Printed right after the `iter` line it describes, so it belongs to
            # the last iteration read.
            parsed = parse_value_line(line)
            if parsed is not None and steps:
                for group, stats in parsed.groups.items():
                    value_sd[parsed.kind].setdefault(group, []).append(
                        [steps[-1], stats.target_sd]
                    )
                    if parsed.kind in value_ev and stats.explained_variance is not None:
                        value_ev[parsed.kind].setdefault(group, []).append(
                            [steps[-1], stats.explained_variance]
                        )
                if parsed.spread is not None and parsed.kind in value_spread:
                    value_spread[parsed.kind].append([steps[-1], parsed.spread])
        elif "eval vs pool:" in line:
            matched = _EVAL.search(line)
            if matched and steps:
                eval_bb.append([steps[-1], float(matched.group(1))])
                eval_rating.append([steps[-1], float(matched.group(2))])
    keep = thin(len(rows), max_points)

    def column(index: int) -> list[float]:
        return [rows[k][index] for k in keep]

    kept_gradients = [(steps[k], gradients[steps[k]]) for k in keep if steps[k] in gradients]
    return WorkerHistory(
        name=name,
        iterations=[steps[k] for k in keep],
        train_bb100=[value * HANDS_PER_RATE for value in column(0)],
        train_bb100_mean=_at(rolling_reward(rows, hands_per_iteration), keep),
        policy=column(1),
        value=column(2),
        entropy=column(3),
        kl=column(4),
        clip=column(5),
        eval_bb100=eval_bb,
        eval_rating=eval_rating,
        value_sd_size=value_sd[SIZE_KIND],
        value_sd_stack=value_sd[STACK_KIND],
        value_ev_stack=value_ev[STACK_KIND],
        value_sd_street=value_sd[STREET_KIND],
        value_ev_street=value_ev[STREET_KIND],
        value_spread_size=value_spread[SIZE_KIND],
        value_spread_stack=value_spread[STACK_KIND],
        grad_policy_mean=[[it, g.policy.mean] for it, g in kept_gradients],
        grad_policy_sd=[[it, g.policy.sd] for it, g in kept_gradients],
        grad_policy_clipped=[[it, g.policy.clipped] for it, g in kept_gradients],
        grad_policy_threshold=kept_gradients[-1][1].policy.threshold if kept_gradients else None,
        grad_critic_mean=[[it, g.critic.mean] for it, g in kept_gradients],
        grad_critic_sd=[[it, g.critic.sd] for it, g in kept_gradients],
        grad_critic_clipped=[[it, g.critic.clipped] for it, g in kept_gradients],
        grad_critic_threshold=kept_gradients[-1][1].critic.threshold if kept_gradients else None,
        style_rate=style_rate,
        style_latest=style_latest,
        style_hands=style_hands,
        style_by_size=style_by_size,
        total_iterations=len(rows),
    )


def worker_history(
    log_dir: Path, generation: int, worker: str, *, max_points: int = DEFAULT_MAX_POINTS
) -> WorkerHistory | None:
    """`parse_worker_history` for one named worker of one generation, or None.

    The name is the one `worker_progress` reports (`w01`), and it is matched
    against the exact filename the loop writes rather than globbed, so nothing
    a caller passes can reach outside the log directory.
    """
    path = Path(log_dir) / f"gen{generation:04d}-{worker}.log"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return parse_worker_history(worker, text, max_points=max_points)
