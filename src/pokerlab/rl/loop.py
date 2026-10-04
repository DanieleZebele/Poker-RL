"""A continuous training loop: N worker processes, one shared store of models.

Processes, not threads. The engine is pure Python and CPU-bound, so N threads
would serialise on the GIL and run at the speed of one; N processes genuinely
use N cores (measured: 30 concurrent runs on a 32-core box, ~57 hands/s each).

The loop runs in generations. Each generation launches N `poker-train` workers.
Every worker draws *its own* opponents from the shared store
(`checkpoints/models/`, see `rl/training_pool.py`), trains, publishes its best
model back into that store, and plays a cross-population rating round
(`rl/global_arena.py`) -- so nothing is seeded, merged or re-ranked here. The
ratings live in the global registry (`rl/global_store.py`), one file per model,
and every machine reads and writes it directly, which is what removed the need
for per-machine pools, an exchange directory and a retired archive.

What the loop still owns is the choreography around that: which workers inherit
weights (and from whom), the once-per-generation benchmark against the frozen
set, sweeping up after runs that were killed, and the status display.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from pokerlab.engine.config import GameConfig
from pokerlab.rl.benchmark import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_BENCHMARK_HANDS,
    DEFAULT_BENCHMARK_SESSIONS,
)
from pokerlab.rl.global_arena import (
    DEFAULT_BENCHMARK_SAMPLE,
    DEFAULT_GAMES_PER_MODEL,
    DEFAULT_GLOBAL_DIR,
    DEFAULT_HANDS_PER_GAME,
    DEFAULT_POPULATION_SAMPLE,
)
from pokerlab.rl.global_store import (
    DEFAULT_LOCK_SECONDS,
    DEFAULT_MODELS_DIR,
    claim_benchmark_arena,
    finish_benchmark_arena,
    load_global_registry,
    load_ranking,
    publish_model,
    read_sidecar,
)

# Re-exported from `rl/monitor.py`, which is deliberately torch-free so the
# dashboard can read a running loop without paying for torch (measured at over
# five minutes on the NFS server under load). `--status` and the tests keep
# importing these from here; monitor.py holds the single definition.
from pokerlab.rl.monitor import (  # noqa: F401 - re-exported for --status and tests
    REWARD_WINDOW_HANDS,
    STAGE_ERROR,
    STAGE_LABELS,
    STAGE_ORDER,
    STAGE_STARTING,
    STAGE_TRAINING,
    STALE_LOG_SECONDS,
    STATE_FILENAME,
    STOP_FILENAME,
    LoopState,
    WorkerProgress,
    format_finishing_line,
    format_sweep_line,
    format_worker_table,
    parse_worker_log,
    stage_cell,
    stage_label,
    worker_progress,
)
from pokerlab.rl.phases import FILL_DRAINING_FILENAME, FILL_STOP_FILENAME
from pokerlab.rl.pool_registry import (
    DEFAULT_ELIMINATION_FRACTION,
    DEFAULT_POOL_SIZE,
    DEFAULT_POPULATION_TRIGGER,
    DEFAULT_PROTECT_PERCENTILE,
    PoolRegistry,
)
from pokerlab.rl.ppo import PPOConfig, build_model_from_checkpoint

# The one thing the loop needs from `train.py`, which it otherwise only ever
# launches as a subprocess: how many rated sessions a validation round plays,
# so the default the loop forwards and the default a standalone `poker-train`
# uses cannot drift apart. No cycle -- train.py does not import loop.py -- and torch is already
# here through `rl/benchmark.py`.
from pokerlab.rl.train import (
    DEFAULT_EVAL_SESSIONS,
    DEFAULT_FILL_DEADLINE_MINUTES,
    DEFAULT_FILL_GAMES_PER_MODEL,
    DEFAULT_FILL_MIN_SESSIONS,
    RUN_METADATA_VERSION,
    TrainConfig,
)
from pokerlab.rl.training_pool import (
    DEFAULT_TOP_N,
    DEFAULT_TOP_SHARE,
    PARENT_TIERS,
    available_labels,
    pick_parents,
)

# Per-machine state (`loop_state.json`, the `STOP` file) lives here; the models
# and the ratings are shared and live elsewhere. The two file names come from
# `rl/monitor.py`, since a watcher needs them too.
DEFAULT_STATE_DIR = Path("checkpoints/state")


@dataclass
class GenerationRecord:
    generation: int
    started: str
    finished: str = ""
    workers: int = 0
    failed: int = 0
    published_models: int = 0
    best_label: str = ""
    best_rating: float = 0.0
    models_total: int = 0
    # Against the frozen set: the only number in this record that is comparable
    # across generations, because it is the only one whose opposition is fixed.
    benchmark_bb100: float | None = None
    benchmark_model: str = ""
    # The best model *this generation produced*, benchmarked separately. The
    # overall best can be -- and for several generations was -- a model that
    # predates the loop, in which case the headline number repeats unchanged and
    # says nothing about whether new training is improving.
    benchmark_new_bb100: float | None = None
    benchmark_new_model: str = ""



_WORKER_DIR = re.compile(r"^gen\d{4}-w\d{2}$")
_LIVE_CHECKPOINT = re.compile(r"^agent-(gen\d{4}-w\d{2})\.pt$")
_ARENA_DIR = re.compile(r"^arena-gen\d{4}$")
# Scratch that is not provably in use is only reclaimed once it has sat untouched
# this long, so a file another process is writing right now is never taken.
STALE_SCRATCH_SECONDS = 3600


@dataclass
class SweepReport:
    """What `sweep_stale_work` reclaimed."""

    work_dirs: int = 0
    salvaged: int = 0
    checkpoints: int = 0
    arena_dirs: int = 0
    partials: int = 0
    scratch_dirs: int = 0
    in_use: int = 0
    freed_bytes: int = 0

    @property
    def total(self) -> int:
        return (
            self.work_dirs + self.checkpoints + self.arena_dirs + self.partials + self.scratch_dirs
        )


def _live_commands() -> list[str] | None:
    """The command line of every running process, or None where /proc is absent
    (then nothing can be proven unused and the sweep does nothing)."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    me = str(os.getpid())
    commands: list[str] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or entry.name == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        commands.append(raw.replace(b"\0", b" ").decode(errors="replace"))
    return commands


def _tree_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total

def _salvage_archives(
    worker_dir: Path,
    models_dir: Path,
    global_dir: Path,
    *,
    machine: str,
    lock_ttl: float = DEFAULT_LOCK_SECONDS,
) -> int | None:
    """Publish an abandoned worker's own archive(s) into the shared store.

    A worker directory now holds nothing but what the worker itself produced
    (its best-so-far `agent-*.pt` and the sidecar recording its rating), so every
    such file is a model that would otherwise be invisible to every ranking. It
    is published under the same name the worker would have used, at the rating
    its sidecar recorded. Returns how many were saved, or None if a copy failed
    -- the caller then leaves the directory alone rather than lose the model.
    """
    saved = 0
    for path in sorted(worker_dir.glob("agent-*.pt")):
        name = f"{machine}-{worker_dir.name}-{path.name}"
        if (models_dir / name).exists():
            continue
        rating, iteration = read_sidecar(path)
        try:
            publish_model(
                path,
                models_dir=models_dir,
                global_dir=global_dir,
                name=name,
                rating=rating,
                iteration=iteration,
                machine=machine,
                lock_ttl=lock_ttl,
            )
        except OSError:
            return None
        saved += 1
    return saved


def sweep_stale_work(
    work_root: Path,
    models_dir: Path,
    global_dir: Path,
    *,
    machine: str,
    scratch_dir: Path | None = None,
    min_age: float = STALE_SCRATCH_SECONDS,
) -> SweepReport:
    """Clear what an interrupted training run left behind.

    A loop killed mid-generation (reboot, crash, `kill -9`) never gets to clean
    up after its workers, and the next run counts on from a new generation
    number, so the old `work/gen<N>-w<K>/` directories, the
    `work/agent-gen<N>-w<K>.pt` live checkpoints and any `work/arena-gen<N>/`
    left by an older version would sit there forever. This removes them, after
    first publishing each abandoned worker's own archive into the shared store --
    it is a trained model that would otherwise be invisible to every ranking.
    The same call runs after each generation's workers exit, which is how a worker
    that crashed before it could publish still gets its model in.

    Safety, in order of importance:
      * Anything a *running process* still names on its command line is left
        alone: a worker can outlive a supervisor that died, and deleting its
        directory would corrupt it. Liveness is read from `/proc`, which only
        sees this host -- which is exactly right, since `work_root` is this
        machine's own directory and no other host ever uses it. Where `/proc`
        does not exist nothing is deleted.
      * A directory whose archive could not be saved is kept.
      * Only names of the exact shapes this loop creates are touched.
    Also removes this machine's own stale `.partial` files in the models
    directory (a publish that died mid-copy) and stale `global-arena-shard-*`
    scratch left in the temp directory by a killed global round.
    """
    report = SweepReport()
    commands = _live_commands()
    if commands is None:
        return report

    def in_use(token: str) -> bool:
        return any(token in command for command in commands)

    if work_root.is_dir():
        for entry in sorted(work_root.iterdir()):
            name = entry.name
            try:
                if entry.is_dir() and _WORKER_DIR.match(name):
                    if in_use(name):
                        report.in_use += 1
                        continue
                    saved = _salvage_archives(entry, models_dir, global_dir, machine=machine)
                    if saved is None:
                        continue
                    report.salvaged += saved
                    report.freed_bytes += _tree_size(entry)
                    shutil.rmtree(entry, ignore_errors=True)
                    report.work_dirs += 1
                elif entry.is_file() and (live := _LIVE_CHECKPOINT.match(name)):
                    if in_use(live.group(1)):
                        report.in_use += 1
                        continue
                    report.freed_bytes += entry.stat().st_size
                    entry.unlink()
                    report.checkpoints += 1
                elif entry.is_dir() and _ARENA_DIR.match(name):
                    if in_use(name):
                        report.in_use += 1
                        continue
                    report.freed_bytes += _tree_size(entry)
                    shutil.rmtree(entry, ignore_errors=True)
                    report.arena_dirs += 1
            except OSError:
                continue

    now = time.time()
    if models_dir.is_dir():
        for partial in models_dir.glob(f".{machine}-*.partial"):
            try:
                if now - partial.stat().st_mtime >= STALE_SCRATCH_SECONDS:
                    report.freed_bytes += partial.stat().st_size
                    partial.unlink()
                    report.partials += 1
            except OSError:
                continue

    scratch_root = scratch_dir if scratch_dir is not None else Path(tempfile.gettempdir())
    for scratch in scratch_root.glob("global-arena-shard-*"):
        try:
            if scratch.is_dir() and now - scratch.stat().st_mtime >= min_age and not in_use(scratch.name):
                report.freed_bytes += _tree_size(scratch)
                shutil.rmtree(scratch, ignore_errors=True)
                report.scratch_dirs += 1
        except OSError:
            continue
    return report


# --- hyperparameters, one draw per worker -------------------------------------
#
# Every worker of a generation used to be launched with the *same* settings, so
# the fleet produced thousands of runs of one configuration and no evidence
# about any other. These ladders are what makes a generation a sweep instead:
# each worker gets its own point, recorded in the model it publishes
# (`train.run_metadata`), so an outcome can afterwards be attributed to the
# settings that produced it.
#
# **Each axis is a ladder of allowed values, and both arms move on it.** The
# sampled arm draws a rung uniformly and independently per axis -- independence
# is what lets the mean outcome over one axis be read as that axis's effect,
# with the others averaged out rather than confounded.
#
# **The two arms now move by different mechanisms, and that is deliberate.** The
# sampled arm draws a rung; the inherit arm multiplies its parent's value by one
# of `HP_MULTIPLIERS` and is *not* snapped back onto the ladder, so a lineage can
# hold values no rung holds and can walk past the ends of the ladder entirely.
# Changed at the user's request, from an earlier +/-1 rung step clamped to the
# ends. Two reasons it is the better fit for that arm:
#   * **The ends of these ladders are a guess, and the clamp made them
#     load-bearing.** A parent already on the top rung stayed there two thirds of
#     the time (both +1, clamped, and 0 returned it), so the ladder's author
#     decided in advance the furthest the fleet could ever go. Nothing measured
#     those ends.
#   * **The real bound is selection, and that one is measured.** A parent is
#     drawn from the `--pool-top-n` best-rated models, so a lineage that walks
#     its lr up to something that breaks training rates badly and stops being a
#     parent. An arithmetic clamp guesses where the edge is; the ranking finds
#     out.
# The cost, accepted: a lineage really can drift a long way. The step is a
# random walk in log space, so over N generations its spread is about
# ln(1.2)*sqrt(2N/3) -- roughly 27x either way over 500 generations, which would
# put `hands` anywhere between ~12 and ~8600. Nothing stops that but the
# ranking.
#
# **Only two limits survive, and neither is a rung.** A probability axis is
# capped at 1.0 (see `_HP_PROBABILITY_AXES`), because a probability above 1 is
# not one; and the count axes are *rounded*, never truncated, which is what
# keeps them off zero -- see `_typed`. Nothing has a lower cap, and none is
# needed: multiplying a positive number by 0.8 never reaches 0.
#
# **Multiplication now works on every axis, which it did not before.** One of
# the two original arguments for a ladder was `self_share`, whose bottom rung was
# 0.0 and which no multiplier can ever leave; that axis went with the snapshots,
# and no ladder has a 0 rung any more. The other argument -- that an integer
# multiplied drifts off any grid an analysis could group by -- still stands and is
# simply accepted: inherited points are a continuum now. That costs nothing that
# was not already lost, because an inherited point never could be read as a
# response curve (it correlates with its parent's quality by construction). The
# sampled arm is still on the grid, and it is still the readable one.
#
# The centre rung of every ladder is what the fleet runs today, so the sampled
# arm is a draw *around* the known-good point rather than a jump away from it.
#
# There used to be a `self_share` axis here -- how much of the opponent field was
# the run's own frozen snapshots. It is gone with the snapshots themselves,
# removed at the user's request: every opponent seat now faces a previously
# trained model from the store, so there is nothing left to divide.
HP_LADDERS: dict[str, tuple[float, ...]] = {
    "lr": (1.9e-4, 2.4e-4, 3.0e-4, 3.8e-4, 4.7e-4),
    "hands": (320, 400, 512, 640, 800),
    "opponent_probability": (0.32, 0.40, 0.50, 0.62, 0.78),
    "ppo_epochs": (2, 3, 4, 5, 6),
    "clip_epsilon": (0.13, 0.16, 0.20, 0.25, 0.31),
    # Added later, centred on the values every earlier run used (`PPOConfig`'s
    # defaults and `TrainConfig.lam`), so the sampled arm is again a draw around
    # the known-good point. `minibatch_size` sets how many gradient steps an
    # epoch takes, `gae_lambda` the bias/variance of the advantages,
    # `value_coef` how loudly the critic speaks against the policy, and
    # `max_grad_norm` how large one step may be.
    "minibatch_size": (512, 724, 1024, 1448, 2048),
    "gae_lambda": (0.90, 0.93, 0.95, 0.97, 0.99),
    "value_coef": (0.25, 0.35, 0.5, 0.7, 1.0),
    "max_grad_norm": (0.25, 0.35, 0.5, 0.7, 1.0),
    # Centred on 0.0, which is what the fleet runs (see
    # `PPOConfig.entropy_coefficient` for why it is off). Reopened at the user's
    # request; a positive bonus buys per-decision dithering, not strategy, so
    # selection is what decides whether any lineage keeps one.
    "entropy_coef": (0.0, 0.0, 0.0, 0.001, 0.003),
    # **How strong a field the worker trains against.** These two were held
    # fixed with `--pool-models` at first, on the grounds that who a run trains
    # against decides what its result means; they were then opened at the user's
    # request, to diversify the *strength* of the opposition across workers as
    # well as the optimiser settings.
    #
    # `draw_training_pool` fills `--pool-models` seats from two sources: a
    # `pool_top_share` fraction drawn uniformly from the `pool_top_n` best-rated
    # models, and the rest uniformly from the whole store. So the share decides
    # *how much* of the field is elite and `top_n` decides *how* elite that part
    # is -- a smaller `top_n` is a narrower, stronger band. Together they move
    # the mean rating of the drawn field, which is exactly the "pool strength"
    # being diversified, and which `--status` already reports per worker in its
    # `pool` column: the sweep should visibly spread a column that was until now
    # nearly identical across every row.
    #
    # **`--pool-models` stays fixed**, deliberately: it is the *size* of the
    # field, not its strength, and it is also what the run's cost scales with
    # (each drawn model is loaded into memory). Varying it would confound a
    # strength axis with a cost axis.
    "pool_top_share": (0.32, 0.40, 0.50, 0.62, 0.78),
    "pool_top_n": (51, 64, 80, 100, 125, 156, 195),
}

# The axes whose values are counts, so a value is cast back to `int`. Everything
# else is a float. This exists so `HP_LADDERS` is the only place an axis has to
# be added: both arms build a `Hyperparameters` from the dict generically, and
# the type is the one thing the ladder itself cannot say (0.0 and 0 look alike).
_HP_INT_AXES = frozenset({"hands", "ppo_epochs", "pool_top_n", "minibatch_size"})

# What a parent that predates an axis was effectively trained with. Without it
# every published model (none carries the four newest axes) would look unusable
# to `perturb_hyperparameters` and the whole fleet would fall back to sampling
# at once. Only axes added after the metadata schema was fixed belong here; the
# original ones stay strict, so a half-written parent is still refused.
_HP_DEFAULTS: dict[str, float] = {
    "minibatch_size": PPOConfig.minibatch_size,
    "gae_lambda": TrainConfig.lam,
    "value_coef": PPOConfig.value_coefficient,
    "max_grad_norm": PPOConfig.max_grad_norm,
    "entropy_coef": PPOConfig.entropy_coefficient,
}

# Zero is absorbing under a multiplier, and 0.0 is where this axis lives, so an
# upward draw from exactly zero lands on this value instead of staying put.
# Downward draws from zero stay zero; from a positive value they decay as usual.
_HP_SEED_FROM_ZERO: dict[str, float] = {"entropy_coef": 1e-3}

# What the inherit arm multiplies its parent's value by, one independent draw per
# axis. This is ordinary PBT's x{0.8, 1.0, 1.25} with the user's own 1.2, and it
# replaced a +/-1 step along `HP_LADDERS` -- see the note above HP_LADDERS for
# why an unbounded multiplier suits that arm and a bounded ladder suits the
# other. Keeping 1.0 in the set is what lets an axis stay put: with only the two
# moving multipliers every axis of every inherited worker would move every
# generation, and with seven axes no lineage would ever hold a setting still.
#
# **These numbers are not symmetric in log space, and the drift is measurable.**
# 1.2 is not 1/0.8, so the geometric mean of the set is (0.96)^(1/3) = 0.9865:
# every axis of an inherited lineage shrinks by ~1.35% per generation with
# nothing opposing it but selection. Simulated over 400 free lineages with no
# selection at all, 60 generations from lr 3.0e-4 put the median at 1.27e-4 --
# exactly the 0.9865^60 = 0.44 the geometric mean predicts -- with a spread from
# 2.9e-6 to 1.2e-2. Canonical PBT uses x1.25 for precisely this reason: 1.25 is
# 1/0.8, so (0.8, 1.0, 1.25) has a geometric mean of exactly 1 and the walk is
# unbiased. Keeping 1.2 is the user's choice; the unbiased counterparts, if the
# drift is ever unwanted, are (0.8, 1.0, 1.25) or (1/1.2, 1.0, 1.2).
HP_MULTIPLIERS: tuple[float, ...] = (0.8, 1.0, 1.2)

# The axes that are a probability. Their product is capped at 1.0 -- a
# probability above 1 is not one, and both of these reach code that would
# silently do something odd with it (`opponent_probability` is compared against
# `rng.random()`, and `pool_top_share` decides how many of `--pool-models` seats
# come from the top band). There is deliberately no lower cap, here or anywhere:
# multiplying a positive value by 0.8 can never reach 0.
_HP_PROBABILITY_AXES = frozenset({"opponent_probability", "pool_top_share"})

# Axes perturbed through their complement, `1 - value`. For the GAE lambda the
# quantity that matters is `1 - lambda`: the effective horizon is
# `1 / (1 - lambda)`, so x0.8 / x1.2 on lambda itself would jump 0.95 to 0.76
# (horizon 20 -> 4) or past 1.0, while on the complement they give 0.96 / 0.94
# and can never reach 1. Stored, passed and reported as lambda everywhere else.
_HP_COMPLEMENT_AXES = frozenset({"gae_lambda"})
_COMPLEMENT_FLOOR = 1e-3

# Which arm produced a worker's settings, recorded in its published model as
# `hp_arm`. Still recorded, and still not cosmetic: an inherited point
# correlates with its parent's quality by construction -- the parent was drawn
# from the top 100 -- so it cannot be read as a response curve, while an
# independent draw can. What has changed is that there is no longer a cohort of
# independent draws to compare against: see `hyperparameter_plan`.
#
# `HP_ARM_SAMPLED` is now reached only by a `poker-loop` run driven by hand,
# where `launch_worker` is given no plan at all. Production never produces it.
HP_ARM_SAMPLED = "sampled"
HP_ARM_INHERITED = "inherited"
# **The one arm that is not inheritance, and it is not a choice.** A worker can
# only inherit from a parent whose checkpoint actually carries the metadata; with
# no parent, or a parent that predates the metadata, there is literally nothing
# to perturb, so that worker draws a rung of `HP_LADDERS` instead and is labelled
# here. Since the deliberate sampled arm was removed this is the *whole* of the
# non-inherited population, which makes the count worth watching rather than
# worth ignoring: it says how often inheritance was available at all, and it is
# the only thing left keeping any worker near the ladder's known-good centre.
HP_ARM_FALLBACK = "sampled-fallback"


@dataclass(frozen=True)
class Hyperparameters:
    """One worker's point in the sweep, and which arm chose it."""

    lr: float
    hands: int
    opponent_probability: float
    ppo_epochs: int
    clip_epsilon: float
    # How strong a field this worker draws: what share of its `--pool-models`
    # seats come from the best-rated band, and how deep that band is.
    pool_top_share: float
    pool_top_n: int
    minibatch_size: int = PPOConfig.minibatch_size
    gae_lambda: float = TrainConfig.lam
    value_coef: float = PPOConfig.value_coefficient
    max_grad_norm: float = PPOConfig.max_grad_norm
    entropy_coef: float = PPOConfig.entropy_coefficient
    arm: str = HP_ARM_SAMPLED


def _typed(values: dict[str, float]) -> dict[str, float | int]:
    """One value per axis, cast to the type that axis is measured in.

    The count axes are **rounded, never truncated**: `int()` would take 1.6 down
    to 1 and then 0.8 to 0, and `ppo_epochs` 0 is a run that never updates its
    policy at all. On the sampled path every value is already a ladder rung, so
    the rounding is the identity there; the inherit path has already produced
    integers of its own through `_moved_count`, which has a stronger job to do --
    see its docstring.
    """
    return {
        axis: round(value) if axis in _HP_INT_AXES else float(value)
        for axis, value in values.items()
    }


def _moved_count(before: float, after: float, multiplier: float) -> int:
    """A count axis's perturbed value: **a move has to actually move.**

    Plain rounding does not, and this was a real one-way ratchet rather than an
    edge case. `round(2 * 1.2) = round(2.4) = 2` and `round(2 * 0.8) = 2`, so 2
    was absorbing in *both* directions -- and 2 is the bottom rung of the
    `ppo_epochs` ladder, one step below 3 and two below the fleet's own 4, so
    lineages fell in and could never climb out. Measured on the code before this
    function existed: **62% of lineages sat stuck at `ppo_epochs` 2 after 20
    generations and 96% after 200**, against a median of 4 afterwards. (On
    `hands` and `pool_top_n` the same state exists and is unreachable in
    practice, ~24 consecutive downward draws away.)

    So: round as usual, and only when that would leave the value where it started
    step it one in the multiplier's direction. **Rounding always away from the
    current value would also fix the ratchet and was rejected**: it inflates the
    step wherever rounding was working, turning 6 into 8 or 4 instead of the 7 or
    5 that x1.2 and x0.8 actually ask for -- a 33% step on an axis the user asked
    to move by 20%.

    The floor of 1 is the integer analogue of the fact that multiplying a
    positive value by 0.8 never reaches 0: `ppo_epochs` 0 is not a smaller
    setting, it is the absence of training. 1 is a legitimate value (one PPO
    pass) and it is *reflecting*, not absorbing -- from 1 the up draw gives 2.
    Note this does let a lineage reach 1, which the ladder's bottom rung of 2
    never allowed.
    """
    moved = round(after)
    if multiplier != 1.0 and moved == round(before):
        moved = round(before) + (1 if multiplier > 1.0 else -1)
    return max(1, moved)


def sample_hyperparameters(rng: random.Random) -> Hyperparameters:
    """An independent uniform rung on every axis."""
    drawn = {axis: rng.choice(values) for axis, values in HP_LADDERS.items()}
    return Hyperparameters(**_typed(drawn), arm=HP_ARM_SAMPLED)


def perturb_hyperparameters(
    parent: dict | None, rng: random.Random
) -> Hyperparameters | None:
    """The parent's settings, each axis multiplied by one of `HP_MULTIPLIERS`.

    Returns `None` when the parent carries nothing usable -- no metadata, a
    schema this build does not know, or a missing axis -- so the caller can fall
    back to sampling rather than inventing a lineage that does not exist.

    **The product is unbounded and is never snapped back onto `HP_LADDERS`**, so
    a lineage holds whatever its ancestors' draws multiplied out to and can walk
    clean past the ends of the ladder. That is the point: this arm is a hill
    climb, and a clamp would have the ladder's ends -- which nothing ever
    measured -- decide how far the fleet may go, while the ranking that chooses
    parents is a bound that was measured. See the note above `HP_LADDERS`.

    `HP_LADDERS` is still what says *which* axes exist, so adding an axis there
    is all it takes for both arms to move on it; only its *values* have stopped
    constraining this arm.

    The two limits that remain: a probability is capped at 1.0, and a count is
    floored at 1 by `_moved_count`, which also makes sure a count actually moves
    when the multiplier says it should.
    """
    if not parent or parent.get("schema") != RUN_METADATA_VERSION:
        return None
    moved: dict[str, float] = {}
    for axis in HP_LADDERS:
        inherited = parent.get(axis, _HP_DEFAULTS.get(axis))
        if not isinstance(inherited, (int, float)) or isinstance(inherited, bool):
            return None
        multiplier = rng.choice(HP_MULTIPLIERS)
        if axis in _HP_COMPLEMENT_AXES:
            complement = max(1.0 - float(inherited), _COMPLEMENT_FLOOR)
            moved[axis] = max(0.0, 1.0 - complement * multiplier)
            continue
        value = float(inherited) * multiplier
        if inherited == 0 and multiplier > 1.0 and axis in _HP_SEED_FROM_ZERO:
            value = _HP_SEED_FROM_ZERO[axis]
        if axis in _HP_PROBABILITY_AXES:
            value = min(value, 1.0)
        if axis in _HP_INT_AXES:
            # Direction-aware, because plain rounding leaves a count stuck at 2
            # in both directions -- see `_moved_count`.
            value = _moved_count(float(inherited), value, multiplier)
        moved[axis] = value
    return Hyperparameters(**_typed(moved), arm=HP_ARM_INHERITED)


def hyperparameter_plan(
    workers: int,
    parents: list[dict | None],
    *,
    rng: random.Random,
) -> list[Hyperparameters]:
    """What each worker of a generation trains with: its parent's settings, every
    axis perturbed.

    **Every worker inherits, and the deliberate sampled arm is gone**, removed at
    the user's request together with `--hp-inherit-share`. A worker draws a rung
    of `HP_LADDERS` only when it *cannot* inherit -- no parent, or a parent whose
    checkpoint carries no usable metadata -- and that case is labelled
    `HP_ARM_FALLBACK`, not chosen.

    **What that costs, recorded because it was the reason the split existed.**
    Half the fleet used to be an independent draw, and an independent draw is the
    only thing a response curve can be read off: an inherited point correlates
    with its parent's quality by construction, since the parent was drawn from
    the top 100, so the settings and the selection cannot be separated afterwards.
    With one arm the fleet is a pure search -- it can find a good configuration
    and can no longer say *why* it is good, or tell a good setting from a lucky
    lineage. That is the user's call; the trade is what this paragraph is for.

    **And the second thing it cost, which is easier to miss.** The sampled arm
    was re-drawn from the ladder every generation, so half the fleet sat at the
    known-good centre by construction -- an anchor the search could not drift
    away from. Nothing anchors it now except the ranking that picks parents,
    which matters because `HP_MULTIPLIERS` is not symmetric in log space (see the
    note there): every axis of every lineage shrinks ~1.35% per generation on
    average, and there is no longer a cohort at the centre to pull against it.

    `parents[w]` is the run metadata of worker `w`'s *weight* parent, or `None`.
    The two inheritances are deliberately coupled: perturbing the settings of a
    model whose weights this worker is not starting from would attribute the
    parent's configuration to a run that never had it.
    """
    plan: list[Hyperparameters] = []
    for worker in range(workers):
        inherited = perturb_hyperparameters(parents[worker], rng)
        if inherited is not None:
            plan.append(inherited)
            continue
        plan.append(replace(sample_hyperparameters(rng), arm=HP_ARM_FALLBACK))
    return plan


def effective_hyperparameters(
    hp: Hyperparameters, *, inheriting: bool, fresh_lr: float
) -> Hyperparameters:
    """What a worker will really train with, once `--fresh-lr` has had its say.

    A worker with no parent to resume from trains at `--fresh-lr` whatever the
    sweep drew, because a rate drawn for a warm start means nothing on a cold
    one. That substitution has to happen in *one* place, or the supervisor
    reports the draw while the worker trains at something else -- which is
    exactly what it did: with an empty store the header read
    `lr 0.00019-0.00047` while every worker logged `lr=0.001`. The worker was
    right (its metadata records the rate actually passed), so the summary is
    what had to change.
    """
    if inheriting:
        return hp
    return replace(hp, lr=fresh_lr)


def read_run_metadata(path: Path | None) -> dict | None:
    """The settings a published model was trained with, or `None`.

    Read straight out of the checkpoint the worker is about to resume from, so
    the inherit arm needs no side file and no shared-directory write. Unreadable
    is not an error here -- a model published before this metadata existed is
    the normal case today -- so every failure is one `None` and one fallback to
    sampling.
    """
    if path is None or not Path(path).exists():
        return None
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - a store of ~10,000 files on an NFS mount
        # torch raises whatever the pickle, the zip reader or the mount felt
        # like raising, and every one of them means the same thing here: this
        # worker samples instead. A supervisor must not die reading a parent.
        return None
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    return metadata if isinstance(metadata, dict) and metadata else None


def inheritance_plan(
    ranking: PoolRegistry,
    models_dir: Path,
    workers: int,
    fraction: float,
    *,
    rng: random.Random,
    tiers: Sequence[int | None] = PARENT_TIERS,
) -> list[Path | None]:
    """Which model, if any, each worker starts from.

    Half the workers inheriting and half starting from a random network is a
    deliberate split. Without any inheritance the loop produces an endless
    supply of models that are all exactly `--iterations` deep and never deeper:
    it generates *variety*, not *strength*, and the best model never improves.
    With *every* worker inheriting from the same parent, all of them explore
    around one point and a dead end (a policy collapsed onto folding, say) traps
    the whole population at once. Splitting keeps depth and fresh blood in the
    same loop.

    Inheritors take distinct parents drawn at random from the best-rated models
    on disk (`pick_parents`: top 10/100/1000/all, a quarter each), so the deep half is competing lineages rather than
    copies of one, and a different set every generation.
    """
    inheriting = min(workers, max(0, round(workers * fraction)))
    parents = pick_parents(
        ranking.members, available_labels(models_dir), inheriting, rng=rng, tiers=tiers
    )
    plan: list[Path | None] = [None] * workers
    for index, label in enumerate(parents):
        plan[index] = Path(models_dir) / f"{label}.pt"
    return plan


def launch_worker(
    args: argparse.Namespace,
    worker: int,
    generation: int,
    worker_dir: Path,
    log_path: Path,
    inherit_from: Path | None = None,
    hp: Hyperparameters | None = None,
    fill_stop: Path | None = None,
) -> subprocess.Popen:
    seed = args.seed_base + generation * args.workers + worker
    checkpoint_path = worker_dir.parent / f"agent-{worker_dir.name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)
    # Decided once: the same answer picks the learning rate and whether the
    # worker resumes, so the two cannot drift apart.
    inheriting = inherit_from is not None and inherit_from.exists()
    # No plan means no sweep: the worker runs the fleet's own settings. `main`
    # always builds one, so this is the hand-driven and test path, and it is
    # what keeps `args` the single definition of every default.
    if hp is None:
        hp = Hyperparameters(
            lr=args.lr,
            hands=args.hands,
            opponent_probability=args.opponent_probability,
            ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            pool_top_share=args.pool_top_share,
            pool_top_n=args.pool_top_n,
            minibatch_size=args.minibatch_size,
            gae_lambda=args.gae_lambda,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            entropy_coef=args.entropy_coef,
            arm=HP_ARM_SAMPLED,
        )
    hp = effective_hyperparameters(hp, inheriting=inheriting, fresh_lr=args.fresh_lr)
    command = [
        sys.executable,
        "-u",
        "-m",
        "pokerlab.rl.train",
        "--iterations", str(args.iterations),
        "--hands", str(hp.hands),
        "--players", str(args.players),
        "--stack", str(args.stack),
        "--sb", str(args.sb),
        "--bb", str(args.bb),
        # Already resolved above by `effective_hyperparameters`: an inheritor
        # keeps the swept rate, a worker with no parent takes `--fresh-lr`.
        "--lr", str(hp.lr),
        "--seed", str(seed),
        "--device", args.device,
        "--models-dir", str(args.models_dir),
        "--scratch-dir", str(worker_dir),
        "--archive-prefix", worker_dir.name,
        "--machine", args.machine,
        "--pool-models", str(args.pool_models),
        # Swept: these two decide how strong a field the worker draws. Only
        # `--pool-models`, the field's *size*, is still the fleet's own value.
        "--pool-top-share", str(hp.pool_top_share),
        "--pool-top-n", str(hp.pool_top_n),
        "--opponent-probability", str(hp.opponent_probability),
        "--ppo-epochs", str(hp.ppo_epochs),
        "--clip-epsilon", str(hp.clip_epsilon),
        "--minibatch-size", str(hp.minibatch_size),
        "--gae-lambda", str(hp.gae_lambda),
        "--value-coef", str(hp.value_coef),
        "--max-grad-norm", str(hp.max_grad_norm),
        "--entropy-coef", str(hp.entropy_coef),
        # Recorded by the worker into the model it publishes, never acted on:
        # which arm drew these values is what separates the runs an analysis can
        # read a response curve from.
        "--hp-arm", hp.arm,
        "--eval-every", str(args.eval_every),
        "--eval-sessions", str(args.eval_sessions),
        "--archive-every", str(args.archive_every),
        "--checkpoint", str(checkpoint_path),
        # --benchmark-dir stays: the worker rates its published model against
        # opponents drawn from it (--benchmark-sessions). What is gone is the live
        # per-worker benchmark that used to be forwarded here as
        # --benchmark-every/--benchmark-hands/--benchmark-seed: removed at the
        # user's request, the reading no longer wanted and the hands no longer
        # worth a worker's time. `poker-loop`'s own once-per-generation
        # benchmark (run_generation_benchmark) is a different thing and stays.
        "--benchmark-dir", str(args.benchmark_dir),
        "--global-dir", str(args.global_dir),
        "--global-lock-seconds", str(args.global_lock_seconds),
        "--benchmark-sessions", str(args.benchmark_sessions),
    ]
    if args.elo_fill_in:
        # Only the supervisor can supply the stop file, which is why the phase
        # is off by default in `poker-train`: a hand-run training with nobody to
        # wait for would fill until its deadline.
        command += [
            "--elo-fill-in",
            "--fill-stop-file", str(fill_stop),
            "--fill-min-sessions", str(args.fill_min_sessions),
            "--fill-deadline-minutes", str(args.fill_deadline_minutes),
            "--fill-games-per-model", str(args.fill_games_per_model),
        ]
    if args.global_round:
        command += [
            "--global-round",
            "--global-root", str(args.global_root),
            "--global-sample", str(args.global_sample),
            "--global-benchmark-sample", str(args.global_benchmark_sample),
            "--global-games-per-model", str(args.global_games_per_model),
            "--global-hands-per-game", str(args.global_hands_per_game),
            "--global-trigger-size", str(args.global_trigger_size),
            "--global-eliminate-fraction", str(args.global_eliminate_fraction),
            "--global-protect-percentile", str(args.global_protect_percentile),
        ]
    else:
        command.append("--no-global-round")
    if inheriting:
        # `--resume` reads --checkpoint, so the parent is copied into place
        # first. Only the weights carry over: archives hold no optimizer state,
        # so Adam restarts cold, which costs a short transient and is much
        # cheaper than throwing the weights away too.
        shutil.copy2(inherit_from, checkpoint_path)
        command.append("--resume")
    environment = dict(os.environ)
    # One core per worker: torch's intra-op threads buy nothing on a small MLP
    # and would have N workers fighting over the same cores.
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)


# How often the supervisor looks at its workers while they train. It used to
# block in `process.wait()` and learn nothing until one exited, which is no
# longer possible: a worker in the Elo fill-in phase deliberately does not exit
# until this supervisor tells it to, so blocking on it would deadlock the
# generation. Ten seconds against runs measured in hours costs nothing, and the
# workers themselves only look at the stop file between rounds (~3 minutes), so
# polling faster would buy nothing either.
FILL_POLL_SECONDS = 10.0


def wait_for_workers(
    processes: list[tuple[int, Path, subprocess.Popen]],
    *,
    fill_stop: Path | None = None,
    poll_seconds: float = FILL_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[tuple[int, int]]:
    """Wait for a generation's workers, releasing the ones that are only waiting.

    Returns `(worker, returncode)` for each worker that exited non-zero.

    **Why this is not `process.wait()` any more.** Workers now finish at
    genuinely different times -- `--hands` is drawn per worker -- and the fast
    ones spend the difference playing rating rounds instead of idling. Such a
    worker is waiting for *this* supervisor to tell it the generation is over,
    and this supervisor was waiting for that worker to exit: each would have
    held the other until the worker's own deadline expired hours later.

    The handshake breaks it. A filling worker creates
    `FILL_DRAINING_FILENAME` in its scratch directory, meaning "I am done with
    my own work and only killing time"; when every worker still alive says that,
    the generation has really finished, and the supervisor creates `fill_stop`,
    which every worker checks between rounds. The flag is created once and never
    withdrawn here -- the next generation removes it before launching anything,
    so a stale flag cannot release the next generation's workers the moment they
    start.
    """
    failures: list[tuple[int, int]] = []
    released = False
    remaining = list(processes)
    while remaining:
        still_running = []
        for worker, worker_dir, process in remaining:
            code = process.poll()
            if code is None:
                still_running.append((worker, worker_dir, process))
            elif code != 0:
                failures.append((worker, code))
        remaining = still_running
        if not remaining:
            break
        if (
            fill_stop is not None
            and not released
            and all(
                (worker_dir / FILL_DRAINING_FILENAME).exists()
                for _worker, worker_dir, _process in remaining
            )
        ):
            fill_stop.parent.mkdir(parents=True, exist_ok=True)
            fill_stop.touch()
            released = True
            print(
                f"  generazione completa: {len(remaining)} worker in riempimento elo, "
                "dato il via libera a chiudere",
                flush=True,
            )
        sleep(poll_seconds)
    return failures


# The fleet's machines are 32-core, and `run.sh` caps its own choice at the same
# number (`DEFAULT_WORKER_CEILING`); the two are kept equal on purpose, so
# `poker-loop` run by hand behaves like `./run.sh start` on the same box. Unlike
# run.sh's, this one is *not* reduced by the machine's cores or free memory, so a
# small VM driven by hand should pass `--workers` explicitly.
DEFAULT_WORKERS = 25

# What the `benchmark_arena` run requested by an added anchor is asked to do.
# Values, not flags, on purpose: decided once for the whole fleet. K is not among
# them -- the arena always uses the hyperbolic staircase.
ARENA_MIN_ROUNDS = 50
ARENA_MAX_ROUNDS = 500
# Converged when no anchor has drifted more than this over the drift window.
ARENA_TOLERANCE = 1.0


def run_requested_benchmark_arena(
    args: argparse.Namespace, global_dir: Path, generation: int
) -> bool:
    """Run the `benchmark_arena` an added anchor asked for, if this machine wins it.

    The request is left in the shared global directory by
    `global_arena.add_benchmark_candidates` and claimed here by an atomic rename,
    so exactly one machine in the fleet picks it up.

    **Between two generations, deliberately.** The workers have exited, so the
    whole machine is free and the run gets every core; and because the
    supervisor is the one waiting on it, the job is supervised and logged
    instead of being an unattended process nobody knows about. The cost is
    honest and unavoidable: the next generation starts hours later.

    Returns whether a run happened.
    """
    claimed = claim_benchmark_arena(global_dir, machine=args.machine)
    if claimed is None:
        return False
    claim, request = claimed
    added = len(request.get("added", []))
    log_path = Path(args.log_dir) / f"arena-gen{generation:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", "-m", "pokerlab.rl.benchmark_arena",
        "--root", str(args.global_root),
        "--global-dir", str(global_dir),
        "--machine", args.machine,
        "--workers", str(args.workers),
        "--tolerance", str(ARENA_TOLERANCE),
        "--min-rounds", str(ARENA_MIN_ROUNDS),
        "--max-rounds", str(ARENA_MAX_ROUNDS),
    ]
    print(f"  arena delle ancore per {added} nuovi modelli: "
          f"{ARENA_MIN_ROUNDS}-{ARENA_MAX_ROUNDS} round, "
          f"convergenza {ARENA_TOLERANCE}; log in {log_path}", flush=True)
    started = time.time()
    environment = dict(os.environ)
    # One thread per shard, as everywhere else: the arena shards the play across
    # `--workers` processes and threads would only make them fight.
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        code = subprocess.call(
            command, stdout=handle, stderr=subprocess.STDOUT, env=environment
        )
    finish_benchmark_arena(claim, failed=code != 0)
    elapsed = (time.time() - started) / 60.0
    if code == 0:
        print(f"  arena conclusa in {elapsed:.0f} min", flush=True)
    else:
        print(f"  ARENA FALLITA (codice {code}) dopo {elapsed:.0f} min: vedi {log_path}. "
              f"La richiesta non viene ripetuta da sola.", flush=True)
    return True


def run_generation_benchmark(
    args: argparse.Namespace,
    ranking: PoolRegistry,
    game: GameConfig,
    generation: int = 0,
) -> tuple[float | None, str, float | None, str]:
    """Score two models against the frozen set: the best rated one in the store,
    and the best one *this generation produced*.

    Both are needed, and the second is the one that answers "is the loop
    working". The overall best can be a model that predates the loop, which
    makes the headline benchmark repeat the identical number and say nothing at
    all about the newly trained models.

    Returns `(None, "", None, "")` when no benchmark directory is configured or
    it holds too few opponents -- a missing benchmark must not stop the loop.
    """
    empty = (None, "", None, "")
    benchmark_dir = Path(args.benchmark_dir)
    if args.benchmark_hands <= 0 or not benchmark_dir.is_dir():
        return empty

    from pokerlab.rl.benchmark import load_benchmark_opponents, run_benchmark

    opponents = load_benchmark_opponents(
        benchmark_dir,
        game,
        device=args.device,
        on_skip=lambda path, why: print(f"  benchmark, saltato {path.name}: {why}", flush=True),
    )
    if len(opponents) < game.num_players - 1:
        print(f"  benchmark saltato: solo {len(opponents)} avversari in {benchmark_dir}", flush=True)
        return empty

    on_disk = set(available_labels(args.models_dir))
    models = [
        m for m in ranking.ranked() if m.label in on_disk and not m.frozen and m.games > 0
    ]
    prefix = f"{args.machine}-gen{generation:04d}-"
    fresh_models = [
        m for m in ranking.ranked() if m.label in on_disk and m.label.startswith(prefix)
    ]
    if not models and not fresh_models:
        return empty

    def score(member) -> float | None:
        try:
            model, _checkpoint = build_model_from_checkpoint(
                Path(args.models_dir) / f"{member.label}.pt", device=args.device
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the loop
            print(f"  benchmark saltato, {member.label} illeggibile: {exc}", flush=True)
            return None
        return run_benchmark(
            model,
            opponents,
            game,
            hands=args.benchmark_hands,
            seed=args.benchmark_seed,
            device=args.device,
        ).bb_per_100

    best = models[0] if models else fresh_models[0]
    fresh = fresh_models[0] if fresh_models else None
    # Re-scoring the same file twice would be pure waste: the benchmark is
    # deterministic, so an identical model returns an identical number.
    best_score = score(best)
    fresh_score = None
    fresh_label = ""
    if fresh is not None:
        fresh_label = fresh.label
        fresh_score = best_score if fresh.label == best.label else score(fresh)
    return best_score, best.label, fresh_score, fresh_label


def format_leaderboard(registry: PoolRegistry, limit: int = 15) -> str:
    rows = [f"{'#':>3}  {'modello':<52}{'rating':>8}{'partite':>9}"]
    ranked = [member for member in registry.ranked() if not member.frozen]
    for position, member in enumerate(ranked[:limit], start=1):
        rows.append(f"{position:>3}  {member.label:<52}{member.rating:8.0f}{member.games:9d}")
    return "\n".join(rows)


def render_trend(marks: list[float], width: int = 40) -> str:
    """A bar per generation, scaled to the range actually observed.

    Absolute bb/100 values mean little on their own here; what the eye needs is
    whether the line goes up.
    """
    if len(marks) < 2:
        return ""
    low, high = min(marks), max(marks)
    span = high - low or 1.0
    blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    bars = "".join(blocks[min(int((m - low) / span * len(blocks)), len(blocks) - 1)] for m in marks[-width:])
    return f"  {low:+.0f} {bars} {high:+.0f} bb/100"


def print_status(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    state = LoopState.load(state_dir / STATE_FILENAME)
    ranking = load_ranking(args.global_dir)

    print(f"stato         : {state_dir}")
    print(f"avviato       : {state.started or 'mai'}")
    print(f"generazione   : {state.generation}   supervisor: {state.phase}   worker: {state.workers}")
    rated = sum(1 for m in ranking.members.values() if not m.frozen and m.games > 0)
    print(f"modelli       : {len(available_labels(args.models_dir))} in {args.models_dir}, "
          f"{rated} con almeno una partita valutata")
    benchmark_models = len(list(Path(args.benchmark_dir).rglob("*.pt")))
    print(f"benchmark     : {benchmark_models} avversari fissi in {args.benchmark_dir}")

    rows = worker_progress(Path(args.log_dir), state.generation)
    if rows:
        print(f"\nworker generazione {state.generation}")
        for line in format_worker_table(rows, args.iterations):
            print(line)

    # No leaderboard here on purpose: --status/--watch is about what this
    # machine's workers are doing right now, and fifteen rows of ratings pushed
    # that off the screen. The ranking is still printed once per generation into
    # the supervisor log (`run_loop`), which is where a snapshot of it belongs.
    # No per-generation table and no benchmark trends here: --status/--watch is
    # for what this machine's workers are doing right now, and both were asked
    # to go. The numbers are still recorded in `loop_state.json` history and
    # printed into the supervisor log as each generation ends.


def run_loop(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    global_dir = Path(args.global_dir)
    state_path = state_dir / STATE_FILENAME
    stop_path = state_dir / STOP_FILENAME
    # The supervisor's half of the fill-in handshake. On this machine's own
    # disk, never the shared volume: it releases *this* supervisor's workers,
    # and a shared one would have the first machine to finish a generation stop
    # every other machine's workers too.
    fill_stop = state_dir / FILL_STOP_FILENAME

    state = LoopState.load(state_path)
    state.started = time.strftime("%Y-%m-%d %H:%M:%S")
    state.central_pool = str(models_dir)
    state.workers = args.workers
    state.save(state_path)

    game = GameConfig(
        num_players=args.players, starting_stack=args.stack, small_blind=args.sb, big_blind=args.bb
    )
    plan_rng = random.Random()
    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        print("\narresto richiesto: termino la generazione in corso e mi fermo.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def sweep(label: str) -> None:
        if args.keep_work:
            return
        swept = sweep_stale_work(work_root, models_dir, global_dir, machine=args.machine)
        if swept.total or swept.in_use:
            print(
                f"  {label}: {swept.work_dirs} directory di worker, "
                f"{swept.checkpoints} checkpoint, {swept.arena_dirs} directory arena, "
                f"{swept.partials + swept.scratch_dirs} file/dir temporanei "
                f"({swept.freed_bytes / 1e6:.0f} MB liberati), "
                f"{swept.salvaged} archivi non pubblicati recuperati"
                + (f"; {swept.in_use} ancora in uso, lasciati" if swept.in_use else ""),
                flush=True,
            )

    while not stopping:
        if args.generations and state.generation >= args.generations:
            print(f"raggiunte {args.generations} generazioni, fine.")
            break
        if stop_path.exists():
            print(f"trovato {stop_path}, fine.")
            break

        state.generation += 1
        generation = state.generation
        record = GenerationRecord(
            generation=generation, started=time.strftime("%H:%M:%S"), workers=args.workers
        )
        print(f"\n=== generazione {generation} | {args.workers} worker "
              f"x {args.iterations} iterazioni ===", flush=True)

        sweep("pulizia residui")

        state.phase = "starting"
        state.save(state_path)
        # Before anything is launched: a flag left by the previous generation
        # would release this one's workers the instant they reached the phase.
        fill_stop.unlink(missing_ok=True)
        # The authoritative member files, not the (minutes-old) snapshot: this
        # picks parents, once per generation, so it can afford the read.
        ranking = load_global_registry(global_dir)
        plan = inheritance_plan(
            ranking,
            models_dir,
            args.workers,
            args.inherit_fraction,
            rng=plan_rng,
        )
        # Read once per generation, off the very checkpoints the workers are
        # about to resume from: at most `--workers` files, and only for the
        # workers that have a parent at all.
        hp_plan = hyperparameter_plan(
            args.workers,
            [read_run_metadata(parent) for parent in plan],
            rng=plan_rng,
        )
        # Resolved here, exactly as `launch_worker` resolves it again (the
        # substitution is idempotent), so the header below and the workers' own
        # logs cannot report different learning rates.
        hp_plan = [
            effective_hyperparameters(
                hp,
                inheriting=plan[worker] is not None and plan[worker].exists(),
                fresh_lr=args.fresh_lr,
            )
            for worker, hp in enumerate(hp_plan)
        ]
        processes = []
        for worker in range(args.workers):
            worker_dir = work_root / f"gen{generation:04d}-w{worker:02d}"
            log_path = Path(args.log_dir) / f"gen{generation:04d}-w{worker:02d}.log"
            processes.append(
                (
                    worker,
                    worker_dir,
                    launch_worker(
                        args, worker, generation, worker_dir, log_path,
                        plan[worker], hp_plan[worker], fill_stop,
                    ),
                )
            )
        inheriting = sum(1 for parent in plan if parent is not None)
        print(f"  {inheriting}/{args.workers} ereditano i pesi, "
              f"{args.workers - inheriting} partono da zero (lr {args.fresh_lr:g}); "
              f"ognuno pesca il proprio pool da {len(available_labels(models_dir))} modelli",
              flush=True)
        arms = [hp.arm for hp in hp_plan]
        # The fallback count is the one worth printing, and now more than before:
        # it is the *whole* of the non-inherited population, so it says how often
        # inheritance was available at all. An arm silently empty is exactly what
        # this line exists to make impossible to miss.
        print(f"  iperparametri: {arms.count(HP_ARM_INHERITED)} ereditati e perturbati"
              + (f", {arms.count(HP_ARM_FALLBACK)} campionati per mancanza di "
                 f"metadati nel genitore" if HP_ARM_FALLBACK in arms else "")
              + f"; lr {min(hp.lr for hp in hp_plan):g}-{max(hp.lr for hp in hp_plan):g}, "
              f"mani {min(hp.hands for hp in hp_plan)}-{max(hp.hands for hp in hp_plan)}",
              flush=True)

        state.phase = "training"
        state.save(state_path)
        for worker, code in wait_for_workers(
            processes, fill_stop=fill_stop if args.elo_fill_in else None
        ):
            record.failed += 1
            print(f"  worker {worker} uscito con codice {code}", flush=True)

        # Each worker publishes its own best model as it exits; this catches the
        # ones that died first, and removes the scratch either way.
        state.phase = "publishing"
        state.save(state_path)
        sweep("recupero e pulizia")
        prefix = f"{args.machine}-gen{generation:04d}-w"
        record.published_models = sum(1 for label in available_labels(models_dir) if label.startswith(prefix))
        print(f"  pubblicati {record.published_models} nuovi modelli", flush=True)

        state.phase = "benchmark"
        state.save(state_path)
        ranking = load_global_registry(global_dir)
        (
            record.benchmark_bb100,
            record.benchmark_model,
            record.benchmark_new_bb100,
            record.benchmark_new_model,
        ) = run_generation_benchmark(args, ranking, game, generation)
        if record.benchmark_bb100 is not None:
            print(f"  benchmark, migliore assoluto ({record.benchmark_model}): "
                  f"{record.benchmark_bb100:+.1f} bb/100", flush=True)
        if record.benchmark_new_bb100 is not None:
            print(f"  benchmark, migliore di questa generazione "
                  f"({record.benchmark_new_model}): "
                  f"{record.benchmark_new_bb100:+.1f} bb/100", flush=True)

        ranked = [m for m in ranking.ranked() if not m.frozen]
        best = ranked[0] if ranked else None
        record.best_label = best.label if best else ""
        record.best_rating = best.rating if best else 0.0
        record.models_total = len(available_labels(models_dir))
        record.finished = time.strftime("%H:%M:%S")
        state.history.append(asdict(record))
        state.phase = "idle"
        state.save(state_path)

        print(f"  {record.models_total} modelli nello store", flush=True)
        print(format_leaderboard(ranking, limit=args.top), flush=True)

        # Last, with the machine to itself: a model added to the benchmark
        # anywhere in the fleet leaves a request for the anchors to be settled,
        # and this is the one moment a supervisor has no workers running.
        state.phase = "benchmark_arena"
        state.save(state_path)
        if run_requested_benchmark_arena(args, global_dir, generation):
            state.phase = "idle"
            state.save(state_path)

    state.phase = "stopped"
    state.save(state_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous self-play training: N worker processes, one shared store of models."
    )
    parser.add_argument("--status", action="store_true", help="print the leaderboard and exit")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="concurrent training processes",
    )
    parser.add_argument(
        "--generations", type=int, default=0, help="0 runs until stopped (Ctrl-C or a STOP file)"
    )
    parser.add_argument("--iterations", type=int, default=100, help="iterations per worker run")
    parser.add_argument("--hands", type=int, default=512)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="learning rate of the workers that inherit weights")
    parser.add_argument(
        "--fresh-lr", type=float, default=1e-3,
        help="learning rate of the workers that start from a random network; "
             "higher than --lr because they have further to travel (default 1e-3)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--state-dir", type=Path, default=DEFAULT_STATE_DIR,
        help="this machine's own loop state (loop_state.json) and STOP file",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
        help="the shared store of every trained model",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("checkpoints/work"))
    parser.add_argument("--log-dir", type=Path, default=Path("checkpoints/logs/loop"))
    parser.add_argument("--pool-models", type=int, default=DEFAULT_POOL_SIZE,
                        help="opponents each worker draws for its own run")
    parser.add_argument("--pool-top-share", type=float, default=DEFAULT_TOP_SHARE)
    parser.add_argument("--pool-top-n", type=int, default=DEFAULT_TOP_N)
    # These exist here only as the value a worker gets when no plan is drawn,
    # which is the hand-driven path; what production launches with comes from
    # `hyperparameter_plan`. (`--hp-inherit-share` used to sit here, deciding how
    # much of a generation drew its own settings instead of inheriting them. It
    # is gone with the sampled arm: every worker inherits now, and the only ones
    # that do not are the ones that cannot.)
    parser.add_argument("--ppo-epochs", type=int, default=PPOConfig.epochs)
    parser.add_argument("--clip-epsilon", type=float, default=PPOConfig.clip_epsilon)
    parser.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    parser.add_argument("--gae-lambda", type=float, default=TrainConfig.lam)
    parser.add_argument("--value-coef", type=float, default=PPOConfig.value_coefficient)
    parser.add_argument("--max-grad-norm", type=float, default=PPOConfig.max_grad_norm)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coefficient)
    parser.add_argument(
        "--opponent-probability", type=float, default=0.5,
        help="chance that a non-learner seat gets an opponent instead of another "
        "copy of the learner; see poker-train's help for why it matters",
    )
    # Fewer, far bigger evaluations. At 500 hands split into 25-hand blocks the
    # `eval/100` column carried a 95% interval of +/-255 bb/100 and the `rating`
    # was twenty coin flips -- neither could distinguish two models. 10,000 hands
    # in blocks of 1000 take those to +/-57 bb/100 and a signal twice the noise;
    # running it every 250 iterations instead of every 25 keeps the bill at ~13
    # minutes of a 170-minute run. Same trade as the rated sessions themselves:
    # bigger measurements, not more of them.
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--eval-sessions", type=int, default=DEFAULT_EVAL_SESSIONS,
        help="rated sessions per validation round of a worker, each 1000 hands. "
        "At --eval-every 100 over a 1000-iteration run that is 100 rated "
        "sessions, which continue into the round against the frozen anchors",
    )
    parser.add_argument("--archive-every", type=int, default=50)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="frozen opponents, never seated in training; the only cross-generation metric",
    )
    parser.add_argument(
        "--benchmark-hands", type=int, default=DEFAULT_BENCHMARK_HANDS, help="0 disables"
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=12345,
        help="held fixed so every generation is dealt the identical hands",
    )
    parser.add_argument(
        "--inherit-fraction",
        type=float,
        default=1.0,
        help="share of workers that resume from a top-rated model instead of a "
        "random network (0 = every run starts from scratch)",
    )
    parser.add_argument(
        "--machine",
        default=socket.gethostname().split(".")[0],
        help="identifier prefixed to published models, so hosts cannot collide",
    )
    parser.add_argument(
        "--global-round",
        dest="global_round",
        action="store_true",
        default=True,
        help="have every worker try one cross-population Elo round "
        "(rl/global_arena.py) at the end of its run; on by default",
    )
    parser.add_argument(
        "--no-global-round",
        dest="global_round",
        action="store_false",
        help="disable the end-of-run population round for every worker",
    )
    # On by default here, and off by default in `poker-train`: the phase only
    # makes sense when there are other workers to wait for, and only a
    # supervisor knows when they are done.
    parser.add_argument(
        "--elo-fill-in",
        dest="elo_fill_in",
        action="store_true",
        default=True,
        help="have a worker that finished early keep playing rating rounds "
        "until the rest of its generation catches up; on by default",
    )
    parser.add_argument(
        "--no-elo-fill-in",
        dest="elo_fill_in",
        action="store_false",
        help="let a worker that finished early exit and leave its core idle",
    )
    parser.add_argument(
        "--fill-min-sessions", type=int, default=DEFAULT_FILL_MIN_SESSIONS,
        help="sessions each filling worker plays before the stop file can end "
        "its phase, so a generation still gets a rating phase when every worker "
        "finishes together",
    )
    parser.add_argument(
        "--fill-deadline-minutes", type=float, default=DEFAULT_FILL_DEADLINE_MINUTES,
        help="hard cap on one worker's fill-in phase, in minutes; the only cap "
        "there is, and what stops a worker whose supervisor died from filling "
        "forever",
    )
    parser.add_argument(
        "--fill-games-per-model", type=int, default=DEFAULT_FILL_GAMES_PER_MODEL,
        help="rated sessions each drawn model owes per fill-in round",
    )
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--global-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-sample", type=int, default=DEFAULT_POPULATION_SAMPLE)
    parser.add_argument("--global-benchmark-sample", type=int, default=DEFAULT_BENCHMARK_SAMPLE)
    parser.add_argument(
        "--global-games-per-model", type=int, default=DEFAULT_GAMES_PER_MODEL,
        help="rated sessions each drawn model owes per population round; the "
        "round's cost is very nearly linear in it (~1.1 s per session)",
    )
    parser.add_argument("--global-hands-per-game", type=int, default=DEFAULT_HANDS_PER_GAME)
    parser.add_argument("--global-lock-seconds", type=int, default=DEFAULT_LOCK_SECONDS)
    parser.add_argument("--global-trigger-size", type=int, default=DEFAULT_POPULATION_TRIGGER)
    parser.add_argument(
        "--global-eliminate-fraction", type=float, default=DEFAULT_ELIMINATION_FRACTION
    )
    parser.add_argument(
        "--global-protect-percentile", type=float, default=DEFAULT_PROTECT_PERCENTILE
    )
    parser.add_argument(
        "--benchmark-sessions", type=int, default=DEFAULT_BENCHMARK_SESSIONS,
        help="rated sessions of 1000 hands each worker's published model plays "
        "against opponents drawn at random from the frozen set, before the "
        "population round (0 disables). This is what the model is published "
        "with: its rating and its session count both come out of it",
    )
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--top", type=int, default=15, help="rows shown in the leaderboard")
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="with --status, redraw every N seconds instead of printing once",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="keep each worker's scratch directory and skip the residue sweep",
    )
    args = parser.parse_args()

    if args.status:
        if args.watch <= 0:
            print_status(args)
            return
        try:
            while True:
                # Redraw in place rather than scrolling, so a long watch does
                # not bury the terminal in repeated snapshots.
                print("\033[H\033[J", end="")
                print(f"poker-loop --status   {time.strftime('%H:%M:%S')}   "
                      f"aggiorno ogni {args.watch}s, Ctrl-C per uscire\n")
                print_status(args)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return
        return
    run_loop(args)


if __name__ == "__main__":
    main()
