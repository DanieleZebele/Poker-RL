"""A continuous training loop: N worker processes, one shared store of models.

Processes, not threads. The engine is pure Python and CPU-bound, so N threads
would serialise on the GIL and run at the speed of one; N processes genuinely
use N cores (measured: 30 concurrent runs on a 32-core box, ~57 hands/s each).

The loop runs in generations. Each generation launches N `poker-train` workers.
Every worker draws *its own* opponents from the shared store
(`checkpoints/models/`, see `rl/training_pool.py`), trains, publishes its final
model back into that store, and plays a cross-population rating pass
(`rl/global_arena.py`) -- so nothing is seeded, merged or re-ranked here. The
ratings live in the global registry (`rl/global_store.py`), one file per model,
and every machine reads and writes it directly.

What the loop still owns is the choreography around that: which workers inherit
weights (and from whom), the once-per-generation benchmark against the frozen
set, sweeping up after runs that were killed, and the status display.
"""

from __future__ import annotations

import argparse
import math
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
from dataclasses import asdict, dataclass
from pathlib import Path

from pokerlab.config import (
    ConfigError,
    ConfigReport,
    add_config_arguments,
    diff_settings,
    format_config,
    parse_with_config,
)
from pokerlab.rl.allin_reward import DEFAULT_ALLIN_RUNOUTS
from pokerlab.rl.benchmark import (
    DEFAULT_ANCHOR_ROTATE_EVERY,
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_BENCHMARK_SESSIONS,
    DEFAULT_RESIDENT_ANCHORS,
)
from pokerlab.rl.global_arena import (
    BENCHMARK_GAMES_PERCENTILE,
    BENCHMARK_MARGIN,
    DEFAULT_BENCHMARK_SAMPLE,
    DEFAULT_GLOBAL_DIR,
    DEFAULT_GLOBAL_SESSIONS,
    DEFAULT_POPULATION_SAMPLE,
    DRAW_TIERS,
    backfill_member_styles,
    draw_tiers_text,
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
    DEFAULT_K_SCHEDULE,
    DEFAULT_POOL_SIZE,
    DEFAULT_POPULATION_TRIGGER,
    DEFAULT_PROTECT_PERCENTILE,
    PoolRegistry,
    format_k_schedule,
    k_schedule_text,
)
from pokerlab.rl.ppo import PPOConfig
from pokerlab.rl.rollout import DEFAULT_CONCURRENT_TABLES, DEFAULT_TABLE_HANDS

# The one thing the loop needs from `train.py`, which it otherwise only ever
# launches as a subprocess: how many rated sessions a validation pass plays,
# so the default the loop forwards and the default a standalone `poker-train`
# uses cannot drift apart. No cycle -- train.py does not import loop.py -- and torch is already
# here through `rl/benchmark.py`.
from pokerlab.rl.siblings import sibling_parsers
from pokerlab.rl.styles import add_style_arguments, style_arguments, style_config_from_args
from pokerlab.rl.sweep_log import read_observations
from pokerlab.rl.sweep_optimizer import (
    DEFAULT_EXPLORE,
    DEFAULT_MIN_GAIN,
    DEFAULT_WARMUP,
    DEFAULT_WINDOW,
    SweepPolicy,
    build_policy,
    report_lines,
)
from pokerlab.rl.table_mix import add_table_arguments, table_arguments, table_mix_from_args
from pokerlab.rl.train import (
    DEFAULT_EVAL_SESSIONS,
    DEFAULT_FILL_DEADLINE_MINUTES,
    DEFAULT_FILL_SESSIONS,
    RUN_METADATA_VERSION,
    TrainConfig,
    add_network_arguments,
    check_network_arguments,
    network_shape,
)
from pokerlab.rl.training_pool import (
    DEFAULT_TOP_N,
    DEFAULT_TOP_SHARE,
    PARENT_TIERS,
    available_labels,
    format_parent_tiers,
    parent_tiers_text,
    parse_parent_tiers,
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
    (its final `agent-*.pt` and the sidecar recording its rating), so every
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
        sidecar = read_sidecar(path)
        try:
            publish_model(
                path,
                models_dir=models_dir,
                global_dir=global_dir,
                name=name,
                rating=sidecar.rating,
                style=sidecar.style,
                style_hands=sidecar.style_hands,
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
    scratch left in the temp directory by a killed global pass.
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


# --- hyperparameters, one point per worker ------------------------------------
#
# Every worker of a generation trains with its own settings, so a generation is a
# sweep rather than N copies of one configuration. Each worker's point is
# recorded in the model it publishes (`train.run_metadata`), so an outcome can
# afterwards be attributed to the settings that produced it.
#
# **Every point is a perturbation of another point.** A worker that inherits
# weights multiplies its parent's value on each axis by one of `HP_MULTIPLIERS`;
# a worker with no parent to inherit from does the same to the *starting point*,
# which is what the fleet's own flags (and so `config.toml`) say. The result is
# not snapped onto any grid, so a lineage can hold values no one chose and can
# walk arbitrarily far. Two reasons that is the better fit:
#   * **A clamp would make a guess load-bearing.** Whoever wrote the limits would
#     decide in advance the furthest the fleet could ever go.
#   * **The real bound is selection.** A parent is drawn from the `--pool-top-n`
#     best-rated models, so a lineage that walks its lr up to something that
#     breaks training rates badly and stops being a parent. An arithmetic clamp
#     guesses where the edge is; the ranking finds out.
# The cost, accepted: a lineage really can drift a long way. The step is a
# random walk in log space, so over N generations its spread is about
# ln(1.2)*sqrt(2N/3) -- roughly 27x either way over 500 generations. Nothing
# stops that but the ranking.
#
# **Only two limits survive, and neither is a rung.** A probability axis is
# capped at 1.0 (see `_HP_PROBABILITY_AXES`), because a probability above 1 is
# not one; and the count axes are *rounded*, never truncated, which is what
# keeps them off zero -- see `_typed`. Nothing has a lower cap, and none is
# needed: multiplying a positive number by 0.8 never reaches 0.
#
# An integer multiplied drifts off any grid an analysis could group by, so
# points are a continuum. That costs nothing that was not already lost, because
# an inherited point can never be read as a response curve (it correlates with
# its parent's quality by construction).
#
# The axes that exist. Adding a name here is all it takes for a worker to move
# along it, provided `Hyperparameters`, `--<axis>` and `train.REPORTED_AXES`
# know it too (a test fails otherwise). `--pool-models` is deliberately absent:
# it is the *size* of the field, and what a run's cost scales with, so varying
# it would confound a strength axis with a cost axis. `pool_top_share` and
# `pool_top_n` are how strong a field the worker draws: the share decides *how
# much* of it is elite and `n` *how* elite that part is.
HP_AXES = (
    "lr",
    "hands",
    "opponent_probability",
    "ppo_epochs",
    "clip_epsilon",
    "minibatch_size",
    "gae_lambda",
    "value_coef",
    "policy_max_grad_norm",
    "critic_max_grad_norm",
    "entropy_coef",
    "pool_top_share",
    "pool_top_n",
)

# The axes whose values are counts, so a value is cast back to `int`. Everything
# else is a float. This exists so `HP_AXES` is the only place an axis has to
# be added: both arms build a `Hyperparameters` from the dict generically, and
# the type is the one thing the axis list itself cannot say (0.0 and 0 look alike).
_HP_INT_AXES = frozenset({"hands", "ppo_epochs", "pool_top_n", "minibatch_size"})

# Zero is absorbing under a multiplier, and 0.0 is where this axis lives, so an
# upward draw from exactly zero lands on this value instead of staying put.
# Downward draws from zero stay zero; from a positive value they decay as usual.
_HP_SEED_FROM_ZERO: dict[str, float] = {"entropy_coef": 1e-3}

# What a worker multiplies its starting value by (its parent's, or the fleet's
# own when it has none), one independent draw per axis. This is the *default*:
# `--hp-multipliers` (and so `config.toml`) replaces it. It is ordinary PBT's
# x{0.8, 1.0, 1.25} with 1.2 -- see the note above for why an unbounded
# multiplier suits this search. Keeping 1.0 in the set is what lets an axis stay
# put: with only the two moving multipliers every axis of every worker would move
# every generation, and with twelve axes no lineage would ever hold a setting
# still.
#
# **These numbers are not symmetric in log space, and the drift is measurable.**
# 1.2 is not 1/0.8, so the geometric mean of the set is (0.96)^(1/3) = 0.9865:
# every axis of an inherited lineage shrinks by ~1.35% per generation with
# nothing opposing it but selection: over 60 generations the median lineage
# lands at 0.9865^60 = 0.44 of where it started. Canonical PBT uses x1.25 for
# precisely this reason: 1.25 is 1/0.8, so (0.8, 1.0, 1.25) has a geometric mean
# of exactly 1 and the walk is unbiased. The unbiased counterparts, if the drift
# is ever unwanted, are (0.8, 1.0, 1.25) or (1/1.2, 1.0, 1.2).
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
# `hp_arm`. Not cosmetic: an inherited point correlates with its parent's
# quality by construction -- the parent was drawn from the top 100 -- so it
# cannot be read as a response curve, while a point perturbed from the fleet's own
# starting values can. See `hyperparameter_plan`.
#
# `HP_ARM_SAMPLED` is reached only by a `poker-loop` run driven by hand, where
# `launch_worker` is given no plan at all: the fleet's settings, unperturbed.
# Production never produces it.
HP_ARM_SAMPLED = "sampled"
HP_ARM_INHERITED = "inherited"
# **The one arm that is not inheritance, and it is not a choice.** A worker can
# only inherit from a parent whose checkpoint actually carries the metadata; with
# no parent, or a parent without the metadata, there is nothing of a parent's to
# perturb, so that worker perturbs the fleet's own starting values (the flags, so
# `config.toml`) and is labelled here. The name predates that -- it used to draw
# a rung of a ladder -- and stays because it is recorded in published models.
# This is the *whole* of the non-inherited population, which makes the count
# worth watching: it says how often inheritance was available at all.
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
    policy_max_grad_norm: float = PPOConfig.policy_max_grad_norm
    critic_max_grad_norm: float = PPOConfig.critic_max_grad_norm
    entropy_coef: float = PPOConfig.entropy_coefficient
    arm: str = HP_ARM_SAMPLED


def _typed(values: dict[str, float]) -> dict[str, float | int]:
    """One value per axis, cast to the type that axis is measured in.

    The count axes are **rounded, never truncated**: `int()` would take 1.6 down
    to 1 and then 0.8 to 0, and `ppo_epochs` 0 is a run that never updates its
    policy at all. `_moved_count` has already produced integers of its own for the
    perturbed counts and has a stronger job to do -- see its docstring.
    """
    return {
        axis: round(value) if axis in _HP_INT_AXES else float(value)
        for axis, value in values.items()
    }


def _moved_count(before: float, after: float, multiplier: float) -> int:
    """A count axis's perturbed value: **a move has to actually move.**

    Plain rounding does not, and the failure is a one-way ratchet rather than an
    edge case. `round(2 * 1.2) = round(2.4) = 2` and `round(2 * 0.8) = 2`, so 2
    is absorbing in *both* directions -- and 2 is a perfectly ordinary
    `ppo_epochs`, so lineages fall in and can never climb out. (On
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
    Note this does let a lineage reach 1.
    """
    moved = round(after)
    if multiplier != 1.0 and moved == round(before):
        moved = round(before) + (1 if multiplier > 1.0 else -1)
    return max(1, moved)


def starting_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    """The fleet's own settings, as the flags and `config.toml` resolved them.

    Where a worker with no parent starts, and what a worker launched without a
    plan runs unchanged. `args` stays the single definition of every default.
    """
    return Hyperparameters(
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
        policy_max_grad_norm=args.policy_max_grad_norm,
        critic_max_grad_norm=args.critic_max_grad_norm,
        entropy_coef=args.entropy_coef,
        arm=HP_ARM_SAMPLED,
    )


def _perturbed(
    start: dict,
    rng: random.Random,
    *,
    arm: str,
    multipliers: Sequence[float],
    policy: SweepPolicy | None = None,
) -> Hyperparameters:
    """`start` with each axis multiplied by one of `multipliers`.

    **The product is unbounded and is never snapped back onto anything**, so a
    lineage holds whatever its ancestors' draws multiplied out to and can walk
    arbitrarily far. That is the point: this is a hill climb, and a clamp would
    have limits nobody measured decide how far the fleet may go, while the ranking
    that chooses parents is the bound. See the note above `HP_AXES`.

    The two limits that remain: a probability is capped at 1.0, and a count is
    floored at 1 by `_moved_count`, which also makes sure a count actually moves
    when the multiplier says it should.

    **Which multiplier an axis takes is `policy`'s call when there is one**
    (`rl/sweep_optimizer.py`: tilted toward the steps that gained Elo per unit of
    compute), and a uniform draw from `multipliers` when there is not -- the draw
    the sweep always made, from the same rng in the same order.
    """
    moved: dict[str, float] = {}
    chosen = policy.draw(rng) if policy is not None else None
    for axis in HP_AXES:
        before = start[axis]
        multiplier = chosen[axis] if chosen is not None else rng.choice(multipliers)
        if axis in _HP_COMPLEMENT_AXES:
            complement = max(1.0 - float(before), _COMPLEMENT_FLOOR)
            moved[axis] = max(0.0, 1.0 - complement * multiplier)
            continue
        value = float(before) * multiplier
        if before == 0 and multiplier > 1.0 and axis in _HP_SEED_FROM_ZERO:
            value = _HP_SEED_FROM_ZERO[axis]
        if axis in _HP_PROBABILITY_AXES:
            value = min(value, 1.0)
        if axis in _HP_INT_AXES:
            # Direction-aware, because plain rounding leaves a count stuck at 2
            # in both directions -- see `_moved_count`.
            value = _moved_count(float(before), value, multiplier)
        moved[axis] = value
    return Hyperparameters(**_typed(moved), arm=arm)


def perturb_hyperparameters(
    parent: dict | None,
    rng: random.Random,
    multipliers: Sequence[float] = HP_MULTIPLIERS,
    policy: SweepPolicy | None = None,
) -> Hyperparameters | None:
    """The parent's settings, each axis multiplied by one of `multipliers`.

    Returns `None` when the parent carries nothing usable -- no metadata, a
    schema this build does not know, or a missing axis -- so the caller can start
    from the fleet's own settings rather than invent a lineage that does not
    exist.
    """
    if not parent or parent.get("schema") != RUN_METADATA_VERSION:
        return None
    for axis in HP_AXES:
        inherited = parent.get(axis)
        if not isinstance(inherited, (int, float)) or isinstance(inherited, bool):
            return None
    return _perturbed(
        parent, rng, arm=HP_ARM_INHERITED, multipliers=multipliers, policy=policy
    )


def hyperparameter_plan(
    workers: int,
    parents: list[dict | None],
    start: Hyperparameters,
    *,
    rng: random.Random,
    multipliers: Sequence[float] = HP_MULTIPLIERS,
    policy: SweepPolicy | None = None,
    fallback_arm: str = HP_ARM_FALLBACK,
) -> list[Hyperparameters]:
    """What each worker of a generation trains with: its parent's settings, every
    axis perturbed -- or `start`'s, for a worker with no parent.

    **Every worker perturbs something.** A worker with a parent whose checkpoint
    carries usable metadata perturbs that; any other perturbs `start` (the fleet's
    own flags, so `config.toml`) and is labelled `HP_ARM_FALLBACK`, not chosen.
    That is how the first generation after an empty store is set: write the values
    wanted in the file and every worker starts a step or two around them.

    **What that costs.** An independent point is the only thing a response curve
    can be read off: an inherited point correlates with its parent's quality by
    construction, since the parent was drawn from the top 100, so the settings and
    the selection cannot be separated afterwards. The fleet is a pure search -- it
    can find a good configuration and cannot say *why* it is good, or tell a good
    setting from a lucky lineage.

    **And nothing anchors the search** except the ranking that picks parents,
    which matters because `HP_MULTIPLIERS` is not symmetric in log space (see the
    note there): every axis of every lineage shrinks ~1.35% per generation on
    average, with no cohort at the centre to pull against it.

    `parents[w]` is the run metadata of worker `w`'s *weight* parent, or `None`.
    The two inheritances are deliberately coupled: perturbing the settings of a
    model whose weights this worker is not starting from would attribute the
    parent's configuration to a run that never had it.

    With a `policy` every worker -- the fallback ones too -- takes its multipliers
    from it instead of drawing them uniformly; each worker's call to `policy.draw`
    is its own posterior sample, which is what keeps the fleet trying both
    directions where the evidence is thin.

    `fallback_arm` labels the workers that perturb `start` instead of a parent's
    settings. `run_loop` passes `HP_ARM_SAMPLED` with every parent `None` when
    `--no-inherit-hyperparameters` is set: then *every* worker is one independent
    step from the fleet's own values, which is what that arm has always meant.
    """
    start_values = {axis: getattr(start, axis) for axis in HP_AXES}
    plan: list[Hyperparameters] = []
    for worker in range(workers):
        inherited = perturb_hyperparameters(parents[worker], rng, multipliers, policy)
        if inherited is not None:
            plan.append(inherited)
            continue
        plan.append(
            _perturbed(
                start_values, rng, arm=fallback_arm, multipliers=multipliers, policy=policy
            )
        )
    return plan


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
    *,
    rng: random.Random,
    tiers: Sequence[int | None] = PARENT_TIERS,
) -> list[Path | None]:
    """Which model each worker starts from: every worker inherits.

    Without inheritance the loop only ever produces models exactly
    `--iterations` deep: it generates *variety*, not *strength*, and a
    from-scratch worker trains worse and is far likelier to blow past the `kl`
    threshold early. The variety a fresh start would have supplied comes from the
    hyperparameter sweep instead. A worker gets `None` only when the store cannot
    supply a parent at all (an empty store, the very first run ever, or no model
    rated yet).

    Parents are distinct and drawn at random from the best-rated models on disk
    (`pick_parents`: top 10/100/1000/all, a quarter each), so the population is
    competing lineages rather than copies of one, and a different set every
    generation.
    """
    parents = pick_parents(
        ranking.members, available_labels(models_dir), workers, rng=rng, tiers=tiers
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
    inheriting = inherit_from is not None and inherit_from.exists()
    # No plan means no sweep: the worker runs the fleet's own settings. `main`
    # always builds one, so this is the hand-driven and test path.
    if hp is None:
        hp = starting_hyperparameters(args)
    command = [
        sys.executable,
        "-u",
        "-m",
        "pokerlab.rl.train",
        "--iterations", str(args.iterations),
        "--hands", str(hp.hands),
        *table_arguments(table_mix_from_args(args)),
        "--lr", str(hp.lr),
        # The fleet's network shape, as `config.toml` resolved it when this
        # generation started.
        *[
            token
            for key, value in network_shape(args).items()
            for token in (f"--{key.replace('_', '-')}", str(value))
        ],
        "--equity-model", args.equity_model,
        "--seed", str(seed),
        "--device", args.device,
        "--models-dir", str(args.models_dir),
        "--scratch-dir", str(worker_dir),
        "--archive-prefix", worker_dir.name,
        "--machine", args.machine,
        "--pool-models", str(args.pool_models),
        "--table-hands", str(args.table_hands),
        "--concurrent-tables", str(args.concurrent_tables),
        "--allin-runouts", str(args.allin_runouts),
        *style_arguments(style_config_from_args(args)),
        "--critic-stack-power", str(args.critic_stack_power),
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
        "--policy-max-grad-norm", str(hp.policy_max_grad_norm),
        "--critic-max-grad-norm", str(hp.critic_max_grad_norm),
        "--entropy-coef", str(hp.entropy_coef),
        # Recorded by the worker into the model it publishes, never acted on:
        # which arm drew these values is what separates the runs an analysis can
        # read a response curve from.
        "--hp-arm", hp.arm,
        "--eval-every", str(args.eval_every),
        "--eval-sessions", str(args.eval_sessions),
        "--checkpoint", str(checkpoint_path),
        # The worker rates its published model against opponents drawn from the
        # frozen set (--benchmark-sessions): the only benchmark there is.
        "--benchmark-dir", str(args.benchmark_dir),
        "--global-dir", str(args.global_dir),
        "--global-lock-seconds", str(args.global_lock_seconds),
        "--benchmark-sessions", str(args.benchmark_sessions),
        "--benchmark-resident", str(args.benchmark_resident),
        "--benchmark-rotate-every", str(args.benchmark_rotate_every),
        "--k-schedule", args.k_schedule,
        "--draw-tiers", args.draw_tiers,
        # Every value is already on this command line, resolved by the supervisor
        # when the generation started. A worker reading the file again would only
        # add a way to die: a file saved halfway between the two reads.
        "--config", "",
    ]
    if args.elo_fill_in:
        # Only the supervisor can supply the stop file, which is why the phase
        # is off by default in `poker-train`: a hand-run training with nobody to
        # wait for would fill until its deadline.
        command += [
            "--elo-fill-in",
            "--fill-stop-file", str(fill_stop),
            "--fill-deadline-minutes", str(args.fill_deadline_minutes),
            "--fill-sessions", str(args.fill_sessions),
        ]
    if args.global_elo:
        command += [
            "--global-elo",
            "--global-root", str(args.global_root),
            "--global-sample", str(args.global_sample),
            "--global-benchmark-sample", str(args.global_benchmark_sample),
            "--global-sessions", str(args.global_sessions),
            "--session-hands", str(args.session_hands),
            "--global-trigger-size", str(args.global_trigger_size),
            "--global-eliminate-fraction", str(args.global_eliminate_fraction),
            "--global-protect-percentile", str(args.global_protect_percentile),
            "--benchmark-games-percentile", str(args.benchmark_games_percentile),
            "--benchmark-margin", str(args.benchmark_margin),
        ]
    else:
        command.append("--no-global-elo")
    if inheriting:
        # `--resume` reads --checkpoint, so the parent is copied into place
        # first. Only the weights carry over: archives hold no optimizer state,
        # so Adam restarts cold, which costs a short transient and is much
        # cheaper than throwing the weights away too.
        shutil.copy2(inherit_from, checkpoint_path)
        # The worker records which model it came from, so the finished child can
        # say how much Elo it gained over it (`rl/sweep_log.py`).
        command.append("--resume")
        # Only a worker whose settings are its parent's (perturbed) took a step
        # from it. One that runs the fleet's own values took no step, and a
        # "step" of toml-minus-parent would confound the optimizer's evidence with
        # nothing but how far the parent's lineage had drifted.
        if hp.arm == HP_ARM_INHERITED:
            command += ["--parent-label", inherit_from.stem]
    environment = dict(os.environ)
    # One core per worker: torch's intra-op threads buy nothing on a small MLP
    # and would have N workers fighting over the same cores.
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)


# How often the supervisor looks at its workers while they train. It polls
# instead of blocking in `process.wait()`: a worker in the Elo fill-in phase
# deliberately does not exit until this supervisor tells it to, so blocking on
# it would deadlock the generation. Ten seconds against runs measured in hours costs nothing, and the
# workers themselves only look at the stop file between passes (~3 minutes), so
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

    **Why this is not `process.wait()`.** Workers finish at
    genuinely different times -- `--hands` is drawn per worker -- and the fast
    ones spend the difference playing rating passes instead of idling. Such a
    worker is waiting for *this* supervisor to tell it the generation is over,
    and this supervisor was waiting for that worker to exit: each would have
    held the other until the worker's own deadline expired hours later.

    The handshake breaks it. A filling worker creates
    `FILL_DRAINING_FILENAME` in its scratch directory, meaning "I am done with
    my own work and only killing time"; when every worker still alive says that,
    the generation has really finished, and the supervisor creates `fill_stop`,
    which every worker checks between passes. The flag is created once and never
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


# The most workers a machine runs, set fleet-wide in `config.toml` as
# `worker_ceiling`. It is a *ceiling*, not a count: a small VM is held lower by
# its own cores and free memory (`auto_workers`), which is why this can safely be
# one number for every host while `--workers` itself stays machine-local.
#
# Twenty-five, not "every core": filling a 32-core box (nproc - 2 = 30) is what
# the fleet ran before, and five of seven machines went unresponsive under it;
# the cause was never established, but a machine at its own ceiling has no
# headroom for the end-of-run phases, where every worker of a generation arrives
# at the same moment and each loads ~55 models it did not hold while training.
# Watch this one: if machines start going unresponsive, this and
# `--global-sessions` are the two dials.
DEFAULT_WORKERS = 25
# Each worker is its own Python+torch process at roughly this much once its pool
# is loaded, and the system keeps `RESERVED_MEMORY_MB` for itself.
WORKER_MEMORY_MB = 700
RESERVED_MEMORY_MB = 3000


def available_memory_mb() -> int | None:
    """`MemAvailable` in MB, or None where `/proc/meminfo` is not there."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def auto_workers(
    ceiling: int, *, cores: int | None = None, free_mb: int | None = None
) -> int:
    """How many workers this machine takes: `ceiling`, or less where its cores
    (all but two) or its free memory (after the system's share) say so; never
    below 1. Where the memory cannot be read only the cores and the ceiling count.
    """
    cores = os.cpu_count() or 1 if cores is None else cores
    free_mb = available_memory_mb() if free_mb is None else free_mb
    chosen = min(ceiling, cores - 2)
    if free_mb is not None:
        chosen = min(chosen, (free_mb - RESERVED_MEMORY_MB) // WORKER_MEMORY_MB)
    return max(1, chosen)


def resolve_workers(args: argparse.Namespace) -> argparse.Namespace:
    """`args` with `--workers 0` (auto) turned into a number. Done again after
    every config reload, so editing `worker_ceiling` takes effect at the next
    generation."""
    if args.workers > 0:
        return args
    return replace_namespace(args, workers=auto_workers(args.worker_ceiling))


def replace_namespace(args: argparse.Namespace, **changes) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), **changes})

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
    ]
    # What it is asked to do -- total sessions, session length -- is not decided
    # here: the arena reads it from `config.toml` like everything else, from the
    # very file this supervisor was given. K is not among them: the arena always
    # uses the hyperbolic staircase.
    if args.config is not None:
        command += ["--config", args.config]
    print(f"  arena delle ancore per {added} nuovi modelli; log in {log_path}", flush=True)
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


def format_leaderboard(registry: PoolRegistry, limit: int = 15) -> str:
    rows = [f"{'#':>3}  {'modello':<52}{'rating':>8}{'partite':>9}"]
    ranked = [member for member in registry.ranked() if not member.frozen]
    for position, member in enumerate(ranked[:limit], start=1):
        rows.append(f"{position:>3}  {member.label:<52}{member.rating:8.0f}{member.games:9d}")
    return "\n".join(rows)


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


def build_sweep_policy(args: argparse.Namespace, global_dir: Path) -> SweepPolicy | None:
    """The optimizer's policy for this generation, or None when it must not steer.

    Read once per generation from what the finished children recorded
    (`sweep_log`), and printed, because an optimizer whose estimates cannot be seen
    is one nobody can distrust. **With `--no-sweep-optimizer` it still reads, fits
    and prints** -- the report is how one decides whether to turn it on -- and only
    withholds the policy, so the workers draw uniformly as before. **It never stops
    a generation**: whatever goes wrong reading or fitting is reported and the
    workers draw uniformly.
    """
    try:
        policy = build_policy(
            read_observations(global_dir, args.sweep_window),
            HP_AXES,
            _HP_COMPLEMENT_AXES,
            args.hp_multipliers,
            explore=args.sweep_explore,
            warmup=args.sweep_warmup,
            min_gain=args.sweep_min_gain,
        )
    except Exception as error:  # noqa: BLE001 - evidence off a shared volume
        print(f"  sweep: ottimizzatore saltato per errore: {error}", flush=True)
        return None
    for line in report_lines(policy, steering=args.sweep_optimizer):
        print(line, flush=True)
    return policy if args.sweep_optimizer else None


def run_loop(
    args: argparse.Namespace,
    reload: Callable[[], tuple[argparse.Namespace, ConfigReport]] | None = None,
) -> None:
    """Run generations until stopped. With `reload`, `config.toml` is re-read at
    the start of every generation (see `refresh_args`)."""
    args = resolve_workers(args)
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

    styles_tried: set[str] = set()

    def backfill_styles() -> None:
        # A member published before the ranking carried a style gets the one its own
        # checkpoint recorded. Best effort: the passes fill the rest as they seat it.
        try:
            filled = backfill_member_styles(
                global_dir,
                models_dir.parent,
                machine=args.machine,
                read_metadata=read_run_metadata,
                skip=styles_tried,
            )
        except Exception as error:  # noqa: BLE001 - a supervisor must not die on bookkeeping
            print(f"  stili dei modelli: saltato per errore ({error})", flush=True)
            return
        if filled:
            print(f"  stili dei modelli: {filled} aggiunti dai checkpoint", flush=True)

    while not stopping:
        if args.generations and state.generation >= args.generations:
            print(f"raggiunte {args.generations} generazioni, fine.")
            break
        if stop_path.exists():
            print(f"trovato {stop_path}, fine.")
            break

        if reload is not None:
            args = refresh_args(args, reload)
        state.generation += 1
        generation = state.generation
        record = GenerationRecord(
            generation=generation, started=time.strftime("%H:%M:%S"), workers=args.workers
        )
        print(f"\n=== generazione {generation} | {args.workers} worker "
              f"x {args.iterations} iterazioni ===", flush=True)

        sweep("pulizia residui")
        backfill_styles()

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
            rng=plan_rng,
            tiers=parse_parent_tiers(args.parent_tiers),
        )
        # Read once per generation, off the very checkpoints the workers are
        # about to resume from: at most `--workers` files, and only for the
        # workers that have a parent at all.
        sweep_policy = build_sweep_policy(args, global_dir)
        if args.inherit_hyperparameters:
            parent_settings = [read_run_metadata(parent) for parent in plan]
            fallback_arm = HP_ARM_FALLBACK
        else:
            # The weights are inherited, the settings are not: every worker is one
            # independent step from the fleet's own values (`config.toml`), whoever
            # its parent was, and is labelled `sampled`.
            parent_settings = [None] * args.workers
            fallback_arm = HP_ARM_SAMPLED
        hp_plan = hyperparameter_plan(
            args.workers,
            parent_settings,
            starting_hyperparameters(args),
            rng=plan_rng,
            multipliers=args.hp_multipliers,
            policy=sweep_policy,
            fallback_arm=fallback_arm,
        )
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
              f"{args.workers - inheriting} partono da zero; "
              f"ognuno pesca il proprio pool da {len(available_labels(models_dir))} modelli",
              flush=True)
        arms = [hp.arm for hp in hp_plan]
        spread = (
            f"lr {min(hp.lr for hp in hp_plan):g}-{max(hp.lr for hp in hp_plan):g}, "
            f"mani {min(hp.hands for hp in hp_plan)}-{max(hp.hands for hp in hp_plan)}"
        )
        if not args.inherit_hyperparameters:
            print(f"  iperparametri: non ereditati, ogni worker perturba i valori di "
                  f"config.toml; {spread}", flush=True)
        else:
            # The fallback count is the one worth printing:
            # it is the *whole* of the non-inherited population, so it says how often
            # inheritance was available at all. An arm silently empty is exactly what
            # this line exists to make impossible to miss.
            print(f"  iperparametri: {arms.count(HP_ARM_INHERITED)} ereditati e perturbati"
                  + (f", {arms.count(HP_ARM_FALLBACK)} dai valori di partenza (nessun "
                     f"genitore con metadati)" if HP_ARM_FALLBACK in arms else "")
                  + f"; {spread}", flush=True)

        state.phase = "training"
        state.save(state_path)
        for worker, code in wait_for_workers(
            processes, fill_stop=fill_stop if args.elo_fill_in else None
        ):
            record.failed += 1
            print(f"  worker {worker} uscito con codice {code}", flush=True)

        # Each worker publishes its own model as it exits; this catches the
        # ones that died first, and removes the scratch either way.
        state.phase = "publishing"
        state.save(state_path)
        sweep("recupero e pulizia")
        prefix = f"{args.machine}-gen{generation:04d}-w"
        # A model rated well enough in its own run may already have been moved from the
        # store into the benchmark, so both places are counted.
        in_benchmark = (p.stem for p in Path(args.benchmark_dir).rglob("*.pt"))
        record.published_models = sum(
            1 for label in (*available_labels(models_dir), *in_benchmark) if label.startswith(prefix)
        )
        print(f"  pubblicati {record.published_models} nuovi modelli", flush=True)

        ranking = load_global_registry(global_dir)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous self-play training: N worker processes, one shared store of models."
    )
    parser.add_argument("--status", action="store_true", help="print the leaderboard and exit")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="concurrent training processes; 0 = as many as this machine holds, "
             "up to --worker-ceiling (what run.sh passes)",
    )
    parser.add_argument(
        "--worker-ceiling", type=int, default=DEFAULT_WORKERS,
        help="with --workers 0, the most workers any machine runs; a small one "
             "still gets fewer, held by its cores (all but two) and free memory",
    )
    parser.add_argument(
        "--generations", type=int, default=0, help="0 runs until stopped (Ctrl-C or a STOP file)"
    )
    parser.add_argument("--iterations", type=int, default=1000, help="iterations per worker run")
    parser.add_argument("--hands", type=int, default=512)
    add_table_arguments(parser)
    add_network_arguments(parser)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="starting learning rate of the workers with no parent "
                        "to inherit settings from; every worker then perturbs it")
    parser.add_argument(
        "--hp-multipliers", type=float, nargs="+", default=list(HP_MULTIPLIERS),
        help="the factors a worker draws from, independently per axis, to move its "
        "starting settings (its parent's, or the flags' when it has no parent); "
        "repeat a value to make it likelier, keep 1.0 in the list so an axis can "
        "stay put",
    )
    parser.add_argument(
        "--inherit-hyperparameters", dest="inherit_hyperparameters", action="store_true",
        default=True,
        help="a worker starts from its parent's settings and perturbs them; on by default",
    )
    parser.add_argument(
        "--no-inherit-hyperparameters", dest="inherit_hyperparameters", action="store_false",
        help="a worker does not start from its parent's settings: it perturbs the "
        "values of config.toml (or the flags) instead, as one with no parent does. "
        "The weights are still inherited",
    )
    parser.add_argument(
        "--sweep-optimizer", dest="sweep_optimizer", action="store_true", default=True,
        help="choose the multipliers from what the finished children gained in Elo "
        "per unit of compute, instead of drawing them uniformly; on by default, and "
        "uniform until --sweep-warmup children have been seen. Off, the estimate is "
        "still fitted and printed each generation, it just does not steer",
    )
    parser.add_argument(
        "--no-sweep-optimizer", dest="sweep_optimizer", action="store_false",
        help="draw every multiplier uniformly, as the sweep did before the optimizer "
        "(the estimate is still printed)",
    )
    parser.add_argument(
        "--sweep-explore", type=float, default=DEFAULT_EXPLORE,
        help="probability that an axis is drawn uniformly whatever the optimizer "
        "thinks: an axis every worker moves the same way stops varying, and its "
        "effect stops being measurable",
    )
    parser.add_argument(
        "--sweep-warmup", type=int, default=DEFAULT_WARMUP,
        help="children that must have been observed before the optimizer steers",
    )
    parser.add_argument(
        "--sweep-window", type=int, default=DEFAULT_WINDOW,
        help="how many of the most recent children the optimizer reads; older ones "
        "describe a fleet that has moved on",
    )
    parser.add_argument(
        "--sweep-min-gain", type=float, default=DEFAULT_MIN_GAIN,
        help="Elo a unit of relative cost is worth, at least: the objective is gain "
        "per compute, and with no average gain left it would otherwise reward cost",
    )
    parser.add_argument(
        "--device", default="auto", help="cpu, cuda or auto (the default): the GPU if there is one"
    )
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
    # `hyperparameter_plan`.
    parser.add_argument("--ppo-epochs", type=int, default=PPOConfig.epochs)
    parser.add_argument("--clip-epsilon", type=float, default=PPOConfig.clip_epsilon)
    parser.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    parser.add_argument("--gae-lambda", type=float, default=TrainConfig.lam)
    parser.add_argument("--value-coef", type=float, default=PPOConfig.value_coefficient)
    parser.add_argument("--policy-max-grad-norm", type=float, default=PPOConfig.policy_max_grad_norm)
    parser.add_argument("--critic-max-grad-norm", type=float, default=PPOConfig.critic_max_grad_norm)
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
    parser.add_argument(
        "--table-hands", type=int, default=DEFAULT_TABLE_HANDS,
        help="hands a training table keeps the same players (the opponent statistics "
        "fill in over them); forwarded to every worker",
    )
    parser.add_argument(
        "--concurrent-tables", type=int, default=DEFAULT_CONCURRENT_TABLES,
        help="training tables played in turn; forwarded to every worker",
    )
    add_style_arguments(parser)
    parser.add_argument(
        "--allin-runouts", type=int, default=DEFAULT_ALLIN_RUNOUTS,
        help="boards a hand closed before the river is averaged over for its training "
        "reward (0: the chips that moved); forwarded to every worker",
    )
    parser.add_argument(
        "--critic-stack-power", type=float, default=PPOConfig.critic_stack_power,
        help="the critic's loss weighs a decision by its stake in big blinds to this "
        "power, negated (0: off); forwarded to every worker",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--eval-sessions", type=int, default=DEFAULT_EVAL_SESSIONS,
        help="rated sessions per validation pass of a worker, each 1000 hands. "
        "At --eval-every 100 over a 1000-iteration run that is 100 rated "
        "sessions, which continue into the pass against the frozen anchors",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="frozen opponents, never seated in training; each worker rates its model against them",
    )
    parser.add_argument(
        "--machine",
        default=socket.gethostname().split(".")[0],
        help="identifier prefixed to published models, so hosts cannot collide",
    )
    parser.add_argument(
        "--global-elo",
        dest="global_elo",
        action="store_true",
        default=True,
        help="have every worker try one cross-population Elo pass "
        "(rl/global_arena.py) at the end of its run; on by default",
    )
    parser.add_argument(
        "--no-global-elo",
        dest="global_elo",
        action="store_false",
        help="disable the end-of-run population pass for every worker",
    )
    # On by default here, and off by default in `poker-train`: the phase only
    # makes sense when there are other workers to wait for, and only a
    # supervisor knows when they are done.
    parser.add_argument(
        "--elo-fill-in",
        dest="elo_fill_in",
        action="store_true",
        default=True,
        help="have a worker that finished early keep playing rating passes "
        "until the rest of its generation catches up; on by default",
    )
    parser.add_argument(
        "--no-elo-fill-in",
        dest="elo_fill_in",
        action="store_false",
        help="let a worker that finished early exit and leave its core idle",
    )
    parser.add_argument(
        "--fill-deadline-minutes", type=float, default=DEFAULT_FILL_DEADLINE_MINUTES,
        help="hard cap on one worker's fill-in phase, in minutes; the only cap "
        "there is, and what stops a worker whose supervisor died from filling "
        "forever",
    )
    parser.add_argument(
        "--fill-sessions", type=int, default=DEFAULT_FILL_SESSIONS,
        help="rated sessions of one fill-in pass, models drawn at random",
    )
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--global-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-sample", type=int, default=DEFAULT_POPULATION_SAMPLE)
    parser.add_argument("--global-benchmark-sample", type=int, default=DEFAULT_BENCHMARK_SAMPLE)
    parser.add_argument(
        "--draw-tiers", type=draw_tiers_text, default=format_parent_tiers(DRAW_TIERS),
        help="who the population passes (end of run and fill-in) seat: the cutoffs "
             "of the rating ranking, each with an equal share of the seats (a tier is "
             "the best N rated models, or 'all'); a tier listed twice gets twice the "
             "share; forwarded to every worker",
    )
    parser.add_argument(
        "--global-sessions", type=int, default=DEFAULT_GLOBAL_SESSIONS,
        help="rated sessions of the end-of-run population pass, models drawn at "
        "random; the pass's cost is very nearly linear in it (~1.1 s per session)",
    )
    parser.add_argument("--global-lock-seconds", type=int, default=DEFAULT_LOCK_SECONDS)
    parser.add_argument(
        "--k-schedule", type=k_schedule_text, default=format_k_schedule(DEFAULT_K_SCHEDULE),
        help="the Elo K staircase as games:K pairs, e.g. '0:16, 20:11, 45:7.4': the K a "
             "model is rated at once it has played that many sessions; forwarded to every worker",
    )
    parser.add_argument(
        "--parent-tiers", type=parent_tiers_text, default=format_parent_tiers(PARENT_TIERS),
        help="where a worker's parent comes from: each worker picks one of these bands "
             "with equal probability (a band is the best N rated models, or 'all'), then "
             "a model uniformly inside it; a band listed twice is drawn twice as often",
    )
    parser.add_argument("--global-trigger-size", type=int, default=DEFAULT_POPULATION_TRIGGER)
    parser.add_argument(
        "--global-eliminate-fraction", type=float, default=DEFAULT_ELIMINATION_FRACTION
    )
    parser.add_argument(
        "--global-protect-percentile", type=float, default=DEFAULT_PROTECT_PERCENTILE
    )
    parser.add_argument(
        "--benchmark-games-percentile", type=float, default=BENCHMARK_GAMES_PERCENTILE,
        help="a model becomes a frozen anchor only if its games are above this "
        "percentile of the population's (a rating built on few games is not evidence)",
    )
    parser.add_argument(
        "--benchmark-margin", type=float, default=BENCHMARK_MARGIN,
        help="a model becomes a frozen anchor only if its rating is more than this "
        "many points above the best anchor; candidates added together must also "
        "clear each other by more than this",
    )
    parser.add_argument(
        "--benchmark-sessions", type=int, default=DEFAULT_BENCHMARK_SESSIONS,
        help="rated sessions of 1000 hands each worker's published model plays "
        "against opponents drawn at random from the frozen set, before the "
        "population pass (0 disables). This is what the model is published "
        "with: its rating and its session count both come out of it",
    )
    parser.add_argument(
        "--benchmark-resident", type=int, default=DEFAULT_RESIDENT_ANCHORS,
        help="frozen anchors each worker holds in memory at once during that "
        "round (a random slice, dropped for the next)",
    )
    parser.add_argument(
        "--benchmark-rotate-every", type=int, default=DEFAULT_ANCHOR_ROTATE_EVERY,
        help="rated sessions played against one slice of anchors before a new "
        "slice is drawn",
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
    add_config_arguments(parser)
    return parser


def _sibling_parsers() -> list[argparse.ArgumentParser]:
    """Every other CLI that reads `config.toml`: a key meant for one is not a typo here."""
    return sibling_parsers("pokerlab.rl.loop", with_torch=True)


def load_args(argv: Sequence[str]) -> tuple[argparse.Namespace, ConfigReport]:
    """Flags and `config.toml` resolved into one namespace (flags win)."""
    args, report = parse_with_config(build_parser(), argv, siblings=_sibling_parsers)
    # Here and not in argparse's `type=`, so a bad list in the file is a
    # `ConfigError` like any other: a warning on a reload, an exit at startup.
    if not all(math.isfinite(m) and m > 0 for m in args.hp_multipliers):
        raise ConfigError("hp_multipliers must be positive numbers")
    try:
        check_network_arguments(args)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if not 0.0 <= args.sweep_explore <= 1.0:
        raise ConfigError("sweep_explore must be between 0 and 1")
    if args.sweep_warmup < 0 or args.sweep_window < 1:
        raise ConfigError("sweep_warmup must be >= 0 and sweep_window >= 1")
    if not math.isfinite(args.sweep_min_gain) or args.sweep_min_gain < 0:
        raise ConfigError("sweep_min_gain must be a non-negative number")
    return args, report


def refresh_args(
    args: argparse.Namespace,
    reload: Callable[[], tuple[argparse.Namespace, ConfigReport]],
) -> argparse.Namespace:
    """`args` as the file reads *now*, or `args` unchanged if it cannot be read.

    Called once per generation, which is what lets the fleet be retuned without
    restarting a supervisor that lives for weeks. A file saved halfway or with a
    typo must never reach a running loop, so any problem is a warning and the
    previous values carry on.
    """
    try:
        fresh, _report = reload()
    except ConfigError as error:
        print(f"  config: {error}; tengo i valori precedenti", flush=True)
        return args
    fresh = resolve_workers(fresh)
    changes = diff_settings(vars(args), vars(fresh))
    if changes:
        print("  config riletto, cambiato: " + "; ".join(changes), flush=True)
    return fresh


def main() -> None:
    argv = sys.argv[1:]
    parser = build_parser()
    try:
        args, report = load_args(argv)
    except ConfigError as error:
        parser.exit(2, f"{parser.prog}: config: {error}\n")
    if args.print_config:
        print(format_config(vars(args), [k for k, v in report.applied.items() if vars(args)[k] == v]))
        return
    if report.path is not None and not args.status:
        print(f"config: {report.path}, {len(report.applied)} valori", flush=True)

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
    run_loop(args, reload=lambda: load_args(argv))


if __name__ == "__main__":
    main()
