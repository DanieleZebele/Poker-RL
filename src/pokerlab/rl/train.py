"""Self-play PPO training loop, and the `poker-train` entry point.

Requires the `rl` extra (`pip install -e ".[rl]"`).
"""

from __future__ import annotations

import argparse
import math
import random
import socket
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pokerlab.config import (
    ConfigError,
    add_config_arguments,
    format_config,
    parse_with_config,
    resolved_settings,
)
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.allin_reward import DEFAULT_ALLIN_RUNOUTS
from pokerlab.rl.benchmark import (
    DEFAULT_ANCHOR_ROTATE_EVERY,
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_BENCHMARK_SESSIONS,
    DEFAULT_RESIDENT_ANCHORS,
)
from pokerlab.rl.device import resolve_device
from pokerlab.rl.equity_net import (
    encoder_config,
    equity_net_from_checkpoint,
    load_encoder_weights,
    load_equity_checkpoint,
)
from pokerlab.rl.global_arena import (
    BENCHMARK_GAMES_PERCENTILE,
    BENCHMARK_MARGIN,
    DEFAULT_BENCHMARK_SAMPLE,
    DEFAULT_GLOBAL_DIR,
    DEFAULT_GLOBAL_SESSIONS,
    DEFAULT_POPULATION_SAMPLE,
    DRAW_TIERS,
    draw_tiers_text,
)
from pokerlab.rl.global_store import (
    DEFAULT_LOCK_SECONDS,
    DEFAULT_MODELS_DIR,
    load_ranking,
    publish_model,
    write_sidecar,
)
from pokerlab.rl.grad_log import ClipReading, GradientReading, format_gradient_line
from pokerlab.rl.phases import (
    DONE,
    ELO_FILL,
    ELO_PLAY,
    EVALUATION,
    FILL_DRAINING_FILENAME,
    SERIES,
    hyperparameters_marker,
    marker,
    progress_marker,
)
from pokerlab.rl.policy import (
    DEFAULT_HEAD_HIDDEN,
    DEFAULT_HEAD_LAYERS,
    DEFAULT_HIDDEN,
    DEFAULT_NUM_LAYERS,
    PokerActorCritic,
    make_critic_fns,
    make_policy_fn,
)
from pokerlab.rl.pool_registry import (
    DEFAULT_ELIMINATION_FRACTION,
    DEFAULT_K_SCHEDULE,
    DEFAULT_POOL_SIZE,
    DEFAULT_POPULATION_TRIGGER,
    DEFAULT_PROTECT_PERCENTILE,
    DEFAULT_RATING,
    PoolMember,
    PoolRegistry,
    format_k_schedule,
    k_for_games,
    k_schedule_text,
    parse_k_schedule,
)
from pokerlab.rl.ppo import (
    PPOConfig,
    build_batch,
    build_model_from_checkpoint,
    check_compatible,
    checkpoint_shape,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from pokerlab.rl.rollout import (
    DEFAULT_CONCURRENT_TABLES,
    DEFAULT_TABLE_HANDS,
    CriticFns,
    Opponent,
    OpponentPool,
    SelfPlayCollector,
    TableBank,
    policy_opponent,
)
from pokerlab.rl.siblings import sibling_parsers
from pokerlab.rl.style_log import format_style_line, merge_style
from pokerlab.rl.styles import StyleConfig, add_style_arguments, style_config_from_args
from pokerlab.rl.sweep_log import SweepObservation, write_observation
from pokerlab.rl.table_mix import (
    DEFAULT_SESSION_HANDS,
    TableMix,
    add_table_arguments,
    sizes_text,
    table_mix_from_args,
)
from pokerlab.rl.training_pool import (
    DEFAULT_TOP_N,
    DEFAULT_TOP_SHARE,
    available_labels,
    build_training_registry,
    draw_training_pool,
    format_parent_tiers,
    parse_parent_tiers,
)
from pokerlab.rl.value_diagnostics import (
    ValueDiagnostics,
    format_value_diagnostics,
    value_diagnostics,
)

# Where a run keeps its final model until it is published to the shared store.
DEFAULT_SCRATCH_DIR = Path("checkpoints/scratch")

# The live learner is a session participant but never a pool member: it changes
# every iteration, so persisting it would rate a moving target. Its rating is
# carried on the trainer and written beside each checkpoint it saves.
LEARNER_LABEL = "learner"

# Hands per rated block inside `evaluate_against_pool`, i.e. how long one of its
# sessions is. Elo reads only the *sign* of a block's chip delta, so a short
# session is close to a coin flip: the stronger side finishes ahead barely more
# often than not, and the rating equilibrates far closer to the pool than the
# strength deserves. Session length is the only lever that moves that
# equilibrium; a lower K shrinks the jitter around it and cannot move it.
# One rated session is 1,000 hands, everywhere: the in-run validation, the pass
# against the frozen anchors and the population passes all use the same length,
# because the Elo scale itself is defined by how often a session of that length
# picks the stronger model. A pass of a different length would be a different
# scale silently sharing the same numbers.
# A pass is `--eval-sessions` sessions, with opponents re-drawn from the pool
# before each one.
DEFAULT_EVAL_SESSIONS = 10


def announce(stage: str) -> None:
    """Write the `phase: <stage>` line `poker-loop --status` looks for.

    Its own line, flushed at once: the watcher reads this log while the run is
    still going, and a stage that lasts minutes (the population pass plays
    thousands of hands without printing anything) would otherwise show as
    whatever the last ordinary line said.
    """
    print(marker(stage), flush=True)


# One progress line at most this often. The two stages that report take ~50 and
# ~90 minutes, so this is ~100 and ~180 lines each: enough that `--status` is
# never more than half a minute stale, few enough that the log stays readable
# and the watcher, which re-reads the whole file on every poll, stays cheap.
PROGRESS_EVERY_SECONDS = 30.0

# The `valore ...` lines (`rl/value_diagnostics.py`) are printed on the first
# iteration and then every this many. Two lines of ~12 groups each are too long
# to repeat every iteration, and the spread they report moves slowly.
VALUE_DIAGNOSTICS_EVERY = 10


class PhaseProgress:
    """Throttled `avanzamento` lines for a long, otherwise silent stage.

    The population pass and the benchmark pass each take the better part of
    an hour and print, between them, four lines. `--status` and the dashboard
    could therefore only say "elo: gioco" and leave the operator guessing
    whether the worker was working or wedged -- the 10-minute stale-log warning
    fires routinely inside both -- and could not answer the question actually
    being asked, which is how much is left.

    The pass itself reports every session and knows nothing about cadence;
    deciding how often that reaches the log belongs here, with the rest of what
    a run prints. The ETA is the plainest possible extrapolation of the elapsed
    time, which is the right model for these two stages: every session is the
    same number of hands, so the rate really is flat.
    """

    def __init__(
        self,
        stage: str,
        *,
        every_seconds: float = PROGRESS_EVERY_SECONDS,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stage = stage
        self.every_seconds = every_seconds
        self._now = now
        self._started = now()
        self._last: float | None = None

    def __call__(self, done: int, total: int, detail: str = "") -> None:
        current = self._now()
        # Always the first line (so a stage announces its size as it starts) and
        # always the last (so a finished stage does not sit at 97% forever).
        if self._last is not None and done < total and current - self._last < self.every_seconds:
            return
        self._last = current
        elapsed = current - self._started
        eta = elapsed * (total - done) / done if 0 < done < total and elapsed > 0 else None
        print(progress_marker(self.stage, done, total, detail=detail, eta_seconds=eta), flush=True)


def registry_opponents(
    registry: PoolRegistry,
    mix: TableMix,
    *,
    device: str | torch.device = "cpu",
    count: int | None = None,
    members: Sequence[PoolMember] | None = None,
    on_skip: Callable[[Path, str], None] | None = None,
) -> list[Opponent]:
    """Seat fillers from a registry of models, best rated first.

    This is the torch-side glue for `pool_registry`, which is deliberately
    torch-free. `fill_slots` may hand back the same member twice when the pool
    is short of `count`; the model behind it is loaded once and the resulting
    `Opponent` reused, so a duplicate costs a list entry and not a second copy
    of the weights in memory.

    A checkpoint that fails to load is skipped with a reason, never fatal --
    the pool directory is user-owned and one bad file must not stop training.
    """
    slots = (
        list(members)
        if members is not None
        else registry.fill_slots(registry.max_models if count is None else count)
    )
    built: dict[str, Opponent] = {}
    opponents: list[Opponent] = []
    for member in slots:
        if member.label in built:
            opponents.append(built[member.label])
            continue
        path = Path(registry.directory) / member.ref
        try:
            model, _checkpoint = build_model_from_checkpoint(path, device=device)
        except Exception as exc:  # noqa: BLE001 - see docstring
            if on_skip is not None:
                on_skip(path, str(exc))
            continue
        opponent = policy_opponent(member.label, make_policy_fn(model, device=device), mix)
        built[member.label] = opponent
        opponents.append(opponent)
    return opponents


@dataclass(frozen=True)
class TrainConfig:
    hands_per_iteration: int = 256
    opponent_probability: float = 0.5
    # A table keeps its players for this many hands, and `concurrent_tables` of them
    # are played in turn: see `rollout.DEFAULT_TABLE_HANDS`.
    table_hands: int = DEFAULT_TABLE_HANDS
    concurrent_tables: int = DEFAULT_CONCURRENT_TABLES
    # Boards a hand closed before the river is averaged over for its training reward
    # (`rl/allin_reward.py`); 0 trains on the chips that moved.
    allin_runouts: int = DEFAULT_ALLIN_RUNOUTS
    # How the pool models in the opponent seats are given a style (`rl/styles.py`).
    styles: StyleConfig = field(default_factory=StyleConfig)
    # There is deliberately no knob for seating the run's own past selves: a
    # snapshot is a copy of the network being trained, so it drifts with it and
    # anchors nothing, and every seat it took would be a seat not facing an
    # independently trained model from the pool.
    gamma: float = 1.0
    lam: float = 0.95
    # None derives big_blind / starting_stack, which puts a stack-sized swing at
    # a value target of ~1.0. See SelfPlayCollector's reward_scale.
    reward_scale: float | None = None


class SelfPlayTrainer:
    def __init__(
        self,
        mix: TableMix,
        train_config: TrainConfig | None = None,
        ppo_config: PPOConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        rng: random.Random | None = None,
        model: PokerActorCritic,
        critic: CriticFns,
        extra_opponents: Sequence[Opponent] = (),
        registry: PoolRegistry | None = None,
        initial_rating: float = DEFAULT_RATING,
        k_schedule: Sequence[tuple[int, float]] = DEFAULT_K_SCHEDULE,
    ) -> None:
        self._mix = mix
        self._k_schedule = tuple(k_schedule)
        self._registry = registry
        # Carried, not persisted: see LEARNER_LABEL. The model this run
        # publishes enters the global ranking at whatever rating the learner has
        # earned by then.
        #
        # **`initial_rating` is the parent's published rating when this run
        # inherits its weights**, not the 1500 baseline. A child of a model rated
        # 1620 does not start out as an unknown quantity: it starts as that model
        # plus some training, so the prior is informative and throwing it away
        # would make the run spend its first rated sessions re-discovering where
        # its own lineage sits. What keeps this from compounding an ancestor's
        # error down a lineage is that the prior is *weak*: at the schedule's
        # offset the inherited value carries about 6% of the weight of the final
        # rating after 600 rated sessions, the rest coming from the sessions
        # actually played. See `DEFAULT_K_SCHEDULE` for the arithmetic.
        self._learner_rating = float(initial_rating)
        # How many rated sessions the learner has played in *this* run, which is
        # what its K is read from (`DEFAULT_K_SCHEDULE`). It is not a member of
        # the registry -- see LEARNER_LABEL -- so nothing else counts them. It
        # starts at **zero even when the rating is inherited**, and the two
        # together are exactly what the schedule's hyperbolic gain expects: a
        # starting point worth something, and no claim to having earned it.
        # `rate_against_benchmark` continues this same count, so one run is rated
        # on one schedule from its first validation session to publication.
        self._learner_games = 0
        self._train = train_config if train_config is not None else TrainConfig()
        self._ppo = ppo_config if ppo_config is not None else PPOConfig()
        self._device = device
        self._rng = rng if rng is not None else random.Random()
        self._model = model.to(device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._ppo.learning_rate)
        self._iteration = 0
        self._value_diagnostics: ValueDiagnostics | None = None

        self._pool = OpponentPool(list(extra_opponents))
        self._reward_scale = (
            self._train.reward_scale
            if self._train.reward_scale is not None
            else mix.big_blind / mix.starting_stack
        )
        # The closure reads the live model, so updated weights take effect with
        # no rebuilding between iterations.
        self._collector = SelfPlayCollector(
            mix,
            make_policy_fn(self._model, device=device),
            rng=self._rng,
            opponent_pool=self._pool,
            opponent_probability=self._train.opponent_probability,
            reward_scale=self._reward_scale,
            table_hands=self._train.table_hands,
            concurrent_tables=self._train.concurrent_tables,
            allin_runouts=self._train.allin_runouts,
            styles=self._train.styles,
            critic=critic,
        )

    @property
    def model(self) -> PokerActorCritic:
        return self._model

    @property
    def style_hands(self) -> int:
        """Learner-seat hands behind `style_rates`."""
        return self._collector.style_hands

    @property
    def style_by_group(self) -> dict[str, tuple[int, dict[str, tuple[int, int]]]]:
        """The learner's style per group of table sizes: `{group: (hands, rates)}`."""
        return self._collector.style_by_group

    @property
    def style_rates(self) -> dict[str, tuple[int, int]]:
        """How this model plays: `(events, opportunities)` per statistic, over its
        recent training hands (`rollout.STYLE_WINDOW`)."""
        return self._collector.style_rates

    @property
    def value_diagnostics(self) -> ValueDiagnostics | None:
        """The critic's target spread by table size and stack, last iteration."""
        return self._value_diagnostics

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def learner_rating(self) -> float:
        return self._learner_rating

    @property
    def learner_games(self) -> int:
        """Rated sessions the learner has played in this run; sets its K."""
        return self._learner_games

    @property
    def registry(self) -> PoolRegistry | None:
        return self._registry

    def evaluate_against_pool(
        self,
        sessions: int = DEFAULT_EVAL_SESSIONS,
        *,
        seed: int | None = None,
        hands: int = DEFAULT_SESSION_HANDS,
        device: str | torch.device | None = None,
    ) -> float:
        """Play the ranked pool, return bb/100, and update everyone's rating.

        `sessions` rated sessions of `hands` hands each, with a fresh draw of
        opponents from the pool and the learner in a fresh seat before every one.
        The default is 10 sessions of 1,000 hands, and both halves of that matter:
        CLAUDE.md measures the bb/100 of a *single* session as having a ~350-570
        bb/100 spread across seeds, so one session could never rank anything,
        while a session shorter than 1,000 hands would be on a different Elo
        scale than every other rated result in the project (see `table_mix.DEFAULT_SESSION_HANDS`).
        The rating accumulates across every validation pass of the run and then
        into `rate_against_benchmark`, on one continuous session count.

        Seats are swapped through `SeatProxy` so the same `Table` lives across the
        whole pass and the button keeps rotating -- rebuilding it per session
        would reset the button to the same seat every time and hand whoever sits
        there a systematic positional edge.

        **What this number is not.** The rating these sessions build is measured
        against a field of *live* models whose own ratings carry error, and it
        rates a moving target: the weights change between one pass and the next,
        so a session played at iteration 100 scored a network that no longer
        exists. That is why the validation is deliberately the *smaller* half of
        a run's rated sessions and why the published rating is earned against the
        pinned anchors.

        **The learner is rated through `DEFAULT_K_SCHEDULE`, on its own session
        count, not at the flat K.** It is not a registry member, so it has no
        `games` the schedule could read. At a flat K the rating could not travel:
        a session moves it by at most `K * 0.5`, so the first evaluation's
        sessions are worth only a few tens of points against a pool whose mean
        sits well away from the starting rating. The schedule's first tiers are
        what let a run's first passes move the rating at all, and they expire on
        their own as the learner's session count grows.

        The returned bb/100 is measured from stack deltas, for the same reason
        `evaluate` does: hands the learner wins without acting produce no
        trajectory and are systematically winning ones.
        """
        if self._registry is None:
            raise ValueError("evaluate_against_pool needs a registry; none was configured")

        device = self._device if device is None else device
        mix = self._mix
        needed = mix.max_players - 1
        opponents = registry_opponents(
            self._registry,
            mix,
            device=device,
            count=max(self._registry.max_models, needed),
        )
        if len(opponents) < needed:
            raise ValueError(
                f"the pool can seat only {len(opponents)} opponents, "
                f"a {mix.max_players}-handed table needs {needed}"
            )

        streams = random.Random(seed)
        rng = random.Random(streams.random())
        bot_rng = random.Random(streams.random())
        seat_rng = random.Random(streams.random())
        size_rng = random.Random(streams.random())

        bank = TableBank(mix, rng)
        learners = [
            RLAgentPlayer(
                f"s{seat}",
                LEARNER_LABEL,
                policy_fn=make_policy_fn(self._model, device=device),
                big_blind=mix.big_blind,
                starting_stack=mix.starting_stack,
            )
            for seat in range(mix.max_players)
        ]

        won = 0
        for _session in range(sessions):
            # The size is drawn per session and held for all of it: a session
            # compares the same participants over the same hands.
            num_players = mix.draw_size(size_rng)
            proxies = bank.seats(num_players)
            learner_seat = seat_rng.randrange(num_players)
            # sample() draws positions, so a member that fill_slots duplicated
            # can legitimately occupy two seats -- that is what weighting a
            # short pool by duplication means.
            drawn = seat_rng.sample(opponents, num_players - 1)

            occupants: dict[int, str] = {}
            draw = iter(drawn)
            for seat, proxy in enumerate(proxies):
                if seat == learner_seat:
                    proxy.inner = learners[seat]
                    proxy.name = LEARNER_LABEL
                    occupants[seat] = LEARNER_LABEL
                else:
                    opponent = next(draw)
                    proxy.inner = opponent.factory(proxy.player_id, opponent.label, bot_rng)
                    proxy.name = opponent.label
                    occupants[seat] = opponent.label

            block_delta = bank.play_session(num_players, hands)
            won += block_delta[learner_seat]

            # A member seated twice contributes both seats to one result, so
            # it is rated on how its strategy did, not on which chair it sat in.
            results: dict[str, float] = {}
            for seat, label in occupants.items():
                results[label] = results.get(label, 0.0) + block_delta[seat]
            ratings = {
                label: self._learner_rating
                if label == LEARNER_LABEL
                else self._registry.members[label].rating
                if label in self._registry.members
                else DEFAULT_RATING
                for label in results
            }
            # The learner is not a registry member, so the schedule cannot read
            # its games; it counts them itself and hands over the K. Read before
            # the increment, exactly as `record_session_with_ratings` does for a
            # member: a session is rated at the experience its player had when
            # it sat down.
            k_factors = {
                label: k_for_games(member.games, self._k_schedule)
                for label, member in self._registry.members.items()
                if label in results
            }
            k_factors[LEARNER_LABEL] = k_for_games(self._learner_games, self._k_schedule)
            deltas = self._registry.record_session_with_ratings(
                results, ratings, k_factors=k_factors
            )
            self._learner_rating += deltas.get(LEARNER_LABEL, 0.0)
            self._learner_games += 1

        return won / mix.big_blind / (sessions * hands) * 100

    def archive(
        self, path: str | Path, *, iteration: int, metadata: dict | None = None
    ) -> None:
        """Save the current weights, with the learner rating beside them.

        Calling this again with the *same* `path` overwrites in place, which is
        how `main()` adds the run's outcomes to the model it already wrote.
        Nothing is added to any
        registry: the model is published to the shared store once, at the end of
        the run (`publish_model`), so a label never changes weights after it
        exists. The sidecar carries the rating so that even a run that dies
        before publishing can have its model salvaged at the rating it earned.

        `metadata` rides inside the checkpoint (`save_checkpoint` has always had
        the slot and nothing ever used it). It is what the run was *configured
        with* -- see `run_metadata` -- and it exists for two jobs that cannot be
        done without it: attributing an outcome to the settings that produced
        it, and letting the next generation inherit those settings from a parent
        instead of guessing. No new file, no NFS concern, and it travels with
        the published model automatically because the archive *is* what gets
        published.
        """
        save_checkpoint(path, self._model, iteration=iteration, metadata=metadata)
        write_sidecar(
            path,
            rating=self._learner_rating,
            style=self.style_rates,
            style_hands=self.style_hands,
        )

    def train_iteration(self) -> dict[str, float]:
        self._iteration += 1
        trajectories = self._collector.collect(
            self._train.hands_per_iteration, gamma=self._train.gamma, lam=self._train.lam
        )
        # Before the update: these are the values the policy collected with.
        self._value_diagnostics = value_diagnostics(trajectories)
        batch = build_batch(trajectories, device=self._device)
        stats = ppo_update(self._model, self._optimizer, batch, self._ppo)
        stats["iteration"] = float(self._iteration)
        stats["decisions"] = float(len(batch))
        stats["reward_bb"] = sum(t.reward for t in trajectories) / max(len(trajectories), 1)
        return stats

    def train(
        self, iterations: int, *, checkpoint_path: str | Path | None = None
    ) -> list[dict[str, float]]:
        history = []
        for _ in range(iterations):
            stats = self.train_iteration()
            history.append(stats)
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    self._model,
                    self._optimizer,
                    iteration=self._iteration,
                    metadata={"table_weights": list(self._mix.weights)},
                )
        return history


# Bumped whenever the *meaning* of a field changes, so an analysis can refuse a
# mixture of schemas rather than average across them.
RUN_METADATA_VERSION = 1

# The axes `poker-loop` draws per worker, in the order a reader wants them: the
# arm first (it says whether the rest can be read as an independent draw), then
# the settings. This is the list the worker *reports*; the axes themselves
# live in `loop.HP_AXES`, and a test requires every one of them to appear
# here -- adding an axis to the sweep without making it visible would produce
# runs whose configuration cannot be recovered from their own log.
REPORTED_AXES = (
    "hp_arm",
    "lr",
    "hands",
    "ppo_epochs",
    "clip_epsilon",
    "minibatch_size",
    "gae_lambda",
    "value_coef",
    "policy_max_grad_norm",
    "critic_max_grad_norm",
    "entropy_coef",
    "opponent_probability",
    # How strong a field this run drew. Reported like the rest, and worth
    # reading next to the `pool rating: media ...` line printed just below,
    # which is the strength these two actually produced.
    "pool_top_share",
    "pool_top_n",
)


def add_network_arguments(parser: argparse.ArgumentParser) -> None:
    """The four numbers that make a network's shape, for every CLI that builds one.

    They are the fleet's: `config.toml`'s `[network]` section sets them and
    `poker-loop` forwards them to every worker. A model is rebuilt from the shape its
    checkpoint recorded, so changing them affects only the networks trained from now
    on -- and a worker whose parent has a different shape starts from scratch.
    """
    parser.add_argument(
        "--hidden", type=int, default=DEFAULT_HIDDEN,
        help="width of the layers of each trunk (the policy's and the critic's)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=DEFAULT_NUM_LAYERS,
        help="number of layers in each trunk (the policy's and the critic's)",
    )
    parser.add_argument(
        "--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN,
        help="width of the hidden layers of each head (policy and value)",
    )
    parser.add_argument(
        "--head-layers", type=int, default=DEFAULT_HEAD_LAYERS,
        help="hidden layers in each head before its output layer; 0 is a head that is "
        "the output layer alone",
    )
    parser.add_argument(
        "--equity-model", type=str, default="",
        help="checkpoint of an equity network (studies/equity_net), required: the policy reads "
        "its per-player encoder in place of the card planes and the critic, a network of its "
        "own, reads the equity of every player in place of them",
    )


def network_shape(args: argparse.Namespace) -> dict[str, int]:
    """The shape the flags (or `config.toml`) ask for."""
    return {
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "head_hidden": args.head_hidden,
        "head_layers": args.head_layers,
    }


def parent_shape_mismatch(path: str | Path, shape: dict[str, int], equity: dict[str, int]) -> dict | None:
    """The shape of the checkpoint at `path` if it is not `shape` (with `equity`, the shape of
    the equity encoder it is built with), else None.

    Raises `IncompatibleCheckpointError` for a checkpoint that is not this build's at
    all (another encoding or feature version): that is a broken parent, not a
    different one, and resuming from it was never going to work.
    """
    parent = torch.load(path, map_location="cpu", weights_only=True)
    check_compatible(parent, str(path))
    found = checkpoint_shape(parent)
    if found != shape:
        return found
    # The same shape, but built with another equity encoder: the weights cannot be exchanged.
    found_equity = parent["equity"]
    return None if found_equity == equity else {**found, "equity": found_equity}


def check_network_arguments(args: argparse.Namespace) -> None:
    """Refuse a shape that cannot be built, naming the flag, before anything starts."""
    for name, minimum in (("hidden", 1), ("num_layers", 1), ("head_hidden", 1), ("head_layers", 0)):
        if getattr(args, name) < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {getattr(args, name)}")
    if not args.equity_model:
        raise ValueError("--equity-model is required: set equity_model in config.toml or pass the flag")
    if not Path(args.equity_model).is_file():
        raise ValueError(f"--equity-model {args.equity_model}: no such file")
    style_config_from_args(args)  # raises, naming the flag, on a style setting that cannot be used


def run_metadata(args: argparse.Namespace) -> dict:
    """What this run was configured with, to ride inside the published model.

    `save_checkpoint` has always had a `metadata` slot and nothing ever filled
    it. Filling it is what makes two things possible that are otherwise not:
    **attributing** an outcome to the settings that produced it, and letting the
    next generation **inherit** those settings from a parent instead of guessing
    them. It costs no new file and no shared-directory write, because the
    archive this rides in *is* what `publish_model` copies into the store.

    Only what the run was *set to* goes here. The outcomes are added later, once
    they exist (see `outcome_metadata`): the benchmark rating and bb/100 are
    not known until after the last archive is written.

    `hp_arm` is recorded, never acted on: which arm a worker was in is a
    `poker-loop` decision, and without it an analysis cannot separate the runs
    whose settings were drawn independently -- the ones a response curve can be
    read from -- from the ones that inherited them, whose settings correlate
    with their parent's quality by construction.
    """
    return {
        "schema": RUN_METADATA_VERSION,
        "hp_arm": args.hp_arm,
        "machine": args.machine,
        "seed": args.seed,
        "resumed": bool(args.resume),
        "parent_label": args.parent_label,
        **network_shape(args),
        "iterations": args.iterations,
        "hands": args.hands,
        "table_weights": list(args.table_weights),
        "stack_min_bb": args.stack_min_bb,
        "stack_max_bb": args.stack_max_bb,
        "sb": args.sb,
        "bb": args.bb,
        "lr": args.lr,
        "ppo_epochs": args.ppo_epochs,
        "clip_epsilon": args.clip_epsilon,
        "minibatch_size": args.minibatch_size,
        "gae_lambda": args.gae_lambda,
        "value_coef": args.value_coef,
        "policy_max_grad_norm": args.policy_max_grad_norm,
        "critic_max_grad_norm": args.critic_max_grad_norm,
        "entropy_coef": args.entropy_coef,
        "opponent_probability": args.opponent_probability,
        "pool_models": args.pool_models,
        "table_hands": args.table_hands,
        "concurrent_tables": args.concurrent_tables,
        "pool_top_share": args.pool_top_share,
        "pool_top_n": args.pool_top_n,
        # Everything else the run resolved to (evaluation, population pass,
        # fill-in...): with a config file, a run's settings are no longer
        # recoverable from its command line.
        "settings": resolved_settings(vars(args)),
    }


def record_sweep_observation(
    args: argparse.Namespace,
    *,
    label: str,
    parent_rating: float,
    parent_settings: dict[str, float],
    rating: float,
    cpu_seconds: float,
) -> None:
    """Leave the evidence the sweep optimizer learns from, if this run has a parent.

    Only a run that resumed from a named parent whose settings are on record has a
    step to report; any other leaves nothing, and a failure to write is reported and
    swallowed -- a finished, published model must not become a failed run over a
    side file.
    """
    if not (args.resume and args.parent_label and parent_settings):
        return
    recorded = run_metadata(args)
    settings = {
        axis: recorded[axis]
        for axis in REPORTED_AXES
        if isinstance(recorded.get(axis), int | float) and not isinstance(recorded.get(axis), bool)
    }
    try:
        write_observation(
            args.global_dir,
            SweepObservation(
                label=label,
                parent=args.parent_label,
                parent_rating=parent_rating,
                rating=rating,
                cpu_seconds=cpu_seconds,
                settings=settings,
                parent_settings=parent_settings,
                machine=args.machine,
            ),
        )
    except OSError as error:
        print(f"sweep: osservazione non scritta ({error})", flush=True)
        return
    print(
        f"sweep: guadagno {rating - parent_rating:+.0f} Elo sul genitore "
        f"{args.parent_label} ({parent_rating:.0f} -> {rating:.0f}), "
        f"{cpu_seconds / 60:.0f} min di CPU",
        flush=True,
    )


# ---- the Elo fill-in phase ---------------------------------------------------
#
# A worker that reaches the end of its run while the rest of its generation is
# still training would otherwise exit and leave its core idle until the slowest
# worker finished. With `--hands` drawn per worker (`loop.hyperparameter_plan`)
# run lengths are *deliberately* ragged, so several workers a generation finish
# well early.
#
# **Why spend it on Elo and not on more training.** Two reasons. Training more
# would publish more models, and the store's problem is not that it holds too
# few but that few of them have a rating worth anything: a pass seats ~50
# models, so a given one comes up rarely and can sit for many generations on
# the number its own training run published. And a
# longer run is not comparable with the others in its generation, which is the
# whole point of the sweep -- the fast workers are fast because they drew fewer
# hands, and giving those runs extra iterations would erase the very axis being
# measured. Rating passes cost nothing to the experiment: they touch no
# published model's weights, only what is known about them.
#
# So the fast workers become the fleet's rating engine, which is where the work
# belongs: they are idle, and the ratings are what everything else reads.

# The safety cap is expressed in **minutes only** -- a
# maximum number of passes would be a cap on work, and what actually has to be
# bounded is how long a worker can hold its core while its supervisor waits.
# 150 minutes is the order of one generation, so a worker that never hears from
# its supervisor (a supervisor killed mid-generation, a state directory that
# moved) stops within about the time the generation would have taken anyway
# rather than filling forever.
DEFAULT_FILL_DEADLINE_MINUTES = 150
# **Ten sessions a pass, against the usual ~55 models drawn.** The end-of-run
# pass is `DEFAULT_GLOBAL_SESSIONS` (100) sessions; a fill-in pass is a unit
# of *waiting*, so it has to be short enough that the worker notices the stop flag
# soon after it goes up and short enough to re-draw often. Ten sessions are
# 10,000 hands and ~3 minutes, and the worker re-reads the ranking between
# passes, so a long wait is many independent draws rather than one stale one. The
# cost it does not avoid is loading the drawn models: ~203 MB and a few seconds
# every pass, which is why the pass is not made shorter still.
DEFAULT_FILL_SESSIONS = 10


def ranking_for_draw(global_dir) -> dict[str, dict]:
    """The ratings `tiered_draw` ranks by, read fresh from the shared store."""
    return {
        label: {"rating": member.rating, "games": member.games}
        for label, member in load_ranking(global_dir).members.items()
    }


def run_elo_fill_in(args, mix) -> tuple[int, int]:
    """Play rating passes until the rest of the generation catches up.

    Returns `(passes, sessions)`. Stops on the first of: the supervisor's stop
    flag (checked between passes, so a pass in progress always finishes), the
    deadline, or a pass that plays nothing at all -- a store too small to seat a table would otherwise
    spin.

    **Pruning is off and the draw is biased, and the two go together.** These
    passes lift `trigger_size` to `NO_PRUNE_TRIGGER` and draw through
    `tiered_draw`, which gives a quarter of the seats to each of ranks 1-10,
    11-100, 101-1,000 and the rest. That is the right place to spend a waiting
    worker's time -- the top is the only part of the ranking anything reads,
    `pick_parents` draws from the best 100. Eligibility for deletion is a percentile of `games`, so a biased draw
    makes the often-seated eligible sooner and leaves the rarely-drawn tail
    permanently immune: a fill-in pass that could prune would eat the middle of
    the population instead of its bottom, and it cannot.
    """
    from pokerlab.rl.global_arena import (
        NO_PRUNE_TRIGGER,
        run_population_sessions,
        tiered_draw,
    )

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    draining = scratch / FILL_DRAINING_FILENAME
    stop_file = Path(args.fill_stop_file) if args.fill_stop_file else None
    deadline = time.monotonic() + max(0.0, args.fill_deadline_minutes) * 60.0

    announce(ELO_FILL)
    # Written *before* the first pass, so the supervisor learns this worker is
    # only waiting as soon as it is true. Writing it after would have the
    # supervisor hold the whole generation for a worker already filling in.
    draining.touch()
    passes = sessions = 0
    try:
        while time.monotonic() < deadline:
            if stop_file is not None and stop_file.exists():
                break
            # Re-read every pass: the previous one just moved the ratings the
            # bias is computed from, and other machines moved them too.
            ratings = ranking_for_draw(args.global_dir)
            report = run_population_sessions(
                global_dir=args.global_dir,
                root=args.global_root,
                mix=mix,
                machine=args.machine,
                population_sample=args.global_sample,
                benchmark_sample=args.global_benchmark_sample,
                sessions=args.fill_sessions,
                session_hands=args.session_hands,
                device=args.device,
                lock_ttl=args.global_lock_seconds,
                trigger_size=NO_PRUNE_TRIGGER,
                k_schedule=parse_k_schedule(args.k_schedule),
                draw=tiered_draw(ratings, tiers=parse_parent_tiers(args.draw_tiers)),
                on_skip=lambda path, why: print(f"  riempimento elo, saltato {path}: {why}"),
            )
            if report.played == 0:
                print("riempimento elo: popolazione insufficiente per un tavolo, esco")
                break
            passes += 1
            # `sessions_played`, not `sessions`: the latter counts what this
            # merge folded in from every machine, which on a busy fleet is
            # thousands and would misreport what this worker played.
            sessions += report.sessions_played
            remaining = deadline - time.monotonic()
            # No ETA on purpose: the number this worker knows is its safety cap,
            # and reporting that as "time left" would make the generation's
            # `fine :` line quote hours for the one worker nobody is waiting on
            # (see `monitor.format_finishing_line`).
            print(
                f"riempimento elo: passata {passes}, {sessions} sessioni giocate, "
                f"tetto fra {max(0, round(remaining / 60))}m",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - the run is already complete and
        # published by this point; a failure here must cost the extra passes and
        # nothing else, exactly as the end-of-run pass's own guard does.
        print(f"riempimento elo interrotto per errore: {exc}")
    finally:
        # Always, including on the error path: a marker left behind would have
        # the next generation's supervisor read a stale directory as a worker
        # that is already draining.
        draining.unlink(missing_ok=True)
    print(f"riempimento elo: {passes} passate, {sessions} sessioni giocate")
    return passes, sessions


def inherited_rating(path: str | Path, *, default: float = DEFAULT_RATING) -> float:
    """The rating a resumed parent was published with, or `default`.

    A run that inherits its weights inherits its starting rating too: the child
    is that parent plus some training, not an unknown quantity, so starting it at
    the 1500 baseline would spend its first rated sessions re-discovering where
    its own lineage already sits. See `SelfPlayTrainer.__init__` for why this does
    not compound an ancestor's error (the prior is weak by construction) and
    `pool_registry.DEFAULT_K_SCHEDULE` for the offset that encodes how weak.

    `publish_rating` is what a model was registered with -- the number the global
    ranking holds for it. `pool_rating` is the fallback when it is missing:
    worse, being measured against one run's drawn pool, but far better than the
    baseline. Anything unreadable falls through to `default`, which is the same
    thing a fresh run does.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - a missing or foreign file is not fatal
        return float(default)
    metadata = checkpoint.get("metadata") or {}
    for key in ("publish_rating", "pool_rating"):
        value = metadata.get(key)
        if isinstance(value, int | float) and math.isfinite(value):
            return float(value)
    return float(default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a poker agent with self-play PPO.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--hands", type=int, default=256, help="hands collected per iteration")
    add_table_arguments(parser)
    parser.add_argument("--lr", type=float, default=3e-4)
    add_network_arguments(parser)
    # The two PPO knobs worth sweeping, exposed so a per-worker draw can reach
    # them. `epochs x clip_epsilon` is the classic stability trade -- more
    # passes over the same batch move further from the policy that collected
    # it, and the clip is what bounds how far. Adding a flag is safe; removing
    # one while a supervisor runs is what takes the fleet down (see "Never
    # delete a `poker-train` CLI flag while a supervisor is running").
    parser.add_argument(
        "--ppo-epochs", type=int, default=PPOConfig.epochs,
        help="passes PPO makes over each collected batch",
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=PPOConfig.clip_epsilon,
        help="PPO's ratio clip: how far the updated policy may move from the "
        "one that collected the batch",
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=PPOConfig.minibatch_size,
        help="decisions per PPO minibatch; smaller means more gradient steps per epoch",
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=TrainConfig.lam,
        help="GAE lambda: how far advantages look ahead before trusting the critic",
    )
    parser.add_argument(
        "--value-coef", type=float, default=PPOConfig.value_coefficient,
        help="weight of the value loss in the PPO loss",
    )
    parser.add_argument(
        "--policy-max-grad-norm", type=float, default=PPOConfig.policy_max_grad_norm,
        help="gradient clipping norm of the policy's weights",
    )
    parser.add_argument(
        "--critic-max-grad-norm", type=float, default=PPOConfig.critic_max_grad_norm,
        help="gradient clipping norm of the critic's weights: a clip of its own, since the two "
        "networks share no weight and their gradients live on different scales",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=PPOConfig.entropy_coefficient,
        help="weight of the entropy bonus in the PPO loss. Zero by default: a "
        "bonus here buys per-decision dithering, not different strategies, since "
        "it perturbs each decision independently while a bluff is a sequence. See "
        "PPOConfig.entropy_coefficient for the measurements behind that",
    )
    parser.add_argument(
        "--device", default="auto",
        help="cpu, cuda or auto (the default): the GPU if there is one",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/agent.pt"))
    parser.add_argument("--resume", action="store_true", help="load --checkpoint before training")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="the shared store of every trained model: opponents are drawn from "
        "here and this run's final model is published here when it ends",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_SCRATCH_DIR,
        help="where this run keeps its final model until it is published",
    )
    parser.add_argument(
        "--archive-prefix",
        default="",
        help="inserted in the published model's name after the machine id, e.g. "
        "gen0155-w07, so a model's name says which run produced it",
    )
    parser.add_argument(
        "--machine",
        default=socket.gethostname().split(".")[0],
        help="this host's id: prefixed to published model names so machines "
        "cannot collide, and named in the locks it takes",
    )
    parser.add_argument(
        "--pool-models",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="how many opponents to draw from the shared store for this run "
        "(0 disables); every run draws its own, so runs face different fields",
    )
    parser.add_argument(
        "--pool-top-share", type=float, default=DEFAULT_TOP_SHARE,
        help="share of those opponents drawn from the best-rated models; every "
        "remaining seat is drawn uniformly from the whole store",
    )
    parser.add_argument(
        "--pool-top-n", type=int, default=DEFAULT_TOP_N,
        help="how many of the best-rated models count as 'the top' for the draw",
    )
    parser.add_argument(
        "--opponent-probability",
        type=float,
        default=TrainConfig.opponent_probability,
        help="chance that a seat the learner does not hold goes to an opponent "
        "rather than to another copy of the learner. At 6-max the default 0.5 "
        "leaves 2.5 of 6 seats to opponents, every one of them a previously "
        "trained pool model; 1.0 doubles it",
    )
    parser.add_argument(
        "--table-hands", type=int, default=TrainConfig.table_hands,
        help="hands a training table keeps the same players: what the model reads about "
        "an opponent (VPIP, 3-bet...) is then true of the one in front of it, and it "
        "fills in over these hands. 1 redraws the table every hand and the statistics "
        "never exist",
    )
    parser.add_argument(
        "--concurrent-tables", type=int, default=TrainConfig.concurrent_tables,
        help="tables played in turn, hand by hand, so a batch is not the same few "
        "opponents repeated for --table-hands hands",
    )
    add_style_arguments(parser)
    parser.add_argument(
        "--allin-runouts", type=int, default=TrainConfig.allin_runouts,
        help="a hand whose betting closed before the river is trained on its expected "
        "result over this many boards instead of the one that came (unbiased, and it "
        "drops the runout's luck from the value target); 0 trains on the chips that "
        "moved. Training only: every reported and rated result is the real chips",
    )
    parser.add_argument(
        "--critic-stack-power", type=float, default=PPOConfig.critic_stack_power,
        help="weigh each decision in the critic's loss by its chips at stake in big "
        "blinds to this power, negated (0: every decision alike). The target's spread "
        "grows with the stake, so without it the critic learns the deep hands and "
        "ignores the short ones; a weight on the state alone leaves the policy's "
        "objective unchanged",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--hp-arm", default="",
        help="which hyperparameter arm `poker-loop` put this worker in "
        "(\"sampled\" or \"inherited\"); recorded in the published model and "
        "never acted on, so an analysis can tell the two apart",
    )
    parser.add_argument(
        "--parent-label", default="",
        help="the published model this run resumed from, given by `poker-loop`; "
        "recorded in the published model and, with the parent's settings, in the "
        "sweep log the optimizer learns from (`rl/sweep_log.py`)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=100,
        help="iterations between validation passes (0 disables)",
    )
    parser.add_argument(
        "--eval-sessions", type=int, default=DEFAULT_EVAL_SESSIONS,
        help="rated sessions per validation pass, each --session-hands hands "
        "with opponents re-drawn from the pool. Over a 1000-iteration run at "
        "--eval-every 100 that is 100 rated sessions, which then continue into "
        "the pass against the frozen anchors on one session count",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR,
        help="the frozen opponent set the benchmark pass rates this run's "
        "published model against (--benchmark-sessions) -- never seated in training",
    )
    parser.add_argument(
        "--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR,
        help="the cross-machine population Elo registry (rl/global_arena.py)",
    )
    parser.add_argument(
        "--global-root", type=Path, default=Path("checkpoints"),
        help="the checkpoint root: the model store (models/) and the benchmark "
        "sets (benchmark/) the population passes draw from are found under it",
    )
    parser.add_argument("--global-sample", type=int, default=DEFAULT_POPULATION_SAMPLE)
    parser.add_argument("--global-benchmark-sample", type=int, default=DEFAULT_BENCHMARK_SAMPLE)
    parser.add_argument(
        "--draw-tiers", type=draw_tiers_text, default=format_parent_tiers(DRAW_TIERS),
        help="who the population passes (end of run and fill-in) seat: the cutoffs "
             "of the rating ranking, each with an equal share of the seats (a tier is "
             "the best N rated models, or 'all'); a tier listed twice gets twice the "
             "share. '10, 100, 1000, all' is a quarter each of ranks 1-10, 11-100, "
             "101-1000 and the rest",
    )
    parser.add_argument(
        "--global-sessions", type=int, default=DEFAULT_GLOBAL_SESSIONS,
        help="rated sessions of the end-of-run population pass, each seating "
        "models drawn at random from the sample; the pass's cost is very nearly "
        "linear in it (~1.1 s per session)",
    )
    parser.add_argument(
        "--benchmark-sessions", type=int, default=DEFAULT_BENCHMARK_SESSIONS,
        help="rated sessions of --session-hands hands this run's model plays "
        "against opponents drawn at random from --benchmark-dir before it is "
        "published (0 disables). The rating it earns there is what it is "
        "published with, and its session count continues the one the validation "
        "passes built",
    )
    parser.add_argument(
        "--benchmark-resident", type=int, default=DEFAULT_RESIDENT_ANCHORS,
        help="frozen anchors held in memory at once during the round against "
        "them: each slice is drawn at random and dropped for the next, so this "
        "bounds the worker's memory while the round still faces the whole set",
    )
    parser.add_argument(
        "--benchmark-rotate-every", type=int, default=DEFAULT_ANCHOR_ROTATE_EVERY,
        help="rated sessions played against one slice of --benchmark-resident "
        "anchors before a new slice is drawn (lower: more checkpoint loads, "
        "more varied field)",
    )
    parser.add_argument(
        "--global-lock-seconds", type=int, default=DEFAULT_LOCK_SECONDS,
        help="how long a per-model lock lives before it counts as abandoned",
    )
    parser.add_argument(
        "--k-schedule", type=k_schedule_text, default=format_k_schedule(DEFAULT_K_SCHEDULE),
        help="the Elo K staircase as games:K pairs, e.g. '0:16, 20:11, 45:7.4': the K a "
             "model is rated at once it has played that many sessions. Applies to the "
             "learner, the pass against the anchors and the population merge",
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
        "--global-elo", dest="global_elo", action="store_true", default=True,
        help="after this run finishes, try one cross-machine population Elo pass "
        "(on by default)",
    )
    parser.add_argument(
        "--no-global-elo", dest="global_elo", action="store_false",
        help="disable the end-of-run population pass entirely",
    )
    # Off by default, and that is the right way round: a hand-run `poker-train`
    # should end when it ends, not sit for two hours playing rating passes. Only
    # `poker-loop` turns it on, because only a supervisor has other workers to
    # wait for and a stop flag to tell this one when they are done.
    parser.add_argument(
        "--elo-fill-in", dest="elo_fill_in", action="store_true", default=False,
        help="after the end-of-run pass, keep playing rating passes until the "
        "supervisor's stop file appears or the deadline passes; a worker that "
        "finished early spends the wait on ratings instead of idling",
    )
    parser.add_argument(
        "--fill-stop-file", default="",
        help="the supervisor writes this file once every worker of the "
        "generation has finished or is itself filling in; empty means nothing "
        "will ever say stop, so only the deadline ends the phase",
    )
    parser.add_argument(
        "--fill-deadline-minutes", type=float, default=DEFAULT_FILL_DEADLINE_MINUTES,
        help="hard cap on the phase, in minutes; the only cap there is",
    )
    parser.add_argument(
        "--fill-sessions", type=int, default=DEFAULT_FILL_SESSIONS,
        help="rated sessions of one fill-in pass, each seating models drawn at "
        "random from the sample; small on purpose, so the worker re-draws often "
        "and stops soon after being told to",
    )
    add_config_arguments(parser)
    return parser


def _sibling_parsers() -> list[argparse.ArgumentParser]:
    """Every other CLI that reads `config.toml`: a key meant for one is not a typo here."""
    return sibling_parsers("pokerlab.rl.train", with_torch=True)


def main() -> None:
    parser = build_parser()
    try:
        args, report = parse_with_config(parser, siblings=_sibling_parsers)
    except ConfigError as error:
        parser.exit(2, f"{parser.prog}: config: {error}\n")
    args.device = resolve_device(args.device)
    if args.print_config:
        print(format_config(vars(args), [k for k, v in report.applied.items() if vars(args)[k] == v]))
        return
    if report.path is not None:
        print(f"config: {report.path}, {len(report.applied)} valori", flush=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    mix = table_mix_from_args(args)
    k_schedule = parse_k_schedule(args.k_schedule)

    # This run's own draw from the shared store: different for every run, so
    # workers do not all train against the same field. Rated against the global
    # ranking's snapshot, and held only in memory (see `build_training_registry`).
    draw_rng = random.Random(args.seed)
    ranking = load_ranking(args.global_dir).members
    drawn = draw_training_pool(
        ranking,
        available_labels(args.models_dir),
        size=args.pool_models,
        rng=draw_rng,
        top_share=args.pool_top_share,
        top_n=args.pool_top_n,
    )
    registry = build_training_registry(args.models_dir, drawn)

    archived = (
        registry_opponents(
            registry,
            mix,
            device=args.device,
            count=args.pool_models,
            on_skip=lambda path, why: print(f"skipping {path.name}: {why}"),
        )
        if args.pool_models > 0
        else []
    )
    check_network_arguments(args)
    shape = network_shape(args)
    # The policy gets the equity network's per-player encoder (frozen, kept in the model's
    # own weights) and the critic the full network's output (training only).
    equity_saved = load_equity_checkpoint(args.equity_model)
    equity_config = encoder_config(equity_saved)
    model = PokerActorCritic(**shape, equity=equity_config)
    load_encoder_weights(model.equity_encoder, equity_saved)
    critic = make_critic_fns(model, equity_net_from_checkpoint(equity_saved), device=args.device)
    if args.resume:
        # A parent of another shape cannot be loaded into this network, and
        # `pick_parents` draws by rating, not by shape, so after the shape in
        # `config.toml` changes this is an ordinary event, not an error: the worker
        # trains the new shape from scratch (and, having taken no step from that
        # parent, records no sweep observation).
        parent_shape = parent_shape_mismatch(args.checkpoint, shape, equity_config)
        if parent_shape is not None:
            print(
                f"parent has a different network ({parent_shape}), this run builds "
                f"{ {**shape, 'equity': equity_config} }: "
                "starting from scratch instead of resuming",
                flush=True,
            )
            args.resume = False
    trainer = SelfPlayTrainer(
        mix,
        TrainConfig(
            hands_per_iteration=args.hands,
            opponent_probability=args.opponent_probability,
            table_hands=args.table_hands,
            concurrent_tables=args.concurrent_tables,
            allin_runouts=args.allin_runouts,
            styles=style_config_from_args(args),
            lam=args.gae_lambda,
        ),
        PPOConfig(
            learning_rate=args.lr,
            entropy_coefficient=args.entropy_coef,
            epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            minibatch_size=args.minibatch_size,
            value_coefficient=args.value_coef,
            policy_max_grad_norm=args.policy_max_grad_norm,
            critic_max_grad_norm=args.critic_max_grad_norm,
            critic_stack_power=args.critic_stack_power,
        ),
        device=args.device,
        rng=random.Random(args.seed),
        model=model,
        critic=critic,
        extra_opponents=archived,
        registry=registry,
        initial_rating=inherited_rating(args.checkpoint) if args.resume else DEFAULT_RATING,
        k_schedule=k_schedule,
    )
    # What the parent was trained with and published at: the other half of the
    # step this run is about to take, for the sweep log written at publication.
    parent_settings: dict[str, float] = {}
    parent_rating = trainer.learner_rating
    if args.resume:
        resumed = load_checkpoint(args.checkpoint, trainer.model, device=args.device)
        # the file is the source of truth for the frozen encoder
        load_encoder_weights(trainer.model.equity_encoder, equity_saved)
        parent_settings = {
            axis: value
            for axis, value in (resumed.get("metadata") or {}).items()
            if axis in REPORTED_AXES
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        }
        print(
            f"resumed from {args.checkpoint} "
            f"(rating ereditato {trainer.learner_rating:.0f})"
        )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)
    archive_path = Path(args.scratch_dir) / f"agent-{run_id}.pt"
    print(f"device {args.device} | tables {sizes_text(args.table_weights)} | {args.hands} hands/iteration")
    # Built from `run_metadata`, not from `args` directly, so the line in the log
    # and the metadata inside the published checkpoint cannot disagree about what
    # this run was.
    recorded = run_metadata(args)
    print(
        hyperparameters_marker({axis: recorded[axis] for axis in REPORTED_AXES if axis in recorded}),
        flush=True,
    )
    models = registry.ranked()
    if models:
        print(f"pool: drew {len(models)} of {len(available_labels(args.models_dir))} models "
              f"from {args.models_dir}, seating {len(archived)}")
        print(registry.summary(limit=5))
        # On its own line and in a fixed shape, because `rl/monitor.py` parses it
        # into the `pool` column of `--status` and the dashboard. Without it the
        # `eval` and `rating` columns are uninterpretable: they say how the learner
        # did, and this says against whom. Two workers of the same generation draw
        # different fields, so a rating of 1520 can mean opposite things.
        pool_ratings = [m.rating for m in registry.members.values()]
        if pool_ratings:
            print(
                f"pool rating: media {statistics.fmean(pool_ratings):.0f} "
                f"min {min(pool_ratings):.0f} max {max(pool_ratings):.0f}",
                flush=True,
            )
    else:
        print(f"no models in {args.models_dir} yet -- plain self-play for this run")
    for _ in range(args.iterations):
        stats = trainer.train_iteration()
        print(
            f"iter {int(stats['iteration']):4}  "
            f"reward {stats['reward_bb']:+7.2f} bb  "
            f"policy {stats['policy_loss']:+.4f}  "
            f"value {stats['value_loss']:8.3f}  "
            f"entropy {stats['entropy']:.3f}  "
            f"kl {stats['approx_kl']:.4f}  "
            f"clip {stats['clip_fraction']:.3f}"
        )
        # Every iteration, so the dashboard can draw it like the `iter` columns; a line
        # of its own so the `iter` format the status parsers read does not change.
        print(
            format_gradient_line(
                GradientReading(
                    policy=ClipReading(
                        stats["grad_norm_policy"], stats["grad_norm_policy_sd"],
                        stats["grad_clipped_policy"], args.policy_max_grad_norm,
                    ),
                    critic=ClipReading(
                        stats["grad_norm_critic"], stats["grad_norm_critic_sd"],
                        stats["grad_clipped_critic"], args.critic_max_grad_norm,
                    ),
                    steps=int(stats["grad_steps"]),
                )
            )
        )
        iteration = int(stats["iteration"])
        if trainer.value_diagnostics is not None and (
            iteration == 1 or iteration % VALUE_DIAGNOSTICS_EVERY == 0
        ):
            for line in format_value_diagnostics(trainer.value_diagnostics):
                print(line)
            if trainer.style_hands:
                print(format_style_line(trainer.style_hands, trainer.style_rates))
                for group, (hands, rates) in trainer.style_by_group.items():
                    print(format_style_line(hands, rates, group=group))
            # Starts with "vantaggi", so the status parsers (which read only
            # `iter ` lines) are unaffected, like the `valore` lines.
            print(
                f"vantaggi grezzi: media {stats['adv_mean']:+.4f}  "
                f"std {stats['adv_std']:.4f}"
            )
        if args.eval_every and (
            iteration % args.eval_every == 0
            # The last iteration always evaluates, whatever the interval: this
            # reading is the run's final word on the learner, and it is also the
            # rating the model is published with whenever the benchmark pass
            # cannot run.
            or iteration == args.iterations
        ):
            # Nobody to play against yet (an empty store): rating waits for a
            # later run, but the live checkpoint is still saved.
            if registry.members:
                announce(EVALUATION)
                win_rate = trainer.evaluate_against_pool(
                    args.eval_sessions,
                    seed=args.seed,
                    hands=args.session_hands,
                )
                print(
                    f"        eval vs pool: {win_rate:+.1f} bb/100  "
                    f"rating {trainer.learner_rating:.0f}"
                )
            save_checkpoint(
                args.checkpoint, trainer.model, trainer.optimizer, iteration=trainer.iteration
            )
            print(f"        saved {args.checkpoint}")

    # **The model that is kept is the one the run ends with**, and it is written
    # once, here. It has to be written before the benchmark pass: that pass
    # measures `trainer.model` -- the live weights -- and publishes the rating it
    # earns, so the archive has to *be* those weights or the number would
    # describe a different model than the one that goes into the store. Written
    # this early, a worker that dies during the benchmark still leaves a file
    # (and the sidecar with its rating) for the loop's sweep to publish.
    #
    # **The latest checkpoint, not the best-rated one.** The rating comes from a
    # few thousand hands against this run's own drawn pool, a noisy measurement,
    # while training improving the model is a reliable prior: selecting on a noisy
    # rating throws away part of the improvement the later checkpoint carries.
    # Re-measure this (duplicate-deck duel of a mid-run checkpoint against the
    # final one from the same seed) before changing it. Replace the blind rule
    # only with a *good* measurement -- the benchmark bb/100 is deterministic and
    # scored against models never seated in training, so selecting on it would
    # also catch the runs where the later model genuinely is worse.
    pool_rating = trainer.learner_rating
    final_iteration = trainer.iteration
    trainer.archive(archive_path, iteration=final_iteration, metadata=run_metadata(args))
    print(f"        archived {archive_path} (iterazione {final_iteration}, "
          f"rating {pool_rating:.0f})")

    # **The benchmark pass runs before publishing, and its rating is what the
    # model is published with.** The alternative -- publish first, on the rating
    # `evaluate_against_pool` measured -- puts a model into the global ranking on
    # a number earned against the 50 opponents this one run happened to draw,
    # which orders the top of the ranking unreliably. The frozen anchors
    # are a fixed scale and are never seated in training, so the rating they give
    # means the same thing for every model on every machine.
    #
    # **These sessions are deliberately NOT queued for the global merge.** They
    # are already in the rating the model is published with, and
    # `_apply_session` would re-apply the identical evidence a second time from
    # the member's own games count -- worse than applying it once (simulated),
    # because re-using the same outcomes cannot add information and does add
    # movement. The model is registered with the rating *and* the session
    # count this pass earned (`games_after`), which is what makes the local
    # computation authoritative rather than a report.
    prefix = f"{args.archive_prefix}-" if args.archive_prefix else ""
    label = f"{args.machine}-{prefix}{archive_path.stem}"
    publish_rating = pool_rating
    publish_games = trainer.learner_games
    benchmark_bb100: float | None = None
    benchmark_by_size: dict[str, float] = {}
    benchmark_hands = 0
    # How the model plays: what training measured, then refreshed by the benchmark
    # pass below, whose half a million hands dwarf the training window.
    final_style = {name: list(pair) for name, pair in trainer.style_rates.items()}
    final_style_hands = trainer.style_hands

    if args.benchmark_sessions > 0:
        announce(SERIES)
        from pokerlab.rl.benchmark import rate_against_benchmark

        anchors = {
            label_: member.rating
            for label_, member in load_ranking(args.global_dir).members.items()
        }
        rated = rate_against_benchmark(
            trainer.model,
            args.benchmark_dir,
            mix,
            label=label,
            rating=pool_rating,
            games=trainer.learner_games,
            schedule=k_schedule,
            anchor_ratings=anchors,
            sessions=args.benchmark_sessions,
            hands=args.session_hands,
            resident=args.benchmark_resident,
            rotate_every=args.benchmark_rotate_every,
            device=args.device,
            seed=args.seed or 0,
            on_skip=lambda path, why: print(f"  benchmark, saltato {path}: {why}"),
            on_progress=PhaseProgress(SERIES),
        )
        if rated is not None:
            publish_rating = rated.rating_after
            publish_games = rated.games_after
            benchmark_bb100, benchmark_hands = rated.bb_per_100, rated.hands
            final_style, final_style_hands = merge_style(
                final_style, final_style_hands, rated.style, rated.style_hands
            )
            print(
                f"        benchmark: {rated.bb_per_100:+.1f} bb/100 su "
                f"{rated.sessions} sessioni contro {rated.opponents} ancore, "
                f"rating {publish_rating:.0f} "
                f"(era {pool_rating:.0f} contro il pool, "
                f"{publish_games} partite)"
            )
            # A diagnostic and nothing else: the rating is one number over the
            # whole mixture, this says whether a model is lopsided across sizes.
            benchmark_by_size = {str(n): bb for n, bb in rated.bb_per_100_by_size.items()}
            print(
                "        per tavolo: "
                + "  ".join(f"{n} giocatori {bb:+.1f}" for n, bb in rated.bb_per_100_by_size.items())
                + " bb/100"
            )

    # The outcomes only exist now, after the benchmark pass, so the archive
    # is rewritten once with them before it is copied into the store. Same
    # weights (`trainer.model` is what the pass just measured, and what the
    # last archive holds), one 3 MB write, and the published model then
    # carries both what it was configured with and what that produced --
    # which is what makes a population of runs analysable at all.
    #
    # `benchmark_bb100` is the response variable to read: `--benchmark-sessions`
    # sessions of 1,000 hands against the frozen anchors, which are never
    # seated in training and whose ratings are pinned.
    trainer.archive(
        archive_path,
        iteration=final_iteration,
        metadata={
            **run_metadata(args),
            "pool_rating": pool_rating,
            "publish_rating": publish_rating,
            "publish_games": publish_games,
            "benchmark_bb100": benchmark_bb100,
            "benchmark_bb100_by_size": benchmark_by_size,
            "benchmark_hands": benchmark_hands,
            # How the model plays at the end of training (events, opportunities over
            # its last `STYLE_WINDOW` seat-hands), kept with the weights it describes.
            "style": final_style,
            "style_hands": final_style_hands,
        },
    )
    # The CPU this child cost, up to its publication: training, validation and the
    # benchmark pass. Process time and not wall time -- a machine running twenty
    # workers would otherwise bill a setting for its neighbours.
    cpu_seconds = time.process_time()
    published = publish_model(
        archive_path,
        models_dir=args.models_dir,
        global_dir=args.global_dir,
        name=f"{args.machine}-{prefix}{archive_path.name}",
        rating=publish_rating,
        games=publish_games,
        style=final_style,
        style_hands=final_style_hands,
        machine=args.machine,
        lock_ttl=args.global_lock_seconds,
    )
    if published is not None:
        print(
            f"published {published.label} "
            f"(rating {publish_rating:.0f}, {publish_games} partite)"
        )
        archive_path.unlink(missing_ok=True)
        archive_path.with_suffix(".json").unlink(missing_ok=True)
        record_sweep_observation(
            args,
            label=published.label,
            parent_rating=parent_rating,
            parent_settings=parent_settings,
            rating=publish_rating,
            cpu_seconds=cpu_seconds,
        )

    if args.global_elo:
        try:
            from pokerlab.rl.global_arena import run_population_sessions, tiered_draw

            report = run_population_sessions(
                global_dir=args.global_dir,
                root=args.global_root,
                mix=mix,
                machine=args.machine,
                population_sample=args.global_sample,
                benchmark_sample=args.global_benchmark_sample,
                sessions=args.global_sessions,
                session_hands=args.session_hands,
                device=args.device,
                lock_ttl=args.global_lock_seconds,
                draw=tiered_draw(
                    ranking_for_draw(args.global_dir),
                    tiers=parse_parent_tiers(args.draw_tiers),
                ),
                trigger_size=args.global_trigger_size,
                eliminate_fraction=args.global_eliminate_fraction,
                protect_percentile=args.global_protect_percentile,
                k_schedule=k_schedule,
                on_skip=lambda path, why: print(f"  global pass, saltato {path}: {why}"),
                on_phase=announce,
                on_progress=PhaseProgress(ELO_PLAY),
            )
        except Exception as exc:  # noqa: BLE001 - a shared cross-machine pass
            # must never turn an otherwise-successful training run into a
            # failed process exit; the next run, here or elsewhere, just
            # tries again.
            print(f"global pass saltato per errore: {exc}")
        else:
            if report.sessions == 0 and report.deferred_sessions == 0:
                print("global pass: popolazione insufficiente per un tavolo, salto questa passata")
                if report.ghosts_dropped:
                    print(f"  ripulite {report.ghosts_dropped} voci fantasma dal registro")
            else:
                line = (
                    f"global pass: {report.sessions} sessioni, "
                    f"{report.participants} modelli coinvolti"
                )
                if report.deferred_sessions:
                    line += f", {report.deferred_sessions} rimandate (modelli occupati)"
                if report.ghosts_dropped:
                    line += f", {report.ghosts_dropped} voci fantasma ripulite"
                if report.triggered_elimination:
                    line += (
                        f", eliminati {report.eliminated}, "
                        f"{report.deleted_files} file cancellati"
                    )
                print(line)
        try:
            from pokerlab.rl.global_arena import add_benchmark_candidates

            for member in add_benchmark_candidates(
                global_dir=args.global_dir,
                root=args.global_root,
                machine=args.machine,
                games_percentile=args.benchmark_games_percentile,
                margin=args.benchmark_margin,
                minimum_anchors=mix.max_players - 1,
                lock_ttl=args.global_lock_seconds,
                on_skip=lambda path, why: print(f"  benchmark, saltato {path}: {why}"),
            ):
                print(
                    f"benchmark: aggiunto {member.label} "
                    f"(rating {member.rating:.0f}, {member.games} partite)"
                )
        except Exception as exc:  # noqa: BLE001 - same reasoning as the pass above
            print(f"aggiunta al benchmark saltata per errore: {exc}")

    # Only after the run's own pass: the fill-in passes are extra, and a worker
    # whose own results were not merged has nothing to fill in *for*.
    if args.elo_fill_in:
        run_elo_fill_in(args, mix)

    # Whatever happened above -- a pass skipped, an error absorbed, no pass
    # asked for -- the run got to its end; the watcher's "still working" ends here.
    announce(DONE)


if __name__ == "__main__":
    main()
