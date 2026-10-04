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
from dataclasses import dataclass
from pathlib import Path

import torch

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.benchmark import (
    DEFAULT_BENCHMARK_DIR,
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
    load_ranking,
    publish_model,
    write_sidecar,
)
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
from pokerlab.rl.policy import PokerActorCritic, make_policy_fn
from pokerlab.rl.pool_registry import (
    DEFAULT_ELIMINATION_FRACTION,
    DEFAULT_K_SCHEDULE,
    DEFAULT_POOL_SIZE,
    DEFAULT_POPULATION_TRIGGER,
    DEFAULT_PROTECT_PERCENTILE,
    DEFAULT_RATING,
    PoolMember,
    PoolRegistry,
    k_for_games,
)
from pokerlab.rl.ppo import (
    PPOConfig,
    build_batch,
    build_model_from_checkpoint,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from pokerlab.rl.rollout import (
    Opponent,
    OpponentPool,
    SeatProxy,
    SelfPlayCollector,
    policy_opponent,
)
from pokerlab.rl.training_pool import (
    DEFAULT_TOP_N,
    DEFAULT_TOP_SHARE,
    available_labels,
    build_training_registry,
    draw_training_pool,
)

# Where a run keeps its best-so-far model until it is published to the shared
# store at the end of the run.
DEFAULT_SCRATCH_DIR = Path("checkpoints/scratch")

# The live learner is a session participant but never a pool member: it changes
# every iteration, so persisting it would rate a moving target. Its rating is
# carried on the trainer and written beside each checkpoint it saves.
LEARNER_LABEL = "learner"

# Hands per rated block inside `evaluate_against_pool`, i.e. how long one of its
# sessions is. It was 25, which made the in-run rating a sequence of coin flips:
# Elo reads only the *sign* of a block's chip delta, and at 25 hands the stronger
# side finishes ahead about 52% of the time, which equilibrates ~14 rating points
# above the pool instead of the ~150 the same strength deserves. A thousand hands
# takes that to ~63% and ~91 points, against an unchanged accumulated noise of
# ~46 -- the signal goes from a quarter of the noise to twice it. Session length
# is the only lever that moves it; a lower K shrinks the jitter around the
# equilibrium and cannot move the equilibrium itself. Same argument, same number,
# as `global_arena.DEFAULT_HANDS_PER_GAME`.
# One rated session is 1,000 hands, everywhere: the in-run validation, the round
# against the frozen anchors and the population rounds all use the same length,
# because the Elo scale itself is defined by how often a session of that length
# picks the stronger model (measured: 69.7% for a 136-point gap, against the
# 68.6% the logistic predicts). A round of a different length would be a
# different scale silently sharing the same numbers.
#
# `--eval-hands` and `--eval-rotate-every` used to set this pair and are gone:
# they asked for a hand count that had to be a multiple of a block size, which
# said nothing about what a reader wants to know -- how many rated results an
# evaluation produces. A round is now `--eval-sessions` sessions, full stop, with
# opponents re-drawn from the pool before each one.
SESSION_HANDS = 1000
DEFAULT_EVAL_SESSIONS = 10


def announce(stage: str) -> None:
    """Write the `phase: <stage>` line `poker-loop --status` looks for.

    Its own line, flushed at once: the watcher reads this log while the run is
    still going, and a stage that lasts minutes (the population round plays
    thousands of hands without printing anything) would otherwise show as
    whatever the last ordinary line said.
    """
    print(marker(stage), flush=True)


# One progress line at most this often. The two stages that report take ~50 and
# ~90 minutes, so this is ~100 and ~180 lines each: enough that `--status` is
# never more than half a minute stale, few enough that the log stays readable
# and the watcher, which re-reads the whole file on every poll, stays cheap.
PROGRESS_EVERY_SECONDS = 30.0


class PhaseProgress:
    """Throttled `avanzamento` lines for a long, otherwise silent stage.

    The population round and the benchmark round each take the better part of
    an hour and print, between them, four lines. `--status` and the dashboard
    could therefore only say "elo: gioco" and leave the operator guessing
    whether the worker was working or wedged -- the 10-minute stale-log warning
    fires routinely inside both -- and could not answer the question actually
    being asked, which is how much is left.

    The round itself reports every session and knows nothing about cadence;
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
    game: GameConfig,
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
        if not member.is_model:
            if on_skip is not None:
                on_skip(Path(member.ref), "not a model")
            continue
        path = Path(registry.directory) / member.ref
        try:
            model, _checkpoint = build_model_from_checkpoint(path, device=device)
        except Exception as exc:  # noqa: BLE001 - see docstring
            if on_skip is not None:
                on_skip(path, str(exc))
            continue
        opponent = policy_opponent(member.label, make_policy_fn(model, device=device), game)
        built[member.label] = opponent
        opponents.append(opponent)
    return opponents


@dataclass(frozen=True)
class TrainConfig:
    hands_per_iteration: int = 256
    opponent_probability: float = 0.5
    # There is deliberately no knob for seating the run's own past selves. The
    # trainer used to freeze a snapshot of the learner every `snapshot_every`
    # iterations and give it a `snapshot_share` of the opponent seats; that was
    # removed at the user's request. A snapshot is a copy of the network being
    # trained, so it drifts with it and anchors nothing, and every seat it took
    # was a seat not facing an independently trained model from the pool.
    gamma: float = 1.0
    lam: float = 0.95
    # None derives big_blind / starting_stack, which puts a stack-sized swing at
    # a value target of ~1.0. See SelfPlayCollector's reward_scale.
    reward_scale: float | None = None


class SelfPlayTrainer:
    def __init__(
        self,
        game_config: GameConfig,
        train_config: TrainConfig | None = None,
        ppo_config: PPOConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        rng: random.Random | None = None,
        model: PokerActorCritic | None = None,
        extra_opponents: Sequence[Opponent] = (),
        registry: PoolRegistry | None = None,
        initial_rating: float = DEFAULT_RATING,
    ) -> None:
        self._game = game_config
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
        self._model = (model if model is not None else PokerActorCritic()).to(device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._ppo.learning_rate)
        self._iteration = 0

        self._pool = OpponentPool(list(extra_opponents))
        self._reward_scale = (
            self._train.reward_scale
            if self._train.reward_scale is not None
            else game_config.big_blind / game_config.starting_stack
        )
        # The closure reads the live model, so updated weights take effect with
        # no rebuilding between iterations.
        self._collector = SelfPlayCollector(
            game_config,
            make_policy_fn(self._model, device=device),
            rng=self._rng,
            opponent_pool=self._pool,
            opponent_probability=self._train.opponent_probability,
            reward_scale=self._reward_scale,
        )

    @property
    def model(self) -> PokerActorCritic:
        return self._model

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
        hands: int = SESSION_HANDS,
        device: str | torch.device | None = None,
    ) -> float:
        """Play the ranked pool, return bb/100, and update everyone's rating.

        `sessions` rated sessions of `hands` hands each, with a fresh draw of
        opponents from the pool and the learner in a fresh seat before every one.
        The default is 10 sessions of 1,000 hands, and both halves of that matter:
        CLAUDE.md measures the bb/100 of a *single* session as having a ~350-570
        bb/100 spread across seeds, so one session could never rank anything,
        while a session shorter than 1,000 hands would be on a different Elo
        scale than every other rated result in the project (see `SESSION_HANDS`).
        The rating accumulates across every validation round of the run and then
        into `rate_against_benchmark`, on one continuous session count.

        Seats are swapped through `SeatProxy` so the same `Table` lives across the
        whole round and the button keeps rotating -- rebuilding it per session
        would reset the button to the same seat every time and hand whoever sits
        there a systematic positional edge.

        **What this number is not.** The rating these sessions build is measured
        against a field of *live* models whose own ratings carry error, and it
        rates a moving target: the weights change between one round and the next,
        so a session played at iteration 100 scored a network that no longer
        exists, and the duplicate-deck duel in CLAUDE.md puts a run's final model
        +56.7 bb/100 ahead of its mid-run self. That is why the validation is
        deliberately the *smaller* half of a run's rated sessions -- 100 against
        500 -- and why the published rating is earned against the pinned anchors.

        **The learner is rated through `DEFAULT_K_SCHEDULE`, on its own session
        count, not at the flat K.** It is not a registry member, so it has no
        `games` the schedule could read and it used to fall through to
        `DEFAULT_K_FACTOR` for every session of every run. At a flat 8 the rating
        could not travel: a session moves it by at most `K * 0.5`, so the 10
        sessions of the first evaluation were worth 40 points against pools whose
        mean sits near 1580, and a simulated learner of *identical* strength to
        its pool finished a whole 40-session run at 1530. That was visible on the
        fleet -- 17 live workers all rated between 1469 and 1534 with bb/100 from
        -686 to +47, which is the rating saying nothing at all. The schedule's
        first tiers (16 below 25 rated sessions, 9 to 75) are what let a run's
        first rounds move the rating at all, and they expire on their own: by the
        fourth round the learner is at 5.5 and falling.

        The returned bb/100 is measured from stack deltas, for the same reason
        `evaluate` does: hands the learner wins without acting produce no
        trajectory and are systematically winning ones.
        """
        if self._registry is None:
            raise ValueError("evaluate_against_pool needs a registry; none was configured")

        device = self._device if device is None else device
        num_players = self._game.num_players
        opponents = registry_opponents(
            self._registry,
            self._game,
            device=device,
            count=max(self._registry.max_models, num_players - 1),
        )
        if len(opponents) < num_players - 1:
            raise ValueError(
                f"the pool can seat only {len(opponents)} opponents, "
                f"a {num_players}-handed table needs {num_players - 1}"
            )

        streams = random.Random(seed)
        rng = random.Random(streams.random())
        bot_rng = random.Random(streams.random())
        seat_rng = random.Random(streams.random())

        proxies = [SeatProxy(f"s{seat}", f"S{seat}") for seat in range(num_players)]
        learners = [
            RLAgentPlayer(
                f"s{seat}",
                LEARNER_LABEL,
                policy_fn=make_policy_fn(self._model, device=device),
                big_blind=self._game.big_blind,
                starting_stack=self._game.starting_stack,
            )
            for seat in range(num_players)
        ]
        table = Table(self._game, list(proxies), rng=rng)

        won = 0
        for _session in range(sessions):
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

            block_delta = [0] * num_players
            for _ in range(hands):
                before = list(table.stacks)
                table.play_hand()
                for seat in range(num_players):
                    block_delta[seat] += table.stacks[seat] - before[seat]
                won += table.stacks[learner_seat] - before[learner_seat]
                table.stacks = [self._game.starting_stack] * num_players

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
            deltas = self._registry.record_session_with_ratings(
                results,
                ratings,
                k_factors={
                    LEARNER_LABEL: k_for_games(self._learner_games, DEFAULT_K_SCHEDULE)
                },
            )
            self._learner_rating += deltas.get(LEARNER_LABEL, 0.0)
            self._learner_games += 1

        return won / self._game.big_blind / (sessions * hands) * 100

    def archive(
        self, path: str | Path, *, iteration: int, metadata: dict | None = None
    ) -> None:
        """Save the current weights, with the learner rating beside them.

        Calling this again with the *same* `path` overwrites in place, which is
        how `main()` keeps only the run's best model. Nothing is added to any
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
        write_sidecar(path, rating=self._learner_rating, iteration=iteration)

    def train_iteration(self) -> dict[str, float]:
        self._iteration += 1
        trajectories = self._collector.collect(
            self._train.hands_per_iteration, gamma=self._train.gamma, lam=self._train.lam
        )
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
                    metadata={"num_players": self._game.num_players},
                )
        return history


# Bumped whenever the *meaning* of a field changes, so an analysis can refuse a
# mixture of schemas rather than average across them.
RUN_METADATA_VERSION = 1

# The axes `poker-loop` draws per worker, in the order a reader wants them: the
# arm first (it says whether the rest can be read as an independent draw), then
# the settings. This is the list the worker *reports*; the ladders themselves
# live in `loop.HP_LADDERS`, and a test requires every ladder axis to appear
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
    "max_grad_norm",
    "entropy_coef",
    "opponent_probability",
    # How strong a field this run drew. Reported like the rest, and worth
    # reading next to the `pool rating: media ...` line printed just below,
    # which is the strength these two actually produced.
    "pool_top_share",
    "pool_top_n",
)


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
        "iterations": args.iterations,
        "hands": args.hands,
        "players": args.players,
        "stack": args.stack,
        "sb": args.sb,
        "bb": args.bb,
        "lr": args.lr,
        "ppo_epochs": args.ppo_epochs,
        "clip_epsilon": args.clip_epsilon,
        "minibatch_size": args.minibatch_size,
        "gae_lambda": args.gae_lambda,
        "value_coef": args.value_coef,
        "max_grad_norm": args.max_grad_norm,
        "entropy_coef": args.entropy_coef,
        "opponent_probability": args.opponent_probability,
        "pool_models": args.pool_models,
        "pool_top_share": args.pool_top_share,
        "pool_top_n": args.pool_top_n,
    }


# ---- the Elo fill-in phase ---------------------------------------------------
#
# A worker that reaches the end of its run while the rest of its generation is
# still training used to exit and leave its core idle until the slowest worker
# finished. With `--hands` now drawn per worker (`loop.hyperparameter_plan`) that
# idle stretch is no longer an accident of scheduling: run lengths are
# *deliberately* ragged, so several workers a generation will finish well early,
# and on a 25-worker box that is hours of cores doing nothing every generation.
#
# **Why spend it on Elo and not on more training.** Two reasons. Training more
# would publish more models, and the store's problem is not that it holds too
# few (~9,600) but that almost none of them have a rating worth anything: a
# round seats ~50 models, so a given one comes up about 0.5% of the time and can
# sit for many generations on the number its own training run published. And a
# longer run is not comparable with the others in its generation, which is the
# whole point of the sweep -- the fast workers are fast because they drew fewer
# hands, and giving those runs extra iterations would erase the very axis being
# measured. Rating rounds cost nothing to the experiment: they touch no
# published model's weights, only what is known about them.
#
# So the fast workers become the fleet's rating engine, which is where the work
# belongs: they are idle, and the ratings are what everything else reads.

# Enough that a generation always gets a rating phase even when every worker
# finishes together, which is the case the user asked to protect: without it,
# a generation whose workers all drew the same `--hands` would do no fill-in at
# all. ~10 sessions a round, so ~5 rounds, ~17 minutes.
DEFAULT_FILL_MIN_SESSIONS = 50
# The safety cap is expressed in **minutes only**, at the user's decision -- a
# maximum number of rounds would be a cap on work, and what actually has to be
# bounded is how long a worker can hold its core while its supervisor waits.
# 150 minutes is the order of one generation, so a worker that never hears from
# its supervisor (a supervisor killed mid-generation, a state directory that
# moved) stops within about the time the generation would have taken anyway
# rather than filling forever.
DEFAULT_FILL_DEADLINE_MINUTES = 150
# **One game per model, against the usual ~55 drawn.** The end-of-run round owes
# 50, which is ~485 sessions and ~90 minutes; a fill-in round is a unit of
# *waiting*, so it has to be short enough that the worker notices the stop flag
# soon after it goes up and short enough to re-draw often. At 1 game each that is
# ~10 sessions, ~10,000 hands and ~3 minutes, and the worker re-reads the ranking
# between rounds, so a long wait is many independent draws rather than one stale
# one. The cost it does not avoid is loading the drawn models: ~203 MB and a few
# seconds every round, which is why the round is not made shorter still.
DEFAULT_FILL_GAMES_PER_MODEL = 1


def ranking_for_draw(global_dir) -> dict[str, dict]:
    """The ratings `tiered_draw` ranks by, read fresh from the shared store."""
    return {
        label: {"rating": member.rating, "games": member.games}
        for label, member in load_ranking(global_dir).members.items()
    }


def run_elo_fill_in(args, game) -> tuple[int, int]:
    """Play rating rounds until the rest of the generation catches up.

    Returns `(rounds, sessions)`. Stops on the first of: the supervisor's stop
    flag (but never before `--fill-min-sessions`), the deadline, or a round that
    plays nothing at all -- a store too small to seat a table would otherwise
    spin.

    **Pruning is off and the draw is biased, and the two go together.** These
    rounds lift `trigger_size` to `NO_PRUNE_TRIGGER` and draw through
    `tiered_draw`, which gives a quarter of the seats to each of ranks 1-10,
    11-100, 101-1,000 and the rest. That is the right place to spend a waiting
    worker's time -- the top is the only part of the ranking anything reads,
    `pick_parents` draws from the best 100, and the ordering there was measured
    wrong. Eligibility for deletion is a percentile of `games`, so a biased draw
    makes the often-seated eligible sooner and leaves the rarely-drawn tail
    permanently immune: a fill-in round that could prune would eat the middle of
    the population instead of its bottom, and it cannot.
    """
    from pokerlab.rl.global_arena import (
        NO_PRUNE_TRIGGER,
        run_population_round,
        tiered_draw,
    )

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    draining = scratch / FILL_DRAINING_FILENAME
    stop_file = Path(args.fill_stop_file) if args.fill_stop_file else None
    deadline = time.monotonic() + max(0.0, args.fill_deadline_minutes) * 60.0

    announce(ELO_FILL)
    # Written *before* the first round, so the supervisor learns this worker is
    # only waiting as soon as it is true. Writing it after would have the
    # supervisor hold the whole generation for a worker already filling in.
    draining.touch()
    rounds = sessions = 0
    try:
        while time.monotonic() < deadline:
            if sessions >= args.fill_min_sessions and stop_file is not None and stop_file.exists():
                break
            # Re-read every round: the previous one just moved the ratings the
            # bias is computed from, and other machines moved them too.
            ratings = ranking_for_draw(args.global_dir)
            report = run_population_round(
                global_dir=args.global_dir,
                root=args.global_root,
                game=game,
                machine=args.machine,
                population_sample=args.global_sample,
                benchmark_sample=args.global_benchmark_sample,
                games_per_model=args.fill_games_per_model,
                hands_per_game=args.global_hands_per_game,
                device=args.device,
                lock_ttl=args.global_lock_seconds,
                trigger_size=NO_PRUNE_TRIGGER,
                draw=tiered_draw(ratings),
                on_skip=lambda path, why: print(f"  riempimento elo, saltato {path}: {why}"),
            )
            if report.played == 0:
                print("riempimento elo: popolazione insufficiente per un tavolo, esco")
                break
            rounds += 1
            # `sessions_played`, not `sessions`: the latter counts what this
            # merge folded in from every machine, which on a busy fleet is
            # thousands and would satisfy the minimum on the first round without
            # this worker having played anything.
            sessions += report.sessions_played
            remaining = deadline - time.monotonic()
            # No ETA on purpose: the number this worker knows is its safety cap,
            # and reporting that as "time left" would make the generation's
            # `fine :` line quote hours for the one worker nobody is waiting on
            # (see `monitor.format_finishing_line`).
            print(
                progress_marker(
                    ELO_FILL,
                    min(sessions, args.fill_min_sessions),
                    args.fill_min_sessions,
                    detail=f"giro {rounds}, {sessions} sessioni giocate, "
                    f"tetto fra {max(0, round(remaining / 60))}m",
                ),
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - the run is already complete and
        # published by this point; a failure here must cost the extra rounds and
        # nothing else, exactly as the end-of-run round's own guard does.
        print(f"riempimento elo interrotto per errore: {exc}")
    finally:
        # Always, including on the error path: a marker left behind would have
        # the next generation's supervisor read a stale directory as a worker
        # that is already draining.
        draining.unlink(missing_ok=True)
    print(f"riempimento elo: {rounds} giri, {sessions} sessioni giocate")
    return rounds, sessions


def inherited_rating(path: str | Path, *, default: float = DEFAULT_RATING) -> float:
    """The rating a resumed parent was published with, or `default`.

    A run that inherits its weights inherits its starting rating too: the child
    is that parent plus some training, not an unknown quantity, so starting it at
    the 1500 baseline would spend its first rated sessions re-discovering where
    its own lineage already sits. See `SelfPlayTrainer.__init__` for why this does
    not compound an ancestor's error (the prior is weak by construction) and
    `pool_registry.DEFAULT_K_SCHEDULE` for the offset that encodes how weak.

    `publish_rating` is what a model was registered with -- the number the global
    ranking holds for it -- and every model published since the metadata existed
    carries it, so this works on the store as it stands today with no migration.
    `pool_rating` is the fallback for the few that predate it: worse, being
    measured against one run's drawn pool, but far better than the baseline.
    Anything older, or unreadable, falls through to `default`, which is the same
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a poker agent with self-play PPO.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--hands", type=int, default=256, help="hands collected per iteration")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    # The two PPO knobs worth sweeping, exposed so a per-worker draw can reach
    # them. `epochs x clip_epsilon` is the classic stability trade -- more
    # passes over the same batch move further from the policy that collected
    # it, and the clip is what bounds how far -- and it is a likelier
    # explanation than the learning rate for the `kl` of 0.11-0.15 measured at
    # iteration 1 in from-scratch runs, against a "step too large" threshold of
    # 0.05. Adding a flag is safe; removing one is what takes the fleet down
    # (see "Never delete a `poker-train` CLI flag while a supervisor is
    # running"), so these are cheap to try and permanent to keep.
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
        "--max-grad-norm", type=float, default=PPOConfig.max_grad_norm,
        help="gradient clipping norm",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=PPOConfig.entropy_coefficient,
        help="weight of the entropy bonus in the PPO loss. Zero by default: a "
        "bonus here buys per-decision dithering, not different strategies, since "
        "it perturbs each decision independently while a bluff is a sequence. See "
        "PPOConfig.entropy_coefficient for the measurements behind that",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/agent.pt"))
    parser.add_argument("--resume", action="store_true", help="load --checkpoint before training")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="the shared store of every trained model: opponents are drawn from "
        "here and this run's best model is published here when it ends",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_SCRATCH_DIR,
        help="where this run keeps its latest checkpoint until it is published",
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
        "--archive-every",
        type=int,
        default=25,
        help="iterations between checks of whether the current model is this "
        "run's best so far; only the best is kept on disk, overwritten in "
        "place as a better one appears (0 disables archiving entirely)",
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--hp-arm", default="",
        help="which hyperparameter arm `poker-loop` put this worker in "
        "(\"sampled\" or \"inherited\"); recorded in the published model and "
        "never acted on, so an analysis can tell the two apart",
    )
    parser.add_argument(
        "--eval-every", type=int, default=100,
        help="iterations between validation rounds (0 disables)",
    )
    parser.add_argument(
        "--eval-sessions", type=int, default=DEFAULT_EVAL_SESSIONS,
        help=f"rated sessions per validation round, each {SESSION_HANDS} hands "
        "with opponents re-drawn from the pool. Over a 1000-iteration run at "
        "--eval-every 100 that is 100 rated sessions, which then continue into "
        "the round against the frozen anchors on one session count",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR,
        help="the frozen opponent set the benchmark round rates this run's "
        "published model against (--benchmark-sessions) -- never seated in training",
    )
    parser.add_argument(
        "--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR,
        help="the cross-machine population Elo registry (rl/global_arena.py)",
    )
    parser.add_argument(
        "--global-root", type=Path, default=Path("checkpoints"),
        help="the checkpoint root: the model store (models/) and the benchmark "
        "sets (benchmark/) the population rounds draw from are found under it",
    )
    parser.add_argument("--global-sample", type=int, default=DEFAULT_POPULATION_SAMPLE)
    parser.add_argument("--global-benchmark-sample", type=int, default=DEFAULT_BENCHMARK_SAMPLE)
    parser.add_argument(
        "--global-games-per-model", type=int, default=DEFAULT_GAMES_PER_MODEL,
        help="rated sessions each drawn model owes per population round; the "
        "round's cost is very nearly linear in it (~1.1 s per session)",
    )
    parser.add_argument("--global-hands-per-game", type=int, default=DEFAULT_HANDS_PER_GAME)
    parser.add_argument(
        "--benchmark-sessions", type=int, default=DEFAULT_BENCHMARK_SESSIONS,
        help=f"rated sessions of {SESSION_HANDS} hands this run's model plays "
        "against opponents drawn at random from --benchmark-dir before it is "
        "published (0 disables). The rating it earns there is what it is "
        "published with, and its session count continues the one the validation "
        "rounds built",
    )
    parser.add_argument(
        "--global-lock-seconds", type=int, default=DEFAULT_LOCK_SECONDS,
        help="how long a per-model lock lives before it counts as abandoned",
    )
    parser.add_argument("--global-trigger-size", type=int, default=DEFAULT_POPULATION_TRIGGER)
    parser.add_argument(
        "--global-eliminate-fraction", type=float, default=DEFAULT_ELIMINATION_FRACTION
    )
    parser.add_argument(
        "--global-protect-percentile", type=float, default=DEFAULT_PROTECT_PERCENTILE
    )
    # Accepted and ignored. A `poker-loop` supervisor is a process that can run
    # for weeks -- the fleet's have been up since generation 1 -- and it forwards
    # the flag list *its own* code knew about when it started. So removing a flag
    # from `poker-train` does not merely retire it: it kills every worker of
    # every running supervisor at that supervisor's next generation, because
    # argparse exits 2 on an unrecognised argument. The loop then spins through
    # generations in ~90 s each, producing nothing, while `--status` still says
    # "training".
    #
    # That is not hypothetical. On 2026-09-28 at 20:47 `--global-benchmark-refresh`
    # was removed from here; five machines ran their last real generation between
    # 23:08 and 00:29 and then burned ~250 empty generations each overnight.
    # `--pool-random-share` and `--pool-fresh-n` had been retired earlier and were
    # in the same forwarded list.
    #
    # **The rule this encodes: a flag removed from `poker-train` stays parseable
    # until every supervisor in the fleet has been restarted.** The value is
    # discarded, so the behaviour is the new behaviour -- only the parsing is
    # backwards compatible. Delete these once no supervisor old enough to pass
    # them is left running (`poker-loop --status` shows each machine's
    # generation, and a restart resets it to 1).
    #
    # `--benchmark-every`/`--benchmark-hands`/`--benchmark-seed` are the live
    # per-worker benchmark, removed at the user's request because the reading was
    # no longer wanted and the hands it played cost every worker time.
    # `--benchmark-dir` is *not* here: it is still a working flag, because the
    # frozen set it names is what the benchmark round rates a published model
    # against. `--self-share` is the share of opponent seats that went to this
    # run's own frozen snapshots: the snapshots are gone, so the value is now
    # discarded, but a supervisor started before that change still passes it.
    for retired in (
        "--pool-random-share",
        "--pool-fresh-n",
        "--global-benchmark-refresh",
        "--benchmark-every",
        "--benchmark-hands",
        "--benchmark-seed",
        "--self-share",
    ):
        parser.add_argument(retired, help=argparse.SUPPRESS)

    parser.add_argument(
        "--global-round", dest="global_round", action="store_true", default=True,
        help="after this run finishes, try one cross-machine population Elo round "
        "(on by default)",
    )
    parser.add_argument(
        "--no-global-round", dest="global_round", action="store_false",
        help="disable the end-of-run population round entirely",
    )
    # Off by default, and that is the right way round: a hand-run `poker-train`
    # should end when it ends, not sit for two hours playing rating rounds. Only
    # `poker-loop` turns it on, because only a supervisor has other workers to
    # wait for and a stop flag to tell this one when they are done.
    parser.add_argument(
        "--elo-fill-in", dest="elo_fill_in", action="store_true", default=False,
        help="after the end-of-run round, keep playing rating rounds until the "
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
        "--fill-min-sessions", type=int, default=DEFAULT_FILL_MIN_SESSIONS,
        help="sessions this worker plays before the stop file can end the "
        "phase, so a generation whose workers all finish together still gets a "
        "rating phase",
    )
    parser.add_argument(
        "--fill-deadline-minutes", type=float, default=DEFAULT_FILL_DEADLINE_MINUTES,
        help="hard cap on the phase, in minutes; the only cap there is",
    )
    parser.add_argument(
        "--fill-games-per-model", type=int, default=DEFAULT_FILL_GAMES_PER_MODEL,
        help="rated sessions each drawn model owes per fill-in round; small on "
        "purpose, so the worker re-draws often and stops soon after being told to",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    game = GameConfig(
        num_players=args.players, starting_stack=args.stack, small_blind=args.sb, big_blind=args.bb
    )

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
            game,
            device=args.device,
            count=args.pool_models,
            on_skip=lambda path, why: print(f"skipping {path.name}: {why}"),
        )
        if args.pool_models > 0
        else []
    )
    trainer = SelfPlayTrainer(
        game,
        TrainConfig(
            hands_per_iteration=args.hands,
            opponent_probability=args.opponent_probability,
            lam=args.gae_lambda,
        ),
        PPOConfig(
            learning_rate=args.lr,
            entropy_coefficient=args.entropy_coef,
            epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            minibatch_size=args.minibatch_size,
            value_coefficient=args.value_coef,
            max_grad_norm=args.max_grad_norm,
        ),
        device=args.device,
        rng=random.Random(args.seed),
        extra_opponents=archived,
        registry=registry,
        initial_rating=inherited_rating(args.checkpoint) if args.resume else DEFAULT_RATING,
    )
    if args.resume:
        load_checkpoint(args.checkpoint, trainer.model, device=args.device)
        print(
            f"resumed from {args.checkpoint} "
            f"(rating ereditato {trainer.learner_rating:.0f})"
        )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)
    archive_path = Path(args.scratch_dir) / f"agent-{run_id}.pt"
    # Not "the best so far" any more: the run archives its *latest* checkpoint,
    # so these carry whatever the last archive was rated at. See the archiving
    # block below for why the best-rated rule was dropped.
    archived_rating = float("-inf")
    archived_iteration = 0
    print(f"device {args.device} | {args.players} seats | {args.hands} hands/iteration")
    # Built from `run_metadata`, not from `args` directly, so the line in the log
    # and the metadata inside the published checkpoint cannot disagree about what
    # this run was.
    recorded = run_metadata(args)
    print(
        hyperparameters_marker({axis: recorded[axis] for axis in REPORTED_AXES if axis in recorded}),
        flush=True,
    )
    models = registry.models()
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
        iteration = int(stats["iteration"])
        if args.eval_every and (
            iteration % args.eval_every == 0
            # The last iteration always evaluates, whatever the interval: this
            # reading is the run's final word on the learner, and it is also the
            # rating the model is published with whenever the benchmark round
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
                )
                print(
                    f"        eval vs pool: {win_rate:+.1f} bb/100  "
                    f"rating {trainer.learner_rating:.0f}"
                )
            save_checkpoint(
                args.checkpoint, trainer.model, trainer.optimizer, iteration=trainer.iteration
            )
            print(f"        saved {args.checkpoint}")

        if args.archive_every and (
            iteration % args.archive_every == 0
            # The last iteration always archives, whatever the interval. The
            # benchmark round below measures `trainer.model` -- the live
            # weights -- and publishes the rating it earns, so the archive has
            # to *be* those weights or the number would describe a different
            # model than the one that goes into the store.
            or iteration == args.iterations
        ):
            # **The latest checkpoint wins, not the best-rated one.** It used to
            # keep whichever had the highest `learner_rating`, which sounds
            # obviously right and measured badly: the rating comes from 500 hands
            # against this run's own drawn pool, and across 27,610 real runs it
            # called the mid-run model better than the final one 47% of the time
            # -- a coin flip. Meanwhile a duplicate-deck duel of the iteration-100
            # model against the iteration-50 model *from the same seed* put the
            # later one ahead in all 6 seeds, by +56.7 bb/100 (t = 7.5).
            #
            # So the choice is between a reliable prior (training improves the
            # model) and an unreliable measurement (this rating). Picking by the
            # rating captures only 30.1 of those 56.7 bb/100; always taking the
            # last captures all of it. A measurement would have to be right ~95%
            # of the time to beat the blind rule, which this one is not.
            #
            # Replace this with a *good* measurement when there is one -- the
            # benchmark bb/100 is deterministic and scored against models never
            # seated in training, so selecting on it would also catch the runs
            # where the later model genuinely is worse. That needs the frozen set
            # rebuilt first (it is down to 5 models).
            archived_rating = trainer.learner_rating
            archived_iteration = iteration
            trainer.archive(archive_path, iteration=iteration, metadata=run_metadata(args))
            print(f"        archived {archive_path} (iterazione {iteration}, "
                  f"rating {trainer.learner_rating:.0f})")

    # **The benchmark round runs before publishing, and its rating is what the
    # model is published with.** The alternative -- publish first, on the rating
    # `evaluate_against_pool` measured -- puts a model into the global ranking on
    # a number earned against the 50 opponents this one run happened to draw,
    # which is the measurement shown to get the order at the top of the ranking
    # outright wrong (see "The ranking is wrong at the top"). The frozen anchors
    # are a fixed scale and are never seated in training, so the rating they give
    # means the same thing for every model on every machine.
    #
    # **These sessions are deliberately NOT queued for the global merge.** They
    # are already in the rating the model is published with, and
    # `_apply_session` would re-apply the identical evidence a second time from
    # the member's own games count -- measured as 10-20% worse than applying it
    # once, because re-using the same outcomes cannot add information and does
    # add movement. The model is registered with the rating *and* the session
    # count this round earned (`games_after`), which is what makes the local
    # computation authoritative rather than a report.
    published = None
    if archive_path.exists():
        prefix = f"{args.archive_prefix}-" if args.archive_prefix else ""
        label = f"{args.machine}-{prefix}{archive_path.stem}"
        publish_rating = archived_rating
        publish_games = trainer.learner_games
        benchmark_bb100: float | None = None
        benchmark_hands = 0

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
                game,
                label=label,
                rating=archived_rating,
                games=trainer.learner_games,
                anchor_ratings=anchors,
                sessions=args.benchmark_sessions,
                device=args.device,
                seed=args.seed or 0,
                on_skip=lambda path, why: print(f"  benchmark, saltato {path}: {why}"),
                on_progress=PhaseProgress(SERIES),
            )
            if rated is not None:
                publish_rating = rated.rating_after
                publish_games = rated.games_after
                benchmark_bb100, benchmark_hands = rated.bb_per_100, rated.hands
                print(
                    f"        benchmark: {rated.bb_per_100:+.1f} bb/100 su "
                    f"{rated.sessions} sessioni contro {rated.opponents} ancore, "
                    f"rating {publish_rating:.0f} "
                    f"(era {archived_rating:.0f} contro il pool, "
                    f"{publish_games} partite)"
                )

        # The outcomes only exist now, after the benchmark round, so the archive
        # is rewritten once with them before it is copied into the store. Same
        # weights (`trainer.model` is what the round just measured, and what the
        # last archive holds), one 3 MB write, and the published model then
        # carries both what it was configured with and what that produced --
        # which is what makes a population of runs analysable at all.
        #
        # `benchmark_bb100` is the response variable to read: 500 sessions of
        # 1,000 hands against the frozen anchors, which are never seated in
        # training and whose ratings are pinned, so 500,000 hands and a 95%
        # interval of roughly +/-8 bb/100. It replaced `series_bb100`, which was
        # the same quantity broken down by series; the breakdown went with the
        # series and the precision did not change, because the hand count did not.
        trainer.archive(
            archive_path,
            iteration=archived_iteration,
            metadata={
                **run_metadata(args),
                "pool_rating": archived_rating,
                "publish_rating": publish_rating,
                "publish_games": publish_games,
                "benchmark_bb100": benchmark_bb100,
                "benchmark_hands": benchmark_hands,
            },
        )
        published = publish_model(
            archive_path,
            models_dir=args.models_dir,
            global_dir=args.global_dir,
            name=f"{args.machine}-{prefix}{archive_path.name}",
            rating=publish_rating,
            games=publish_games,
            iteration=archived_iteration,
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

    if args.global_round:
        try:
            from pokerlab.rl.global_arena import run_population_round, tiered_draw

            report = run_population_round(
                global_dir=args.global_dir,
                root=args.global_root,
                game=game,
                machine=args.machine,
                population_sample=args.global_sample,
                benchmark_sample=args.global_benchmark_sample,
                games_per_model=args.global_games_per_model,
                hands_per_game=args.global_hands_per_game,
                device=args.device,
                lock_ttl=args.global_lock_seconds,
                draw=tiered_draw(ranking_for_draw(args.global_dir)),
                trigger_size=args.global_trigger_size,
                eliminate_fraction=args.global_eliminate_fraction,
                protect_percentile=args.global_protect_percentile,
                on_skip=lambda path, why: print(f"  global round, saltato {path}: {why}"),
                on_phase=announce,
                on_progress=PhaseProgress(ELO_PLAY),
            )
        except Exception as exc:  # noqa: BLE001 - a shared cross-machine round
            # must never turn an otherwise-successful training run into a
            # failed process exit; the next run, here or elsewhere, just
            # tries again.
            print(f"global round saltato per errore: {exc}")
        else:
            if report.sessions == 0 and report.deferred_sessions == 0:
                print("global round: popolazione insufficiente per un tavolo, salto questo giro")
                if report.ghosts_dropped:
                    print(f"  ripulite {report.ghosts_dropped} voci fantasma dal registro")
            else:
                line = (
                    f"global round: {report.sessions} sessioni, "
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
                lock_ttl=args.global_lock_seconds,
                on_skip=lambda path, why: print(f"  benchmark, saltato {path}: {why}"),
            ):
                print(
                    f"benchmark: aggiunto {member.label} "
                    f"(rating {member.rating:.0f}, {member.games} partite)"
                )
        except Exception as exc:  # noqa: BLE001 - same reasoning as the round above
            print(f"aggiunta al benchmark saltata per errore: {exc}")

    # Only after the run's own round: the fill-in rounds are extra, and a worker
    # whose own results were not merged has nothing to fill in *for*.
    if args.elo_fill_in:
        run_elo_fill_in(args, game)

    # Whatever happened above -- a round skipped, an error absorbed, no round
    # asked for -- the run got to its end; the watcher's "still working" ends here.
    announce(DONE)


if __name__ == "__main__":
    main()
