"""A frozen opponent set, for measuring whether the loop actually improves.

The ratings in `pool_registry` answer "is this model better than the others in
the pool right now". They cannot answer "is generation 40 better than generation
1", because the pool they are measured against is replaced as training goes on:
a rating that holds steady at 1600 across forty generations could mean no
progress at all, or it could mean the agent improved exactly as fast as its
opposition. A rating is relative by construction.

A benchmark fixes the opposition instead. The same set of models, the same deals,
the same seats, every generation -- so the only thing that changes between two
measurements is the model being measured.

Two properties make the comparison honest, and both are easy to lose:

- **The benchmark models must never be seated in training.** Otherwise the agent
  is trained on its own test set and the number measures memorisation of
  specific opponents rather than strength. `poker-loop` keeps them in a separate
  directory that the training pool never reads.
- **Everything else must be held fixed.** `Table` consumes its `random.Random`
  only to shuffle the deck, exactly once per hand, so a fixed seed deals an
  identical sequence of hands no matter how the betting goes -- and therefore no
  matter which model is being benchmarked. Seats and opponent draws are assigned
  by block index rather than sampled, and torch's global RNG is reseeded, so two
  runs of the same model return the same number.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.policy import PokerActorCritic, make_policy_fn
from pokerlab.rl.pool_registry import (
    DEFAULT_K_SCHEDULE,
    k_for_games,
    pairwise_elo_delta,
)
from pokerlab.rl.pool_registry import DEFAULT_RATING as DEFAULT_ANCHOR_RATING
from pokerlab.rl.ppo import build_model_from_checkpoint
from pokerlab.rl.rollout import Opponent, SeatProxy, policy_opponent

DEFAULT_BENCHMARK_DIR = Path("checkpoints/benchmark")
DEFAULT_BENCHMARK_HANDS = 3000
DEFAULT_ROTATE_EVERY = 25


@dataclass(frozen=True)
class BenchmarkResult:
    bb_per_100: float
    hands: int
    won_chips: int
    opponents: int
    per_opponent: dict[str, float] = field(default_factory=dict)


def load_benchmark_opponents(
    directory: str | Path,
    game: GameConfig,
    *,
    device: str | torch.device = "cpu",
    on_skip: Callable[[Path, str], None] | None = None,
    recursive: bool = True,
) -> list[Opponent]:
    """Every checkpoint in the directory, in a fixed order.

    Sorted by file name rather than by mtime: the benchmark's composition and
    the order opponents are seated in must not depend on when the files happened
    to be written, or copying the directory would change the measurement.

    The root benchmark directory is a container for numbered `benchmark_<N>/`
    series directories. The default is therefore recursive so a benchmark run
    sees the whole frozen benchmark set, not just whatever blessed files happen
    to sit directly in the root. The explicit `recursive=False` switch keeps the
    root-only behaviour available for callers that really want a non-growing,
    static set.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    opponents: list[Opponent] = []
    paths = directory.rglob("*.pt") if recursive else directory.glob("*.pt")
    for path in sorted(paths):
        try:
            model, _checkpoint = build_model_from_checkpoint(path, device=device)
        except Exception as exc:  # noqa: BLE001 - a user-owned directory
            if on_skip is not None:
                on_skip(path, str(exc))
            continue
        opponents.append(policy_opponent(path.stem, make_policy_fn(model, device=device), game))
    return opponents


def run_benchmark(
    model: PokerActorCritic,
    opponents: Sequence[Opponent],
    game: GameConfig,
    *,
    hands: int = DEFAULT_BENCHMARK_HANDS,
    seed: int = 12345,
    rotate_every: int = DEFAULT_ROTATE_EVERY,
    device: str | torch.device = "cpu",
) -> BenchmarkResult:
    """Play `model` against the frozen set and report bb/100.

    Measured from stack deltas rather than from collected trajectories, for the
    same reason `evaluate` is: hands the agent wins without ever acting produce
    no trajectory and are systematically winning ones.

    Deterministic by construction -- the deal seed, the seat rotation and the
    opponent draw are all functions of the block index, and torch is reseeded --
    so a difference between two generations is a difference between two models,
    not between two samples.
    """
    if len(opponents) < game.num_players - 1:
        raise ValueError(
            f"the benchmark has {len(opponents)} opponents, a "
            f"{game.num_players}-handed table needs {game.num_players - 1}"
        )

    torch.manual_seed(seed)
    num_players = game.num_players
    proxies = [SeatProxy(f"s{seat}", f"S{seat}") for seat in range(num_players)]
    learners = [
        RLAgentPlayer(
            f"s{seat}",
            "benchmarked",
            policy_fn=make_policy_fn(model, device=device),
            big_blind=game.big_blind,
            starting_stack=game.starting_stack,
        )
        for seat in range(num_players)
    ]
    table = Table(game, list(proxies), rng=random.Random(seed))
    bot_rng = random.Random(seed + 1)

    won = 0
    played = 0
    block_index = 0
    faced: dict[str, list[int]] = {}

    while played < hands:
        block = min(rotate_every, hands - played)
        learner_seat = block_index % num_players
        # A deterministic walk through the fixed set rather than a sample: every
        # generation must face the same opponents in the same seats.
        draw = [
            opponents[(block_index * (num_players - 1) + offset) % len(opponents)]
            for offset in range(num_players - 1)
        ]

        seated_labels: list[str] = []
        draw_iter = iter(draw)
        for seat, proxy in enumerate(proxies):
            if seat == learner_seat:
                proxy.inner = learners[seat]
                proxy.name = "benchmarked"
            else:
                opponent = next(draw_iter)
                proxy.inner = opponent.factory(proxy.player_id, opponent.label, bot_rng)
                proxy.name = opponent.label
                seated_labels.append(opponent.label)

        block_won = 0
        for _ in range(block):
            before = table.stacks[learner_seat]
            table.play_hand()
            block_won += table.stacks[learner_seat] - before
            table.stacks = [game.starting_stack] * num_players
        won += block_won
        played += block
        block_index += 1

        # Credited to every opponent at the table, since a hand is played
        # against the field rather than against one seat.
        for label in seated_labels:
            faced.setdefault(label, []).append(block_won)

    per_opponent = {
        label: sum(blocks) / game.big_blind / (len(blocks) * rotate_every) * 100
        for label, blocks in faced.items()
    }
    return BenchmarkResult(
        bb_per_100=won / game.big_blind / hands * 100,
        hands=hands,
        won_chips=won,
        opponents=len(opponents),
        per_opponent=per_opponent,
    )




# ---- rating a fresh model against the frozen anchors ------------------------

# **500 sessions of 1,000 hands, drawn at random from the whole frozen set.**
# There is deliberately no notion of a *series* here any more: this round is a
# measurement of one number -- the rating a model is published with -- and
# breaking it down by which era of the population a model beats was information
# nobody acted on, bought at the price of every session being played against a
# field chosen for its label rather than for what it tells us.
#
# **Why 500, and why the count continues from training.** A 1000-hand session
# measures a rating with a standard deviation of 238 points (see
# `pool_registry.DEFAULT_K_SCHEDULE` for where that comes from), so precision
# improves only as `1/sqrt(sessions)`: a run's 100 validation sessions leave a
# 95% band of about +/-42 points, 500 more here take it to **+/-20**, and halving
# that again to +/-10 would cost 2,000 (11 hours a run, four times the hands).
# 500 is the point where the round costs what the old 11-series arrangement cost
# -- 2.8 hours against 3.1 -- so the fleet's model production rate is unchanged
# while the published rating improves from about +/-25 to +/-20. It is also
# 500,000 hands, which keeps `benchmark_bb100` at the same +/-8 precision the
# hyperparameter sweep already reads it with.
#
# **The sessions are rated on the learner's own running count**
# (`games`, continued from `SelfPlayTrainer.learner_games`), through
# `DEFAULT_K_SCHEDULE`, not at a flat K. A flat 8 -- what this round used to use
# -- settles at a jitter of +/-35 points whatever the session count, because a
# fixed K is an exponential average with a fixed effective window: more sessions
# buy nothing at all. The schedule's falling K is what turns 500 sessions into
# 500 sessions' worth of evidence.
#
# **Anchors are loaded in rotating slices, not all at once.** The frozen set is
# 55 models today and only grows; holding all of them resident would cost ~440 MB
# per worker, and every worker of a generation reaches this phase at about the
# same moment, so 25 of them would want ~11 GB of transient allocation together
# -- the profile of the unexplained fleet incident in CLAUDE.md. A slice of
# `resident` anchors is drawn at random, played for `rotate_every` sessions and
# dropped, so the resident cost is ~80 MB and the round still faces the whole set.
DEFAULT_BENCHMARK_SESSIONS = 500
DEFAULT_SESSION_HANDS = 1000
DEFAULT_RESIDENT_ANCHORS = 10
DEFAULT_ANCHOR_ROTATE_EVERY = 50


@dataclass(frozen=True)
class BenchmarkRating:
    """What the round against the frozen anchors concluded about one model."""

    bb_per_100: float
    sessions: int
    hands: int
    # Distinct anchors that actually took a seat, across every slice.
    opponents: int
    rating_after: float
    # The learner's session count after the round: what the model is registered
    # with, so a population round rates it at the experience it really has.
    games_after: int
    # Chip deltas per participant, one dict per session, in the shape the global
    # merge consumes (`global_arena._apply_session`).
    raw_sessions: list[dict[str, float]] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)


def anchor_paths(directory: str | Path) -> list[Path]:
    """Every frozen checkpoint under `directory`, recursively, sorted.

    Sorted so a seeded draw is reproducible: `rglob` order is filesystem order
    and differs between machines sharing the volume.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("*.pt"))


def rate_against_benchmark(
    model: PokerActorCritic,
    directory: str | Path,
    game: GameConfig,
    *,
    label: str,
    rating: float,
    games: int = 0,
    anchor_ratings: dict[str, float] | None = None,
    sessions: int = DEFAULT_BENCHMARK_SESSIONS,
    hands: int = DEFAULT_SESSION_HANDS,
    resident: int = DEFAULT_RESIDENT_ANCHORS,
    rotate_every: int = DEFAULT_ANCHOR_ROTATE_EVERY,
    schedule: Sequence[tuple[int, float]] = DEFAULT_K_SCHEDULE,
    device: str | torch.device = "cpu",
    seed: int = 0,
    on_skip: Callable[[Path, str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> BenchmarkRating | None:
    """Play a freshly trained model against the frozen anchors and rate it.

    This is the number a model is published with. A model enters the global
    ranking with the rating its own run measured and only earns real games when a
    population round happens to draw it -- about a 0.5% chance per round out of
    ~9,600 models -- so without this round it could sit on an unearned number for
    many generations. The anchors are the right opponents for it: their ratings
    are pinned, so the result is measured against a fixed scale rather than a
    moving field, and they are never seated in training, so it is not
    memorisation.

    `games` continues the learner's own rated-session count from training, and
    every session is rated at `k_for_games` of the count it had when it sat down
    -- the same rule `record_session_with_ratings` applies to a registry member.

    Returns `None` when the frozen set cannot seat a table at all, which is not
    an error: a fresh installation has no anchors yet and the run still publishes
    at the rating its validation measured.

    `raw_sessions` carries the per-session chip deltas in the exact shape
    `global_arena` merges, for a caller that wants to queue them; `main()`
    deliberately does not, because these sessions are already in the rating and
    merging them would apply the same evidence twice.
    """
    anchor_ratings = anchor_ratings or {}
    seats = game.num_players
    available = anchor_paths(directory)
    if len(available) < seats - 1:
        if on_skip is not None:
            on_skip(
                Path(str(directory)),
                f"solo {len(available)} ancore, un tavolo da {seats} ne chiede {seats - 1}",
            )
        return None

    rng = random.Random(seed)
    bot_rng = random.Random(rng.random())
    proxies = [SeatProxy(f"s{seat}", f"S{seat}") for seat in range(seats)]
    learners = [
        RLAgentPlayer(
            f"s{seat}", label,
            policy_fn=make_policy_fn(model, device=device),
            big_blind=game.big_blind, starting_stack=game.starting_stack,
        )
        for seat in range(seats)
    ]
    table = Table(game, list(proxies), rng=rng)

    current = float(rating)
    played_games = int(games)
    won = 0
    raw: list[dict[str, float]] = []
    paths: dict[str, str] = {}
    seen: set[str] = set()
    slice_size = max(seats - 1, min(resident, len(available)))
    rotate = max(1, rotate_every)
    opponents: list[Opponent] = []
    played = 0

    while played < sessions:
        if played % rotate == 0 or len(opponents) < seats - 1:
            # A fresh random slice: loaded, played, and dropped before the next
            # one, which is what keeps the resident cost at one slice.
            if len(available) < seats - 1:
                break
            opponents = []
            slice_size = max(seats - 1, min(resident, len(available)))
            for path in rng.sample(available, slice_size):
                try:
                    loaded, _checkpoint = build_model_from_checkpoint(path, device=device)
                except Exception as exc:  # noqa: BLE001 - a user-owned directory
                    # Struck off, not merely skipped: a bad file must be tried
                    # once, not once per slice, or a round of ten slices reports
                    # the same failure ten times and can spin if a whole slice
                    # happens to be bad.
                    available = [other for other in available if other != path]
                    if on_skip is not None:
                        on_skip(path, str(exc))
                    continue
                opponents.append(
                    policy_opponent(path.stem, make_policy_fn(loaded, device=device), game)
                )
                paths[path.stem] = str(path)
                seen.add(path.stem)
            if len(opponents) < seats - 1:
                continue

        learner_seat = rng.randrange(seats)
        drawn = rng.sample(opponents, seats - 1)
        occupants: dict[int, str] = {}
        draw_iter = iter(drawn)
        for seat, proxy in enumerate(proxies):
            if seat == learner_seat:
                proxy.inner = learners[seat]
                proxy.name = label
                occupants[seat] = label
            else:
                opponent = next(draw_iter)
                proxy.inner = opponent.factory(proxy.player_id, opponent.label, bot_rng)
                proxy.name = opponent.label
                occupants[seat] = opponent.label

        deltas = [0] * seats
        for _hand in range(hands):
            before = list(table.stacks)
            table.play_hand()
            for seat in range(seats):
                deltas[seat] += table.stacks[seat] - before[seat]
            table.stacks = [game.starting_stack] * seats
        won += deltas[learner_seat]

        session: dict[str, float] = {}
        for seat, who in occupants.items():
            session[who] = session.get(who, 0.0) + deltas[seat]
        raw.append(session)

        ratings = {
            who: current if who == label else anchor_ratings.get(who, DEFAULT_ANCHOR_RATING)
            for who in session
        }
        # Read before the increment: a session is rated at the experience its
        # player had when it sat down.
        k = k_for_games(played_games, schedule)
        current += pairwise_elo_delta(session, ratings, k_factor=k).get(label, 0.0)
        played_games += 1
        played += 1
        if on_progress is not None:
            # The running figures ride in the detail so the watcher can show
            # them while the round is still playing: until it ends, the
            # `ancore/100` and `elo ancore` cells would otherwise stay empty
            # for hours. Read by `monitor._SERIES_DETAIL`.
            running = won / game.big_blind / (played * hands) * 100
            on_progress(played, sessions, f"rating {current:.0f}, {running:+.1f} bb/100")

    if played == 0:
        return None
    return BenchmarkRating(
        bb_per_100=won / game.big_blind / (played * hands) * 100,
        sessions=played,
        hands=played * hands,
        opponents=len(seen),
        rating_after=current,
        games_after=played_games,
        raw_sessions=raw,
        paths=paths,
    )
