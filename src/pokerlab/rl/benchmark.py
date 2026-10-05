"""A frozen opponent set, and the pass that rates a new model against it.

The ratings in `pool_registry` are relative: they say whether a model beats the
others in the pool right now, and the pool is replaced as training goes on. The
frozen anchors in `checkpoints/benchmark/` are the fixed reference the scale is
measured against. They are **never seated in training** -- otherwise the agent is
trained on its own test set and the number measures memorisation of specific
opponents rather than strength -- and their ratings are pinned.

`rate_against_benchmark` is what each worker runs on its own model right before
publishing it; see below for how many sessions and why.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.policy import PokerActorCritic, make_policy_fn
from pokerlab.rl.pool_registry import (
    DEFAULT_K_SCHEDULE,
    k_for_games,
    pairwise_elo_delta,
)
from pokerlab.rl.pool_registry import DEFAULT_RATING as DEFAULT_ANCHOR_RATING
from pokerlab.rl.ppo import build_model_from_checkpoint
from pokerlab.rl.rollout import Opponent, TableBank, policy_opponent
from pokerlab.rl.table_mix import DEFAULT_SESSION_HANDS, TableMix

DEFAULT_BENCHMARK_DIR = Path("checkpoints/benchmark")


# ---- rating a fresh model against the frozen anchors ------------------------

# **500 sessions of 1,000 hands, drawn at random from the whole frozen set.**
# This pass is a measurement of one number -- the rating a model is published
# with -- not a breakdown by which part of the population a model beats.
#
# **Why 500, and why the count continues from training.** A 1000-hand session
# measures a rating with a standard deviation of about 238 points (see
# `pool_registry.DEFAULT_K_SCHEDULE` for where that comes from), so precision
# improves only as `1/sqrt(sessions)`: halving the band again costs four times
# the sessions. 500 sessions are also 500,000 hands, which keeps
# `benchmark_bb100` precise enough for the hyperparameter sweep to read.
#
# **The sessions are rated on the learner's own running count**
# (`games`, continued from `SelfPlayTrainer.learner_games`), through
# `DEFAULT_K_SCHEDULE`, not at a flat K. A fixed K is an exponential average
# with a fixed effective window, so more sessions buy nothing past a point; the
# schedule's falling K is what turns 500 sessions into 500 sessions' worth of
# evidence.
#
# **Anchors are loaded in rotating slices, not all at once.** The frozen set only
# grows, and every worker of a generation reaches this phase at about the same
# moment, so holding the whole set resident in each worker would multiply into a
# large transient allocation across a machine. A slice of `resident` anchors is
# drawn at random, played for `rotate_every` sessions and dropped, so the
# resident cost stays small and the pass still faces the whole set.
DEFAULT_BENCHMARK_SESSIONS = 500
DEFAULT_RESIDENT_ANCHORS = 10
DEFAULT_ANCHOR_ROTATE_EVERY = 50


@dataclass(frozen=True)
class BenchmarkRating:
    """What the pass against the frozen anchors concluded about one model."""

    bb_per_100: float
    sessions: int
    hands: int
    # Distinct anchors that actually took a seat, across every slice.
    opponents: int
    rating_after: float
    # The learner's session count after the pass: what the model is registered
    # with, so a population pass rates it at the experience it really has.
    games_after: int
    # Chip deltas per participant, one dict per session, in the shape the global
    # merge consumes (`global_arena._apply_session`).
    raw_sessions: list[dict[str, float]] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    # The learner's bb/100 at each table size it played, a diagnostic to read and
    # nothing more: the rating is one number over the whole mixture.
    bb_per_100_by_size: dict[int, float] = field(default_factory=dict)


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
    mix: TableMix,
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
    population pass happens to draw it -- about a 0.5% chance per pass out of
    a large store -- so without this pass it could sit on an unearned number for
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
    # The largest table that can be drawn decides how many anchors the set must
    # hold: a pass that could only seat the small tables would measure a
    # different mixture than the weights say, so it does not run at all.
    needed = mix.max_players - 1
    available = anchor_paths(directory)
    if len(available) < needed:
        if on_skip is not None:
            on_skip(
                Path(str(directory)),
                f"solo {len(available)} ancore, un tavolo da {mix.max_players} ne chiede {needed}",
            )
        return None

    rng = random.Random(seed)
    bot_rng = random.Random(rng.random())
    size_rng = random.Random(rng.random())
    bank = TableBank(mix, rng)
    learners = [
        RLAgentPlayer(
            f"s{seat}", label,
            policy_fn=make_policy_fn(model, device=device),
            big_blind=mix.big_blind, starting_stack=mix.starting_stack,
        )
        for seat in range(mix.max_players)
    ]

    current = float(rating)
    played_games = int(games)
    won = 0
    raw: list[dict[str, float]] = []
    paths: dict[str, str] = {}
    seen: set[str] = set()
    won_by_size: dict[int, int] = {}
    hands_by_size: dict[int, int] = {}
    rotate = max(1, rotate_every)
    opponents: list[Opponent] = []
    played = 0

    while played < sessions:
        if played % rotate == 0 or len(opponents) < needed:
            # A fresh random slice: loaded, played, and dropped before the next
            # one, which is what keeps the resident cost at one slice.
            if len(available) < needed:
                break
            opponents = []
            slice_size = max(needed, min(resident, len(available)))
            for path in rng.sample(available, slice_size):
                try:
                    loaded, _checkpoint = build_model_from_checkpoint(path, device=device)
                except Exception as exc:  # noqa: BLE001 - a user-owned directory
                    # Struck off, not merely skipped: a bad file must be tried
                    # once, not once per slice, or a pass of ten slices reports
                    # the same failure ten times and can spin if a whole slice
                    # happens to be bad.
                    available = [other for other in available if other != path]
                    if on_skip is not None:
                        on_skip(path, str(exc))
                    continue
                opponents.append(
                    policy_opponent(path.stem, make_policy_fn(loaded, device=device), mix)
                )
                paths[path.stem] = str(path)
                seen.add(path.stem)
            if len(opponents) < needed:
                continue

        # Drawn per session and held for all of it (see `table_mix`).
        num_players = mix.draw_size(size_rng)
        proxies = bank.seats(num_players)
        learner_seat = rng.randrange(num_players)
        drawn = rng.sample(opponents, num_players - 1)
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

        deltas = bank.play_session(num_players, hands)
        won += deltas[learner_seat]
        won_by_size[num_players] = won_by_size.get(num_players, 0) + deltas[learner_seat]
        hands_by_size[num_players] = hands_by_size.get(num_players, 0) + hands

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
        current += pairwise_elo_delta(
            session, ratings, k_factors=dict.fromkeys(session, k)
        ).get(label, 0.0)
        played_games += 1
        played += 1
        if on_progress is not None:
            # The running figures ride in the detail so the watcher can show
            # them while the pass is still playing: until it ends, the
            # `ancore/100` and `elo ancore` cells would otherwise stay empty
            # for hours. Read by `monitor._SERIES_DETAIL`.
            running = won / mix.big_blind / (played * hands) * 100
            on_progress(played, sessions, f"rating {current:.0f}, {running:+.1f} bb/100")

    if played == 0:
        return None
    return BenchmarkRating(
        bb_per_100=won / mix.big_blind / (played * hands) * 100,
        sessions=played,
        hands=played * hands,
        opponents=len(seen),
        rating_after=current,
        games_after=played_games,
        raw_sessions=raw,
        paths=paths,
        bb_per_100_by_size={
            size: won_by_size[size] / mix.big_blind / hands_by_size[size] * 100
            for size in sorted(won_by_size)
        },
    )
