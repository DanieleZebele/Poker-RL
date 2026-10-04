"""Ratings for poker models: Elo bookkeeping shared by every ranking in the project.

The problem this solves is stated in CLAUDE.md: `SelfPlayTrainer.evaluate` is
dominated by noise (a spread of ~350-570 bb/100 across seeds on one fixed
model), so ranking models on a single win-rate measurement would rank the
luckiest, not the strongest. Ratings fix that by *accumulating*: every rated
session contributes one more result to each participant's running rating, so
hundreds of games' worth of evidence decide the ranking rather than one sample
from a heavy-tailed distribution.

The one persistent ranking is the global registry (`rl/global_store.py`, one
file per model); a training run also builds a small *in-memory* registry from the
opponents it drew, purely to score its learner against them.

Pure Python on purpose -- JSON and arithmetic, no torch. Turning a ranked member
back into a seat-filling `Opponent` needs `build_model_from_checkpoint`, and
that glue lives in `rl/train.py`, which already imports torch. Keeping the
ranking logic torch-free is what puts it in the ordinary test suite (see
`tests/unit/test_rl_pool_registry.py`) with no extra dependency, the same
argument that keeps `features.py` and `action_space.py` pure.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

REGISTRY_FILENAME = "registry.json"

# How many opponents a training run seats.
DEFAULT_POOL_SIZE = 20
DEFAULT_RATING = 1500.0
# The flat K, the fallback for a participant the registry knows nothing about.
# It used to be what rated the learner's own in-run evaluation as well; that
# now goes through `DEFAULT_K_SCHEDULE` like everything else (see the burn-in
# tiers below and `SelfPlayTrainer.evaluate_against_pool`), because a flat 8 was
# measured as unable to move the learner off 1500 inside a run: 40 rated
# sessions against a pool averaging ~1580 left a model of *identical* strength
# at 1530, and every one of 17 live workers sat between 1469 and 1534 whatever
# its bb/100. `benchmark.rate_against_benchmark` used to be the last caller of
# the flat K and no longer is: it continues the learner's own session count into
# `DEFAULT_K_SCHEDULE`, so a run is rated on one schedule from its first
# validation session to the moment it publishes. Nothing in the rating path uses
# a flat K any more -- what is left is the fallback for a participant nobody
# names, and `benchmark_arena`'s own `--k`, which is a separate dial.
#
# **Lowered from 24 to 8 at the user's decision (24 -> 12 -> 8): 24 was far too
# aggressive** for a rating meant to accumulate. At 24 a single session won
# outright against an equally rated field moved the learner 12 points, so a
# round's rated sessions could swing its rating by a couple of hundred points on
# evidence that CLAUDE.md measures as barely better than a coin flip per session
# (a 1000-hand session picks the stronger of two models 69.7% of the time). That
# number is not cosmetic: it is what a model is published into the global ranking
# with, so the noise in it becomes the newcomer's entry point. At 8 the same
# session moves it 4 points. Note that 24 survives as the schedule's *first*
# tier, where it is bounded to a model's first 10 rated games rather than
# applied to all of them -- which is the difference between a burn-in and a
# permanently jumpy rating.
DEFAULT_K_FACTOR = 8.0
# How the K-factor falls as a model accumulates rated games: `(games_at_least, K)`
# tiers, the last one whose threshold has been reached wins. A model that has
# played a thousand games has a rating that is already well determined, so one
# more session should nudge it, not shove it; a newcomer has to move fast to get
# to where it belongs. Used by the global registry, and -- through the
# `k_factors` override on `record_session_with_ratings` -- by the learner's own
# in-run evaluation, which counts its rated sessions itself because it is not a
# member. The thresholds are calibrated on the real population, and the
# distribution that matters is the one over the models the schedule can actually
# *move* -- the non-frozen ones, since a frozen anchor is scored but never has
# its own delta applied. Measured over the 9,433 non-frozen members on the
# volume: median 911 rated games, 90th percentile ~1,780, max 2,727.
#
# **The tiers are a 10-step staircase approximating the optimal gain, and the
# derivation is worth keeping because it is what fixes every number below.** A
# 1000-hand rated session is a measurement of a rating with a standard deviation
# of **238 points**: Elo reads only the *sign* of each pair's chip delta, and the
# zero-sum structure of a session correlates the learner's five pairwise
# comparisons at exactly 0.5, so they are worth 2.18 independent Bernoullis.
# For a quantity that does not move, the gain extracting all of that evidence and
# no more is Kalman's, and in Elo's parametrisation it is exactly hyperbolic:
#
#     K_t = 1 / (slope * (t + V/P0)),   slope = 0.001421,   V = 238^2 = 56,864
#
# with `t` the rated sessions played and `P0` the variance of the starting
# rating's error. **`V/P0` is where "we already know roughly where this model
# sits" is encoded**, and it is the one number here that is an estimate rather
# than a measurement. A run inherits its parent's rating rather than starting at
# 1500, so `P0` is how far a child's strength differs from its parent's -- taken
# at sd 40, giving `V/P0 = 35.5` and a first-session K of 20 instead of the 105
# an uninformative prior would ask for. Setting it too low throws the inherited
# rating away; too high and the inherited error persists, which matters because
# `pick_parents` draws from the top 100 and that is a positive feedback loop.
# Worth measuring properly one day, with a duplicate-deck duel of a model against
# its own parent.
#
# The staircase is geometric in `(t + 35)`: **20 tiers, each K a factor 1.4745
# below the one above, from 16 down to 0.01.** That ratio is what ties the two ends
# together (`16 / 1.4745^19 = 0.01`), and 0.01 sits at `704/0.01 - 35` = ~70,000
# rated games, which is where the large thresholds at the bottom come from.
#
# **It was 10 tiers halving from 16 to 0.10, and going to 20 improved the part that
# is actually used.** A staircase is necessarily low at the start of a tier and
# high at the end; at ratio 1.76 that was a **25%** swing inside each tier, which
# cost 3% of the final precision of a published rating (a 95% band of 19.9 points
# against the curve's 19.3, measured over 15,000 simulated runs). At 1.4745 the
# worst tier is **4.5%** off the curve, and **12 of the 20 tiers fall below 3,000
# games** -- the range the population can actually reach -- against 8 of the 10
# before. So this is not only headroom at the bottom: the live part got finer.
#
# **One schedule rates everything**, which is the point: the learner's in-run
# validation, its round against the frozen anchors, and every population-round
# session of every stored model all read it off the same session count, so a
# rating means one thing everywhere. A run takes its ~100 validation sessions from
# 16 down to 5.0, its 500 sessions against the anchors end at 1.05, and the stored
# population is refined from there. Which tier covers what is spelled out just
# above the tuple, in one place rather than two.
#
# **The history before this, because the direction of travel is measured and the
# destination was not.** The tiers used to be a burn-in (24 below 10 games, 8
# from 10) prepended to four settled tiers, and the burn-in existed because a
# rating starting at 1500 could not travel: a
# session moves it by at most `K * 0.5`, so at the old first tier of 3.0 -- and
# equally at the flat 8 the learner used to be rated at -- the first ten
# sessions of a new model's life were worth 15 and 40 points respectively
# against fields whose mean sits near 1580. Measured on the fleet: 17 live
# workers, all rated between 1469 and 1534, with bb/100 from -686 to +47, and a
# simulation of a learner *exactly* as strong as its pool finishing a full run
# at 1537. At 24 for the first 10 games and 8 for the next 20 the same learner
# reads 1565 after one evaluation instead of 1524, for a spread of +/-13 points
# against +/-5 -- a burn-in buys responsiveness with noise, and that trade is
# only sound because it expires.
#
# **The lower tiers were lowered three times, because the ranking could not tell
# its own best models apart.** Measured directly, 3 copies of one model against
# 3 of another with duplicate decks (the same shuffles replayed with the teams
# swapped, so card luck cancels): the #1 beat the #100 by only +6 bb/100 across
# 24,000 hands -- indistinguishable -- and the #20 *lost* to the #100 by
# 37 bb/100 (t = -6.1) although the ranking put it 16 points and 80 places
# higher. A null control of the #1 against itself returned +2.0 bb/100
# (t = 0.43), so the method was not inventing the effect. Everything the loop
# does rests on this order being right: it picks parents from the top 100, so a
# ranking that is wrong at the top makes that choice effectively random.
#
# The route was 24/16/12/8/6 -> 12/8/6/4/3 -> 4.0/1.0/0.5/0.1 -> 3.0/1.0/0.3/0.1
# -> those four with a burn-in (24 below 10 games, 8 from 10) prepended -> today's
# ten steps, and then today's twenty, which replace the lot with an approximation
# of the optimal gain.
# Simulation against a population whose true skill *is* known scored the ordering
# 0.889 for the first schedule and 0.935 for the second, and improved
# monotonically at every point of a grid that bottomed out at 6/4/3/2/2 (0.9604)
# -- so the *direction* of travel was measured early, while every value below that
# grid was chosen by hand until this schedule, which is the first that follows
# from a derivation rather than from judgement.
#
# Where each tier bites today, read against the games distribution above:
#   * 16, 11 and 7.4, below 80 games: the first validation rounds of a run, where
#     the rating still has distance to travel from its inherited starting point.
#   * 5.0, from 80: where a run's ~100 validation sessions end and the round
#     against the anchors begins.
#   * 3.4 down to 1.05, from 140 to 750: the round against the anchors, which is
#     where a published rating is actually earned. A model publishes with ~600 rated
#     sessions, so **1.05 is the tier it enters the global ranking in**.
#   * 0.72 down to 0.22, from 750 to 3,750: the stored population as it stands,
#     refined 50 games at a time whenever a population round draws it. Measured over
#     the non-frozen members, the distribution is median 911 rated games, 90th
#     percentile ~1,780 and max 2,727, so this is where a rating that has been
#     around actually sits. At 0.33 a session moves it by at most a third of a point
#     -- nearly held still, which is the point: the ranking's job here is to keep an
#     order steady rather than to track a moving strength.
#   * 0.15 and below, from 3,750 games: headroom, reached by nobody who can move
#     today. It exists because the population's game counts only grow, and they
#     grow faster than they used to now that a model publishes with ~600 rated
#     sessions instead of zero. The bottom tier -- 0.01 at 60,000 games -- is
#     inside the range the *frozen anchors* occupy (up to 115,372 games), and their
#     deltas are discarded anyway, so read everything below ~0.2 as a promise the
#     curve keeps rather than as a live tier.
DEFAULT_K_SCHEDULE: tuple[tuple[int, float], ...] = (
    (0, 16.0),
    (20, 11.0),
    (45, 7.4),
    (80, 5.0),
    (140, 3.4),
    (225, 2.3),
    (325, 1.6),
    (500, 1.05),
    (750, 0.72),
    (1200, 0.49),
    (1700, 0.33),
    (2500, 0.22),
    (3750, 0.15),
    (5500, 0.10),
    (8500, 0.07),
    (12000, 0.047),
    (18000, 0.032),
    (27500, 0.022),
    (40000, 0.015),
    (60000, 0.01),
)
# Population-scale pruning of the global registry (see
# `rl/global_arena.py::run_population_round`). The caller turns these into two numbers for `eliminate_lowest_rated`: how
# many networks to remove (`fraction` of the eligible models, whenever the real
# on-disk population reaches `DEFAULT_POPULATION_TRIGGER`) and the minimum games
# a network needs to count as reliably rated (the `DEFAULT_PROTECT_PERCENTILE`
# of games played among the models rated so far).
DEFAULT_POPULATION_TRIGGER = 10_000
DEFAULT_ELIMINATION_FRACTION = 0.05
# **The 25th percentile, lowered from the 50th.** The protection exists so a
# rating built on a handful of games is never grounds for deletion, and at the
# real population that bar is still comfortably met: measured over the 9,588
# rated, non-frozen models on disk, the 25th percentile of games played is
# **627 rated sessions** of 1,000 hands each, against 962 at the 50th. Nobody
# is being deleted on thin evidence at either setting.
#
# What changes is *who can be reached*. Eligibility is a percentile of games,
# so any draw that seats some models more often than others makes the
# well-played ones eligible sooner and leaves the rarely-drawn tail permanently
# immune -- and the pass then eats the middle of the population instead of its
# bottom, which is the opposite of what pruning is for. That pressure grows
# with `tiered_draw` (see `global_arena.py`), which deliberately seats the
# top bands far more often. Lowering the bar to the 25th percentile widens
# the eligible set from 50% to 75% of the population, so the tail comes back
# within reach.
#
# The cost, which is accepted: a pass removes `DEFAULT_ELIMINATION_FRACTION` of
# the *eligible* set, so 25% of 7,193 instead of 25% of 4,799 -- about 1,800
# models a pass rather than 1,200. Prunes therefore fire less often and bite
# harder.
DEFAULT_PROTECT_PERCENTILE = 25.0

MODEL = "model"


@dataclass
class PoolMember:
    """One rated pool entry: a saved checkpoint.

    In the global registry `ref` is the checkpoint's path under the checkpoint
    root (`models/` or a `benchmark/` series), refreshed every time the model
    plays a round and rewritten when it is promoted to a benchmark directory. In
    the in-memory registry of a training run it is just the file name inside the
    models directory.
    """

    label: str
    kind: str
    ref: str
    rating: float = DEFAULT_RATING
    games: int = 0
    iteration: int = 0
    # bb/100 against the fixed *selection* half of the benchmark set. Unlike
    # `rating`, this is on one absolute scale: the opponents never change and
    # never train, so a score means the same thing on every machine and in
    # every generation. `rating` only ever says "better than the rest of this
    # pool right now" -- measured proof: Shark, identical code everywhere,
    # rates 1387 on one machine and 1487 on another.
    benchmark: float | None = None
    # A frozen member's rating is a fixed reference point: it shapes *other*
    # members' deltas but never moves itself -- see `record_session_with_
    # ratings`. Used for the benchmark anchors, and for the opponents a training
    # run scores its learner against (so a run never edits anyone else's rating).
    frozen: bool = False

    @property
    def is_model(self) -> bool:
        return self.kind == MODEL


def expected_score(rating: float, opponent_rating: float) -> float:
    """The standard logistic Elo expectation."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def interpolated_percentile(values: Sequence[float], pct: float) -> float:
    """The `pct`-th percentile (0-100) of `values`, by linear interpolation.

    Matches the usual default ("linear") method: at `pct=50` this is exactly
    the median for both odd- and even-length inputs. Written out explicitly
    rather than reached for `statistics.median()` because
    `eliminate_by_percentile`'s `protect_percentile` argument has to actually
    vary -- a games-played protection threshold that only ever means "the
    median" regardless of what is asked for would silently ignore the
    argument it claims to take.
    """
    if not values:
        raise ValueError("interpolated_percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def k_for_games(games: int, schedule: Sequence[tuple[int, float]] = DEFAULT_K_SCHEDULE) -> float:
    """The K-factor for a model that has played `games` rated games."""
    k = schedule[0][1]
    for threshold, value in schedule:
        if games >= threshold:
            k = value
    return k


def pairwise_elo_delta(
    results: dict[str, float],
    ratings: dict[str, float],
    *,
    k_factor: float = DEFAULT_K_FACTOR,
    k_factors: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Rating changes for one multiplayer session, from each seat's chip delta.

    Elo is a two-player system, so a 6-handed session is scored as every pair
    of participants having played each other: a participant who finished the
    session with more chips than another scores 1 against them, fewer scores 0,
    equal scores 0.5 (a genuine outcome in poker -- a chopped pot leaves both
    stacks unchanged, see CLAUDE.md).

    The per-pair changes are divided by the number of opponents faced, so a
    6-handed session moves a rating about as much as a heads-up one rather than
    five times as much; otherwise table size would silently rescale K.

    Every update is computed against the ratings as they were *before* the
    session, never against a partially updated table, so the result does not
    depend on the order the participants happen to be iterated in.

    `k_factors` gives individual participants their own K (a veteran moves less
    than a newcomer at the same table); anyone missing from it uses `k_factor`.
    Each participant's change depends only on its *own* K, so the two sides of a
    pair are no longer equal and opposite and a session is no longer exactly
    zero-sum in rating. That is the price of letting veterans move slowly.
    """
    labels = list(results)
    deltas = {label: 0.0 for label in labels}
    if len(labels) < 2:
        return deltas

    for label in labels:
        own_k = k_factor if k_factors is None else k_factors.get(label, k_factor)
        for other in labels:
            if other == label:
                continue
            if results[label] > results[other]:
                score = 1.0
            elif results[label] < results[other]:
                score = 0.0
            else:
                score = 0.5
            expected = expected_score(ratings[label], ratings[other])
            deltas[label] += own_k * (score - expected)
        deltas[label] /= len(labels) - 1
    return deltas


@dataclass
class PoolRegistry:
    """A set of rated models, with the Elo arithmetic to update them."""

    directory: Path
    max_models: int = DEFAULT_POOL_SIZE
    k_factor: float = DEFAULT_K_FACTOR
    members: dict[str, PoolMember] = field(default_factory=dict)
    # When set, each registered member's K comes from its own games played
    # (`k_for_games`) instead of the flat `k_factor`. Off (None) for a
    # per-machine training pool, whose members are re-rated by the arena every
    # generation and must stay responsive; on for the global registry.
    k_schedule: tuple[tuple[int, float], ...] | None = None

    @property
    def path(self) -> Path:
        return Path(self.directory) / REGISTRY_FILENAME

    # ---- persistence ----------------------------------------------------

    @classmethod
    def load(cls, directory: str | Path, **kwargs) -> PoolRegistry:
        """Read the registry, tolerating a missing or unreadable file.

        A corrupt registry is a recoverable annoyance -- the ratings are
        derived data that rebuild themselves over a few sessions -- whereas
        refusing to start training over it would not be. Same reasoning as
        `archived_opponents` skipping a bad checkpoint instead of dying.
        """
        registry = cls(directory=Path(directory), **kwargs)
        try:
            raw = json.loads(registry.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return registry
        for entry in raw.get("members", []):
            try:
                member = PoolMember(**entry)
            except TypeError:
                continue
            registry.members[member.label] = member
        return registry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "max_models": self.max_models,
            "members": [asdict(m) for m in self.ranked()],
        }
        # Write-then-rename: a crash mid-write must not leave a truncated file
        # that the next run silently reads as an empty ranking.
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    # ---- ranking --------------------------------------------------------

    def ranked(self) -> list[PoolMember]:
        return sorted(self.members.values(), key=lambda m: (-m.rating, m.label))

    def models(self) -> list[PoolMember]:
        return [m for m in self.ranked() if m.is_model]

    def record_session(self, results: dict[str, float]) -> dict[str, float]:
        """Fold one session's per-participant chip deltas into the ratings.

        Participants that are not registered members (the live learner, most
        importantly) are rated for the purposes of the pairwise comparison but
        are not persisted; the caller gets every delta back, so it can carry
        the learner's own rating forward itself.
        """
        ratings = {
            label: self.members[label].rating if label in self.members else DEFAULT_RATING
            for label in results
        }
        return self.record_session_with_ratings(results, ratings)

    def record_session_with_ratings(
        self,
        results: dict[str, float],
        ratings: dict[str, float],
        *,
        k_factors: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Apply one session's deltas, except to a frozen member's rating.

        A frozen member still appears in `ratings` and is scored normally
        inside `pairwise_elo_delta`, so it still shapes every *other*
        participant's expected score and delta exactly like a normal
        opponent -- only its own rating is held fixed (`games` still counts,
        for honest bookkeeping). The returned dict still carries its
        unapplied delta, for a caller that wants to know how it fared without
        persisting the result.

        `k_factors` gives named participants a K the registry cannot work out
        for itself, and it wins over the schedule. It exists for exactly one
        caller: the live learner of a training run, which is deliberately not a
        member (it changes every iteration, so persisting it would rate a moving
        target) and therefore has no `games` here for `k_for_games` to read.
        `SelfPlayTrainer` counts its own rated sessions and passes the K that
        follows, which is what lets the learner have a burn-in instead of the
        flat K -- see `SelfPlayTrainer.evaluate_against_pool`.
        """
        schedule_factors = None
        if self.k_schedule is not None:
            # Read before `games` is incremented below: a session is rated at
            # the experience the model had when it sat down.
            schedule_factors = {
                label: k_for_games(self.members[label].games, self.k_schedule)
                for label in results
                if label in self.members
            }
        if k_factors is not None:
            schedule_factors = {**(schedule_factors or {}), **k_factors}
        k_factors = schedule_factors
        deltas = pairwise_elo_delta(
            results, ratings, k_factor=self.k_factor, k_factors=k_factors
        )
        for label, delta in deltas.items():
            member = self.members.get(label)
            if member is not None:
                member.games += 1
                if not member.frozen:
                    member.rating += delta
        return deltas

    def eliminate_lowest_rated(
        self,
        *,
        count: int,
        games_threshold: float,
        among: Collection[str] | None = None,
    ) -> list[PoolMember]:
        """Remove up to `count` of the lowest-rated members that have played
        at least `games_threshold` games.

        Deciding *whether* to prune at all, and how reliable a rating must be
        (`games_threshold`), is the caller's job: `global_arena.
        run_population_round` derives both from the real on-disk population,
        which this registry cannot see -- see its docstring for why `count` is
        sized against the *eligible* set here rather than the raw population.
        Frozen members are never eligible; `among`, when given, restricts
        eligibility to those labels (the networks that still exist on disk),
        so a stale entry can never use up a place in the quota.

        Capped, not all-or-nothing: if fewer than `count` members qualify,
        every one of them goes rather than none. The alternative -- waiting
        for `count` to be simultaneously available -- can stall forever if
        `count` is large relative to how much of the population ever gets
        rated at once (it is; see the docstring on the caller).

        Only edits `self.members` -- deleting the files, and deciding
        *whether* to call this at all, are the caller's job (`global_arena.py
        ::run_population_round`). This is a genuinely destructive step, done
        deliberately to reclaim disk space: there is no "retired" holding area
        any more, an eliminated model is gone.
        """
        eligible = [
            m
            for m in self.models()
            if not m.frozen
            and m.games >= games_threshold
            and (among is None or m.label in among)
        ]
        if count <= 0 or not eligible:
            return []
        doomed = sorted(eligible, key=lambda m: (m.rating, m.label))[:count]
        for member in doomed:
            del self.members[member.label]
        return doomed

    # ---- seating --------------------------------------------------------

    def fill_slots(self, count: int) -> list[PoolMember]:
        """Exactly `count` seat fillers, best rated first.

        When there are fewer than `count` distinct members the remainder is
        made up by cycling the models again. A repeated member is not a
        second registry entry: it is the same model seated twice, which
        simply weights it more heavily in the sampling.
        """
        models = self.models()
        if count <= 0 or not models:
            return []
        return [models[index % len(models)] for index in range(count)]

    def summary(self, limit: int = 10) -> str:
        lines = [f"{'membro':<34}{'rating':>8}{'partite':>9}"]
        for member in self.ranked()[:limit]:
            lines.append(f"{member.label:<34}{member.rating:8.0f}{member.games:9d}")
        hidden = len(self.members) - min(limit, len(self.members))
        if hidden > 0:
            lines.append(f"... e altri {hidden}")
        return "\n".join(lines)
