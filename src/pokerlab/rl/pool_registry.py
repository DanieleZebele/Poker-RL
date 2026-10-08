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

import argparse
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from itertools import pairwise
from pathlib import Path

REGISTRY_FILENAME = "registry.json"

# How many opponents a training run seats.
DEFAULT_POOL_SIZE = 20
DEFAULT_RATING = 1500.0
# How the K-factor falls as a model accumulates rated games: `(games_at_least, K)`
# tiers, the last one whose threshold has been reached wins. A model that has
# played a thousand games has a rating that is already well determined, so one
# more session should nudge it, not shove it; a newcomer has to move fast to get
# to where it belongs. Used by the global registry, and -- through the
# `k_factors` override on `record_session_with_ratings` -- by the learner's own
# in-run evaluation, which counts its rated sessions itself because it is not a
# member. One schedule rates everything, so a rating means one thing everywhere:
# a run takes its validation sessions through the top tiers, its sessions
# against the anchors through the middle ones, and the stored population is
# refined from there.
#
# **The tiers are a staircase approximating the optimal gain, and the
# derivation is what fixes every number below.** A 1000-hand rated session is a
# measurement of a rating with a standard deviation of about **238 points**: Elo
# reads only the *sign* of each pair's chip delta, and the zero-sum structure of
# a session correlates the learner's five pairwise comparisons at exactly 0.5, so
# they are worth 2.18 independent Bernoullis. For a quantity that does not move,
# the gain extracting all of that evidence and no more is Kalman's, and in Elo's
# parametrisation it is exactly hyperbolic:
#
#     K_t = 1 / (slope * (t + V/P0)),   slope = 0.001421,   V = 238^2 = 56,864
#
# with `t` the rated sessions played and `P0` the variance of the starting
# rating's error. **`V/P0` is where "we already know roughly where this model
# sits" is encoded**, and it is the one number here that is an estimate. A run
# inherits its parent's rating rather than starting at 1500, so `P0` is how far a
# child's strength differs from its parent's -- taken at sd 40, giving
# `V/P0 = 35.5` and a first-session K of 20 instead of the 105 an uninformative
# prior would ask for. Setting it too low throws the inherited rating away; too
# high and the inherited error persists, which matters because `pick_parents`
# draws from the top 100 and that is a positive feedback loop.
#
# The staircase is geometric in `(t + 35)`: **20 tiers, each K a factor 1.4745
# below the one above, from 16 down to 0.01.** That ratio ties the two ends
# together (`16 / 1.4745^19 = 0.01`), and 0.01 sits at `704/0.01 - 35` = ~70,000
# rated games. A staircase is necessarily low at the start of a tier and high at
# the end; at this ratio the worst tier is ~4.5% off the curve.
#
# Where each tier bites:
#   * 16, 11 and 7.4, below 80 games: the first validation passes of a run, where
#     the rating still has distance to travel from its inherited starting point.
#   * 5.0 down to 1.05, from 80 to 750: the rest of a run's validation, then the
#     pass against the anchors, which is where a published rating is actually
#     earned. A model publishes with its run's rated sessions behind it.
#   * 0.72 and below: the stored population, refined whenever a population pass
#     draws a model. The ranking's job here is to keep an order steady rather
#     than to track a moving strength.
#   * The bottom tiers are headroom for game counts that only grow. Frozen
#     anchors have their deltas discarded anyway, so read anything below ~0.2 as a
#     promise the curve keeps rather than as a live tier.
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
def format_k_schedule(schedule: Sequence[tuple[int, float]]) -> str:
    """`schedule` as the text the `--k-schedule` flag and `config.toml` use:
    `"0:16, 20:11, 45:7.4, ..."`, one `games:K` pair per tier."""
    return ", ".join(f"{threshold}:{k:g}" for threshold, k in schedule)


def parse_k_schedule(text: str) -> tuple[tuple[int, float], ...]:
    """The inverse of `format_k_schedule`; `ValueError` says what is wrong.

    A schedule is a staircase, so it has to be one: it starts at 0 games (every
    model has a K from its first session), thresholds strictly increase, and
    every K is positive and finite.
    """
    pairs: list[tuple[int, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        games, separator, k = part.partition(":")
        if not separator:
            raise ValueError(f"'{part}' is not games:K")
        try:
            pairs.append((int(games), float(k)))
        except ValueError:
            raise ValueError(f"'{part}' is not games:K with an integer and a number") from None
    if not pairs:
        raise ValueError("a K schedule needs at least one games:K tier")
    if pairs[0][0] != 0:
        raise ValueError("the first tier must start at 0 games")
    for (before, _), (after, _) in pairwise(pairs):
        if after <= before:
            raise ValueError("tier thresholds must strictly increase")
    for _, k in pairs:
        if not 0.0 < k < float("inf"):
            raise ValueError("every K must be positive and finite")
    return tuple(pairs)


def k_schedule_text(text: str) -> str:
    """argparse `type=` for a K schedule: validates, returns the canonical text.

    The text itself is what travels (flag, file, worker command line, the
    settings a model records), so it stays a string rather than a tuple.
    """
    try:
        return format_k_schedule(parse_k_schedule(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


# Population-scale pruning of the global registry (see
# `rl/global_arena.py::run_population_sessions`). The caller turns these into two numbers for `eliminate_lowest_rated`: how
# many networks to remove (`fraction` of the eligible models, whenever the real
# on-disk population reaches `DEFAULT_POPULATION_TRIGGER`) and the minimum games
# a network needs to count as reliably rated (the `DEFAULT_PROTECT_PERCENTILE`
# of games played among the models rated so far).
DEFAULT_POPULATION_TRIGGER = 10_000
DEFAULT_ELIMINATION_FRACTION = 0.05
# **The 25th percentile.** The protection exists so a rating built on a handful
# of games is never grounds for deletion; at this percentile the bar is still
# well above a handful of rated sessions.
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

@dataclass
class PoolMember:
    """One rated pool entry: a saved checkpoint.

    In the global registry `ref` is the checkpoint's path under the checkpoint
    root (`models/` or a `benchmark/` series), refreshed every time the model
    plays a pass and rewritten when it is promoted to a benchmark directory. In
    the in-memory registry of a training run it is just the file name inside the
    models directory.
    """

    label: str
    ref: str
    rating: float = DEFAULT_RATING
    games: int = 0
    # How the model plays: `[events, opportunities]` of each statistic of
    # `engine/stats.py` (VPIP, PFR, 3-bet, ...) over its last `style_hands` seat-hands
    # (at most `style_log.STYLE_WINDOW`). Set by the run that trained it and refreshed
    # by every pass that seats it (`style_log.merge_style`). Empty for a model never
    # measured. Descriptive only: nothing ranks or draws on it.
    style: dict[str, list[int]] = field(default_factory=dict)
    style_hands: int = 0
    # A frozen member's rating is a fixed reference point: it shapes *other*
    # members' deltas but never moves itself -- see `record_session_with_
    # ratings`. Used for the benchmark anchors, and for the opponents a training
    # run scores its learner against (so a run never edits anyone else's rating).
    frozen: bool = False


def member_from_json(entry: Mapping) -> PoolMember:
    """A `PoolMember` from a stored entry: the fields it knows are taken, and what it
    does not (a file written by another version) is left out rather than refused, so a
    store loads whichever version wrote it and a field the member now has but the file
    lacks simply takes its default."""
    known = {f.name for f in fields(PoolMember)}
    return PoolMember(**{k: v for k, v in entry.items() if k in known})


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
    k_factors: Mapping[str, float],
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

    `k_factors` gives every participant its own K (a veteran moves less than a
    newcomer at the same table), read from the staircase by whoever calls this;
    there is no flat fallback, so a participant missing from it is an error
    rather than a guess. Each participant's change depends only on its *own* K, so the two sides of a
    pair are no longer equal and opposite and a session is no longer exactly
    zero-sum in rating. That is the price of letting veterans move slowly.
    """
    labels = list(results)
    deltas = {label: 0.0 for label in labels}
    if len(labels) < 2:
        return deltas

    for label in labels:
        if label not in k_factors:
            raise ValueError(f"no K-factor for {label!r}: every participant needs one")
        own_k = k_factors[label]
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
    members: dict[str, PoolMember] = field(default_factory=dict)
    # Each registered member's K comes from its own games played (`k_for_games`).
    # None for a registry that is only read, or whose caller names every K in
    # `k_factors` itself (a training run's in-memory registry).
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
                member = member_from_json(entry)
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
        text = json.dumps(payload, indent=2)
        # A style pair on one line, so a member stays readable by eye.
        text = re.sub(r"\[\s+(\d+),\s+(\d+)\s+\]", r"[\1, \2]", text)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)

    # ---- ranking --------------------------------------------------------

    def ranked(self) -> list[PoolMember]:
        return sorted(self.members.values(), key=lambda m: (-m.rating, m.label))

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
        """Apply one session's deltas, except to a frozen member.

        A frozen member still appears in `ratings` and is scored normally
        inside `pairwise_elo_delta`, so it still shapes every *other*
        participant's expected score and delta exactly like a normal
        opponent -- but nothing of its own moves: neither its rating nor its
        `games`. An anchor's rating and the games behind it belong to
        `benchmark_arena` alone, which reads `games` to pick each anchor's K; a
        count that grew with every pass that merely seated it as a yardstick
        would make the arena treat the anchor as more settled than its own
        games among the anchors say. The returned dict still carries its
        unapplied delta, for a caller that wants to know how it fared without
        persisting the result.

        `k_factors` gives named participants a K the registry cannot work out
        for itself, and it wins over the schedule. Anyone with neither a
        schedule entry nor a `k_factors` entry makes the session an error. Its
        main caller is the live learner of a training run, which is deliberately not a
        member (it changes every iteration, so persisting it would rate a moving
        target) and therefore has no `games` here for `k_for_games` to read.
        `SelfPlayTrainer` counts its own rated sessions and passes the K that
        follows, which is what lets the learner have a burn-in instead of the
        flat K -- see `SelfPlayTrainer.evaluate_against_pool`.
        """
        schedule_factors: dict[str, float] = {}
        if self.k_schedule is not None:
            # Read before `games` is incremented below: a session is rated at
            # the experience the model had when it sat down.
            schedule_factors = {
                label: k_for_games(self.members[label].games, self.k_schedule)
                for label in results
                if label in self.members
            }
        if k_factors is not None:
            schedule_factors.update(k_factors)
        deltas = pairwise_elo_delta(results, ratings, k_factors=schedule_factors)
        for label, delta in deltas.items():
            member = self.members.get(label)
            if member is not None and not member.frozen:
                member.games += 1
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
        run_population_sessions` derives both from the real on-disk population,
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
        ::run_population_sessions`). This is a genuinely destructive step, done
        deliberately to reclaim disk space: an eliminated model is gone.
        """
        eligible = [
            m
            for m in self.ranked()
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
        models = self.ranked()
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
