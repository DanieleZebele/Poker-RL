"""How many hands does a session need before the stronger model actually wins it?

Everything the ranking does rests on one implicit assumption: that when two
models play a session, the one that finishes ahead is the better one. That is
not a fact, it is a probability, and the probability depends entirely on how
long the session is. A poker session is a coin weighted by skill -- at ten hands
the weighting is invisible and the winner is whoever caught cards, at a hundred
thousand it is all that is left.

This module measures that curve for a concrete pair of models:

    P(the stronger model finishes ahead)  as a function of session length

and it answers the inverse question -- how many hands a session must last for
that probability to reach 75%, 90%, 95%, 99%.

Three ideas make it cheap enough to actually run.

- **One long stream is many sessions.** `Table` is re-stacked after every hand
  here (as it is in every evaluation path in this project), so consecutive hands
  are independent and identically distributed. A stream of 10,000 hands
  therefore *is* 1,000 sessions of 10 hands, or 10 of 1,000, depending only on
  where it is chopped. Every session length in the grid is read off the same
  played hands instead of being played separately.
- **Duplicate decks give the ground truth.** The question presupposes knowing
  which model really is stronger, and an ordinary measurement of that is exactly
  as noisy as the thing being studied. Each stream is therefore played twice
  from the same seed, with the two models' seats swapped the second time. `Table`
  consumes its `random.Random` only to shuffle, once per hand, so both
  arrangements see an identical sequence of deals and the card luck cancels in
  the difference: `(normal - mirrored) / 2` is the skill effect alone. Note that
  mirroring does not change the estimate of the edge -- the pooled mean is
  exactly the mean of the paired differences -- it changes how precisely that
  estimate is known, which is the whole point.
- **The normal approximation extrapolates.** With a per-hand edge `mu` and a
  per-hand noise `sigma`, a session of `H` hands is won with probability
  `Phi(mu * sqrt(H) / sigma)`, and `mu` and `sigma` are measured to good
  precision from the *total* hands played, not from the handful of long blocks.
  So the table below reports a measured frequency wherever there are enough
  blocks to measure one and a predicted probability everywhere, including at
  lengths far past anything that was played. The two columns sitting next to
  each other is also the check on the approximation: the per-hand chip delta is
  heavy-tailed, so the two are expected to disagree at ten hands and to agree
  from a few hundred on.

Run it by hand; nothing imports it:

    OMP_NUM_THREADS=1 python -m pokerlab.rl.duel_power --rank-a 1 --rank-b 100

The statistics half of the file is plain Python and torch-free, and the play
half imports torch lazily, for the reason `features.py` and `pool_registry.py`
are pure: the arithmetic that decides what the answer *is* belongs in the
ordinary test suite with no extra dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table

DEFAULT_GLOBAL_DIR = Path("checkpoints/global")
DEFAULT_ROOT = Path("checkpoints")
DEFAULT_FIELD_DIR = Path("checkpoints/benchmark")

# The grid deliberately runs well past what anyone plays: the population pass's
# session is 1,000 hands and so is a rated session everywhere else, so the lengths
# have to be readable straight off the table, and the lengths beyond them are
# what say how far short they fall.
DEFAULT_LENGTHS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000)
DEFAULT_TARGETS = (0.75, 0.90, 0.95, 0.99)
# The lengths the distribution section draws a histogram for. One order of
# magnitude apart, because the spread of a session result only halves when the
# hands quadruple: anything closer together looks like the same picture twice.
DEFAULT_DISTRIBUTION_LENGTHS = (100, 1000, 10000, 100000)
DEFAULT_STREAMS = 8
DEFAULT_HANDS = 2000

Z95 = 1.959963984540054


# ---- statistics ------------------------------------------------------------
# Pure Python on purpose: no torch, no numpy. These functions decide what the
# experiment concludes, so they are the part that must be unit-tested.


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """Inverse of `normal_cdf`, by bisection.

    Bisection rather than one of the usual rational approximations: it is four
    lines, it cannot be subtly wrong in the tail, and it is called a handful of
    times per run.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p deve stare in (0, 1), non {p}")
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if normal_cdf(middle) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def wilson_interval(score: float, trials: int, z: float = Z95) -> tuple[float, float]:
    """A confidence interval for a proportion that stays inside [0, 1].

    The textbook `p +- z*sqrt(p(1-p)/n)` is useless exactly where this table
    needs an interval -- near 0 and 1, where the long sessions live -- since it
    happily reports an upper bound above 100%.
    """
    if trials <= 0:
        return (0.0, 1.0)
    rate = score / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
    spread /= denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass(frozen=True)
class BlockOutcome:
    """How often the first model came out ahead over sessions of one length."""

    hands: int
    sessions: int
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        """Wins plus half the draws.

        A drawn session is not a degenerate case in poker -- a chopped pot
        genuinely leaves two stacks level -- and at ten hands it happens often
        enough to matter. Half a win each is the same convention Elo uses.
        """
        return self.wins + 0.5 * self.draws

    @property
    def rate(self) -> float:
        return self.score / self.sessions if self.sessions else float("nan")


def block_outcomes(streams: Sequence[Sequence[float]], hands: int) -> BlockOutcome:
    """Chop every stream into sessions of `hands` hands and count who won them.

    A trailing partial block is dropped rather than counted short: a 600-hand
    remainder scored as if it were a 1,000-hand session would quietly mix two
    different measurements into one row.
    """
    wins = draws = losses = 0
    for stream in streams:
        for start in range(0, len(stream) - hands + 1, hands):
            total = sum(stream[start : start + hands])
            if total > 0:
                wins += 1
            elif total < 0:
                losses += 1
            else:
                draws += 1
    return BlockOutcome(
        hands=hands, sessions=wins + draws + losses, wins=wins, draws=draws, losses=losses
    )


def win_probability(edge: float, sigma: float, hands: int) -> float:
    """P(a session of `hands` hands is won), under the normal approximation.

    The sum of `H` i.i.d. per-hand deltas has mean `H*edge` and standard
    deviation `sigma*sqrt(H)`, so it is positive with probability
    `Phi(edge*sqrt(H)/sigma)`. Everything in this file that extrapolates past
    the hands actually played rests on this one line.
    """
    if sigma <= 0:
        return 1.0 if edge > 0 else (0.0 if edge < 0 else 0.5)
    return normal_cdf(edge * math.sqrt(hands) / sigma)


def hands_for_probability(edge: float, sigma: float, target: float) -> int | None:
    """The session length at which `win_probability` reaches `target`.

    `None` when the edge is zero or points the wrong way: no session length
    identifies the stronger of two models that are equally strong.
    """
    if edge <= 0 or sigma <= 0:
        return None
    z = normal_quantile(target)
    if z <= 0:
        return 1
    return math.ceil((z * sigma / edge) ** 2)


@dataclass
class DuelMeasurement:
    """The played hands, and every conclusion that can be drawn from them.

    `normal` and `mirrored` hold one list of per-hand scores per stream, a score
    being the chip delta of model A's seats minus that of model B's seats.
    `mirrored[i]` is already expressed from A's point of view (the raw result of
    the swapped arrangement, negated), and is aligned hand by hand with
    `normal[i]` because both were dealt from the same seed.
    """

    label_a: str
    label_b: str
    seats_per_team: int
    big_blind: int
    normal: list[list[float]] = field(default_factory=list)
    mirrored: list[list[float]] = field(default_factory=list)

    @property
    def mirror(self) -> bool:
        return bool(self.mirrored)

    @property
    def streams(self) -> list[list[float]]:
        """Every stream, both arrangements, as independent sessions to chop.

        Blocks never straddle two streams or two arrangements, so each block is
        a real session played from one seating.
        """
        return [*self.normal, *self.mirrored]

    @property
    def hands_played(self) -> int:
        return sum(len(stream) for stream in self.streams)

    def scores(self) -> list[float]:
        return [value for stream in self.streams for value in stream]

    def paired(self) -> list[float]:
        """The two arrangements of one hand, averaged: the deal's luck cancelled.

        Both lists are read from A's side, so the average is
        `(normal + mirrored) / 2` here -- the same thing as half the difference
        between the two raw arrangements, since the swapped one is stored
        negated. Whatever the cards were worth to a *seat* enters the first term
        with one sign and the second with the other, and drops out; what is left
        is the difference the two networks made playing them.
        """
        out: list[float] = []
        for plain, swapped in zip(self.normal, self.mirrored):
            out.extend((a + b) / 2.0 for a, b in zip(plain, swapped))
        return out

    def sigma(self) -> float:
        """Per-hand standard deviation of an ordinary (non-mirrored) session."""
        scores = self.scores()
        return statistics.stdev(scores) if len(scores) > 1 else 0.0

    def edge(self) -> tuple[float, float]:
        """The per-hand edge in chips, and its standard error.

        The point estimate is the mean score either way. With mirrored streams
        the standard error comes from the paired differences, where card luck
        has already cancelled, which is the only thing mirroring buys -- and it
        typically buys an order of magnitude.
        """
        scores = self.scores()
        if not scores:
            return (0.0, float("inf"))
        mean = statistics.fmean(scores)
        sample = self.paired() if self.mirror else scores
        if len(sample) < 2:
            return (mean, float("inf"))
        return (mean, statistics.stdev(sample) / math.sqrt(len(sample)))

    def to_bb100(self, chips_per_hand: float) -> float:
        """Chips per hand -> big blinds per 100 hands, per seat of a team.

        Note what this is: the head-to-head margin between one seat of A and one
        seat of B. In a table split evenly between the two it is twice A's own
        bb/100, because what A wins B loses.
        """
        return chips_per_hand / self.seats_per_team / self.big_blind * 100.0


@dataclass(frozen=True)
class PowerRow:
    hands: int
    measured: BlockOutcome | None
    predicted: float


def power_table(
    measurement: DuelMeasurement, lengths: Sequence[int], orientation: int
) -> list[PowerRow]:
    """One row per session length: what was measured, and what is predicted.

    `orientation` is +1 when model A is the stronger one and -1 when it is B, so
    every row reads as "how often the stronger model won", which is the question
    being asked.
    """
    streams = [[value * orientation for value in stream] for stream in measurement.streams]
    edge, _stderr = measurement.edge()
    sigma = measurement.sigma()
    rows: list[PowerRow] = []
    for hands in lengths:
        outcome = block_outcomes(streams, hands)
        rows.append(
            PowerRow(
                hands=hands,
                measured=outcome if outcome.sessions else None,
                predicted=win_probability(abs(edge), sigma, hands),
            )
        )
    return rows


DEFAULT_SESSIONS_SHOWN = 100


def sample_sessions(
    streams: Sequence[Sequence[float]], hands: int, *, limit: int | None = None
) -> list[float]:
    """`limit` sessions of `hands` hands, taken round-robin across the streams.

    Round-robin, not the first `limit` blocks in order, and the reason is the
    whole point of the exercise: a hundred consecutive blocks out of one stream
    are a hundred sessions from *one* seating, and part of what makes one
    session differ from another is which seats the two models drew. Taking block
    0 of every stream, then block 1 of every stream, spends the independence
    that was paid for.
    """
    per_stream = [
        [
            sum(stream[start : start + hands])
            for start in range(0, len(stream) - hands + 1, hands)
        ]
        for stream in streams
    ]
    out: list[float] = []
    depth = 0
    while any(depth < len(blocks) for blocks in per_stream):
        for blocks in per_stream:
            if depth < len(blocks):
                out.append(blocks[depth])
                if limit is not None and len(out) >= limit:
                    return out
        depth += 1
    return out


def _nice_step(rough: float) -> float:
    """A round bin width near `rough`: 1, 2, 2.5 or 5 times a power of ten.

    A histogram whose bins are 37.4 wide is arithmetically fine and unreadable.
    """
    if rough <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(rough))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if rough <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


def histogram(values: Sequence[float], *, bins: int = 12, width: int = 34) -> list[str]:
    """An ASCII histogram of session results, with zero always on a bin edge.

    Zero on an edge rather than in the middle of a bin because the sign is the
    only thing the ranking actually reads: a bin straddling zero would hide
    exactly the split being counted.
    """
    if not values:
        return ["   (nessuna partita)"]
    span = max(abs(min(values)), abs(max(values))) or 1.0
    step = _nice_step(2 * span / bins)
    low = math.floor(min(values) / step)
    high = math.ceil(max(values) / step)
    counts = [0] * (high - low)
    for value in values:
        index = min(math.floor(value / step) - low, len(counts) - 1)
        counts[max(0, index)] += 1
    tallest = max(counts) or 1

    lines: list[str] = []
    for offset, count in enumerate(counts):
        start = (low + offset) * step
        bar = "#" * round(count / tallest * width)
        lines.append(
            f"   {start:>9,.0f} .. {start + step:>9,.0f}  |{bar:<{width}} {count:>4}"
        )
        if start < 0 <= start + step:
            lines.append(f"   {'':>9} {'':>4} {'':>9}  " + "-" * (width + 5) + "  zero")
    return lines


def format_distributions(
    measurement: DuelMeasurement,
    *,
    lengths: Sequence[int],
    sessions: int = DEFAULT_SESSIONS_SHOWN,
) -> list[str]:
    """How `sessions` sessions of each length actually came out, one by one.

    The power table answers "how often"; this answers "how spread out", which is
    the same question asked in a way that can be seen. At a hundred hands the
    results are a cloud straddling zero and the winner is whoever caught cards;
    at a hundred thousand the cloud has collapsed onto the real edge and the
    same model wins nearly every time. Nothing changed but the length.
    """
    edge, _stderr = measurement.edge()
    orientation = 1 if edge >= 0 else -1
    stronger = measurement.label_a if orientation > 0 else measurement.label_b
    streams = [[value * orientation for value in stream] for stream in measurement.streams]

    lines = [
        f"--- distribuzione dei risultati, {sessions} partite per lunghezza",
        f"    (segno positivo = ha vinto {stronger}, il piu' forte)",
    ]
    if measurement.mirror:
        lines.append(
            "    nota: i flussi sono speculari a coppie, quindi ogni partita ha la sua"
        )
        lines.append(
            "    gemella a carte invertite e l'istogramma e' piu' simmetrico del vero"
        )
    summary: list[tuple[int, int, int, float, float]] = []

    for hands in lengths:
        drawn = sample_sessions(streams, hands, limit=sessions)
        if not drawn:
            lines += [
                "",
                (f"  {hands:,} mani per partita: non ce ne sono, servono almeno "
                 f"{hands:,} mani per flusso"),
            ]
            continue
        # bb/100 per seat, the same unit as the edge above, so a session's result
        # can be read against the true difference between the two models.
        results = [measurement.to_bb100(value / hands) for value in drawn]
        wins = sum(1 for value in drawn if value > 0)
        draws = sum(1 for value in drawn if value == 0)
        spread = statistics.stdev(results) if len(results) > 1 else 0.0
        summary.append((hands, len(results), wins, spread, statistics.median(results)))
        lines += [
            "",
            f"  {len(results)} partite da {hands:,} mani: vinte {wins}, perse "
            f"{len(results) - wins - draws}" + (f", pattate {draws}" if draws else ""),
            (f"  risultato per partita, bb/100 per seggio: da {min(results):+,.1f} a "
             f"{max(results):+,.1f}, mediana {statistics.median(results):+,.1f}, "
             f"dispersione {spread:,.1f}"),
            "",
        ]
        lines += histogram(results)

    if summary:
        lines += [
            "",
            "--- riepilogo: la stessa differenza di forza, vista da lunghezze diverse",
            "",
            "   mani/partita   partite   vinte dal piu' forte   dispersione   mediana",
        ]
        for hands, count, wins, spread, median in summary:
            lines.append(
                f"   {hands:>12,} {count:>9} {wins:>12} ({wins / count * 100:4.0f}%)"
                f"   {spread:>11,.1f}   {median:>+7,.1f}"
            )
        lines += [
            "",
            "   La dispersione si dimezza ogni volta che le mani quadruplicano, la",
            "   mediana no: e' li' che sta il vero divario fra i due modelli. Le",
            "   vittorie si concentrano quando la prima scende sotto la seconda.",
        ]
    return lines


def _plural(count: int, singular: str, plural: str) -> str:
    """"1 seggio" / "3 seggi": the report is read by a person, not parsed."""
    return f"{count:,} {singular if count == 1 else plural}"


def format_report(
    measurement: DuelMeasurement,
    *,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    targets: Sequence[float] = DEFAULT_TARGETS,
    rating_a: float | None = None,
    rating_b: float | None = None,
    session_hands: int | None = None,
) -> list[str]:
    """The whole answer, as printable lines. Pure: takes data, returns text."""
    edge, stderr = measurement.edge()
    sigma = measurement.sigma()
    t_stat = edge / stderr if stderr else 0.0
    orientation = 1 if edge >= 0 else -1
    stronger = measurement.label_a if orientation > 0 else measurement.label_b
    weaker = measurement.label_b if orientation > 0 else measurement.label_a

    def rating(value: float | None) -> str:
        return f"  (rating {value:.1f})" if value is not None else ""

    lines = [
        f"A: {measurement.label_a}{rating(rating_a)}",
        f"B: {measurement.label_b}{rating(rating_b)}",
        (f"{_plural(measurement.seats_per_team, 'seggio', 'seggi')} per modello, "
         f"big blind {measurement.big_blind}"),
        _plural(len(measurement.normal), "flusso", "flussi")
        + (" x 2 disposizioni speculari" if measurement.mirror else "")
        + f" = {measurement.hands_played:,} mani giocate",
        "",
        "--- forza reale "
        + ("(mazzi duplicati: le stesse carte a parti invertite, "
           "la fortuna si annulla)" if measurement.mirror else "(misura diretta, non speculare)"),
    ]
    margin = measurement.to_bb100(edge) * orientation
    margin_error = measurement.to_bb100(stderr) if stderr != float("inf") else float("inf")
    lines.append(
        f"  vantaggio di {stronger}: {margin:+.2f} bb/100 per seggio"
        f"  (+/- {Z95 * margin_error:.2f} al 95%, t = {abs(t_stat):.1f})"
    )
    lines.append(
        f"  rumore di una singola mano: sigma = {measurement.to_bb100(sigma) / 100:.1f} bb "
        f"(stessa unita' del vantaggio qui sopra)"
    )
    if abs(t_stat) < 3.0:
        lines += [
            "",
            "  ATTENZIONE: con t < 3 non e' accertato quale dei due sia il piu' forte.",
            "  La tabella qui sotto misura quanto spesso vince quello che in questa",
            "  misura e' risultato davanti, che potrebbe essere il piu' debole:",
            "  gioca piu' mani (--streams / --hands) prima di crederci.",
        ]

    lines += [
        "",
        f"--- con quale probabilita' una partita la vince {stronger} (il piu' forte)",
        "",
        "   mani/partita    partite     vinte    misurato          IC 95%     previsto",
    ]
    for row in power_table(measurement, lengths, orientation):
        predicted = f"{row.predicted * 100:7.1f}%"
        if row.measured is None:
            lines.append(
                f"   {row.hands:>12,}          -         -           -               -"
                f"   {predicted}"
            )
            continue
        outcome = row.measured
        low, high = wilson_interval(outcome.score, outcome.sessions)
        drawn = f" ({outcome.draws} patte)" if outcome.draws else ""
        lines.append(
            f"   {row.hands:>12,} {outcome.sessions:>10,} {outcome.score:>9,.1f}"
            f"   {outcome.rate * 100:6.1f}%   [{low * 100:5.1f}, {high * 100:5.1f}]"
            f"   {predicted}{drawn}"
        )
    lines += [
        "",
        "   'misurato' e' la frequenza osservata chiudendo davvero delle partite di",
        "   quella lunghezza; 'previsto' e' l'approssimazione normale, che vale anche",
        "   oltre le mani giocate. Dove le due colonne divergono (in genere sotto le",
        "   100 mani) e' la coda pesante del poker: la normale sottostima le patte e",
        "   le mani-monstre. Da qualche centinaio di mani in su devono coincidere.",
        "   L'intervallo e' calcolato come se le partite fossero indipendenti: con i",
        "   mazzi duplicati non lo sono (ogni partita ha la sua gemella a parti",
        "   invertite), quindi se sbaglia e' per eccesso.",
        "",
        f"--- quante mani deve durare una partita perche' {stronger} la vinca:",
    ]
    for target in targets:
        needed = hands_for_probability(abs(edge), sigma, target)
        if needed is None:
            lines.append(f"   {target * 100:5.1f}% delle volte  ->  mai (nessun vantaggio misurato)")
        else:
            lines.append(f"   {target * 100:5.1f}% delle volte  ->  {needed:>12,} mani")
    if session_hands:
        probability = win_probability(abs(edge), sigma, session_hands)
        lines += [
            "",
            f"   Una partita da {session_hands:,} mani -- quella che questo progetto usa per",
            f"   muovere l'Elo -- la vince il piu' forte nel {probability * 100:.1f}% dei casi:",
            f"   {(1 - probability) * 100:.1f} volte su 100 il rating si muove dalla parte",
            f"   sbagliata, e {weaker} guadagna punti su {stronger}.",
        ]
    return lines


# ---- playing ---------------------------------------------------------------
# torch is imported lazily below, never at module import time.


@dataclass(frozen=True)
class StreamJob:
    """One stream's worth of work, in picklable primitives.

    Everything crosses a process boundary as paths and ints, and each worker
    loads the checkpoints itself: `make_policy_fn` returns a closure, which is
    not picklable, so the models cannot be handed to a worker ready-made.
    """

    index: int
    seed: int
    hands: int
    seats_a: tuple[int, ...]
    seats_b: tuple[int, ...]
    field_seats: tuple[int, ...]
    field_paths: tuple[str, ...]
    path_a: str
    path_b: str
    players: int
    stack: int
    small_blind: int
    big_blind: int
    device: str
    mirror: bool

    @property
    def game(self) -> GameConfig:
        return GameConfig(
            num_players=self.players,
            starting_stack=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )


_POLICY_CACHE: dict[tuple[str, str], object] = {}


def _policy(path: str, device: str):
    """A frozen checkpoint as a policy callable, loaded once per process."""
    key = (path, device)
    if key not in _POLICY_CACHE:
        from pokerlab.rl.policy import make_policy_fn
        from pokerlab.rl.ppo import build_model_from_checkpoint

        model, _checkpoint = build_model_from_checkpoint(path, device=device)
        _POLICY_CACHE[key] = make_policy_fn(model, device=device)
    return _POLICY_CACHE[key]


def _play(
    game: GameConfig,
    policy_by_seat: dict[int, object],
    plus: Sequence[int],
    minus: Sequence[int],
    *,
    hands: int,
    seed: int,
) -> list[float]:
    """Play `hands` hands and return the per-hand chip delta of `plus` over `minus`.

    Stacks are reset after every hand, which is what makes the hands
    independent and a block of them a session -- the same `rebuy=True`
    convention the collector and the benchmark use.
    """
    from pokerlab.players.rl_agent import RLAgentPlayer

    players = [
        RLAgentPlayer(
            f"s{seat}",
            f"S{seat}",
            policy_fn=policy_by_seat[seat],
            big_blind=game.big_blind,
            starting_stack=game.starting_stack,
        )
        for seat in range(game.num_players)
    ]
    table = Table(game, players, rng=random.Random(seed))
    scores: list[float] = []
    for _hand in range(hands):
        before = list(table.stacks)
        table.play_hand()
        won = sum(table.stacks[seat] - before[seat] for seat in plus)
        lost = sum(table.stacks[seat] - before[seat] for seat in minus)
        scores.append(float(won - lost))
        table.stacks = [game.starting_stack] * game.num_players
    return scores


def run_stream(job: StreamJob) -> tuple[list[float], list[float]]:
    """One stream, played once and -- unless mirroring is off -- once more swapped.

    Both arrangements are dealt from the same seed, and `Table` touches its
    `random.Random` only to shuffle once per hand, so hand `i` of the mirrored
    run is dealt exactly the cards hand `i` of the plain run was. torch is
    reseeded before each arrangement so a stream reproduces regardless of what
    order the streams happen to run in.
    """
    import torch

    torch.set_num_threads(1)
    game = job.game
    policy_a = _policy(job.path_a, job.device)
    policy_b = _policy(job.path_b, job.device)
    seated = {seat: _policy(path, job.device) for seat, path in zip(job.field_seats, job.field_paths)}

    layout = dict(seated) | {seat: policy_a for seat in job.seats_a}
    layout |= {seat: policy_b for seat in job.seats_b}
    torch.manual_seed(job.seed)
    normal = _play(game, layout, job.seats_a, job.seats_b, hands=job.hands, seed=job.seed)

    mirrored: list[float] = []
    if job.mirror:
        swapped = dict(seated) | {seat: policy_b for seat in job.seats_a}
        swapped |= {seat: policy_a for seat in job.seats_b}
        torch.manual_seed(job.seed)
        raw = _play(game, swapped, job.seats_a, job.seats_b, hands=job.hands, seed=job.seed)
        # The returned delta is "the seats A sat in, minus the seats B sat in",
        # and in this arrangement those are B's and A's: negate to keep every
        # score in the file expressed from A's point of view.
        mirrored = [-value for value in raw]
    return normal, mirrored


# ---- putting a duel together -----------------------------------------------


def resolve_model(
    spec: str, *, global_dir: Path, root: Path
) -> tuple[str, Path, float | None]:
    """Turn `--model-a` into (label, path, rating).

    Three spellings, because the three are wanted in different moods: a path to
    a checkpoint, a model's label, and `#N` for the Nth best model in the global
    ranking -- which is how a question about the ranking is usually phrased
    ("is the 1st really better than the 100th?").
    """
    from pokerlab.rl.global_arena import discover_benchmark_population, discover_population
    from pokerlab.rl.global_store import load_ranking

    ranked = load_ranking(global_dir).models()
    rating_by_label = {member.label: member.rating for member in ranked}

    if spec.startswith("#"):
        index = int(spec[1:])
        if not 1 <= index <= len(ranked):
            raise SystemExit(f"la classifica ha {len(ranked)} modelli, #{index} non esiste")
        member = ranked[index - 1]
        path = Path(member.ref)
        if not path.is_file():
            path = _find_by_label(member.label, root)
        if path is None or not path.is_file():
            raise SystemExit(f"il checkpoint di {member.label} non e' piu' su disco")
        return (member.label, path, member.rating)

    candidate = Path(spec)
    if candidate.is_file():
        return (candidate.stem, candidate, rating_by_label.get(candidate.stem))

    for found in [*discover_population(root), *discover_benchmark_population(root)]:
        if found.label == spec:
            return (found.label, found.path, rating_by_label.get(found.label))
    raise SystemExit(f"non trovo un modello per '{spec}' (ne' file, ne' etichetta, ne' #rango)")


def _find_by_label(label: str, root: Path) -> Path | None:
    from pokerlab.rl.global_arena import discover_benchmark_population, discover_population

    for found in [*discover_population(root), *discover_benchmark_population(root)]:
        if found.label == label:
            return found.path
    return None


def build_jobs(
    *,
    path_a: Path,
    path_b: Path,
    field_paths: Sequence[Path],
    players: int,
    seats_per_team: int,
    stack: int,
    small_blind: int,
    big_blind: int,
    streams: int,
    hands: int,
    mirror: bool,
    device: str,
    seed: int,
) -> list[StreamJob]:
    """One job per stream, each with its own seating and its own deal seed.

    The seats are redrawn per stream rather than fixed, so a lucky seating
    cannot colour the whole experiment: position genuinely matters, and the
    question is about strength, not about who got the button.
    """
    rng = random.Random(seed)
    jobs: list[StreamJob] = []
    for index in range(streams):
        seats = list(range(players))
        rng.shuffle(seats)
        seats_a = tuple(sorted(seats[:seats_per_team]))
        seats_b = tuple(sorted(seats[seats_per_team : 2 * seats_per_team]))
        field_seats = tuple(sorted(seats[2 * seats_per_team :]))
        jobs.append(
            StreamJob(
                index=index,
                # Distinct, widely spaced seeds: two streams sharing a deal
                # sequence would be one stream counted twice.
                seed=rng.randrange(2**31),
                hands=hands,
                seats_a=seats_a,
                seats_b=seats_b,
                field_seats=field_seats,
                field_paths=tuple(str(path) for path in field_paths[: len(field_seats)]),
                path_a=str(path_a),
                path_b=str(path_b),
                players=players,
                stack=stack,
                small_blind=small_blind,
                big_blind=big_blind,
                device=device,
                mirror=mirror,
            )
        )
    return jobs


def measure(
    jobs: Sequence[StreamJob],
    *,
    label_a: str,
    label_b: str,
    jobs_in_parallel: int = 1,
    on_stream=None,
) -> DuelMeasurement:
    """Run every stream and collect the scores.

    `jobs_in_parallel` is one by default even on a 32-core box: this is meant to
    be run on a machine that is usually busy training, and each worker is a full
    torch process holding its own copy of every model.
    """
    measurement = DuelMeasurement(
        label_a=label_a,
        label_b=label_b,
        seats_per_team=len(jobs[0].seats_a),
        big_blind=jobs[0].big_blind,
    )
    if jobs_in_parallel <= 1:
        results = (run_stream(job) for job in jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs_in_parallel)
        results = executor.map(run_stream, jobs)
    for done, (normal, mirrored) in enumerate(results, start=1):
        measurement.normal.append(normal)
        if mirrored:
            measurement.mirrored.append(mirrored)
        if on_stream is not None:
            on_stream(done, len(jobs), measurement)
    if jobs_in_parallel > 1:
        executor.shutdown()
    return measurement


def save_measurement(path: Path, measurement: DuelMeasurement) -> None:
    """Write the per-hand scores so the analysis can be redone without replaying.

    The hands are the expensive part by orders of magnitude; re-reading them to
    try a different grid of session lengths costs nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label_a": measurement.label_a,
                "label_b": measurement.label_b,
                "seats_per_team": measurement.seats_per_team,
                "big_blind": measurement.big_blind,
                "normal": [[int(v) for v in stream] for stream in measurement.normal],
                "mirrored": [[int(v) for v in stream] for stream in measurement.mirrored],
            }
        ),
        encoding="utf-8",
    )


def load_measurement(path: Path) -> DuelMeasurement:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return DuelMeasurement(
        label_a=raw["label_a"],
        label_b=raw["label_b"],
        seats_per_team=raw["seats_per_team"],
        big_blind=raw["big_blind"],
        normal=[[float(v) for v in stream] for stream in raw["normal"]],
        mirrored=[[float(v) for v in stream] for stream in raw.get("mirrored", [])],
    )


def main(argv: Sequence[str] | None = None) -> int:
    from pokerlab.rl.table_mix import DEFAULT_SESSION_HANDS

    parser = argparse.ArgumentParser(
        description="Quante mani deve durare una partita perche' vinca il modello "
        "piu' forte, e con quale probabilita'.",
    )
    parser.add_argument("--model-a", default="#1",
                        help="percorso di un checkpoint, etichetta, oppure #N per l'N-esimo "
                        "della classifica globale (default: #1)")
    parser.add_argument("--model-b", default="#100",
                        help="come --model-a (default: #100)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--mode", choices=("teams", "field"), default="teams",
                        help="teams: meta' tavolo per modello, e' il confronto piu' pulito. "
                        "field: un solo seggio a testa e gli altri riempiti da modelli "
                        "terzi, cioe' esattamente cio' che misura una passata di popolazione")
    parser.add_argument("--field-dir", type=Path, default=DEFAULT_FIELD_DIR,
                        help="da dove pescare i riempitivi in --mode field (default: le "
                        "ancore congelate, che non sono mai sedute in training)")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--streams", type=int, default=DEFAULT_STREAMS,
                        help="flussi indipendenti, ognuno con la sua disposizione dei "
                        f"seggi e le sue carte (default: {DEFAULT_STREAMS})")
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS,
                        help="mani per flusso: e' anche la partita piu' lunga che si puo' "
                        f"misurare davvero (default: {DEFAULT_HANDS})")
    parser.add_argument("--no-mirror", action="store_true",
                        help="non rigiocare ogni flusso a parti invertite. Dimezza il tempo "
                        "e peggiora di un ordine di grandezza la stima di chi sia il piu' forte")
    parser.add_argument("--lengths", default=None,
                        help="lunghezze di partita nella tabella delle probabilita', "
                        "separate da virgola")
    parser.add_argument("--dist-lengths", default=None,
                        help="lunghezze di partita di cui disegnare la distribuzione "
                        "(default: "
                        + ",".join(str(n) for n in DEFAULT_DISTRIBUTION_LENGTHS) + ")")
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS_SHOWN,
                        help="quante partite per lunghezza mostrare nella distribuzione "
                        f"(default: {DEFAULT_SESSIONS_SHOWN})")
    parser.add_argument("--session-hands", type=int, default=DEFAULT_SESSION_HANDS,
                        help="la lunghezza di partita usata davvero dal progetto, "
                        "commentata in fondo al rapporto")
    parser.add_argument("--jobs", type=int, default=1,
                        help="flussi in parallelo, un processo ciascuno (~600 MB l'uno)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=None,
                        help="salva i delta mano per mano, per rianalizzarli senza rigiocare")
    parser.add_argument("--load", type=Path, default=None,
                        help="rianalizza un file salvato con --save, senza giocare nulla")
    args = parser.parse_args(argv)

    def parse_lengths(text: str | None, fallback: Sequence[int]) -> tuple[int, ...]:
        if not text:
            return tuple(fallback)
        return tuple(int(part) for part in text.split(",") if part.strip())

    lengths = parse_lengths(args.lengths, DEFAULT_LENGTHS)
    dist_lengths = parse_lengths(args.dist_lengths, DEFAULT_DISTRIBUTION_LENGTHS)

    if args.load is not None:
        measurement = load_measurement(args.load)
        print(f"=== {measurement.label_a} contro {measurement.label_b} "
              f"(rigiocato da {args.load}) ===")
        for line in format_report(
            measurement, lengths=lengths, session_hands=args.session_hands
        ):
            print(line)
        print()
        for line in format_distributions(
            measurement, lengths=dist_lengths, sessions=args.sessions
        ):
            print(line)
        return 0

    seats_per_team = args.players // 2 if args.mode == "teams" else 1
    if args.mode == "teams" and args.players % 2:
        raise SystemExit("--mode teams vuole un numero pari di giocatori: "
                         "meta' tavolo per modello")
    if args.players < 2 * seats_per_team:
        raise SystemExit(f"servono almeno {2 * seats_per_team} seggi")

    label_a, path_a, rating_a = resolve_model(
        args.model_a, global_dir=args.global_dir, root=args.root
    )
    label_b, path_b, rating_b = resolve_model(
        args.model_b, global_dir=args.global_dir, root=args.root
    )
    if path_a == path_b:
        print("NOTA: i due modelli sono lo stesso file. E' il controllo nullo: "
              "il vantaggio misurato deve risultare zero entro l'errore.")

    field_paths: list[Path] = []
    needed = args.players - 2 * seats_per_team
    if needed:
        pool = sorted(Path(args.field_dir).rglob("*.pt"))
        if len(pool) < needed:
            raise SystemExit(
                f"in {args.field_dir} ci sono {len(pool)} modelli, per un tavolo da "
                f"{args.players} con {2 * seats_per_team} seggi in gara ne servono {needed}"
            )
        field_paths = random.Random(args.seed).sample(pool, needed)

    jobs = build_jobs(
        path_a=path_a, path_b=path_b, field_paths=field_paths,
        players=args.players, seats_per_team=seats_per_team,
        stack=args.stack, small_blind=args.sb, big_blind=args.bb,
        streams=args.streams, hands=args.hands,
        mirror=not args.no_mirror, device=args.device, seed=args.seed,
    )
    total_hands = args.streams * args.hands * (1 if args.no_mirror else 2)

    print(f"=== {label_a} contro {label_b} ===")
    print(f"tavolo da {args.players}, {_plural(seats_per_team, 'seggio', 'seggi')} per modello"
          + (f", {needed} riempitivi da {args.field_dir}" if needed else "")
          + f", stack {args.stack}, blind {args.sb}/{args.bb}")
    print(f"{_plural(args.streams, 'flusso', 'flussi')} x {args.hands:,} mani"
          + ("" if args.no_mirror else " x 2 disposizioni speculari")
          + f" = {total_hands:,} mani da giocare, {args.jobs} in parallelo")
    print("(a ~50 mani/s per processo: circa "
          f"{total_hands / 50 / max(1, args.jobs) / 60:.0f} minuti)", flush=True)

    started = time.time()

    def progress(done: int, count: int, measurement: DuelMeasurement) -> None:
        elapsed = time.time() - started
        # Saved after every stream, not once at the end. A run of this size is
        # hours long, so a crash at the last stream would otherwise throw away
        # everything -- and the partial file is readable with --load while the
        # rest is still playing, which is the only way to see anything before it
        # finishes.
        if args.save is not None:
            save_measurement(args.save, measurement)
        print(f"  flusso {done}/{count}  {elapsed / 60:.1f} min trascorsi, "
              f"~{elapsed / done * (count - done) / 60:.1f} min alla fine", flush=True)

    measurement = measure(
        jobs, label_a=label_a, label_b=label_b,
        jobs_in_parallel=args.jobs, on_stream=progress,
    )
    if args.save is not None:
        save_measurement(args.save, measurement)
        print(f"delta mano per mano salvati in {args.save}")

    print()
    for line in format_report(
        measurement, lengths=lengths, rating_a=rating_a, rating_b=rating_b,
        session_hands=args.session_hands,
    ):
        print(line)
    print()
    for line in format_distributions(
        measurement, lengths=dist_lengths, sessions=args.sessions
    ):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
