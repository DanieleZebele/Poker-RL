"""Which way to move each hyperparameter, learned from what the children did.

The sweep (`loop.hyperparameter_plan`) moves every axis of a worker by one of a few
multipliers, independently per axis. That is a randomised experiment already: each
finished child records the step it took from its parent, the Elo it gained over that
parent and the CPU it cost (`rl/sweep_log.py`). This module reads those and tilts
the draw toward the steps that paid.

**The objective is Elo gained per unit of compute**, which is `J = gain / cost`. For a
small step in one axis, `d log J = d gain / gain - d log cost`, so the step's value in
Elo is

    g_axis = beta_axis - G * elasticity_axis

where `beta` is how much a unit of log-step moves the *gain* and `elasticity` how
much it moves the *log cost*, and `G` is the gain a typical child makes (floored at
`min_gain`, so a fleet whose children no longer improve on average does not turn the
objective upside down: with `G <= 0` more cost would look like a benefit).

**How `beta` is estimated.** A Bayesian ridge regression of the gain on the
realised log-steps of the twelve axes, with an intercept and the parent's rating as
nuisance terms. The parent's rating is there because the parents are drawn from the
top of the ranking, whose ratings are biased upward (a winner's curse), so a high
parent regresses back and a plain average gain would be negative for reasons that
have nothing to do with the settings. The prior shrinks every `beta` toward 0
(`PRIOR_SD`), which is what keeps twelve noisy coefficients from chasing noise.
Noise is large against the effect -- a gain is the difference of two ratings, each
with a standard deviation near 10 Elo, and one step of x1.2 is worth a couple of
Elo -- so the estimate firms up over hundreds of children, not a handful.

**How `elasticity` is estimated.** A ridge regression of the log CPU time on the log
*levels* of the axes. Cost is nearly deterministic in the settings (it is CPU time,
not wall time), so this is precise, and it is what lets the objective discourage
`hands` without anyone writing "hands is expensive" into the code.

**Thompson sampling, one posterior draw per worker.** Each worker draws its own
`beta` from the posterior, so where the evidence is weak the draws disagree and the
fleet keeps trying both directions, and where it is strong they agree and the fleet
follows. For each axis the worker then takes the multiplier that the draw says is
best -- the largest if `g > 0`, the smallest otherwise. On top of that **every axis is
drawn uniformly at random with probability `explore`**: if every worker moved an axis
the same way, that axis would stop varying and its effect would stop being
estimable. Until `warmup` children have been seen the draw is uniform, exactly as the
sweep behaved before this existed.

**What it cannot see.** The model is linear in the step, fitted over a window of the
most recent children, so it measures the *average* slope where the fleet has been
lately. As a lineage nears a good value the slope there flattens and the average
follows it down; it does not know where the optimum is, only which way is uphill.

Pure Python, no torch and no numpy: the matrices are 14x14.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from pokerlab.rl.sweep_log import SweepObservation

# How far a unit of log-step is expected to move the gain, in Elo, before any data:
# the standard deviation of the prior on every `beta`. A step of x1.2 is a log-step
# of 0.18, so 15 means "a couple of Elo, rarely much more".
PRIOR_SD = 15.0
NUISANCE_PRECISION = 1e-6  # the intercept and the parent's rating are left free
NOISE_FLOOR = 1.0  # Elo squared: the noise estimate never claims to be exact
COST_RIDGE = 0.5  # shrinks the elasticities toward 0, in sums of squared log-levels
LEVEL_FLOOR = 1e-3  # an axis that lives at 0 (entropy) has a level of ln(LEVEL_FLOOR)
SEED_STEP = math.log(1.2)  # the step recorded for a move off (or onto) exactly 0
MIN_FIT = 20  # fewer usable children than this and there is nothing to fit

DEFAULT_EXPLORE = 0.35
DEFAULT_WARMUP = 150
DEFAULT_WINDOW = 1500
DEFAULT_MIN_GAIN = 1.0


# ---- the axes, as the regression sees them -------------------------------------


def _effective(axis: str, value: float, complement_axes: frozenset[str]) -> float:
    """The quantity an axis is moved in: `1 - value` for the GAE lambda."""
    return 1.0 - value if axis in complement_axes else value


def log_step(axis: str, parent: float, child: float, complement_axes: frozenset[str]) -> float:
    """The realised log-step from the parent's value to the child's.

    Realised, not planned: a count that was rounded or a probability that was capped
    moved by what the child actually trained with, and that is what had the effect.
    """
    before = _effective(axis, parent, complement_axes)
    after = _effective(axis, child, complement_axes)
    if before > 0 and after > 0:
        return math.log(after / before)
    if after > 0:
        return SEED_STEP
    if before > 0:
        return -SEED_STEP
    return 0.0


def log_level(axis: str, value: float, complement_axes: frozenset[str]) -> float:
    return math.log(max(_effective(axis, value, complement_axes), LEVEL_FLOOR))


# ---- small dense linear algebra --------------------------------------------------


def _cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            partial = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if partial <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lower[i][i] = math.sqrt(partial)
            else:
                lower[i][j] = partial / lower[j][j]
    return lower


def _solve(lower: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve `L L^T x = rhs`."""
    size = len(lower)
    forward = [0.0] * size
    for i in range(size):
        forward[i] = (rhs[i] - sum(lower[i][k] * forward[k] for k in range(i))) / lower[i][i]
    solution = [0.0] * size
    for i in reversed(range(size)):
        solution[i] = (
            forward[i] - sum(lower[k][i] * solution[k] for k in range(i + 1, size))
        ) / lower[i][i]
    return solution


def _inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    lower = _cholesky(matrix)
    size = len(matrix)
    columns = [_solve(lower, [1.0 if r == c else 0.0 for r in range(size)]) for c in range(size)]
    return [[columns[c][r] for c in range(size)] for r in range(size)]


def _gram(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    width = len(rows[0])
    gram = [[0.0] * width for _ in range(width)]
    for row in rows:
        for i in range(width):
            xi = row[i]
            if xi == 0.0:
                continue
            for j in range(i + 1):
                gram[i][j] += xi * row[j]
    for i in range(width):
        for j in range(i):
            gram[j][i] = gram[i][j]
    return gram


def _times(rows: Sequence[Sequence[float]], values: Sequence[float]) -> list[float]:
    width = len(rows[0])
    out = [0.0] * width
    for row, value in zip(rows, values):
        for i in range(width):
            out[i] += row[i] * value
    return out


# ---- the estimate --------------------------------------------------------------


@dataclass(frozen=True)
class SweepEstimate:
    axes: tuple[str, ...]
    observations: int
    beta: tuple[float, ...]  # Elo of gain per unit of log-step, per axis
    covariance: tuple[tuple[float, ...], ...]  # posterior covariance of `beta`
    elasticity: tuple[float, ...]  # d log(cost) / d log(step), per axis
    mean_gain: float
    mean_cost: float  # CPU seconds
    noise_sd: float  # Elo, of a child's gain around the model
    gain_scale: float  # `G`: what a unit of relative cost is worth, in Elo

    def net(self, index: int) -> float:
        """Elo per unit of log-step once the cost is charged."""
        return self.beta[index] - self.gain_scale * self.elasticity[index]

    def beta_sd(self, index: int) -> float:
        return math.sqrt(max(self.covariance[index][index], 0.0))


def fit_estimate(
    observations: Sequence[SweepObservation],
    axes: Sequence[str],
    complement_axes: frozenset[str] = frozenset(),
    *,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> SweepEstimate | None:
    """The estimate from `observations`, or None when there is too little to fit."""
    axes = tuple(axes)
    usable = [
        o
        for o in observations
        if o.cpu_seconds > 0
        and all(a in o.settings and a in o.parent_settings for a in axes)
    ]
    count = len(usable)
    if count < MIN_FIT or not axes:
        return None
    width = len(axes)

    gains = [o.gain for o in usable]
    mean_gain = sum(gains) / count
    parent_ratings = [o.parent_rating for o in usable]
    mean_parent = sum(parent_ratings) / count
    sd_parent = math.sqrt(sum((r - mean_parent) ** 2 for r in parent_ratings) / count) or 1.0

    design = [
        [log_step(a, o.parent_settings[a], o.settings[a], complement_axes) for a in axes]
        + [1.0, (o.parent_rating - mean_parent) / sd_parent]
        for o in usable
    ]
    gram = _gram(design)
    moment = _times(design, gains)
    prior = [1.0 / PRIOR_SD**2] * width + [NUISANCE_PRECISION] * 2

    noise = max(sum((g - mean_gain) ** 2 for g in gains) / count, NOISE_FLOOR)
    coefficients = [0.0] * (width + 2)
    for _ in range(3):
        precision = [
            [gram[i][j] / noise + (prior[i] if i == j else 0.0) for j in range(width + 2)]
            for i in range(width + 2)
        ]
        lower = _cholesky(precision)
        coefficients = _solve(lower, [m / noise for m in moment])
        residuals = [
            g - sum(c * x for c, x in zip(coefficients, row)) for g, row in zip(gains, design)
        ]
        noise = max(sum(r * r for r in residuals) / max(count - 2, 1), NOISE_FLOOR)
    precision = [
        [gram[i][j] / noise + (prior[i] if i == j else 0.0) for j in range(width + 2)]
        for i in range(width + 2)
    ]
    covariance = _inverse(precision)

    costs = [o.cpu_seconds for o in usable]
    log_costs = [math.log(c) for c in costs]
    mean_log_cost = sum(log_costs) / count
    levels = [
        [log_level(a, o.settings[a], complement_axes) for a in axes] for o in usable
    ]
    level_means = [sum(row[i] for row in levels) / count for i in range(width)]
    centred = [[row[i] - level_means[i] for i in range(width)] for row in levels]
    ridge = _gram(centred)
    for i in range(width):
        ridge[i][i] += COST_RIDGE
    elasticity = _solve(
        _cholesky(ridge), _times(centred, [lc - mean_log_cost for lc in log_costs])
    )

    return SweepEstimate(
        axes=axes,
        observations=count,
        beta=tuple(coefficients[:width]),
        covariance=tuple(tuple(covariance[i][:width]) for i in range(width)),
        elasticity=tuple(elasticity),
        mean_gain=mean_gain,
        mean_cost=sum(costs) / count,
        noise_sd=math.sqrt(noise),
        gain_scale=max(mean_gain, min_gain),
    )


# ---- the draw --------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPolicy:
    """How a generation's workers choose their multipliers."""

    axes: tuple[str, ...]
    multipliers: tuple[float, ...]
    explore: float
    observations: int  # children seen in the window, usable or not
    warmup: int
    estimate: SweepEstimate | None

    @property
    def guided(self) -> bool:
        return self.estimate is not None

    def draw(self, rng: random.Random) -> dict[str, float]:
        """One worker's multiplier per axis."""
        estimate = self.estimate
        if estimate is None:
            return {axis: rng.choice(self.multipliers) for axis in self.axes}
        sample = self._posterior_draw(estimate, rng)
        chosen: dict[str, float] = {}
        for index, axis in enumerate(self.axes):
            if rng.random() < self.explore:
                chosen[axis] = rng.choice(self.multipliers)
                continue
            gradient = sample[index] - estimate.gain_scale * estimate.elasticity[index]
            if gradient == 0.0:
                chosen[axis] = rng.choice(self.multipliers)
            else:
                chosen[axis] = max(self.multipliers, key=lambda m: gradient * math.log(m))
        return chosen

    @staticmethod
    def _posterior_draw(estimate: SweepEstimate, rng: random.Random) -> list[float]:
        lower = _cholesky(estimate.covariance)
        noise = [rng.gauss(0.0, 1.0) for _ in estimate.beta]
        return [
            estimate.beta[i] + sum(lower[i][k] * noise[k] for k in range(i + 1))
            for i in range(len(estimate.beta))
        ]


def build_policy(
    observations: Sequence[SweepObservation],
    axes: Sequence[str],
    complement_axes: frozenset[str],
    multipliers: Sequence[float],
    *,
    explore: float = DEFAULT_EXPLORE,
    warmup: int = DEFAULT_WARMUP,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> SweepPolicy:
    """A policy that is uniform until `warmup` children have been seen."""
    estimate = None
    if len(observations) >= warmup:
        estimate = fit_estimate(observations, axes, complement_axes, min_gain=min_gain)
    return SweepPolicy(
        axes=tuple(axes),
        multipliers=tuple(multipliers),
        explore=explore,
        observations=len(observations),
        warmup=warmup,
        estimate=estimate,
    )


# ---- what the supervisor prints --------------------------------------------------


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def report_lines(policy: SweepPolicy, *, steering: bool = True) -> list[str]:
    """The estimate, one line per axis, for the supervisor's log.

    Every effect is for a step of x1.2 -- the multiplier the sweep actually takes --
    in Elo, with the cost charged and not: a reader can see both what an axis does
    to the gain and what it does to the bill. With `steering` false the header says
    the estimate is only being watched, so a log read later cannot be mistaken for
    one that shaped the sweep.
    """
    watching = "" if steering else " (spento: solo osservazione, non guida i sorteggi)"
    estimate = policy.estimate
    if estimate is None and policy.observations >= policy.warmup:
        return [
            (
                f"  sweep: {policy.observations} figli osservati ma meno di {MIN_FIT} "
                f"utilizzabili per la stima (assi o CPU mancanti); i moltiplicatori "
                f"restano casuali{watching}"
            )
        ]
    if estimate is None:
        return [
            (
                f"  sweep: ottimizzatore in riscaldamento, {policy.observations}/{policy.warmup} "
                f"figli osservati; i moltiplicatori sono ancora casuali{watching}"
            )
        ]
    scale = math.log(1.2)
    lines = [
        (
            f"  sweep: ottimizzatore attivo su {estimate.observations} figli: guadagno medio "
            f"{estimate.mean_gain:+.1f} Elo, costo medio {estimate.mean_cost / 60:.0f} min CPU, "
            f"rumore {estimate.noise_sd:.0f} Elo; {policy.explore:.0%} degli assi casuali"
            f"{watching}"
        ),
        "    effetto di un passo x1.2, in Elo (P = probabilità che valga la pena):",
    ]
    for index, axis in enumerate(estimate.axes):
        sd = estimate.beta_sd(index) * scale
        net = estimate.net(index) * scale
        probability = _normal_cdf(net / sd) if sd > 0 else 0.5
        lines.append(
            f"    {axis:<22} lordo {estimate.beta[index] * scale:+6.2f} ±{sd:4.2f}  "
            f"costo {estimate.elasticity[index] * scale * 100:+5.1f}%  "
            f"netto {net:+6.2f}  P {probability:4.0%}"
        )
    return lines
