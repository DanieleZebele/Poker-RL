"""Is the critic's target balanced across table sizes and stack depths?

The reward is one constant for every hand (`SelfPlayCollector.reward_scale`), but
what a hand can win or lose is bounded by the stacks, so the spread of the value
target is not the same everywhere: a short-stacked or heads-up hand moves far
fewer big blinds than a deep one at a full table. If the spread differs a lot
between groups, the squared value loss is dominated by the widest group and the
critic learns the others worse. This module measures it, per iteration, from the
rollouts that were collected anyway.

**What it reports, per group.** The decisions it holds, the standard deviation of
the target (the GAE return, in the unit the critic predicts) and the explained
variance of the values the policy produced against that target
(`1 - Var(target - value) / Var(target)`: 0 is no better than predicting the mean,
negative is worse). Groups are by table size (2-9) and by effective stack
(`STACK_EDGES_BB`). The spread across groups is the largest target sd over the
smallest; around 2x or more is the threshold the TODO sets for needing a scale per
group (or PopArt).

**What it is not.** The values are the ones the policy saw while *collecting*, so
this describes the critic that acted, before the update that follows. Every
decision is one sample, which is how the loss weighs them: a hand with many
decisions counts for more than a hand with one.

Pure Python, like `features.py`: the arithmetic belongs in the ordinary test suite.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Only for the annotation: the monitor and the dashboard read this module's log
# lines and have no use for the engine.
if TYPE_CHECKING:
    from pokerlab.rl.rollout import HandTrajectory

# Effective stack in big blinds: short, medium, deep. 10 bb is where a hand is
# mostly push-or-fold; 30 bb is where postflop play starts to have room.
STACK_EDGES_BB = (10.0, 30.0)
STACK_LABELS = ("<10", "10-30", "30+")
# A group with fewer decisions than this has too noisy a standard deviation to be
# compared with the others (it is shown, but left out of the spread).
MIN_DECISIONS = 100


def stack_label(effective_stack_bb: float) -> str:
    for edge, label in zip(STACK_EDGES_BB, STACK_LABELS):
        if effective_stack_bb < edge:
            return label
    return STACK_LABELS[-1]


@dataclass(frozen=True)
class GroupStats:
    decisions: int
    target_sd: float
    explained_variance: float | None  # None when the target has no spread


@dataclass(frozen=True)
class ValueDiagnostics:
    by_size: dict[int, GroupStats]
    by_stack: dict[str, GroupStats]

    @staticmethod
    def spread(groups: Iterable[GroupStats]) -> float | None:
        """Largest target sd over the smallest, among groups large enough to compare.

        None with fewer than two such groups, or when the smallest has no spread.
        """
        sds = [g.target_sd for g in groups if g.decisions >= MIN_DECISIONS]
        if len(sds) < 2 or min(sds) <= 0:
            return None
        return max(sds) / min(sds)

    @property
    def size_spread(self) -> float | None:
        return self.spread(self.by_size.values())

    @property
    def stack_spread(self) -> float | None:
        return self.spread(self.by_stack.values())


class _Accumulator:
    """Running sums for one group: enough for the sd and the explained variance."""

    __slots__ = ("error", "error_sq", "n", "target", "target_sq")

    def __init__(self) -> None:
        self.n = 0
        self.target = self.target_sq = self.error = self.error_sq = 0.0

    def add(self, value: float, target: float) -> None:
        residual = target - value
        self.n += 1
        self.target += target
        self.target_sq += target * target
        self.error += residual
        self.error_sq += residual * residual

    def stats(self) -> GroupStats:
        mean = self.target / self.n
        variance = max(self.target_sq / self.n - mean * mean, 0.0)
        if variance <= 0.0:
            return GroupStats(self.n, 0.0, None)
        error_mean = self.error / self.n
        error_variance = max(self.error_sq / self.n - error_mean * error_mean, 0.0)
        return GroupStats(self.n, math.sqrt(variance), 1.0 - error_variance / variance)


def value_diagnostics(trajectories: Iterable[HandTrajectory]) -> ValueDiagnostics:
    """Group every decision's `(value, return)` by table size and effective stack."""
    sizes: dict[int, _Accumulator] = {}
    stacks: dict[str, _Accumulator] = {}
    for trajectory in trajectories:
        size_group = sizes.setdefault(trajectory.num_players, _Accumulator())
        stack_group = stacks.setdefault(stack_label(trajectory.effective_stack_bb), _Accumulator())
        for decision, target in zip(trajectory.decisions, trajectory.returns):
            size_group.add(decision.value, target)
            stack_group.add(decision.value, target)
    return ValueDiagnostics(
        by_size={size: group.stats() for size, group in sorted(sizes.items())},
        by_stack={label: stacks[label].stats() for label in STACK_LABELS if label in stacks},
    )


def _cell(name: str, group: GroupStats) -> str:
    variance = "n/a" if group.explained_variance is None else f"{group.explained_variance:+.2f}"
    return f"{name} sd {group.target_sd:.3f} ev {variance} n {group.decisions}"


def _spread_text(spread: float | None) -> str:
    return "-" if spread is None else f"{spread:.1f}x"


def format_value_diagnostics(diagnostics: ValueDiagnostics) -> list[str]:
    """Two lines, both starting with `valore` so nothing that parses the `iter`
    lines of a worker's log can mistake them for one."""
    sizes = " | ".join(_cell(str(size), group) for size, group in diagnostics.by_size.items())
    stacks = " | ".join(_cell(label, group) for label, group in diagnostics.by_stack.items())
    return [
        f"valore per tavolo: {sizes}  (spread {_spread_text(diagnostics.size_spread)})",
        f"valore per stack (bb): {stacks}  (spread {_spread_text(diagnostics.stack_spread)})",
    ]


# ---- reading the lines back --------------------------------------------------
#
# Written and parsed here, next to `format_value_diagnostics`, for the same reason
# `rl/phases.py` holds both halves of its markers: a line one module prints and
# another one parses cannot drift apart if they live in one file.

SIZE_KIND = "size"
STACK_KIND = "stack"
_KINDS = {"tavolo": SIZE_KIND, "stack (bb)": STACK_KIND}
_LINE = re.compile(r"^valore per (tavolo|stack \(bb\)): (.*?)\s+\(spread (-|\d+(?:\.\d+)?)x?\)\s*$")
_CELL = re.compile(r"^(\S+) sd (\d+(?:\.\d+)?) ev (n/a|[+-]\d+(?:\.\d+)?) n (\d+)$")


@dataclass(frozen=True)
class ParsedValueLine:
    kind: str  # SIZE_KIND or STACK_KIND
    groups: dict[str, GroupStats]
    spread: float | None


def parse_value_line(line: str) -> ParsedValueLine | None:
    """The inverse of one line of `format_value_diagnostics`, or None.

    Anchored at both ends, so a line the worker had only half written when the
    log was read does not match and is dropped rather than parsed into a wrong
    point -- the same rule as the `iter` lines.
    """
    matched = _LINE.match(line)
    if matched is None:
        return None
    groups: dict[str, GroupStats] = {}
    for cell in matched.group(2).split(" | "):
        parts = _CELL.match(cell.strip())
        if parts is None:
            return None
        name, sd, variance, decisions = parts.groups()
        groups[name] = GroupStats(
            decisions=int(decisions),
            target_sd=float(sd),
            explained_variance=None if variance == "n/a" else float(variance),
        )
    spread = None if matched.group(3) == "-" else float(matched.group(3))
    return ParsedValueLine(_KINDS[matched.group(1)], groups, spread)
