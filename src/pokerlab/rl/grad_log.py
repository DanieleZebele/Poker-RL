"""How often the gradient-norm clips cut a step, as one line of a training log.

`ppo_update` clips the gradient of every minibatch step, the policy's weights to
`policy_max_grad_norm` and the critic's to `critic_max_grad_norm`: the two networks share
no weight, and their gradients live on different scales, so one clip over both would cut
each by a factor the other decides. A step whose norm is above its threshold is shortened
to it. With Adam, which divides every parameter's step by a running RMS of its own
gradient, a cut by about the same factor on every step cancels out; what a cut changes is
the steps whose factor differs from the usual one. The norm is the one *before* the cut,
which is what `clip_grad_norm_` returns.

    gradienti: policy media 0.6122 sd 0.3139 tagliati 16.7% soglia 1 | critico media 0.1725 sd 0.1133 tagliati 0.0% soglia 1 [36 passi]

A mean alone hides how often a cut fires (half the steps at 0.1 and half at 0.7 average
under a 0.5 threshold), hence the share of steps cut next to it. The closing bracket is
what makes a line cut inside its last number detectable, as in `style_log`. Written and
parsed here, in one file, so the line `poker-train` prints and the one the dashboard reads
cannot drift apart (the rule of `value_diagnostics` and `style_log`). It starts with
`gradienti`, which no parser of the `iter` lines can mistake for one.

Pure Python, no torch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PREFIX = "gradienti:"
_NUMBER = r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
_PART = rf"media {_NUMBER} sd {_NUMBER} tagliati {_NUMBER}% soglia {_NUMBER}"
_LINE = re.compile(rf"^gradienti: policy {_PART} \| critico {_PART} \[(\d+) passi\]\s*$")


@dataclass(frozen=True)
class ClipReading:
    """One network's gradient over an iteration's minibatch steps."""

    mean: float  # mean norm before the cut
    sd: float  # its standard deviation over the steps
    clipped: float  # share of steps whose norm was above the threshold, 0..1
    threshold: float  # the network's max grad norm


@dataclass(frozen=True)
class GradientReading:
    policy: ClipReading
    critic: ClipReading
    steps: int


def _part(reading: ClipReading) -> str:
    return (
        f"media {reading.mean:.4f} sd {reading.sd:.4f} "
        f"tagliati {100 * reading.clipped:.1f}% soglia {reading.threshold:.4g}"
    )


def format_gradient_line(reading: GradientReading) -> str:
    return f"{PREFIX} policy {_part(reading.policy)} | critico {_part(reading.critic)} [{reading.steps} passi]"


def parse_gradient_line(line: str) -> GradientReading | None:
    """The inverse of `format_gradient_line`, or None.

    Anchored at both ends, so a line the worker had only half written when the log was
    read does not match and is dropped rather than parsed into a wrong point."""
    matched = _LINE.match(line)
    if matched is None:
        return None
    numbers = matched.groups()

    def part(offset: int) -> ClipReading:
        mean, sd, clipped, threshold = (float(x) for x in numbers[offset : offset + 4])
        return ClipReading(mean, sd, clipped / 100, threshold)

    return GradientReading(policy=part(0), critic=part(4), steps=int(numbers[8]))
