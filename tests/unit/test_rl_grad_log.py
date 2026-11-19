"""The gradient line: what `poker-train` prints is what the dashboard reads back."""

from __future__ import annotations

import pytest

from pokerlab.rl.grad_log import (
    ClipReading,
    GradientReading,
    NormReading,
    format_gradient_line,
    parse_gradient_line,
)

READING = GradientReading(
    policy=ClipReading(mean=0.6122, sd=0.3139, clipped=0.167, threshold=1.0),
    critic=NormReading(mean=0.1725, sd=0.1133),
    steps=36,
)


def test_the_line_round_trips():
    line = format_gradient_line(READING)
    assert line.startswith("gradienti: policy ")
    back = parse_gradient_line(line)
    assert back.steps == 36
    got, want = back.policy, READING.policy
    assert (got.mean, got.sd, got.clipped, got.threshold) == pytest.approx(
        (want.mean, want.sd, want.clipped, want.threshold), abs=1e-4
    )
    assert (back.critic.mean, back.critic.sd) == pytest.approx((0.1725, 0.1133), abs=1e-4)


def test_a_line_from_before_the_critic_lost_its_clip_is_still_read():
    line = ("gradienti: policy media 0.6122 sd 0.3139 tagliati 16.7% soglia 1 | "
            "critico media 0.1725 sd 0.1133 tagliati 0.0% soglia 1 [36 passi]")
    back = parse_gradient_line(line)
    assert back.critic == NormReading(0.1725, 0.1133) and back.policy.threshold == 1.0 and back.steps == 36


def test_a_tiny_threshold_survives_the_round_trip():
    reading = GradientReading(ClipReading(0.5, 0.1, 1.0, 1e-9), NormReading(0.2, 0.1), 4)
    back = parse_gradient_line(format_gradient_line(reading))
    assert back.policy.threshold == pytest.approx(1e-9)


def test_a_half_written_line_is_not_read():
    line = format_gradient_line(READING)
    assert parse_gradient_line(line[:-1]) is None  # the bracket is missing
    assert parse_gradient_line(line[:-9]) is None  # cut inside the step count
    assert parse_gradient_line("iter    1  reward   +0.54 bb") is None
