"""Opponents with a style: a model's own logits pushed along a few axes of play.

The models of the pool are alike (the HUD studies in `studies/agents` found they barely
tell one kind of opponent from another, and reading the HUD earned nothing), so a share of
the training seats can be given a **style**: the pool model in that seat plays with a push
added to its logits, drawn at random for each table and kept for the table's hands -- long
enough for the opponent statistics to fill in, so the HUD is what tells the learner what
it is facing. Nothing is a fixed rule: a style is a point in a continuous space, most
points near the centre, so there is no archetype to memorise, only players to read.

The axes (each a number around [-1, 1], times a per-axis scale from the calibration
study, `studies/agents/style_calibration.py`):

- **larghezza** (tight <-> loose, preflop): fold against call/raise, weighted by the hand
  (`HAND_WEIGHT`): mostly the marginal hands move, a loose player does not fold aces less
  and a tight one does not open more trash -- which also makes the push cheaper;
- **aggressivita_preflop** (passive <-> aggressive): call against raise, preflop;
- **aggressivita_postflop**: check/call against bet/raise, after the flop;
- **tenacia** (gives up <-> calling station): fold against call, after the flop, facing a
  bet;
- **dimensione** (small <-> big bets): the small raise bins against the big ones;

plus a **temperature** (the logits divided by it: above 1 an erratic player, below 1 a
mechanical one) and a little **noise** on the shape of the push, so that two players at
the same point of the space are not one pattern either.

The push is added to the network's own logits, so the cards still count: only the style
bends. Pure Python, no torch: `push` is what `RLAgentPlayer(style=...)` calls with the
observation, and the policy adds the vector and divides by the temperature
(`rl/policy.py::make_policy_fn`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pokerlab.engine.state import Street
from pokerlab.players.base import Observation
from pokerlab.rl.action_space import ACTION_DIM, ALL_IN_BIN, CHECK_CALL_BIN, FOLD_BIN, POT_FRACTIONS
from pokerlab.rl.hand_tiers import TIER_OF, TIERS, hand_class

AXES = ("larghezza", "aggressivita_preflop", "aggressivita_postflop", "tenacia", "dimensione")
SITUATIONS = ("preflop", "checked", "facing")
GROUPS = ("fold", "call", "small", "big", "allin")

_BIG_FROM = 3 + POT_FRACTIONS.index(0.75)  # the bins of 75% of the pot and up


def group_of(index: int) -> str:
    """The group a bin of the action space falls in."""
    if index == FOLD_BIN:
        return "fold"
    if index == CHECK_CALL_BIN:
        return "call"
    if index == ALL_IN_BIN:
        return "allin"
    return "big" if index >= _BIG_FROM else "small"


_RAISES = {"small": 0.5, "big": 0.5, "allin": 0.5}

# axis -> situation -> group -> push for an axis value of 1 and a scale of 1.
PATTERNS: dict[str, dict[str, dict[str, float]]] = {
    "larghezza": {"preflop": {"fold": -1.0, "call": 0.5, **_RAISES}},
    "aggressivita_preflop": {"preflop": {"call": -0.5, **_RAISES}},
    "aggressivita_postflop": {
        "checked": {"call": -0.5, **_RAISES},
        "facing": {"call": -0.5, **_RAISES},
    },
    "tenacia": {"facing": {"fold": -1.0, "call": 1.0}},
    "dimensione": {
        situation: {"small": -0.5, "big": 0.5, "allin": 0.5} for situation in SITUATIONS
    },
}
assert set(PATTERNS) == set(AXES)

# How much of `larghezza` reaches each band of starting hands: the marginal ones move, the
# best and the worst much less.
HAND_WEIGHT = dict(zip(TIERS, (0.2, 0.5, 1.0, 1.0, 0.7)))

DEFAULT_SCALES = (1.0,) * len(AXES)


def situation_of(observation: Observation) -> str:
    if observation.street == Street.PREFLOP:
        return "preflop"
    return "facing" if observation.current_bet_to_match > observation.my_current_bet else "checked"


@dataclass(frozen=True)
class Style:
    """One player's style: a value per axis, a temperature, and the noise on its shape
    (one number per situation and group, `len(SITUATIONS) * len(GROUPS)`, or none)."""

    axes: tuple[float, ...] = (0.0,) * len(AXES)
    temperature: float = 1.0
    jitter: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.axes) != len(AXES):
            raise ValueError(f"a style has one value per axis {AXES}, got {len(self.axes)}")
        if self.temperature <= 0:
            raise ValueError("the temperature must be positive")
        if self.jitter and len(self.jitter) != len(SITUATIONS) * len(GROUPS):
            raise ValueError("the noise has one number per situation and group")

    def push(self, observation: Observation, scales: Sequence[float] = DEFAULT_SCALES) -> list[float]:
        """What is added to the logits of every bin, in this state."""
        situation = situation_of(observation)
        by_group = dict.fromkeys(GROUPS, 0.0)
        for axis, value, scale in zip(AXES, self.axes, scales):
            pattern = PATTERNS[axis].get(situation)
            if not pattern or not value:
                continue
            weight = HAND_WEIGHT[TIER_OF[hand_class(observation.hole_cards)]] if axis == "larghezza" else 1.0
            for group, unit in pattern.items():
                by_group[group] += value * scale * weight * unit
        if self.jitter:
            row = SITUATIONS.index(situation) * len(GROUPS)
            for offset, group in enumerate(GROUPS):
                by_group[group] += self.jitter[row + offset]
        return [by_group[group_of(index)] for index in range(ACTION_DIM)]


def style_fn(style: Style, scales: Sequence[float]) -> Callable[[Observation], tuple[list[float], float]]:
    """What `RLAgentPlayer(style=...)` takes: the push and the temperature, per state."""
    scales = tuple(scales)
    return lambda observation: (style.push(observation, scales), style.temperature)


def draw_style(
    rng: random.Random, *, spread: float, temperature_spread: float, jitter: float
) -> Style:
    """A style at random: each axis normal around 0 with sd `spread`, cut at +-1 (so most
    players are mildly bent and few are extreme), the temperature log-normal with sd
    `temperature_spread` cut to [1/2, 2], and the noise normal with sd `jitter`."""
    axes = tuple(max(-1.0, min(1.0, rng.gauss(0.0, spread))) for _ in AXES)
    temperature = math.exp(max(-math.log(2), min(math.log(2), rng.gauss(0.0, temperature_spread))))
    noise = tuple(rng.gauss(0.0, jitter) for _ in range(len(SITUATIONS) * len(GROUPS))) if jitter > 0 else ()
    return Style(axes, temperature, noise)


@dataclass(frozen=True)
class StyleConfig:
    """How a training run gives its opponents a style: what share of the pool-model seats
    get one, how far from the centre the axes are drawn (`spread`, the sd of each), the
    spread of the temperature and of the noise, and the per-axis scales the calibration
    study found (`studies/agents/style_calibration.py`). `share` 0 switches it off."""

    share: float = 0.0
    spread: float = 0.5
    temperature_spread: float = 0.2
    jitter: float = 0.1
    scales: tuple[float, ...] = DEFAULT_SCALES

    def __post_init__(self) -> None:
        if not 0.0 <= self.share <= 1.0:
            raise ValueError("style_share is a share of the opponent seats, between 0 and 1")
        if min(self.spread, self.temperature_spread, self.jitter) < 0:
            raise ValueError("style_spread, style_temperature_spread and style_jitter cannot be negative")
        if len(self.scales) != len(AXES) or min(self.scales) < 0:
            raise ValueError(f"style_scales is one non-negative number per axis {AXES}")

    @property
    def enabled(self) -> bool:
        return self.share > 0.0

    def draw(self, rng: random.Random) -> Callable[[Observation], tuple[list[float], float]] | None:
        """The style of one opponent seat for one table, or None (a plain pool model). It
        takes nothing from `rng` when switched off, so a run without styles plays the
        hands it always did."""
        if not self.enabled or rng.random() >= self.share:
            return None
        style = draw_style(rng, spread=self.spread, temperature_spread=self.temperature_spread, jitter=self.jitter)
        return style_fn(style, self.scales)


# ---- the command line --------------------------------------------------------


def add_style_arguments(parser) -> None:
    """The flags that give the opponents a style, for `poker-train` and `poker-loop`."""
    default = StyleConfig()
    parser.add_argument(
        "--style-share", type=float, default=default.share,
        help="share of the opponent seats (pool models) that play with a random style for "
        "a table's hands: the model's own logits pushed along a few axes of play, so the "
        "opponents the HUD describes really differ (0 switches it off)",
    )
    parser.add_argument(
        "--style-spread", type=float, default=default.spread,
        help="sd of each style axis (drawn normal around 0 and cut at +-1): most styled "
        "players are mildly bent, few extreme",
    )
    parser.add_argument(
        "--style-temperature-spread", type=float, default=default.temperature_spread,
        help="sd of the log of a style's temperature (cut to a factor of 2 either way)",
    )
    parser.add_argument(
        "--style-jitter", type=float, default=default.jitter,
        help="sd of the noise on the shape of a style's push, per situation and action group",
    )
    parser.add_argument(
        "--style-scales", type=float, nargs="+", default=list(default.scales),
        help=f"the strength of each axis {', '.join(AXES)}, from the calibration study "
        "(studies/agents/style_calibration.py)",
    )


def style_config_from_args(args) -> StyleConfig:
    return StyleConfig(
        share=args.style_share, spread=args.style_spread, temperature_spread=args.style_temperature_spread,
        jitter=args.style_jitter, scales=tuple(args.style_scales),
    )


def style_arguments(config: StyleConfig) -> list[str]:
    """`config` as the flags `add_style_arguments` parses back (what the loop forwards)."""
    return [
        "--style-share", repr(config.share), "--style-spread", repr(config.spread),
        "--style-temperature-spread", repr(config.temperature_spread), "--style-jitter", repr(config.jitter),
        "--style-scales", *(repr(x) for x in config.scales),
    ]
