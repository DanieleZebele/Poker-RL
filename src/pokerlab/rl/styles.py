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

and, since 2026-11-17, ten axes of detail (`DETAIL_AXES`, off where only five scales are given):

- **posizione**: looser and more aggressive the later it acts (`position_of`: the button 1, the
  first seat after the blinds -1, the blinds 0), preflop; more aggressive in position postflop;
- **aggressivita_flop / _turn / _river**: the postflop aggression of one street alone (c-bet,
  second barrel, river bet);
- **bluff**: bets and raises with nothing (`strength_of`: `debole`, nothing the board does not
  have); **slowplay**: checks and calls with a strong hand (`forte`, a set or better, or two pair
  on an unpaired board) instead of betting -- negative is a player who fast-plays;
- **tre_bet**: re-raises a single raise preflop; **fold_al_tre_bet**: folds its own raise to a
  re-raise;
- **check_raise**: raises a bet after checking the same street;
- **shove_corto**: shoves instead of raising small when its stack is short (from
  `SHORT_FROM_BB` big blinds down, fully at `SHORT_FULL_BB`);

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
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from pokerlab.cards.card import Card
from pokerlab.engine.actions import FORCED_ACTIONS, ActionType
from pokerlab.engine.state import Street
from pokerlab.evaluator.evaluator import HandCategory, evaluate
from pokerlab.players.base import Observation
from pokerlab.rl.action_space import ACTION_DIM, ALL_IN_BIN, CHECK_CALL_BIN, FOLD_BIN, POT_FRACTIONS
from pokerlab.rl.hand_tiers import TIER_OF, TIERS, hand_class

LEGACY_AXES = ("larghezza", "aggressivita_preflop", "aggressivita_postflop", "tenacia", "dimensione")
# Added 2026-11-17, after the first five (so a list of five scales still means the first five,
# and the new ones are off): position, street by street, bluff against value, preflop spots.
DETAIL_AXES = (
    "posizione",
    "aggressivita_flop",
    "aggressivita_turn",
    "aggressivita_river",
    "bluff",
    "slowplay",
    "tre_bet",
    "fold_al_tre_bet",
    "check_raise",
    "shove_corto",
)
AXES = LEGACY_AXES + DETAIL_AXES
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
assert set(PATTERNS) == set(LEGACY_AXES)

# How much of `larghezza` reaches each band of starting hands: the marginal ones move, the
# best and the worst much less.
HAND_WEIGHT = dict(zip(TIERS, (0.2, 0.5, 1.0, 1.0, 0.7)))

DEFAULT_SCALES = (1.0,) * len(AXES)

# `shove_corto` acts from this many big blinds of stack down, fully at `SHORT_FULL_BB` and below.
SHORT_FROM_BB = 25.0
SHORT_FULL_BB = 5.0
_STREET_AXIS = {Street.FLOP: "aggressivita_flop", Street.TURN: "aggressivita_turn", Street.RIVER: "aggressivita_river"}
_AGGRESSIVE = {"call": -0.5, **_RAISES}


def situation_of(observation: Observation) -> str:
    if observation.street == Street.PREFLOP:
        return "preflop"
    return "facing" if observation.current_bet_to_match > observation.my_current_bet else "checked"


def position_of(observation: Observation) -> float:
    """How late I act, in [-1, 1]: the button 1, the first seat after the big blind -1, the seats
    between evenly spaced; the blinds 0 (heads-up the button 1 and the big blind 0)."""
    seats = [seat.seat for seat in observation.seats]
    start = seats.index(observation.button_seat)
    order = seats[start + 1 :] + seats[: start + 1]  # small blind, big blind, ..., button
    if len(seats) == 2:
        return 1.0 if observation.my_seat == observation.button_seat else 0.0
    index = order.index(observation.my_seat)
    if index < 2:
        return 0.0
    others = len(seats) - 2
    return 1.0 if others == 1 else -1.0 + 2.0 * (index - 2) / (others - 1)


def preflop_raises(observation: Observation) -> tuple[int, bool]:
    """How many raises the preflop has seen (a shove above the bet is one; blinds and antes are
    not), and whether one of them was mine."""
    level, raises, mine = 0, 0, False
    for record in observation.action_history:
        if record.street is not Street.PREFLOP:
            continue
        if record.action_type in FORCED_ACTIONS:
            level = max(level, record.amount)
            continue
        if record.amount > level and record.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            raises += 1
            mine = mine or record.seat == observation.my_seat
        level = max(level, record.amount)
    return raises, mine


def _board_category(board: Sequence[Card]) -> HandCategory:
    counts = sorted(Counter(card.rank for card in board).values(), reverse=True) + [0]
    if counts[0] >= 4:
        return HandCategory.FOUR_OF_A_KIND
    if counts[0] == 3:
        return HandCategory.FULL_HOUSE if counts[1] >= 2 else HandCategory.THREE_OF_A_KIND
    if counts[0] == 2:
        return HandCategory.TWO_PAIR if counts[1] == 2 else HandCategory.PAIR
    return HandCategory.HIGH_CARD


@lru_cache(maxsize=65_536)
def strength_of(hole: tuple[Card, Card], board: tuple[Card, ...]) -> str:
    """How much my two cards make of the board, after the flop: `forte` (a set or better, or two
    pair on an unpaired board, my cards in it), `media` (a pair of my own, or better than the
    board by one step), `debole` (nothing the board does not already have)."""
    mine = evaluate((*hole, *board)).category
    shared = _board_category(board)
    if mine <= shared:
        return "debole"
    if mine >= HandCategory.THREE_OF_A_KIND or (mine == HandCategory.TWO_PAIR and shared == HandCategory.HIGH_CARD):
        return "forte"
    return "media"


def _detail(axis: str, observation: Observation, situation: str) -> dict[str, float] | None:
    """The push of one of `DETAIL_AXES` at a value of 1 and a scale of 1, in this state."""
    street = observation.street
    if axis == "posizione":
        where = position_of(observation)
        if situation == "preflop":
            return {"fold": -1.0 * where, "small": 0.5 * where, "big": 0.5 * where, "allin": 0.5 * where}
        return {group: unit * where for group, unit in _RAISES.items()}
    if axis in _STREET_AXIS.values():
        return dict(_AGGRESSIVE) if _STREET_AXIS.get(street) == axis else None
    if axis in ("bluff", "slowplay"):
        if situation == "preflop":
            return None
        strength = strength_of(tuple(observation.hole_cards), tuple(observation.community_cards))
        if axis == "bluff":
            return dict(_AGGRESSIVE) if strength == "debole" else None
        return {"call": 1.0, "small": -0.5, "big": -0.5, "allin": -0.5} if strength == "forte" else None
    if axis in ("tre_bet", "fold_al_tre_bet"):
        if situation != "preflop":
            return None
        raises, mine = preflop_raises(observation)
        if axis == "tre_bet" and raises == 1 and not mine:
            return {"fold": -0.25, "call": -0.25, **_RAISES}
        if axis == "fold_al_tre_bet" and raises >= 2 and mine:
            return {"fold": 1.0, "call": -0.5, **{group: -0.5 for group in _RAISES}}
        return None
    if axis == "check_raise":
        checked = any(
            record.seat == observation.my_seat and record.street is street and record.action_type is ActionType.CHECK
            for record in observation.action_history
        )
        return dict(_AGGRESSIVE) if situation == "facing" and checked else None
    if axis == "shove_corto":
        if not observation.big_blind:
            return None
        stack_bb = (observation.my_stack + observation.my_current_bet) / observation.big_blind
        short = min(1.0, max(0.0, (SHORT_FROM_BB - stack_bb) / (SHORT_FROM_BB - SHORT_FULL_BB)))
        return {"allin": 1.0 * short, "small": -0.5 * short} if short > 0 else None
    raise ValueError(f"unknown style axis {axis!r}")


def full_scales(scales: Sequence[float]) -> tuple[float, ...]:
    """One scale per axis of `AXES`: a list of the first five alone (what was written before the
    detail axes existed) leaves the detail axes at 0, off."""
    scales = tuple(scales)
    if len(scales) == len(LEGACY_AXES):
        return scales + (0.0,) * len(DETAIL_AXES)
    return scales


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
        for axis, value, scale in zip(AXES, self.axes, full_scales(scales)):
            if not value or not scale:
                continue
            pattern = PATTERNS[axis].get(situation) if axis in PATTERNS else _detail(axis, observation, situation)
            if not pattern:
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
        if len(self.scales) not in (len(LEGACY_AXES), len(AXES)) or min(self.scales) < 0:
            raise ValueError(
                f"style_scales is one non-negative number per axis {AXES} (or per one of the first "
                f"{len(LEGACY_AXES)}, which leaves the others off)"
            )
        object.__setattr__(self, "scales", full_scales(self.scales))

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
