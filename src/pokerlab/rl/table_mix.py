"""Which tables the models play on: a size from 2 to 9 and a stack per seat.

Everything in `rl/` used to assume one `GameConfig` (6 seats, 200 chips for
everyone, reset to the same number after every hand). `TableMix` is the
distribution that replaces it: the **table size** is drawn from `weights` and
every seat's **starting stack** is drawn independently, uniform in big blinds
between `stack_min_bb` and `stack_max_bb`, decimals included.

- **Training** draws a size and a set of stacks for every hand.
- **Every rated result** (validation, benchmark, population passes) draws the
  size once per *session* and keeps it for the whole session, while the stacks
  are still redrawn every hand. A session compares the same participants over
  the same hands, which is what `pairwise_elo_delta` assumes; a size that
  changed hand by hand would have a 2-handed hand involve two of the session's
  nine models and leave every pair with a different number of shared hands.
  Over a pass the share of hands at each size is still exactly `weights`.

**Chips stay `int` inside the engine.** The stack is written in big blinds as a
float and converted to whole chips only here, when it is drawn: with the default
blinds of 50/100 a chip is a hundredth of a big blind, so 45.6 BB is 4560 chips.
The blinds are the unit that sets the resolution; nothing downstream sees a
float chip.

`starting_stack` (the *maximum* stack, in chips) is what the features are
normalised by -- a full-depth stack is 1.0 -- so it plays the role the fixed
starting stack used to, and `game.big_blind` / `game.starting_stack` keep
working wherever a normalisation constant is wanted.

Pure Python, like `features.py`: the arithmetic that decides what the models
train and are rated on belongs in the ordinary test suite.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass

from pokerlab.engine.config import GameConfig

MIN_PLAYERS = 2
MAX_PLAYERS = 9
SIZES = tuple(range(MIN_PLAYERS, MAX_PLAYERS + 1))

DEFAULT_TABLE_WEIGHTS = (25.0, 20.0, 15.0, 10.0, 10.0, 10.0, 5.0, 5.0)
# 50/100: one chip is a hundredth of a big blind, which is what lets a stack be
# written as 45.6 BB and still be a whole number of chips.
DEFAULT_SMALL_BLIND = 50
DEFAULT_BIG_BLIND = 100
DEFAULT_STACK_MIN_BB = 1.0
DEFAULT_STACK_MAX_BB = 100.0


def feature_stack_bb(checkpoint: dict) -> float:
    """The stack, in big blinds, a model's features were normalised by in training: the
    deepest stack of the mixture it trained on (`TableMix.starting_stack`), recorded in
    its metadata as `stack_max_bb`. Whatever plays a model outside training -- the GUI's
    table, the spot screen -- must normalise by this, not by the stacks at its own table:
    a table of 250 bb stacks normalised by 250 would show the model every stack and pot
    at 40% of what it learned them as."""
    metadata = checkpoint.get("metadata") or {}
    return float(metadata.get("stack_max_bb", DEFAULT_STACK_MAX_BB))

# A thousand hands per rated session. Elo reads only the *sign* of each pair's
# chip delta, so the length of a session decides how often that sign is right --
# and in a short session it very nearly is not. The spread of a short chip delta
# dwarfs the real gap between two adjacent models, so the stronger of the two
# finishes ahead barely more often than not. That does not merely add noise: Elo
# settles at the rating that reproduces the *observed* win frequency, so a small
# edge equilibrates far closer to the pool than the strength deserves and the
# whole scale comes out compressed. Longer sessions cut the spread as
# `1/sqrt(hands)` and decompress the gaps by the same factor, which is the only
# lever that does -- lowering K shrinks the jitter around the equilibrium but
# cannot move the equilibrium itself.
#
# `games` counts sessions, so the K schedule is indifferent to the length: each
# rated result simply carries more evidence at the same K. The cost is linear in
# hands; fewer, longer sessions is the better trade at a fixed hand budget.
DEFAULT_SESSION_HANDS = 1000


@dataclass(frozen=True)
class TableMix:
    """The distribution of tables, and nothing else: no state, no rng."""

    weights: tuple[float, ...] = DEFAULT_TABLE_WEIGHTS
    stack_min_bb: float = DEFAULT_STACK_MIN_BB
    stack_max_bb: float = DEFAULT_STACK_MAX_BB
    small_blind: int = DEFAULT_SMALL_BLIND
    big_blind: int = DEFAULT_BIG_BLIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", tuple(float(w) for w in self.weights))
        if len(self.weights) != len(SIZES):
            raise ValueError(
                f"table_weights needs one weight per table size {MIN_PLAYERS}-{MAX_PLAYERS} "
                f"({len(SIZES)} numbers), got {len(self.weights)}"
            )
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("table_weights must be non-negative and not all zero")
        if self.small_blind <= 0 or self.big_blind <= 0 or self.small_blind >= self.big_blind:
            raise ValueError("blinds must be positive with small_blind < big_blind")
        if self.stack_min_bb <= 0 or self.stack_min_bb > self.stack_max_bb:
            raise ValueError("stacks need 0 < stack_min_bb <= stack_max_bb")

    @property
    def starting_stack(self) -> int:
        """The deepest stack in chips: the features' normalisation constant."""
        return self._chips(self.stack_max_bb)

    @property
    def sizes(self) -> tuple[int, ...]:
        """The table sizes that can actually be drawn (weight above zero)."""
        return tuple(size for size, w in zip(SIZES, self.weights) if w > 0)

    @property
    def max_players(self) -> int:
        return self.sizes[-1]

    def config(self, num_players: int) -> GameConfig:
        """The `GameConfig` of one table of `num_players`."""
        return GameConfig(
            num_players=num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )

    def draw_size(self, rng: random.Random) -> int:
        return rng.choices(SIZES, weights=self.weights)[0]

    def draw_stacks(self, rng: random.Random, num_players: int) -> list[int]:
        """One stack in chips per seat, drawn independently."""
        return [self._chips(rng.uniform(self.stack_min_bb, self.stack_max_bb)) for _ in range(num_players)]

    def _chips(self, big_blinds: float) -> int:
        return max(1, round(big_blinds * self.big_blind))


# ---- the command line --------------------------------------------------------


def add_table_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags that describe the tables, for every CLI that plays hands."""
    parser.add_argument(
        "--table-weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_TABLE_WEIGHTS),
        help=f"weight of each table size {MIN_PLAYERS}-{MAX_PLAYERS}, in that order "
        f"({len(SIZES)} numbers, relative: they need not add up to 1)",
    )
    parser.add_argument("--stack-min-bb", type=float, default=DEFAULT_STACK_MIN_BB)
    parser.add_argument("--stack-max-bb", type=float, default=DEFAULT_STACK_MAX_BB)
    parser.add_argument(
        "--session-hands",
        type=int,
        default=DEFAULT_SESSION_HANDS,
        help="hands in one rated session: the one length every result uses "
        "(validation, the pass against the anchors, population and arena passes). "
        "It defines the Elo scale, so changing it means a fleet reset",
    )
    parser.add_argument("--sb", type=int, default=DEFAULT_SMALL_BLIND)
    parser.add_argument("--bb", type=int, default=DEFAULT_BIG_BLIND)


def table_mix_from_args(args: argparse.Namespace) -> TableMix:
    return TableMix(
        weights=tuple(args.table_weights),
        stack_min_bb=args.stack_min_bb,
        stack_max_bb=args.stack_max_bb,
        small_blind=args.sb,
        big_blind=args.bb,
    )


def table_arguments(mix: TableMix) -> list[str]:
    """`mix` as the command-line flags `add_table_arguments` parses back."""
    return [
        "--table-weights", *(repr(w) for w in mix.weights),
        "--stack-min-bb", repr(mix.stack_min_bb),
        "--stack-max-bb", repr(mix.stack_max_bb),
        "--sb", str(mix.small_blind),
        "--bb", str(mix.big_blind),
    ]


def sizes_text(weights: Sequence[float]) -> str:
    """A short readable form for a log line: `2:25% 3:20% ...`."""
    total = sum(weights)
    return " ".join(f"{size}:{w / total:.0%}" for size, w in zip(SIZES, weights) if w > 0)
