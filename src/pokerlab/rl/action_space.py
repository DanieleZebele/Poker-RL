"""Fixed-size discrete action space for an RL policy, and the mapping back to
the engine's `Action`.

A No-Limit betting round has a continuous action dimension (any raise-to level
between min_amount and the stack), which a categorical policy cannot express.
This module collapses it to `ACTION_DIM` bins: fold, check/call, a minimum
raise, seven pot-fraction raises, and all-in. Everything here is pure Python --
no numpy, no torch -- so it can be unit-tested inside the existing suite
without the `rl` extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokerlab.engine.actions import Action, ActionType, LegalAction

if TYPE_CHECKING:
    from pokerlab.players.base import Observation

POT_FRACTIONS: tuple[float, ...] = (0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0)

FOLD_BIN = 0
CHECK_CALL_BIN = 1
RAISE_MIN_BIN = 2
FIRST_FRACTION_BIN = 3
ALL_IN_BIN = FIRST_FRACTION_BIN + len(POT_FRACTIONS)
ACTION_DIM = ALL_IN_BIN + 1


def to_call_amount(observation: Observation) -> int:
    return max(0, observation.current_bet_to_match - observation.my_current_bet)


def raise_to_for_fraction(observation: Observation, fraction: float) -> int:
    """A pot-fraction raise expressed as a raise-*to* level.

    `pot_size` is every chip already committed this hand but not the chips we
    would still have to put in to call, so the pot a raise is measured against
    is `pot_size + to_call`. On a fresh street `current_bet_to_match` is 0 and
    this degenerates to `fraction * pot`.
    """
    pot_after_call = observation.pot_size + to_call_amount(observation)
    return observation.current_bet_to_match + round(fraction * pot_after_call)


def _aggressive_legal(legal_actions: list[LegalAction]) -> LegalAction | None:
    """The single BET-or-RAISE entry, if the seat has one.

    Read off `legal_actions` rather than inferred from `to_call == 0`: preflop
    the big blind faces `to_call == 0` with a live bet already posted, and
    there the aggressive option is a RAISE off its own blind, not a fresh BET.
    """
    for legal in legal_actions:
        if legal.action_type in (ActionType.BET, ActionType.RAISE):
            return legal
    return None


def _raise_levels(observation: Observation, aggressive: LegalAction) -> list[tuple[int, int]]:
    """(bin index, raise-to level) for every raise bin, in bin order."""
    levels = [(RAISE_MIN_BIN, aggressive.min_amount)]
    levels.extend(
        (FIRST_FRACTION_BIN + i, raise_to_for_fraction(observation, fraction))
        for i, fraction in enumerate(POT_FRACTIONS)
    )
    return levels


def legal_action_mask(
    observation: Observation,
    legal_actions: list[LegalAction],
    *,
    mask_dominated_folds: bool = True,
) -> list[bool]:
    """Which of the `ACTION_DIM` bins the policy may sample right now."""
    mask = [False] * ACTION_DIM
    types = {legal.action_type for legal in legal_actions}

    # Folding for free is strictly dominated by checking; leaving the bin open
    # only gives the policy a blunder to unlearn.
    fold_is_dominated = mask_dominated_folds and to_call_amount(observation) == 0
    if ActionType.FOLD in types and not fold_is_dominated:
        mask[FOLD_BIN] = True
    if types & {ActionType.CHECK, ActionType.CALL}:
        mask[CHECK_CALL_BIN] = True
    if ActionType.ALL_IN in types:
        mask[ALL_IN_BIN] = True

    aggressive = _aggressive_legal(legal_actions)
    if aggressive is not None:
        seen: set[int] = set()
        for index, raise_to in _raise_levels(observation, aggressive):
            # Out-of-range bins are masked, never clamped: clamping up would
            # duplicate RAISE_MIN and clamping down would duplicate ALL_IN, and
            # two bins mapping to one action split the policy's probability
            # mass over a choice that does not exist. Half-open on max for the
            # same reason -- raising to the shove level is legal, but it is
            # ALL_IN, which bin 10 already owns.
            if not (aggressive.min_amount <= raise_to < aggressive.max_amount):
                continue
            if raise_to in seen:
                continue
            seen.add(raise_to)
            mask[index] = True

    return mask


RAISE_BINS = tuple(range(RAISE_MIN_BIN, ALL_IN_BIN))  # min-raise and the pot fractions, not the shove
# How a raise's size is chosen once a raise bin is drawn (`action_index_to_action`): the bin's own
# level, or the mean of every legal raise bin's level weighted by its probability.
RAISE_SIZINGS = ("bin", "weighted")

# How the models of this process size a raise when they play: set once by a CLI from
# `--raise-sizing` (`set_raise_sizing`), read by `policy.make_policy_fn` when it is not told. A
# process setting rather than a parameter of every function that builds a player, because it has to
# reach every player of every rated session, pool and shard alike; forked copies inherit it, and
# the shards of a pass are handed the flag. The learner a collector trains never uses it: PPO
# trains on the bins it drew. Here, in a module with no torch, so the torch-free CLIs can set it.
_RAISE_SIZING = "bin"


def set_raise_sizing(mode: str) -> None:
    global _RAISE_SIZING
    if mode not in RAISE_SIZINGS:
        raise ValueError(f"raise sizing {mode!r} is not one of {RAISE_SIZINGS}")
    _RAISE_SIZING = mode


def raise_sizing() -> str:
    return _RAISE_SIZING


def add_raise_sizing_argument(parser) -> None:
    """`--raise-sizing`, for every CLI that plays the models (and `poker-loop`, which forwards it)."""
    parser.add_argument(
        "--raise-sizing", choices=RAISE_SIZINGS, default="bin",
        help="how the models size a raise when they play (rated sessions, the pool's opponents, the "
        "GUI): bin, the drawn bin's level, or weighted, the mean of every open raise bin's level "
        "weighted by its probability; the all-in stays a bin. The learner trains on bins either way",
    )


def weighted_raise_to(
    probabilities: list[float], observation: Observation, legal_actions: list[LegalAction], mask: list[bool]
) -> int:
    """The raise-to level of a weighted raise: the mean of the levels of the open raise bins (the
    min-raise and the pot fractions, never the all-in), each weighted by its probability, rounded
    to a chip. Being a mean of legal levels it is legal itself."""
    aggressive = _aggressive_legal(legal_actions)
    if aggressive is None:
        raise ValueError("a weighted raise needs a BET or RAISE to be legal")
    levels = dict(_raise_levels(observation, aggressive))
    open_bins = [index for index in RAISE_BINS if mask[index]]
    total = sum(probabilities[index] for index in open_bins)
    if total <= 0:
        raise ValueError("a weighted raise needs some probability on an open raise bin")
    mean = sum(probabilities[index] * levels[index] for index in open_bins) / total
    return min(max(round(mean), aggressive.min_amount), aggressive.max_amount)


def action_index_to_action(
    index: int,
    observation: Observation,
    legal_actions: list[LegalAction],
    *,
    weights: tuple[list[float], list[bool]] | None = None,
) -> Action:
    """Turn a sampled bin index into the engine's `Action`.

    With `weights` (the policy's probabilities and the mask it drew from), a raise bin raises to
    `weighted_raise_to` instead of its own level: the policy decided to raise with the summed
    probability of the raise bins, and sizes the raise with all of them."""
    if index == FOLD_BIN:
        return Action(ActionType.FOLD)
    if index == CHECK_CALL_BIN:
        types = {legal.action_type for legal in legal_actions}
        # CALL/ALL_IN carry no amount: the engine derives the chips.
        return Action(ActionType.CHECK if ActionType.CHECK in types else ActionType.CALL)
    if index == ALL_IN_BIN:
        return Action(ActionType.ALL_IN)

    aggressive = _aggressive_legal(legal_actions)
    if aggressive is None:
        raise ValueError(f"action bin {index} is a raise, but no BET/RAISE is legal here")
    if weights is not None:
        probabilities, mask = weights
        return Action(aggressive.action_type, weighted_raise_to(probabilities, observation, legal_actions, mask))
    return Action(aggressive.action_type, dict(_raise_levels(observation, aggressive))[index])


def describe_action_bin(index: int) -> str:
    if index == FOLD_BIN:
        return "FOLD"
    if index == CHECK_CALL_BIN:
        return "CHECK/CALL"
    if index == RAISE_MIN_BIN:
        return "RAISE min"
    if index == ALL_IN_BIN:
        return "ALL_IN"
    return f"RAISE {POT_FRACTIONS[index - FIRST_FRACTION_BIN]:.0%} pot"
