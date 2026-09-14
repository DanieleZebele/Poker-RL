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


def action_index_to_action(
    index: int, observation: Observation, legal_actions: list[LegalAction]
) -> Action:
    """Turn a sampled bin index into the engine's `Action`."""
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
