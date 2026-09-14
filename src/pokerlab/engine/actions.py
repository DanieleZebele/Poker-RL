from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_BLIND = "post_blind"  # forced action, logged like any other for a complete history


@dataclass(frozen=True)
class Action:
    """A player's decision.

    `amount` is only meaningful for BET/RAISE, where it is the *raise-to*
    level (the player's total commitment on this street after the action),
    not the incremental chips added. FOLD/CHECK/CALL/ALL_IN ignore `amount`:
    the engine derives the actual chip movement from game state.
    """

    action_type: ActionType
    amount: int = 0


@dataclass(frozen=True)
class LegalAction:
    """One action a player may currently take.

    For BET/RAISE, `min_amount`/`max_amount` bound the legal raise-to level.
    For CALL, both are set to the call-to level (informational). FOLD/CHECK
    leave both as None.
    """

    action_type: ActionType
    min_amount: int | None = None
    max_amount: int | None = None


class IllegalActionError(Exception):
    """Raised when a Player returns an Action not present in compute_legal_actions."""
