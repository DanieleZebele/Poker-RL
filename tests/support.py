"""Minimal test-only players: cheap, deterministic opponents for fuzzing
engine correctness (chip conservation, action legality).

These replace what `players/scripted.py` used to provide before the
hand-coded bot catalog was removed from the product (see CLAUDE.md,
"Heuristic bots, removed"). Nothing here is product-facing -- it exists
purely so engine-level tests keep a cheap opponent to play against, the same
way they always have.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.players.base import Observation, Player

Strategy = Callable[[Observation, list[LegalAction]], Action]


class _StrategyPlayer(Player):
    def __init__(self, player_id: str, name: str, strategy: Strategy) -> None:
        super().__init__(player_id, name)
        self._strategy = strategy

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        return self._strategy(observation, legal_actions)


def make_always_call_bot(player_id: str, name: str = "AlwaysCallBot") -> Player:
    """Never voluntarily folds: checks or calls whenever possible, and goes
    all-in rather than fold if it can't afford a full call."""

    def strategy(observation: Observation, legal_actions: list[LegalAction]) -> Action:
        types = {la.action_type for la in legal_actions}
        if ActionType.CHECK in types:
            return Action(ActionType.CHECK)
        if ActionType.CALL in types:
            return Action(ActionType.CALL)
        if ActionType.ALL_IN in types:
            return Action(ActionType.ALL_IN)
        return Action(ActionType.FOLD)

    return _StrategyPlayer(player_id, name, strategy)


def make_random_legal_bot(
    player_id: str, name: str = "RandomBot", rng: random.Random | None = None
) -> Player:
    """Picks uniformly among whatever is currently legal; useful for
    fuzz-testing the engine (see test_full_hand_flow.py)."""
    rng = rng if rng is not None else random.Random()

    def strategy(observation: Observation, legal_actions: list[LegalAction]) -> Action:
        choice = rng.choice(legal_actions)
        if choice.action_type in (ActionType.BET, ActionType.RAISE):
            assert choice.min_amount is not None and choice.max_amount is not None
            amount = rng.randint(choice.min_amount, choice.max_amount)
            return Action(choice.action_type, amount=amount)
        return Action(choice.action_type)

    return _StrategyPlayer(player_id, name, strategy)
