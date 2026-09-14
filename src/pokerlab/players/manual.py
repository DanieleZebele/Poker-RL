from __future__ import annotations

from pokerlab.cli.render import render_legal_actions, render_observation
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.players.base import Observation, Player


class ManualPlayer(Player):
    """A human, prompted at the terminal for every decision."""

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        print()
        print(render_observation(observation))
        print(f"{self.name}, it's your turn. " + render_legal_actions(legal_actions))

        while True:
            raw = input("> ").strip()
            try:
                index = int(raw)
                legal = legal_actions[index]
            except (ValueError, IndexError):
                print(f"Enter a number between 0 and {len(legal_actions) - 1}.")
                continue

            if legal.action_type not in (ActionType.BET, ActionType.RAISE):
                return Action(legal.action_type)

            assert legal.min_amount is not None and legal.max_amount is not None
            if legal.min_amount == legal.max_amount:
                return Action(legal.action_type, amount=legal.min_amount)

            amount_raw = input(f"amount ({legal.min_amount}-{legal.max_amount}): ").strip()
            try:
                amount = int(amount_raw)
            except ValueError:
                print("Enter a whole number.")
                continue
            if not (legal.min_amount <= amount <= legal.max_amount):
                print(f"Amount must be between {legal.min_amount} and {legal.max_amount}.")
                continue
            return Action(legal.action_type, amount=amount)
