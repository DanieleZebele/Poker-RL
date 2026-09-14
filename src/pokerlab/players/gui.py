from __future__ import annotations

import queue
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pokerlab.engine.actions import Action, LegalAction
from pokerlab.players.base import Observation, Player


@dataclass(frozen=True)
class GuiEvent:
    """One message from the background game thread to the GUI thread.

    kind is one of "your_turn" (payload: (Observation, list[LegalAction])),
    "action_taken" (payload: (player_id, name, Observation, Action), fired
    for every player's action, human or bot, purely for display/logging),
    "hand_started" (payload: dict with hand_id/button_seat/sb_seat/bb_seat/
    small_blind/big_blind/hole_cards -- see Table's on_hand_started),
    "hand_complete" (payload: HandResult), "session_ended_early" (payload:
    number of hands actually played), or "session_complete" (no payload).
    """

    kind: str
    payload: Any = None


class GuiPlayer(Player):
    """Bridges the background game thread to a GUI's event loop.

    `act` publishes a "your_turn" GuiEvent (carrying the Observation and
    legal actions) onto the shared `event_queue` that the GUI polls, then
    blocks on its own private `decisions` queue until the GUI thread pushes
    back an Action (e.g. from a button click); once it has an answer it
    also publishes an "action_taken" event (unblocked -- the human already
    "stepped" by choosing) so the GUI's action log covers the human's own
    moves too, not just bots'. This is the only integration point a GUI
    needs -- Table and the rest of the engine are completely unaware a GUI
    exists, exactly like ManualPlayer just swaps a blocking `queue.get()`
    in for a blocking `input()`.
    """

    def __init__(self, player_id: str, name: str, event_queue: queue.Queue[GuiEvent]) -> None:
        super().__init__(player_id, name)
        self._event_queue = event_queue
        self.decisions: queue.Queue[Action] = queue.Queue()

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self._event_queue.put(GuiEvent("your_turn", (observation, legal_actions)))
        action = self.decisions.get()
        self._event_queue.put(GuiEvent("action_taken", (self.player_id, self.name, observation, action)))
        return action


class SteppingPlayer(Player):
    """Wraps another (non-human) Player so every action it takes is
    reported to the GUI, and -- while `step_mode()` currently returns True
    -- blocks after the action until something pushes onto `step_gate`.

    This lets a human watch bot decisions unfold one at a time (via a
    GUI "Avanti" button) instead of an entire hand resolving instantly, and
    (regardless of step mode) feeds the GUI's action-by-action log. When
    step mode is off, actions are only reported, never delayed -- so a
    long unattended bot-only session isn't forced to be tediously slow.
    """

    def __init__(
        self,
        inner: Player,
        event_queue: queue.Queue[GuiEvent],
        step_gate: queue.Queue[None],
        step_mode: Callable[[], bool],
    ) -> None:
        super().__init__(inner.player_id, inner.name)
        self._inner = inner
        self._event_queue = event_queue
        self._step_gate = step_gate
        self._step_mode = step_mode

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        action = self._inner.act(observation, legal_actions)
        self._event_queue.put(GuiEvent("action_taken", (self.player_id, self.name, observation, action)))
        if self._step_mode():
            self._step_gate.get()
        return action

    def notify(self, event: str, **data: object) -> None:
        self._inner.notify(event, **data)
