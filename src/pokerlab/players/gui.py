from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any

from pokerlab.engine.actions import Action, LegalAction
from pokerlab.players.base import Observation, Player


@dataclass(frozen=True)
class GuiEvent:
    """One message from the background game thread to the GUI thread.

    kind is one of "your_turn" (payload: (Observation, list[LegalAction])),
    "action_taken" (payload: the dict from Table's on_action_applied --
    seat/player_id/name/action/record/observation, the Observation being
    the state *after* the action; fired for every player's action, human
    or bot, purely for display/logging),
    "hand_started" (payload: dict with hand_id/button_seat/sb_seat/bb_seat/
    small_blind/big_blind/hole_cards -- see Table's on_hand_started),
    "street_dealt" (payload: dict with hand_id/street/community_cards/
    betting_closed -- see Table's on_street_dealt; the only report a
    spectator gets of an all-in runout, which calls no Player at all),
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
    back an Action (e.g. from a button click). Reporting the action once it
    has been taken is deliberately *not* done here: a Player only ever sees
    the state from before its own action, so Table's `on_action_applied`
    hook publishes the "action_taken" event instead -- for the human and
    the bots alike, from one place, with the state from after the action.
    This is the only integration point a GUI needs -- Table and the rest of
    the engine are completely unaware a GUI exists: `act()` simply blocks on a
    `queue.get()` until the human's decision arrives.
    """

    def __init__(self, player_id: str, name: str, event_queue: queue.Queue[GuiEvent]) -> None:
        super().__init__(player_id, name)
        self._event_queue = event_queue
        self.decisions: queue.Queue[Action] = queue.Queue()

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self._event_queue.put(GuiEvent("your_turn", (observation, legal_actions)))
        return self.decisions.get()
