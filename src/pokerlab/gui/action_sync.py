"""Rebuild the actions of a hand from what the poker client shows.

The spot screen keeps a script -- the actions so far, in the engine's own turn
order -- and replays it through a real `Table`. This module extends that script
from a *picture* of the table: each seat's bet in front of it, who is out of the
hand, how many board cards there are.

It compares **states, not events**. Two readings are two seconds apart and
several players can act in between, so nothing here tries to see an action
happen. Instead, while the engine's player to act can be shown to have acted
already -- the picture is ahead of the engine -- that action is appended and the
engine asked again; the first time there is no such proof, it stops and waits
for the next reading. Whoever acted in between is recovered then.

What counts as proof that the seat to act (`p`) has acted:

- `p` is shown out of the hand: FOLD (after first matching a bet it had already
  put in, if its chips in front exceed what the engine has);
- `p`'s chips in front exceed the engine's: CALL if they match the bet to call,
  BET/RAISE to that amount if above it, ALL_IN when that is all `p` has;
- the board is further along than the engine's street: the street closed, so
  `p` did what closes it -- CALL when facing a bet, CHECK otherwise (a raise
  whose chips have since been swept into the pot cannot be seen, and is not
  invented);
- someone due to act *after* `p` this street has acted (chips in front above
  the engine's, or out of the hand): `p` acted first, and with chips unchanged
  that can only have been a CHECK;
- your countdown bar is showing (`TableView.my_turn`) while the engine still
  waits on `p`, someone before you: the action reached you, so `p` acted -- a
  CHECK with chips unchanged. A check leaves nothing on the table, so without
  this a round of checks was only discovered when the next card came.

Pure Python over `gui/spot.py`, no Tk, so every rule is tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.gui.spot import Spot, SpotState, replay

STREET_BY_BOARD = {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}
_STREET_ORDER = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER, Street.SHOWDOWN]
# Enough for any hand: it stops far sooner, at the first seat with no proof.
MAX_STEPS = 60


@dataclass
class TableView:
    """The picture of the table, in the engine's seat numbers.

    `bets`: chips in front of each seat whose amount was read (seats missing
    are unknown, not zero). `out`: seats shown out of the hand. `board_cards`:
    how many board cards are showing, or None if the board was not read."""

    bets: dict[int, int] = field(default_factory=dict)
    out: set[int] = field(default_factory=set)
    board_cards: int | None = None
    # The countdown bar is on your seat: the action has reached you, so every
    # seat the engine still has before you has acted. None when not read.
    my_turn: bool | None = None


@dataclass
class SyncResult:
    actions: list[tuple[int, Action]]  # appended, with the seat that took each
    waiting_for: int | None  # the seat the engine waits on, None if the hand ended
    note: str  # why it stopped, for the status line


def _street_index(street: Street) -> int:
    return _STREET_ORDER.index(street)


def _seats_after(state: SpotState, seat: int) -> list[int]:
    """The seats still to act after `seat`, in turn order (active ones only)."""
    seats = sorted(state.bets)
    active = {
        info.seat for info in state.observation.seats if info.status is PlayerStatus.ACTIVE
    } if state.observation else set(seats)
    start = seats.index(seat) if seat in seats else 0
    ordered = seats[start + 1:] + seats[:start]
    return [s for s in ordered if s in active]


def _legal(state: SpotState, kind: ActionType):
    return next((legal for legal in state.legal_actions if legal.action_type is kind), None)


def _to_amount(state: SpotState, seat: int, target: int) -> Action | None:
    """The action that leaves `seat` with `target` chips in front, if legal."""
    current = state.bets.get(seat, 0)
    facing = max(state.bets.values(), default=0)
    stack = state.stacks.get(seat, 0)
    if target == current + stack:
        return Action(ActionType.ALL_IN) if _legal(state, ActionType.ALL_IN) else None
    if target == facing and _legal(state, ActionType.CALL):
        return Action(ActionType.CALL)
    for kind in (ActionType.BET, ActionType.RAISE):
        legal = _legal(state, kind)
        if legal is not None and legal.min_amount <= target <= legal.max_amount:
            return Action(kind, target)
    return None


def next_action(state: SpotState, view: TableView, my_seat: int | None = None) -> tuple[Action | None, str]:
    """The action the picture proves the seat to act has taken, or None and why not."""
    seat = state.to_act
    if view.my_turn and seat == my_seat:
        return None, "tocca a te"
    current = state.bets.get(seat, 0)
    facing = max(state.bets.values(), default=0)
    shown = view.bets.get(seat)
    ahead = (
        view.board_cards in STREET_BY_BOARD
        and _street_index(STREET_BY_BOARD[view.board_cards]) > _street_index(state.street)
    )
    if shown is not None and shown > current and not ahead:
        # Chips in front beyond the engine's: a call, bet or raise to that amount.
        action = _to_amount(state, seat, shown)
        return (action, "") if action else (None, f"importo {shown} non coerente per il posto {seat}")
    if seat in view.out:
        if facing == current and _legal(state, ActionType.CHECK):
            # Out with nothing to call: it checked, and folded later on.
            return (Action(ActionType.CHECK), "") if ahead else (Action(ActionType.FOLD), "")
        return Action(ActionType.FOLD), ""
    if ahead:
        if facing > current:
            return Action(ActionType.CALL) if _legal(state, ActionType.CALL) else Action(ActionType.ALL_IN), ""
        return Action(ActionType.CHECK), ""
    later_acted = any(
        s in view.out or (view.bets.get(s) is not None and view.bets[s] > state.bets.get(s, 0))
        for s in _seats_after(state, seat)
    )
    # Your timer showing while the engine still waits on someone else: the
    # action went round to you, so they acted -- the evidence a check never
    # leaves on the table by itself.
    later_acted = later_acted or (bool(view.my_turn) and my_seat is not None and seat != my_seat)
    if later_acted and facing == current and _legal(state, ActionType.CHECK):
        return Action(ActionType.CHECK), ""
    return None, "attendo"


def sync_actions(spot: Spot, view: TableView) -> tuple[Spot, SyncResult]:
    """Extend `spot.script` as far as the picture proves; the new spot and what
    was added. A step the engine refuses ends it, without the step."""
    added: list[tuple[int, Action]] = []
    note = "attendo"
    for _ in range(MAX_STEPS):
        state = replay(spot)
        if state.finished or state.to_act is None:
            return spot, SyncResult(added, None, "mano conclusa")
        action, note = next_action(state, view, spot.my_seat)
        if action is None:
            return spot, SyncResult(added, state.to_act, note or "attendo")
        candidate = replace(spot, script=[*spot.script, action])
        if replay(candidate).invalid_from is not None:
            return spot, SyncResult(added, state.to_act, f"azione rifiutata per il posto {state.to_act}")
        added.append((state.to_act, action))
        spot = candidate
    return spot, SyncResult(added, None, "troppi passi, interrotto")
