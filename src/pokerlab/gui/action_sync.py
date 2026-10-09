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
  this a round of checks was only discovered when the next card came;
- `p` is you and the board is further on: whatever your bar shows (it may be your turn
  on the next street already), you acted -- CALL facing a bet if you are still in the
  hand, FOLD if your cards are gone, CHECK with nothing to call;
- `p` is you, your bar was seen for this decision (`TableView.my_turn_seen`) and
  is gone, with no chips added and nothing to call: you checked. Without it your
  own check was found only when someone after you acted or the next card came,
  and until then the screen kept waiting on you;
- `p`'s stack on screen (`TableView.stacks`) is lower than the engine's, and the
  table cannot show the chips -- swept into the pot when the next card came, or a
  bet zone not read: the drop is what `p` put in, so a CALL, BET or RAISE to that
  (`_street_chips`). A bet seen in front always wins over the stack;
- the next card came and `p` still has cards in front (`TableView.in_hand`) facing a
  bet: it called, whatever its stack shows (the client may update it a moment late);
- the next card came and `p`'s stack did not move, and `p` is not shown in the hand:
  with nothing to call it checked, facing a bet it folded.

The stacks are the chips *behind*. With the board one street ahead the drop also
holds what `p` has put in on the new street, which is in front of it and is taken
off; two or more streets ahead the drop cannot be split between them, and only "no
chips at all" is used. A stack read as 0 is left out: an empty zone and an all-in
look the same.

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
    # Seats shown *in* the hand (their cards in front of them). Empty when not read.
    in_hand: set[int] = field(default_factory=set)
    board_cards: int | None = None
    # The countdown bar is on your seat: the action has reached you, so every
    # seat the engine still has before you has acted. None when not read.
    my_turn: bool | None = None
    # Your bar was seen at an earlier reading for the decision the engine is waiting on
    # now. Its going away is then your action; without having seen it, a bar not yet
    # drawn (the client is a moment behind) would read as a check you never made.
    my_turn_seen: bool = False
    # Chips behind each seat as read on screen, for the seats whose stack was read
    # (and is above 0). Missing: unknown.
    stacks: dict[int, int] = field(default_factory=dict)
    # How far two amounts may differ and still be the same: the screen writes one
    # decimal of a big blind, so half of that in chips.
    tolerance: int = 0


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


def _to_amount(state: SpotState, seat: int, target: int, tolerance: int = 0) -> Action | None:
    """The action that leaves `seat` with `target` chips in front, if legal; amounts
    within `tolerance` of the all-in, of the call or of the legal range count as them
    (a stack read to one decimal of a big blind is that close, not exact)."""
    current = state.bets.get(seat, 0)
    facing = max(state.bets.values(), default=0)
    stack = state.stacks.get(seat, 0)
    if abs(target - (current + stack)) <= tolerance:
        return Action(ActionType.ALL_IN) if _legal(state, ActionType.ALL_IN) else None
    if abs(target - facing) <= tolerance and _legal(state, ActionType.CALL):
        return Action(ActionType.CALL)
    for kind in (ActionType.BET, ActionType.RAISE):
        legal = _legal(state, kind)
        if legal is not None and legal.min_amount - tolerance <= target <= legal.max_amount + tolerance:
            return Action(kind, min(max(target, legal.min_amount), legal.max_amount))
    return None


def _call(state: SpotState) -> Action:
    return Action(ActionType.CALL) if _legal(state, ActionType.CALL) else Action(ActionType.ALL_IN)


def _street_chips(state: SpotState, view: TableView, seat: int) -> int | None:
    """What `seat` has put in on the engine's street beyond what the engine has, from
    its stack on screen; None when that cannot be told (stack not read, or grown --
    a pot won, a misread -- or the board too far ahead to split the drop)."""
    shown = view.stacks.get(seat)
    if shown is None:
        return None
    drop = state.stacks.get(seat, 0) - shown
    if abs(drop) <= view.tolerance:
        drop = 0
    if drop < 0:
        return None
    gap = 0
    if view.board_cards in STREET_BY_BOARD:
        gap = _street_index(STREET_BY_BOARD[view.board_cards]) - _street_index(state.street)
    if gap <= 0:
        return drop
    if drop == 0:
        return 0  # nothing went in on any street since
    if gap == 1 and view.bets.get(seat) is not None:
        # One street ahead: what is in front now belongs to the new street.
        rest = drop - view.bets[seat]
        if rest >= -view.tolerance:
            return max(0, rest)
    return None


def _stack_dropped(state: SpotState, view: TableView, seat: int) -> bool:
    shown = view.stacks.get(seat)
    return shown is not None and state.stacks.get(seat, 0) - shown > view.tolerance


def next_action(state: SpotState, view: TableView, my_seat: int | None = None) -> tuple[Action | None, str]:
    """The action the picture proves the seat to act has taken, or None and why not."""
    seat = state.to_act
    current = state.bets.get(seat, 0)
    facing = max(state.bets.values(), default=0)
    shown = view.bets.get(seat)
    ahead = (
        view.board_cards in STREET_BY_BOARD
        and _street_index(STREET_BY_BOARD[view.board_cards]) > _street_index(state.street)
    )
    if view.my_turn and seat == my_seat and not ahead:
        # Your bar is up on the engine's street: you are deciding. With the board further
        # on, the bar is your turn on the *next* street, and this decision is long made:
        # heads-up, a call that closes the street brings the card and your turn back at
        # once, and the bar never went away for the call to be seen.
        return None, "tocca a te"
    if (
        view.board_cards in STREET_BY_BOARD
        and _street_index(STREET_BY_BOARD[view.board_cards]) < _street_index(state.street)
    ):
        # The engine has closed the street and moved on, the screen has not dealt the next
        # card yet: the chips in front still belong to the street just closed. Read as this
        # street's, a call that closed the preflop came back as a flop bet, over and over.
        return None, "attendo la carta"
    if shown is not None and shown > current and not ahead:
        # Chips in front beyond the engine's: a call, bet or raise to that amount.
        action = _to_amount(state, seat, shown)
        return (action, "") if action else (None, f"importo {shown} non coerente per il posto {seat}")
    put_in = _street_chips(state, view, seat)
    if put_in and (ahead or shown is None):
        # The table cannot show these chips (swept into the pot, or the zone not read),
        # but the stack went down by them: a call, bet or raise to that.
        action = _to_amount(state, seat, current + put_in, view.tolerance)
        if action is None and ahead and facing > current and seat in view.in_hand:
            # A drop that makes no bet (a stack caught mid-update, a misread), but the card
            # came and the player still holds cards: called.
            return _call(state), ""
        return (action, "") if action else (None, f"stack del posto {seat} non coerente ({put_in})")
    if (
        seat == my_seat and view.my_turn is False and view.my_turn_seen
        and seat not in view.out and facing == current and _legal(state, ActionType.CHECK)
    ):
        # Your bar came and went, nothing went in front of you, nothing to call: a check.
        return Action(ActionType.CHECK), ""
    if seat in view.out:
        if facing == current and _legal(state, ActionType.CHECK):
            # Out with nothing to call: it checked, and folded later on.
            return (Action(ActionType.CHECK), "") if ahead else (Action(ActionType.FOLD), "")
        return Action(ActionType.FOLD), ""
    if ahead:
        if facing > current:
            if seat == my_seat or seat in view.in_hand:
                # The card came and the player still has cards in front (you: you are not
                # out): called. The cards say it better than the stack, which the client
                # may update a moment after the card -- read unchanged, it made a fold of
                # a call, and ended the hand there.
                return _call(state), ""
            if put_in == 0:
                # The next card came and its stack never moved: facing a bet, it did not
                # stay in by calling, so it folded.
                return Action(ActionType.FOLD), ""
            return Action(ActionType.CALL) if _legal(state, ActionType.CALL) else Action(ActionType.ALL_IN), ""
        return Action(ActionType.CHECK), ""
    later_acted = any(
        s in view.out
        or (view.bets.get(s) is not None and view.bets[s] > state.bets.get(s, 0))
        or _stack_dropped(state, view, s)
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
