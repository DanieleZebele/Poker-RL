from __future__ import annotations

import time

from pokerlab.engine.actions import Action, ActionType, IllegalActionError, LegalAction
from pokerlab.engine.state import ActionRecord, HandState, PlayerStatus, Street


def seats_clockwise_from(hand_state: HandState, start_seat: int) -> list[int]:
    """Cyclic seat order starting at `start_seat` (or the next occupied seat
    after it, if `start_seat` itself isn't occupied this hand)."""
    seats_sorted = sorted(ps.seat for ps in hand_state.seats)
    if not seats_sorted:
        return []
    idx = 0
    for i, s in enumerate(seats_sorted):
        if s >= start_seat:
            idx = i
            break
    else:
        idx = 0
    return seats_sorted[idx:] + seats_sorted[:idx]


def compute_legal_actions(hand_state: HandState, seat: int) -> list[LegalAction]:
    ps = hand_state.seat_state(seat)
    if ps.status != PlayerStatus.ACTIVE or ps.stack <= 0:
        return []

    to_call = hand_state.current_bet_to_match - ps.current_bet
    shove_to = ps.current_bet + ps.stack
    actions: list[LegalAction] = [LegalAction(ActionType.FOLD)]

    if to_call <= 0:
        actions.append(LegalAction(ActionType.CHECK))
        if hand_state.current_bet_to_match == 0:
            # No bet yet this street: the option is a fresh opening bet.
            if shove_to >= hand_state.big_blind:
                actions.append(LegalAction(ActionType.BET, min_amount=hand_state.big_blind, max_amount=shove_to))
        elif seat not in hand_state.raise_barred:
            # to_call == 0 but current_bet_to_match > 0: this is the big
            # blind option preflop (everyone just called the BB, nobody
            # raised) -- the player already has chips in as the current
            # bet and can raise it further, not "open" a new one.
            min_raise_to = hand_state.current_bet_to_match + hand_state.min_raise
            if min_raise_to <= shove_to:
                actions.append(LegalAction(ActionType.RAISE, min_amount=min_raise_to, max_amount=shove_to))
    else:
        if ps.stack >= to_call:
            actions.append(
                LegalAction(
                    ActionType.CALL,
                    min_amount=hand_state.current_bet_to_match,
                    max_amount=hand_state.current_bet_to_match,
                )
            )
        if ps.stack > to_call and seat not in hand_state.raise_barred:
            min_raise_to = hand_state.current_bet_to_match + hand_state.min_raise
            if min_raise_to <= shove_to:
                actions.append(LegalAction(ActionType.RAISE, min_amount=min_raise_to, max_amount=shove_to))

    actions.append(LegalAction(ActionType.ALL_IN, min_amount=shove_to, max_amount=shove_to))
    return actions


def _is_legal(action: Action, legal_actions: list[LegalAction]) -> bool:
    for legal in legal_actions:
        if legal.action_type != action.action_type:
            continue
        # Only BET/RAISE carry a caller-chosen amount; CALL/ALL_IN amounts
        # are derived from state (see Action's docstring), so their
        # LegalAction min/max (though populated, for display purposes) is
        # not something the caller needs to match.
        if action.action_type in (ActionType.BET, ActionType.RAISE):
            return legal.min_amount <= action.amount <= legal.max_amount
        return True
    return False


def _handle_new_bet_level(hand_state: HandState, actor_seat: int, old_level: int, new_level: int) -> None:
    increase = new_level - old_level
    full_raise = increase >= hand_state.min_raise
    hand_state.current_bet_to_match = new_level
    others = {ps.seat for ps in hand_state.actionable_seats() if ps.seat != actor_seat}

    if full_raise:
        hand_state.min_raise = increase
        hand_state.raise_barred = set()
        hand_state.to_act = set(others)
    else:
        already_acted = {ps.seat for ps in hand_state.actionable_seats()} - hand_state.to_act
        hand_state.raise_barred |= already_acted - {actor_seat}
        hand_state.to_act |= others


def apply_action(hand_state: HandState, seat: int, action: Action) -> None:
    legal_actions = compute_legal_actions(hand_state, seat)
    if not _is_legal(action, legal_actions):
        raise IllegalActionError(
            f"seat {seat} attempted illegal action {action}; legal actions were {legal_actions}"
        )

    ps = hand_state.seat_state(seat)
    pot_before = hand_state.pot_total()
    stack_before = ps.stack
    hand_state.to_act.discard(seat)

    if action.action_type == ActionType.FOLD:
        ps.status = PlayerStatus.FOLDED
    elif action.action_type == ActionType.CHECK:
        pass
    elif action.action_type == ActionType.CALL:
        to_call = hand_state.current_bet_to_match - ps.current_bet
        ps.commit(to_call)
    elif action.action_type in (ActionType.BET, ActionType.RAISE):
        old_level = hand_state.current_bet_to_match
        increment = action.amount - ps.current_bet
        ps.commit(increment)
        _handle_new_bet_level(hand_state, seat, old_level, action.amount)
    elif action.action_type == ActionType.ALL_IN:
        old_level = hand_state.current_bet_to_match
        shove_to = ps.current_bet + ps.stack
        ps.commit(ps.stack)
        if shove_to > old_level:
            _handle_new_bet_level(hand_state, seat, old_level, shove_to)
        # else: short all-in call for less than the current bet; the bet
        # level doesn't move and nobody else needs to respond further.

    hand_state.action_log.append(
        ActionRecord(
            street=hand_state.street,
            seat=seat,
            player_id=ps.player_id,
            action_type=action.action_type,
            amount=ps.current_bet,
            stack_before=stack_before,
            stack_after=ps.stack,
            pot_before=pot_before,
            timestamp=time.time(),
        )
    )


def post_blinds(hand_state: HandState, sb_seat: int, bb_seat: int) -> None:
    for ps, blind in ((hand_state.seat_state(sb_seat), hand_state.small_blind), (hand_state.seat_state(bb_seat), hand_state.big_blind)):
        pot_before = hand_state.pot_total()
        stack_before = ps.stack
        ps.commit(min(blind, ps.stack))
        hand_state.action_log.append(
            ActionRecord(
                street=hand_state.street,
                seat=ps.seat,
                player_id=ps.player_id,
                action_type=ActionType.POST_BLIND,
                amount=ps.current_bet,
                stack_before=stack_before,
                stack_after=ps.stack,
                pot_before=pot_before,
                timestamp=time.time(),
            )
        )
    hand_state.current_bet_to_match = hand_state.big_blind
    hand_state.min_raise = hand_state.big_blind
    hand_state.raise_barred = set()
    hand_state.to_act = {ps.seat for ps in hand_state.actionable_seats()}


def start_new_street_betting(hand_state: HandState, street: Street) -> None:
    hand_state.street = street
    for ps in hand_state.seats:
        ps.current_bet = 0
    hand_state.current_bet_to_match = 0
    hand_state.min_raise = hand_state.big_blind
    hand_state.raise_barred = set()
    hand_state.to_act = {ps.seat for ps in hand_state.actionable_seats()}
