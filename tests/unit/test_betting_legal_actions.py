import pytest

from pokerlab.cards.deck import Deck
from pokerlab.engine.actions import Action, ActionType, IllegalActionError
from pokerlab.engine.betting import (
    apply_action,
    compute_legal_actions,
    post_blinds,
    start_new_street_betting,
)
from pokerlab.engine.state import HandState, PlayerState, Street


def make_hand(stacks: dict[int, int], small_blind=1, big_blind=2, button_seat=0) -> HandState:
    seats = [
        PlayerState(seat=s, player_id=f"p{s}", name=f"P{s}", stack=stack)
        for s, stack in sorted(stacks.items())
    ]
    return HandState(
        hand_id="h1",
        button_seat=button_seat,
        seats=seats,
        deck=Deck(),
        small_blind=small_blind,
        big_blind=big_blind,
    )


def action_types(legal):
    return {la.action_type for la in legal}


def test_preflop_legal_actions_after_blinds_three_handed():
    # seats 0,1,2 ; button=0, sb=1, bb=2
    hs = make_hand({0: 200, 1: 200, 2: 200}, button_seat=0)
    post_blinds(hs, sb_seat=1, bb_seat=2)

    # Seat 0 (UTG, first to act preflop with 3 players) faces the big blind.
    legal = compute_legal_actions(hs, 0)
    assert action_types(legal) == {ActionType.FOLD, ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN}
    call = next(la for la in legal if la.action_type == ActionType.CALL)
    assert call.min_amount == call.max_amount == 2
    raise_ = next(la for la in legal if la.action_type == ActionType.RAISE)
    assert raise_.min_amount == 4  # min raise-to = current bet (2) + min_raise increment (2)


def test_big_blind_option_can_raise_when_everyone_just_called():
    # 3-handed: button=0, sb=1, bb=2. UTG (seat 0) and SB (seat 1) both just
    # call the big blind, so the BB (seat 2) faces to_call == 0 -- but
    # current_bet_to_match is still 2 (the BB itself), not 0. The BB must
    # still get the option to raise here, not just check/fold.
    hs = make_hand({0: 200, 1: 200, 2: 200}, button_seat=0)
    post_blinds(hs, sb_seat=1, bb_seat=2)
    apply_action(hs, 0, Action(ActionType.CALL))
    apply_action(hs, 1, Action(ActionType.CALL))

    legal = compute_legal_actions(hs, 2)
    types = action_types(legal)
    assert ActionType.CHECK in types
    assert ActionType.RAISE in types
    assert ActionType.BET not in types  # there's already a live bet (the BB), this is a raise not an open
    raise_ = next(la for la in legal if la.action_type == ActionType.RAISE)
    assert raise_.min_amount == 4  # min raise-to = current bet (2) + min_raise increment (2)

    # And actually taking that option should work like any other raise.
    apply_action(hs, 2, Action(ActionType.RAISE, amount=6))
    assert hs.current_bet_to_match == 6
    assert 0 in hs.to_act and 1 in hs.to_act  # both must respond to the raise


def test_check_available_when_nothing_to_call():
    hs = make_hand({0: 200, 1: 200}, button_seat=0)
    start_new_street_betting(hs, Street.FLOP)
    legal = compute_legal_actions(hs, 0)
    assert ActionType.CHECK in action_types(legal)
    assert ActionType.CALL not in action_types(legal)
    assert ActionType.FOLD in action_types(legal)


def test_opening_bet_min_is_big_blind():
    hs = make_hand({0: 200, 1: 200}, big_blind=4, button_seat=0)
    start_new_street_betting(hs, Street.FLOP)
    legal = compute_legal_actions(hs, 0)
    bet = next(la for la in legal if la.action_type == ActionType.BET)
    assert bet.min_amount == 4
    assert bet.max_amount == 200


def test_short_stack_cannot_afford_call_only_fold_or_all_in():
    hs = make_hand({0: 200, 1: 5}, button_seat=0)
    post_blinds(hs, sb_seat=0, bb_seat=1)  # heads-up: button posts SB
    apply_action(hs, 0, Action(ActionType.RAISE, amount=20))  # button raises to 20
    legal = compute_legal_actions(hs, 1)
    types = action_types(legal)
    assert ActionType.CALL not in types  # stack (5-1 already posted=4 left) can't cover 20
    assert ActionType.FOLD in types
    assert ActionType.ALL_IN in types
    assert ActionType.RAISE not in types


def test_illegal_action_raises():
    hs = make_hand({0: 200, 1: 200}, button_seat=0)
    start_new_street_betting(hs, Street.FLOP)
    with pytest.raises(IllegalActionError):
        apply_action(hs, 0, Action(ActionType.CALL))  # nothing to call, CALL isn't legal


def test_full_raise_reopens_action_for_everyone_including_earlier_actor():
    hs = make_hand({0: 500, 1: 500, 2: 500}, button_seat=0)
    start_new_street_betting(hs, Street.FLOP)
    # order doesn't matter for this state-only test; drive it manually.
    apply_action(hs, 0, Action(ActionType.BET, amount=20))  # seat0 bets 20 (min_raise now 20)
    apply_action(hs, 1, Action(ActionType.CALL))  # seat1 calls, now done unless reopened
    assert 1 not in hs.to_act
    apply_action(hs, 2, Action(ActionType.RAISE, amount=60))  # full raise: +40 >= min_raise(20)
    # seat1 already called at the old level; a FULL raise must reopen for them.
    assert 1 in hs.to_act
    legal1 = compute_legal_actions(hs, 1)
    assert ActionType.RAISE in action_types(legal1)


def test_short_all_in_raise_does_not_reopen_for_players_who_already_acted():
    hs = make_hand({0: 500, 1: 500, 2: 12}, button_seat=0)
    start_new_street_betting(hs, Street.FLOP)
    apply_action(hs, 0, Action(ActionType.BET, amount=20))  # min_raise = 20
    apply_action(hs, 1, Action(ActionType.CALL))  # seat1 already responded to the 20-level
    assert 1 not in hs.to_act
    # seat2 shoves all-in for 12, which is LESS than current_bet_to_match (20) -> not even a raise.
    apply_action(hs, 2, Action(ActionType.ALL_IN))
    assert hs.current_bet_to_match == 20  # unchanged: seat2's all-in didn't cover the call
    assert 1 not in hs.to_act  # seat1 does not need to act again


def test_short_all_in_raise_above_current_bet_bars_reraise_for_already_acted_player():
    hs = make_hand({0: 500, 1: 500, 2: 27}, button_seat=0)
    start_new_street_betting(hs, Street.FLOP)
    apply_action(hs, 0, Action(ActionType.BET, amount=20))  # min_raise = 20, level=20
    apply_action(hs, 1, Action(ActionType.CALL))  # seat1 already acted at level 20
    assert 1 not in hs.to_act
    # seat2 shoves all-in for 27 total (> 20 call, but the +7 raise is below the 20 min-raise).
    apply_action(hs, 2, Action(ActionType.ALL_IN))
    assert hs.current_bet_to_match == 27
    # seat1 must respond to the new amount (owes 7 more) ...
    assert 1 in hs.to_act
    # ... but is barred from raising again, since this wasn't a full raise.
    legal1 = compute_legal_actions(hs, 1)
    assert ActionType.RAISE not in action_types(legal1)
    assert ActionType.CALL in action_types(legal1)

    # seat0, who had NOT yet acted again since their own opening bet... wait seat0 opened,
    # so they already "acted" too and should also be barred.
    legal0 = compute_legal_actions(hs, 0)
    assert ActionType.RAISE not in action_types(legal0)


def test_short_all_in_raise_does_not_bar_a_player_who_had_not_acted_yet():
    hs = make_hand({0: 500, 1: 500, 2: 500, 3: 27}, button_seat=0)
    start_new_street_betting(hs, Street.FLOP)
    apply_action(hs, 0, Action(ActionType.BET, amount=20))  # min_raise = 20
    # seat1 has NOT acted yet when seat3's short all-in comes in (out of a simplified manual order,
    # simulating seat3 acting before seat1 gets their turn).
    apply_action(hs, 3, Action(ActionType.ALL_IN))  # 27 total, +7 raise, short of the 20 min-raise
    assert hs.current_bet_to_match == 27
    legal1 = compute_legal_actions(hs, 1)
    assert ActionType.RAISE in action_types(legal1)  # seat1 never acted yet, keeps full rights
