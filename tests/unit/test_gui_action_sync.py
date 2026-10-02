"""Rebuilding the actions of a hand from pictures of the table.

The engine seats a 6-max hand as: 0 button, 1 small blind (1 chip), 2 big blind
(2 chips), 3 UTG first to act preflop, then 4, 5.
"""

from pokerlab.engine.actions import Action, ActionType
from pokerlab.gui.action_sync import TableView, sync_actions
from pokerlab.gui.spot import Spot, replay

BLINDS = {1: 1, 2: 2}


def spot(script=(), my_seat=0):
    return Spot(num_players=6, my_seat=my_seat, script=list(script))


def kinds(result):
    return [(seat, action.action_type, action.amount) for seat, action in result.actions]


def test_only_the_blinds_shown_adds_nothing_and_waits_for_utg():
    new, result = sync_actions(spot(), TableView(bets=dict(BLINDS), board_cards=0))
    assert result.actions == [] and result.waiting_for == 3 and new.script == []


def test_several_players_acting_between_two_readings_are_all_recovered():
    view = TableView(bets={**BLINDS, 3: 6, 5: 6}, out={4}, board_cards=0)
    new, result = sync_actions(spot(), view)
    assert kinds(result) == [
        (3, ActionType.RAISE, 6), (4, ActionType.FOLD, 0), (5, ActionType.CALL, 0),
    ]
    assert result.waiting_for == 0  # the button has not acted yet
    assert sync_actions(new, view)[1].actions == []  # the same picture again adds nothing


def test_a_check_is_seen_when_a_later_player_acted():
    # Preflop: everyone folds to the blinds, small blind calls, big blind checks.
    script = [Action(ActionType.FOLD)] * 4 + [Action(ActionType.CALL), Action(ActionType.CHECK)]
    on_flop = spot(script)
    assert replay(on_flop).to_act == 1
    view = TableView(bets={1: 0, 2: 4}, out={0, 3, 4, 5}, board_cards=3)
    _new, result = sync_actions(on_flop, view)
    assert kinds(result) == [(1, ActionType.CHECK, 0), (2, ActionType.BET, 4)]
    assert result.waiting_for == 1  # back to the small blind, facing the bet


def test_a_street_dealt_closes_the_one_before():
    # UTG and the button call, everyone else in place; then the flop shows.
    view = TableView(bets={}, out={4, 5}, board_cards=3)
    preflop = spot([Action(ActionType.CALL)])  # UTG called already
    _new, result = sync_actions(preflop, view)
    assert kinds(result)[:4] == [
        (4, ActionType.FOLD, 0), (5, ActionType.FOLD, 0), (0, ActionType.CALL, 0), (1, ActionType.CALL, 0),
    ]
    assert (2, ActionType.CHECK, 0) in kinds(result)


def test_your_turn_with_nothing_new_waits_for_you():
    _new, result = sync_actions(spot(my_seat=3), TableView(bets=dict(BLINDS), board_cards=0))
    assert result.actions == [] and result.waiting_for == 3


def test_an_amount_the_engine_cannot_make_stops_without_inventing():
    view = TableView(bets={**BLINDS, 3: 3}, board_cards=0)  # a raise to 3 is below the minimum
    _new, result = sync_actions(spot(), view)
    assert result.actions == [] and result.waiting_for == 3 and "non coerente" in result.note


def test_a_shove_is_an_all_in():
    view = TableView(bets={**BLINDS, 3: 200}, board_cards=0)
    _new, result = sync_actions(spot(), view)
    assert kinds(result)[0] == (3, ActionType.ALL_IN, 0)


def test_your_timer_turns_the_unseen_checks_into_checks():
    # Flop, blinds checked preflop with everyone else folded; you are the big blind (2).
    script = [Action(ActionType.FOLD)] * 4 + [Action(ActionType.CALL), Action(ActionType.CHECK)]
    on_flop = Spot(num_players=6, my_seat=2, script=script)
    no_chips = TableView(bets={1: 0, 2: 0}, out={0, 3, 4, 5}, board_cards=3)
    _new, result = sync_actions(on_flop, no_chips)
    assert result.actions == [] and result.waiting_for == 1  # a check leaves no trace
    with_timer = TableView(bets={1: 0, 2: 0}, out={0, 3, 4, 5}, board_cards=3, my_turn=True)
    _new, result = sync_actions(on_flop, with_timer)
    assert kinds(result) == [(1, ActionType.CHECK, 0)]
    assert result.waiting_for == 2 and result.note == "tocca a te"


def test_your_timer_does_not_invent_a_call():
    # UTG faces the big blind: with your timer up it must have called or folded,
    # and neither shows -- so nothing is invented.
    view = TableView(bets=dict(BLINDS), board_cards=0, my_turn=True)
    _new, result = sync_actions(spot(my_seat=5), view)
    assert result.actions == [] and result.waiting_for == 3
