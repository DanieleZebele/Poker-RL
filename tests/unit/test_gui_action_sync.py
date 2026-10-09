"""Rebuilding the actions of a hand from pictures of the table.

The engine seats a 6-max hand as: 0 button, 1 small blind (1 chip), 2 big blind
(2 chips), 3 UTG first to act preflop, then 4, 5.
"""

from pokerlab.engine.actions import Action, ActionType
from pokerlab.gui.action_sync import TableView, sync_actions
from pokerlab.gui.spot import Spot, parse_card, replay

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


def test_your_bar_gone_with_nothing_in_front_is_your_check():
    """Checked to you on the flop: your bar shows, then goes, and nothing is in front
    of you -- you checked. Without having seen the bar, its absence proves nothing."""
    call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
    board = (parse_card("Ks"), parse_card("8d"), parse_card("3h"))
    spot = Spot(num_players=3, my_seat=0, script=[call, call, check, check, check], board=board)
    assert replay(spot).to_act == 0  # the button, last on the flop: you

    waiting = TableView(bets={0: 0, 1: 0, 2: 0}, board_cards=3, my_turn=True)
    _, result = sync_actions(spot, waiting)
    assert result.actions == [] and result.waiting_for == 0

    gone_unseen = TableView(bets={0: 0, 1: 0, 2: 0}, board_cards=3, my_turn=False)
    assert sync_actions(spot, gone_unseen)[1].actions == []  # bar never seen: wait

    gone = TableView(bets={0: 0, 1: 0, 2: 0}, board_cards=3, my_turn=False, my_turn_seen=True)
    _, result = sync_actions(spot, gone)
    assert result.actions[:1] == [(0, Action(ActionType.CHECK))]

    bet = TableView(bets={0: 4, 1: 0, 2: 0}, board_cards=3, my_turn=False, my_turn_seen=True)
    assert sync_actions(spot, bet)[1].actions[:1] == [(0, Action(ActionType.BET, 4))]  # chips: a bet


def test_heads_up_your_call_is_seen_when_the_card_comes_with_your_bar_still_up():
    """Heads-up, you are the big blind (seat 1): the button raises, you call, the flop
    comes and you are first to act on it -- the bar never went away, so it was your call
    that the new card proved, not a decision still to make."""
    raised = Spot(num_players=2, my_seat=1, script=[Action(ActionType.RAISE, 6)])
    assert replay(raised).to_act == 1

    still = TableView(bets={0: 6, 1: 2}, board_cards=0, my_turn=True)
    assert sync_actions(raised, still)[1].actions == []  # deciding: nothing yet

    flop = TableView(bets={0: 0, 1: 0}, board_cards=3, my_turn=True)
    _, result = sync_actions(raised, flop)
    assert result.actions == [(1, Action(ActionType.CALL))]
    assert result.waiting_for == 1 and result.note == "tocca a te"  # your turn on the flop

    unmoved = TableView(bets={0: 0, 1: 0}, board_cards=3, my_turn=True, stacks={0: 194, 1: 198})
    assert sync_actions(raised, unmoved)[1].actions == [(1, Action(ActionType.CALL))]  # your cards win

    folded = TableView(bets={0: 0, 1: 0}, out={1}, board_cards=0, my_turn=False)
    assert sync_actions(raised, folded)[1].actions == [(1, Action(ActionType.FOLD))]


# --- stacks (`TableView.stacks`): chips behind, the default spot at blinds 1/2, 200 each.
# Three-handed and limped: everyone has 198 behind on the flop; seat 1 (SB) acts first.

def _limped_flop(*flop_actions):
    call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
    board = (parse_card("Ks"), parse_card("8d"), parse_card("3h"), parse_card("2c"))
    return Spot(num_players=3, my_seat=0, script=[call, call, check, *flop_actions], board=board)


def test_a_bet_swept_into_the_pot_is_read_off_the_stacks():
    """The SB bet 3 BB on the flop and both called; the turn came before any reading saw
    the chips. The stacks went down by 6 each: a bet and two calls."""
    view = TableView(bets={0: 0, 1: 0, 2: 0}, board_cards=4, stacks={0: 192, 1: 192, 2: 192})
    _, result = sync_actions(_limped_flop(), view)
    assert result.actions == [
        (1, Action(ActionType.BET, 6)), (2, Action(ActionType.CALL)), (0, Action(ActionType.CALL)),
    ]


def test_a_new_card_with_the_stacks_unchanged_is_a_round_of_checks():
    view = TableView(bets={0: 0, 1: 0, 2: 0}, board_cards=4, stacks={0: 198, 1: 198, 2: 198})
    _, result = sync_actions(_limped_flop(), view)
    assert [a.action_type for _, a in result.actions] == [ActionType.CHECK] * 3


def test_facing_a_bet_a_stack_that_never_moved_has_folded():
    """The SB's flop bet is known; the turn came, the BB's stack is where it was (it did
    not call) and the button's went down by the bet (it did)."""
    view = TableView(bets={0: 0, 1: 0, 2: 0}, board_cards=4, stacks={0: 192, 1: 192, 2: 198})
    _, result = sync_actions(_limped_flop(Action(ActionType.BET, 6)), view)
    assert result.actions == [(2, Action(ActionType.FOLD)), (0, Action(ActionType.CALL))]


def test_a_player_still_holding_cards_when_the_card_comes_called_whatever_the_stack():
    """Heads-up, the button (you) bet 4,5 BB on the flop, the big blind called and the
    chips went straight into the pot. At the reading that shows the turn its stack is
    not updated yet: it still has its cards, so it called -- not folded."""
    call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
    board = (parse_card("Ks"), parse_card("8d"), parse_card("3h"))
    spot = Spot(num_players=2, my_seat=0, board=board,
                script=[Action(ActionType.RAISE, 6), call, check, Action(ActionType.BET, 9)])
    assert replay(spot).to_act == 1
    late = TableView(bets={0: 0, 1: 0}, board_cards=4, in_hand={0, 1}, stacks={0: 185, 1: 194})
    assert sync_actions(spot, late)[1].actions[:1] == [(1, Action(ActionType.CALL))]
    halfway = TableView(bets={0: 0, 1: 0}, board_cards=4, in_hand={0, 1}, stacks={0: 185, 1: 190})
    assert sync_actions(spot, halfway)[1].actions[:1] == [(1, Action(ActionType.CALL))]
    gone = TableView(bets={0: 0, 1: 0}, board_cards=4, out={1}, stacks={0: 185, 1: 194})
    assert sync_actions(spot, gone)[1].actions[:1] == [(1, Action(ActionType.FOLD))]


def test_the_bet_on_the_table_wins_over_the_stack():
    call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
    board = (parse_card("Ks"), parse_card("8d"), parse_card("3h"))
    spot = Spot(num_players=3, my_seat=0, script=[call, call, check], board=board)
    # 3 BB in front of the SB, but its stack reads 4 BB lower: the table is believed
    view = TableView(bets={0: 0, 1: 6, 2: 0}, board_cards=3, stacks={0: 198, 1: 190, 2: 198})
    _, result = sync_actions(spot, view)
    assert result.actions[:1] == [(1, Action(ActionType.BET, 6))]


def test_a_bet_zone_not_read_falls_back_on_the_stack():
    call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
    board = (parse_card("Ks"), parse_card("8d"), parse_card("3h"))
    spot = Spot(num_players=3, my_seat=0, script=[call, call, check], board=board)
    view = TableView(bets={0: 0, 2: 0}, board_cards=3, stacks={0: 198, 1: 190, 2: 198})  # seat 1 unread
    _, result = sync_actions(spot, view)
    assert result.actions[:1] == [(1, Action(ActionType.BET, 8))]


def test_a_drop_over_two_streets_cannot_be_split_and_is_not_used():
    from pokerlab.gui.action_sync import _street_chips

    state = replay(_limped_flop())  # the engine on the flop
    river = TableView(bets={0: 0, 1: 0, 2: 0}, board_cards=5, stacks={0: 190, 1: 198, 2: 198})
    assert _street_chips(state, river, 0) is None  # 8 chips over flop and turn: how split?
    assert _street_chips(state, river, 1) == 0  # nothing at all: that is known


def test_chips_left_in_front_of_a_closed_street_are_not_the_next_street_s():
    """The big blind's call closed the preflop: the engine is on the flop, the screen
    has not dealt it and still shows 3 BB in front of both. Those are preflop chips:
    nothing happens until the flop comes."""
    raise_to_3 = Action(ActionType.RAISE, 6)
    fold, call = Action(ActionType.FOLD), Action(ActionType.CALL)
    spot = Spot(num_players=3, my_seat=1, script=[raise_to_3, fold, call])  # BTN raises, SB folds, BB calls
    assert replay(spot).street.value == "flop"
    view = TableView(bets={0: 6, 1: 0, 2: 6}, out={1}, board_cards=0)
    _, result = sync_actions(spot, view)
    assert result.actions == [] and result.note == "attendo la carta"
