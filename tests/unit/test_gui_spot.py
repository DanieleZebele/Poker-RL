import gc

import pytest

from pokerlab.engine.actions import Action, ActionType
from pokerlab.gui.spot import (
    ArrangingRandom,
    Spot,
    deal_positions,
    parse_card,
    position_names,
    replay,
)

tk = pytest.importorskip("tkinter")


def test_cards_round_trip():
    assert str(parse_card("Ah")) and parse_card("tc").rank == parse_card("Tc").rank
    with pytest.raises(ValueError):
        parse_card("1x")


def test_position_names():
    assert position_names(2) == ["BTN/SB", "BB"]
    assert position_names(6) == ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
    assert position_names(9)[3:] == ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"]


def test_the_cards_land_where_deal_positions_says():
    spot = Spot(num_players=4, my_seat=2, hole_cards=(parse_card("Ah"), parse_card("Kh")),
                board=(parse_card("2c"), parse_card("7d"), parse_card("Js")))
    placements = spot.placements()
    slots = deal_positions(4)
    assert placements[slots["seat2"][0]] == parse_card("Ah")
    assert placements[slots["flop"][2]] == parse_card("Js")
    deck = list(range(52))
    ArrangingRandom({0: 5}, seed=1).shuffle(deck)
    assert deck[0] == 5 and sorted(deck) == list(range(52))


def test_replay_walks_the_engine_turn_order():
    spot = Spot(num_players=3, my_seat=0, hole_cards=(parse_card("Ah"), parse_card("Kh")))
    state = replay(spot)
    assert state.to_act == 0 and not state.finished  # 3-max: BTN acts first preflop
    spot.script = [Action(ActionType.FOLD)]
    assert replay(spot).to_act == 1
    spot.script = [Action(ActionType.FOLD), Action(ActionType.FOLD)]
    assert replay(spot).finished


def test_a_stale_script_is_cut_where_it_stops_being_legal():
    spot = Spot(num_players=3, script=[Action(ActionType.FOLD), Action(ActionType.FOLD),
                                       Action(ActionType.FOLD)])
    state = replay(spot)
    assert state.finished and state.invalid_from == 2


def test_an_illegal_amount_is_dropped_not_raised():
    spot = Spot(num_players=3, script=[Action(ActionType.RAISE, 1)])
    state = replay(spot)
    assert state.invalid_from == 0 and state.to_act == 0


def test_validation_errors_are_clear():
    with pytest.raises(ValueError):
        Spot(num_players=1).validate()
    with pytest.raises(ValueError):
        Spot(num_players=3, my_seat=3).validate()
    with pytest.raises(ValueError):
        Spot(hole_cards=(parse_card("Ah"), parse_card("Ah"))).validate()


def test_advice_from_an_untrained_model():
    pytest.importorskip("torch")
    from pokerlab.gui.spot import advise
    from pokerlab.rl.policy import PokerActorCritic

    spot = Spot(num_players=3, my_seat=0, hole_cards=(parse_card("Ah"), parse_card("Kh")))
    state = replay(spot)
    out = advise(state.observation, state.legal_actions,
                 [("m", 1500.0, PokerActorCritic(hidden=16, num_layers=1))],
                 big_blind=2, starting_stack=200)
    assert len(out) == 1
    assert abs(sum(b.probability for b in out[0].bins) - 1.0) < 1e-4
    assert all(b.legal for b in out[0].bins)


@pytest.fixture(scope="module")
def app():
    from pokerlab.gui.app import PokerGuiApp

    application = PokerGuiApp()
    application.withdraw()
    yield application
    gc.collect()
    application.destroy()
    gc.collect()


def seat(frame, *chairs):
    for chair in chairs:
        frame.add_player(chair)


def test_the_button_decides_who_acts_first(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        seat(frame, 3, 6)  # 3 players: chairs 0 (you), 3, 6; the button starts on you
        assert frame.layout.order() == [0, 3, 6]
        assert frame.state.to_act == 0  # 3 players: the button acts first preflop
        frame.set_dealer(3)
        assert frame.layout.order() == [3, 6, 0]
        # You are now the 3rd seat from the button: the big blind, who acts last.
        assert frame.build_spot().my_seat == 2
    finally:
        frame.destroy()
        gc.collect()


def test_a_player_can_be_added_between_two_already_seated(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        seat(frame, 2, 6)
        frame.add_player(4)  # between chairs 2 and 6, which are already there
        assert frame.layout.order() == [0, 2, 4, 6]
        assert frame.state is not None and frame.state.to_act is not None
    finally:
        frame.destroy()
        gc.collect()


def test_fewer_than_two_players_asks_for_an_opponent(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        assert frame.state is None
        assert "avversario" in frame.warning.cget("text")
    finally:
        frame.destroy()
        gc.collect()


def test_actions_appear_on_the_chair_of_whoever_is_to_act(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        seat(frame, 3, 6)
        turn_chair = frame.layout.chair_of(frame.state.to_act)
        for chair, ui in frame.chair_ui.items():
            has_buttons = bool(ui.actions.winfo_children())
            assert has_buttons == (chair == turn_chair)
        frame._append(Action(ActionType.FOLD))
        assert frame.layout.chair_of(frame.state.to_act) != turn_chair
        frame._undo()
        assert frame.layout.chair_of(frame.state.to_act) == turn_chair
    finally:
        frame.destroy()
        gc.collect()


def test_changing_the_seating_clears_the_actions(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        seat(frame, 3, 6)
        frame._append(Action(ActionType.FOLD))
        assert frame.script
        frame.set_dealer(6)
        assert frame.script == []
    finally:
        frame.destroy()
        gc.collect()


def test_cards_are_set_with_buttons_and_never_repeat(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        seat(frame, 3)
        picker = frame._pick_hole(0)
        picker.choose_rank("A")
        picker.choose_suit("h")
        assert frame.hole[0] == parse_card("Ah") and frame.hole_buttons[0].cget("text") == "A♥"

        # The ace of hearts is in use: it cannot be chosen for the board.
        board = frame._pick_board(0)
        board.choose_rank("A")
        board.choose_suit("h")
        assert frame.board == []
        board.choose_suit("s")
        assert frame.board == [parse_card("As")]
        # The board fills in order: slot 3 is not available before slot 1.
        assert frame._pick_board(3) is None
        assert frame.board_buttons[3].cget("state") == "disabled"
        frame._pick_board(0).clear()
        assert frame.board == []
    finally:
        frame.destroy()
        gc.collect()


def test_both_cards_in_and_my_turn_consults_the_models(app, monkeypatch):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    calls = []
    monkeypatch.setattr(frame, "_consult", lambda state, spot: calls.append(state.to_act))
    try:
        seat(frame, 3)
        frame.refresh()
        assert calls == []  # no cards yet
        for slot, text in enumerate(("Ah", "Kh")):
            picker = frame._pick_hole(slot)
            picker.choose_rank(text[0])
            picker.choose_suit(text[1])
        assert calls and calls[-1] == frame.build_spot().my_seat
        count = len(calls)
        frame._append(Action(ActionType.FOLD))
        assert len(calls) == count  # someone else's turn: nothing asked
    finally:
        frame.destroy()
        gc.collect()


def test_the_default_stack_follows_unedited_seats(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        seat(frame, 3)
        frame.stack_vars[3].set("500")
        frame.stack_var.set("300")
        frame._default_stack_changed()
        assert frame.stack_vars[0].get() == "300" and frame.stack_vars[3].get() == "500"
    finally:
        frame.destroy()
        gc.collect()


def test_the_wheel_over_a_raise_amount_moves_it_by_a_big_blind_within_the_limits(app):
    import tkinter as tk

    from pokerlab.engine.actions import LegalAction
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        legal = LegalAction(ActionType.RAISE, 4, 10)
        var = tk.StringVar(value="4")
        frame._wheel_step(var, legal, 1)
        assert var.get() == "6"  # one big blind (2) up
        for _ in range(5):
            frame._wheel_step(var, legal, 1)
        assert var.get() == "10"  # never above the engine's maximum
        for _ in range(9):
            frame._wheel_step(var, legal, -1)
        assert var.get() == "4"  # nor below the minimum
        var.set("abc")
        frame._wheel_step(var, legal, 1)
        assert var.get() == "6"  # a half-typed amount starts from the minimum
        frame.bb_var.set("x")
        frame._wheel_step(var, legal, 1)
        assert var.get() == "7"  # an unreadable big blind falls back to a step of 1
    finally:
        frame.destroy()
        gc.collect()


def test_the_wheel_is_bound_on_the_raise_entry_in_both_directions(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        seat(frame, 3)
        ui = frame.chair_ui[frame.layout.chair_of(frame.state.to_act)]
        entries = [
            child
            for line in ui.actions.winfo_children()
            for child in line.winfo_children()
            if child.winfo_class() == "Entry"
        ]
        assert entries, "a raise amount is on offer in this spot"
        for sequence in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            assert entries[0].bind(sequence)
    finally:
        frame.destroy()
        gc.collect()
