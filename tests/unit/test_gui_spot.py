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
    from support import tiny_model

    from pokerlab.gui.spot import advise

    spot = Spot(num_players=3, my_seat=0, hole_cards=(parse_card("Ah"), parse_card("Kh")))
    state = replay(spot)
    out = advise(state.observation, state.legal_actions,
                 [("m", 1500.0, tiny_model(hidden=16, num_layers=1))],
                 big_blind=2)
    assert len(out) == 1
    assert abs(sum(b.probability for b in out[0].bins) - 1.0) < 1e-4
    assert all(b.legal for b in out[0].bins)


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


def test_the_wheel_over_a_raise_amount_moves_it_by_a_big_blind_within_the_limits(app):
    import tkinter as tk

    from pokerlab.engine.actions import LegalAction
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        legal = LegalAction(ActionType.RAISE, 200, 500)  # chips: 2 to 5 big blinds
        var = tk.StringVar(value="2")  # the field is in big blinds
        frame._wheel_step(var, legal, 1)
        assert var.get() == "3"  # one big blind up
        for _ in range(5):
            frame._wheel_step(var, legal, 1)
        assert var.get() == "5"  # never above the engine's maximum
        for _ in range(9):
            frame._wheel_step(var, legal, -1)
        assert var.get() == "2"  # nor below the minimum
        var.set("abc")
        frame._wheel_step(var, legal, 1)
        assert var.get() == "3"  # a half-typed amount starts from the minimum
        var.set("2,5")  # a comma decimal, as the client writes it
        frame._wheel_step(var, legal, 1)
        assert var.get() == "3,5"
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



def test_a_popup_opens_under_the_mouse_and_stays_on_its_monitor(app, monkeypatch):
    import tkinter as tk

    from pokerlab.gui import spot_view

    monkeypatch.setattr(spot_view, "_monitor_bounds", lambda x, y, fallback: (1920, 0, 3840, 1040))
    window = tk.Toplevel(app)
    try:
        tk.Frame(window, width=300, height=200).pack()
        monkeypatch.setattr(window, "winfo_pointerxy", lambda: (2500, 400))
        assert spot_view.place_near_pointer(window) == (2500 - 150, 380)
        monkeypatch.setattr(window, "winfo_pointerxy", lambda: (3830, 1030))  # bottom-right corner
        assert spot_view.place_near_pointer(window) == (3840 - 300, 1040 - 200)
        monkeypatch.setattr(window, "winfo_pointerxy", lambda: (1925, 5))  # left edge of monitor 2
        assert spot_view.place_near_pointer(window) == (1920, 0)
    finally:
        window.destroy()
        gc.collect()


def test_your_box_and_all_its_actions_fit_on_the_table(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    try:
        frame.pack()
        seat(frame, 3, 6)  # three-handed the button acts first: your actions are showing
        assert frame.layout.chair_of(frame.state.to_act) == 0
        app.update()
        holder = frame.chair_ui[0].box.master
        assert frame.chair_ui[0].actions.winfo_children()
        assert holder.winfo_y() + holder.winfo_reqheight() <= int(frame.canvas.cget("height"))
    finally:
        frame.destroy()
        gc.collect()


def test_no_chair_is_cut_off_whoever_is_to_act(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    app.deiconify()  # geometry and pointer events are only real on a mapped window
    try:
        frame.pack()
        seat(frame, *range(1, 8))
        for _ in range(7):  # the turn goes round every chair, top ones included
            app.update()
            holder = frame.chair_ui[frame.layout.chair_of(frame.state.to_act)].holder
            assert holder.winfo_y() >= 0 and holder.winfo_x() >= 0
            assert holder.winfo_y() + holder.winfo_reqheight() <= int(frame.canvas.cget("height"))
            frame._append(Action(ActionType.CALL))
    finally:
        app.withdraw()
        frame.destroy()
        gc.collect()


def test_a_press_anywhere_on_a_chair_brings_it_in_front(app):
    from pokerlab.gui.spot_view import SpotFrame, _raise_tag

    frame = SpotFrame(app)
    app.deiconify()  # Tk delivers no pointer event to a withdrawn window
    try:
        frame.pack()
        seat(frame, 3, 4, 6)
        app.update()

        def front():  # Tk's own child list is in stacking order, topmost last
            return frame.nametowidget(frame.tk.splitlist(frame.tk.call("winfo", "children", frame.canvas))[-1])

        acting = frame.chair_ui[frame.layout.chair_of(frame.state.to_act)].holder
        assert front() == acting  # whoever acts is put in front by itself
        other = next(ui for c, ui in frame.chair_ui.items() if c in (3, 4, 6) and ui.holder != acting)
        button = next(w for w in other.box.winfo_children()[0].winfo_children() if w.winfo_class() == "Button")
        tags = button.bindtags()
        assert tags[0] == _raise_tag(other.holder) and "Button" in tags  # the click still reaches it
        button.event_generate("<ButtonPress-1>", x=2, y=2)
        app.update()
        assert front() == other.holder
        # action buttons are rebuilt at every step and must be raisable too
        frame._append(Action(ActionType.CALL))
        acting = frame.chair_ui[frame.layout.chair_of(frame.state.to_act)]
        assert all(_raise_tag(acting.holder) in w.bindtags() for w in acting.actions.winfo_children())
    finally:
        app.withdraw()
        frame.destroy()
        gc.collect()


def test_big_blind_amounts_are_shown_and_read_with_a_comma():
    from pokerlab.gui.spot import bb_number, describe_action, format_bb, parse_bb

    assert bb_number(37, 2) == "18,5" and bb_number(6, 2) == "3" and format_bb(1, 2) == "0,5 BB"
    assert parse_bb("18,5", 2) == 37 and parse_bb("18.5", 2) == 37 and parse_bb(" 100 ", 2) == 200
    with pytest.raises(ValueError):
        parse_bb("tanti", 2)
    assert describe_action(Action(ActionType.RAISE, 7), None, 2) == "raise a 3,5 BB"
    assert describe_action(Action(ActionType.RAISE, 7)) == "raise a 7"  # chips without a big blind


def test_the_models_page_shows_every_amount_in_big_blinds(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        seat(frame, 3, 6)
        frame._append(Action(ActionType.RAISE, 350))  # 3,5 BB
        assert frame.build_spot().starting_stack == 10000  # 100 BB, the default
        assert "piatto 5 BB" in frame.situation.cget("text")  # 0,5 + 1 + 3,5
        assert "raise a 3,5 BB" in frame.log.get("1.0", "end")
        texts = [w.cget("text") for line in frame.chair_ui[frame.layout.chair_of(frame.state.to_act)]
                 .actions.winfo_children() for w in ([line] + line.winfo_children())
                 if w.winfo_class() == "Button"]
        assert any(t.startswith("Raise a (") and t.endswith(" BB)") for t in texts)
        assert any(t.startswith("Call ") and t.endswith(" BB") for t in texts)
    finally:
        frame.destroy()
        gc.collect()


def test_the_models_are_shown_the_statistics_the_spot_was_given_and_the_replay_records_nothing():
    vector = (1.0,) + (0.25,) * 19
    spot = Spot(num_players=3, starting_stack=200, seat_stats={1: vector})

    state = replay(spot)

    assert dict(state.observation.seat_stats) == {1: vector}
    assert replay(Spot(num_players=3, starting_stack=200)).observation.seat_stats == {}


def test_a_replay_reports_the_blinds_and_the_scripted_actions_but_not_the_ones_that_finish_the_hand():
    raised = Action(ActionType.RAISE, 6)
    spot = Spot(num_players=3, starting_stack=200, script=[raised])

    state = replay(spot)
    assert [(r.action_type, r.amount) for r in state.records] == [
        (ActionType.POST_BLIND, 1),
        (ActionType.POST_BLIND, 2),
        (ActionType.RAISE, 6),
    ]

    folds = replay(replace_script(spot, [Action(ActionType.FOLD), Action(ActionType.FOLD)]))
    assert folds.finished
    assert [r.action_type for r in folds.records][-2:] == [ActionType.FOLD, ActionType.FOLD]


def replace_script(spot, script):
    from dataclasses import replace

    return replace(spot, script=script)


def test_the_likeliest_action_of_the_top_model_is_in_bold(app, monkeypatch):
    from pokerlab.gui import spot_view
    from pokerlab.gui.spot import BinAdvice, ModelAdvice

    def opinion(label, rating):
        best = BinAdvice(1, "check/call", Action(ActionType.CALL), 0.7, True)
        other = BinAdvice(0, "fold", Action(ActionType.FOLD), 0.3, True)
        return ModelAdvice(label, rating, best, [best, other], 0.1)

    frame = spot_view.SpotFrame(app)
    try:
        monkeypatch.setattr(spot_view, "advise", lambda *a, **k: [opinion("primo", 1900), opinion("secondo", 1800)])
        frame.models = [object()]
        # their places in the global ranking: the third did not load, the second is #3 there
        frame._model_ranks = {"primo": 1, "secondo": 3}
        spot = Spot(num_players=3, starting_stack=200)
        frame._consult(replay(spot), spot)

        text = frame.advice.get("1.0", "end")
        assert text.splitlines()[0].startswith("#1 ") and "#3 " in text
        assert "primo" not in text and "secondo" not in text  # numbers only, no labels
        ranges = [str(r) for r in frame.advice.tag_ranges("bold")]
        assert ranges == ["2.0", "2.end"] or (len(ranges) == 2 and ranges[0].startswith("2."))
        bold = frame.advice.get(*frame.advice.tag_ranges("bold"))
        # the tag's font must still exist once Python has collected its garbage: a font
        # object dropped deletes the Tk font, and the line then shows plain
        gc.collect()
        font_name = frame.advice.tag_cget("bold", "font")
        assert str(font_name) in app.tk.call("font", "names")
        assert app.tk.call("font", "actual", font_name, "-weight") == "bold"
        assert bold.strip().startswith("->") and "70%" in bold
        assert "#3" not in bold
    finally:
        frame.destroy()
        gc.collect()


def test_the_table_grows_with_the_screen_and_the_felt_stays_clear_of_the_boxes():
    from pokerlab.gui.spot_view import (
        BOX_BOTTOM,
        BOX_SIDE,
        BOX_TOP,
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        FELT_MIN_RADII,
        RESERVED_HEIGHT,
        SIDE_PANEL_WIDTH,
        table_geometry,
    )

    for screen in ((1920, 1080), (2560, 1440), (1536, 864)):
        geometry = table_geometry(*screen)
        width, height = geometry.canvas
        assert width >= CANVAS_WIDTH and height >= CANVAS_HEIGHT
        if geometry.canvas != (CANVAS_WIDTH, CANVAS_HEIGHT):
            assert width + SIDE_PANEL_WIDTH <= screen[0] + 1 and height + RESERVED_HEIGHT <= screen[1] + 1
        (cx, cy), (a, b) = geometry.center, geometry.table_radii
        assert cx - a >= BOX_SIDE and cx + a <= width - BOX_SIDE  # clear of the side boxes
        if b > FELT_MIN_RADII[1]:  # at its minimum the board's height wins over the margin
            assert cy - b >= BOX_TOP and cy + b <= height - BOX_BOTTOM  # of the top ones and of yours
        assert BOX_TOP <= cy <= height - BOX_BOTTOM
    assert table_geometry(2560, 1440).table_radii[0] > table_geometry(1920, 1080).table_radii[0]
    assert table_geometry(1024, 600).canvas == (CANVAS_WIDTH, CANVAS_HEIGHT)


def test_the_advice_is_what_the_model_plays_at_a_table():
    """The spot's numbers are the policy's own: the same features, mask and normalisation
    `RLAgentPlayer` uses when the model sits at a table -- the stack constant from training
    (100 bb), whatever the spot's stacks are."""
    pytest.importorskip("torch")
    from support import tiny_model

    from pokerlab.gui.spot import advise
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.policy import make_policy_fn

    model = tiny_model(hidden=16, num_layers=1)
    model.feature_stack_bb = 100.0
    spot = Spot(num_players=4, my_seat=3, starting_stack=25000, small_blind=50, big_blind=100,
                hole_cards=(parse_card("Ah"), parse_card("Kh")))  # 250 bb stacks
    state = replay(spot)
    advice = advise(state.observation, state.legal_actions, [("m", 1500.0, model)], big_blind=100)[0]
    player = RLAgentPlayer("m", "m", policy_fn=make_policy_fn(model), big_blind=100, starting_stack=10000)
    played = player.action_probabilities(state.observation, state.legal_actions)
    advised = {b.index: b.probability for b in advice.bins}  # the legal bins only
    assert advised == pytest.approx({index: played[index] for index in advised})
    assert sum(played[index] for index in advised) == pytest.approx(1.0)  # nothing legal left out


def test_with_nothing_to_call_folding_is_never_advised():
    pytest.importorskip("torch")
    from support import tiny_model

    from pokerlab.gui.spot import advise

    model = tiny_model(hidden=16, num_layers=1)
    call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
    board = (parse_card("Ks"), parse_card("8d"), parse_card("3h"))
    for spot in (
        Spot(num_players=6, my_seat=2, script=[call] * 5),  # the big blind after limps
        Spot(num_players=3, my_seat=0, script=[call, call, check, check, check], board=board),  # checked to
    ):
        state = replay(spot)
        assert state.to_act == spot.my_seat
        advice = advise(state.observation, state.legal_actions, [("m", 1500.0, model)], big_blind=spot.big_blind)[0]
        assert all(b.action.action_type is not ActionType.FOLD for b in advice.bins)


def test_a_model_is_normalised_by_the_stack_it_was_trained_with():
    from pokerlab.rl.table_mix import DEFAULT_STACK_MAX_BB, feature_stack_bb

    assert feature_stack_bb({"metadata": {"stack_max_bb": 60.0}}) == 60.0
    assert feature_stack_bb({"metadata": {}}) == DEFAULT_STACK_MAX_BB == feature_stack_bb({})
