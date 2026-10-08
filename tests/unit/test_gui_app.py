import queue
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter")
from tkinter import ttk

import pokerlab.gui.app as app_module
from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.history import SCHEMA_VERSION, HandHistory
from pokerlab.engine.state import ActionRecord, PlayerStatus, Street
from pokerlab.engine.table import HandResult
from pokerlab.gui.app import (
    DEALER_BACKGROUND,
    TURN_BACKGROUND,
    ActionReporter,
    TableFrame,
    _bot_spec_label,
    _bot_spec_to_key_string,
    _describe_action,
    _describe_combination,
    _format_hand_summary,
    _pot_fraction_raise_to,
)
from pokerlab.players.base import Observation, SeatPublicInfo
from pokerlab.players.gui import GuiEvent, GuiPlayer


def _card_item_counts(widgets: dict) -> list[int]:
    return [len(c.find_all()) for c in widgets["cards"]]


@pytest.fixture
def table_frame(app):
    # TableFrame's `master` doubles as both its Tk parent widget and "the
    # app" (it calls self.app.show_setup), so it needs a real PokerGuiApp,
    # not a bare tk.Tk().
    frame = TableFrame(
        app,
        num_players=3,
        event_queue=queue.Queue(),
        human_player=None,
        history_path="unused.jsonl",
        seat_names={0: "A", 1: "B", 2: "C"},
        step_gate=queue.Queue(),
        step_mode_state={"on": False},
    )
    yield frame
    frame.destroy()


def test_bot_spec_to_key_string_for_a_model():
    """Every bot spec is a trained model given by path."""
    assert _bot_spec_to_key_string({"key": "model", "path": "checkpoints/pool/agent.pt"}) == (
        "model:checkpoints/pool/agent.pt"
    )


def test_bot_spec_label_shows_the_model_name():
    label = _bot_spec_label({"key": "model", "path": "checkpoints/pool/agent-iter00100.pt"})
    assert "agent-iter00100" in label


def test_hide_seat_removes_the_box_from_the_grid(table_frame):
    box = table_frame.seat_widgets[1]["frame"]
    assert box.grid_info()  # visible initially

    table_frame._hide_seat(1)

    assert not table_frame.seat_widgets[1]["frame"].grid_info()
    assert 1 in table_frame._busted_seats


def test_hide_seat_is_idempotent(table_frame):
    table_frame._hide_seat(2)
    table_frame._hide_seat(2)  # must not raise or double-count
    assert table_frame._busted_seats == {2}


def test_hide_seat_also_removes_the_dealer_label(table_frame):
    label = table_frame.seat_widgets[1]["dealer_label"]
    assert label.grid_info()  # visible initially

    table_frame._hide_seat(1)

    assert not label.grid_info()


def test_render_observation_marks_only_the_button_seat_as_dealer(table_frame):
    seats = (
        SeatPublicInfo(seat=0, name="A", stack=200, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False),
        SeatPublicInfo(seat=1, name="B", stack=200, current_bet=0, status=PlayerStatus.ACTIVE, is_button=True),
        SeatPublicInfo(seat=2, name="C", stack=200, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False),
    )
    observation = Observation(
        street=Street.PREFLOP,
        hole_cards=(),
        community_cards=(),
        pot_size=0,
        current_bet_to_match=0,
        min_raise=2,
        my_seat=0,
        my_stack=200,
        my_current_bet=0,
        seats=seats,
        button_seat=1,
        action_history=(),
    )

    table_frame._render_observation(observation)

    assert table_frame.seat_widgets[0]["dealer"].get() == ""
    assert table_frame.seat_widgets[1]["dealer"].get() == "D"
    assert table_frame.seat_widgets[2]["dealer"].get() == ""


def test_reset_seats_for_new_hand_hides_non_participants_and_shows_backs(table_frame):
    table_frame._all_hole_cards = {0: (Card.parse("Ah"), Card.parse("Kh")), 1: (Card.parse("2c"), Card.parse("7d"))}

    table_frame._reset_seats_for_new_hand([0, 1])  # seat 2 busted before this hand

    assert 2 in table_frame._busted_seats
    assert not table_frame.seat_widgets[2]["frame"].grid_info()
    for seat in (0, 1):
        assert table_frame.seat_widgets[seat]["frame"].grid_info()
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [2, 2]  # card-backs, spy off


def test_reset_seats_for_new_hand_reveals_cards_when_spy_is_on(table_frame):
    hole_cards = {0: (Card.parse("Ah"), Card.parse("Kh")), 1: (Card.parse("2c"), Card.parse("7d"))}
    table_frame._all_hole_cards = hole_cards
    table_frame.spy_var.set(True)

    table_frame._reset_seats_for_new_hand(hole_cards.keys())

    for seat in (0, 1):
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [4, 4]  # revealed faces


def test_draw_seat_cards_shows_empty_slot_for_folded_seat_even_with_spy_on(table_frame):
    table_frame._all_hole_cards = {1: (Card.parse("2c"), Card.parse("7d"))}
    table_frame.spy_var.set(True)
    folded = SeatPublicInfo(seat=1, name="B", stack=100, current_bet=0, status=PlayerStatus.FOLDED, is_button=False)

    table_frame._draw_seat_cards(folded, is_me=False)

    assert _card_item_counts(table_frame.seat_widgets[1]) == [1, 1]  # empty slot, not revealed


def test_draw_seat_cards_never_reveals_my_own_seat_via_spy(table_frame):
    table_frame._all_hole_cards = {0: (Card.parse("Ah"), Card.parse("Kh"))}
    table_frame.spy_var.set(True)
    me = SeatPublicInfo(seat=0, name="A", stack=100, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False)

    table_frame._draw_seat_cards(me, is_me=True)

    assert _card_item_counts(table_frame.seat_widgets[0]) == [1, 1]  # empty -- shown separately above instead


def test_render_observation_keeps_human_hole_cards_when_other_player_acts(table_frame):
    table_frame._human_seat = 0
    table_frame._all_hole_cards = {
        0: (Card.parse("Ah"), Card.parse("Kh")),
        1: (Card.parse("2c"), Card.parse("7d")),
    }
    seats = (
        SeatPublicInfo(seat=0, name="A", stack=200, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False),
        SeatPublicInfo(seat=1, name="B", stack=200, current_bet=0, status=PlayerStatus.ACTIVE, is_button=True),
    )
    observation = Observation(
        street=Street.PREFLOP,
        hole_cards=(Card.parse("2c"), Card.parse("7d")),
        community_cards=(),
        pot_size=0,
        current_bet_to_match=0,
        min_raise=2,
        my_seat=1,
        my_stack=200,
        my_current_bet=0,
        seats=seats,
        button_seat=1,
        action_history=(),
    )

    table_frame._render_observation(observation)

    assert _card_item_counts({"cards": table_frame.hole_canvases}) == [4, 4]
    assert str(table_frame.menu_button.cget("state")) == "normal"


# --------------------------------------------------------------------------
# Raise sizing: mouse wheel over the slider, and the pot-fraction shortcuts
# --------------------------------------------------------------------------


def _flop_observation(pot_size=100, to_match=20, my_bet=0, my_stack=200, my_seat=0, folded=()):
    seats = tuple(
        SeatPublicInfo(
            seat=i,
            name=n,
            stack=200,
            current_bet=0,
            status=PlayerStatus.FOLDED if i in folded else PlayerStatus.ACTIVE,
            is_button=(i == 1),
        )
        for i, n in enumerate(["A", "B", "C"])
    )
    return Observation(
        street=Street.FLOP,
        hole_cards=(Card.parse("Ah"), Card.parse("Kh")),
        community_cards=(Card.parse("2h"), Card.parse("9s"), Card.parse("Jd")),
        pot_size=pot_size,
        current_bet_to_match=to_match,
        min_raise=20,
        my_seat=my_seat,
        my_stack=my_stack,
        my_current_bet=my_bet,
        seats=seats,
        button_seat=1,
        action_history=(),
    )


def _widgets_under(widget, out=None):
    out = [] if out is None else out
    out.append(widget)
    for child in widget.winfo_children():
        _widgets_under(child, out)
    return out


def _of_class(widget, cls):
    return [w for w in _widgets_under(widget) if w.winfo_class() == cls]


def _amount_shown(frame):
    """The 'Raise: <n>' label the panel keeps its chosen amount in."""
    for label in _of_class(frame.actions_frame, "TLabel"):
        name = label.cget("textvariable")
        if not name:
            continue
        text = frame.tk.globalgetvar(name)
        if ":" in text and text.split(":")[0] in ("Raise", "Bet"):
            return int(text.split(":")[1])
    raise AssertionError("no bet/raise amount label found")


def test_pot_fraction_raise_to_sizes_off_the_pot_after_the_call():
    """A pot-sized raise is call-first-then-bet-the-pot: with 100 in the
    middle and 20 to call, the pot is 120 once matched, so the raise-to
    level is 20 + 120 = 140 -- not 100 and not 120."""
    observation = _flop_observation(pot_size=100, to_match=20, my_bet=0)

    assert _pot_fraction_raise_to(observation, 1.0) == 140
    assert _pot_fraction_raise_to(observation, 0.5) == 80
    assert _pot_fraction_raise_to(observation, 0.30) == 56


def test_pot_fraction_raise_to_counts_chips_already_in_front_of_me():
    """my_current_bet is part of the raise-to level (a street total, see
    Action's docstring), so having already put 10 in must not make the
    same sizing come out smaller."""
    observation = _flop_observation(pot_size=100, to_match=20, my_bet=10)

    assert _pot_fraction_raise_to(observation, 1.0) == 20 + (100 + 10)


def test_pot_fraction_buttons_set_the_amount_without_submitting(table_frame):
    observation = _flop_observation()
    table_frame._render_actions(observation, [LegalAction(ActionType.RAISE, 40, 200)])

    buttons = {b.cget("text"): b for b in _of_class(table_frame.actions_frame, "TButton")}
    assert {"30%", "50%", "66%", "Piatto"} <= set(buttons)

    buttons["Piatto"].invoke()
    assert _amount_shown(table_frame) == 140
    buttons["50%"].invoke()
    assert _amount_shown(table_frame) == 80
    # Still the player's turn: nothing was sent to the engine.
    assert table_frame.human_player is None


def test_pot_fraction_buttons_clamp_into_the_legal_range(table_frame):
    """A pot-sized raise of 140 is illegal when the stack only allows 90 --
    the button offers the legal maximum rather than an amount the engine
    would reject."""
    observation = _flop_observation()
    table_frame._render_actions(observation, [LegalAction(ActionType.RAISE, 40, 90)])

    buttons = {b.cget("text"): b for b in _of_class(table_frame.actions_frame, "TButton")}
    buttons["Piatto"].invoke()

    assert _amount_shown(table_frame) == 90


def test_mouse_wheel_over_the_slider_moves_the_bet_by_one_big_blind(app):
    frame = TableFrame(
        app,
        num_players=3,
        event_queue=queue.Queue(),
        human_player=None,
        history_path="unused.jsonl",
        seat_names={0: "A", 1: "B", 2: "C"},
        step_gate=queue.Queue(),
        step_mode_state={"on": False},
        big_blind=2,
    )
    try:
        frame._render_actions(_flop_observation(), [LegalAction(ActionType.RAISE, 40, 200)])
        scale = _of_class(frame.actions_frame, "TScale")[0]
        frame.update()
        # Start away from either bound, so a clamp cannot be mistaken for a
        # step in the wrong direction (which is how an inverted wheel hides).
        {b.cget("text"): b for b in _of_class(frame.actions_frame, "TButton")}["50%"].invoke()
        start = _amount_shown(frame)
        assert start == 80

        scale.event_generate("<Button-4>", x=5, y=5)  # X11 wheel up
        assert _amount_shown(frame) == start + 2
        scale.event_generate("<Button-5>", x=5, y=5)  # X11 wheel down
        scale.event_generate("<Button-5>", x=5, y=5)
        assert _amount_shown(frame) == start - 2

        # Windows/macOS report a signed delta on <MouseWheel> instead.
        scale.event_generate("<MouseWheel>", x=5, y=5, delta=120)
        assert _amount_shown(frame) == start
    finally:
        frame.destroy()


def test_mouse_wheel_cannot_scroll_past_the_legal_bounds(app):
    frame = TableFrame(
        app,
        num_players=3,
        event_queue=queue.Queue(),
        human_player=None,
        history_path="unused.jsonl",
        seat_names={0: "A", 1: "B", 2: "C"},
        step_gate=queue.Queue(),
        step_mode_state={"on": False},
        big_blind=2,
    )
    try:
        frame._render_actions(_flop_observation(), [LegalAction(ActionType.RAISE, 40, 60)])
        scale = _of_class(frame.actions_frame, "TScale")[0]
        frame.update()

        for _ in range(50):
            scale.event_generate("<Button-5>", x=5, y=5)
        assert _amount_shown(frame) == 40
        for _ in range(100):
            scale.event_generate("<Button-4>", x=5, y=5)
        assert _amount_shown(frame) == 60
    finally:
        frame.destroy()


def test_confirm_submits_the_amount_chosen_by_a_shortcut(app):
    human = GuiPlayer("h", "Io", queue.Queue())
    frame = TableFrame(
        app,
        num_players=3,
        event_queue=queue.Queue(),
        human_player=human,
        history_path="unused.jsonl",
        seat_names={0: "Io", 1: "B", 2: "C"},
        step_gate=queue.Queue(),
        step_mode_state={"on": False},
        big_blind=2,
    )
    try:
        frame._render_actions(_flop_observation(), [LegalAction(ActionType.RAISE, 40, 200)])
        buttons = {b.cget("text"): b for b in _of_class(frame.actions_frame, "TButton")}
        buttons["66%"].invoke()
        buttons["Conferma raise"].invoke()

        action = human.decisions.get_nowait()
        assert action.action_type == ActionType.RAISE
        assert action.amount == _pot_fraction_raise_to(_flop_observation(), 0.66)
    finally:
        frame.destroy()


# --------------------------------------------------------------------------
# Dealer highlight
# --------------------------------------------------------------------------


def _seat_style(table_frame, seat):
    return str(table_frame.seat_widgets[seat]["frame"].cget("style"))


def test_the_dealer_seat_box_is_tinted_whole(table_frame):
    table_frame._set_dealer_seat(1)

    assert _seat_style(table_frame, 1) == "Dealer.TLabelframe"
    widgets = table_frame.seat_widgets[1]
    # The tint has to reach the ttk children too: they take their
    # background from their style, not from the parent frame.
    assert all(str(label.cget("style")) == "Dealer.TLabel" for label in widgets["labels"])
    assert str(widgets["cards_frame"].cget("style")) == "Dealer.TFrame"
    assert all(str(c.cget("background")) == DEALER_BACKGROUND for c in widgets["cards"])


def test_only_one_seat_is_tinted_and_the_tint_follows_the_button(table_frame):
    table_frame._set_dealer_seat(1)
    table_frame._set_dealer_seat(2)

    assert _seat_style(table_frame, 2) == "Dealer.TLabelframe"
    for seat in (0, 1):
        assert _seat_style(table_frame, seat) == "Seat.TLabelframe"
        widgets = table_frame.seat_widgets[seat]
        assert all(str(c.cget("background")) != DEALER_BACKGROUND for c in widgets["cards"])


def test_render_observation_tints_the_button_seat(table_frame):
    observation = _flop_observation()  # button_seat=1

    table_frame._render_observation(observation)

    assert _seat_style(table_frame, 1) == "Dealer.TLabelframe"
    assert _seat_style(table_frame, 0) == "Seat.TLabelframe"


# --------------------------------------------------------------------------
# All-in runout, and the end-of-hand reveal
# --------------------------------------------------------------------------


_HOLE = {
    0: (Card.parse("Ah"), Card.parse("Kh")),
    1: (Card.parse("2c"), Card.parse("7d")),
    2: (Card.parse("Qs"), Card.parse("Qd")),
}
_BOARD = [Card.parse("2h"), Card.parse("9s"), Card.parse("Jd"), Card.parse("4c"), Card.parse("Qc")]


def _hand_started_event(button_seat=2):
    return GuiEvent(
        "hand_started",
        {
            "hand_id": "hand-1",
            "button_seat": button_seat,
            "sb_seat": 0,
            "bb_seat": 1,
            "small_blind": 1,
            "big_blind": 2,
            "hole_cards": dict(_HOLE),
        },
    )


def _street_event(street, cards, betting_closed):
    return GuiEvent(
        "street_dealt",
        {"hand_id": "hand-1", "street": street, "community_cards": cards, "betting_closed": betting_closed},
    )


def _hand_result(final_stacks=None, payouts=None):
    final_stacks = final_stacks if final_stacks is not None else {0: 0, 1: 0, 2: 600}
    history = HandHistory(
        schema_version=SCHEMA_VERSION,
        hand_id="hand-1",
        started_at=0.0,
        num_players=3,
        small_blind=1,
        big_blind=2,
        button_seat=2,
        starting_stacks={0: 200, 1: 200, 2: 200},
        seat_names={0: "A", 1: "B", 2: "C"},
        community_cards=list(_BOARD),
        actions=[],
        hole_cards=dict(_HOLE),
        payouts=payouts if payouts is not None else {2: 600},
        final_stacks=final_stacks,
    )
    return HandResult(hand_id="hand-1", payouts=history.payouts, final_stacks=final_stacks, hand_history=history)


def test_an_all_in_runout_draws_the_board_nobody_acted_on(table_frame):
    """The bug this fixes: with everyone all-in, Table calls no Player, so
    the GUI saw nothing between the last bet and the final stacks and the
    board never appeared."""
    table_frame._handle_event(_hand_started_event())

    table_frame._handle_event(_street_event(Street.FLOP, _BOARD[:3], betting_closed=True))

    assert [len(c.find_all()) for c in table_frame.board_canvases] == [4, 4, 4, 1, 1]
    assert "All-in" in table_frame.waiting_var.get()


def test_an_all_in_runout_turns_over_everyone_still_in_the_pot(table_frame):
    table_frame._handle_event(_hand_started_event())

    table_frame._handle_event(_street_event(Street.FLOP, _BOARD[:3], betting_closed=True))

    for seat in (0, 1, 2):
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [4, 4]


def test_an_all_in_runout_keeps_a_folded_players_cards_mucked(table_frame):
    """Who has mucked is read off the Observation, which now carries the
    state from after the action -- so a fold shows up in it immediately."""
    table_frame._handle_event(_hand_started_event())
    # The hook's Observation is the state *after* the action, so the seat
    # that just folded already reads FOLDED in it -- which it never did
    # when the GUI was fed the pre-action snapshot.
    table_frame._handle_event(
        _action_event(
            seat=1,
            name="B",
            action_type=ActionType.FOLD,
            observation=_flop_observation(my_seat=1, folded=(1,)),
        )
    )

    table_frame._handle_event(_street_event(Street.FLOP, _BOARD[:3], betting_closed=True))

    assert table_frame._folded_seats() == {1}
    assert _card_item_counts(table_frame.seat_widgets[1]) == [1, 1]  # mucked
    assert _card_item_counts(table_frame.seat_widgets[0]) == [4, 4]
    assert _card_item_counts(table_frame.seat_widgets[2]) == [4, 4]


def test_an_ordinary_street_is_drawn_without_revealing_anyone(table_frame):
    table_frame._handle_event(_hand_started_event())

    table_frame._handle_event(_street_event(Street.FLOP, _BOARD[:3], betting_closed=False))

    assert [len(c.find_all()) for c in table_frame.board_canvases] == [4, 4, 4, 1, 1]
    for seat in (0, 1, 2):
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [2, 2]  # still backs


def test_the_end_of_a_hand_reveals_every_participant(table_frame):
    table_frame._handle_event(_hand_started_event())

    table_frame._handle_event(GuiEvent("hand_complete", _hand_result()))

    assert [len(c.find_all()) for c in table_frame.board_canvases] == [4, 4, 4, 4, 4]
    for seat in (0, 1, 2):
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [4, 4]


def test_a_seat_that_just_busted_is_not_hidden_before_its_cards_are_shown(table_frame):
    """A player who went all-in and lost is exactly the one whose cards are
    worth seeing, so the seat survives the reveal and is only hidden when
    the next hand starts."""
    result = _hand_result(final_stacks={0: 0, 1: 0, 2: 600})
    table_frame._handle_event(_hand_started_event())

    table_frame._handle_event(GuiEvent("hand_complete", result))
    assert table_frame._busted_seats == set()
    assert _card_item_counts(table_frame.seat_widgets[0]) == [4, 4]

    table_frame._handle_event(_hand_started_event())
    assert table_frame._busted_seats == {0, 1}


def test_the_hand_stays_on_the_table_until_the_next_one_starts(table_frame):
    result = _hand_result(final_stacks={0: 100, 1: 100, 2: 400})
    table_frame._handle_event(_hand_started_event())
    table_frame._handle_event(GuiEvent("hand_complete", result))
    table_frame._handle_event(GuiEvent("awaiting_next_hand"))
    table_frame.update()

    assert [len(c.find_all()) for c in table_frame.board_canvases] == [4, 4, 4, 4, 4]
    for seat in (0, 1, 2):
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [4, 4]

    table_frame._handle_event(_hand_started_event())

    assert [len(c.find_all()) for c in table_frame.board_canvases] == [1] * 5
    assert [len(c.find_all()) for c in table_frame.hole_canvases] == [1, 1]
    assert table_frame._busted_seats == set()


def test_the_next_hand_button_releases_the_session_thread(table_frame):
    table_frame._handle_event(GuiEvent("awaiting_next_hand"))
    assert table_frame.next_hand_gate.empty()
    assert table_frame.next_hand_button is not None

    table_frame.next_hand_button.invoke()

    assert table_frame.next_hand_gate.qsize() == 1
    assert table_frame.next_hand_button is None


def test_the_session_waits_for_the_next_hand_button_between_hands():
    import threading

    table = SimpleNamespace(stacks=[100, 100], play_hand=lambda: SimpleNamespace(hand_history=None))
    events: queue.Queue = queue.Queue()
    gate: queue.Queue = queue.Queue()
    writer = SimpleNamespace(close=lambda: None)
    thread = threading.Thread(target=app_module._run_session, args=(table, 2, events, writer, gate), daemon=True)
    thread.start()

    assert events.get(timeout=2).kind == "hand_complete"
    assert events.get(timeout=2).kind == "awaiting_next_hand"
    thread.join(timeout=0.3)
    assert thread.is_alive()  # held: the button has not been pressed

    gate.put(None)

    assert events.get(timeout=2).kind == "hand_complete"
    assert events.get(timeout=2).kind == "session_complete"
    thread.join(timeout=2)
    assert not thread.is_alive()


# --------------------------------------------------------------------------
# The log's winning combination
# --------------------------------------------------------------------------


def test_describe_combination_names_the_best_five_in_italian():
    text = _describe_combination((Card.parse("Qs"), Card.parse("Qd")), _BOARD)

    assert text is not None
    assert text.startswith("Tris")  # QsQd + Qc on the board
    assert "Qs" in text and "Qd" in text and "Qc" in text


def test_describe_combination_has_nothing_to_say_before_the_flop():
    """Fewer than five cards: `evaluate` would raise rather than guess, so
    a hand won preflop simply has no combination to show."""
    assert _describe_combination((Card.parse("Ah"), Card.parse("Kh")), []) is None
    assert _describe_combination(None, _BOARD) is None


def test_describe_combination_distinguishes_a_royal_flush():
    board = [Card.parse(c) for c in ("Qh", "Jh", "Th", "2c", "3d")]
    royal = _describe_combination((Card.parse("Ah"), Card.parse("Kh")), board)
    lower = _describe_combination((Card.parse("9h"), Card.parse("8h")), board)

    assert royal.startswith("Scala reale")
    assert lower.startswith("Scala colore")


def test_the_hand_summary_names_the_winning_combination():
    summary = _format_hand_summary(_hand_result())

    assert "Pot vinto da: C (+600) con Tris" in summary
    assert "Board: 2h 9s Jd 4c Qc" in summary
    # Every contestant's own combination is shown next to their cards.
    assert "A: Ah Kh  ->  " in summary
    assert "Coppia" in summary  # B's 2c7d pairs the board's 2h


def test_a_pot_won_without_a_showdown_claims_no_combination():
    """Nothing was shown and the board may not even be complete, so naming
    a "winning combination" there would be inventing one."""
    from pokerlab.engine.state import ActionRecord

    result = _hand_result(final_stacks={0: 210, 1: 195, 2: 195}, payouts={0: 210})
    result.hand_history.actions = [
        ActionRecord(Street.PREFLOP, seat, f"p{seat}", ActionType.FOLD, 0, 200, 200, 3, 0.0) for seat in (1, 2)
    ]

    summary = _format_hand_summary(result)

    assert "senza showdown" in summary
    assert "con " not in summary.split("Pot vinto da:")[1]
    assert "Showdown:" not in summary


def test_the_winning_five_is_printed_strongest_first():
    """HandRank keeps best_five in combination order, which buries the pair
    among the kickers; the log sorts it the way a player reads a hand."""
    board = [Card.parse(c) for c in ("Tc", "5c", "Th", "2d", "Kd")]

    text = _describe_combination((Card.parse("Ah"), Card.parse("7h")), board)

    assert text == "Coppia (Tc Th Ah Kd 7h)"


# --------------------------------------------------------------------------
# Actions are reported from the engine hook, with the state AFTER the action
# --------------------------------------------------------------------------


def _record(seat=0, action_type=ActionType.CALL, amount=10, stack_before=200, stack_after=190, street=Street.FLOP):
    return ActionRecord(
        street=street,
        seat=seat,
        player_id=f"p{seat}",
        action_type=action_type,
        amount=amount,
        stack_before=stack_before,
        stack_after=stack_after,
        pot_before=20,
        timestamp=0.0,
    )


def _action_event(seat=0, name="A", action_type=ActionType.CALL, observation=None, **record_kwargs):
    record = _record(seat=seat, action_type=action_type, **record_kwargs)
    return GuiEvent(
        "action_taken",
        {
            "hand_id": "hand-1",
            "seat": seat,
            "player_id": f"p{seat}",
            "name": name,
            "action": Action(action_type),
            "record": record,
            "observation": observation if observation is not None else _flop_observation(my_seat=seat),
        },
    )


def test_describe_action_reads_the_chips_off_the_engines_record():
    """CALL and ALL_IN leave Action.amount meaningless, so the chips come
    from stack_before - stack_after; BET/RAISE report the record's amount,
    which is the player's street total, i.e. the raise-to level."""
    assert _describe_action(_record(action_type=ActionType.CALL, stack_before=200, stack_after=180)) == "call 20"
    assert _describe_action(_record(action_type=ActionType.ALL_IN, stack_before=75, stack_after=0)) == "all-in (75)"
    assert _describe_action(_record(action_type=ActionType.RAISE, amount=60)) == "raise to 60"
    assert _describe_action(_record(action_type=ActionType.BET, amount=25)) == "bet 25"
    assert _describe_action(_record(action_type=ActionType.CHECK)) == "check"
    assert _describe_action(_record(action_type=ActionType.FOLD)) == "fold"


def test_an_action_is_logged_with_the_street_it_was_taken_on(table_frame):
    table_frame._handle_event(_hand_started_event())

    table_frame._handle_event(
        _action_event(seat=2, name="C", action_type=ActionType.CALL, stack_before=200, stack_after=185)
    )

    assert "[FLOP] C: call 15" in table_frame.log.get("1.0", "end")


def test_rendering_an_acting_bot_never_labels_it_as_the_human(table_frame):
    """`Observation.my_seat` belongs to whoever is acting, so reading "me"
    off it marked that bot "(tu)" and drew its cards as an empty slot --
    it looked folded, and in spectator mode one bot always wore the
    marker. Identity is the GUI's own `_human_seat`."""
    table_frame._all_hole_cards = {s: (Card.parse("Ah"), Card.parse("Kh")) for s in range(3)}
    table_frame._reset_seats_for_new_hand([0, 1, 2])

    table_frame._render_observation(_flop_observation(my_seat=1))  # a bot acts

    assert table_frame.seat_widgets[1]["name"].get() == "B"
    assert "(tu)" not in " ".join(table_frame.seat_widgets[s]["name"].get() for s in range(3))
    for seat in (0, 1, 2):
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [2, 2]  # backs, none mucked


def test_the_human_seat_keeps_its_marker_whoever_is_acting(table_frame):
    table_frame._human_seat = 0

    table_frame._render_observation(_flop_observation(my_seat=2))  # a bot acts

    assert table_frame.seat_widgets[0]["name"].get() == "A (tu)"
    assert table_frame.seat_widgets[2]["name"].get() == "C"


def test_spectator_mode_shows_no_hole_cards_of_its_own(table_frame):
    """With nobody sitting in, "Le tue carte" has nothing to show -- it used
    to fall back to the acting player's cards and flip between opponents'
    hands every action."""
    assert table_frame._human_seat is None

    table_frame._render_observation(_flop_observation(my_seat=1))

    assert [len(c.find_all()) for c in table_frame.hole_canvases] == [1, 1]


def test_action_reporter_publishes_the_action_then_paces_a_bot():
    events: queue.Queue = queue.Queue()
    reporter = ActionReporter(events, queue.Queue(), {"on": False}, human_seat=0, delay_seconds=0.05)
    info = _action_event(seat=1, name="B").payload

    started = time.monotonic()
    reporter(info)

    assert events.get_nowait().payload is info
    assert time.monotonic() - started >= 0.05


def test_action_reporter_does_not_pause_on_the_humans_own_action():
    """They already "stepped" by clicking."""
    events: queue.Queue = queue.Queue()
    reporter = ActionReporter(events, queue.Queue(), {"on": True}, human_seat=0, delay_seconds=5.0)

    started = time.monotonic()
    reporter(_action_event(seat=0, name="A").payload)

    assert events.get_nowait().kind == "action_taken"
    assert time.monotonic() - started < 1.0


def test_action_reporter_blocks_on_the_step_gate_in_step_mode():
    events: queue.Queue = queue.Queue()
    gate: queue.Queue = queue.Queue()
    reporter = ActionReporter(events, gate, {"on": True}, human_seat=0, delay_seconds=0)
    done = threading.Event()
    worker = threading.Thread(target=lambda: (reporter(_action_event(seat=1).payload), done.set()))
    worker.start()
    try:
        assert not done.wait(timeout=0.2), "should be blocked until 'Avanti'"
        gate.put(None)
        assert done.wait(timeout=2)
    finally:
        worker.join(timeout=2)


def test_a_runout_with_no_action_at_all_reveals_everyone(table_frame):
    """Every seat all-in on its own blind: the preflop betting round runs
    nobody, so there is no Observation to read statuses from and nobody
    has folded."""
    table_frame._handle_event(_hand_started_event())
    assert table_frame._last_observation is None

    table_frame._handle_event(_street_event(Street.FLOP, _BOARD[:3], betting_closed=True))

    assert table_frame._folded_seats() == set()
    for seat in (0, 1, 2):
        assert _card_item_counts(table_frame.seat_widgets[seat]) == [4, 4]


def test_a_session_gives_the_models_their_opponents_statistics(app, tmp_path, monkeypatch):
    """The table of a GUI session has a `StatsTracker`, so a model seated there reads its
    opponents' VPIP, PFR... like at every table it was trained and rated on (without one
    every opponent is "unknown" for the whole session)."""
    pytest.importorskip("torch")
    from support import tiny_model

    import pokerlab.gui.app as app_module
    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.stats import StatsTracker
    from pokerlab.rl.ppo import save_checkpoint

    checkpoint = tmp_path / "agent.pt"
    save_checkpoint(checkpoint, tiny_model())
    built: list = []
    real_table = app_module.Table

    def recording_table(*args, **kwargs):
        built.append(real_table(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(app_module, "Table", recording_table)
    monkeypatch.setattr(app_module, "_run_session", lambda *args: None)  # build the table, play nothing
    monkeypatch.chdir(tmp_path)  # the session opens its hand-history file in the working directory
    config = GameConfig(num_players=3, starting_stack=2000, small_blind=10, big_blind=20)
    app.start_session(config, 0, [f"model:{checkpoint}"] * 3, hands=1, seed=1)

    table = built[0]
    assert isinstance(table.stats_tracker, StatsTracker)
    for _ in range(3):
        table.stacks = [2000] * 3
        table.play_hand()
    assert all(table.stats_tracker.hands(player.player_id) == 3 for player in table.players)


# ---- blinds that go up -------------------------------------------------------------


def test_the_table_follows_the_blinds_of_each_hand_and_says_when_they_go_up(table_frame):
    started = _hand_started_event()
    table_frame._handle_event(started)
    assert table_frame.big_blind == 2
    assert "bui salgono" not in table_frame.log.get("1.0", "end")

    raised = _hand_started_event()
    raised.payload.update(small_blind=3, big_blind=5, hand_id="hand-2")
    table_frame._handle_event(raised)
    assert table_frame.big_blind == 5  # the mouse wheel moves a raise in big blinds
    text = table_frame.log.get("1.0", "end")
    assert "I bui salgono: 3/5" in text and "big blind prima: 2" in text


def test_the_setup_screen_reads_the_blind_increase_from_its_two_fields(app):
    from pokerlab.engine.config import BlindSchedule
    from pokerlab.gui.app import SetupFrame

    setup = SetupFrame(app)
    try:
        assert setup._blind_schedule() is None  # the first field empty: the blinds never move
        setup.blind_every_var.set("5")
        setup.blind_factor_var.set("1,2")  # a comma, as it is written here
        assert setup._blind_schedule() == BlindSchedule(every=5, factor=1.2)
        setup.blind_factor_var.set("1.5")
        assert setup._blind_schedule() == BlindSchedule(every=5, factor=1.5)
        for every, factor in (("0", "1.2"), ("5", "0.8"), ("cinque", "1.2"), ("5", "molto")):
            setup.blind_every_var.set(every)
            setup.blind_factor_var.set(factor)
            with pytest.raises(ValueError):
                setup._blind_schedule()
    finally:
        setup.destroy()


def test_a_session_raises_the_blinds_on_schedule(app, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from support import tiny_model

    import pokerlab.gui.app as app_module
    from pokerlab.engine.config import BlindSchedule, GameConfig
    from pokerlab.rl.ppo import save_checkpoint

    checkpoint = tmp_path / "agent.pt"
    save_checkpoint(checkpoint, tiny_model())
    built: list = []
    real_table = app_module.Table

    def recording_table(*args, **kwargs):
        built.append(real_table(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(app_module, "Table", recording_table)
    monkeypatch.setattr(app_module, "_run_session", lambda *args: None)
    monkeypatch.chdir(tmp_path)
    config = GameConfig(num_players=3, starting_stack=2000, small_blind=10, big_blind=20)
    app.start_session(config, 0, [f"model:{checkpoint}"] * 3, hands=1, seed=1, blind_schedule=BlindSchedule(2, 1.5))

    table = built[0]
    assert table.blind_schedule == BlindSchedule(2, 1.5)
    played = []
    for _ in range(5):
        table.stacks = [2000] * 3
        played.append(table.play_hand().hand_history.big_blind)
    assert played == [20, 20, 30, 30, 45]


# --------------------------------------------------------------------------
# The seat whose turn it is, and each seat's last action
# --------------------------------------------------------------------------


def test_the_seat_whose_turn_it_is_is_painted_green_and_the_paint_follows_the_turn(table_frame):
    table_frame._set_turn_seat(2)
    widgets = table_frame.seat_widgets[2]
    assert _seat_style(table_frame, 2) == "Turn.TLabelframe"
    assert all(str(label.cget("style")) == "Turn.TLabel" for label in widgets["labels"])
    assert str(widgets["cards_frame"].cget("style")) == "Turn.TFrame"
    assert all(str(c.cget("background")) == TURN_BACKGROUND for c in widgets["cards"])

    table_frame._set_turn_seat(0)
    assert _seat_style(table_frame, 0) == "Turn.TLabelframe"
    assert _seat_style(table_frame, 2) == "Seat.TLabelframe"  # only one seat at a time

    table_frame._set_turn_seat(None)
    assert all(_seat_style(table_frame, seat) == "Seat.TLabelframe" for seat in range(3))


def test_the_turn_wins_over_the_dealer_and_the_dealer_colour_comes_back(table_frame):
    table_frame._set_dealer_seat(1)
    table_frame._set_turn_seat(1)
    assert _seat_style(table_frame, 1) == "Turn.TLabelframe"
    table_frame._set_turn_seat(2)
    assert _seat_style(table_frame, 1) == "Dealer.TLabelframe"
    assert _seat_style(table_frame, 2) == "Turn.TLabelframe"


def test_the_events_of_a_hand_move_the_green_seat(table_frame):
    started = _hand_started_event()
    started.payload["first_actor"] = 2
    table_frame._handle_event(started)
    assert _seat_style(table_frame, 2) == "Turn.TLabelframe"

    action = _action_event(seat=2)
    action.payload["next_seat"] = 0  # announced before player 0 is asked
    table_frame._handle_event(action)
    assert _seat_style(table_frame, 0) == "Turn.TLabelframe"
    assert _seat_style(table_frame, 2) == "Seat.TLabelframe"

    table_frame._handle_event(GuiEvent("your_turn", (_flop_observation(my_seat=1), [])))
    assert _seat_style(table_frame, 1) == "Turn.TLabelframe"

    street = GuiEvent("street_dealt", {"hand_id": "hand-1", "street": Street.TURN,
                                      "community_cards": _BOARD[:4], "betting_closed": False, "first_actor": 2})
    table_frame._handle_event(street)
    assert _seat_style(table_frame, 2) == "Turn.TLabelframe"

    table_frame._handle_event(_action_event(seat=2))  # a payload without next_seat: nobody is due
    assert all(_seat_style(table_frame, seat) != "Turn.TLabelframe" for seat in range(3))


def test_each_seat_shows_its_last_action_under_its_bet(table_frame):
    history = (
        _record(seat=0, action_type=ActionType.POST_BLIND, amount=1, street=Street.PREFLOP),
        _record(seat=1, action_type=ActionType.CALL, stack_before=200, stack_after=190, street=Street.PREFLOP),
        _record(seat=1, action_type=ActionType.RAISE, amount=60, street=Street.FLOP),
        _record(seat=2, action_type=ActionType.FOLD, street=Street.FLOP),
        _record(seat=2, action_type=ActionType.ALL_IN, stack_before=75, stack_after=0, street=Street.FLOP),
    )
    observation = replace(_flop_observation(), action_history=history)
    table_frame._render_observation(observation)

    shown = {seat: table_frame.seat_widgets[seat]["last_action"].get() for seat in range(3)}
    assert shown == {0: "", 1: "Raise to 60", 2: "All-in (75)"}  # the blind is not a choice; the latest wins
    # it sits in the box right under "Bet:", above the status
    order = [str(label.cget("textvariable")) for label in table_frame.seat_widgets[1]["labels"]]
    widgets = table_frame.seat_widgets[1]
    assert order.index(str(widgets["last_action"])) == order.index(str(widgets["bet"])) + 1

    table_frame._handle_event(_hand_started_event())  # a new hand starts clean
    assert all(table_frame.seat_widgets[seat]["last_action"].get() == "" for seat in range(3))


def test_an_action_of_an_earlier_street_is_not_shown(table_frame):
    history = (
        _record(seat=1, action_type=ActionType.CALL, stack_before=200, stack_after=190, street=Street.PREFLOP),
        _record(seat=2, action_type=ActionType.RAISE, amount=60, street=Street.FLOP),
    )
    table_frame._render_observation(replace(_flop_observation(), action_history=history))

    shown = {seat: table_frame.seat_widgets[seat]["last_action"].get() for seat in range(3)}
    assert shown == {0: "", 1: "", 2: "Raise to 60"}


def test_dealing_a_street_clears_the_last_actions(table_frame):
    table_frame.seat_widgets[1]["last_action"].set("Call 10")

    table_frame._handle_event(_street_event(Street.FLOP, _BOARD[:3], betting_closed=False))

    assert all(table_frame.seat_widgets[seat]["last_action"].get() == "" for seat in range(3))


def test_a_folded_seat_is_dark_grey_until_the_next_hand(table_frame):
    table_frame._render_observation(_flop_observation(folded=(2,)))
    assert _seat_style(table_frame, 2) == "Folded.TLabelframe"
    assert _seat_style(table_frame, 0) != "Folded.TLabelframe"

    table_frame._handle_event(_hand_started_event())
    assert _seat_style(table_frame, 2) != "Folded.TLabelframe"


def test_the_green_seat_is_the_one_that_acts_next_over_real_hands(table_frame):
    """Real hands (random bots, the real ActionReporter and hooks) replayed into the frame: each
    time an action arrives, the seat painted green is the seat that took it. That is what the
    player watching sees -- the green box is the one about to act, and it is already green while
    the previous action is still on screen."""
    import random

    from support import make_random_legal_bot

    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.table import Table
    from pokerlab.gui.app import ActionReporter

    events: queue.Queue = queue.Queue()
    rng = random.Random(5)
    players = [make_random_legal_bot(f"p{i}", f"B{i}", rng=random.Random(rng.random())) for i in range(3)]
    table = Table(
        GameConfig(num_players=3, starting_stack=400, small_blind=2, big_blind=4),
        players,
        rng=random.Random(5),
        on_hand_started=lambda info: events.put(GuiEvent("hand_started", info)),
        on_street_dealt=lambda info: events.put(GuiEvent("street_dealt", info)),
        on_action_applied=ActionReporter(events, queue.Queue(), {"on": False}, None, delay_seconds=0),
    )
    for _ in range(25):
        table.stacks = [rng.randint(20, 400) for _ in range(3)]
        table.play_hand()

    green_checked = 0
    while not events.empty():
        event = events.get()
        if event.kind == "action_taken":
            greens = [seat for seat in range(3) if _seat_style(table_frame, seat) == "Turn.TLabelframe"]
            assert greens == [event.payload["seat"]], f"{greens} painted, {event.payload['seat']} acted"
            green_checked += 1
        table_frame._handle_event(event)
    assert green_checked > 25


# --------------------------------------------------------------------------
# A bot's action probabilities, in step mode with the opponents' cards shown
# --------------------------------------------------------------------------


def _probability_rows(frame):
    """`{label: percent}` of the rows the panel shows."""
    rows = {}
    for row in frame.probability_rows.winfo_children():
        texts = [child.cget("text") for child in row.winfo_children() if isinstance(child, ttk.Label)]
        rows[texts[0]] = float(texts[1].rstrip("%"))
    return rows


def _decision_for_seat(seat=1):
    from pokerlab.engine.actions import ActionType as T

    observation = _flop_observation(my_seat=seat, to_match=20)
    legal = [LegalAction(T.FOLD), LegalAction(T.CALL), LegalAction(T.RAISE, min_amount=40, max_amount=200),
             LegalAction(T.ALL_IN)]
    return observation, legal


def _fake_probabilities(observation, legal):
    from pokerlab.rl.action_space import (
        ACTION_DIM,
        ALL_IN_BIN,
        CHECK_CALL_BIN,
        FOLD_BIN,
        RAISE_MIN_BIN,
    )

    probabilities = [0.0] * ACTION_DIM
    probabilities[FOLD_BIN], probabilities[CHECK_CALL_BIN], probabilities[RAISE_MIN_BIN] = 0.125, 0.5, 0.25
    probabilities[ALL_IN_BIN] = 0.125
    return probabilities


def _panel_frame(app, advisors):
    frame = TableFrame(
        app, num_players=3, event_queue=queue.Queue(), human_player=None, history_path="unused.jsonl",
        seat_names={0: "A", 1: "B", 2: "C"}, step_gate=queue.Queue(), step_mode_state={"on": False},
        advisors=advisors,
    )
    return frame


def test_the_panel_shows_each_legal_actions_probability_for_the_bot_about_to_act(app):
    frame = _panel_frame(app, {1: _fake_probabilities})
    try:
        frame.step_mode_state.update(on=True, spy=True)
        action = _action_event(seat=0)
        action.payload["next_seat"] = 1
        action.payload["next_view"] = _decision_for_seat(1)
        frame._handle_event(action)

        rows = _probability_rows(frame)
        assert rows["fold"] == 12.5 and rows["check/call"] == 50.0
        assert rows["min-raise -> 40"] == 25.0 and rows["all-in"] == 12.5  # the raise shows its amount
        # the other legal raise sizes are listed too, at zero; an illegal action never is
        assert any("pot" in label and probability == 0.0 for label, probability in rows.items())
        assert sum(rows.values()) == 100.0
        assert "Tocca a B" in frame.probability_hint.get() and "Ah Kh" in frame.probability_hint.get()
    finally:
        frame.destroy()


def test_the_panel_waits_for_step_mode_and_the_spy_and_answers_the_toggles(app):
    frame = _panel_frame(app, {1: _fake_probabilities})
    try:
        action = _action_event(seat=0)
        action.payload["next_seat"], action.payload["next_view"] = 1, _decision_for_seat(1)
        frame._handle_event(action)
        assert _probability_rows(frame) == {} and "Attiva" in frame.probability_hint.get()

        frame.step_mode_var.set(True)
        frame._on_step_mode_toggled()
        assert _probability_rows(frame) == {}  # the spy is still off
        frame.spy_var.set(True)
        frame._on_spy_toggled()
        assert frame.step_mode_state["spy"] is True
        assert _probability_rows(frame)["check/call"] == 50.0  # the decision being held shows at once

        frame.spy_var.set(False)
        frame._on_spy_toggled()
        assert _probability_rows(frame) == {} and frame.step_mode_state["spy"] is False
    finally:
        frame.destroy()


def test_the_panel_says_so_when_the_one_to_act_is_not_a_bot_or_nobody_is(app):
    frame = _panel_frame(app, {1: _fake_probabilities})  # seat 2 has no advisor
    try:
        frame.step_mode_state.update(on=True, spy=True)
        frame._show_probabilities(_decision_for_seat(2))
        assert _probability_rows(frame) == {} and "non e' un bot" in frame.probability_hint.get()
        frame._show_probabilities(None)
        assert "Nessun bot" in frame.probability_hint.get()
    finally:
        frame.destroy()


def test_the_game_is_held_before_a_hand_or_street_opened_by_a_bot_only_when_both_modes_are_on():
    from pokerlab.gui.app import wait_for_first_bot

    def released(first_actor, state, human_seat=0):
        gate: queue.Queue = queue.Queue()
        done = threading.Event()

        def run():
            wait_for_first_bot({"first_actor": first_actor}, human_seat, state, gate)
            done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        finished_alone = done.wait(0.3)
        gate.put(None)  # the "Avanti" click
        thread.join(2)
        return finished_alone

    assert released(1, {"on": True, "spy": True}) is False  # a bot opens: it waits for Avanti
    assert released(0, {"on": True, "spy": True}) is True  # the human opens: nothing to look at
    assert released(None, {"on": True, "spy": True}) is True  # nobody is due
    assert released(1, {"on": True, "spy": False}) is True  # without the spy: the old behaviour
    assert released(1, {"on": False, "spy": True}) is True  # without step mode: the old behaviour


def test_real_models_real_hands_show_probabilities_that_agree_with_what_they_do(app):
    """Hands played by models (`RLAgentPlayer`, the real hooks and reporter) replayed into a frame
    in step mode with the spy on: every time a bot is about to act the panel lists its legal
    actions summing to 100%, and the action it then takes is one of them."""
    pytest.importorskip("torch")
    import random

    from support import tiny_model

    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.table import Table
    from pokerlab.gui.app import ActionReporter
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.policy import make_policy_fn

    events: queue.Queue = queue.Queue()
    model = tiny_model()
    players = [
        RLAgentPlayer(f"p{i}", f"B{i}", policy_fn=make_policy_fn(model), big_blind=4, starting_stack=400)
        for i in range(3)
    ]
    table = Table(
        GameConfig(num_players=3, starting_stack=400, small_blind=2, big_blind=4),
        players,
        rng=random.Random(2),
        on_hand_started=lambda info: events.put(GuiEvent("hand_started", info)),
        on_street_dealt=lambda info: events.put(GuiEvent("street_dealt", info)),
        on_action_applied=ActionReporter(events, queue.Queue(), {"on": False}, None, delay_seconds=0),
    )
    for _ in range(8):
        table.stacks = [400] * 3
        table.play_hand()

    def labels_of(action_type):
        """The panel rows an action of this type can have come from."""
        if action_type == ActionType.FOLD:
            return {"fold"}
        if action_type in (ActionType.CHECK, ActionType.CALL):
            return {"check/call"}
        if action_type == ActionType.ALL_IN:
            return {"all-in"}
        return None  # a bet or raise: any row that is not one of the three above

    frame = _panel_frame(app, {seat: player.action_probabilities for seat, player in enumerate(players)})
    try:
        frame.step_mode_state.update(on=True, spy=True)
        shown: list = []  # (seat the panel was about, its rows) for the player due
        while not events.empty():
            event = events.get()
            if event.kind == "action_taken":
                seat, rows = shown[-1]
                assert seat == event.payload["seat"], "the panel was not about the player who acted"
                labels = {label.split(" ->")[0] for label in rows}
                allowed = labels_of(event.payload["record"].action_type)
                if allowed is None:
                    assert labels - {"fold", "check/call", "all-in"}, "a raise that was not among the actions"
                else:
                    assert labels & allowed, f"{event.payload['record'].action_type} not among {labels}"
            frame._handle_event(event)
            view = {
                "action_taken": event.payload.get("next_view"),
                "hand_started": event.payload.get("first_actor_view"),
                "street_dealt": event.payload.get("first_actor_view"),
            }.get(event.kind)
            if view is not None:
                rows = _probability_rows(frame)
                assert sum(rows.values()) == pytest.approx(100.0, abs=0.6)
                shown.append((view[0].my_seat, rows))
        assert len(shown) > 20
    finally:
        frame.destroy()
