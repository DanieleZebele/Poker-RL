import queue

import pytest

tk = pytest.importorskip("tkinter")

from pokerlab.cards.card import Card
from pokerlab.engine.state import PlayerStatus
from pokerlab.gui.app import PokerGuiApp, TableFrame, _bot_spec_label, _bot_spec_to_key_string
from pokerlab.players.base import SeatPublicInfo


def _card_item_counts(widgets: dict) -> list[int]:
    return [len(c.find_all()) for c in widgets["cards"]]


@pytest.fixture(scope="module")
def app():
    # One Tk interpreter for the whole file: creating/destroying tk.Tk()
    # repeatedly within a single process has proven unstable with this
    # project's Tcl/Tk install (see _fix_tcl_tk_library_paths), so each
    # test below gets a fresh TableFrame parented to this single long-lived
    # root instead of its own PokerGuiApp.
    application = PokerGuiApp()
    application.withdraw()
    yield application
    application.destroy()


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


def test_bot_spec_to_key_string_for_catalog_and_custom():
    assert _bot_spec_to_key_string({"key": "shark"}) == "shark"
    custom = {"key": "custom", "params": {"tightness": 0.2, "aggression": 0.8, "bluff_frequency": 0.3, "size_variance": 0.5}}
    key_string = _bot_spec_to_key_string(custom)
    assert key_string.startswith("custom:")
    assert "tightness=0.2" in key_string
    assert "aggression=0.8" in key_string


def test_bot_spec_label_shows_difficulty_for_catalog_bots():
    label = _bot_spec_label({"key": "shark"})
    assert "Shark" in label
    assert "[5]" in label


def test_bot_spec_label_shows_params_for_custom_bots():
    custom = {"key": "custom", "params": {"tightness": 0.2, "aggression": 0.8, "bluff_frequency": 0.3, "size_variance": 0.5}}
    label = _bot_spec_label(custom)
    assert "Custom" in label
    assert "0.20" in label and "0.80" in label


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
