"""The seating arithmetic of the spot table (no Tkinter)."""

from __future__ import annotations

import pytest

from pokerlab.gui.spot_table import CHAIRS, USER_CHAIR, TableLayout, chair_slot


def test_the_user_starts_alone_with_the_button():
    layout = TableLayout()
    assert layout.chairs == {USER_CHAIR} and layout.dealer == USER_CHAIR
    assert layout.order() == [USER_CHAIR]


def test_play_order_is_clockwise_from_the_dealer():
    layout = TableLayout()
    for chair in (2, 5, 7):
        layout.add(chair)
    assert layout.order() == [0, 2, 5, 7]
    layout.set_dealer(5)
    assert layout.order() == [5, 7, 0, 2]
    assert layout.seat_of(5) == 0 and layout.seat_of(0) == 2
    assert layout.chair_of(3) == 2


def test_a_player_can_be_inserted_between_two_others():
    layout = TableLayout()
    layout.add(2)
    layout.add(6)
    assert layout.add(4)
    assert layout.order() == [0, 2, 4, 6]


def test_occupied_or_missing_chairs_are_refused():
    layout = TableLayout()
    assert not layout.add(USER_CHAIR) and not layout.add(CHAIRS) and not layout.add(-1)
    assert not layout.remove(USER_CHAIR) and not layout.remove(4)
    assert not layout.set_dealer(4)


def test_removing_the_dealer_gives_the_button_back_to_the_user():
    layout = TableLayout()
    layout.add(3)
    layout.set_dealer(3)
    layout.remove(3)
    assert layout.dealer == USER_CHAIR


def test_chairs_run_clockwise_from_the_bottom_as_seen_on_screen():
    size = (800.0, 600.0)
    assert chair_slot(0, size) == (400.0, 600.0, "s")  # you, bottom middle
    assert chair_slot(1, size) == (0.0, 600.0, "sw")  # then the bottom-left corner
    assert chair_slot(2, size) == (0.0, 300.0, "w")  # up the left side
    assert chair_slot(4, size) == (400.0, 0.0, "n")  # across the top
    assert chair_slot(6, size) == (800.0, 300.0, "e")  # and down the right
    assert chair_slot(3, size, margin=10)[:2] == pytest.approx((10.0, 10.0))
    # every box pinned by the side facing its edge, so it grows inwards
    assert {chair_slot(c, size)[2] for c in range(CHAIRS)} == {"s", "sw", "w", "nw", "n", "ne", "e", "se"}
    assert layout_free_chairs() == list(range(1, CHAIRS))


def layout_free_chairs():
    return TableLayout().free_chairs()
