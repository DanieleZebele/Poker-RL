"""The seating arithmetic of the spot table (no Tkinter)."""

from __future__ import annotations

import pytest

from pokerlab.gui.spot_table import CHAIRS, USER_CHAIR, TableLayout, chair_position


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
    center, radii = (400.0, 300.0), (300.0, 200.0)
    bottom = chair_position(0, center, radii)
    assert bottom == pytest.approx((400.0, 500.0))
    left = chair_position(2, center, radii)  # a quarter-ish turn: to the left of the bottom
    assert left[0] < center[0]
    top = [chair_position(c, center, radii) for c in (4, 5)]
    assert all(y < center[1] for _, y in top)
    right = chair_position(7, center, radii)
    assert right[0] > center[0]
    assert layout_free_chairs() == list(range(1, CHAIRS))


def layout_free_chairs():
    return TableLayout().free_chairs()
