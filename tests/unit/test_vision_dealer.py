"""The dealer-button rule, on crops drawn like the client's: gold disc on green felt."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from pokerlab.vision.dealer import PRESENT_FRACTION, find_dealer, gold_fraction, has_dealer

FELT = (65, 120, 40)  # BGR, H ~68 like the real felt
GOLD = (40, 200, 235)  # BGR of the button's gold


def zone(button=False, cover=0.0):
    image = np.full((37, 38, 3), FELT, np.uint8)
    if button:
        cv2.circle(image, (19, 18), 16, GOLD, -1)
        cv2.putText(image, "D", (11, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 80, 110), 2)
        if cover:  # something drawn over part of it (a card, chips)
            image[:, : int(38 * cover)] = (200, 200, 200)
    return image


def test_the_felt_has_no_gold_and_the_button_is_mostly_gold():
    assert gold_fraction(zone()) == 0.0
    assert gold_fraction(zone(button=True)) > 0.5
    assert not has_dealer(zone()) and has_dealer(zone(button=True))


def test_a_half_covered_button_is_still_seen():
    assert PRESENT_FRACTION < gold_fraction(zone(button=True, cover=0.5))


def test_the_seat_with_the_button_is_found_and_none_is_an_answer():
    reading = find_dealer({0: zone(), 1: zone(), 3: zone(button=True), 5: zone()})
    assert reading.seat == 3 and not reading.ambiguous
    assert find_dealer({0: zone(), 2: zone()}).seat is None


def test_two_buttons_pick_the_more_gold_one_and_say_so():
    reading = find_dealer({1: zone(button=True, cover=0.4), 4: zone(button=True)})
    assert reading.seat == 4 and reading.ambiguous
