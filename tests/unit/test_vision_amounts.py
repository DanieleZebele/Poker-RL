"""The bet/pot reader, on pills drawn like the client's: white digits, then BB."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from pokerlab.vision.amounts import (
    AmountReader,
    amount_value,
    examples_from_crop,
    has_amount,
    layout,
    white_mask,
)

FELT = (60, 110, 40)
PILL = (40, 55, 30)


def pill(text):
    """`text` BB in white on a dark pill, the comma drawn small at the baseline."""
    image = np.full((33, 110, 3), FELT, np.uint8)
    if not text:
        return image
    cv2.rectangle(image, (2, 3), (107, 30), PILL, -1)
    x = 8
    for char in text:
        if char == ",":
            cv2.rectangle(image, (x, 23), (x + 2, 27), (255, 255, 255), -1)
            x += 6
            continue
        cv2.putText(image, char, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        x += 14
    for _ in range(2):
        cv2.putText(image, "B", (x + 4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        x += 15
    return image


def reader_from(*texts, fallback=()):
    examples = []
    for text in texts:
        examples += examples_from_crop(pill(text), text, text)
    return AmountReader(examples, list(fallback))


def test_the_bb_is_dropped_and_the_comma_found():
    found = layout(white_mask(pill("18,5")))
    assert found.ok and len(found.digits) == 3 and found.comma_after == 1


def test_amounts_are_read_with_their_comma():
    reader = reader_from("10", "23", "45", "67", "89")
    for text in ("0,5", "18,5", "7", "236"):
        assert reader.read(pill(text)).text == text


def test_no_pill_is_no_amount_even_with_white_stripes_around():
    reader = reader_from("10")
    empty = pill("")
    cv2.rectangle(empty, (20, 0), (24, 4), (255, 255, 255), -1)  # a chip's stripes
    cv2.rectangle(empty, (60, 0), (67, 2), (255, 255, 255), -1)
    assert not has_amount(empty)
    assert reader.read(empty).text == ""


def test_a_digit_the_bets_do_not_match_well_is_read_from_the_fallback():
    fallback = examples_from_crop(pill("9"), "9", "card")
    reader = reader_from("10", "2345678", fallback=fallback)
    assert reader.read(pill("19")).text == "19"


def test_the_value_is_in_big_blinds():
    assert amount_value("18,5") == 18.5 and amount_value("7") == 7.0 and amount_value("") is None


def test_loops_tell_digits_apart_whatever_the_correlation():
    from pokerlab.vision.amounts import glyph_holes

    holes = {}
    for digit in "0123456789":
        example = examples_from_crop(pill(digit), digit, digit)[0]
        holes[digit] = glyph_holes(example.glyph)
    # (OpenCV's drawn "0" has a slash through it, so two loops: the client's has one)
    assert holes["8"] == 2 and holes["3"] == 0 and holes["6"] == 1 and holes["1"] == 0


def yellow_stack(text):
    """`text` BB in yellow, as the client writes the stacks."""
    image = np.full((33, 110, 3), FELT, np.uint8)
    if not text:
        return image  # nobody there: felt only
    x = 8
    for char in text + "BB":
        if char == ",":
            cv2.rectangle(image, (x, 23), (x + 2, 27), (40, 210, 240), -1)
            x += 6
            continue
        cv2.putText(image, char, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 210, 240), 2)
        x += 14 if char != "B" else 15
    return image


def test_a_yellow_stack_is_read_with_the_white_bets_as_examples():
    reader = reader_from("10", "23", "45", "67", "89")  # white bet digits only
    assert reader.read(yellow_stack("47,5"), "stack_6_2").text == "47,5"
    assert reader.read(yellow_stack(""), "stack_6_2").text == ""  # felt alone: no stack
    # read as a bet, the yellow lettering is not there at all
    assert reader.read(yellow_stack("47"), "bet_6_2").text == ""


def test_a_one_pixel_hole_is_not_a_loop():
    """A real "6" from the client closed its stroke round one stray pixel, the
    hole was counted as a second loop and the 6 was read as an 8."""
    from pokerlab.vision.amounts import glyph_holes

    six = examples_from_crop(pill("6"), "6", "6")[0].glyph.copy()
    side = round(len(six) ** 0.5)
    grid = six.reshape(side, side)
    rows, cols = np.nonzero(grid > 0)
    row, col = rows[len(rows) // 2], cols[len(cols) // 2]  # somewhere inside the stroke
    grid[row - 1:row + 2, col - 1:col + 2] = grid.max()
    grid[row, col] = min(grid.min(), 0)  # a pinhole in the ink
    assert glyph_holes(grid.ravel()) == 1


def test_an_amount_has_at_most_one_decimal():
    reader = reader_from("10", "23", "45", "67", "89")
    assert reader.read(yellow_stack("47,5"), "stack_6_2").text == "47,5"
    assert reader.read(yellow_stack("4,75"), "stack_6_2") is None  # a misread, not a stack
    assert reader.read(pill("4,75")) is None  # bets too
    assert reader.read(pill("4,5")).text == "4,5"


def test_the_cards_lend_only_digits_the_bets_have_no_example_of():
    """A pot "0,7" whose 0 matched the bets' zeros only weakly went to the card ranks,
    which have no 0, and came back "9,7": a digit the bets have is read against them."""
    fallback = examples_from_crop(pill("9"), "9", "card") + examples_from_crop(pill("6"), "6", "card")
    reader = reader_from("10", "23", "45", "78", "9", fallback=fallback)
    assert {e.digit for e in reader._fallback.examples} == {"6"}  # the bets have a 9
    assert reader.read(pill("16")).text == "16"  # and a 6 still comes from the cards
