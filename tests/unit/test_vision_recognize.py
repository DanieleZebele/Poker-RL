"""The card reader, on synthetic crops drawn like the client's four-colour deck."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from pokerlab.vision.recognize import (
    CardRecognizer,
    examples_from_crop,
    has_card,
    rank_glyph_box,
    split_slots,
    white_mask,
)
from pokerlab.vision.regions import BOARD, HOLE_CARDS

FELT = (65, 174, 116)
SUIT_COLOURS = {"s": (49, 53, 52), "h": (52, 68, 216), "d": (204, 106, 51), "c": (41, 179, 46)}
TEXT = {"T": "10"}


def draw_card(canvas, x, rank, suit, width=90):
    """A card at `x`: suit-coloured body, white rank top-left, a white pip below."""
    h = canvas.shape[0]
    cv2.rectangle(canvas, (x, 4), (x + width, h - 4), SUIT_COLOURS[suit], -1)
    cv2.putText(canvas, TEXT.get(rank, rank), (x + 6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.circle(canvas, (x + width // 2 + 10, h - 30), 16, (255, 255, 255), -1)


def board(cards):
    canvas = np.full((138, 486, 3), FELT, np.uint8)
    for i, card in enumerate(cards):
        draw_card(canvas, round(i * 97.2) + 3, card[0], card[1])
    return canvas


def hole(cards):
    canvas = np.full((98, 170, 3), FELT, np.uint8)
    if cards:
        draw_card(canvas, 6, cards[0][0], cards[0][1], width=86)
        draw_card(canvas, 80, cards[1][0], cards[1][1], width=86)  # overlaps the left one
    return canvas


def recognizer_from(crops):
    examples = []
    for image, zone, cards in crops:
        examples += examples_from_crop(image, zone, cards, "synthetic")
    return CardRecognizer(examples)


def test_the_board_is_cut_in_five_and_the_hand_left_of_the_middle():
    assert [s.shape[1] for s in split_slots(board([]), BOARD)] == [97, 97, 98, 97, 97]
    left, right = split_slots(hole([]), HOLE_CARDS)
    assert left.shape[1] == 80 and right.shape[1] == 90  # not 85/85: see SLOT_EDGES


def test_the_felt_is_an_empty_slot_and_a_card_is_not():
    slots = split_slots(board(["Ah"]), BOARD)
    assert has_card(slots[0]) and not has_card(slots[1])


def test_a_ten_is_one_glyph_and_something_drawn_top_right_is_ignored():
    image = board(["Td"])
    slot = split_slots(image, BOARD)[0]
    cv2.circle(slot, (75, 14), 12, (255, 255, 255), -1)  # an overlay above the rank, on the right
    x, _y, w, _h = rank_glyph_box(white_mask(slot))
    assert x < 12 and w > 20 and x + w < 50  # both digits, and not the overlay


def test_rank_and_suit_are_read_separately_so_an_unseen_card_is_still_read():
    rec = recognizer_from([
        (board(["Ah", "Ks", "Qd", "Jc", "Th"]), BOARD, ["Ah", "Ks", "Qd", "Jc", "Th"]),
        (board(["9s", "8d", "7c", "Ts", "Ad"]), BOARD, ["9s", "8d", "7c", "Ts", "Ad"]),
    ])
    # Qh, Kc and 9d were never shown as whole cards.
    assert [m.card for m in rec.recognize(board(["Qh", "Kc", "9d"]), BOARD)] == ["Qh", "Kc", "9d"]


def test_the_hand_reads_both_cards_or_none():
    rec = recognizer_from([
        (board(["Ah", "Ks", "Qd", "Jc", "7h"]), BOARD, ["Ah", "Ks", "Qd", "Jc", "7h"]),
        (hole(["Ah", "Ks"]), HOLE_CARDS, ["Ah", "Ks"]),
    ])
    assert [m.card for m in rec.recognize(hole(["Qd", "Jc"]), HOLE_CARDS)] == ["Qd", "Jc"]
    assert rec.recognize(hole([]), HOLE_CARDS) == []


def test_a_board_stops_at_its_first_empty_slot():
    rec = recognizer_from([(board(["Ah", "Ks", "Qd"]), BOARD, ["Ah", "Ks", "Qd"])])
    image = board(["Ah", "Ks", "Qd"])
    draw_card(image, round(4 * 97.2) + 3, "K", "s")  # something in slot 5 with slot 4 empty
    assert [m.card for m in rec.recognize(image, BOARD)] == ["Ah", "Ks", "Qd"]


@pytest.mark.parametrize("bgr, suit", [
    ((49, 53, 52), "s"),     # black / dark grey, as measured
    ((52, 68, 216), "h"),    # red
    ((41, 179, 46), "c"),    # bright green
    ((204, 106, 51), "d"),   # blue
    ((65, 174, 116), None),  # the felt: green, but not the clubs' green
    ((40, 120, 60), None),   # a dull dark green, as the felt gets in shadow
])
def test_the_suit_is_the_colour_of_the_card(bgr, suit):
    from pokerlab.vision.recognize import classify_suit

    assert classify_suit(np.array(bgr)) == suit


def test_a_suit_needs_no_examples():
    rec = recognizer_from([(board(["As", "Ks", "Qs"]), BOARD, ["As", "Ks", "Qs"])])
    assert [m.card for m in rec.recognize(board(["Ah", "Kd", "Qc"]), BOARD)] == ["Ah", "Kd", "Qc"]
