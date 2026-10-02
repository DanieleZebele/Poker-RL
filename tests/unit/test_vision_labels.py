import json

import pytest

from pokerlab.vision.labels import label_error, label_path, load_label, save_label
from pokerlab.vision.regions import BOARD, HOLE_CARDS


@pytest.mark.parametrize("count, ok", [(0, True), (1, False), (2, True), (3, False)])
def test_hole_cards_are_two_or_none(count, ok):
    cards = ["Ah", "Kd", "Qs"][:count]
    assert (label_error(HOLE_CARDS, cards) is None) is ok


@pytest.mark.parametrize("count, ok", [(0, True), (1, False), (2, False), (3, True), (4, True), (5, True)])
def test_a_board_is_none_a_flop_a_turn_or_a_river(count, ok):
    cards = ["2c", "7d", "Th", "Js", "Ac"][:count]
    assert (label_error(BOARD, cards) is None) is ok


def test_bad_cards_and_repeats_are_refused():
    assert label_error(HOLE_CARDS, ["Ah", "Ah"])
    assert label_error(HOLE_CARDS, ["Ah", "1x"])
    assert label_error(HOLE_CARDS, "AhKd")
    assert label_error("altro", [])


def test_a_label_sits_next_to_its_crop_and_keeps_the_order(tmp_path):
    png = tmp_path / "board-20261001-101500.png"
    written = save_label(png, BOARD, ["Th", "2c", "7d", "As"])
    assert written == label_path(png) == tmp_path / "board-20261001-101500.json"
    label = load_label(png)
    assert label["cards"] == ["Th", "2c", "7d", "As"]  # left to right, not sorted
    assert label["zone"] == BOARD and label["image"] == png.name
    assert not list(tmp_path.glob(".*.partial"))


def test_no_cards_visible_is_a_real_label(tmp_path):
    png = tmp_path / "hole_cards-x.png"
    save_label(png, HOLE_CARDS, [])
    assert load_label(png)["cards"] == []


def test_an_invalid_label_is_never_written(tmp_path):
    png = tmp_path / "board-x.png"
    with pytest.raises(ValueError):
        save_label(png, BOARD, ["Ah", "Kd"])
    assert not label_path(png).exists()


def test_a_missing_or_broken_label_reads_as_none(tmp_path):
    png = tmp_path / "board-x.png"
    assert load_label(png) is None
    label_path(png).write_text("{not json", encoding="utf-8")
    assert load_label(png) is None
    label_path(png).write_text(json.dumps({"zone": BOARD}), encoding="utf-8")
    assert load_label(png) is None
