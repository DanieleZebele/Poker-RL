"""The saved screen regions need no dependency at all."""

from __future__ import annotations

import json

import pytest

from pokerlab.vision.regions import (
    BOARD,
    HOLE_CARDS,
    Region,
    RegionConfig,
    load_regions,
    save_regions,
)


def test_a_region_must_have_a_positive_size():
    with pytest.raises(ValueError):
        Region(0, 0, 0, 10)
    with pytest.raises(ValueError):
        Region(0, 0, 10, -1)
    assert Region(-1920, 100, 300, 80).as_monitor() == {
        "left": -1920, "top": 100, "width": 300, "height": 80,
    }


def test_regions_round_trip_through_the_file(tmp_path):
    path = tmp_path / "vision" / "regions.json"
    config = RegionConfig()
    config.set(HOLE_CARDS, Region(10, 20, 120, 60))
    config.set(BOARD, Region(-300, 40, 500, 70))
    save_regions(config, path)

    loaded = load_regions(path)
    assert loaded.get(HOLE_CARDS) == Region(10, 20, 120, 60)
    assert loaded.get(BOARD) == Region(-300, 40, 500, 70)
    assert not list(path.parent.glob(".*.partial")), "no half-written file is left behind"


def test_a_missing_or_corrupt_file_means_no_regions(tmp_path):
    assert load_regions(tmp_path / "absent.json").regions == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_regions(broken).regions == {}
    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text(json.dumps({"regions": [1, 2]}), encoding="utf-8")
    assert load_regions(wrong_shape).regions == {}


def test_one_bad_entry_does_not_take_the_others_with_it(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(
        json.dumps({"regions": {
            HOLE_CARDS: {"left": 1, "top": 2, "width": 30, "height": 40},
            BOARD: {"left": 1, "top": 2, "width": 0, "height": 40},
            "other": {"nope": 1},
        }}),
        encoding="utf-8",
    )
    loaded = load_regions(path)
    assert set(loaded.regions) == {HOLE_CARDS}


def test_a_zone_can_be_cleared():
    config = RegionConfig()
    config.set(BOARD, Region(0, 0, 5, 5))
    config.clear(BOARD)
    config.clear(BOARD)  # clearing what is not there is not an error
    assert config.get(BOARD) is None
