"""Capturing the screen needs the `vision` extra and a display."""

from __future__ import annotations

import pytest

pytest.importorskip("mss")
pytest.importorskip("numpy")

from pokerlab.vision.capture import (
    frame_to_png_bytes,
    grab_monitor,
    grab_region,
    list_monitors,
    save_png,
)
from pokerlab.vision.regions import Region


@pytest.fixture(scope="module")
def screen():
    try:
        return list_monitors()
    except Exception as exc:  # noqa: BLE001 - no display on this machine
        pytest.skip(f"nessuno schermo da catturare: {exc}")


def test_a_region_comes_back_as_a_bgr_frame_of_its_own_size(screen):
    monitor = screen[0]
    frame = grab_region(Region(monitor["left"], monitor["top"], 64, 32))
    assert frame.shape == (32, 64, 3)
    assert frame.dtype.name == "uint8"


def test_a_whole_monitor_comes_with_its_geometry(screen):
    frame, geometry = grab_monitor(1)
    assert frame.shape[:2] == (geometry["height"], geometry["width"])
    with pytest.raises(ValueError):
        grab_monitor(len(screen) + 1)


def test_a_frame_is_written_as_a_png_without_an_imaging_library(screen, tmp_path):
    frame = grab_region(Region(screen[0]["left"], screen[0]["top"], 20, 10))
    data = frame_to_png_bytes(frame)
    assert data.startswith(b"\x89PNG")
    path = save_png(frame, tmp_path / "crops" / "x.png")
    assert path.read_bytes() == data
