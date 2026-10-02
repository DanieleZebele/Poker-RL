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


def test_several_regions_come_out_of_one_grab_each_in_its_place(monkeypatch):
    np = pytest.importorskip("numpy")
    from pokerlab.vision import capture
    from pokerlab.vision.regions import Region

    grabs = []

    class FakeScreen:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grab(self, monitor):
            grabs.append(dict(monitor))
            # every pixel holds its own absolute (x, y), BGRA
            ys, xs = np.mgrid[monitor["top"]:monitor["top"] + monitor["height"],
                              monitor["left"]:monitor["left"] + monitor["width"]]
            return np.dstack([xs, ys, np.zeros_like(xs), np.zeros_like(xs)]).astype(np.int32)

    monkeypatch.setattr(capture, "_screen", FakeScreen)
    regions = [Region(100, 50, 20, 10), Region(-30, 200, 5, 5), Region(400, 60, 7, 3)]
    frames = capture.grab_regions(regions)
    assert len(grabs) == 1  # one grab for all of them
    for region, frame in zip(regions, frames, strict=True):
        assert frame.shape[:2] == (region.height, region.width)
        assert (frame[0, 0, 0], frame[0, 0, 1]) == (region.left, region.top)
        assert (frame[-1, -1, 0], frame[-1, -1, 1]) == (region.left + region.width - 1, region.top + region.height - 1)
    assert capture.grab_regions([]) == []
