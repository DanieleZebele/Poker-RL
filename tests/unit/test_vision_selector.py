"""The region selector: the coordinate arithmetic, and the overlay without a mouse."""

from __future__ import annotations

import gc
import struct
import zlib

import pytest

from pokerlab.vision.regions import Region
from pokerlab.vision.selector import MIN_SIDE, normalize_drag, scale_to_pixels

tk = pytest.importorskip("tkinter")


def test_a_drag_in_any_direction_is_the_same_rectangle():
    assert normalize_drag(10, 20, 50, 80) == (10, 20, 40, 60)
    assert normalize_drag(50, 80, 10, 20) == (10, 20, 40, 60)
    assert normalize_drag(50, 20, 10, 80) == (10, 20, 40, 60)


def test_equal_sizes_leave_the_rectangle_alone():
    assert scale_to_pixels((10, 20, 40, 60), (800, 600), (800, 600)) == (10, 20, 40, 60)


def test_a_scaled_display_stretches_the_rectangle_by_the_same_factor():
    # Canvas two thirds of the picture, as on a display set to 150%.
    assert scale_to_pixels((100, 40, 200, 80), (800, 600), (1200, 900)) == (150, 60, 300, 120)
    # Different factors per axis are respected.
    assert scale_to_pixels((10, 10, 10, 10), (100, 100), (200, 400)) == (20, 40, 20, 40)


def png(width: int, height: int) -> bytes:
    """A flat grey PNG, built by hand so the test needs no imaging library."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes([128, 128, 128]) * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


@pytest.fixture(scope="module")
def root():
    window = tk.Tk()
    window.withdraw()
    yield window
    gc.collect()
    window.destroy()
    gc.collect()


def make(root, origin=(0, 0)):
    from pokerlab.vision.selector import RegionSelector

    return RegionSelector(root, png(200, 100), (200, 100), origin)


def test_dragging_and_confirming_returns_the_region_in_desktop_pixels(root):
    selector = make(root, origin=(1920, 40))  # a second monitor to the right
    selector.press(150, 70)
    selector.drag(110, 30)  # dragged up and to the left
    selector.release(110, 30)
    selector.confirm()
    assert selector.result == Region(1920 + 110, 40 + 30, 40, 40)


def test_the_rectangle_cannot_leave_the_picture(root):
    selector = make(root)
    selector.press(150, 70)
    selector.drag(500, 500)
    selector.confirm()
    assert selector.result == Region(150, 70, 50, 30)


def test_a_click_without_a_drag_is_not_a_selection(root):
    selector = make(root)
    selector.press(50, 50)
    selector.release(50 + MIN_SIDE - 1, 50 + MIN_SIDE - 1)
    selector.confirm()
    assert selector.result is None
    assert selector.winfo_exists(), "stays open until something real is chosen"
    selector.cancel()
    assert selector.result is None


def test_escape_gives_up(root):
    selector = make(root)
    selector.press(10, 10)
    selector.drag(100, 80)
    selector.cancel()
    assert selector.result is None


def test_the_overlay_can_be_closed_with_its_own_buttons(root):
    """Keys alone are not enough: some window managers give an override-redirect
    window no keyboard focus, and the overlay then could not be closed."""
    selector = make(root)
    selector.press(10, 10)
    selector.drag(100, 80)
    selector.confirm_button.invoke()
    assert selector.result == Region(10, 10, 90, 70)
    assert not selector.winfo_exists()

    other = make(root)
    other.cancel_button.invoke()
    assert other.result is None and not other.winfo_exists()


def test_a_right_click_and_a_double_click_are_wired(root):
    selector = make(root)
    try:
        assert selector.canvas.bind("<ButtonPress-3>")
        assert selector.canvas.bind("<Double-Button-1>")
        assert selector.canvas.bind("<Return>") and selector.canvas.bind("<Escape>")
    finally:
        selector.cancel()


def test_the_overlay_takes_focus_only_when_it_is_on_screen(root):
    selector = make(root)
    selector.present()  # must not raise or hang, mapped or not
    selector.cancel()
