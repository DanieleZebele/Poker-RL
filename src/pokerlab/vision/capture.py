"""Grabbing the screen.

`mss` and `numpy` come with the `vision` extra and are imported inside the
functions, the same way torch is everywhere else: nothing that does not capture a
screen pays for them, and a machine without the extra can still open the GUI.

A frame is a `numpy` array of shape `(height, width, 3)` in **BGR** order, which
is what OpenCV expects, so the recognition step can take it as it is.
"""

from __future__ import annotations

from pathlib import Path

from pokerlab.vision.regions import Region

INSTALL_HINT = 'serve l\'extra vision: pip install -e ".[vision]"'


class VisionUnavailable(RuntimeError):
    """The `vision` extra is not installed."""


def _mss():
    try:
        import mss
        import mss.tools
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise VisionUnavailable(INSTALL_HINT) from exc
    return mss


def _screen():
    """An `mss` session. Newer releases call the class `MSS` and deprecate `mss`;
    older ones only have `mss`, so either is accepted."""
    mss = _mss()
    return getattr(mss, "MSS", None)() if hasattr(mss, "MSS") else mss.mss()


def _to_frame(shot):
    import numpy as np

    # mss gives BGRA; the alpha channel is always opaque and of no use here.
    return np.ascontiguousarray(np.asarray(shot)[:, :, :3])


def list_monitors() -> list[dict[str, int]]:
    """The physical monitors, `mss` numbering: index 1 is the first one.

    (Index 0 of `mss`'s own list is the whole virtual desktop and is left out.)
    """
    with _screen() as screen:
        return [dict(monitor) for monitor in screen.monitors[1:]]


def grab_monitor(index: int = 1) -> tuple[object, dict[str, int]]:
    """A whole monitor as a frame, with the monitor's own geometry."""
    with _screen() as screen:
        if not 1 <= index < len(screen.monitors):
            raise ValueError(f"il monitor {index} non esiste: ce ne sono {len(screen.monitors) - 1}")
        monitor = dict(screen.monitors[index])
        return _to_frame(screen.grab(monitor)), monitor


def grab_region(region: Region) -> object:
    """Just the rectangle of the virtual desktop that was selected."""
    with _screen() as screen:
        return _to_frame(screen.grab(region.as_monitor()))


def frame_to_png_bytes(frame) -> bytes:
    """A frame as PNG data, for Tk's `PhotoImage` or for a file -- no Pillow."""
    mss = _mss()
    height, width = frame.shape[:2]
    return mss.tools.to_png(frame[:, :, ::-1].tobytes(), (width, height))


def save_png(frame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(frame_to_png_bytes(frame))
    return path
