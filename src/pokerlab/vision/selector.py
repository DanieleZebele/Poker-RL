"""Pick a rectangle of the screen by dragging the mouse over it.

The selector photographs a monitor, hides pokerlab's own window first so it is not
in the picture, and shows the photograph full screen and dimmed. You drag a
rectangle over the part you care about -- where your cards appear, say -- and
confirm with Return (Esc gives up). What comes back is a `Region` in the pixels of
the virtual desktop, ready for `capture.grab_region`.

Tk's coordinates and the screenshot's pixels are not guaranteed to be the same
thing: on a scaled Windows display they differ by the scale factor. So nothing
here assumes they match -- the rectangle is converted by comparing the size of the
canvas with the size of the picture (`scale_to_pixels`), and that arithmetic is
plain Python and tested on its own.
"""

from __future__ import annotations

import base64
import time
import tkinter as tk

from pokerlab.vision.regions import Region

# Smaller than this and it is a click, not a selection.
MIN_SIDE = 4
# The overlay covers the whole screen and takes the mouse, so it must never be
# able to stay forever: it gives itself up after this long.
GIVE_UP_SECONDS = 180


def normalize_drag(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    """`(left, top, width, height)` of a drag, whichever way it went."""
    return (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))


def scale_to_pixels(
    rect: tuple[float, float, float, float],
    canvas_size: tuple[float, float],
    image_size: tuple[float, float],
) -> tuple[int, int, int, int]:
    """A rectangle in canvas coordinates, in pixels of the screenshot.

    Equal sizes give the identity. On a display scaled to 150% the canvas can be
    two thirds of the picture, and the rectangle is stretched by the same factor.
    """
    sx = image_size[0] / canvas_size[0]
    sy = image_size[1] / canvas_size[1]
    left, top, width, height = rect
    return (round(left * sx), round(top * sy), round(width * sx), round(height * sy))


class RegionSelector(tk.Toplevel):
    """The full-screen overlay. `result` is the chosen `Region`, or None."""

    def __init__(
        self,
        master,
        png_data: bytes,
        image_size: tuple[int, int],
        origin: tuple[int, int] = (0, 0),
        label: str | None = None,
    ) -> None:
        """`label` names the zone being set at the top of the overlay: two
        sections with identical "Posto 5" buttons once had the dealer zone of
        seat 5 drawn round that player's whole box, with nothing on screen to
        say which one was being set."""
        super().__init__(master)
        self.result: Region | None = None
        self._image_size = image_size
        self._origin = origin
        self._start: tuple[float, float] | None = None
        self._rect: tuple[float, float, float, float] | None = None
        width, height = image_size

        self.overrideredirect(True)
        self.geometry(f"{width}x{height}+{origin[0]}+{origin[1]}")
        try:
            self.attributes("-topmost", True)
        except tk.TclError:  # pragma: no cover - not every window manager has it
            pass
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self._picture = tk.PhotoImage(data=base64.b64encode(png_data).decode("ascii"))
        self.canvas.create_image(0, 0, image=self._picture, anchor="nw")
        self.canvas.create_rectangle(0, 0, width, height, fill="black", stipple="gray50", outline="")
        self.canvas.create_text(
            width / 2, 28, fill="white", font=("TkDefaultFont", 14, "bold"),
            text=(f"{label}:  " if label else "")
            + "Trascina un rettangolo sulla zona  -  Invio conferma, Esc annulla",
        )
        self._box = self.canvas.create_rectangle(0, 0, 0, 0, outline="#ffd43b", width=2)
        # Buttons as well as keys: an override-redirect window gets no keyboard
        # focus from some Linux window managers, so Return and Esc alone can leave
        # a full-screen overlay nobody can close.
        bar = tk.Frame(self.canvas)
        self.confirm_button = tk.Button(bar, text="Conferma (Invio)", command=self.confirm)
        self.cancel_button = tk.Button(bar, text="Annulla (Esc)", command=self.cancel)
        self.confirm_button.pack(side="left", padx=4)
        self.cancel_button.pack(side="left", padx=4)
        self.canvas.create_window(width / 2, 70, window=bar)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.press(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.drag(e.x, e.y))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.release(e.x, e.y))
        self.canvas.bind("<Double-Button-1>", lambda _e: self.confirm())
        self.canvas.bind("<ButtonPress-3>", lambda _e: self.cancel())
        for widget in (self, self.canvas):
            widget.bind("<Return>", lambda _e: self.confirm())
            widget.bind("<Escape>", lambda _e: self.cancel())
        self.after(GIVE_UP_SECONDS * 1000, self.cancel)

    # -- the mouse ------------------------------------------------------------

    def press(self, x: float, y: float) -> None:
        self._start = (x, y)
        self._rect = None
        self.canvas.coords(self._box, x, y, x, y)

    def drag(self, x: float, y: float) -> None:
        if self._start is None:
            return
        # Kept inside the picture: a rectangle off the edge is not on the screen.
        width, height = self._image_size
        x = min(max(x, 0), width)
        y = min(max(y, 0), height)
        left, top, w, h = normalize_drag(*self._start, x, y)
        self._rect = (left, top, w, h)
        self.canvas.coords(self._box, left, top, left + w, top + h)

    def release(self, x: float, y: float) -> None:
        self.drag(x, y)

    # -- the outcome ----------------------------------------------------------

    def confirm(self) -> None:
        """Keep the selection, if it is big enough to be one."""
        if self._rect is None or min(self._rect[2], self._rect[3]) < MIN_SIDE:
            return  # nothing chosen yet: stay open rather than return an empty zone
        canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if min(canvas_size) <= 1:  # not mapped yet (a headless test): the requested size
            canvas_size = self._image_size
        left, top, width, height = scale_to_pixels(self._rect, canvas_size, self._image_size)
        if width > 0 and height > 0:
            self.result = Region(self._origin[0] + left, self._origin[1] + top, width, height)
        if self.winfo_exists():
            self.destroy()

    def cancel(self) -> None:
        self.result = None
        if self.winfo_exists():
            self.destroy()

    def present(self) -> None:
        """Take the keyboard and the mouse, once the window is really on screen.

        `focus_force` before the window is mapped does nothing, which is how the
        overlay used to ignore Return and Esc.
        """
        # Waited for with a limit, never with `wait_visibility`, which blocks
        # forever if the window is not mapped (and then nothing can close it).
        deadline = time.time() + 1.0
        self.update()
        while not self.winfo_viewable() and time.time() < deadline:
            self.update()
            time.sleep(0.02)
        try:
            self.grab_set()
        except tk.TclError:  # pragma: no cover - not mapped, or no display to grab on
            pass
        self.focus_force()
        self.canvas.focus_set()


def select_region(parent, monitor: int = 1, label: str | None = None) -> Region | None:
    """Photograph `monitor`, let the user drag a rectangle over it, return it.

    The calling window is hidden first and brought back whatever happens, so the
    picture is of the client and not of pokerlab, and a failed capture cannot
    leave the application invisible.
    """
    from pokerlab.vision.capture import frame_to_png_bytes, grab_monitor

    root = parent.winfo_toplevel()
    root.withdraw()
    try:
        root.update()
        time.sleep(0.25)  # let the window really disappear before the shot
        frame, geometry = grab_monitor(monitor)
        height, width = frame.shape[:2]
        selector = RegionSelector(
            root, frame_to_png_bytes(frame), (width, height), (geometry["left"], geometry["top"]),
            label=label,
        )
        selector.present()
        root.wait_window(selector)
        return selector.result
    finally:
        root.deiconify()
