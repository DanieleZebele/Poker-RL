"""The "ask the models" screen: a table you fill in with buttons.

An oval table with eight chairs round it, yours at the bottom. Press "+" on a chair
to seat an opponent, give them a name and a stack, and mark who has the button;
the order of play follows from where the button is. The board sits in the middle
and your two cards under your chair, and every card is chosen by pressing buttons
-- rank, then suit -- never by typing. When it is someone's turn their actions
appear on their own chair; when it is yours the top models answer by themselves.

All the poker lives in `gui/spot.py` and the seating arithmetic in
`gui/spot_table.py`; this is only the form around them. The state of the screen
is the layout, your cards, the board and the list of actions, and every edit
rebuilds a `Spot` from them and replays it, so what is drawn is always what the
engine reached rather than something this frame tracked.
"""

from __future__ import annotations

import itertools
import math
import random
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from types import SimpleNamespace

from pokerlab.cards.card import Card
from pokerlab.cli.play import discover_global_top_models
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.engine.stats import StatsTracker
from pokerlab.gui.spot import (
    DEFAULT_ADVISORS,
    Spot,
    SpotState,
    advise,
    bb_number,
    describe_action,
    format_bb,
    format_card,
    load_advisors,
    parse_bb,
    parse_card,
    position_names,
    replay,
)
from pokerlab.gui.spot_table import (
    CHAIRS,
    USER_CHAIR,
    TableLayout,
    chair_for_client_seat,
    chair_slot,
)
from pokerlab.vision.regions import DEALER_TABLE_SIZES

# One wheel notch over a bet or raise amount moves it by this many big blinds.
WHEEL_STEP_BIG_BLINDS = 1
# The engine's blinds in chips: one chip is a hundredth of a big blind, the blinds
# the models train at (`rl/table_mix.py`). Fixed, since every amount on screen is in
# big blinds; a chip that small is what lets an ante of 0,1 BB -- or 0,12 -- exist.
ENGINE_SMALL_BLIND = 50
ENGINE_BIG_BLIND = 100
DEFAULT_STACK_BB = "100"
# How often the hand and the board are read off the poker client.
# Half a second: a reading costs ~60 ms on
# the Tk thread (one grab of every zone, see `capture.grab_regions`), so this
# keeps the screen ~88% free, and catches a raise before it is swept into the pot.
SCREEN_POLL_MS = 500
# How many readings in a row must show a seat empty before its player's statistics are
# forgotten (one second at `SCREEN_POLL_MS`).
EMPTY_READINGS_TO_FORGET = 2
# How a seat state read off the screen is shown (`vision.labels.SEAT_STATES`).
SEAT_STATE_LABELS = {"in_gioco": "in gioco", "fuori": "fold", "sit_out": "sit-out", "libero": "liberi",
                     "reazione": "reazioni"}
# The states that put a seated player out of the hand: tagged on their chair, and taken
# by `action_sync` as a fold when their turn comes. An empty seat ("libero") among them:
# a player who vanishes mid-hand -- left the table, disconnected -- has folded as far as
# the hand goes, and without it the rebuild waited on them for ever.
# A new hand shows up twice: the button moves and your new cards come, often a reading
# or two apart. The second sign within this many seconds of the first, of the other
# kind, is the same hand starting, not another one (`_hand_boundary`).
SAME_DEAL_SECONDS = 5.0
# A scripted FOLD or ALL_IN the screen keeps contradicting for this many readings in a
# row (the player still holds cards; still has chips behind) is taken out, with what
# came after it, and the hand rebuilt from there (`_heal_script`). Not one: a stack is
# updated a moment late, and a misread frame passes.
HEAL_READINGS = 3
# Every reading that changed something, written here for whoever has to find out why the
# rebuild went wrong (only while the screen is read; `_log_reading`).
SCREEN_LOG = Path("hand_histories") / "spot_screen.log"
SCREEN_LOG_MAX_BYTES = 5_000_000
SEAT_MARKS = {"fuori": "fold", "sit_out": "sit-out", "libero": "uscito"}
OUT_OF_HAND = "#9e9e9e"
# A chair's box, and the same box dark once its player is out of the hand.
BOX_BG = "#f4f1ea"
FOLDED_BG = "#3c3c3c"
FOLDED_FG = "#e6e6e6"

# The table at its smallest; `table_geometry` makes it fill the screen. The chairs
# sit along its edges (`spot_table.chair_slot`), three a side, so it has to hold
# three boxes across and three down.
CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 620
# Space kept between the chairs and the edge of the canvas.
CANVAS_MARGIN = 8
# The room the boxes take from each edge, which the felt in the middle stays out of.
# A box does not scale with the table: an opponent with the statistics and its
# actions is ~300 px wide and up to ~245 tall, yours with the cards ~314 tall.
BOX_SIDE = 310
BOX_TOP = 250
BOX_BOTTOM = 330
# The felt never smaller than this, whatever the boxes leave: the board must fit.
FELT_MIN_RADII = (150, 90)
# What the screen keeps for everything but the table: the side panel (advice, log)
# on the right, and above and below the window's title bar, the top row of buttons
# and the taskbar (measured: 118 px on a 1536x864 Windows screen, maximised).
SIDE_PANEL_WIDTH = 360
RESERVED_HEIGHT = 125
# The statistics a chair shows, in the order of `engine/stats.py::STATS`, three a line.
STAT_LABELS = {
    "vpip": "VPIP", "pfr": "PFR", "three_bet": "3bet",
    "fold_to_three_bet": "F3bet", "steal": "Steal", "aggression": "Aggr",
    "cbet": "Cbet", "fold_to_cbet": "FCbet", "wtsd": "WTSD",
}
STATS_PER_LINE = 3
# Lines of the models' advice panel: the action drawn, every option of the first model
# (two or three lines wrapped), then a line a model; the list of actions above it takes
# the rest of the height.
ADVICE_LINES = 12
# The chairs in the middle of the top and bottom edges have room sideways and none
# up and down (the felt is right there): their cards and actions go to the right of
# the player's details instead of under them, or the box grew over the felt.
WIDE_CHAIRS = (USER_CHAIR, 4)


@dataclass(frozen=True)
class TableGeometry:
    canvas: tuple[int, int]
    center: tuple[float, float]
    table_radii: tuple[float, float]


def table_geometry(screen_width: int, screen_height: int) -> TableGeometry:
    """The canvas filling the screen next to the side panel (never below the base
    size: a screen smaller than that scrolls), and the felt in the space the boxes
    along the edges leave in the middle (`BOX_SIDE`/`BOX_TOP`/`BOX_BOTTOM`): the
    boxes keep their size, so a bigger screen gives the room to the table between
    them. Filled, not scaled: a canvas taller than the screen cut the bottom row of
    boxes off on a 1536x864 screen."""
    width = max(CANVAS_WIDTH, screen_width - SIDE_PANEL_WIDTH)
    height = max(CANVAS_HEIGHT, screen_height - RESERVED_HEIGHT)
    top, bottom = BOX_TOP + CANVAS_MARGIN, height - BOX_BOTTOM - CANVAS_MARGIN
    radii = (
        max(FELT_MIN_RADII[0], width / 2 - BOX_SIDE - 2 * CANVAS_MARGIN),
        max(FELT_MIN_RADII[1], (bottom - top) / 2 - CANVAS_MARGIN),
    )
    return TableGeometry(canvas=(width, height), center=(width / 2, (top + bottom) / 2), table_radii=radii)


FELT = "#1f6b3a"
RAIL = "#5b3a1a"
HIGHLIGHT = "#ffd43b"
DEALER_BUTTON = "#f2c94c"
# Each seat's bet on the felt: a dark chip with the amount, on the rail in the
# direction of its chair.
BET_CHIP_BG = "#1b1b1b"
BET_CHIP_FG = "#ffd43b"
# Where along the way from the centre to the rail the bets sit (1 = on the rail).
BET_RADIUS = 0.97

RANKS = "23456789TJQKA"
SUITS = "shdc"
_SUIT_GLYPHS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
_RED = "#c0392b"


def card_text(card: Card | None) -> str:
    """`Ah` -> `A♥`, `Td` -> `10♦`; an empty slot reads `?`."""
    if card is None:
        return "?"
    rank, suit = format_card(card)
    return f"{'10' if rank == 'T' else rank}{_SUIT_GLYPHS[suit]}"


def card_color(card: Card | None) -> str:
    if card is not None and format_card(card)[1] in "hd":
        return _RED
    return "black"


def _monitor_bounds(x: int, y: int, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """`(left, top, right, bottom)` of the work area of the monitor holding the
    point, so a popup is kept on the screen the mouse is on. Windows only (the
    process is DPI-aware, see `app._make_windows_dpi_aware`, so these are the
    same pixels Tk uses); anywhere else, or if the call fails, `fallback`."""
    import sys

    if sys.platform != "win32":
        return fallback
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        monitor = ctypes.windll.user32.MonitorFromPoint(wintypes.POINT(x, y), 2)  # nearest
        info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
        if not ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return fallback
        work = info.rcWork
        return (work.left, work.top, work.right, work.bottom)
    except (AttributeError, OSError):
        return fallback


_raise_tags = itertools.count()


def _paint_box(box: tk.Widget, *, folded: bool) -> None:
    """A chair's box and everything in it but its buttons and fields, dark when its
    player is out of the hand (FOLDED_BG, light text), in the usual colours otherwise."""
    bg = FOLDED_BG if folded else BOX_BG
    widgets = [box]
    while widgets:
        widget = widgets.pop()
        widgets.extend(widget.winfo_children())
        if widget.winfo_class() == "Frame":
            widget.configure(bg=bg)
        elif widget.winfo_class() == "Label":
            widget.configure(bg=bg, fg=FOLDED_FG if folded else "black")


def _raise_tag(holder: tk.Widget) -> str:
    """A bindtag unique to this holder for the life of the process. Not
    `winfo_id()`: Windows reuses window handles, so a new chair could inherit a
    destroyed one's class binding -- whose `lift()` then fails and aborts the
    binding script before the new one runs."""
    if not hasattr(holder, "_raise_tag"):
        holder._raise_tag = f"RaiseChair{next(_raise_tags)}"
    return holder._raise_tag


def _make_raisable(holder: tk.Widget) -> None:
    """Make a press anywhere on `holder` -- its own background or any widget in
    it, a button included -- bring it in front of the chairs it overlaps.

    Chairs are canvas windows, and overlapping embedded windows stack by widget
    order, not by canvas item order, so `lift()` on the holder is what raises it.
    Done with an extra bindtag, placed first, on every descendant: the press
    raises the box and then carries on to the widget's own bindings, so the
    button that was clicked still does its job. Re-run after the action buttons
    are rebuilt, which `refresh` does at every step; a widget already tagged is
    left alone."""
    tag = _raise_tag(holder)
    stack = [holder]
    while stack:
        widget = stack.pop()
        tags = widget.bindtags()
        if tag not in tags:
            widget.bindtags((tag, *tags))
        stack.extend(widget.winfo_children())


def place_near_pointer(window: tk.Toplevel) -> tuple[int, int]:
    """Open `window` where the mouse is -- centred on it horizontally, just
    below it -- instead of wherever the window manager drops it (the top-left
    corner of the first screen, on Windows). Kept inside that monitor."""
    window.update_idletasks()
    x, y = window.winfo_pointerxy()
    width, height = window.winfo_reqwidth(), window.winfo_reqheight()
    left, top, right, bottom = _monitor_bounds(
        x, y, (0, 0, window.winfo_screenwidth(), window.winfo_screenheight())
    )
    px = min(max(x - width // 2, left), max(left, right - width))
    py = min(max(y - 20, top), max(top, bottom - height))
    window.geometry(f"+{px}+{py}")
    return px, py


class CardPicker(tk.Toplevel):
    """A popup with a button per rank and then a button per suit.

    Two steps on purpose: thirteen plus four buttons are quick to hit and cannot
    be mistyped, where fifty-two would not fit on screen at a readable size.
    Cards already used elsewhere are disabled in the suit step, so the same card
    can never be chosen twice.
    """

    def __init__(self, master, *, taken: set[Card], on_pick, title: str, can_clear: bool) -> None:
        super().__init__(master)
        self.title(title)
        self.transient(master.winfo_toplevel())
        self._taken = taken
        self._on_pick = on_pick
        self._can_clear = can_clear
        self.rank: str | None = None
        self.body = ttk.Frame(self, padding=10)
        self.body.pack()
        self._show_ranks()
        place_near_pointer(self)
        try:
            self.grab_set()
        except tk.TclError:  # no display to grab on (headless tests)
            pass

    def _clear_body(self) -> None:
        for widget in self.body.winfo_children():
            widget.destroy()

    def _show_ranks(self) -> None:
        self._clear_body()
        ttk.Label(self.body, text="Valore").grid(row=0, column=0, columnspan=7, pady=(0, 4))
        for index, symbol in enumerate(RANKS):
            ttk.Button(
                self.body, text="10" if symbol == "T" else symbol, width=4,
                command=lambda s=symbol: self.choose_rank(s),
            ).grid(row=1 + index // 7, column=index % 7, padx=2, pady=2)
        row = ttk.Frame(self.body)
        row.grid(row=4, column=0, columnspan=7, pady=(8, 0))
        if self._can_clear:
            ttk.Button(row, text="Togli la carta", command=self.clear).pack(side="left", padx=4)
        ttk.Button(row, text="Annulla", command=self.destroy).pack(side="left", padx=4)

    def _show_suits(self) -> None:
        self._clear_body()
        ttk.Label(self.body, text="Seme").grid(row=0, column=0, columnspan=4, pady=(0, 4))
        for index, suit in enumerate(SUITS):
            used = parse_card(f"{self.rank}{suit}") in self._taken
            tk.Button(
                self.body, text=_SUIT_GLYPHS[suit], width=4, font=("TkDefaultFont", 16),
                fg=_RED if suit in "hd" else "black",
                state="disabled" if used else "normal",
                command=lambda s=suit: self.choose_suit(s),
            ).grid(row=1, column=index, padx=3, pady=2)
        row = ttk.Frame(self.body)
        row.grid(row=2, column=0, columnspan=4, pady=(8, 0))
        ttk.Button(row, text="Indietro", command=self._show_ranks).pack(side="left", padx=4)
        ttk.Button(row, text="Annulla", command=self.destroy).pack(side="left", padx=4)

    def choose_rank(self, symbol: str) -> None:
        self.rank = symbol
        self._show_suits()

    def choose_suit(self, suit: str) -> None:
        if self.rank is None:
            return
        card = parse_card(f"{self.rank}{suit}")
        if card in self._taken:
            return
        self._on_pick(card)
        self.destroy()

    def clear(self) -> None:
        self._on_pick(None)
        self.destroy()


class SpotFrame(ttk.Frame):
    def __init__(self, master, *, read_screen: bool = False, table_size: int | None = None) -> None:
        """`read_screen` starts reading the hand and the board off the poker
        client every `SCREEN_POLL_MS` (the app turns it on; tests leave it off,
        or they would photograph whatever screen they run on).

        `table_size` is the client's table, 6 or 8 seats: which set of seat
        zones is read (`player_6_*` or `player_8_*`, ...) and how client seats
        map onto the chairs. Without one, the last choice made in this app."""
        # Checked before the widget exists: a half-built frame left in the app
        # would fail when the app later destroys it.
        if table_size is None:
            table_size = getattr(master, "spot_table_size", DEALER_TABLE_SIZES[0])
        if table_size not in DEALER_TABLE_SIZES:
            raise ValueError(f"tavolo da {table_size} non gestito: {DEALER_TABLE_SIZES}")
        super().__init__(master, padding=8)
        self.app = master
        self.table_geom = table_geometry(self.winfo_screenwidth(), self.winfo_screenheight())
        self.table_size = table_size
        self.table_size_var = tk.IntVar(value=table_size)
        self.layout = TableLayout()
        self.script: list[Action] = []
        self.state: SpotState | None = None
        self.models: list | None = None  # loaded lazily, once
        self._model_ranks: dict[str, int] = {}  # label -> place in the global ranking
        self.hole: list[Card | None] = [None, None]
        self.board: list[Card] = []

        # Every amount on this screen is in big blinds. Underneath, the engine
        # counts chips at blinds of 50/100 -- one chip is a hundredth of a big
        # blind, and these are the blinds the models train at -- and the conversion
        # happens only at the edges (`format_bb` to show, `parse_bb` to read input).
        # Tournaments: what everyone pays before the blinds, in BB. Read off the
        # screen at every preflop reading (`_read_ante`), or typed.
        self.ante_var = tk.StringVar(value="0")
        # On: the models are always shown every opponent as never seen before (no
        # statistics), as if each hand were against new players. The statistics are
        # still kept and shown; only what the policy reads changes. Kept on the app, so
        # the screen reopens with it.
        self.fresh_opponents_var = tk.BooleanVar(value=getattr(master, "spot_fresh_opponents", False))
        self.sb_var = tk.StringVar(value=str(ENGINE_SMALL_BLIND))
        self.bb_var = tk.StringVar(value=str(ENGINE_BIG_BLIND))
        self.name_vars = [
            tk.StringVar(value="Tu" if chair == USER_CHAIR else f"Avv. {chair}")
            for chair in range(CHAIRS)
        ]
        self.stack_vars = [tk.StringVar(value=DEFAULT_STACK_BB) for _ in range(CHAIRS)]
        self.status_var = tk.StringVar()
        self.chair_ui: dict[int, SimpleNamespace] = {}
        self.board_buttons: list[tk.Button] = []
        self.hole_buttons: list[tk.Button] = []
        self.screen_var = tk.StringVar(value="Schermo: lettura spenta")
        self._reader = None
        self._last_reading = None
        self._last_hand: list[str] | None = None
        self._last_dealer: int | None = None  # client seat of the button last seen
        self._seat_states: dict[int, str] = {}  # chair -> state, from the last reading
        self._seating_read = False  # whether the seats have been taken off the screen yet
        self._sync_note = ""  # what the last action rebuild did, for the status line
        self._last_view = None  # the last reading as `action_sync` saw it (engine seats, chips)
        # Chairs whose starting stack this hand was read off the screen (`_apply_stacks`):
        # only their stacks on screen are evidence of chips put in.
        self._stacks_from_screen: set[int] = set()
        # The script's length when your countdown bar was seen while the engine waited on
        # you: that decision's bar has been seen, so its going away is your action.
        self._my_turn_mark: int | None = None
        self._screen_stacks: dict[int, float] = {}  # chair -> stack read now, in BB
        self._screen_job: str | None = None
        # What the players at the table have done over the hands seen on the screen, read by
        # the models as at every table they were trained on. Keyed by chair, which is what
        # stays put from one hand to the next; see `_record_finished_hand`.
        self.stats = StatsTracker()
        # Chairs read empty in a row, and the chairs whose player has left during the hand
        # on the screen: see `_note_players_leaving`.
        self._empty_readings: dict[int, int] = {}
        self._left_chairs: set[int] = set()
        # Scripted actions the screen contradicts, and for how many readings in a row
        # (`_heal_script`).
        self._contradicted: dict[tuple, int] = {}
        # The action drawn for the decision being advised (`_sampled_action`), and which one.
        self._rng = random.Random()
        self._draw = None
        self._draw_key = None
        # The signs of the last new hand seen ("dealer", "hole") and when the first came.
        self._boundary_signs: set[str] = set()
        self._boundary_time = float("-inf")

        self._build()
        self.refresh()
        if read_screen:
            self._start_screen_reading()

    # -- layout -------------------------------------------------------------

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", pady=(0, 6))
        ttk.Label(top, text="Tavolo").pack(side="left", padx=(0, 3))
        for size in DEALER_TABLE_SIZES:
            ttk.Radiobutton(
                top, text=f"{size} giocatori", value=size, variable=self.table_size_var,
                command=lambda: self.set_table_size(self.table_size_var.get()),
            ).pack(side="left", padx=(0, 3))
        ttk.Label(top, text="Blind 0,5 / 1 BB").pack(side="left", padx=(9, 12))
        ttk.Label(top, text="Ante (BB)").pack(side="left", padx=(0, 3))
        ante = ttk.Entry(top, textvariable=self.ante_var, width=5)
        ante.pack(side="left", padx=(0, 10))
        ante.bind("<Return>", lambda _e: self.refresh())
        ttk.Button(top, text="Annulla ultima azione", command=self._undo).pack(side="left", padx=3)
        ttk.Button(top, text="Azzera azioni", command=self._clear).pack(side="left", padx=3)
        ttk.Button(top, text="Azzera statistiche", command=self.reset_stats).pack(side="left", padx=3)
        ttk.Checkbutton(
            top, text="Policy: avversari sempre nuovi (statistiche azzerate)",
            variable=self.fresh_opponents_var, command=self._fresh_opponents_toggled,
        ).pack(side="left", padx=6)
        ttk.Button(top, text="Torna al menu", command=self.app.show_setup).pack(side="right")

        main = ttk.Frame(self)
        main.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(
            main, width=self.table_geom.canvas[0], height=self.table_geom.canvas[1],
            highlightthickness=0, bg="#2b2b2b",
        )
        self.canvas.pack(side="left")
        self._draw_table()
        for chair in range(CHAIRS):
            self._build_chair(chair)

        side = ttk.Frame(main)
        side.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.warning = ttk.Label(side, foreground="#b00020", wraplength=300, justify="left")
        self.warning.pack(anchor="w")
        self.situation = ttk.Label(side, justify="left", font=("TkDefaultFont", 10, "bold"))
        self.situation.pack(anchor="w", pady=(2, 4))
        ttk.Label(side, textvariable=self.screen_var, foreground="#555555", wraplength=300,
                  justify="left").pack(anchor="w", pady=(0, 4))
        # The models' advice keeps a fixed height at the bottom (packed first, from the
        # bottom, so a short window squeezes the actions, not it); the list of actions
        # takes every line left above it, and always shows its last ones (`_write_log`).
        advisors = ttk.LabelFrame(side, text=f"Consiglio dei top {DEFAULT_ADVISORS}", padding=6)
        advisors.pack(side="bottom", fill="x", pady=(8, 0))
        # No "ask" button: the models answer by
        # themselves whenever it is your turn with both your cards known.
        ttk.Label(advisors, textvariable=self.status_var, wraplength=300).pack(anchor="w", pady=2)
        self.advice = tk.Text(advisors, width=36, height=ADVICE_LINES, state="disabled", wrap="word")
        self.advice.pack(fill="x")
        ttk.Label(side, text="Azioni").pack(anchor="w")
        log_box = ttk.Frame(side)
        log_box.pack(fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_box, orient="vertical")
        log_scroll.pack(side="right", fill="y")
        self.log = tk.Text(log_box, height=8, width=34, state="disabled", yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.configure(command=self.log.yview)

    def _draw_table(self) -> None:
        cx, cy = self.table_geom.center
        a, b = self.table_geom.table_radii
        self.canvas.create_oval(cx - a, cy - b, cx + a, cy + b, fill=FELT, outline=RAIL, width=10)
        self.pot_item = self.canvas.create_text(
            cx, cy - 55, text="", fill="white", font=("TkDefaultFont", 14, "bold")
        )
        board = tk.Frame(self.canvas, bg=FELT)
        for slot in range(5):
            button = tk.Button(
                board, text="?", width=3, height=2, font=("TkDefaultFont", 12, "bold"),
                command=lambda s=slot: self._pick_board(s),
            )
            button.pack(side="left", padx=2)
            self.board_buttons.append(button)
        self.canvas.create_window(cx, cy, window=board)
        self.street_item = self.canvas.create_text(
            cx, cy + 52, text="", fill="white", font=("TkDefaultFont", 11)
        )
        # Every chair's bet, drawn on the rail facing it (`_show_bets`).
        self.bet_items: dict[int, tuple[int, int]] = {}
        for chair in range(CHAIRS):
            x, y = self._bet_position(chair)
            chip = self.canvas.create_rectangle(x, y, x, y, fill=BET_CHIP_BG, outline=BET_CHIP_FG,
                                                width=2, state="hidden")
            text = self.canvas.create_text(x, y, text="", fill=BET_CHIP_FG,
                                           font=("TkDefaultFont", 12, "bold"), state="hidden")
            self.bet_items[chair] = (chip, text)

    def _bet_position(self, chair: int) -> tuple[float, float]:
        """On the rail, in the direction of the chair's box from the centre of the felt."""
        cx, cy = self.table_geom.center
        a, b = self.table_geom.table_radii
        x, y, _anchor = chair_slot(chair, self.table_geom.canvas, CANVAS_MARGIN)
        angle = math.atan2((y - cy) / b, (x - cx) / a)
        return cx + BET_RADIUS * a * math.cos(angle), cy + BET_RADIUS * b * math.sin(angle)

    def _show_bets(self, bets: dict[int, int]) -> None:
        """Each chair's chips in front this street (`bets` by chair, in chips), as a
        number on the felt; nothing for a chair with none."""
        for chair, (chip, text) in self.bet_items.items():
            amount = bets.get(chair, 0)
            if amount <= 0:
                self.canvas.itemconfigure(chip, state="hidden")
                self.canvas.itemconfigure(text, state="hidden", text="")
                continue
            self.canvas.itemconfigure(text, text=self._bb(amount), state="normal")
            x0, y0, x1, y1 = self.canvas.bbox(text)
            self.canvas.coords(chip, x0 - 6, y0 - 3, x1 + 6, y1 + 3)
            self.canvas.itemconfigure(chip, state="normal")
            self.canvas.tag_raise(text, chip)

    def _build_chair(self, chair: int) -> None:
        x, y, anchor = chair_slot(chair, self.table_geom.canvas, CANVAS_MARGIN)
        holder = tk.Frame(self.canvas, bg="#2b2b2b")
        # Pinned by the side facing the edge, so the box grows inwards when its
        # actions appear and stays on the canvas.
        self.canvas.create_window(x, y, window=holder, anchor=anchor)
        plus = tk.Button(
            holder, text="+", width=3, font=("TkDefaultFont", 12, "bold"),
            command=lambda c=chair: self.add_player(c),
        )
        box = tk.Frame(holder, bd=2, relief="ridge", bg="#f4f1ea", highlightthickness=3,
                       highlightbackground="#f4f1ea")
        if chair in WIDE_CHAIRS:
            columns = tk.Frame(box, bg="#f4f1ea")
            columns.pack(fill="x")
            details = tk.Frame(columns, bg="#f4f1ea")
            details.pack(side="left", anchor="n")
            play = tk.Frame(columns, bg="#f4f1ea")
            play.pack(side="left", anchor="n", padx=(6, 0))
        else:
            details = play = box
        head = tk.Frame(details, bg="#f4f1ea")
        head.pack(fill="x")
        tk.Entry(head, textvariable=self.name_vars[chair], width=10).pack(side="left")
        dealer = tk.Button(head, text="D", width=2, command=lambda c=chair: self.set_dealer(c))
        dealer.pack(side="left", padx=2)
        if chair != USER_CHAIR:
            tk.Button(head, text="x", width=2, command=lambda c=chair: self.remove_player(c)).pack(
                side="left"
            )
        row = tk.Frame(details, bg="#f4f1ea")
        row.pack(fill="x", pady=1)
        # The stack this hand *started* with: the engine takes every bet off it
        # itself, which is why it is not rewritten from the screen mid-hand. What
        # is left now is shown under it (`_show_state`, "resta").
        tk.Label(row, text="inizio BB", bg="#f4f1ea").pack(side="left")
        entry = tk.Entry(row, textvariable=self.stack_vars[chair], width=7)
        entry.pack(side="left", padx=2)
        # Only Return applies a typed value. A FocusOut binding would rebuild the
        # action buttons under the mouse whenever one entry is clicked after
        # another, and the click on the second would be lost.
        entry.bind("<Return>", lambda _e: self.refresh())
        info = tk.Label(details, text="", bg="#f4f1ea", justify="left", font=("TkDefaultFont", 9))
        info.pack(fill="x")
        if chair == USER_CHAIR:
            cards = tk.Frame(play, bg="#f4f1ea")
            cards.pack(pady=2)
            for slot in range(2):
                button = tk.Button(
                    cards, text="?", width=3, height=2, font=("TkDefaultFont", 12, "bold"),
                    command=lambda s=slot: self._pick_hole(s),
                )
                button.pack(side="left", padx=2)
                self.hole_buttons.append(button)
        actions = tk.Frame(play, bg="#f4f1ea")
        actions.pack(fill="x")
        self.chair_ui[chair] = SimpleNamespace(
            plus=plus, box=box, dealer=dealer, info=info, actions=actions, holder=holder
        )
        self.bind_class(_raise_tag(holder), "<ButtonPress-1>", lambda _e, h=holder: h.lift())
        _make_raisable(holder)

    # -- the table ----------------------------------------------------------

    def add_player(self, chair: int) -> None:
        if self.layout.add(chair):
            self.stack_vars[chair].set(DEFAULT_STACK_BB)
            self.name_vars[chair].set(f"Avv. {chair}")
            self._seating_changed()

    def remove_player(self, chair: int) -> None:
        if self.layout.remove(chair):
            self._seating_changed()

    def set_dealer(self, chair: int) -> None:
        if self.layout.set_dealer(chair):
            self._seating_changed()

    def _seating_changed(self) -> None:
        # The actions belong to one particular seating: who acts after whom
        # changes with the button and with who is at the table, so a script kept
        # across the change would be read as something nobody chose.
        self.script.clear()
        self.refresh()

    # -- the cards ----------------------------------------------------------

    def _taken(self, *, except_card: Card | None = None) -> set[Card]:
        used = {card for card in self.hole if card is not None} | set(self.board)
        used.discard(except_card)
        return used

    def _open_picker(self, title: str, current: Card | None, on_pick) -> CardPicker:
        return CardPicker(
            self, taken=self._taken(except_card=current), on_pick=on_pick, title=title,
            can_clear=current is not None,
        )

    def _pick_hole(self, slot: int) -> CardPicker:
        def done(card: Card | None) -> None:
            self.hole[slot] = card
            self.refresh()

        return self._open_picker(f"Tua carta {slot + 1}", self.hole[slot], done)

    def _pick_board(self, slot: int) -> CardPicker | None:
        if slot > len(self.board):
            return None  # the board is filled in order: flop, turn, river

        def done(card: Card | None) -> None:
            if card is None:
                del self.board[slot]
            elif slot == len(self.board):
                self.board.append(card)
            else:
                self.board[slot] = card
            self.refresh()

        current = self.board[slot] if slot < len(self.board) else None
        label = "Flop" if slot < 3 else ("Turn" if slot == 3 else "River")
        return self._open_picker(f"{label}, carta {slot + 1}", current, done)

    # -- the spot -----------------------------------------------------------

    @staticmethod
    def _number(var: tk.StringVar, name: str) -> int:
        try:
            return int(var.get())
        except ValueError:
            raise ValueError(f"{name}: serve un numero intero") from None

    def _big_blind(self) -> int:
        """The engine's big blind in chips (2): what one BB on screen is worth."""
        return self._number(self.bb_var, "Big blind")

    def _chips(self, var: tk.StringVar, name: str) -> int:
        """A big-blind amount typed by the user, in the engine's chips."""
        try:
            chips = parse_bb(var.get(), self._big_blind())
        except ValueError:
            raise ValueError(f"{name}: serve un numero di BB (es. 100 o 37,5)") from None
        if chips <= 0:
            raise ValueError(f"{name}: deve essere più di 0 BB")
        return chips

    def _ante(self) -> int:
        """The ante in the engine's chips; empty is none."""
        text = self.ante_var.get().strip()
        if not text:
            return 0
        try:
            return parse_bb(text, self._big_blind())
        except ValueError:
            raise ValueError("Ante: serve un numero di BB (es. 0 o 0,1)") from None

    def _has_ante(self) -> bool:
        try:
            return self._ante() > 0
        except ValueError:
            return False

    def _bb(self, chips: int) -> str:
        """Chips as big blinds for display ("18,5 BB")."""
        return format_bb(chips, self._big_blind())

    def build_spot(self) -> Spot:
        if self.layout.players < 2:
            raise ValueError("Aggiungi almeno un avversario con il tasto +")
        default = parse_bb(DEFAULT_STACK_BB, self._big_blind())
        stacks = {}
        for seat, chair in enumerate(self.layout.order()):
            text = self.stack_vars[chair].get().strip()
            stacks[seat] = self._chips(self.stack_vars[chair], "Stack") if text else default
        hole = tuple(self.hole) if all(card is not None for card in self.hole) else None
        seat_stats = {} if self.fresh_opponents_var.get() else self.stats.vectors(
            {seat: self._stats_id(chair) for seat, chair in enumerate(self.layout.order())}
        )
        return Spot(
            num_players=self.layout.players,
            starting_stack=default,
            small_blind=self._number(self.sb_var, "Small blind"),
            big_blind=self._number(self.bb_var, "Big blind"),
            ante=self._ante(),
            my_seat=self.layout.seat_of(USER_CHAIR),
            hole_cards=hole,
            board=tuple(self.board),
            script=list(self.script),
            stacks=stacks,
            seat_stats=seat_stats,
        )

    def _fresh_opponents_toggled(self) -> None:
        self.app.spot_fresh_opponents = self.fresh_opponents_var.get()
        self.refresh()

    @staticmethod
    def _stats_id(chair: int) -> str:
        return f"chair{chair}"

    def _note_players_leaving(self) -> None:
        """Forget a player's statistics as soon as their seat is read empty in
        `EMPTY_READINGS_TO_FORGET` readings in a row, whenever that is: whoever sits there
        next is someone else. Not only when the seating is applied (at a new hand), which
        missed a player who stood up mid-hand and was replaced before the next deal -- the
        newcomer inherited their numbers. Two readings, so one misread frame does not wipe
        a player's history."""
        from pokerlab.vision.labels import SEAT_EMPTY

        for chair, state in self._seat_states.items():
            if state != SEAT_EMPTY:
                self._empty_readings.pop(chair, None)
                continue
            self._empty_readings[chair] = self._empty_readings.get(chair, 0) + 1
            if self._empty_readings[chair] >= EMPTY_READINGS_TO_FORGET:
                self.stats.forget(self._stats_id(chair))
                self._left_chairs.add(chair)

    def reset_stats(self) -> None:
        """Forget everything seen so far (another table, say) and redraw."""
        self.stats = StatsTracker()
        self.refresh()

    def _record_finished_hand(self) -> None:
        """Give the tracker the hand the screen has just left, as the actions rebuilt from it.

        Called at a hand boundary (a new deal, the button moving) *before* the new seating is
        applied, while the layout, the script and the chairs still describe that hand. Only
        hands read off the screen are recorded, never a spot built by hand.

        What is counted is what the readings proved (see `action_sync`). WTSD included: a
        player who saw the flop and never folded reached the showdown unless everyone else
        folded, and the folds are in the actions. The limit is the end of the hand: a last
        fold missed because the next deal replaced the hand between two readings makes the
        player left look like they reached the showdown, and a showdown whose face-up cards
        are read as "out" can add a fold that never happened. Every seat at the table is
        counted as dealt in; a player who sat down mid-hand and waits for the next deal gets
        one spurious chance, which is noise."""
        if not self.script:
            return
        try:
            state = replay(self.build_spot())
        except ValueError:
            return
        if not state.records:
            return
        seats = range(self.layout.players)
        ids = {seat: self._stats_id(chair) for seat, chair in enumerate(self.layout.order())}
        self.stats.record_hand(
            state.records,
            dealt=list(seats),
            button_seat=0,
            board_cards=self._board_reached(state.records),
            player_ids=ids,
        )

    def _hand_boundary(self, change, now: float) -> bool:
        """Whether this reading ends the hand the screen was showing.

        A deal is announced by two signs, the button moving and your new cards, and they
        often land in different readings. Recorded at each, every real hand counted twice:
        the second time as whatever `action_sync` had made of the new deal in between (the
        players read "out" before their cards arrive, so folds). So the other sign within
        `SAME_DEAL_SECONDS` of the first is the same deal: it clears the actions, as before,
        but records nothing. The same sign again, or after that, is a new hand."""
        signs = {"dealer"} if change.dealer is not None else set()
        if change.new_hand:
            signs.add("hole")
        if not signs:
            return False
        if now - self._boundary_time <= SAME_DEAL_SECONDS and not signs & self._boundary_signs:
            self._boundary_signs |= signs
            return False
        self._boundary_signs, self._boundary_time = signs, now
        return True

    def _board_reached(self, records) -> int:
        """How many board cards the hand had: the board last read, or what the streets of the
        recorded actions imply if the screen had already cleared it by the next deal."""
        by_street = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
        return max([len(self.board), *(by_street.get(record.street, 0) for record in records)])

    def _show_stats(self) -> None:
        """Every seated player's statistics under their chair, yours included, whatever
        state the spot is in: added at the end of every refresh, so they stay on screen
        when the rebuilt hand is over or the table is not playable, which used to leave
        them out."""
        for chair in self.layout.chairs:
            info = self.chair_ui[chair].info
            text = info.cget("text")
            stats = self._stats_text(chair)
            info.configure(text=f"{text}\n{stats}" if text else stats)

    def _stats_text(self, chair: int) -> str:
        """Every statistic of the player on this chair: the rate and, in brackets, the
        chances it is counted over -- 50% of 2 and 50% of 200 are not the same fact. "-"
        for a statistic with no chance yet; a line saying so before the first hand."""
        player = self._stats_id(chair)
        hands = self.stats.hands(player)
        if not hands:
            return "statistiche: nessuna mano registrata"
        rates = self.stats.rates(player)
        cells = []
        for name, label in STAT_LABELS.items():
            events, chances = rates[name]
            cells.append(f"{label} {events / chances:.0%} ({chances})" if chances else f"{label} -")
        lines = ["  ".join(cells[i:i + STATS_PER_LINE]) for i in range(0, len(cells), STATS_PER_LINE)]
        return "\n".join([f"statistiche su {hands} mani", *lines])

    def _player_name(self, chair: int) -> str:
        return self.name_vars[chair].get().strip() or f"Posto {chair}"

    def refresh(self) -> None:
        try:
            self._refresh()
        finally:
            for ui in self.chair_ui.values():
                _make_raisable(ui.holder)  # the action buttons were just rebuilt
            if self.state is not None and not self.state.finished and self.state.to_act is not None:
                # Whoever acts goes in front, so their actions are never hidden.
                self.chair_ui[self.layout.chair_of(self.state.to_act)].holder.lift()
            self._show_stats()
            self._mark_seat_states()
            self._fit_canvas()

    def _fit_canvas(self) -> None:
        """Keep every chair on the canvas: slide the whole table right/down when
        a box would stick out of the top or the left (a chair at the top grows
        upwards when its actions appear -- ~100 px to ~245 -- and was cut off),
        then grow the canvas to the rest, and the window with it, within the
        screen. Never shrinks or slides back, so the table does not jump around."""
        self.update_idletasks()
        bbox = self._content_bbox()
        if bbox is None:
            return
        dx, dy = max(0, CANVAS_MARGIN - bbox[0]), max(0, CANVAS_MARGIN - bbox[1])
        if dx or dy:
            self.canvas.move("all", dx, dy)
            bbox = (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)
        width = max(self.table_geom.canvas[0], bbox[2] + CANVAS_MARGIN)
        height = max(self.table_geom.canvas[1], bbox[3] + CANVAS_MARGIN)
        if (width, height) != (int(self.canvas.cget("width")), int(self.canvas.cget("height"))):
            self.canvas.configure(width=width, height=height)
            self.update_idletasks()
        top = self.winfo_toplevel()
        want_w = min(top.winfo_reqwidth(), top.winfo_screenwidth() - 40)
        want_h = min(top.winfo_reqheight(), top.winfo_screenheight() - 80)
        if want_w > top.winfo_width() or want_h > top.winfo_height():
            top.geometry(f"{max(want_w, top.winfo_width())}x{max(want_h, top.winfo_height())}")

    def _content_bbox(self) -> tuple[int, int, int, int] | None:
        """What the canvas holds, from each chair's *requested* size.

        `canvas.bbox` lags here: right after the action buttons are rebuilt it
        still has the window items at their old size, so a box that had just
        grown upwards was measured where it used to be and left 7 px off the
        top. A widget's requested size is current as soon as it is packed."""
        anchors = {"n": (0.5, 0), "s": (0.5, 1), "center": (0.5, 0.5), "e": (1, 0.5), "w": (0, 0.5),
                   "nw": (0, 0), "ne": (1, 0), "sw": (0, 1), "se": (1, 1)}
        boxes = []
        for item in self.canvas.find_all():
            if self.canvas.type(item) == "window":
                widget = self.nametowidget(self.canvas.itemcget(item, "window"))
                if not widget.winfo_ismapped() and not widget.winfo_reqwidth():
                    continue
                x, y = self.canvas.coords(item)
                fx, fy = anchors[self.canvas.itemcget(item, "anchor")]
                w, h = widget.winfo_reqwidth(), widget.winfo_reqheight()
                boxes.append((x - fx * w, y - fy * h, x + (1 - fx) * w, y + (1 - fy) * h))
            else:
                box = self.canvas.bbox(item)
                if box is not None:
                    boxes.append(box)
        if not boxes:
            return None
        return (
            round(min(b[0] for b in boxes)), round(min(b[1] for b in boxes)),
            round(max(b[2] for b in boxes)), round(max(b[3] for b in boxes)),
        )

    def _refresh(self) -> None:
        for chair, ui in self.chair_ui.items():
            ui.plus.pack_forget()
            ui.box.pack_forget()
            if chair in self.layout.chairs:
                ui.box.pack()
                ui.dealer.configure(
                    bg=DEALER_BUTTON if chair == self.layout.dealer else "#e0e0e0",
                    relief="sunken" if chair == self.layout.dealer else "raised",
                )
            else:
                ui.plus.pack()
            ui.box.configure(highlightbackground="#f4f1ea")
            _paint_box(ui.box, folded=False)
            ui.info.configure(text="")
            for widget in ui.actions.winfo_children():
                widget.destroy()
            # A frame whose last child is destroyed keeps the size the children gave it
            # (Tk only recomputes it when a packed child changes): without this a chair
            # that has acted stays as tall as its action buttons were, empty.
            ui.actions.configure(width=1, height=1)
        self._draw_cards()
        self.warning.configure(text="")
        self.canvas.itemconfigure(self.pot_item, text="")
        self.canvas.itemconfigure(self.street_item, text="")
        self._show_bets({})
        try:
            spot = self.build_spot()
            self.state = replay(spot)
        except ValueError as exc:
            self.state = None
            self.situation.configure(text="")
            self.warning.configure(text=str(exc))
            self._write(self.log, "")
            self._write(self.advice, "")
            return
        state = self.state
        if state.invalid_from is not None:
            # The table changed under the script: keep the part that still holds.
            self.script = self.script[: state.invalid_from]
            self.warning.configure(
                text=f"Le azioni dalla n. {state.invalid_from + 1} non valgono più e sono state tolte."
            )
        names = position_names(spot.num_players)
        for seat, chair in enumerate(self.layout.order()):
            self.chair_ui[chair].info.configure(text=names[seat])
        self._write_log(state, names)
        if state.finished:
            self.situation.configure(text="Mano conclusa: annulla l'ultima azione per riaprirla.")
            self._write(self.advice, "")
            return
        self._show_state(spot, state, names)
        for legal in state.legal_actions:
            self._action_widget(self.chair_ui[self.layout.chair_of(state.to_act)].actions, legal, state)
        # Your turn with your cards in: the models answer without being asked.
        if state.to_act == spot.my_seat and spot.hole_cards is not None:
            self._consult(state, spot)
        else:
            self._write(self.advice, "")

    def _show_state(self, spot: Spot, state: SpotState, names: list[str]) -> None:
        board = " ".join(format_card(c) for c in state.board) or "-"
        turn_chair = self.layout.chair_of(state.to_act)
        mine = "  <- tocca a te" if state.to_act == spot.my_seat else ""
        self.situation.configure(
            text=f"{state.street.name.lower()} | piatto {self._bb(state.pot)} | board {board}\n"
            f"Parla: {self._player_name(turn_chair)} ({names[state.to_act]}){mine}"
        )
        self.canvas.itemconfigure(self.pot_item, text=f"Piatto {self._bb(state.pot)}")
        self.canvas.itemconfigure(self.street_item, text=state.street.name.lower())
        self._show_bets({self.layout.chair_of(seat): chips for seat, chips in state.bets.items()})
        self.chair_ui[turn_chair].box.configure(highlightbackground=HIGHLIGHT)
        folded = {
            info.seat for info in state.observation.seats if info.status is PlayerStatus.FOLDED
        }
        for seat, chair in enumerate(self.layout.order()):
            text = names[seat]
            if seat in folded:
                text += "  FOLD"
            elif state.bets.get(seat):
                text += f"  puntata {self._bb(state.bets[seat])}"
            if seat in state.stacks:
                text += f"\nresta {self._bb(state.stacks[seat])}"
                shown = self._screen_stacks.get(chair)
                if shown is not None and abs(shown - state.stacks[seat] / self._big_blind()) > 0.01:
                    # The screen disagrees with what the rebuilt actions leave:
                    # a missed or misread action, worth a look.
                    text += f"  ≠ schermo {self._bb(round(shown * self._big_blind()))}"
            self.chair_ui[chair].info.configure(text=text)
        if len(state.board) > len(spot.board):
            self.warning.configure(
                text=(self.warning.cget("text") + " Imposta le carte del board al centro: "
                      "quelle usate finora sono casuali.").strip()
            )

    def _draw_cards(self) -> None:
        for slot, button in enumerate(self.hole_buttons):
            card = self.hole[slot]
            button.configure(text=card_text(card), fg=card_color(card))
        for slot, button in enumerate(self.board_buttons):
            card = self.board[slot] if slot < len(self.board) else None
            button.configure(
                text=card_text(card), fg=card_color(card),
                state="normal" if slot <= len(self.board) else "disabled",
            )

    def _action_widget(self, parent: tk.Frame, legal: LegalAction, state: SpotState) -> None:
        if legal.action_type in (ActionType.BET, ActionType.RAISE):
            verb = "Bet" if legal.action_type is ActionType.BET else "Raise a"
            big_blind = self._big_blind()
            var = tk.StringVar(value=bb_number(legal.min_amount, big_blind))
            line = tk.Frame(parent, bg="#f4f1ea")
            line.pack(fill="x", pady=1)
            entry = tk.Entry(line, textvariable=var, width=6)
            entry.pack(side="left")
            self._bind_wheel(entry, var, legal)

            low, high = bb_number(legal.min_amount, big_blind), bb_number(legal.max_amount, big_blind)

            def go(legal=legal, var=var) -> None:
                try:
                    amount = parse_bb(var.get(), big_blind)
                except ValueError:
                    return
                if not legal.min_amount <= amount <= legal.max_amount:
                    messagebox.showwarning("Importo", f"Tra {low} e {high} BB.")
                    return
                self._append(Action(legal.action_type, amount))

            button = tk.Button(line, text=f"{verb} ({low}-{high} BB)", command=go)
            button.pack(side="left", padx=2)
            self._bind_wheel(button, var, legal)
            return
        label = describe_action(Action(legal.action_type), state.observation, self._big_blind())
        label = label[:1].upper() + label[1:]  # not .capitalize(): it would turn "BB" into "bb"
        tk.Button(
            parent, text=label, command=lambda t=legal.action_type: self._append(Action(t))
        ).pack(fill="x", pady=1)

    def _wheel_step(self, var: tk.StringVar, legal: LegalAction, direction: int) -> None:
        """Move the amount (in BB) by one big blind, kept inside what the engine allows."""
        big_blind = self._big_blind()
        try:
            current = parse_bb(var.get(), big_blind)
        except ValueError:
            current = legal.min_amount
        step = max(1, WHEEL_STEP_BIG_BLINDS * big_blind)
        chips = min(legal.max_amount, max(legal.min_amount, current + direction * step))
        var.set(bb_number(chips, big_blind))

    def _bind_wheel(self, widget: tk.Widget, var: tk.StringVar, legal: LegalAction) -> None:
        """Wheel up raises the amount, wheel down lowers it.

        X11 reports the wheel as Button-4 (up) and Button-5 (down), Windows and
        macOS as <MouseWheel> with a signed delta. Each sequence is bound to its
        own direction instead of reading `event.num`, which is not dependable:
        the same gotcha as the raise slider of the table screen.
        """
        for sequence, direction in (("<Button-4>", 1), ("<Button-5>", -1)):
            widget.bind(sequence, lambda _e, d=direction: self._wheel_step(var, legal, d))
        widget.bind(
            "<MouseWheel>",
            lambda e: self._wheel_step(var, legal, 1 if e.delta > 0 else -1),
        )

    # -- reading the screen -------------------------------------------------

    def set_table_size(self, size: int) -> bool:
        """Read the client as a `size`-seat table (6 or 8) from now on.

        The seat zones and the seat->chair mapping both change, so everything
        remembered from earlier readings is dropped and the next reading is
        taken as the first: it seats the players afresh. The choice is kept on
        the app, so the screen reopens on it. False if nothing changed."""
        if size not in DEALER_TABLE_SIZES or size == self.table_size:
            self.table_size_var.set(self.table_size)
            return False
        self.table_size = size
        self.table_size_var.set(size)
        self.app.spot_table_size = size
        if self._reader is not None:
            self._reader.players = size
        self._last_reading = None
        self._last_dealer = None
        self._seat_states = {}
        self._empty_readings = {}
        self._seating_read = False
        self._screen_stacks = {}
        self._stacks_from_screen = set()
        self.refresh()
        return True

    def _start_screen_reading(self) -> None:
        from pokerlab.gui.screen_reader import ScreenReader, VisionNotAvailable

        try:
            self._reader = ScreenReader(players=self.table_size)
        except VisionNotAvailable as exc:
            self.screen_var.set(f"Schermo: lettura non disponibile, {exc}")
            return
        self._screen_job = self.after(0, self._poll_screen)

    def _poll_screen(self) -> None:
        """Read the client and apply what changed; then again in a moment."""
        from pokerlab.gui.screen_reader import VisionNotAvailable

        self._screen_job = None
        try:
            reading = self._reader.read()
        except VisionNotAvailable as exc:
            self.screen_var.set(f"Schermo: lettura non disponibile, {exc}")
            return  # permanent: no point retrying
        except Exception as exc:  # noqa: BLE001 - a bad tick must not stop the reading
            self.screen_var.set(f"Schermo: errore di lettura, {exc}")
        else:
            self.apply_reading(reading)
        if self.winfo_exists():
            self._screen_job = self.after(SCREEN_POLL_MS, self._poll_screen)

    def apply_reading(self, reading) -> bool:
        """Put what the screen shows into the spot, if the screen changed.

        A new hand (different hole cards) clears the actions, which belonged to
        the previous one; seats, stacks and dealer are kept. The button moving
        puts it on the chair matching that client seat (`chair_for_client_seat`),
        seating a player there if the chair was empty -- the button is always in
        front of someone -- and, being a new hand too, clears the actions.
        Returns whether anything was applied."""
        from pokerlab.gui.screen_reader import diff_reading
        from pokerlab.vision.labels import SEAT_EMPTY

        change = diff_reading(self._last_reading, reading, self._last_hand, self._last_dealer)
        self._last_reading = reading
        if self._hand_boundary(change, time.monotonic()):
            # The hand on the screen is over: count it while the seating still describes it.
            self._record_finished_hand()
            self._my_turn_mark = None  # a bar seen in the last hand says nothing about this one
            # ... and then forget whoever left during it, or the hand they played would be
            # the first one of the player sitting there next.
            for chair in self._left_chairs:
                self.stats.forget(self._stats_id(chair))
            self._left_chairs.clear()
        ante_changed = self._read_ante(reading, new_hand=change.new_hand or change.dealer is not None)
        if reading.seats is not None:
            self._seat_states = {
                chair_for_client_seat(self.table_size, seat): state for seat, state in reading.seats.items()
            }
            self._note_players_leaving()
            # Who sits where changes only between hands: applied mid-hand it would
            # wipe the actions being entered every time someone stood up. The one
            # exception is a player in sit-out whose blind shows up only after the
            # new hand was seen: before anyone has acted, they are seated then.
            if (change.dealer is not None or change.new_hand or not self._seating_read
                    or ante_changed or self._missing_sit_out_blind(reading)):
                self._apply_seating(reading.seats, reading.bets)
                self._apply_stacks(reading)
        if change.dealer is not None:
            self._last_dealer = change.dealer
            chair = chair_for_client_seat(self.table_size, change.dealer)
            if (reading.seats or {}).get(change.dealer) == SEAT_EMPTY and chair not in self.layout.chairs:
                # The button on an empty seat: its player has gone (a dead button). Nobody
                # is seated there for it; it passes to the player before.
                self.layout.set_dealer(self.layout.previous_occupied(chair))
            else:
                if self.layout.add(chair):
                    self.stack_vars[chair].set(DEFAULT_STACK_BB)
                    self.name_vars[chair].set(f"Avv. {chair}")
                self.layout.set_dealer(chair)
            self.script.clear()
        if change.hole is not None:
            cards = [parse_card(c) for c in change.hole]
            self.hole = cards if cards else [None, None]
            if change.hole:
                self._last_hand = change.hole
            if change.new_hand:
                self.script.clear()
        if change.board is not None:
            self.board = [parse_card(c) for c in change.board]
        synced = self._sync_actions(reading)
        stacks_moved = self._note_screen_stacks(reading)
        hole = " ".join(card_text(c) for c in self.hole if c) or "-"
        board = " ".join(card_text(c) for c in self.board) or "-"
        problems = f" | {'; '.join(reading.problems)}" if reading.problems else ""
        dealer = "" if reading.dealer is None else f", dealer posto {reading.dealer}"
        seats = ""
        if reading.seats is not None:
            counts = {label: sum(state == key for state in reading.seats.values())
                      for key, label in SEAT_STATE_LABELS.items()}
            seats = ", posti: " + ", ".join(f"{n} {label}" for label, n in counts.items() if n)
        tracked = sum(
            1 for chair in self.layout.chairs if chair != USER_CHAIR and self.stats.hands(self._stats_id(chair))
        )
        statistics = f" | statistiche: {tracked} avversari osservati" if tracked else ""
        self.screen_var.set(
            f"Schermo ({time.strftime('%H:%M:%S')}): mano {hole}, board {board}{dealer}{seats}"
            f"{self._sync_note}{statistics}{problems}"
        )
        if change.any or synced or stacks_moved or ante_changed:
            self.refresh()
        self._log_reading(reading, change.any or synced or ante_changed)
        return change.any or synced or ante_changed

    def _table_view(self, reading):
        """The reading in the engine's seat numbers and chips, for `action_sync`."""
        from pokerlab.gui.action_sync import TableView
        from pokerlab.vision.labels import SEAT_IN_HAND

        try:
            big_blind = int(self.bb_var.get())
        except ValueError:
            return None
        view = TableView(
            board_cards=None if reading.board is None else len(reading.board), my_turn=reading.my_turn,
            tolerance=max(1, round(0.05 * big_blind)),  # half of the 0,1 BB the screen writes
        )
        for client_seat, amount in (reading.stacks or {}).items():
            chair = chair_for_client_seat(self.table_size, client_seat)
            # Only a chair whose starting stack this hand came from the screen: against a
            # default 100 BB, a stack read later would "drop" by a bet nobody made.
            # 0 is left out too: an empty zone and an all-in look the same.
            if chair in self._stacks_from_screen and chair in self.layout.chairs and amount > 0:
                view.stacks[self.layout.seat_of(chair)] = round(amount * big_blind)
        for client_seat, amount in (reading.bets or {}).items():
            chair = chair_for_client_seat(self.table_size, client_seat)
            if chair in self.layout.chairs:
                view.bets[self.layout.seat_of(chair)] = round(amount * big_blind)
        for client_seat, state in (reading.seats or {}).items():
            chair = chair_for_client_seat(self.table_size, client_seat)
            if chair in self.layout.chairs and state in SEAT_MARKS:
                view.out.add(self.layout.seat_of(chair))
            elif chair in self.layout.chairs and state == SEAT_IN_HAND:
                view.in_hand.add(self.layout.seat_of(chair))
        return view

    def _sync_actions(self, reading) -> bool:
        """Append the actions the picture proves were taken (`action_sync`), and
        check the pot read against the pot the actions add up to. Returns
        whether any action was added."""
        from pokerlab.gui.action_sync import sync_actions

        self._sync_note = ""
        self._last_view = None
        if reading.bets is None and not reading.my_turn:
            return False  # neither amounts nor your timer read: nothing to go on
        view = self._table_view(reading)
        if view is None:
            return False
        view.my_turn_seen = self._my_turn_mark == len(self.script)
        self._last_view = view
        try:
            healed = self._heal_script(view)
            spot = self.build_spot()
            synced, result = sync_actions(spot, view)
        except ValueError:
            return False  # the table is not playable yet (one player, say)
        if result.actions:
            self.script = list(synced.script)
        if result.waiting_for == synced.my_seat and reading.my_turn:
            self._my_turn_mark = len(self.script)
        names = position_names(synced.num_players)
        added = ", ".join(
            f"{names[seat]} {describe_action(action, None, synced.big_blind)}"
            for seat, action in result.actions
        )
        waiting = "" if result.waiting_for is None else f"tocca a {names[result.waiting_for]}"
        parts = [p for p in (f"+ {added}" if added else "", waiting) if p]
        if result.waiting_for is None:
            parts.append(result.note)  # "mano conclusa": said, not left blank
        if healed:
            parts.insert(0, healed)
        self._sync_note = f" | azioni: {'; '.join(parts)}" if parts else ""
        self._sync_note += self._pot_check(synced, reading, view)
        return bool(result.actions) or bool(healed)

    def _heal_script(self, view) -> str:
        """Take back a FOLD or an ALL_IN the screen has kept contradicting for
        `HEAL_READINGS` readings in a row, with every action after it; what it says, or "".

        An action is never taken back otherwise, so one wrong fold (a player read "out" for
        a frame) or a wrong all-in (a starting stack read short) ended the rebuilt hand, or
        changed it, while the real one went on: nobody lit up, your turn was not seen, and
        the screen waited for the next deal. The proof against a FOLD is the player still
        holding cards (`TableView.in_hand`); against an ALL_IN, chips still behind on
        screen. The next rebuild (`action_sync`) puts back whatever the screen does prove."""
        state = replay(self.build_spot())
        now: dict[tuple, str] = {}
        for index, (seat, action) in enumerate(state.taken[: len(self.script)]):
            if action.action_type is ActionType.FOLD and seat in view.in_hand:
                now[(index, seat, action)] = "ha ancora le carte"
            elif action.action_type is ActionType.ALL_IN and view.stacks.get(seat, 0) > view.tolerance:
                now[(index, seat, action)] = "ha ancora chips"
        self._contradicted = {key: self._contradicted.get(key, 0) + 1 for key in now}
        due = sorted(key for key, count in self._contradicted.items() if count >= HEAL_READINGS)
        if not due:
            return ""
        index, seat, action = due[0]
        self.script = self.script[:index]
        self._contradicted = {}
        name = position_names(len(state.stacks))[seat]
        return f"tolto {name} {describe_action(action, None, self._big_blind())} ({now[due[0]]})"

    def _log_reading(self, reading, changed: bool) -> None:
        """One line per reading that changed something, in `SCREEN_LOG`: what was read and
        what the rebuild made of it. Never lets a failure to write stop the reading."""
        if not changed or self._reader is None:
            return
        try:
            SCREEN_LOG.parent.mkdir(parents=True, exist_ok=True)
            if SCREEN_LOG.exists() and SCREEN_LOG.stat().st_size > SCREEN_LOG_MAX_BYTES:
                SCREEN_LOG.replace(SCREEN_LOG.with_suffix(".log.1"))
            script = " ".join(
                f"{a.action_type.name.lower()}{'' if not a.amount else a.amount}" for a in self.script
            )
            state = self.state
            where = ("-" if state is None else "finita" if state.finished
                     else f"tocca seat {state.to_act} {state.street.name.lower()}")
            line = (
                f"{time.strftime('%H:%M:%S')} hole={reading.hole} board={reading.board} "
                f"dealer={reading.dealer} seats={reading.seats} bets={reading.bets} "
                f"stacks={reading.stacks} pot={reading.pot} my_turn={reading.my_turn} "
                f"| script[{len(self.script)}]={script} | {where}{self._sync_note}\n"
            )
            with SCREEN_LOG.open("a", encoding="utf-8") as out:
                out.write(line)
        except OSError:
            pass

    def _pot_check(self, spot, reading, view) -> str:
        """A warning when the pot the actions add up to is not the one shown:
        the client's pot is the earlier streets, so the bets in front are added
        to it before comparing. Silent when it matches or cannot be checked."""
        if reading.pot is None or reading.bets is None:
            return ""
        try:
            state = replay(spot)
            big_blind = int(self.bb_var.get())
        except ValueError:
            return ""
        if state.finished:
            return ""
        shown = round(reading.pot * big_blind) + sum(view.bets.values())
        if shown == state.pot:
            return ""
        return (f" | ATTENZIONE piatto: letto {shown / big_blind:g} BB, "
                f"dalle azioni {state.pot / big_blind:g} BB")

    def _read_ante(self, reading, *, new_hand: bool = False) -> bool:
        """Take the ante off a preflop reading (`screen_reader.ante_from_pot`), rounded to
        the engine's chip. Only while nobody has acted -- or at a `new_hand`, whose reading
        still finds the last hand's actions, cleared right after: the pot does not change
        during the preflop, and a misread mid-hand would rewrite a hand already rebuilt.
        True if the ante changed."""
        from pokerlab.gui.screen_reader import ante_from_pot

        if (self.script and not new_hand) or not reading.seats:
            return False
        ante_bb = ante_from_pot(reading.seats, reading.pot, reading.board)
        if ante_bb is None:
            return False
        try:
            current = self._ante()
        except ValueError:
            current = None
        chips = round(ante_bb * ENGINE_BIG_BLIND)
        if chips == current:
            return False
        self.ante_var.set(bb_number(chips, ENGINE_BIG_BLIND))
        return True

    def _missing_sit_out_blind(self, reading) -> bool:
        """A player sitting out has a blind in front of them -- or, with an ante, is
        dealt in at all -- but no chair at the table, and nobody has acted yet -- so
        seating them rewrites nothing."""
        from pokerlab.gui.screen_reader import seated_seats

        if self.script or not reading.seats:
            return False
        return any(
            chair_for_client_seat(self.table_size, seat) not in self.layout.chairs
            for seat in seated_seats(reading.seats, reading.bets, antes=self._has_ante())
        )

    def _apply_seating(self, seats: dict[int, str], bets: dict[int, float] | None = None) -> bool:
        """Seat the players the screen shows and stand up the ones it does not,
        on the chairs mapped to client seats only (a chair the user filled by
        hand elsewhere is left alone). Clears the actions if anything moved.
        The button stays where it was if that chair is still occupied. A player
        sitting out with a blind in front of them (`bets`) is seated too."""
        from pokerlab.gui.screen_reader import seated_seats
        from pokerlab.vision.labels import SEAT_EMPTY

        self._seating_read = True
        seated = seated_seats(seats, bets, antes=self._has_ante())
        moved = False
        for seat, state in seats.items():
            chair = chair_for_client_seat(self.table_size, seat)
            if state == SEAT_EMPTY:
                self.stats.forget(self._stats_id(chair))  # whoever sits there next is someone else
            if seat in seated and self.layout.add(chair):
                self.stack_vars[chair].set(DEFAULT_STACK_BB)
                self.name_vars[chair].set(f"Avv. {chair}")
                moved = True
            elif seat not in seated and self.layout.remove(chair):
                moved = True
        if self._last_dealer is not None:
            # `remove` passes the button to the player before if its chair emptied;
            # put it back where it was last seen, if that chair is still occupied.
            self.layout.set_dealer(chair_for_client_seat(self.table_size, self._last_dealer))
        if moved:
            self.script.clear()
        return moved

    def _note_screen_stacks(self, reading) -> bool:
        """Keep the stacks read now (by chair, in BB) to compare with what the
        engine leaves each player; True if they changed, so the chairs redraw."""

        if reading.stacks is None:
            return False
        stacks = {
            chair_for_client_seat(self.table_size, seat): value
            for seat, value in reading.stacks.items()
            if value > 0
        }
        moved = stacks != self._screen_stacks
        self._screen_stacks = stacks
        return moved

    def _apply_stacks(self, reading) -> None:
        """Each seated player's stack at the start of the hand, from the screen:
        the stack shown plus the chips already in front of it (the blinds,
        preflop) plus the ante it has already paid into the pot. Applied with the seating -- at a new hand -- and never mid-hand,
        when a changed starting stack would rewrite a hand already rebuilt. A
        seat whose stack was not read keeps the one it had."""

        self._stacks_from_screen = set()
        if not reading.stacks:
            return
        try:
            ante = self._ante()
        except ValueError:
            ante = 0
        for client_seat, stack in reading.stacks.items():
            chair = chair_for_client_seat(self.table_size, client_seat)
            if chair in self.layout.chairs and stack > 0:
                in_front = (reading.bets or {}).get(client_seat, 0.0)
                chips = round((stack + in_front) * ENGINE_BIG_BLIND) + ante
                self.stack_vars[chair].set(bb_number(chips, ENGINE_BIG_BLIND))
                self._stacks_from_screen.add(chair)

    def _mark_seat_states(self) -> None:
        """Darken the box of every player out of the hand: folded in the rebuilt hand,
        or shown by the screen as out ("fold"), sitting out, or gone ("uscito"), which
        is also tagged -- information only: the FOLD goes into the actions when their
        turn comes (`action_sync`), not here."""
        acting = None
        if self.state is not None and not self.state.finished and self.state.to_act is not None:
            acting = self.layout.chair_of(self.state.to_act)
        if self.state is not None:
            order = self.layout.order()
            for seat, action in self.state.taken:
                if action.action_type is ActionType.FOLD and seat < len(order):
                    box = self.chair_ui[order[seat]].box
                    box.configure(highlightbackground=OUT_OF_HAND)
                    _paint_box(box, folded=True)
        for chair, state in self._seat_states.items():
            if chair not in self.layout.chairs or state not in SEAT_MARKS or chair == acting:
                continue
            ui = self.chair_ui[chair]
            text = ui.info.cget("text")
            ui.info.configure(text=f"{text} · {SEAT_MARKS[state]}" if text else SEAT_MARKS[state])
            ui.box.configure(highlightbackground=OUT_OF_HAND)
            _paint_box(ui.box, folded=True)

    def destroy(self) -> None:
        if self._screen_job is not None:
            self.after_cancel(self._screen_job)
            self._screen_job = None
        super().destroy()

    def _append(self, action: Action) -> None:
        self.script.append(action)
        self.refresh()

    def _undo(self) -> None:
        if self.script:
            self.script.pop()
        self.refresh()

    def _clear(self) -> None:
        self.script.clear()
        self.refresh()

    def _write_log(self, state: SpotState, names: list[str]) -> None:
        lines = []
        for number, (seat, action) in enumerate(state.taken, start=1):
            name = self._player_name(self.layout.chair_of(seat))
            lines.append(f"{number}. {name} ({names[seat]}): {describe_action(action, None, self._big_blind())}")
        self._write(self.log, "\n".join(lines))
        self.log.see("end")  # the latest actions always in view

    @staticmethod
    def _write(widget: tk.Text, text: str, bold_lines: tuple[int, ...] = ()) -> None:
        """Replace the text; the (1-based) `bold_lines` are shown in bold."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        if bold_lines:
            # Two traps, both of which showed the line plain while it was tagged bold:
            # `tkfont.Font(font=..., weight=...)` copies the font and ignores the other
            # options, so the weight is set apart; and a `Font` deletes its Tk font when
            # the Python object is collected, so it is kept on the widget.
            if not hasattr(widget, "bold_font"):
                widget.bold_font = tkfont.Font(font=widget.cget("font"))
                widget.bold_font.configure(weight="bold")
            widget.tag_configure("bold", font=widget.bold_font)
            for line in bold_lines:
                widget.tag_add("bold", f"{line}.0", f"{line}.end")
        widget.configure(state="disabled")

    # -- the models ---------------------------------------------------------


    def _consult(self, state: SpotState, spot: Spot) -> None:
        if not self._load_models():
            return
        try:
            advice = advise(
                state.observation,
                state.legal_actions,
                self.models,
                big_blind=spot.big_blind,
            )
        except Exception as exc:  # noqa: BLE001 - shown, not fatal
            self.status_var.set(f"Errore: {exc}")
            return
        missing = self._bets_not_rebuilt(state)
        self.status_var.set(
            f"ATTENZIONE: {missing}: il consiglio è per una mano diversa da quella sullo schermo."
            if missing else f"{len(advice)} modelli consultati."
        )
        lines = []
        bold_lines: tuple[int, ...] = ()
        if advice:
            # The action to take, alone and in capitals, in bold on the first line: drawn
            # from the best-rated model's probabilities, as it would play at a table, not
            # its likeliest action. `upper`, never `capitalize`, which lowercases "BB".
            chosen = self._sampled_action(advice[0], spot)
            lines += [describe_action(chosen.action, state.observation, spot.big_blind).upper(), ""]
            bold_lines = (1,)
        for index, item in enumerate(advice, start=1):
            # The model by its place in the global ranking (#1 is the best-rated), not by
            # its long label, which only took room.
            rank = self._model_ranks.get(item.label, index)
            # One line a model, its likeliest action; every option only for the first, the
            # one the action is drawn from -- all five in full did not fit the panel.
            lines.append(
                f"#{rank} (elo {item.rating:.0f}) -> "
                f"{describe_action(item.best.action, state.observation, spot.big_blind)}"
                f" {item.best.probability:.0%}"
            )
            if index > 1:
                continue
            # Every option as the action it is in this hand ("check", "bet 2,25 BB", "raise a
            # 6 BB"), not the bin's generic name ("check/call", "50% pot"), which does not
            # say whether it is a check or a call, a bet or a raise.
            lines.append(
                "     "
                + ", ".join(
                    f"{describe_action(b.action, state.observation, spot.big_blind)} {b.probability:.0%}"
                    for b in item.bins if b.probability >= 0.01
                )
            )
        self._write(self.advice, "\n".join(lines), bold_lines)

    def _sampled_action(self, item, spot: Spot):
        """One of the model's bins drawn by its probabilities, once per decision: the advice
        is rewritten at every reading, and a new draw each time would flicker between
        actions. A decision is the hand and the actions so far; a new one draws again."""
        key = (item.label, spot.hole_cards, tuple(spot.board), tuple(spot.script), spot.my_seat)
        if self._draw_key != key or self._draw is None:
            options = [b for b in item.bins if b.legal and b.probability > 0] or [item.best]
            self._draw = self._rng.choices(options, weights=[b.probability for b in options])[0]
            self._draw_key = key
        return self._draw

    def _bets_not_rebuilt(self, state: SpotState) -> str:
        """The bets the screen shows in front of players that the rebuilt hand does not
        have, as text, or "" when they agree. The advice is asked of the engine's hand: a
        bet missing from it turns a call into a check and a raise into a bet."""
        view = self._last_view
        if view is None or state.observation is None:
            return ""
        names = position_names(len(state.stacks))
        big_blind = self._big_blind()
        missing = [
            f"{names[seat]} ha {self._bb(chips)} davanti"
            for seat, chips in sorted(view.bets.items())
            if chips > state.bets.get(seat, 0) + big_blind // 100  # a chip of rounding
        ]
        return "; ".join(missing)

    def _load_models(self) -> bool:
        if self.models is not None:
            return True
        top = discover_global_top_models(limit=DEFAULT_ADVISORS)
        if not top:
            self.status_var.set("Nessun modello addestrato trovato.")
            return False
        self.status_var.set("Carico i modelli...")
        self.update_idletasks()
        try:
            loaded, failed = load_advisors([(label, rating, str(path)) for label, path, rating in top])
        except ImportError:
            self.status_var.set("Serve l'extra rl (torch) per usare i modelli.")
            return False
        if failed:
            messagebox.showwarning(
                "Modelli non caricati", "\n".join(f"{label}: {why}" for label, why in failed)
            )
        if not loaded:
            self.status_var.set("Nessun modello caricabile.")
            return False
        self.models = loaded
        self._model_ranks = {label: rank for rank, (label, _path, _rating) in enumerate(top, start=1)}
        return True
