"""The "collect vision data" screen: zones of the poker client and labelled crops.

Reached from the main menu ("Collect vision data"), on its own so that collecting
examples does not crowd the "ask the models" table. Two zones of the screen -- your
cards and the board -- are chosen by dragging a rectangle over the client
(`vision.selector`); "Salva ..." captures a zone and opens a `CropLabeler`, where
the visible cards are entered left to right with the same card buttons as the spot
table, and only "Conferma" writes the PNG and its label to `vision_data/crops/`.
These are the examples the card recognition is built from. The dealer and player
sections collect their own crops the same way, and every section has an
"Anteprima" showing its zones as captured now. Checking what is *read* is left to
the spot screen, which reads every zone every half second.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from pokerlab.cards.card import Card
from pokerlab.gui.spot import format_card
from pokerlab.gui.spot_view import CardPicker, card_color, card_text, place_near_pointer
from pokerlab.vision.regions import (
    BOARD,
    DEALER_TABLE_SIZES,
    DEALER_ZONE_MAX_SIDE,
    HOLE_CARDS,
    POT,
    TURN_TIMER,
    bet_region_name,
    dealer_region_name,
    load_regions,
    player_region_name,
    save_regions,
    stack_region_name,
)

# The table size whose per-seat zones the screen opens on; the "Tavolo" choice
# at the top switches the seat sections between `DEALER_TABLE_SIZES` (6 and 8),
# each with zones of its own.
DEALER_PLAYERS = 6
# The card zones, in the order their buttons appear.
CARD_ZONES = ((HOLE_CARDS, "Mie carte"), (BOARD, "Board"))
# How many lines one wheel notch scrolls the sections.
WHEEL_LINES = 3
# How a seat state read off the screen is captioned in the preview.
SEAT_STATE_TEXT = {"in_gioco": "in gioco", "fuori": "fuori", "sit_out": "sit-out", "libero": "libero",
                   "reazione": "reazione"}

class CropLabeler(tk.Toplevel):
    """Say which cards a captured crop shows, left to right, then confirm.

    Nothing is written until "Conferma": the PNG and its label are saved
    together, so every crop on disk has a label. "Annulla" (or closing the
    window) writes nothing. The slots are filled in order -- a slot is disabled
    until the one to its left is set -- so the label is always the on-screen
    order with no gaps, which is what makes it usable as ground truth. "Nessuna
    carta visibile" is a real answer (folded, street not dealt) and is confirmed
    the same way. `vision.labels.save_label` refuses a count the zone cannot show
    (one hole card, a two-card board), and Conferma stays disabled until the
    count is valid.
    """

    def __init__(self, master, *, png_path, zone: str, frame=None, on_done=None) -> None:
        from pokerlab.vision.labels import SLOTS, VALID_COUNTS

        super().__init__(master)
        self.png_path = png_path
        self.zone = zone
        self.frame = frame  # the captured pixels, written only on Conferma
        self.cards: list[Card] = []
        self.empty = False  # "Nessuna carta visibile" chosen
        self.result: list[str] | None = None  # the saved label; None if cancelled
        self._on_done = on_done
        self._slots = SLOTS[zone]
        zone_name = {HOLE_CARDS: "le tue carte", BOARD: "il board"}[zone]
        self.title(f"Etichetta: {zone_name}")
        self.transient(master.winfo_toplevel())
        body = ttk.Frame(self, padding=10)
        body.pack()
        self._image = self._load_image(frame, png_path)
        if self._image is not None:
            ttk.Label(body, image=self._image).pack(pady=(0, 6))
        counts = " o ".join(str(n) for n in VALID_COUNTS[zone])
        ttk.Label(
            body, justify="left",
            text=f"Che carte si vedono in {zone_name}? Da sinistra a destra.\n"
            f"Validi: {counts} carte. Poi premi Conferma.",
        ).pack(anchor="w")
        row = ttk.Frame(body)
        row.pack(pady=6)
        self.slot_buttons = [
            tk.Button(row, width=4, font=("TkDefaultFont", 14), command=lambda i=i: self._open(i))
            for i in range(self._slots)
        ]
        for button in self.slot_buttons:
            button.pack(side="left", padx=3)
        self.empty_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            body, text="Nessuna carta visibile", variable=self.empty_var, command=self._toggle_empty
        ).pack(anchor="w")
        self.status = ttk.Label(body, foreground="#b00020")
        self.status.pack(anchor="w")
        buttons = ttk.Frame(body)
        buttons.pack(pady=(6, 0))
        self.confirm_button = ttk.Button(buttons, text="Conferma", command=self.confirm)
        self.confirm_button.pack(side="left", padx=3)
        ttk.Button(buttons, text="Annulla", command=self.cancel).pack(side="left", padx=3)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Return>", lambda _e: self.confirm())
        self._refresh()
        place_near_pointer(self)

    @staticmethod
    def _load_image(frame, png_path):
        """The crop, enlarged by a whole factor if small; None if it cannot be shown."""
        import base64

        try:
            if frame is not None:
                from pokerlab.vision.capture import frame_to_png_bytes

                image = tk.PhotoImage(data=base64.b64encode(frame_to_png_bytes(frame)).decode("ascii"))
            else:
                image = tk.PhotoImage(file=str(png_path))
        except Exception:  # noqa: BLE001 - a preview that fails must not stop the labelling
            return None
        factor = max(1, min(4, 300 // max(1, image.width())))
        return image.zoom(factor) if factor > 1 else image

    def _open(self, slot: int) -> CardPicker | None:
        if self.empty or slot > len(self.cards):
            return None  # filled in order, left to right
        current = self.cards[slot] if slot < len(self.cards) else None
        return CardPicker(
            self, taken=set(self.cards) - {current}, title=f"Carta {slot + 1} da sinistra",
            can_clear=current is not None, on_pick=lambda card: self.pick(slot, card),
        )

    def pick(self, slot: int, card: Card | None) -> None:
        """Set (or, with None, remove) the card in `slot`; later ones shift left."""
        if card is None:
            if slot < len(self.cards):
                del self.cards[slot]
        elif slot == len(self.cards):
            self.cards.append(card)
        elif slot < len(self.cards):
            self.cards[slot] = card
        self._refresh()

    def set_empty(self, empty: bool) -> None:
        """Mark the crop as showing no card (clears any card chosen)."""
        self.empty = empty
        self.empty_var.set(empty)
        if empty:
            self.cards.clear()
        self._refresh()

    def _toggle_empty(self) -> None:
        self.set_empty(self.empty_var.get())

    def _labels(self) -> list[str]:
        return [format_card(card) for card in self.cards]

    def ready(self) -> bool:
        """Whether Conferma would save: no card ticked, or a valid count."""
        from pokerlab.vision.labels import label_error

        return self.empty or (bool(self.cards) and label_error(self.zone, self._labels()) is None)

    def _refresh(self) -> None:
        from pokerlab.vision.labels import label_error

        for index, button in enumerate(self.slot_buttons):
            card = self.cards[index] if index < len(self.cards) else None
            open_slot = not self.empty and index == len(self.cards)
            button.configure(
                text=card_text(card) if card or open_slot else "",
                fg=card_color(card),
                state="normal" if not self.empty and index <= len(self.cards) else "disabled",
            )
        error = label_error(self.zone, self._labels()) if self.cards else None
        self.status.configure(text=error or "")
        self.confirm_button.configure(state="normal" if self.ready() else "disabled")

    def confirm(self) -> None:
        """Write the PNG and its label together, if the label is complete."""
        from pokerlab.vision.labels import save_label

        if not self.ready():
            return
        labels = [] if self.empty else self._labels()
        if self.frame is not None:
            from pokerlab.vision.capture import save_png

            save_png(self.frame, self.png_path)
        save_label(self.png_path, self.zone, labels)
        self.result = labels
        self._close()

    def cancel(self) -> None:
        """Close without writing anything."""
        self.result = None
        self._close()

    def _close(self) -> None:
        if self.winfo_exists():
            self.destroy()
        if self._on_done is not None:
            self._on_done(self)


class SeatLabeler(tk.Toplevel):
    """One crop per seat side by side, each given one of a few states with
    buttons -- the dealer zones (present / absent) and the player boxes (in the
    hand / out / empty) are labelled this same way.

    Like `CropLabeler`, nothing is written until "Conferma", and then every crop
    is written with its label through `write(png_path, zone, value)` -- every
    seat is an example, whatever its state. `check(choices)` returns why the
    choices cannot be saved (Conferma stays disabled), or None. A seat starts on
    `defaults[zone]`, or on nothing, which then has to be chosen."""

    def __init__(self, master, *, shots: list, title: str, question: str, options: list,
                 write, check=None, defaults: dict | None = None, on_done=None) -> None:
        """`shots`: `(seat, zone, frame, png_path)`; `options`: `(value, text)`."""
        super().__init__(master)
        self.shots = shots
        self.result: dict[str, str] | None = None  # zone -> value, once saved
        self._write = write
        self._check = check
        self._on_done = on_done
        self._images = []
        self.choices: dict[str, tk.StringVar] = {}
        self.title(title)
        self.transient(master.winfo_toplevel())
        body = ttk.Frame(self, padding=10)
        body.pack()
        span = max(1, len(shots))
        ttk.Label(body, text=question).grid(row=0, column=0, columnspan=span, sticky="w", pady=(0, 6))
        defaults = defaults or {}
        for column, (seat, zone, frame, _path) in enumerate(shots):
            ttk.Label(body, text=f"Posto {seat}{' (tu)' if seat == 0 else ''}").grid(row=1, column=column)
            image = CropLabeler._load_image(frame, None)
            if image is not None:
                self._images.append(image)
                ttk.Label(body, image=image).grid(row=2, column=column, padx=4)
            var = tk.StringVar(value=defaults.get(zone) or "")
            self.choices[zone] = var
            for row, (value, text) in enumerate(options, start=3):
                ttk.Radiobutton(body, text=text, value=value, variable=var, command=self._refresh).grid(
                    row=row, column=column, sticky="w", padx=4
                )
        below = 3 + len(options)
        self.status = ttk.Label(body, foreground="#b00020")
        self.status.grid(row=below, column=0, columnspan=span, sticky="w", pady=(6, 0))
        buttons = ttk.Frame(body)
        buttons.grid(row=below + 1, column=0, columnspan=span, pady=(6, 0))
        self.confirm_button = ttk.Button(buttons, text="Conferma", command=self.confirm)
        self.confirm_button.pack(side="left", padx=3)
        ttk.Button(buttons, text="Annulla", command=self.cancel).pack(side="left", padx=3)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Return>", lambda _e: self.confirm())
        self._refresh()
        place_near_pointer(self)

    def choose(self, zone: str, value: str) -> None:
        self.choices[zone].set(value)
        self._refresh()

    def _problem(self) -> str | None:
        values = {zone: var.get() for zone, var in self.choices.items()}
        if any(not value for value in values.values()):
            return "Scegli uno stato per ogni posto."
        return self._check(values) if self._check else None

    def _refresh(self) -> None:
        problem = self._problem()
        self.status.configure(text=problem or "")
        self.confirm_button.configure(state="disabled" if problem else "normal")

    def confirm(self) -> None:
        from pokerlab.vision.capture import save_png

        if self._problem():
            return
        result = {}
        for _seat, zone, frame, path in self.shots:
            value = self.choices[zone].get()
            save_png(frame, path)
            self._write(path, zone, value)
            result[zone] = value
        self.result = result
        self._close()

    def cancel(self) -> None:
        self.result = None
        self._close()

    def _close(self) -> None:
        if self.winfo_exists():
            self.destroy()
        if self._on_done is not None:
            self._on_done(self)


def _one_dealer_at_most(values: dict[str, str]) -> str | None:
    if sum(value == "si" for value in values.values()) > 1:
        return "Il dealer è uno solo: segna al massimo un posto."
    return None


class DealerLabeler(SeatLabeler):
    """Every dealer zone marked "Presente" / "Non presente", at most one present
    (none is fine, between hands); written with `labels.save_dealer_label`."""

    def __init__(self, master, *, shots: list, on_done=None) -> None:
        from pokerlab.vision.labels import save_dealer_label

        super().__init__(
            master, shots=shots, title="Dove è il dealer?",
            question="Per ogni posto: c'è il gettone del dealer?",
            options=[("si", "Presente"), ("no", "Non presente")],
            write=lambda path, zone, value: save_dealer_label(path, zone, value == "si"),
            check=_one_dealer_at_most, defaults={zone: "no" for _s, zone, _f, _p in shots},
            on_done=on_done,
        )

    def set_present(self, zone: str, present: bool) -> None:
        self.choose(zone, "si" if present else "no")


class AmountLabeler(tk.Toplevel):
    """The pot and every bet zone side by side, each with a field for the amount
    *as written on screen* ("" when nothing is shown). Like the other labellers,
    nothing is written until "Conferma", and then every crop is, with its label
    (`labels.save_amount_label`); Conferma waits until every field is valid."""

    def __init__(self, master, *, shots: list, on_done=None) -> None:
        """`shots`: `(caption, zone, frame, png_path)` for each captured zone."""
        super().__init__(master)
        self.shots = shots
        self.result: dict[str, str] | None = None  # zone -> text, once saved
        self._on_done = on_done
        self._images = []
        self.values: dict[str, tk.StringVar] = {}
        self.title("Quanto c'è scritto?")
        self.transient(master.winfo_toplevel())
        body = ttk.Frame(self, padding=10)
        body.pack()
        span = max(1, len(shots))
        ttk.Label(
            body, justify="left",
            text="Scrivi ogni importo esattamente come appare (separatori e K compresi); "
            "lascia vuoto se non c'è niente.",
        ).grid(row=0, column=0, columnspan=span, sticky="w", pady=(0, 6))
        first = None
        for column, (caption, zone, frame, _path) in enumerate(shots):
            ttk.Label(body, text=caption).grid(row=1, column=column)
            image = CropLabeler._load_image(frame, None)
            if image is not None:
                self._images.append(image)
                ttk.Label(body, image=image).grid(row=2, column=column, padx=4)
            var = tk.StringVar()
            var.trace_add("write", lambda *_a: self._refresh())
            self.values[zone] = var
            entry = ttk.Entry(body, textvariable=var, width=10, justify="center")
            entry.grid(row=3, column=column, padx=4, pady=(2, 0))
            first = first or entry
        self.status = ttk.Label(body, foreground="#b00020")
        self.status.grid(row=4, column=0, columnspan=span, sticky="w", pady=(6, 0))
        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=span, pady=(6, 0))
        self.confirm_button = ttk.Button(buttons, text="Conferma", command=self.confirm)
        self.confirm_button.pack(side="left", padx=3)
        ttk.Button(buttons, text="Annulla", command=self.cancel).pack(side="left", padx=3)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Return>", lambda _e: self.confirm())
        self._refresh()
        place_near_pointer(self)
        if first is not None:
            first.focus_set()

    def set_value(self, zone: str, text: str) -> None:
        self.values[zone].set(text)

    def _problem(self) -> str | None:
        from pokerlab.vision.labels import amount_error

        for caption, zone, _frame, _path in self.shots:
            error = amount_error(self.values[zone].get().strip())
            if error:
                return f"{caption}: {error}"
        return None

    def _refresh(self) -> None:
        problem = self._problem()
        self.status.configure(text=problem or "")
        self.confirm_button.configure(state="disabled" if problem else "normal")

    def confirm(self) -> None:
        from pokerlab.vision.capture import save_png
        from pokerlab.vision.labels import save_amount_label

        if self._problem():
            return
        result = {}
        for _caption, zone, frame, path in self.shots:
            text = self.values[zone].get().strip()
            save_png(frame, path)
            save_amount_label(path, zone, text)
            result[zone] = text
        self.result = result
        self._close()

    def cancel(self) -> None:
        self.result = None
        self._close()

    def _close(self) -> None:
        if self.winfo_exists():
            self.destroy()
        if self._on_done is not None:
            self._on_done(self)


class VisionFrame(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=12)
        self.app = master
        self.players = DEALER_PLAYERS  # the table size the seat sections collect
        self.players_size_var = tk.IntVar(value=self.players)
        self.regions = load_regions()
        self.vision_var = tk.StringVar()
        self.card_zone_buttons: dict[str, tuple[str, ttk.Button]] = {}
        self._preview_images: list[tk.PhotoImage] = []  # Tk drops an image nobody holds
        self.labeler: CropLabeler | None = None
        self.dealer_var = tk.StringVar()
        self.dealer_zone_buttons: list[ttk.Button] = []
        self.dealer_labeler: DealerLabeler | None = None
        self.players_var = tk.StringVar()
        self.player_zone_buttons: list[ttk.Button] = []
        self.players_labeler: SeatLabeler | None = None
        # The state last given to each seat: the next capture starts from it,
        # since between two captures usually only a seat or two has changed.
        self.last_seat_states: dict[int, str] = {}
        self.amounts_var = tk.StringVar()
        self.amount_zone_buttons: dict[str, tuple[str, ttk.Button]] = {}
        self.amounts_labeler: AmountLabeler | None = None
        self.turn_var = tk.StringVar()
        self.stacks_var = tk.StringVar()
        self.stack_zone_buttons: dict[str, tuple[str, ttk.Button]] = {}
        self.stacks_labeler: AmountLabeler | None = None
        self.turn_labeler: SeatLabeler | None = None
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", pady=(0, 8))
        ttk.Label(top, text="Collect vision data", font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Button(top, text="Torna al menu", command=self.app.show_setup).pack(side="right")
        # The sections outgrow a window, so they sit in a canvas that scrolls,
        # with the wheel anywhere over them (`_on_wheel`), the title staying put.
        holder = ttk.Frame(self)
        holder.pack(side="top", fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(holder, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(holder, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.scroll_canvas)
        body_item = self.scroll_canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>",
            lambda _e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")),
        )
        self.scroll_canvas.bind(
            "<Configure>", lambda e: self.scroll_canvas.itemconfigure(body_item, width=e.width)
        )
        self._build_vision(self.body)
        # The per-seat sections (dealer, players, bets, stacks) belong to one
        # table size; choosing another rebuilds them on that size's zones.
        size_row = ttk.Frame(self.body)
        size_row.pack(fill="x", pady=(12, 0))
        ttk.Label(size_row, text="Tavolo:", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(0, 6))
        for size in DEALER_TABLE_SIZES:
            ttk.Radiobutton(
                size_row, text=f"{size} giocatori", value=size, variable=self.players_size_var,
                command=lambda: self.set_players(self.players_size_var.get()),
            ).pack(side="left", padx=(0, 6))
        self.seat_sections = ttk.Frame(self.body)
        self.seat_sections.pack(fill="x")
        self._build_seat_sections()
        self._build_turn(self.body)
        # Bound for the whole application, because the wheel event goes to the
        # widget under the pointer -- a button, a label -- not to the canvas;
        # `_on_wheel` ignores anything outside this screen, and `destroy`
        # unbinds, so no other screen inherits it.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(sequence, self._on_wheel, add="+")

    def _build_seat_sections(self) -> None:
        for child in self.seat_sections.winfo_children():
            child.destroy()
        self._build_dealer(self.seat_sections)
        self._build_players(self.seat_sections)
        self._build_amounts(self.seat_sections)
        self._build_stacks(self.seat_sections)

    def set_players(self, size: int) -> bool:
        """Show the seat sections of a `size`-seat table (6 or 8). The zones of
        the other size stay saved; only which ones these sections set changes.
        False if nothing changed."""
        if size not in DEALER_TABLE_SIZES or size == self.players:
            self.players_size_var.set(self.players)
            return False
        self.players = size
        self.players_size_var.set(size)
        self.last_seat_states = {}  # another table: other seats
        self._build_seat_sections()
        return True

    def _on_wheel(self, event) -> str | None:
        """Scroll the sections, if the pointer is over this screen. Windows and
        macOS send <MouseWheel> with a signed delta, X11 Button-4 (up) and
        Button-5 (down) -- each read from its own sequence, not from
        `event.num`, which is not dependable (see the spot screen)."""
        widget = event.widget
        if not isinstance(widget, tk.Misc) or not str(widget).startswith(str(self)):
            return None  # a popup, another screen: not ours to scroll
        if widget.winfo_toplevel() is not self.winfo_toplevel():
            return None
        if getattr(event, "delta", 0):
            step = -1 if event.delta > 0 else 1
        else:
            step = -1 if event.num == 4 else 1
        self.scroll_canvas.yview_scroll(step * WHEEL_LINES, "units")
        return "break"

    def destroy(self) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.unbind_all(sequence)
        super().destroy()

    # -- the screen ---------------------------------------------------------

    def _build_vision(self, parent: ttk.Frame) -> None:
        """The card zones, laid out like every other section: a row of zone
        buttons (ticked once set), then the actions with the status beside them."""
        box = ttk.LabelFrame(parent, text="Carte (mano e board)", padding=6)
        box.pack(fill="x", pady=(8, 0))
        zones = ttk.Frame(box)
        zones.pack(fill="x")
        self.card_zone_buttons = {}
        for zone, caption in CARD_ZONES:
            button = ttk.Button(zones, width=11, command=lambda z=zone: self._set_zone(z))
            button.pack(side="left", padx=(0, 4))
            self.card_zone_buttons[zone] = (caption, button)
        actions = ttk.Frame(box)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Button(actions, text="Anteprima", command=self._preview).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Salva carte", command=lambda: self._save_crops([HOLE_CARDS])).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(actions, text="Salva board", command=lambda: self._save_crops([BOARD])).pack(side="left")
        ttk.Label(actions, textvariable=self.vision_var, justify="left", wraplength=440).pack(
            side="left", padx=(8, 0)
        )
        self._update_vision_text()

    def _update_vision_text(self, extra: str = "") -> None:
        missing = []
        for zone, (caption, button) in self.card_zone_buttons.items():
            is_set = self.regions.get(zone) is not None
            button.configure(text=f"{'✓ ' if is_set else ''}{caption}")
            if not is_set:
                missing.append(caption)
        status = f"zone mancanti: {', '.join(missing)}" if missing else "tutte le zone impostate"
        self.vision_var.set(status + (f" | {extra}" if extra else ""))

    def _set_zone(self, name: str) -> None:
        """Drag a rectangle over the client; it is remembered between sessions."""
        caption = dict(CARD_ZONES).get(name, name)
        self._select_and_save(name, self._update_vision_text, label=f"ZONA {caption.upper()}")
        self._update_vision_text()

    def _select_and_save(self, name: str, report, label: str | None = None) -> bool:
        """The rectangle-drag for any zone; `report` shows the outcome, `label`
        is written on the overlay so it is plain which zone is being set."""
        from pokerlab.vision.capture import VisionUnavailable
        from pokerlab.vision.selector import select_region

        try:
            region = select_region(self, label=label)
        except VisionUnavailable as exc:
            report(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - a screen grab can fail in many ways
            report(f"Cattura non riuscita: {exc}")
            return False
        if region is None:
            report("Selezione annullata.")
            return False
        self.regions.set(name, region)
        save_regions(self.regions)
        report("Zona salvata.")
        return True

    # -- stacks -------------------------------------------------------------

    def _stack_zones(self) -> list[tuple[str, str]]:
        """`(caption, zone)` for every seat's stack."""
        return [
            (f"Stack {seat}{' (tu)' if seat == 0 else ''}", stack_region_name(self.players, seat))
            for seat in range(self.players)
        ]

    def _build_stacks(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text=f"Stack ({self.players} giocatori)", padding=6)
        box.pack(fill="x", pady=(12, 0))
        zones = ttk.Frame(box)
        zones.pack(fill="x")
        self.stack_zone_buttons = {}
        for caption, zone in self._stack_zones():
            button = ttk.Button(zones, width=11, command=lambda z=zone, c=caption: self._set_stack_zone(z, c))
            button.pack(side="left", padx=(0, 4))
            self.stack_zone_buttons[zone] = (caption, button)
        actions = ttk.Frame(box)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Button(actions, text="Anteprima", command=self._preview_stacks).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Salva stack", command=self._save_stacks).pack(side="left")
        ttk.Label(actions, textvariable=self.stacks_var, justify="left", wraplength=440).pack(
            side="left", padx=(8, 0)
        )
        self._update_stacks_text()

    def _update_stacks_text(self, extra: str = "") -> None:
        missing = []
        for zone, (caption, button) in self.stack_zone_buttons.items():
            is_set = self.regions.get(zone) is not None
            button.configure(text=f"{'✓ ' if is_set else ''}{caption.replace(' (tu)', '')}")
            if not is_set:
                missing.append(caption.replace(" (tu)", ""))
        status = f"zone mancanti: {', '.join(missing)}" if missing else "tutte le zone impostate"
        self.stacks_var.set(status + (f" | {extra}" if extra else ""))

    def _set_stack_zone(self, zone: str, caption: str) -> None:
        self._select_and_save(zone, self._update_stacks_text, label=caption.upper())
        self._update_stacks_text()

    def _stack_frames(self) -> list[tuple[str, str, object]]:
        wanted = [(caption, zone) for caption, zone in self._stack_zones() if self.regions.get(zone)]
        frames = self._capture_zones([zone for _caption, zone in wanted])
        return [(caption, zone, frames[zone]) for caption, zone in wanted if zone in frames]

    def _preview_stacks(self) -> tk.Toplevel | None:
        shots = self._stack_frames()
        if not shots:
            self._update_stacks_text("imposta prima almeno una zona")
            return None
        try:
            from pokerlab.vision.amounts import build_reader, load_amounts
            from pokerlab.vision.labels import AMOUNTS_DIR, STACKS_DIR
        except ImportError:
            def describe(_frame) -> str:
                return ""
        else:
            # Bets and stacks share the client's digits: both lend examples.
            reader = build_reader(load_amounts(AMOUNTS_DIR) + load_amounts(STACKS_DIR))

            def describe(frame, zone) -> str:
                got = reader.read(frame, zone)
                if got is None:
                    return "non leggibile"
                return f"{got.text} BB" if got.text else "nessuno"
        return self._show_preview(
            "Anteprima: stack", [(caption, frame, describe(frame, zone)) for caption, zone, frame in shots],
            columns=len(shots),
        )

    def _save_stacks(self) -> AmountLabeler | None:
        """Capture every stack zone set, and ask what each one says."""
        import time

        from pokerlab.vision.labels import STACKS_DIR

        captured = self._stack_frames()
        if not captured:
            self._update_stacks_text("imposta prima almeno una zona")
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shots = [(caption, zone, frame, STACKS_DIR / f"{zone}-{stamp}.png") for caption, zone, frame in captured]

        def done(labeler: AmountLabeler) -> None:
            if self.winfo_exists():
                saved = "non salvato" if labeler.result is None else f"salvati {len(labeler.result)} ritagli"
                self._update_stacks_text(saved)

        self.stacks_labeler = AmountLabeler(self, shots=shots, on_done=done)
        return self.stacks_labeler

    # -- your turn (the countdown bar) --------------------------------------

    def _build_turn(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Mio turno (barra del tempo)", padding=6)
        box.pack(fill="x", pady=(12, 0))
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.turn_zone_button = ttk.Button(row, width=18, command=self._set_turn_zone)
        self.turn_zone_button.pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Anteprima", command=self._preview_turn).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Salva turno", command=self._save_turn).pack(side="left")
        ttk.Label(row, textvariable=self.turn_var, justify="left", wraplength=360).pack(
            side="left", padx=(8, 0)
        )
        self._update_turn_text()

    def _update_turn_text(self, extra: str = "") -> None:
        is_set = self.regions.get(TURN_TIMER) is not None
        self.turn_zone_button.configure(text=f"{'✓ ' if is_set else ''}Zona barra del tempo")
        status = "zona impostata" if is_set else "zona mancante"
        self.turn_var.set(status + (f" | {extra}" if extra else ""))

    def _set_turn_zone(self) -> None:
        self._select_and_save(TURN_TIMER, self._update_turn_text, label="BARRA DEL TEMPO (il tuo turno)")
        self._update_turn_text()

    def _preview_turn(self) -> tk.Toplevel | None:
        frames = self._capture_zones([TURN_TIMER]) if self.regions.get(TURN_TIMER) else {}
        if not frames:
            self._update_turn_text("imposta prima la zona")
            return None
        frame = frames[TURN_TIMER]
        try:
            from pokerlab.vision.turn import TurnReader, load_turns
        except ImportError:
            reading = ""
        else:
            got = TurnReader(load_turns()).read(frame)
            reading = {True: "TOCCA A TE", False: "non è il tuo turno", None: "servono esempi di entrambi"}[got]
        return self._show_preview("Anteprima: barra del tempo", [("Barra del tempo", frame, reading)], columns=1)

    def _save_turn(self) -> SeatLabeler | None:
        """Capture the timer zone and ask whether the bar is showing."""
        import time

        from pokerlab.vision.labels import TURN_DIR, save_turn_label

        frames = self._capture_zones([TURN_TIMER]) if self.regions.get(TURN_TIMER) else {}
        if not frames:
            self._update_turn_text("imposta prima la zona")
            return None
        path = TURN_DIR / f"{TURN_TIMER}-{time.strftime('%Y%m%d-%H%M%S')}.png"

        def done(labeler: SeatLabeler) -> None:
            if self.winfo_exists():
                saved = "non salvato" if labeler.result is None else "salvato"
                self._update_turn_text(saved)

        self.turn_labeler = SeatLabeler(
            self, shots=[(0, TURN_TIMER, frames[TURN_TIMER], path)],
            title="C'è la barra del tempo?", question="È il tuo turno (barra del tempo visibile)?",
            options=[("si", "Presente"), ("no", "Non presente")],
            write=lambda png, zone, value: save_turn_label(png, zone, value == "si"),
            on_done=done,
        )
        return self.turn_labeler

    # -- bets and the pot ---------------------------------------------------

    def _amount_zones(self) -> list[tuple[str, str]]:
        """`(caption, zone)`: the pot first, then every seat's bet."""
        return [("Piatto", POT)] + [
            (f"Puntata {seat}{' (tu)' if seat == 0 else ''}", bet_region_name(self.players, seat))
            for seat in range(self.players)
        ]

    def _build_amounts(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text=f"Puntate e piatto ({self.players} giocatori)", padding=6)
        box.pack(fill="x", pady=(12, 0))
        zones = ttk.Frame(box)
        zones.pack(fill="x", pady=(4, 0))
        self.amount_zone_buttons = {}
        for caption, zone in self._amount_zones():
            button = ttk.Button(zones, width=11, command=lambda z=zone, c=caption: self._set_amount_zone(z, c))
            button.pack(side="left", padx=(0, 4))
            self.amount_zone_buttons[zone] = (caption, button)
        actions = ttk.Frame(box)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Button(actions, text="Anteprima", command=self._preview_amounts).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Salva puntate", command=self._save_amounts).pack(side="left")
        ttk.Label(actions, textvariable=self.amounts_var, justify="left", wraplength=440).pack(
            side="left", padx=(8, 0)
        )
        self._update_amounts_text()

    def _update_amounts_text(self, extra: str = "") -> None:
        missing = []
        for zone, (caption, button) in self.amount_zone_buttons.items():
            is_set = self.regions.get(zone) is not None
            button.configure(text=f"{'✓ ' if is_set else ''}{caption.replace(' (tu)', '')}")
            if not is_set:
                missing.append(caption.replace(" (tu)", ""))
        status = f"zone mancanti: {', '.join(missing)}" if missing else "tutte le zone impostate"
        self.amounts_var.set(status + (f" | {extra}" if extra else ""))

    def _set_amount_zone(self, zone: str, caption: str) -> None:
        self._select_and_save(zone, self._update_amounts_text, label=caption.upper())
        self._update_amounts_text()

    def _amount_frames(self) -> list[tuple[str, str, object]]:
        """`(caption, zone, frame)` for every amount zone set, pot first."""
        wanted = [(caption, zone) for caption, zone in self._amount_zones() if self.regions.get(zone)]
        frames = self._capture_zones([zone for _caption, zone in wanted])
        return [(caption, zone, frames[zone]) for caption, zone in wanted if zone in frames]

    def _preview_amounts(self) -> tk.Toplevel | None:
        shots = self._amount_frames()
        if not shots:
            self._update_amounts_text("imposta prima almeno una zona")
            return None
        try:
            from pokerlab.vision.amounts import build_reader, load_amounts
        except ImportError:
            def describe(_frame) -> str:
                return ""
        else:
            reader = build_reader(load_amounts())

            def describe(frame) -> str:
                got = reader.read(frame)
                if got is None:
                    return "non leggibile"
                return f"{got.text} BB" if got.text else "nessuna"
        return self._show_preview(
            "Anteprima: puntate e piatto",
            [(caption, frame, describe(frame)) for caption, _zone, frame in shots],
            columns=len(shots),
        )

    def _save_amounts(self) -> AmountLabeler | None:
        """Capture the pot and every bet zone set, and ask what each says."""
        import time

        from pokerlab.vision.labels import AMOUNTS_DIR

        captured = self._amount_frames()
        if not captured:
            self._update_amounts_text("imposta prima almeno una zona")
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shots = [(caption, zone, frame, AMOUNTS_DIR / f"{zone}-{stamp}.png") for caption, zone, frame in captured]

        def done(labeler: AmountLabeler) -> None:
            if self.winfo_exists():
                saved = "non salvato" if labeler.result is None else f"salvati {len(labeler.result)} ritagli"
                self._update_amounts_text(saved)

        self.amounts_labeler = AmountLabeler(self, shots=shots, on_done=done)
        return self.amounts_labeler

    # -- the players' seats -------------------------------------------------

    def _build_players(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text=f"Giocatori ({self.players} giocatori)", padding=6)
        box.pack(fill="x", pady=(12, 0))
        zones = ttk.Frame(box)
        zones.pack(fill="x", pady=(4, 0))
        self.player_zone_buttons = []
        for seat in range(self.players):
            button = ttk.Button(zones, width=11, command=lambda s=seat: self._set_player_zone(s))
            button.pack(side="left", padx=(0, 4))
            self.player_zone_buttons.append(button)
        actions = ttk.Frame(box)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Button(actions, text="Anteprima", command=self._preview_players).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Salva giocatori", command=self._save_players).pack(side="left")
        ttk.Label(actions, textvariable=self.players_var, justify="left", wraplength=440).pack(
            side="left", padx=(8, 0)
        )
        self._update_players_text()

    def _update_players_text(self, extra: str = "") -> None:
        missing = []
        for seat, button in enumerate(self.player_zone_buttons):
            is_set = self.regions.get(player_region_name(self.players, seat)) is not None
            button.configure(text=f"{'✓ ' if is_set else ''}Giocatore {seat}{' (tu)' if seat == 0 else ''}")
            if not is_set:
                missing.append(str(seat))
        status = f"zone mancanti: posti {', '.join(missing)}" if missing else "tutte le zone impostate"
        self.players_var.set(status + (f" | {extra}" if extra else ""))

    def _set_player_zone(self, seat: int) -> None:
        self._select_and_save(
            player_region_name(self.players, seat), self._update_players_text,
            label=f"RIQUADRO GIOCATORE, posto {seat}",
        )
        self._update_players_text()

    def _save_players(self) -> SeatLabeler | None:
        """Capture every player zone set and ask the state of each seat."""
        import time

        from pokerlab.vision.labels import (
            PLAYERS_DIR,
            SEAT_EMPTY,
            SEAT_IN_HAND,
            SEAT_OUT,
            SEAT_REACTION,
            SEAT_SIT_OUT,
            save_player_label,
        )

        names = [player_region_name(self.players, seat) for seat in range(self.players)]
        frames = self._capture_zones([n for n in names if self.regions.get(n) is not None])
        if not frames:
            self._update_players_text("imposta prima almeno una zona")
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shots = sorted(
            (int(name.rsplit("_", 1)[1]), name, frame, PLAYERS_DIR / f"{name}-{stamp}.png")
            for name, frame in frames.items()
        )

        def done(labeler: SeatLabeler) -> None:
            if labeler.result is not None:
                for seat, zone, _frame, _path in labeler.shots:
                    self.last_seat_states[seat] = labeler.result[zone]
            if self.winfo_exists():
                saved = "non salvato" if labeler.result is None else f"salvati {len(labeler.result)} ritagli"
                self._update_players_text(saved)

        self.players_labeler = SeatLabeler(
            self, shots=shots, title="Stato dei giocatori",
            question="Per ogni posto: com'è il giocatore?",
            options=[(SEAT_IN_HAND, "In gioco"), (SEAT_OUT, "Fuori"), (SEAT_SIT_OUT, "Sit-out"),
                     (SEAT_EMPTY, "Libero"), (SEAT_REACTION, "Reazione")],
            write=save_player_label,
            defaults={zone: self.last_seat_states.get(seat) for seat, zone, _f, _p in shots},
            on_done=done,
        )
        return self.players_labeler

    # -- the dealer button --------------------------------------------------

    def _build_dealer(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text=f"Dealer ({self.players} giocatori)", padding=6)
        box.pack(fill="x", pady=(12, 0))
        zones = ttk.Frame(box)
        zones.pack(fill="x", pady=(4, 0))
        self.dealer_zone_buttons = []
        for seat in range(self.players):
            button = ttk.Button(zones, width=11, command=lambda s=seat: self._set_dealer_zone(s))
            button.pack(side="left", padx=(0, 4))
            self.dealer_zone_buttons.append(button)
        actions = ttk.Frame(box)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Button(actions, text="Anteprima", command=self._preview_dealer).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Salva dealer", command=self._save_dealer).pack(side="left")
        ttk.Label(actions, textvariable=self.dealer_var, justify="left", wraplength=440).pack(
            side="left", padx=(8, 0)
        )
        self._update_dealer_text()

    def _update_dealer_text(self, extra: str = "") -> None:
        missing, too_big = [], []
        for seat, button in enumerate(self.dealer_zone_buttons):
            region = self.regions.get(dealer_region_name(self.players, seat))
            button.configure(text=f"{'✓ ' if region else ''}Gettone {seat}{' (tu)' if seat == 0 else ''}")
            if region is None:
                missing.append(str(seat))
            elif max(region.width, region.height) > DEALER_ZONE_MAX_SIDE:
                too_big.append(f"{seat} ({region.width}x{region.height})")
        status = f"zone mancanti: posti {', '.join(missing)}" if missing else "tutte le zone impostate"
        if too_big:
            status += (f" | zona troppo grande per un gettone, posto {', '.join(too_big)}: "
                       "rifalla stretta attorno al gettone")
        self.dealer_var.set(status + (f" | {extra}" if extra else ""))

    def _set_dealer_zone(self, seat: int) -> None:
        self._select_and_save(
            dealer_region_name(self.players, seat), self._update_dealer_text,
            label=f"GETTONE DEALER, posto {seat} (solo il gettone)",
        )
        self._update_dealer_text()

    def _save_dealer(self) -> DealerLabeler | None:
        """Capture every dealer zone set and ask which one holds the button."""
        import time

        from pokerlab.vision.labels import DEALER_DIR

        names = [dealer_region_name(self.players, seat) for seat in range(self.players)]
        frames = self._capture_zones([n for n in names if self.regions.get(n) is not None])
        if not frames:
            self._update_dealer_text("imposta prima almeno una zona")
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shots = [
            (int(name.rsplit("_", 1)[1]), name, frame, DEALER_DIR / f"{name}-{stamp}.png")
            for name, frame in frames.items()
        ]

        def done(labeler: DealerLabeler) -> None:
            if self.winfo_exists():
                saved = "non salvato" if labeler.result is None else f"salvati {len(labeler.result)} ritagli"
                self._update_dealer_text(saved)

        self.dealer_labeler = DealerLabeler(self, shots=sorted(shots), on_done=done)
        return self.dealer_labeler

    def _capture_zones(self, names: list[str] | None = None) -> dict:
        """A frame per configured zone (only `names`, if given), or an empty
        dict with the reason shown."""
        from pokerlab.vision.capture import VisionUnavailable, grab_region

        zones = {
            name: region for name, region in self.regions.regions.items()
            if names is None or name in names
        }
        if not zones:
            if names is not None and len(names) == 1:
                label = {HOLE_CARDS: "mie carte", BOARD: "board"}.get(names[0], names[0])
                self._update_vision_text(f"Imposta prima la zona {label}.")
            else:
                self._update_vision_text("Imposta prima almeno una zona.")
            return {}
        frames = {}
        try:
            for name, region in zones.items():
                frames[name] = grab_region(region)
        except VisionUnavailable as exc:
            self._update_vision_text(str(exc))
            return {}
        except Exception as exc:  # noqa: BLE001
            self._update_vision_text(f"Cattura non riuscita: {exc}")
            return {}
        return frames

    def _preview(self) -> tk.Toplevel | None:
        """What the program sees in the card zones right now."""
        frames = self._capture_zones([HOLE_CARDS, BOARD])
        if not frames:
            return None
        names = {HOLE_CARDS: "Mano", BOARD: "Board"}
        items = [(names[zone], frame, "") for zone, frame in frames.items()]
        self._update_vision_text(f"{len(frames)} zone catturate.")
        return self._show_preview("Anteprima: carte", items, columns=1)

    def _preview_dealer(self) -> tk.Toplevel | None:
        """Every dealer zone set, side by side, with its share of gold now."""
        names = [dealer_region_name(self.players, seat) for seat in range(self.players)]
        frames = self._capture_zones([n for n in names if self.regions.get(n) is not None])
        if not frames:
            self._update_dealer_text("imposta prima almeno una zona")
            return None
        try:
            from pokerlab.vision.dealer import PRESENT_FRACTION, gold_fraction
        except ImportError:
            def describe(_frame) -> str:
                return ""
        else:
            def describe(frame) -> str:
                share = gold_fraction(frame)
                return f"{'GETTONE' if share >= PRESENT_FRACTION else 'vuoto'} (oro {share:.0%})"
        items = [
            (f"Gettone {name.rsplit('_', 1)[1]}", frame, describe(frame))
            for name, frame in sorted(frames.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1]))
        ]
        return self._show_preview("Anteprima: gettone dealer", items, columns=self.players)

    def _preview_players(self) -> tk.Toplevel | None:
        """Every player zone set, side by side, with the state read now."""
        names = [player_region_name(self.players, seat) for seat in range(self.players)]
        frames = self._capture_zones([n for n in names if self.regions.get(n) is not None])
        if not frames:
            self._update_players_text("imposta prima almeno una zona")
            return None
        try:
            from pokerlab.vision.seats import (
                empty_backgrounds,
                load_seats,
                reaction_thumbnails,
                read_seat,
                sit_out_templates,
            )
        except ImportError:
            def describe(_seat, _name, _frame) -> str:
                return ""
        else:
            labelled = load_seats()
            templates = sit_out_templates(labelled)
            backgrounds = empty_backgrounds(labelled)
            reactions = reaction_thumbnails(labelled)

            def describe(seat, name, frame) -> str:
                state = read_seat(frame, seat, templates, backgrounds.get(name), reactions).state
                return SEAT_STATE_TEXT.get(state, "")
        items = []
        for name, frame in sorted(frames.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1])):
            seat = int(name.rsplit("_", 1)[1])
            items.append((f"Giocatore {seat}", frame, describe(seat, name, frame)))
        return self._show_preview("Anteprima: giocatori", items, columns=self.players)

    def _show_preview(self, title: str, items: list, *, columns: int) -> tk.Toplevel:
        """A window of `(caption, frame, reading)` tiles, `columns` to a row.
        The images are kept on the frame: Tk drops an image nobody holds."""
        import base64

        from pokerlab.vision.capture import frame_to_png_bytes

        window = tk.Toplevel(self)
        window.title(title)
        self._preview_images = []
        for index, (caption, frame, reading) in enumerate(items):
            tile = ttk.Frame(window, padding=6)
            tile.grid(row=index // columns, column=index % columns, sticky="n")
            ttk.Label(tile, text=caption, font=("TkDefaultFont", 10, "bold")).pack()
            try:
                image = tk.PhotoImage(data=base64.b64encode(frame_to_png_bytes(frame)).decode("ascii"))
            except Exception:  # noqa: BLE001 - the caption and reading still show
                image = None
            if image is not None:
                self._preview_images.append(image)
                ttk.Label(tile, image=image).pack(pady=2)
            if reading:
                ttk.Label(tile, text=reading).pack()
        place_near_pointer(window)
        return window

    def _save_crops(self, names: list[str] | None = None) -> None:
        """Capture the zones (only `names`, if given) and ask for their cards.

        The pictures are held in memory: each one, with its label, is written
        to `CROPS_DIR` only when its window is confirmed -- the examples the
        recognition is built from."""
        import time

        from pokerlab.vision.labels import CROPS_DIR

        # Card zones only: the dealer zones live in the same regions file and
        # have their own section, labels and folder.
        frames = self._capture_zones(names if names is not None else [HOLE_CARDS, BOARD])
        if not frames:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        pending = [(name, frame, CROPS_DIR / f"{name}-{stamp}.png") for name, frame in frames.items()]
        self._label_crops(pending)

    def _label_crops(self, pending: list, saved: int = 0) -> CropLabeler | None:
        """One labelling window per captured crop, one after the other."""
        from pokerlab.vision.labels import CROPS_DIR

        if not pending:
            return None
        (zone, frame, path), rest = pending[0], pending[1:]

        def done(labeler: CropLabeler) -> None:
            count = saved + (labeler.result is not None)
            self._update_vision_text(f"Ritagli salvati: {count} in {CROPS_DIR}")
            if self.winfo_exists():
                self._label_crops(rest, count)

        self.labeler = CropLabeler(self, png_path=path, zone=zone, frame=frame, on_done=done)
        return self.labeler
