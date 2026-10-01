"""The "ask the models" screen: a table you fill in with buttons.

An oval table with nine chairs round it, yours at the bottom. Press "+" on a chair
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

import tkinter as tk
from tkinter import messagebox, ttk
from types import SimpleNamespace

from pokerlab.cards.card import Card
from pokerlab.cli.play import discover_global_top_models
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.state import PlayerStatus
from pokerlab.gui.spot import (
    DEFAULT_ADVISORS,
    Spot,
    SpotState,
    advise,
    describe_action,
    format_card,
    load_advisors,
    parse_card,
    position_names,
    replay,
)
from pokerlab.gui.spot_table import (
    CHAIRS,
    USER_CHAIR,
    TableLayout,
    chair_position,
)

# One wheel notch over a bet or raise amount moves it by this many big blinds.
WHEEL_STEP_BIG_BLINDS = 1

CANVAS_WIDTH = 860
CANVAS_HEIGHT = 650
CENTER = (CANVAS_WIDTH / 2, 300)
# Where the chairs sit, and the (smaller) felt inside them.
CHAIR_RADII = (340, 225)
TABLE_RADII = (270, 135)

FELT = "#1f6b3a"
RAIL = "#5b3a1a"
HIGHLIGHT = "#ffd43b"
DEALER_BUTTON = "#f2c94c"

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
    def __init__(self, master) -> None:
        super().__init__(master, padding=8)
        self.app = master
        self.layout = TableLayout()
        self.script: list[Action] = []
        self.state: SpotState | None = None
        self.models: list | None = None  # loaded lazily, once
        self.hole: list[Card | None] = [None, None]
        self.board: list[Card] = []

        self.stack_var = tk.StringVar(value="200")
        self.sb_var = tk.StringVar(value="1")
        self.bb_var = tk.StringVar(value="2")
        self.name_vars = [
            tk.StringVar(value="Tu" if chair == USER_CHAIR else f"Avv. {chair}")
            for chair in range(CHAIRS)
        ]
        self.stack_vars = [tk.StringVar(value="200") for _ in range(CHAIRS)]
        self._default_stack = "200"
        self.status_var = tk.StringVar()
        self.chair_ui: dict[int, SimpleNamespace] = {}
        self.board_buttons: list[tk.Button] = []
        self.hole_buttons: list[tk.Button] = []

        self._build()
        self.refresh()

    # -- layout -------------------------------------------------------------

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", pady=(0, 6))
        for label, var in (("Small blind", self.sb_var), ("Big blind", self.bb_var)):
            ttk.Label(top, text=label).pack(side="left", padx=(0, 3))
            entry = ttk.Entry(top, textvariable=var, width=6)
            entry.pack(side="left", padx=(0, 10))
            entry.bind("<Return>", lambda _e: self.refresh())
        ttk.Label(top, text="Stack di partenza").pack(side="left", padx=(0, 3))
        stack = ttk.Entry(top, textvariable=self.stack_var, width=8)
        stack.pack(side="left", padx=(0, 10))
        stack.bind("<Return>", lambda _e: self._default_stack_changed())
        ttk.Button(top, text="Annulla ultima azione", command=self._undo).pack(side="left", padx=3)
        ttk.Button(top, text="Azzera azioni", command=self._clear).pack(side="left", padx=3)
        ttk.Button(top, text="Torna al menu", command=self.app.show_setup).pack(side="right")

        main = ttk.Frame(self)
        main.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(
            main, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, highlightthickness=0, bg="#2b2b2b"
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
        ttk.Label(side, text="Azioni").pack(anchor="w")
        self.log = tk.Text(side, height=8, width=36, state="disabled")
        self.log.pack(fill="x")
        advisors = ttk.LabelFrame(side, text=f"Consiglio dei top {DEFAULT_ADVISORS}", padding=6)
        advisors.pack(fill="both", expand=True, pady=(8, 0))
        self.ask_button = ttk.Button(advisors, text="Chiedi ai modelli", command=self._ask)
        self.ask_button.pack(anchor="w")
        ttk.Label(advisors, textvariable=self.status_var, wraplength=300).pack(anchor="w", pady=2)
        self.advice = tk.Text(advisors, width=36, state="disabled", wrap="word")
        self.advice.pack(fill="both", expand=True)

    def _draw_table(self) -> None:
        cx, cy = CENTER
        a, b = TABLE_RADII
        self.canvas.create_oval(cx - a, cy - b, cx + a, cy + b, fill=FELT, outline=RAIL, width=10)
        self.pot_item = self.canvas.create_text(
            cx, cy - 62, text="", fill="white", font=("TkDefaultFont", 12, "bold")
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
            cx, cy + 58, text="", fill="white", font=("TkDefaultFont", 10)
        )

    def _build_chair(self, chair: int) -> None:
        x, y = chair_position(chair, CENTER, CHAIR_RADII)
        holder = tk.Frame(self.canvas, bg="#2b2b2b")
        self.canvas.create_window(x, y, window=holder, anchor="center")
        plus = tk.Button(
            holder, text="+", width=3, font=("TkDefaultFont", 12, "bold"),
            command=lambda c=chair: self.add_player(c),
        )
        box = tk.Frame(holder, bd=2, relief="ridge", bg="#f4f1ea", highlightthickness=3,
                       highlightbackground="#f4f1ea")
        head = tk.Frame(box, bg="#f4f1ea")
        head.pack(fill="x")
        tk.Entry(head, textvariable=self.name_vars[chair], width=10).pack(side="left")
        dealer = tk.Button(head, text="D", width=2, command=lambda c=chair: self.set_dealer(c))
        dealer.pack(side="left", padx=2)
        if chair != USER_CHAIR:
            tk.Button(head, text="x", width=2, command=lambda c=chair: self.remove_player(c)).pack(
                side="left"
            )
        row = tk.Frame(box, bg="#f4f1ea")
        row.pack(fill="x", pady=1)
        tk.Label(row, text="stack", bg="#f4f1ea").pack(side="left")
        entry = tk.Entry(row, textvariable=self.stack_vars[chair], width=7)
        entry.pack(side="left", padx=2)
        # Only Return applies a typed value. A FocusOut binding would rebuild the
        # action buttons under the mouse whenever one entry is clicked after
        # another, and the click on the second would be lost.
        entry.bind("<Return>", lambda _e: self.refresh())
        info = tk.Label(box, text="", bg="#f4f1ea", justify="left", font=("TkDefaultFont", 9))
        info.pack(fill="x")
        if chair == USER_CHAIR:
            cards = tk.Frame(box, bg="#f4f1ea")
            cards.pack(pady=2)
            for slot in range(2):
                button = tk.Button(
                    cards, text="?", width=3, height=2, font=("TkDefaultFont", 12, "bold"),
                    command=lambda s=slot: self._pick_hole(s),
                )
                button.pack(side="left", padx=2)
                self.hole_buttons.append(button)
        actions = tk.Frame(box, bg="#f4f1ea")
        actions.pack(fill="x")
        self.chair_ui[chair] = SimpleNamespace(
            plus=plus, box=box, dealer=dealer, info=info, actions=actions
        )

    # -- the table ----------------------------------------------------------

    def add_player(self, chair: int) -> None:
        if self.layout.add(chair):
            self.stack_vars[chair].set(self.stack_var.get())
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

    def _default_stack_changed(self) -> None:
        old, new = self._default_stack, self.stack_var.get()
        if old != new:
            for chair in range(CHAIRS):
                if self.stack_vars[chair].get() == old:
                    self.stack_vars[chair].set(new)
            self._default_stack = new
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

    def build_spot(self) -> Spot:
        if self.layout.players < 2:
            raise ValueError("Aggiungi almeno un avversario con il tasto +")
        default = self._number(self.stack_var, "Stack di partenza")
        stacks = {}
        for seat, chair in enumerate(self.layout.order()):
            text = self.stack_vars[chair].get().strip()
            stacks[seat] = self._number(self.stack_vars[chair], "Stack") if text else default
        hole = tuple(self.hole) if all(card is not None for card in self.hole) else None
        return Spot(
            num_players=self.layout.players,
            starting_stack=default,
            small_blind=self._number(self.sb_var, "Small blind"),
            big_blind=self._number(self.bb_var, "Big blind"),
            my_seat=self.layout.seat_of(USER_CHAIR),
            hole_cards=hole,
            board=tuple(self.board),
            script=list(self.script),
            stacks=stacks,
        )

    def _player_name(self, chair: int) -> str:
        return self.name_vars[chair].get().strip() or f"Posto {chair}"

    def refresh(self) -> None:
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
            ui.info.configure(text="")
            for widget in ui.actions.winfo_children():
                widget.destroy()
        self._draw_cards()
        self.warning.configure(text="")
        self.canvas.itemconfigure(self.pot_item, text="")
        self.canvas.itemconfigure(self.street_item, text="")
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
            text=f"{state.street.name.lower()} | piatto {state.pot} | board {board}\n"
            f"Parla: {self._player_name(turn_chair)} ({names[state.to_act]}){mine}"
        )
        self.canvas.itemconfigure(self.pot_item, text=f"Piatto {state.pot}")
        self.canvas.itemconfigure(self.street_item, text=state.street.name.lower())
        self.chair_ui[turn_chair].box.configure(highlightbackground=HIGHLIGHT)
        folded = {
            info.seat for info in state.observation.seats if info.status is PlayerStatus.FOLDED
        }
        for seat, chair in enumerate(self.layout.order()):
            text = names[seat]
            if seat in folded:
                text += "  FOLD"
            elif state.bets.get(seat):
                text += f"  puntata {state.bets[seat]}"
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
            var = tk.StringVar(value=str(legal.min_amount))
            line = tk.Frame(parent, bg="#f4f1ea")
            line.pack(fill="x", pady=1)
            entry = tk.Entry(line, textvariable=var, width=6)
            entry.pack(side="left")
            self._bind_wheel(entry, var, legal)

            def go(legal=legal, var=var) -> None:
                try:
                    amount = int(var.get())
                except ValueError:
                    return
                if not legal.min_amount <= amount <= legal.max_amount:
                    messagebox.showwarning("Importo", f"Tra {legal.min_amount} e {legal.max_amount}.")
                    return
                self._append(Action(legal.action_type, amount))

            button = tk.Button(
                line, text=f"{verb} ({legal.min_amount}-{legal.max_amount})", command=go
            )
            button.pack(side="left", padx=2)
            self._bind_wheel(button, var, legal)
            return
        label = describe_action(Action(legal.action_type), state.observation).capitalize()
        tk.Button(
            parent, text=label, command=lambda t=legal.action_type: self._append(Action(t))
        ).pack(fill="x", pady=1)

    def _wheel_step(self, var: tk.StringVar, legal: LegalAction, direction: int) -> None:
        """Move the amount by one big blind, kept inside what the engine allows."""
        try:
            current = int(var.get())
        except ValueError:
            current = legal.min_amount
        try:
            step = max(1, WHEEL_STEP_BIG_BLINDS * int(self.bb_var.get()))
        except ValueError:
            step = 1
        var.set(str(min(legal.max_amount, max(legal.min_amount, current + direction * step))))

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
            lines.append(f"{number}. {name} ({names[seat]}): {describe_action(action)}")
        self._write(self.log, "\n".join(lines))

    @staticmethod
    def _write(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    # -- the models ---------------------------------------------------------

    def _ask(self) -> None:
        # Whatever is on the table is what gets asked: no separate "apply".
        self.refresh()
        state = self.state
        if state is None or state.finished or state.observation is None:
            self.status_var.set("Niente da chiedere: sistema prima il tavolo.")
            return
        spot = self.build_spot()
        if state.to_act != spot.my_seat:
            self.status_var.set("Tocca a un altro: premi le sue azioni fino al tuo turno.")
            return
        if spot.hole_cards is None:
            self.status_var.set("Imposta le tue due carte.")
            return
        self._consult(state, spot)

    def _consult(self, state: SpotState, spot: Spot) -> None:
        if not self._load_models():
            return
        try:
            advice = advise(
                state.observation,
                state.legal_actions,
                self.models,
                big_blind=spot.big_blind,
                starting_stack=spot.starting_stack,
            )
        except Exception as exc:  # noqa: BLE001 - shown, not fatal
            self.status_var.set(f"Errore: {exc}")
            return
        self.status_var.set(f"{len(advice)} modelli consultati.")
        lines = []
        for item in advice:
            lines.append(
                f"{item.label}  (elo {item.rating:.0f})\n"
                f"  -> {describe_action(item.best.action, state.observation)}"
                f"  {item.best.probability:.0%}   valore {item.value:+.2f}"
            )
            lines.append(
                "     "
                + ", ".join(
                    f"{b.label} {b.probability:.0%}" for b in item.bins if b.probability >= 0.01
                )
            )
        self._write(self.advice, "\n".join(lines))

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
        return True
