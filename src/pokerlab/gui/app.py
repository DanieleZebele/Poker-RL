from __future__ import annotations

import os
import queue
import random
import sys
import threading
import time
from pathlib import Path


def _fix_tcl_tk_library_paths() -> None:
    """Some Windows Python installs (seen on this project's own machine)
    don't register TCL_LIBRARY/TK_LIBRARY correctly, so tkinter looks for
    init.tcl under "<prefix>/lib/tcl8.6" when the files actually live under
    "<prefix>/tcl/tcl8.6". Point the env vars at the real location before
    tkinter is imported, but only as a fallback -- a correctly configured
    install is left untouched."""
    base = Path(sys.base_prefix)
    for var, subdir in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        if os.environ.get(var):
            continue
        candidate = base / "tcl" / subdir
        if candidate.is_dir():
            os.environ[var] = str(candidate)


_fix_tcl_tk_library_paths()

import tkinter as tk
from tkinter import messagebox, ttk

from pokerlab.cli.play import (
    build_players,
    discover_global_top_models,
    discover_trained_models,
    validate_bot_key,
)
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import HandHistoryWriter
from pokerlab.engine.state import ActionRecord, PlayerStatus
from pokerlab.engine.table import HandResult, Table
from pokerlab.evaluator.evaluator import HandCategory, evaluate
from pokerlab.gui.cards_canvas import (
    draw_card_back,
    draw_card_face,
    draw_empty_slot,
    new_card_canvas,
)
from pokerlab.players.base import Observation, Player
from pokerlab.players.gui import GuiEvent, GuiPlayer

# How long a bot's action stays on the table before the next one fires.
# Without it a whole hand's actions land between two of the GUI's 100ms
# polls and render as one instant jump -- this is what makes an unattended
# bot-only session watchable at all, not just step mode.
BOT_ACTION_DELAY_SECONDS = 1.0

# Gives the human time to read the previous hand's showdown/final stacks
# before the next hand's cards start appearing.
NEW_HAND_DELAY_SECONDS = 3.0

# How long every participant's hole cards stay face-up on the table once a
# hand is over, before the table is cleared for the next deal. The hand is
# finished by then, so nothing is leaked that could affect play -- and
# seeing what the bots actually held is the whole point of a testing GUI.
SHOWDOWN_REVEAL_SECONDS = 1.0

# Beat after a new community card lands, before the street's first action.
# The engine deals the street and the next player acts microseconds later,
# so without this the two events reach the GUI inside one 100ms poll and
# the board and the next move appear as a single jump -- the card reads as
# arriving *instead of* the action.
NEW_STREET_DELAY_SECONDS = 1.0

# Pause after each community card of a runout nobody can act on (everyone
# still in the hand is all-in). Without it the engine deals flop, turn and
# river and settles the hand between two of the GUI's 100ms polls, so an
# all-in appears to skip straight from the bet to the final stacks -- the
# player never sees how it ended.
ALL_IN_STREET_DELAY_SECONDS = 1.4

# Raise shortcuts offered next to the slider, as a fraction of the pot
# *after* the call (the standard way a pot-sized bet is reckoned).
POT_FRACTION_PRESETS: tuple[tuple[float, str], ...] = (
    (0.30, "30%"),
    (0.50, "50%"),
    (0.66, "66%"),
    (1.00, "Piatto"),
)

# One mouse-wheel notch over the raise slider moves the bet by this many
# big blinds.
WHEEL_STEP_BIG_BLINDS = 1

# The dealer's seat box is tinted whole (frame, labels and card
# backgrounds), not just tagged with a "D" above it.
DEALER_BACKGROUND = "#f6d55c"

_STATUS_LABELS = {
    PlayerStatus.ACTIVE: "",
    PlayerStatus.FOLDED: "Folded",
    PlayerStatus.ALL_IN: "All-in",
    PlayerStatus.BUSTED: "Out",
}

_SIMPLE_ACTION_LABELS = {
    ActionType.FOLD: "Fold",
    ActionType.CHECK: "Check",
    ActionType.CALL: "Call",
    ActionType.ALL_IN: "All-in",
}


def _describe_action(record: ActionRecord) -> str:
    """Human-readable description of an action for the live log.

    Read off the engine's own ActionRecord rather than re-derived from an
    Observation: CALL and ALL_IN ignore Action.amount (the engine works
    the chips out from state -- see Action's docstring), and the record
    already knows exactly what moved. `stack_before - stack_after` is the
    chips actually committed; `record.amount` is the player's street total
    after the action, which is precisely the raise-to level a BET/RAISE
    reports.
    """
    committed = record.stack_before - record.stack_after
    if record.action_type == ActionType.FOLD:
        return "fold"
    if record.action_type == ActionType.CHECK:
        return "check"
    if record.action_type == ActionType.CALL:
        return f"call {committed}"
    if record.action_type == ActionType.ALL_IN:
        return f"all-in ({committed})"
    if record.action_type == ActionType.BET:
        return f"bet {record.amount}"
    if record.action_type == ActionType.RAISE:
        return f"raise to {record.amount}"
    return record.action_type.value


_CATEGORY_NAMES_IT = {
    HandCategory.HIGH_CARD: "Carta alta",
    HandCategory.PAIR: "Coppia",
    HandCategory.TWO_PAIR: "Doppia coppia",
    HandCategory.THREE_OF_A_KIND: "Tris",
    HandCategory.STRAIGHT: "Scala",
    HandCategory.FLUSH: "Colore",
    HandCategory.FULL_HOUSE: "Full",
    HandCategory.FOUR_OF_A_KIND: "Poker",
    HandCategory.STRAIGHT_FLUSH: "Scala colore",
}


def _describe_combination(hole, board) -> str | None:
    """The best five-card hand a seat makes out of its hole cards plus the
    board, named in Italian like the rest of the GUI, e.g.
    "Doppia coppia (Ah Ad Kh Kd 5c)".

    Returns None when fewer than five cards are known between the two --
    a hand won before the flop has no combination to show, and `evaluate`
    would raise rather than guess. The category names are kept here rather
    than in the evaluator because they are presentation, and the evaluator
    is the one file in this project deliberately left alone (see CLAUDE.md).
    """
    if not hole:
        return None
    cards = (*hole, *board)
    if len(cards) < 5:
        return None
    rank = evaluate(cards)
    name = _CATEGORY_NAMES_IT[rank.category]
    if rank.category == HandCategory.STRAIGHT_FLUSH and rank.tiebreakers[0] == 14:
        name = "Scala reale"
    return f"{name} ({' '.join(str(c) for c in _order_for_display(rank.best_five))})"


def _order_for_display(best_five):
    """The winning five sorted the way a player reads them: the cards that
    make the hand first, kickers after, each group descending. HandRank
    keeps `best_five` in whatever order the combination was generated in,
    which prints a pair of tens as "Ah 7h Tc Th Kd" -- correct, but the
    pair is the last thing the eye finds."""
    counts: dict[int, int] = {}
    for card in best_five:
        counts[card.rank.value] = counts.get(card.rank.value, 0) + 1
    return sorted(best_five, key=lambda c: (-counts[c.rank.value], -c.rank.value))


def _format_hand_summary(result: HandResult) -> str:
    hh = result.hand_history
    names = hh.seat_names
    board = hh.community_cards
    lines = [f"--- fine mano {result.hand_id} ---"]
    if board:
        lines.append(f"Board: {' '.join(str(c) for c in board)}")
    folded = {a.seat for a in hh.actions if a.action_type == ActionType.FOLD}
    contested = [s for s in hh.starting_stacks if s not in folded]
    showdown = len(contested) >= 2
    if showdown:
        lines.append("Showdown:")
        for s in contested:
            hole = hh.hole_cards.get(s)
            card_text = " ".join(str(c) for c in hole) if hole else "?"
            combination = _describe_combination(hole, board)
            suffix = f"  ->  {combination}" if combination else ""
            lines.append(f"  {names[s]}: {card_text}{suffix}")
    winner_parts = []
    for s, amt in result.payouts.items():
        part = f"{names[s]} (+{amt})"
        # Only at a real showdown: a pot taken down by everyone folding was
        # never shown, and the board may not even be complete, so naming a
        # "winning combination" there would be inventing one.
        combination = _describe_combination(hh.hole_cards.get(s), board) if showdown else None
        if combination:
            part += f" con {combination}"
        winner_parts.append(part)
    lines.append("Pot vinto da: " + ", ".join(winner_parts) + ("" if showdown else " (senza showdown)"))
    stacks = ", ".join(f"{names[s]}: {stack}" for s, stack in result.final_stacks.items())
    lines.append(f"Stack finali: {stacks}")
    return "\n".join(lines)


def _pot_fraction_raise_to(observation: Observation, fraction: float) -> int:
    """The raise-to level for betting `fraction` of the pot.

    Standard poker sizing: the player first matches the outstanding bet,
    which grows the pot, and only then bets the fraction of *that* pot. The
    result is a total commitment for the street, which is what
    Action.amount means for BET/RAISE (see Action's docstring) -- not the
    incremental chips added. The caller is responsible for clamping it into
    the legal [min_amount, max_amount] range.
    """
    to_call = max(0, observation.current_bet_to_match - observation.my_current_bet)
    pot_after_call = observation.pot_size + to_call
    return observation.my_current_bet + to_call + round(fraction * pot_after_call)


class ActionReporter:
    """Table's `on_action_applied` hook: publish the action, then pace.

    Both halves used to live in a wrapper around `Player.act()`, and that
    put them in the wrong order. A wrapper only ever sees the state from
    *before* its own action -- the engine applies it once `act()` has
    returned -- so the table showed the move as not yet made and then held
    that stale picture for the whole pause. Worse, the correction only
    arrived with the *next* action, so the last action of a street was
    never drawn at all: the next thing to happen is the new community
    card. Running from the engine hook instead means what the pause holds
    on screen is the finished action, chips in and pot updated.

    Called on the session thread, which is what makes blocking legal here
    (the same place `on_street_dealt` waits out an all-in runout).
    `delay_seconds` is overridable so tests do not sleep for real.
    """

    def __init__(
        self,
        event_queue: queue.Queue[GuiEvent],
        step_gate: queue.Queue[None],
        step_mode_state: dict[str, bool],
        human_seat: int | None,
        delay_seconds: float = BOT_ACTION_DELAY_SECONDS,
    ) -> None:
        self._event_queue = event_queue
        self._step_gate = step_gate
        self._step_mode_state = step_mode_state
        self._human_seat = human_seat
        self._delay_seconds = delay_seconds

    def __call__(self, info: dict) -> None:
        self._event_queue.put(GuiEvent("action_taken", info))
        if info["seat"] == self._human_seat:
            return  # they already "stepped" by clicking
        if self._step_mode_state["on"]:
            self._step_gate.get()
        else:
            time.sleep(self._delay_seconds)


def _run_session(table: Table, num_hands: int, event_queue: queue.Queue[GuiEvent], writer: HandHistoryWriter) -> None:
    """Runs on a background thread: drives the actual game loop. This
    thread can be blocked by a GuiPlayer.act() call waiting on the human's
    next click, or (in step mode) by start_session's on_action_applied
    callback waiting on the "Avanti" button -- there is no separate
    "pause"/"stop mid-hand"
    mechanism beyond that; closing the window (a daemon thread) is how a
    session gets abandoned. Also blocked, deliberately, by
    `NEW_HAND_DELAY_SECONDS` between hands (not before the first), so the
    table does not race straight into the next deal before anyone has had
    a chance to read the previous hand's showdown.
    """
    try:
        for i in range(num_hands):
            if sum(1 for s in table.stacks if s > 0) < 2:
                event_queue.put(GuiEvent("session_ended_early", i))
                return
            if i > 0:
                time.sleep(NEW_HAND_DELAY_SECONDS)
            result = table.play_hand()
            event_queue.put(GuiEvent("hand_complete", result))
        event_queue.put(GuiEvent("session_complete"))
    finally:
        writer.close()


def _bot_spec_to_key_string(spec: dict) -> str:
    """Turn a setup-screen bot spec ({"key": "model", "path": ...}) into the
    plain 'model:<path>' string build_players/validate_bot_key understand."""
    return f"model:{spec['path']}"


def _bot_spec_label(spec: dict) -> str:
    return f"[RL] {Path(spec['path']).stem[:20]}"


class AddBotDialog(tk.Toplevel):
    """Modal dialog opened by the "+" button: pick a trained model to seat.

    There is no hand-coded bot catalog any more -- every bot seat is a
    trained checkpoint, offered by path rather than a hardcoded key: a key
    pointing at a file that may later go missing would break the setup
    screen for everyone, while a path only ever fails for whoever picks it.
    """

    def __init__(self, master: tk.Widget, on_confirm) -> None:
        super().__init__(master)
        self.title("Aggiungi bot")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self._on_confirm = on_confirm

        self._models = discover_trained_models(limit=20)
        ttk.Label(
            self, text="Modello addestrato (i migliori per rating):", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", padx=12, pady=(12, 4))
        if self._models:
            self.model_var = tk.StringVar(value=self._model_choices()[0])
            ttk.Combobox(
                self,
                textvariable=self.model_var,
                values=self._model_choices(),
                state="readonly",
                width=52,
            ).pack(anchor="w", padx=24)
        else:
            self.model_var = tk.StringVar(value="")
            ttk.Label(self, text="nessun modello addestrato trovato in checkpoints/").pack(
                anchor="w", padx=24
            )

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=12, pady=12)
        ttk.Button(buttons, text="Annulla", command=self.destroy).pack(side="right")
        ttk.Button(
            buttons, text="Aggiungi", command=self._confirm, state="normal" if self._models else "disabled"
        ).pack(side="right", padx=(0, 8))

    def _model_choices(self) -> list[str]:
        """Rating first: it is the only ordering that means anything here, and
        it is what the user is picking on."""
        return [f"{rating:.0f}  {label}" for label, _path, rating in self._models]

    def _selected_model_path(self) -> str:
        index = self._model_choices().index(self.model_var.get())
        return str(self._models[index][1])

    def _confirm(self) -> None:
        if not self._models:
            return
        self._on_confirm({"key": "model", "path": self._selected_model_path()})
        self.destroy()


class SetupFrame(ttk.Frame):
    def __init__(self, master: PokerGuiApp) -> None:
        super().__init__(master, padding=16)
        self.app = master
        # Default proposal: one instance each of the top 6 models by rating
        # in the cross-machine global Elo registry (see
        # discover_global_top_models) -- a sensible starting table rather
        # than an empty one, since the global scale is the one comparable
        # measure of "best" across the whole fleet. Still just a starting
        # point: the player can remove any of these or add others through
        # the same "+" dialog as always.
        self.bot_specs: list[dict] = [
            {"key": "model", "path": str(path)}
            for _label, path, _rating in discover_global_top_models(limit=6)
        ]

        self.human_var = tk.BooleanVar(value=True)
        self.step_mode_var = tk.BooleanVar(value=False)
        self.stack_var = tk.StringVar(value="200")
        self.sb_var = tk.StringVar(value="1")
        self.bb_var = tk.StringVar(value="2")
        self.hands_var = tk.StringVar(value="20")
        self.seed_var = tk.StringVar(value="")

        ttk.Label(self, text="pokerlab -- test dal vivo", font=("TkDefaultFont", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 12)
        )

        fields = [
            ("Stack iniziale", self.stack_var),
            ("Small blind", self.sb_var),
            ("Big blind", self.bb_var),
            ("Numero di mani", self.hands_var),
            ("Seed (vuoto = casuale)", self.seed_var),
        ]
        for row, (label, var) in enumerate(fields, start=1):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(self, textvariable=var, width=28).grid(row=row, column=1, sticky="w", pady=2)

        next_row = len(fields) + 1
        ttk.Checkbutton(self, text="Io gioco (seat 0)", variable=self.human_var, command=self._render_bot_slots).grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(
            self,
            text="Modalita' passo-passo fin dall'inizio (avanti manuale sulle azioni dei bot)",
            variable=self.step_mode_var,
        ).grid(row=next_row + 1, column=0, columnspan=2, sticky="w")

        ttk.Label(self, text="Bot in partita:", font=("TkDefaultFont", 10, "bold")).grid(
            row=next_row + 2, column=0, columnspan=2, sticky="w", pady=(12, 2)
        )
        self.bots_frame = ttk.Frame(self)
        self.bots_frame.grid(row=next_row + 3, column=0, columnspan=2, sticky="w")
        self.total_players_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.total_players_var).grid(
            row=next_row + 4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        button_row = next_row + 5
        ttk.Button(self, text="Avvia", command=self._start).grid(row=button_row, column=1, pady=(16, 0), sticky="e")
        ttk.Button(self, text="Chiedi ai modelli (spot)", command=self.app.show_spot).grid(
            row=button_row, column=0, pady=(16, 0), sticky="w"
        )

        self._render_bot_slots()

    def _render_bot_slots(self) -> None:
        for widget in self.bots_frame.winfo_children():
            widget.destroy()
        for i, spec in enumerate(self.bot_specs):
            box = ttk.Frame(self.bots_frame, relief="groove", borderwidth=2, padding=6)
            box.grid(row=0, column=i, padx=4, pady=4, sticky="n")
            ttk.Label(box, text=_bot_spec_label(spec), justify="center").pack()
            ttk.Button(box, text="-", width=3, command=lambda i=i: self._remove_bot(i)).pack(pady=(4, 0))
        add_box = ttk.Frame(self.bots_frame, relief="groove", borderwidth=2, padding=6)
        add_box.grid(row=0, column=len(self.bot_specs), padx=4, pady=4, sticky="n")
        ttk.Label(add_box, text="Aggiungi\nbot", justify="center").pack()
        ttk.Button(add_box, text="+", width=3, command=self._open_add_bot_dialog).pack(pady=(4, 0))

        total = len(self.bot_specs) + (1 if self.human_var.get() else 0)
        self.total_players_var.set(f"Giocatori totali: {total} (min 2, max 9)")

    def _remove_bot(self, index: int) -> None:
        del self.bot_specs[index]
        self._render_bot_slots()

    def _open_add_bot_dialog(self) -> None:
        max_bots = 9 - (1 if self.human_var.get() else 0)
        if len(self.bot_specs) >= max_bots:
            messagebox.showwarning("Limite raggiunto", f"Puoi avere al massimo {max_bots} bot con l'impostazione attuale.")
            return
        AddBotDialog(self, on_confirm=self._add_bot)

    def _add_bot(self, spec: dict) -> None:
        self.bot_specs.append(spec)
        self._render_bot_slots()

    def _start(self) -> None:
        try:
            stack = int(self.stack_var.get())
            sb = int(self.sb_var.get())
            bb = int(self.bb_var.get())
            hands = int(self.hands_var.get())
            seed_text = self.seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
            human_seats = 1 if self.human_var.get() else 0
            bot_keys = [_bot_spec_to_key_string(spec) for spec in self.bot_specs]
            for key in bot_keys:
                validate_bot_key(key)
            num_players = human_seats + len(bot_keys)
            config = GameConfig(num_players=num_players, starting_stack=stack, small_blind=sb, big_blind=bb)
        except ValueError as e:
            messagebox.showerror("Configurazione non valida", str(e))
            return

        self.app.start_session(config, human_seats, bot_keys, hands, seed, self.step_mode_var.get())


class TableFrame(ttk.Frame):
    def __init__(
        self,
        master: PokerGuiApp,
        num_players: int,
        event_queue: queue.Queue[GuiEvent],
        human_player: GuiPlayer | None,
        history_path: Path,
        seat_names: dict[int, str],
        step_gate: queue.Queue[None],
        step_mode_state: dict[str, bool],
        big_blind: int = 2,
    ) -> None:
        super().__init__(master, padding=12)
        self.app = master
        self.event_queue = event_queue
        self.human_player = human_player
        self.seat_names = seat_names
        self.step_gate = step_gate
        self.step_mode_state = step_mode_state
        self.big_blind = big_blind
        self._all_hole_cards: dict[int, tuple] = {}
        self._last_observation: Observation | None = None
        self._busted_seats: set[int] = set()
        self._pending_clear: str | None = None
        # Per build_players' convention, a human always sits at seat 0.
        self._human_seat: int | None = 0 if human_player is not None else None
        self._seat_background = self._configure_seat_styles()

        header = ttk.Frame(self)
        header.pack(fill="x")
        self.street_var = tk.StringVar(value="")
        self.pot_var = tk.StringVar(value="Pot: 0")
        ttk.Label(header, textvariable=self.street_var, font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.pot_var).pack(side="left", padx=16)
        ttk.Label(header, text="Board:").pack(side="left", padx=(16, 4))
        self.board_canvases = [new_card_canvas(header) for _ in range(5)]
        for canvas in self.board_canvases:
            canvas.pack(side="left", padx=2)

        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(6, 0))
        self.spy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Spia carte avversari", variable=self.spy_var, command=self._on_spy_toggled).pack(
            side="left"
        )
        self.step_mode_var = tk.BooleanVar(value=step_mode_state["on"])
        ttk.Checkbutton(
            controls, text="Modalita' passo-passo (bot)", variable=self.step_mode_var, command=self._on_step_mode_toggled
        ).pack(side="left", padx=(16, 0))
        ttk.Button(controls, text="Avanti ->", command=self._on_advance).pack(side="left", padx=(16, 0))

        seats_frame = ttk.Frame(self)
        seats_frame.pack(fill="x", pady=10)
        self.seat_widgets: dict[int, dict[str, object]] = {}
        for seat in range(num_players):
            dealer_var = tk.StringVar(value="")
            dealer_label = ttk.Label(
                seats_frame, textvariable=dealer_var, font=("TkDefaultFont", 12, "bold"), anchor="center"
            )
            dealer_label.grid(row=0, column=seat, sticky="ew")
            box = ttk.LabelFrame(seats_frame, text=f"Seat {seat}", style="Seat.TLabelframe")
            box.grid(row=1, column=seat, padx=4, sticky="n")
            name_var = tk.StringVar(value="-")
            stack_var = tk.StringVar(value="-")
            bet_var = tk.StringVar(value="")
            status_var = tk.StringVar(value="")
            labels = []
            for var in (name_var, stack_var, bet_var, status_var):
                label = ttk.Label(box, textvariable=var, width=16, style="Seat.TLabel")
                label.pack(anchor="w")
                labels.append(label)
            cards_frame = ttk.Frame(box, style="Seat.TFrame")
            cards_frame.pack(anchor="w", pady=(2, 0))
            back_canvases = [new_card_canvas(cards_frame) for _ in range(2)]
            for canvas in back_canvases:
                canvas.pack(side="left", padx=1)
            self.seat_widgets[seat] = {
                "frame": box,
                "dealer_label": dealer_label,
                "dealer": dealer_var,
                "name": name_var,
                "stack": stack_var,
                "bet": bet_var,
                "status": status_var,
                "cards": back_canvases,
                "labels": labels,
                "cards_frame": cards_frame,
                "is_dealer": False,
            }

        hole_frame = ttk.Frame(self)
        hole_frame.pack(anchor="w")
        ttk.Label(hole_frame, text="Le tue carte:", font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=(0, 6))
        self.hole_canvases = [new_card_canvas(hole_frame) for _ in range(2)]
        for canvas in self.hole_canvases:
            canvas.pack(side="left", padx=2)

        self.actions_frame = ttk.Frame(self)
        self.actions_frame.pack(fill="x", pady=8)
        self.waiting_var = tk.StringVar(value="I bot stanno giocando..." if human_player else "Modalita' spettatore.")
        ttk.Label(self.actions_frame, textvariable=self.waiting_var).pack(side="left")

        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(footer, text=f"Hand history: {history_path}").pack(side="left")
        self.menu_button = ttk.Button(footer, text="Torna al menu", command=self.app.show_setup, state="normal")
        self.menu_button.pack(side="right")

        ttk.Label(self, text="Storico mani:").pack(anchor="w", pady=(8, 0))
        log_frame = ttk.Frame(self)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=14, state="disabled", wrap="word")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.after(100, self._poll_events)

    def _configure_seat_styles(self) -> str:
        """Two parallel sets of ttk styles, "Seat.*" and "Dealer.*", so a
        seat's whole box can be tinted by swapping styles rather than by
        recolouring each widget by hand.

        ttk widgets take their background from their style, not from a
        `bg` option, which is why the labels and the inner card frame need
        styles of their own: restyling only the LabelFrame would leave the
        text sitting on default-grey patches inside a yellow box. The card
        Canvases are classic tk widgets and are recoloured directly.

        Returns the theme's ordinary background, which is both the
        non-dealer colour and what the card Canvases revert to.
        """
        style = ttk.Style(self)
        background = style.lookup("TLabelframe", "background") or style.lookup("TFrame", "background")
        for prefix, colour in (("Seat", background), ("Dealer", DEALER_BACKGROUND)):
            style.configure(f"{prefix}.TLabelframe", background=colour)
            style.configure(f"{prefix}.TLabelframe.Label", background=colour)
            style.configure(f"{prefix}.TLabel", background=colour)
            style.configure(f"{prefix}.TFrame", background=colour)
        return background

    def _set_dealer_seat(self, dealer_seat: int | None) -> None:
        """Tint the dealer's entire seat box and un-tint everyone else's."""
        for seat, widgets in self.seat_widgets.items():
            is_dealer = seat == dealer_seat
            if widgets["is_dealer"] == is_dealer:
                continue
            widgets["is_dealer"] = is_dealer
            prefix = "Dealer" if is_dealer else "Seat"
            colour = DEALER_BACKGROUND if is_dealer else self._seat_background
            widgets["frame"].configure(style=f"{prefix}.TLabelframe")
            widgets["cards_frame"].configure(style=f"{prefix}.TFrame")
            for label in widgets["labels"]:
                label.configure(style=f"{prefix}.TLabel")
            for canvas in widgets["cards"]:
                canvas.configure(background=colour)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_hand_start(self, info: dict) -> None:
        sb_name = self.seat_names.get(info["sb_seat"], f"seat {info['sb_seat']}")
        bb_name = self.seat_names.get(info["bb_seat"], f"seat {info['bb_seat']}")
        self._log(
            f"=== {info['hand_id']} ===\n"
            f"  {sb_name} posta small blind {info['small_blind']}\n"
            f"  {bb_name} posta big blind {info['big_blind']}"
        )

    def _log_action(self, name: str, record: ActionRecord) -> None:
        self._log(f"[{record.street.value.upper()}] {name}: {_describe_action(record)}")

    def _on_spy_toggled(self) -> None:
        if self._last_observation is not None:
            for seat_info in self._last_observation.seats:
                self._draw_seat_cards(seat_info, is_me=(seat_info.seat == self._human_seat))

    def _on_step_mode_toggled(self) -> None:
        self.step_mode_state["on"] = self.step_mode_var.get()
        if self.step_mode_state["on"]:
            # Drain any stale release left over from turning step mode off
            # below, so re-enabling it doesn't silently skip the very next pause.
            while True:
                try:
                    self.step_gate.get_nowait()
                except queue.Empty:
                    break
        else:
            # A bot may already be blocked waiting on the gate from just
            # before this toggle; release it immediately instead of leaving
            # the session stuck until one more manual "Avanti" click.
            self.step_gate.put(None)

    def _on_advance(self) -> None:
        self.step_gate.put(None)

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _handle_event(self, event: GuiEvent) -> None:
        if event.kind == "hand_started":
            self._cancel_pending_clear()
            self._all_hole_cards = event.payload["hole_cards"]
            self._last_observation = None
            self._reset_seats_for_new_hand(self._all_hole_cards.keys())
            self._set_dealer_seat(event.payload["button_seat"])
            self._log_hand_start(event.payload)
        elif event.kind == "your_turn":
            observation, legal_actions = event.payload
            self.waiting_var.set("")
            self._render_observation(observation)
            self._render_actions(observation, legal_actions)
        elif event.kind == "action_taken":
            info = event.payload
            self._render_observation(info["observation"])
            self._log_action(info["name"], info["record"])
        elif event.kind == "street_dealt":
            self._render_street_dealt(event.payload)
        elif event.kind == "hand_complete":
            self._log(_format_hand_summary(event.payload))
            self._reveal_hand_end(event.payload)
            self.waiting_var.set("I bot stanno giocando..." if self.human_player else "Modalita' spettatore.")
        elif event.kind == "session_ended_early":
            self._log(f"Sessione terminata dopo {event.payload} mani: meno di 2 giocatori con chip.")
            self._session_over()
        elif event.kind == "session_complete":
            self._log("Sessione completata.")
            self._session_over()

    def _session_over(self) -> None:
        self.waiting_var.set("Sessione finita.")
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.actions_frame, textvariable=self.waiting_var).pack(side="left")
        self.menu_button.configure(state="normal")

    def _hide_seat(self, seat: int) -> None:
        """A player with 0 chips is out for the rest of the session --
        remove their box from the table entirely rather than just greying
        it out, per the user's request."""
        if seat in self._busted_seats:
            return
        self._busted_seats.add(seat)
        self.seat_widgets[seat]["frame"].grid_remove()
        self.seat_widgets[seat]["dealer_label"].grid_remove()

    def _reset_seats_for_new_hand(self, participant_seats) -> None:
        """Called right on "hand_started", before any action of the new
        hand has actually happened. Without this, the seat display (and
        the spy toggle in particular) would keep showing the *previous*
        hand's status/cards until the new hand's first action_taken/
        your_turn event arrives -- a brief but real stale-data window,
        especially noticeable in step mode where that first action might
        be paused for a while."""
        participants = set(participant_seats)
        for seat, widgets in self.seat_widgets.items():
            if seat not in participants:
                self._hide_seat(seat)
                continue
            widgets["dealer"].set("")
            widgets["name"].set(self.seat_names.get(seat, f"seat {seat}"))
            widgets["stack"].set("-")
            widgets["bet"].set("")
            widgets["status"].set("")
            if seat == self._human_seat:
                for canvas in widgets["cards"]:
                    draw_empty_slot(canvas)
            elif self.spy_var.get() and seat in self._all_hole_cards:
                for canvas, card in zip(widgets["cards"], self._all_hole_cards[seat]):
                    draw_card_face(canvas, card)
            else:
                for canvas in widgets["cards"]:
                    draw_card_back(canvas)

    def _draw_seat_cards(self, seat_info, is_me: bool) -> None:
        widgets = self.seat_widgets[seat_info.seat]
        canvases = widgets["cards"]
        still_in_hand = seat_info.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        if is_me or not still_in_hand:
            for canvas in canvases:
                draw_empty_slot(canvas)
            return
        if self.spy_var.get() and seat_info.seat in self._all_hole_cards:
            hole = self._all_hole_cards[seat_info.seat]
            for canvas, card in zip(canvases, hole):
                draw_card_face(canvas, card)
            return
        for canvas in canvases:
            draw_card_back(canvas)

    def _render_observation(self, observation: Observation) -> None:
        self._last_observation = observation
        self.street_var.set(observation.street.value.upper())
        self.pot_var.set(f"Pot: {observation.pot_size}")
        self._draw_board(observation.community_cards)
        self._set_dealer_seat(observation.button_seat)

        # Same rule as the "(tu)" marker above: with nobody sitting in,
        # this panel has nothing to show. It used to fall back to
        # `observation.hole_cards`, which in spectator mode is whichever
        # bot just acted -- so "Le tue carte" flipped between opponents'
        # hands every action.
        if self._human_seat is not None:
            own_hole_cards = self._all_hole_cards.get(self._human_seat, observation.hole_cards)
            for canvas, card in zip(self.hole_canvases, own_hole_cards):
                draw_card_face(canvas, card)

        present_seats = {s.seat for s in observation.seats}
        for seat in self.seat_widgets:
            if seat not in present_seats:
                self._hide_seat(seat)
        for seat_info in observation.seats:
            widgets = self.seat_widgets[seat_info.seat]
            widgets["dealer"].set("D" if seat_info.is_button else "")
            # "Who am I" is a property of this GUI, never of the Observation
            # being rendered: most of them belong to whichever player just
            # acted, so reading `my_seat` labelled that bot "(tu)" and drew
            # its cards as an empty slot -- it looked like it had folded,
            # and in spectator mode one bot always wore the marker.
            is_me = seat_info.seat == self._human_seat
            widgets["name"].set(seat_info.name + (" (tu)" if is_me else ""))
            widgets["stack"].set(f"Stack: {seat_info.stack}")
            widgets["bet"].set(f"Bet: {seat_info.current_bet}")
            widgets["status"].set(_STATUS_LABELS.get(seat_info.status, ""))
            self._draw_seat_cards(seat_info, is_me=is_me)

    def _render_street_dealt(self, info: dict) -> None:
        """The flop/turn/river as the engine deals it, straight from Table's
        on_street_dealt hook.

        For an ordinary street this merely beats the street's first action
        to the draw. For a runout nobody can act on (`betting_closed` --
        everyone still in the hand is all-in) it is the *only* thing that
        ever shows the board: no Player.act() is called, so the GUI used to
        see nothing at all between the last bet and the final stacks, which
        is what made an all-in look like it skipped its own ending. The
        pacing that makes the runout watchable lives on the session thread,
        in start_session's callback, not here.
        """
        self.street_var.set(info["street"].value.upper())
        self._draw_board(info["community_cards"])
        if info["betting_closed"]:
            self.waiting_var.set("All-in: si va a vedere il board...")
            self._reveal_live_hole_cards()

    def _draw_board(self, community_cards) -> None:
        for i, canvas in enumerate(self.board_canvases):
            if i < len(community_cards):
                draw_card_face(canvas, community_cards[i])
            else:
                draw_empty_slot(canvas)

    def _folded_seats(self) -> set[int]:
        """Who has mucked, read straight off the last Observation rendered.

        That Observation is the state *after* the action it came with (see
        Table's on_action_applied), so a fold is visible in it immediately
        and the GUI needs no fold-tracking of its own. There is none at all
        when a runout begins with no action having been taken -- every seat
        all-in on its own blind -- and then nobody has folded.
        """
        if self._last_observation is None:
            return set()
        return {s.seat for s in self._last_observation.seats if s.status == PlayerStatus.FOLDED}

    def _reveal_live_hole_cards(self) -> None:
        """Turn over everyone still contesting the pot -- what a real poker
        room does the moment an all-in is called and there is nothing left
        to protect. Folded seats keep their cards mucked."""
        folded = self._folded_seats()
        for seat, hole in self._all_hole_cards.items():
            if seat in folded or seat in self._busted_seats:
                continue
            if seat == self._human_seat:
                continue  # drawn full-size above the seat grid instead
            for canvas, card in zip(self.seat_widgets[seat]["cards"], hole):
                draw_card_face(canvas, card)

    def _reveal_hand_end(self, result: HandResult) -> None:
        """Everyone's cards face-up on the finished board, held for
        SHOWDOWN_REVEAL_SECONDS before the table is cleared.

        Nothing is hidden yet, not even a seat that just busted: a player
        who went all-in and lost is precisely the one whose cards are worth
        seeing, and hiding the seat here would delete them the instant they
        became showable. _clear_after_showdown does the hiding.
        """
        hh = result.hand_history
        self._draw_board(hh.community_cards)
        for seat, widgets in self.seat_widgets.items():
            if seat not in hh.final_stacks:
                self._hide_seat(seat)
                continue
            widgets["name"].set(hh.seat_names[seat])
            widgets["stack"].set(f"Stack: {hh.final_stacks[seat]}")
            widgets["bet"].set("")
            widgets["status"].set("")
            hole = hh.hole_cards.get(seat)
            if hole and seat != self._human_seat:
                for canvas, card in zip(widgets["cards"], hole):
                    draw_card_face(canvas, card)
            else:
                for canvas in widgets["cards"]:
                    draw_empty_slot(canvas)
        own = hh.hole_cards.get(self._human_seat) if self._human_seat is not None else None
        if own:
            for canvas, card in zip(self.hole_canvases, own):
                draw_card_face(canvas, card)
        self._cancel_pending_clear()
        self._pending_clear = self.after(
            int(SHOWDOWN_REVEAL_SECONDS * 1000), lambda: self._clear_after_showdown(result)
        )

    def _cancel_pending_clear(self) -> None:
        if self._pending_clear is not None:
            self.after_cancel(self._pending_clear)
            self._pending_clear = None

    def _clear_after_showdown(self, result: HandResult) -> None:
        """Runs SHOWDOWN_REVEAL_SECONDS after a hand ends. Guarded because
        it is a timer callback: leaving the table for the menu destroys
        this frame while the timer is still armed."""
        self._pending_clear = None
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        for widgets in self.seat_widgets.values():
            for canvas in widgets["cards"]:
                draw_empty_slot(canvas)
        for canvas in self.board_canvases:
            draw_empty_slot(canvas)
        for canvas in self.hole_canvases:
            draw_empty_slot(canvas)
        for seat, stack in result.hand_history.final_stacks.items():
            if stack == 0:
                self._hide_seat(seat)

    def _render_actions(self, observation: Observation, legal_actions: list[LegalAction]) -> None:
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.actions_frame, text="Tocca a te:").pack(side="left", padx=(0, 8))
        for la in legal_actions:
            if la.action_type in (ActionType.BET, ActionType.RAISE):
                self._add_amount_action(la, observation)
            else:
                label = _SIMPLE_ACTION_LABELS[la.action_type]
                if la.min_amount is not None:
                    label += f" ({la.min_amount})"
                ttk.Button(self.actions_frame, text=label, command=lambda la=la: self._submit(Action(la.action_type))).pack(
                    side="left", padx=4
                )

    def _add_amount_action(self, la: LegalAction, observation: Observation) -> None:
        """The bet/raise panel: a slider, a row of pot-fraction shortcuts,
        and a confirm button.

        The slider is also scrollable -- a wheel notch over it moves the
        bet by WHEEL_STEP_BIG_BLINDS big blinds -- because dragging a
        160-pixel scale to a precise chip count is hopeless, and the
        shortcuts only cover four sizings. Every route into the amount
        goes through `apply`, which is the single place that clamps to the
        legal [min_amount, max_amount] range.
        """
        assert la.min_amount is not None and la.max_amount is not None
        verb = "Bet" if la.action_type == ActionType.BET else "Raise"
        frame = ttk.Frame(self.actions_frame)
        frame.pack(side="left", padx=6)
        chosen = {"amount": la.min_amount}
        amount_text = tk.StringVar(value=f"{verb}: {la.min_amount}")

        def apply(amount: float) -> int:
            clamped = max(la.min_amount, min(la.max_amount, round(amount)))
            chosen["amount"] = clamped
            amount_text.set(f"{verb}: {clamped}")
            return clamped

        def on_move(raw_value: str) -> None:
            apply(float(raw_value))

        label = ttk.Label(frame, textvariable=amount_text)
        label.pack()

        scale = None
        if la.max_amount > la.min_amount:
            scale = ttk.Scale(
                frame, from_=la.min_amount, to=la.max_amount, orient="horizontal", length=180, command=on_move
            )
            scale.pack()

        def set_amount(amount: float) -> None:
            clamped = apply(amount)
            if scale is not None:
                # Fires on_move again, which is a harmless no-op re-apply --
                # on_move never writes back to the scale, so this cannot loop.
                scale.set(clamped)

        def nudge(direction: int) -> str:
            step = max(1, WHEEL_STEP_BIG_BLINDS * self.big_blind)
            set_amount(chosen["amount"] + direction * step)
            return "break"

        # X11 reports the wheel as Button-4 (up) and Button-5 (down);
        # Windows and macOS send <MouseWheel> with a signed delta. Each
        # sequence is bound to its own direction rather than one handler
        # inspecting `event.num`, because that field is not dependable --
        # this build reports 8 and 9 for a synthesised Button-4/5, which
        # silently inverted the scroll direction until it was measured.
        bindings = {
            "<Button-4>": lambda _e: nudge(1),
            "<Button-5>": lambda _e: nudge(-1),
            "<MouseWheel>": lambda e: nudge(1 if e.delta > 0 else -1),
        }
        scroll_targets = [frame, label] + ([scale] if scale is not None else [])
        for widget in scroll_targets:
            for sequence, handler in bindings.items():
                widget.bind(sequence, handler)

        presets = ttk.Frame(frame)
        presets.pack(pady=(2, 0))
        for fraction, text in POT_FRACTION_PRESETS:
            target = _pot_fraction_raise_to(observation, fraction)
            ttk.Button(
                presets, text=text, width=6, command=lambda t=target: set_amount(t)
            ).pack(side="left", padx=1)

        ttk.Button(
            frame,
            text=f"Conferma {verb.lower()}",
            command=lambda: self._submit(Action(la.action_type, amount=chosen["amount"])),
        ).pack(pady=(2, 0))

    def _submit(self, action: Action) -> None:
        assert self.human_player is not None
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
        self.waiting_var.set("I bot stanno giocando...")
        ttk.Label(self.actions_frame, textvariable=self.waiting_var).pack(side="left")
        self.human_player.decisions.put(action)


class PokerGuiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("pokerlab")
        self.geometry("1000x760")
        self.minsize(760, 560)
        self._content: ttk.Frame | None = None
        self.show_setup()

    def _show_frame(self, frame: ttk.Frame) -> None:
        if self._content is not None:
            self._content.destroy()
        self._content = frame
        frame.pack(fill="both", expand=True)

    def show_setup(self) -> None:
        self._show_frame(SetupFrame(self))

    def show_spot(self) -> None:
        from pokerlab.gui.spot_view import SpotFrame

        # The table and the side panel together are wider than the default window.
        self.geometry("1240x760")
        self._show_frame(SpotFrame(self))

    def start_session(
        self,
        config: GameConfig,
        human_seats: int,
        bot_keys: list[str] | None,
        hands: int,
        seed: int | None,
        step_mode: bool = False,
    ) -> None:
        rng = random.Random(seed)
        event_queue: queue.Queue[GuiEvent] = queue.Queue()
        step_gate: queue.Queue[None] = queue.Queue()
        step_mode_state: dict[str, bool] = {"on": step_mode}
        holder: dict[str, GuiPlayer] = {}

        def human_factory(player_id: str, name: str) -> GuiPlayer:
            gui_player = GuiPlayer(player_id, name, event_queue)
            holder["player"] = gui_player
            return gui_player

        base_players = build_players(
            config.num_players,
            human_seats,
            rng,
            bot_keys,
            human_player_factory=human_factory,
            # Needed only by `model:` specs: RLAgentPlayer normalises its
            # features by big_blind and starting_stack, which an
            # Observation deliberately does not carry.
            game=config,
        )
        players: list[Player] = list(base_players)
        seat_names = {seat: p.name for seat, p in enumerate(players)}
        # Per build_players' convention, a human always sits at seat 0.
        human_seat = 0 if human_seats else None

        history_dir = Path("hand_histories")
        history_path = history_dir / f"session_{int(rng.random() * 1_000_000):06d}.jsonl"
        writer = HandHistoryWriter(history_path)

        def on_hand_started(info: dict) -> None:
            event_queue.put(GuiEvent("hand_started", info))

        def on_street_dealt(info: dict) -> None:
            # Both sleeps run on the session thread, deliberately: pacing
            # here rather than inside Table keeps the engine free of
            # display concerns, exactly like ActionReporter does for
            # actions. Without the all-in branch the engine would deal
            # flop, turn and river and settle the hand between two of the
            # GUI's 100ms polls; without the ordinary one the new card and
            # the street's first action land in the same poll.
            event_queue.put(GuiEvent("street_dealt", info))
            time.sleep(ALL_IN_STREET_DELAY_SECONDS if info["betting_closed"] else NEW_STREET_DELAY_SECONDS)

        table = Table(
            config,
            players,
            rng=rng,
            history_writer=writer,
            on_hand_started=on_hand_started,
            on_street_dealt=on_street_dealt,
            on_action_applied=ActionReporter(event_queue, step_gate, step_mode_state, human_seat),
        )

        table_frame = TableFrame(
            self,
            config.num_players,
            event_queue,
            holder.get("player"),
            history_path,
            seat_names,
            step_gate,
            step_mode_state,
            big_blind=config.big_blind,
        )
        self._show_frame(table_frame)

        thread = threading.Thread(target=_run_session, args=(table, hands, event_queue, writer), daemon=True)
        thread.start()


def main() -> None:
    app = PokerGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
