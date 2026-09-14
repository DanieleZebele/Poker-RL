from __future__ import annotations

import os
import queue
import random
import sys
import threading
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

from pokerlab.cli.play import CUSTOM_PARAM_DEFAULTS, build_players, validate_bot_key
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import HandHistoryWriter
from pokerlab.engine.state import PlayerStatus
from pokerlab.engine.table import HandResult, Table
from pokerlab.gui.cards_canvas import (
    draw_card_back,
    draw_card_face,
    draw_empty_slot,
    new_card_canvas,
)
from pokerlab.players.base import Observation, Player
from pokerlab.players.gui import GuiEvent, GuiPlayer, SteppingPlayer
from pokerlab.players.scripted import get_bot_profile, list_bot_profiles

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


def _describe_action(observation: Observation, action: Action) -> str:
    """Human-readable description of an action for the live log. CALL and
    ALL_IN ignore Action.amount (the engine derives it from state -- see
    Action's docstring), so the actual chip amount is computed here from
    the Observation instead of read off the Action."""
    if action.action_type == ActionType.FOLD:
        return "fold"
    if action.action_type == ActionType.CHECK:
        return "check"
    if action.action_type == ActionType.CALL:
        return f"call {observation.current_bet_to_match - observation.my_current_bet}"
    if action.action_type == ActionType.ALL_IN:
        return f"all-in ({observation.my_stack})"
    if action.action_type == ActionType.BET:
        return f"bet {action.amount}"
    if action.action_type == ActionType.RAISE:
        return f"raise to {action.amount}"
    return action.action_type.value


def _format_hand_summary(result: HandResult) -> str:
    hh = result.hand_history
    names = hh.seat_names
    lines = [f"--- fine mano {result.hand_id} ---"]
    folded = {a.seat for a in hh.actions if a.action_type == ActionType.FOLD}
    contested = [s for s in hh.starting_stacks if s not in folded]
    if len(contested) >= 2:
        lines.append("Showdown:")
        for s in contested:
            hole = hh.hole_cards.get(s)
            card_text = " ".join(str(c) for c in hole) if hole else "?"
            lines.append(f"  {names[s]}: {card_text}")
    winners = ", ".join(f"{names[s]} (+{amt})" for s, amt in result.payouts.items())
    lines.append(f"Pot vinto da: {winners}")
    stacks = ", ".join(f"{names[s]}: {stack}" for s, stack in result.final_stacks.items())
    lines.append(f"Stack finali: {stacks}")
    return "\n".join(lines)


def _run_session(table: Table, num_hands: int, event_queue: queue.Queue[GuiEvent], writer: HandHistoryWriter) -> None:
    """Runs on a background thread: drives the actual game loop. This
    thread can be blocked by a GuiPlayer.act() call waiting on the human's
    next click, or (in step mode) by a SteppingPlayer waiting on the
    "Avanti" button -- there is no separate "pause"/"stop mid-hand"
    mechanism beyond that; closing the window (a daemon thread) is how a
    session gets abandoned.
    """
    try:
        for i in range(num_hands):
            if sum(1 for s in table.stacks if s > 0) < 2:
                event_queue.put(GuiEvent("session_ended_early", i))
                return
            result = table.play_hand()
            event_queue.put(GuiEvent("hand_complete", result))
        event_queue.put(GuiEvent("session_complete"))
    finally:
        writer.close()


def _bot_spec_to_key_string(spec: dict) -> str:
    """Turn a setup-screen bot spec ({"key": "shark"} or {"key": "custom",
    "params": {...}}) back into the plain string BOT_CATALOG keys / the
    CLI's --bots already understand, so build_players/validate_bot_key need
    no changes at all to support the visual bot builder below."""
    if spec["key"] != "custom":
        return spec["key"]
    parts = ";".join(f"{name}={value}" for name, value in spec["params"].items())
    return f"custom:{parts}"


def _bot_spec_label(spec: dict) -> str:
    if spec["key"] != "custom":
        profile = get_bot_profile(spec["key"])
        return f"[{profile.difficulty}] {profile.label}"
    p = spec["params"]
    return f"Custom\nT={p['tightness']:.2f}  A={p['aggression']:.2f}\nB={p['bluff_frequency']:.2f}  V={p['size_variance']:.2f}"


class AddBotDialog(tk.Toplevel):
    """Modal dialog opened by the "+" button: pick a catalog bot, or
    "Custom" to reveal four sliders for tightness/aggression/bluff_frequency/
    size_variance (see make_heuristic_bot's docstring for what each means)."""

    def __init__(self, master: tk.Widget, on_confirm) -> None:
        super().__init__(master)
        self.title("Aggiungi bot")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self._on_confirm = on_confirm

        self.choice_var = tk.StringVar(value=list_bot_profiles()[0].key)
        ttk.Label(self, text="Scegli il tipo di bot:", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        for profile in list_bot_profiles():
            ttk.Radiobutton(
                self,
                text=f"[{profile.difficulty}] {profile.label} -- {profile.description}",
                variable=self.choice_var,
                value=profile.key,
                command=self._on_choice_changed,
            ).pack(anchor="w", padx=24)
        ttk.Radiobutton(
            self,
            text="Custom (imposta i parametri manualmente)",
            variable=self.choice_var,
            value="custom",
            command=self._on_choice_changed,
        ).pack(anchor="w", padx=24, pady=(4, 0))

        self.custom_frame = ttk.Frame(self, padding=(24, 8, 12, 0))
        self.param_vars: dict[str, tk.DoubleVar] = {}
        self.param_labels: dict[str, tk.StringVar] = {}
        for name, default in CUSTOM_PARAM_DEFAULTS.items():
            var = tk.DoubleVar(value=default)
            label_var = tk.StringVar(value=f"{default:.2f}")
            self.param_vars[name] = var
            self.param_labels[name] = label_var
            row = ttk.Frame(self.custom_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=15).pack(side="left")
            ttk.Scale(
                row, from_=0.0, to=1.0, orient="horizontal", length=160,
                command=lambda raw, n=name: self._on_param_moved(n, raw),
            ).pack(side="left", padx=6)
            ttk.Label(row, textvariable=label_var, width=5).pack(side="left")

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=12, pady=12)
        ttk.Button(buttons, text="Annulla", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Aggiungi", command=self._confirm).pack(side="right", padx=(0, 8))

        self._on_choice_changed()

    def _on_param_moved(self, name: str, raw_value: str) -> None:
        value = round(float(raw_value), 2)
        self.param_vars[name].set(value)
        self.param_labels[name].set(f"{value:.2f}")

    def _on_choice_changed(self) -> None:
        if self.choice_var.get() == "custom":
            self.custom_frame.pack(fill="x")
        else:
            self.custom_frame.pack_forget()

    def _confirm(self) -> None:
        key = self.choice_var.get()
        if key == "custom":
            params = {name: round(var.get(), 2) for name, var in self.param_vars.items()}
            self._on_confirm({"key": "custom", "params": params})
        else:
            self._on_confirm({"key": key})
        self.destroy()


class SetupFrame(ttk.Frame):
    def __init__(self, master: PokerGuiApp) -> None:
        super().__init__(master, padding=16)
        self.app = master
        self.bot_specs: list[dict] = [{"key": profile.key} for profile in list_bot_profiles()]

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
    ) -> None:
        super().__init__(master, padding=12)
        self.app = master
        self.event_queue = event_queue
        self.human_player = human_player
        self.seat_names = seat_names
        self.step_gate = step_gate
        self.step_mode_state = step_mode_state
        self._all_hole_cards: dict[int, tuple] = {}
        self._last_observation: Observation | None = None
        self._busted_seats: set[int] = set()
        # Per build_players' convention, a human always sits at seat 0.
        self._human_seat: int | None = 0 if human_player is not None else None

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
            box = ttk.LabelFrame(seats_frame, text=f"Seat {seat}")
            box.grid(row=0, column=seat, padx=4, sticky="n")
            name_var = tk.StringVar(value="-")
            stack_var = tk.StringVar(value="-")
            bet_var = tk.StringVar(value="")
            status_var = tk.StringVar(value="")
            for var in (name_var, stack_var, bet_var, status_var):
                ttk.Label(box, textvariable=var, width=16).pack(anchor="w")
            cards_frame = ttk.Frame(box)
            cards_frame.pack(anchor="w", pady=(2, 0))
            back_canvases = [new_card_canvas(cards_frame) for _ in range(2)]
            for canvas in back_canvases:
                canvas.pack(side="left", padx=1)
            self.seat_widgets[seat] = {
                "frame": box,
                "name": name_var,
                "stack": stack_var,
                "bet": bet_var,
                "status": status_var,
                "cards": back_canvases,
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
        self.menu_button = ttk.Button(footer, text="Torna al menu", command=self.app.show_setup, state="disabled")
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

    def _log_action(self, name: str, observation: Observation, action: Action) -> None:
        self._log(f"[{observation.street.value.upper()}] {name}: {_describe_action(observation, action)}")

    def _on_spy_toggled(self) -> None:
        if self._last_observation is not None:
            for seat_info in self._last_observation.seats:
                self._draw_seat_cards(seat_info, is_me=(seat_info.seat == self._last_observation.my_seat))

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
            self._all_hole_cards = event.payload["hole_cards"]
            self._reset_seats_for_new_hand(self._all_hole_cards.keys())
            self._log_hand_start(event.payload)
        elif event.kind == "your_turn":
            observation, legal_actions = event.payload
            self.waiting_var.set("")
            self._render_observation(observation)
            self._render_actions(legal_actions)
        elif event.kind == "action_taken":
            _player_id, name, observation, action = event.payload
            self._render_observation(observation)
            self._log_action(name, observation, action)
        elif event.kind == "hand_complete":
            self._log(_format_hand_summary(event.payload))
            self._render_final_stacks(event.payload)
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
        for i, canvas in enumerate(self.board_canvases):
            if i < len(observation.community_cards):
                draw_card_face(canvas, observation.community_cards[i])
            else:
                draw_empty_slot(canvas)
        for canvas, card in zip(self.hole_canvases, observation.hole_cards):
            draw_card_face(canvas, card)

        present_seats = {s.seat for s in observation.seats}
        for seat in self.seat_widgets:
            if seat not in present_seats:
                self._hide_seat(seat)
        for seat_info in observation.seats:
            widgets = self.seat_widgets[seat_info.seat]
            marker = " (D)" if seat_info.is_button else ""
            marker += " (tu)" if seat_info.seat == observation.my_seat else ""
            widgets["name"].set(seat_info.name + marker)
            widgets["stack"].set(f"Stack: {seat_info.stack}")
            widgets["bet"].set(f"Bet: {seat_info.current_bet}")
            widgets["status"].set(_STATUS_LABELS.get(seat_info.status, ""))
            self._draw_seat_cards(seat_info, is_me=(seat_info.seat == observation.my_seat))

    def _render_final_stacks(self, result: HandResult) -> None:
        hh = result.hand_history
        for seat, widgets in self.seat_widgets.items():
            for canvas in widgets["cards"]:
                draw_empty_slot(canvas)
            if seat not in hh.final_stacks:
                self._hide_seat(seat)
                continue
            widgets["name"].set(hh.seat_names[seat])
            widgets["stack"].set(f"Stack: {hh.final_stacks[seat]}")
            widgets["bet"].set("")
            widgets["status"].set("")
            if hh.final_stacks[seat] == 0:
                self._hide_seat(seat)
        for canvas in self.board_canvases:
            draw_empty_slot(canvas)
        for canvas in self.hole_canvases:
            draw_empty_slot(canvas)

    def _render_actions(self, legal_actions: list[LegalAction]) -> None:
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.actions_frame, text="Tocca a te:").pack(side="left", padx=(0, 8))
        for la in legal_actions:
            if la.action_type in (ActionType.BET, ActionType.RAISE):
                self._add_amount_action(la)
            else:
                label = _SIMPLE_ACTION_LABELS[la.action_type]
                if la.min_amount is not None:
                    label += f" ({la.min_amount})"
                ttk.Button(self.actions_frame, text=label, command=lambda la=la: self._submit(Action(la.action_type))).pack(
                    side="left", padx=4
                )

    def _add_amount_action(self, la: LegalAction) -> None:
        assert la.min_amount is not None and la.max_amount is not None
        verb = "Bet" if la.action_type == ActionType.BET else "Raise"
        frame = ttk.Frame(self.actions_frame)
        frame.pack(side="left", padx=6)
        chosen = {"amount": la.min_amount}
        amount_text = tk.StringVar(value=f"{verb}: {la.min_amount}")

        def on_move(raw_value: str) -> None:
            chosen["amount"] = round(float(raw_value))
            amount_text.set(f"{verb}: {chosen['amount']}")

        ttk.Label(frame, textvariable=amount_text).pack()
        if la.max_amount > la.min_amount:
            ttk.Scale(frame, from_=la.min_amount, to=la.max_amount, orient="horizontal", length=160, command=on_move).pack()
        ttk.Button(
            frame,
            text=f"Conferma {verb.lower()}",
            command=lambda: self._submit(Action(la.action_type, amount=chosen["amount"])),
        ).pack()

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

        base_players = build_players(config.num_players, human_seats, rng, bot_keys, human_player_factory=human_factory)
        players: list[Player] = [
            p if seat < human_seats else SteppingPlayer(p, event_queue, step_gate, step_mode=lambda: step_mode_state["on"])
            for seat, p in enumerate(base_players)
        ]
        seat_names = {seat: p.name for seat, p in enumerate(players)}

        history_dir = Path("hand_histories")
        history_path = history_dir / f"session_{int(rng.random() * 1_000_000):06d}.jsonl"
        writer = HandHistoryWriter(history_path)

        def on_hand_started(info: dict) -> None:
            event_queue.put(GuiEvent("hand_started", info))

        table = Table(config, players, rng=rng, history_writer=writer, on_hand_started=on_hand_started)

        table_frame = TableFrame(
            self,
            config.num_players,
            event_queue,
            holder.get("player"),
            history_path,
            seat_names,
            step_gate,
            step_mode_state,
        )
        self._show_frame(table_frame)

        thread = threading.Thread(target=_run_session, args=(table, hands, event_queue, writer), daemon=True)
        thread.start()


def main() -> None:
    app = PokerGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
