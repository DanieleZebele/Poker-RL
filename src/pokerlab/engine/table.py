from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pokerlab.cards.deck import Deck
from pokerlab.engine.betting import (
    apply_action,
    compute_legal_actions,
    post_blinds,
    seats_clockwise_from,
    start_new_street_betting,
)
from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import SCHEMA_VERSION, HandHistory, HandHistoryWriter
from pokerlab.engine.pots import compute_pots, distribute_pots
from pokerlab.engine.state import HandState, PlayerState, PlayerStatus, Street

if TYPE_CHECKING:
    from pokerlab.players.base import Player


@dataclass
class HandResult:
    hand_id: str
    payouts: dict[int, int]
    final_stacks: dict[int, int]
    hand_history: HandHistory


class Table:
    """Drives full hands over a fixed list of Players, uniformly -- it never
    branches on whether a seat is a ManualPlayer, a ScriptedBot, or (later)
    an RL agent. Persists chip stacks and the button across hands."""

    def __init__(
        self,
        config: GameConfig,
        players: list[Player],
        rng: random.Random | None = None,
        history_writer: HandHistoryWriter | None = None,
        on_hand_started: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """`on_hand_started`, if given, is called once per hand right after
        blinds are posted (before any betting), with a dict of
        {hand_id, button_seat, sb_seat, bb_seat, small_blind, big_blind,
        hole_cards}. This is an optional spectator hook only -- e.g. a GUI
        showing blind postings and a "spy on opponents' cards" debug toggle
        -- nothing in the engine or in Player depends on it.
        """
        if len(players) != config.num_players:
            raise ValueError(
                f"config.num_players ({config.num_players}) does not match len(players) ({len(players)})"
            )
        self.config = config
        self.players = players
        self._rng = rng if rng is not None else random.Random()
        self._history_writer = history_writer
        self._on_hand_started = on_hand_started
        self.stacks: list[int] = [config.starting_stack] * config.num_players
        self._button_seat: int | None = None
        self._hand_counter = 0

    def _eligible_seats(self) -> list[int]:
        return [seat for seat, stack in enumerate(self.stacks) if stack > 0]

    def _advance_button(self, eligible: list[int]) -> int:
        if self._button_seat is None or self._button_seat not in eligible:
            return min(eligible)
        pos = eligible.index(self._button_seat)
        return eligible[(pos + 1) % len(eligible)]

    def play_hand(self) -> HandResult:
        eligible = self._eligible_seats()
        if len(eligible) < 2:
            raise RuntimeError("cannot play a hand with fewer than 2 players holding chips")

        button_seat = self._advance_button(eligible)
        self._button_seat = button_seat
        self._hand_counter += 1
        hand_id = f"hand-{self._hand_counter}-{uuid.uuid4().hex[:8]}"

        seats = [
            PlayerState(
                seat=seat,
                player_id=self.players[seat].player_id,
                name=self.players[seat].name,
                stack=self.stacks[seat],
            )
            for seat in eligible
        ]
        deck = Deck(rng=self._rng)
        deck.shuffle()
        for ps in seats:
            ps.hole_cards = (deck.deal(1)[0], deck.deal(1)[0])

        hand_state = HandState(
            hand_id=hand_id,
            button_seat=button_seat,
            seats=seats,
            deck=deck,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )
        starting_stacks = {ps.seat: ps.stack for ps in seats}
        hole_cards = {ps.seat: ps.hole_cards for ps in seats}

        order = seats_clockwise_from(hand_state, button_seat)
        if len(eligible) == 2:
            sb_seat, bb_seat = order[0], order[1]
        else:
            sb_seat, bb_seat = order[1], order[2]
        post_blinds(hand_state, sb_seat, bb_seat)

        if self._on_hand_started is not None:
            self._on_hand_started(
                {
                    "hand_id": hand_id,
                    "button_seat": button_seat,
                    "sb_seat": sb_seat,
                    "bb_seat": bb_seat,
                    "small_blind": self.config.small_blind,
                    "big_blind": self.config.big_blind,
                    "hole_cards": dict(hole_cards),
                }
            )

        preflop_first_actor = sb_seat if len(eligible) == 2 else order[3 % len(order)]
        self._run_betting_round(hand_state, preflop_first_actor)

        streets: list[tuple[Street, int]] = [(Street.FLOP, 3), (Street.TURN, 1), (Street.RIVER, 1)]
        for street, num_cards in streets:
            if len(hand_state.hand_active_seats()) <= 1:
                break
            hand_state.community_cards.extend(hand_state.deck.deal(num_cards))
            start_new_street_betting(hand_state, street)
            postflop_first_actor = seats_clockwise_from(hand_state, button_seat + 1)[0]
            self._run_betting_round(hand_state, postflop_first_actor)

        hand_state.street = Street.SHOWDOWN
        pots = compute_pots(hand_state.seats)
        payouts = distribute_pots(pots, hand_state.seats, hand_state.community_cards, button_seat)

        for ps in hand_state.seats:
            self.stacks[ps.seat] = ps.stack

        hand_history = HandHistory(
            schema_version=SCHEMA_VERSION,
            hand_id=hand_id,
            started_at=time.time(),
            num_players=len(eligible),
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            button_seat=button_seat,
            starting_stacks=starting_stacks,
            seat_names={ps.seat: ps.name for ps in seats},
            community_cards=list(hand_state.community_cards),
            actions=list(hand_state.action_log),
            hole_cards=hole_cards,
            payouts=payouts,
            final_stacks={ps.seat: ps.stack for ps in hand_state.seats},
        )
        if self._history_writer is not None:
            self._history_writer.append(hand_history)

        return HandResult(
            hand_id=hand_id,
            payouts=payouts,
            final_stacks=hand_history.final_stacks,
            hand_history=hand_history,
        )

    def play_session(self, num_hands: int) -> list[HandResult]:
        results = []
        for _ in range(num_hands):
            if len(self._eligible_seats()) < 2:
                break
            results.append(self.play_hand())
        return results

    def _run_betting_round(self, hand_state: HandState, first_actor_seat: int) -> None:
        if len(hand_state.actionable_seats()) <= 1:
            hand_state.to_act = set()
            return

        from pokerlab.players.base import build_observation  # local import: avoid a module cycle

        order = seats_clockwise_from(hand_state, first_actor_seat)
        max_iterations = 1000
        iterations = 0
        while hand_state.to_act:
            progressed = False
            for seat in order:
                if seat not in hand_state.to_act:
                    continue
                ps = hand_state.seat_state(seat)
                if ps.status != PlayerStatus.ACTIVE or ps.stack <= 0:
                    hand_state.to_act.discard(seat)
                    continue
                legal = compute_legal_actions(hand_state, seat)
                observation = build_observation(hand_state, seat)
                action = self.players[seat].act(observation, legal)
                apply_action(hand_state, seat, action)
                progressed = True
                if len(hand_state.hand_active_seats()) <= 1:
                    hand_state.to_act = set()
                    return
            iterations += 1
            if not progressed or iterations > max_iterations:
                break
