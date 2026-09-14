from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.state import ActionRecord, HandState, PlayerStatus, Street


@dataclass(frozen=True)
class SeatPublicInfo:
    seat: int
    name: str
    stack: int
    current_bet: int
    status: PlayerStatus
    is_button: bool


@dataclass(frozen=True)
class Observation:
    """Everything a Player may legally see when it is asked to act: full
    public state, plus its own hole cards and stack. Deliberately a plain,
    JSON-friendly dataclass of primitives -- NOT a tensor. Encoding this
    into a model-ready representation is the job of the future RL env
    wrapper (see pokerlab.rl.env), not of the engine or of Player itself.
    """

    street: Street
    hole_cards: tuple[Card, Card]
    community_cards: tuple[Card, ...]
    pot_size: int
    current_bet_to_match: int
    min_raise: int
    my_seat: int
    my_stack: int
    my_current_bet: int
    seats: tuple[SeatPublicInfo, ...]
    button_seat: int
    action_history: tuple[ActionRecord, ...]  # this hand only, so far


def build_observation(hand_state: HandState, seat: int) -> Observation:
    ps = hand_state.seat_state(seat)
    assert ps.hole_cards is not None, "observation requested for a seat with no hole cards dealt"
    seats = tuple(
        SeatPublicInfo(
            seat=other.seat,
            name=other.name,
            stack=other.stack,
            current_bet=other.current_bet,
            status=other.status,
            is_button=(other.seat == hand_state.button_seat),
        )
        for other in sorted(hand_state.seats, key=lambda p: p.seat)
    )
    return Observation(
        street=hand_state.street,
        hole_cards=ps.hole_cards,
        community_cards=tuple(hand_state.community_cards),
        pot_size=hand_state.pot_total(),
        current_bet_to_match=hand_state.current_bet_to_match,
        min_raise=hand_state.min_raise,
        my_seat=seat,
        my_stack=ps.stack,
        my_current_bet=ps.current_bet,
        seats=seats,
        button_seat=hand_state.button_seat,
        action_history=tuple(hand_state.action_log),
    )


class Player(ABC):
    """The single extension point for every kind of decision-maker: a human
    at the terminal, a scripted bot, and (in a future session) an RL agent
    or a GUI-driven player. The engine only ever calls `act`; it never
    branches on what kind of Player it is talking to.
    """

    def __init__(self, player_id: str, name: str) -> None:
        self.player_id = player_id
        self.name = name

    @abstractmethod
    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action: ...

    def notify(self, event: str, **data: object) -> None:
        """Optional hook for things like 'new_hand', 'opponent_action',
        'showdown'. No-op by default; ManualPlayer overrides it to render
        updates to the terminal."""
