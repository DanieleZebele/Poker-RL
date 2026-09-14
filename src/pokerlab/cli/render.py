from __future__ import annotations

from pokerlab.engine.actions import ActionType, LegalAction
from pokerlab.engine.state import PlayerStatus
from pokerlab.players.base import Observation

_STATUS_LABEL = {
    PlayerStatus.ACTIVE: "",
    PlayerStatus.FOLDED: "folded",
    PlayerStatus.ALL_IN: "all-in",
    PlayerStatus.BUSTED: "out",
}


def render_observation(observation: Observation) -> str:
    lines = [
        f"-- {observation.street.value.upper()} -- pot: {observation.pot_size}",
        f"board: {' '.join(str(c) for c in observation.community_cards) or '(none yet)'}",
        (
            f"your hand: {' '.join(str(c) for c in observation.hole_cards)}"
            f"  (seat {observation.my_seat}, stack {observation.my_stack}, bet {observation.my_current_bet})"
        ),
    ]
    for seat in observation.seats:
        marker = " (button)" if seat.is_button else ""
        marker += " (you)" if seat.seat == observation.my_seat else ""
        status = f" [{_STATUS_LABEL[seat.status]}]" if _STATUS_LABEL[seat.status] else ""
        lines.append(f"  seat {seat.seat} {seat.name}: stack {seat.stack}, bet {seat.current_bet}{status}{marker}")
    return "\n".join(lines)


def render_legal_actions(legal_actions: list[LegalAction]) -> str:
    parts = []
    for i, la in enumerate(legal_actions):
        if la.action_type in (ActionType.BET, ActionType.RAISE):
            parts.append(f"[{i}] {la.action_type.value} ({la.min_amount}-{la.max_amount})")
        elif la.action_type in (ActionType.CALL, ActionType.ALL_IN) and la.min_amount is not None:
            parts.append(f"[{i}] {la.action_type.value} ({la.min_amount})")
        else:
            parts.append(f"[{i}] {la.action_type.value}")
    return "  ".join(parts)
