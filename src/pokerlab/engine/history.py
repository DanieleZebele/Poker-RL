from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pokerlab.cards.card import Card
from pokerlab.engine.actions import ActionType
from pokerlab.engine.state import ActionRecord, Street

SCHEMA_VERSION = 1


@dataclass
class HandHistory:
    """A fully self-contained record of one played hand: enough to replay
    or feed into a future RL training data pipeline without needing any
    other session state."""

    schema_version: int
    hand_id: str
    started_at: float
    num_players: int
    small_blind: int
    big_blind: int
    button_seat: int
    starting_stacks: dict[int, int]
    seat_names: dict[int, str]
    community_cards: list[Card]
    actions: list[ActionRecord]
    hole_cards: dict[int, tuple[Card, Card]]
    payouts: dict[int, int]
    final_stacks: dict[int, int]


def _record_to_dict(r: ActionRecord) -> dict[str, Any]:
    return {
        "street": r.street.value,
        "seat": r.seat,
        "player_id": r.player_id,
        "action_type": r.action_type.value,
        "amount": r.amount,
        "stack_before": r.stack_before,
        "stack_after": r.stack_after,
        "pot_before": r.pot_before,
        "timestamp": r.timestamp,
    }


def _record_from_dict(d: dict[str, Any]) -> ActionRecord:
    return ActionRecord(
        street=Street(d["street"]),
        seat=d["seat"],
        player_id=d["player_id"],
        action_type=ActionType(d["action_type"]),
        amount=d["amount"],
        stack_before=d["stack_before"],
        stack_after=d["stack_after"],
        pot_before=d["pot_before"],
        timestamp=d["timestamp"],
    )


def _to_jsonable(hh: HandHistory) -> dict[str, Any]:
    """Manual (not dataclasses.asdict) conversion, since asdict would
    recursively flatten Card into {rank, suit} instead of a compact "Ah"
    string, which is what we want for a hand-history file meant to stay
    human-readable and simple to re-parse."""
    return {
        "schema_version": hh.schema_version,
        "hand_id": hh.hand_id,
        "started_at": hh.started_at,
        "num_players": hh.num_players,
        "small_blind": hh.small_blind,
        "big_blind": hh.big_blind,
        "button_seat": hh.button_seat,
        "starting_stacks": {str(k): v for k, v in hh.starting_stacks.items()},
        "seat_names": {str(k): v for k, v in hh.seat_names.items()},
        "community_cards": [str(c) for c in hh.community_cards],
        "actions": [_record_to_dict(r) for r in hh.actions],
        "hole_cards": {str(k): [str(v[0]), str(v[1])] for k, v in hh.hole_cards.items()},
        "payouts": {str(k): v for k, v in hh.payouts.items()},
        "final_stacks": {str(k): v for k, v in hh.final_stacks.items()},
    }


def _hand_history_from_dict(d: dict[str, Any]) -> HandHistory:
    def card_pair(pair: list[str]) -> tuple[Card, Card]:
        return (Card.parse(pair[0]), Card.parse(pair[1]))

    return HandHistory(
        schema_version=d["schema_version"],
        hand_id=d["hand_id"],
        started_at=d["started_at"],
        num_players=d["num_players"],
        small_blind=d["small_blind"],
        big_blind=d["big_blind"],
        button_seat=d["button_seat"],
        starting_stacks={int(k): v for k, v in d["starting_stacks"].items()},
        seat_names={int(k): v for k, v in d["seat_names"].items()},
        community_cards=[Card.parse(c) for c in d["community_cards"]],
        actions=[_record_from_dict(r) for r in d["actions"]],
        hole_cards={int(k): card_pair(v) for k, v in d["hole_cards"].items()},
        payouts={int(k): v for k, v in d["payouts"].items()},
        final_stacks={int(k): v for k, v in d["final_stacks"].items()},
    )


class HandHistoryWriter:
    """Appends hands as JSON Lines, one hand per line, flushing after every
    write so a crash mid-session never loses more than the in-flight hand."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def append(self, hand_history: HandHistory) -> None:
        self._file.write(json.dumps(_to_jsonable(hand_history)) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class HandHistoryReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_hands(self):
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield _hand_history_from_dict(json.loads(line))
