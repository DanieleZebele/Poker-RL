"""Design-only stub for a future live table-recognition module: capturing a
poker application's window and identifying cards/players from the image.

NOT IMPLEMENTED in this session -- signatures and docstrings only. Building
this requires the `vision` extra (`pip install -e ".[vision]"`, i.e.
opencv-python/numpy/mss), which is deliberately not part of the core
install.

Intended eventual usage: a TableStateReader implementation would run
alongside a live Table (or independently, just to log what it sees),
periodically calling capture_frame() + read_table_state() and turning the
result into either a pokerlab.players.base.Observation (to drive an RL
agent playing on a real client) or into a HandHistory-compatible record
(to build a dataset from watched games). Neither of those integration
points requires changes to pokerlab.engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pokerlab.cards.card import Card
    from pokerlab.players.base import SeatPublicInfo


@dataclass(frozen=True)
class RecognizedTableState:
    community_cards: list[Card] | None
    hole_cards: tuple[Card, Card] | None
    seats: list[SeatPublicInfo]
    pot_size: int | None
    confidence: float


class TableStateReader(ABC):
    @abstractmethod
    def capture_frame(self) -> Any:  # -> np.ndarray, once opencv/numpy are dependencies
        raise NotImplementedError("Vision section not implemented yet -- see module docstring")

    @abstractmethod
    def read_table_state(self, frame: Any) -> RecognizedTableState:
        raise NotImplementedError("Vision section not implemented yet -- see module docstring")
