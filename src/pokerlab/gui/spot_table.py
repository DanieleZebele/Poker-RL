"""The seating of the "ask the models" table: nine chairs round an oval.

Pure Python, no Tkinter, so the part that decides who acts after whom is
testable without a display.

There are nine chairs, numbered clockwise as seen on screen starting from the one
at the bottom, which is always yours. The player holding the dealer button is the
one the engine calls seat 0 (`gui/spot.py` documents why: `Table` puts the button
on the lowest seat on a table's first hand), and the occupied chairs read
clockwise from the dealer are seats 0, 1, 2, ... -- so placing the button *is*
describing the order of play, and nobody has to type a seat number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CHAIRS = 9
# The chair at the bottom of the table. The user is always seated, so it is never
# removed, and it is where "your cards" live.
USER_CHAIR = 0


# Where a seat of the poker client sits among these nine chairs, by table size:
# client seat k (0 = you at the bottom, then clockwise on screen) goes to the
# chair nearest its angle. 6-max seats are 60 degrees apart and chairs 40, so
# seats 2 and 4 land exactly (chairs 3, 6), seats 1 and 5 symmetrically 20
# degrees off (chairs 2, 7), and seat 3, straight across, between two chairs:
# chair 4 was picked, 5 would be as good.
CLIENT_SEAT_CHAIRS = {6: (0, 2, 3, 4, 6, 7)}


def chair_for_client_seat(players: int, seat: int) -> int:
    return CLIENT_SEAT_CHAIRS[players][seat]


def chair_position(
    chair: int, center: tuple[float, float], radii: tuple[float, float]
) -> tuple[float, float]:
    """Where a chair sits on screen (y grows downwards).

    Chair 0 is at the bottom and the numbers increase clockwise *as seen on
    screen*: from the bottom the next chair is to the left, then up the left side,
    across the top and down the right.
    """
    angle = 2 * math.pi * chair / CHAIRS
    return (
        center[0] - radii[0] * math.sin(angle),
        center[1] + radii[1] * math.cos(angle),
    )


@dataclass
class TableLayout:
    """Which chairs are occupied and who has the button."""

    chairs: set[int] = field(default_factory=lambda: {USER_CHAIR})
    dealer: int = USER_CHAIR

    def add(self, chair: int) -> bool:
        """Seat someone. False when the chair does not exist or is taken."""
        if not 0 <= chair < CHAIRS or chair in self.chairs:
            return False
        self.chairs.add(chair)
        return True

    def remove(self, chair: int) -> bool:
        """Empty a chair. The user's chair cannot be emptied; if the button was
        on the one removed it goes back to the user."""
        if chair == USER_CHAIR or chair not in self.chairs:
            return False
        self.chairs.discard(chair)
        if self.dealer == chair:
            self.dealer = USER_CHAIR
        return True

    def set_dealer(self, chair: int) -> bool:
        if chair not in self.chairs:
            return False
        self.dealer = chair
        return True

    def order(self) -> list[int]:
        """Occupied chairs clockwise starting from the dealer: seat 0, 1, 2, ..."""
        return sorted(self.chairs, key=lambda chair: (chair - self.dealer) % CHAIRS)

    @property
    def players(self) -> int:
        return len(self.chairs)

    def seat_of(self, chair: int) -> int:
        return self.order().index(chair)

    def chair_of(self, seat: int) -> int:
        return self.order()[seat]

    def free_chairs(self) -> list[int]:
        return [chair for chair in range(CHAIRS) if chair not in self.chairs]
