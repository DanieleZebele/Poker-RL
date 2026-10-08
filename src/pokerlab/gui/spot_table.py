"""The seating of the "ask the models" table: eight chairs round the screen's edges.

Pure Python, no Tkinter, so the part that decides who acts after whom is
testable without a display.

There are eight chairs, numbered clockwise as seen on screen starting from the one
at the bottom, which is always yours. The player holding the dealer button is the
one the engine calls seat 0 (`gui/spot.py` documents why: `Table` puts the button
on the lowest seat on a table's first hand), and the occupied chairs read
clockwise from the dealer are seats 0, 1, 2, ... -- so placing the button *is*
describing the order of play, and nobody has to type a seat number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHAIRS = 8
# The chair at the bottom of the table. The user is always seated, so it is never
# removed, and it is where "your cards" live.
USER_CHAIR = 0


# Where a seat of the poker client sits among these eight chairs, by table size:
# client seat k (0 = you at the bottom, then clockwise on screen) goes to the
# chair nearest its angle. At 8-max the client's seats are the chairs themselves,
# 45 degrees apart like them. 6-max seats are 60 degrees apart, so seat 3,
# straight across, lands exactly (chair 4) and the others 15 degrees off theirs,
# symmetrically (chairs 1, 3 on the left, 5, 7 on the right).
CLIENT_SEAT_CHAIRS = {6: (0, 1, 3, 4, 5, 7), 8: tuple(range(8))}


def chair_for_client_seat(players: int, seat: int) -> int:
    return CLIENT_SEAT_CHAIRS[players][seat]


# Where each chair's box sits on the canvas, as fractions of its width and height,
# and which corner or side of the box is pinned there. Chair 0 is at the bottom and
# the numbers increase clockwise *as seen on screen*: bottom-left corner, the middle
# of the left side, top-left corner, top, and down the right -- a square round the
# felt rather than an oval. Each box is pinned by the side facing the edge, so it
# grows inwards (a box at the top grows down when its actions appear) and the boxes
# never cover one another as long as the canvas holds three of them a side.
_SLOTS = {
    0: (0.5, 1.0, "s"),
    1: (0.0, 1.0, "sw"),
    2: (0.0, 0.5, "w"),
    3: (0.0, 0.0, "nw"),
    4: (0.5, 0.0, "n"),
    5: (1.0, 0.0, "ne"),
    6: (1.0, 0.5, "e"),
    7: (1.0, 1.0, "se"),
}
assert len(_SLOTS) == CHAIRS


def chair_slot(chair: int, size: tuple[float, float], margin: float = 0) -> tuple[float, float, str]:
    """`(x, y, anchor)` of a chair's box on a canvas of `size` (y grows downwards),
    `margin` in from the edges; `anchor` is Tk's, the point of the box placed there."""
    fx, fy, anchor = _SLOTS[chair]
    width, height = size
    return margin + fx * (width - 2 * margin), margin + fy * (height - 2 * margin), anchor


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
        on the one removed it passes to the player before it (`previous_occupied`)."""
        if chair == USER_CHAIR or chair not in self.chairs:
            return False
        self.chairs.discard(chair)
        if self.dealer == chair:
            self.dealer = self.previous_occupied(chair)
        return True

    def previous_occupied(self, chair: int) -> int:
        """The first occupied chair counter-clockwise from `chair` (not `chair` itself):
        where the button goes when its player has gone. The blinds then stay on the
        same two players -- the first occupied chairs after the empty one -- which is
        what a client's dead button does too. Your chair is always occupied, so
        there is always one."""
        for step in range(1, CHAIRS):
            candidate = (chair - step) % CHAIRS
            if candidate in self.chairs:
                return candidate
        return USER_CHAIR

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
