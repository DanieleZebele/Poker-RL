"""Dealing the cards you ask for, through the engine's own dealing.

Used where a situation has to be rebuilt with chosen cards: the spot screen (`gui/spot.py`)
and the search (`rl/search.py`). Nothing in the engine knows about it: it is a
`random.Random` that `Table` is given in place of its own.
"""

from __future__ import annotations

import random

from pokerlab.cards.card import Card


class ArrangingRandom(random.Random):
    """A `random.Random` whose `shuffle` deals the cards you asked for.

    `Deck.shuffle()` calls `rng.shuffle(cards)` on a freshly ordered 52-card
    list, and that is the only thing a hand uses its rng for. Overriding it is
    therefore enough to control the whole deal without any engine change, and it
    keeps the engine dealing normally rather than being handed a rigged deck.

    `placements` maps a position in the deal order to a card. Positions the caller
    left out are filled with whatever is left, shuffled by the underlying
    generator so two unspecified opponents do not always hold the same hand.
    """

    def __init__(self, placements: dict[int, Card], seed: int | None = None) -> None:
        super().__init__(seed)
        self._placements = dict(placements)

    def shuffle(self, x, random=None) -> None:
        wanted = set(self._placements.values())
        if len(wanted) != len(self._placements):
            raise ValueError("la stessa carta e' stata assegnata a due posizioni")
        rest = [card for card in x if card not in wanted]
        super().shuffle(rest)
        spare = iter(rest)
        arranged = [self._placements.get(index) or next(spare) for index in range(len(x))]
        x[:] = arranged


def deal_positions(num_players: int) -> dict[str, list[int]]:
    """Where each thing sits in the deal order `play_hand` consumes.

    Two cards per seated player in seat order, then the flop, the turn and the
    river with no burn card. Pinned by a test that plays a hand and checks the
    cards came out where this says they would -- if the engine's dealing ever
    changes, that is the test that fails.
    """
    hole = {seat: [2 * seat, 2 * seat + 1] for seat in range(num_players)}
    board = 2 * num_players
    return {
        **{f"seat{seat}": slots for seat, slots in hole.items()},
        "flop": [board, board + 1, board + 2],
        "turn": [board + 3],
        "river": [board + 4],
    }
