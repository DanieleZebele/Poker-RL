from __future__ import annotations

import random

from pokerlab.cards.card import Card, Rank, Suit


class Deck:
    """A standard 52-card deck with an injectable RNG for reproducible tests
    and, later, reproducible RL episode seeding."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._cards: list[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def deal(self, n: int) -> list[Card]:
        if n < 0:
            raise ValueError("cannot deal a negative number of cards")
        if n > len(self._cards):
            raise ValueError("not enough cards left in the deck")
        dealt, self._cards = self._cards[:n], self._cards[n:]
        return dealt

    def __len__(self) -> int:
        return len(self._cards)
