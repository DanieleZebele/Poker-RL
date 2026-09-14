from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        return _RANK_SYMBOLS[self]


_RANK_SYMBOLS = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

_SYMBOL_TO_RANK = {v: k for k, v in _RANK_SYMBOLS.items()}


class Suit(Enum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"

    @property
    def symbol(self) -> str:
        return self.value


_SYMBOL_TO_SUIT = {s.value: s for s in Suit}


@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.symbol}"

    def __repr__(self) -> str:
        return f"Card({self!s})"

    @staticmethod
    def parse(text: str) -> Card:
        """Parse a two-character string like 'Ah' or 'Td' into a Card."""
        if len(text) != 2:
            raise ValueError(f"invalid card string: {text!r}")
        rank_symbol, suit_symbol = text[0].upper(), text[1].lower()
        if rank_symbol not in _SYMBOL_TO_RANK:
            raise ValueError(f"invalid rank in card string: {text!r}")
        if suit_symbol not in _SYMBOL_TO_SUIT:
            raise ValueError(f"invalid suit in card string: {text!r}")
        return Card(_SYMBOL_TO_RANK[rank_symbol], _SYMBOL_TO_SUIT[suit_symbol])
