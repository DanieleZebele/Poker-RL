"""Draws playing cards as small Tkinter Canvas graphics -- no image files
or extra dependencies (e.g. Pillow) needed, everything is vector shapes and
text drawn directly on a Canvas widget."""

from __future__ import annotations

import tkinter as tk

from pokerlab.cards.card import Card, Suit

CARD_WIDTH = 46
CARD_HEIGHT = 66

_SUIT_SYMBOLS = {
    Suit.SPADES: "♠",
    Suit.HEARTS: "♥",
    Suit.DIAMONDS: "♦",
    Suit.CLUBS: "♣",
}
_SUIT_COLORS = {
    Suit.SPADES: "black",
    Suit.CLUBS: "black",
    Suit.HEARTS: "#c0392b",
    Suit.DIAMONDS: "#c0392b",
}

_BACK_FILL = "#2b4c8c"
_BACK_PATTERN = "#c9d6f0"
_EMPTY_OUTLINE = "#999999"


def new_card_canvas(parent: tk.Widget) -> tk.Canvas:
    """A blank, empty-slot canvas ready to be passed to the draw_* functions."""
    canvas = tk.Canvas(parent, width=CARD_WIDTH, height=CARD_HEIGHT, highlightthickness=0)
    draw_empty_slot(canvas)
    return canvas


def draw_card_face(canvas: tk.Canvas, card: Card) -> None:
    canvas.delete("all")
    w, h = CARD_WIDTH, CARD_HEIGHT
    color = _SUIT_COLORS[card.suit]
    symbol = _SUIT_SYMBOLS[card.suit]
    canvas.create_rectangle(1, 1, w - 1, h - 1, fill="white", outline="black", width=1)
    canvas.create_text(5, 4, text=card.rank.symbol, anchor="nw", fill=color, font=("TkDefaultFont", 10, "bold"))
    canvas.create_text(5, 17, text=symbol, anchor="nw", fill=color, font=("TkDefaultFont", 10, "bold"))
    canvas.create_text(w / 2, h / 2 + 4, text=symbol, fill=color, font=("TkDefaultFont", 18, "bold"))


def draw_card_back(canvas: tk.Canvas) -> None:
    canvas.delete("all")
    w, h = CARD_WIDTH, CARD_HEIGHT
    canvas.create_rectangle(1, 1, w - 1, h - 1, fill=_BACK_FILL, outline="black", width=1)
    canvas.create_rectangle(5, 5, w - 5, h - 5, outline=_BACK_PATTERN, width=1)


def draw_empty_slot(canvas: tk.Canvas) -> None:
    """A dashed outline for a card that hasn't been dealt (yet)."""
    canvas.delete("all")
    w, h = CARD_WIDTH, CARD_HEIGHT
    canvas.create_rectangle(1, 1, w - 1, h - 1, outline=_EMPTY_OUTLINE, dash=(3, 2))
