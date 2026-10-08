"""Fresh random deals with the pot share each player actually won.

There is no stored dataset: every call draws new cards. The label of a deal is the
*realised* result of one random completion of the board, scored with the project's own
evaluator -- who has the best five-card hand, with a tie splitting the pot. That is a
noisy label but an unbiased one: its expectation over the board still to come is exactly
the player's equity (expected share of the pot), so a network trained on it with a
squared error learns the equity itself.

Cards are indexed 0..51 as `suit * 13 + rank - 2` (clubs, diamonds, hearts, spades; the
same index `pokerlab.rl.features` uses), so a deal can later be fed to the real
observation encoding unchanged.

Pure Python, no torch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pokerlab.cards import Card, Rank, Suit
from pokerlab.evaluator.evaluator import evaluate

SUITS = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
MIN_PLAYERS = 2
MAX_PLAYERS = 9
# How many board cards are already visible: preflop, flop, turn, river.
STREETS = (0, 3, 4, 5)


def card_from_index(index: int) -> Card:
    return Card(Rank(index % 13 + 2), SUITS[index // 13])


@dataclass(frozen=True)
class Deal:
    holes: tuple[tuple[int, int], ...]  # two cards per player
    board: tuple[int, ...]  # all five cards, of which only `visible` are known
    visible: int  # 0, 3, 4 or 5
    shares: tuple[float, ...]  # each player's share of the pot; sums to 1

    @property
    def players(self) -> int:
        return len(self.holes)

    @property
    def known_board(self) -> tuple[int, ...]:
        return self.board[: self.visible]


def pot_shares(holes: tuple[tuple[int, int], ...], board: tuple[int, ...]) -> tuple[float, ...]:
    """Each player's share of the pot when `board` (five cards) is the full runout."""
    cards = [card_from_index(c) for c in board]
    ranks = [evaluate([card_from_index(c) for c in hole] + cards) for hole in holes]
    best = max(ranks)
    winners = sum(1 for rank in ranks if rank == best)
    return tuple(1.0 / winners if rank == best else 0.0 for rank in ranks)


def sample_deal(
    rng: random.Random,
    street_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    min_players: int = MIN_PLAYERS,
    max_players: int = MAX_PLAYERS,
) -> Deal:
    """One new deal: a number of players (uniform), their cards, a full board of which
    a street picked by `street_weights` is visible, and the result."""
    players = rng.randint(min_players, max_players)
    drawn = rng.sample(range(52), 2 * players + 5)
    holes = tuple((drawn[2 * i], drawn[2 * i + 1]) for i in range(players))
    board = tuple(drawn[2 * players :])
    visible = rng.choices(STREETS, weights=street_weights)[0]
    return Deal(holes, board, visible, pot_shares(holes, board))
