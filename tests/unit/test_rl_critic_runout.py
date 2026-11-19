"""The critic in hindsight (`--critic-runout`, off by default): the table announces the five
community cards a hand will have, and a training critic can read its equities on them."""

from __future__ import annotations

import random

import pytest
from support import make_always_call_bot, make_random_legal_bot

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import deal_view
from pokerlab.rl.features import card_index


@pytest.mark.parametrize("num_players", [2, 3, 6, 9])
def test_the_announced_runout_is_the_board_the_hand_deals(num_players):
    """The deck is shuffled once and no card is burnt: the runout read when the hole cards
    are dealt is the board, card for card, as far as the hand gets."""
    rng = random.Random(num_players)
    started: list[dict] = []
    players = [
        make_always_call_bot(f"p{seat}", f"P{seat}") if seat % 2 else make_random_legal_bot(f"p{seat}", f"P{seat}", rng)
        for seat in range(num_players)
    ]
    table = Table(GameConfig(num_players=num_players, starting_stack=200, small_blind=1, big_blind=2),
                  players, rng=rng, on_hand_started=started.append)
    rivers = 0
    for _ in range(80):
        table.stacks = [200] * num_players
        history = table.play_hand().hand_history
        runout = started[-1]["runout"]
        board = list(history.community_cards)
        assert len(runout) == 5 and list(runout[: len(board)]) == board
        holes = {card for pair in started[-1]["hole_cards"].values() for card in pair}
        assert not holes & set(runout)
        rivers += len(board) == 5
    assert rivers > 0


def test_a_deal_view_with_the_runout_shows_the_whole_board_and_refuses_another():
    """At every decision the view's board is the runout, whose first cards are the ones out;
    without the runout it is the visible board; a runout that does not continue it is refused."""
    rng = random.Random(5)
    started: list[dict] = []
    seen: list = []
    caller = make_always_call_bot("p0", "P0")

    class Probe(type(caller)):
        def act(self, observation, legal_actions):
            seen.append((observation, started[-1]))
            return super().act(observation, legal_actions)

    players = [make_always_call_bot(f"p{seat}", f"P{seat}") for seat in range(3)]
    probe = players[0]
    probe.__class__ = Probe
    table = Table(GameConfig(num_players=3, starting_stack=200, small_blind=1, big_blind=2),
                  players, rng=rng, on_hand_started=started.append)
    for _ in range(10):
        table.stacks = [200] * 3
        table.play_hand()
    later = [(o, info) for o, info in seen if o.community_cards]
    assert later
    for observation, info in seen:
        visible = tuple(card_index(c) for c in observation.community_cards)
        assert deal_view(observation, info["hole_cards"]).board == visible
        view = deal_view(observation, info["hole_cards"], info["runout"])
        assert view.board == tuple(card_index(c) for c in info["runout"])
    observation, info = later[0]
    wrong = (info["runout"][1], info["runout"][0], *info["runout"][2:])
    with pytest.raises(ValueError):
        deal_view(observation, info["hole_cards"], wrong)
