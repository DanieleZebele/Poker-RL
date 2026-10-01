import random

import pytest
from support import make_always_call_bot, make_random_legal_bot

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table


def build_mixed_table(num_players: int, seed: int, starting_stack: int = 500) -> Table:
    config = GameConfig(num_players=num_players, starting_stack=starting_stack, small_blind=1, big_blind=2)
    players = []
    for i in range(num_players):
        if i % 2 == 0:
            players.append(make_always_call_bot(f"p{i}", f"AC{i}"))
        else:
            # Picks uniformly among whatever is legal, BET/RAISE included, so
            # this alone already exercises every action type the fuzz needs.
            players.append(make_random_legal_bot(f"p{i}", f"R{i}", rng=random.Random(seed * 31 + i)))
    return Table(config, players, rng=random.Random(seed))


@pytest.mark.parametrize("num_players", range(2, 10))
@pytest.mark.parametrize("seed", range(5))
def test_chip_conservation_holds_across_a_session(num_players, seed):
    table = build_mixed_table(num_players, seed)
    for _ in range(200):
        if sum(1 for s in table.stacks if s > 0) < 2:
            break
        total_before = sum(table.stacks)
        table.play_hand()
        assert sum(table.stacks) == total_before, "chip conservation invariant violated"


def test_single_hand_produces_a_balanced_result():
    table = build_mixed_table(num_players=6, seed=99)
    result = table.play_hand()
    hh = result.hand_history
    assert sum(hh.final_stacks.values()) == sum(hh.starting_stacks.values())
    assert sum(hh.payouts.values()) > 0
    assert set(hh.payouts.keys()) <= set(hh.starting_stacks.keys())


def test_showdown_reveals_hole_cards_for_every_dealt_seat():
    table = build_mixed_table(num_players=4, seed=5)
    result = table.play_hand()
    assert set(result.hand_history.hole_cards.keys()) == set(result.hand_history.starting_stacks.keys())
    for hole in result.hand_history.hole_cards.values():
        assert len(hole) == 2
