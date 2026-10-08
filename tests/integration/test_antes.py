"""Antes: paid by everyone before the blinds, into the pot but not into the bet to match."""

import random

import pytest
from support import make_always_call_bot, make_random_legal_bot

from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.stats import analyse_hand
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.rl.features import committed_by_seat


class _Recorder(Player):
    """Folds whenever it can (checks otherwise) and keeps every observation it was shown."""

    def __init__(self, player_id: str, seen: list) -> None:
        super().__init__(player_id, player_id)
        self.seen = seen

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self.seen.append((observation, legal_actions))
        types = {legal.action_type for legal in legal_actions}
        return Action(ActionType.CHECK if ActionType.CHECK in types else ActionType.FOLD)


def test_the_antes_go_in_the_pot_but_not_in_the_bet_to_match():
    seen: list = []
    config = GameConfig(num_players=4, starting_stack=1000, small_blind=50, big_blind=100, ante=10)
    table = Table(config, [_Recorder(f"p{i}", seen) for i in range(4)], rng=random.Random(1))
    result = table.play_hand()

    observation, legal = seen[0]  # the first player asked, under the gun
    assert observation.pot_size == 4 * 10 + 50 + 100
    assert observation.current_bet_to_match == 100
    assert {info.seat: info.current_bet for info in observation.seats} == {0: 0, 1: 50, 2: 100, 3: 0}
    call = next(action for action in legal if action.action_type is ActionType.CALL)
    assert call.min_amount == 100  # the ante does not count towards the call

    antes = [r for r in result.hand_history.actions if r.action_type is ActionType.POST_ANTE]
    assert [r.seat for r in antes] == [1, 2, 3, 0]  # from the small blind round
    assert all(r.amount == 0 and r.stack_before - r.stack_after == 10 for r in antes)
    blinds = [r for r in result.hand_history.actions if r.action_type is ActionType.POST_BLIND]
    assert result.hand_history.actions.index(blinds[0]) == len(antes)  # antes come first
    # everyone folds to the big blind: it takes the antes and the small blind
    assert result.final_stacks == {0: 990, 1: 940, 2: 1080, 3: 990}


def test_a_player_the_ante_puts_all_in_wins_only_what_it_paid_into():
    config = GameConfig(num_players=3, starting_stack=1000, small_blind=50, big_blind=100, ante=10)
    players = [make_always_call_bot(f"p{i}") for i in range(3)]
    for seed in range(40):
        table = Table(config, players, rng=random.Random(seed))
        table.stacks = [6, 1000, 1000]  # seat 0 (the button) has less than the ante
        result = table.play_hand()
        hh = result.hand_history
        assert sum(hh.final_stacks.values()) == 2006
        ante = next(r for r in hh.actions if r.action_type is ActionType.POST_ANTE and r.seat == 0)
        assert ante.stack_after == 0
        # its pot: 6 from each of the three; it never acted
        assert hh.final_stacks[0] in (0, 18, 9, 6)
        assert all(r.seat != 0 or r.action_type in (ActionType.POST_ANTE, ActionType.POST_BLIND)
                   for r in hh.actions)


@pytest.mark.parametrize("seed", range(5))
def test_chip_conservation_holds_with_antes_and_uneven_stacks(seed):
    """The fuzz of `test_full_hand_flow` with antes: stacks down to less than the ante, so
    some players are all-in before the blinds are even posted."""
    rng = random.Random(seed)
    for _ in range(300):
        num_players = rng.randint(2, 9)
        config = GameConfig(num_players=num_players, starting_stack=300, small_blind=5, big_blind=10,
                            ante=rng.randint(1, 4))
        players = [make_random_legal_bot(f"p{i}", rng=random.Random(rng.random())) for i in range(num_players)]
        table = Table(config, players, rng=random.Random(rng.random()))
        table.stacks = [rng.randint(1, 300) for _ in range(num_players)]
        total_before = sum(table.stacks)
        table.play_hand()
        assert sum(table.stacks) == total_before, "chip conservation invariant violated"


def test_the_features_count_the_antes_in_what_each_seat_has_put_in():
    seen: list = []
    config = GameConfig(num_players=5, starting_stack=1000, small_blind=50, big_blind=100, ante=10)
    table = Table(config, [_Recorder(f"p{i}", seen) for i in range(5)], rng=random.Random(3))
    table.play_hand()
    for observation, _legal in seen:
        committed = committed_by_seat(observation)
        assert sum(committed.values()) == observation.pot_size
        assert all(committed[seat] >= 10 for seat in range(5))


def test_an_ante_is_no_action_in_the_statistics():
    """The same hand counted with and without its ante records gives the same counts: an
    ante opens no steal, VPIP or 3-bet chance."""
    config = GameConfig(num_players=6, starting_stack=1000, small_blind=50, big_blind=100, ante=10)
    for seed in range(20):
        players = [make_random_legal_bot(f"p{i}", rng=random.Random(seed * 7 + i)) for i in range(6)]
        hh = Table(config, players, rng=random.Random(seed)).play_hand().hand_history
        kwargs = {
            "dealt": sorted(hh.starting_stacks), "button_seat": hh.button_seat,
            "board_cards": len(hh.community_cards),
            "player_ids": {seat: f"p{seat}" for seat in hh.starting_stacks},
        }
        without = [r for r in hh.actions if r.action_type is not ActionType.POST_ANTE]
        assert analyse_hand(hh.actions, **kwargs) == analyse_hand(without, **kwargs)
