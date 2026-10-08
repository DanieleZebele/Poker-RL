from __future__ import annotations

import random

import pytest
from support import fake_collector, fixed_mix

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import SCHEMA_VERSION, HandHistory
from pokerlab.engine.state import ActionRecord, Street
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.players.rl_agent import PolicyDecision
from pokerlab.rl.allin_reward import closing_street, expected_deltas


def cards(text: str) -> list[Card]:
    return [Card.parse(card) for card in text.split()]


def record(street: Street, seat: int, action_type: ActionType) -> ActionRecord:
    return ActionRecord(street, seat, f"p{seat}", action_type, 0, 0, 0, 0, 0.0)


def heads_up(actions: list[ActionRecord], *, board: str, payouts: dict[int, int], final: dict[int, int]):
    """AhAs (seat 0, button) against KhKs (seat 1), 1000 chips each."""
    return HandHistory(
        schema_version=SCHEMA_VERSION, hand_id="h", started_at=0.0, num_players=2,
        small_blind=1, big_blind=2, button_seat=0,
        starting_stacks={0: 1000, 1: 1000}, seat_names={0: "a", 1: "b"},
        community_cards=cards(board), actions=actions,
        hole_cards={0: tuple(cards("Ah As")), 1: tuple(cards("Kh Ks"))},
        payouts=payouts, final_stacks=final,
    )


BLINDS = [record(Street.PREFLOP, 0, ActionType.POST_BLIND), record(Street.PREFLOP, 1, ActionType.POST_BLIND)]
TO_THE_TURN = [
    *BLINDS,
    record(Street.PREFLOP, 0, ActionType.CALL), record(Street.PREFLOP, 1, ActionType.CHECK),
    record(Street.FLOP, 1, ActionType.CHECK), record(Street.FLOP, 0, ActionType.CHECK),
]


def all_in_on_the_turn() -> HandHistory:
    return heads_up(
        [*TO_THE_TURN, record(Street.TURN, 1, ActionType.ALL_IN), record(Street.TURN, 0, ActionType.CALL)],
        board="2c 7d 9h Jc 3s", payouts={0: 2000}, final={0: 2000, 1: 0},
    )


def test_an_all_in_on_the_turn_is_worth_the_share_of_rivers_each_side_wins():
    """Kings win on the two kings left, aces on the other 42 of the 44 rivers: with as
    many runouts as rivers, every river is played once and the expectation is exact."""
    expected = expected_deltas(all_in_on_the_turn(), random.Random(0), runouts=44)
    assert expected == pytest.approx({0: 1000 * 40 / 44, 1: -1000 * 40 / 44})


def test_a_hand_closes_on_the_street_of_its_last_action():
    assert closing_street(all_in_on_the_turn()) == Street.TURN
    preflop = heads_up(
        [*BLINDS, record(Street.PREFLOP, 0, ActionType.ALL_IN), record(Street.PREFLOP, 1, ActionType.CALL)],
        board="2c 7d 9h Jc 3s", payouts={0: 2000}, final={0: 2000, 1: 0},
    )
    assert closing_street(preflop) == Street.PREFLOP


def test_every_seat_all_in_on_its_blind_closes_preflop():
    blinds_only = heads_up(BLINDS, board="2c 7d 9h Jc 3s", payouts={0: 2000}, final={0: 2000, 1: 0})
    assert closing_street(blinds_only) == Street.PREFLOP


def test_a_bet_river_or_a_hand_won_by_a_fold_keeps_its_real_result():
    river = heads_up(
        [
            *TO_THE_TURN,
            record(Street.TURN, 1, ActionType.CHECK), record(Street.TURN, 0, ActionType.CHECK),
            record(Street.RIVER, 1, ActionType.CHECK), record(Street.RIVER, 0, ActionType.CHECK),
        ],
        board="2c 7d 9h Jc 3s", payouts={0: 4}, final={0: 1002, 1: 998},
    )
    folded = heads_up(
        [*BLINDS, record(Street.PREFLOP, 0, ActionType.FOLD)],
        board="", payouts={1: 3}, final={0: 999, 1: 1001},
    )
    for hand in (river, folded):
        assert closing_street(hand) is None
        assert expected_deltas(hand, random.Random(0), runouts=50) is None


def test_no_runouts_means_no_expectation():
    assert expected_deltas(all_in_on_the_turn(), random.Random(0), runouts=0) is None


def test_a_preflop_all_in_is_estimated_without_bias():
    """Aces against kings all-in preflop: about 82% for the aces."""
    hand = heads_up(
        [*BLINDS, record(Street.PREFLOP, 0, ActionType.ALL_IN), record(Street.PREFLOP, 1, ActionType.CALL)],
        board="2c 7d 9h Jc 3s", payouts={0: 2000}, final={0: 2000, 1: 0},
    )
    expected = expected_deltas(hand, random.Random(1), runouts=2000)
    assert expected[0] == pytest.approx(1000 * (2 * 0.82 - 1), abs=60)
    assert expected[0] + expected[1] == pytest.approx(0.0)


class _Shove(Player):
    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        types = {legal.action_type for legal in legal_actions}
        return Action(ActionType.ALL_IN if ActionType.ALL_IN in types else ActionType.CALL)


class _Fold(Player):
    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        types = {legal.action_type for legal in legal_actions}
        return Action(ActionType.CHECK if ActionType.CHECK in types else ActionType.FOLD)


def test_side_pots_conserve_chips_and_a_folded_seat_loses_exactly_what_it_put_in():
    players = [_Shove("p0", "a"), _Shove("p1", "b"), _Shove("p2", "c"), _Fold("p3", "d")]
    table = Table(GameConfig(num_players=4, starting_stack=500, small_blind=1, big_blind=2), players,
                  rng=random.Random(3))
    for hand_number in range(8):
        table.stacks = [100, 300, 500, 400]
        hand = table.play_hand().hand_history
        assert closing_street(hand) == Street.PREFLOP
        expected = expected_deltas(hand, random.Random(hand_number), runouts=30)
        assert sum(expected.values()) == pytest.approx(0.0)
        real = {seat: hand.final_stacks[seat] - hand.starting_stacks[seat] for seat in hand.starting_stacks}
        assert expected[3] == real[3]  # the folder: a blind or nothing, whatever the board
        assert expected[0] >= -100 and expected[1] >= -300  # nobody loses more than they had


def shove_policy(features: list[float], mask: list[bool]) -> PolicyDecision:
    """The largest legal bin: all-in whenever it is allowed."""
    return PolicyDecision(action_index=max(i for i, ok in enumerate(mask) if ok))


def test_the_collector_trains_on_the_expectation_and_reports_the_chips():
    collector = fake_collector(fixed_mix(3, stack_bb=50.0), shove_policy, rng=random.Random(5), allin_runouts=40)
    differed = False
    for _ in range(20):
        trajectories = collector.collect(1)  # every seat is the learner: the whole hand
        assert sum(t.reward for t in trajectories) == pytest.approx(0.0)
        assert sum(t.decisions[-1].reward for t in trajectories) == pytest.approx(0.0)
        differed |= any(t.decisions[-1].reward != t.reward for t in trajectories)
    assert differed


def test_a_negative_number_of_runouts_is_refused():
    with pytest.raises(ValueError, match="allin_runouts"):
        fake_collector(fixed_mix(3), shove_policy, allin_runouts=-1)
