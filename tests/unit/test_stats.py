from __future__ import annotations

import random

import pytest
from support import make_random_legal_bot

from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import ActionRecord, Street
from pokerlab.engine.stats import (
    STAT_SLOTS,
    STATS,
    USED_SLOTS,
    WINDOW,
    StatsTracker,
    analyse_hand,
)
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player

P, F, T, R = Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER
POST, FOLD, CHECK = ActionType.POST_BLIND, ActionType.FOLD, ActionType.CHECK
CALL, BET, RAISE, ALL_IN = ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN

SIX = [0, 1, 2, 3, 4, 5]  # button 0: small blind 1, big blind 2, cutoff 5
BLINDS = [(P, 1, POST, 50), (P, 2, POST, 100)]


def hand(*steps):
    return [
        ActionRecord(
            street=street, seat=seat, player_id=f"p{seat}", action_type=kind, amount=amount,
            stack_before=0, stack_after=0, pot_before=0, timestamp=0.0,
        )
        for street, seat, kind, amount in steps
    ]


def counts(steps, *, board=0, dealt=SIX, button=0):
    return analyse_hand(
        hand(*steps), dealt=dealt, button_seat=button, board_cards=board,
        player_ids={seat: f"p{seat}" for seat in dealt},
    )


def stat(result, seat: int, name: str) -> tuple[int, int]:
    """`(events, opportunities)` of one statistic for one seat in one hand."""
    i = STATS.index(name)
    return result[f"p{seat}"].events[i], result[f"p{seat}"].opportunities[i]


# ---- preflop -----------------------------------------------------------------


def test_vpip_counts_a_voluntary_call_but_not_a_blind_or_a_checked_option():
    result = counts(
        [*BLINDS, (P, 3, CALL, 100), (P, 4, FOLD, 0), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
         (P, 1, CALL, 100), (P, 2, CHECK, 100)]
    )
    assert stat(result, 3, "vpip") == (1, 1)
    assert stat(result, 1, "vpip") == (1, 1)  # completing the small blind is voluntary
    assert stat(result, 2, "vpip") == (0, 1)  # the big blind's free check is not
    assert stat(result, 4, "vpip") == (0, 1)


def test_an_all_in_for_no_more_than_the_bet_is_a_call_not_a_raise():
    result = counts(
        [*BLINDS, (P, 3, RAISE, 300), (P, 4, ALL_IN, 250), (P, 5, ALL_IN, 1000),
         (P, 0, FOLD, 0), (P, 1, FOLD, 0), (P, 2, FOLD, 0)]
    )
    assert stat(result, 4, "pfr") == (0, 1) and stat(result, 4, "vpip") == (1, 1)
    assert stat(result, 5, "pfr") == (1, 1)
    assert stat(result, 3, "pfr") == (1, 1)


def test_three_bet_is_counted_for_a_player_facing_exactly_one_raise():
    result = counts(
        [*BLINDS, (P, 3, RAISE, 300), (P, 4, RAISE, 900), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, FOLD, 0), (P, 3, FOLD, 0)]
    )
    assert stat(result, 4, "three_bet") == (1, 1)
    assert stat(result, 3, "three_bet") == (0, 0)  # the opener was not facing a raise
    assert stat(result, 5, "three_bet") == (0, 0)  # facing a 3-bet, not an open
    assert stat(result, 2, "three_bet") == (0, 0)


def test_a_call_of_the_open_is_an_opportunity_that_was_not_taken():
    result = counts(
        [*BLINDS, (P, 3, RAISE, 300), (P, 4, CALL, 300), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, FOLD, 0)]
    )
    assert stat(result, 4, "three_bet") == (0, 1)
    assert stat(result, 5, "three_bet") == (0, 1)


def test_fold_to_three_bet_belongs_to_the_opener_who_is_re_raised():
    result = counts(
        [*BLINDS, (P, 3, RAISE, 300), (P, 4, RAISE, 900), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, FOLD, 0), (P, 3, FOLD, 0)]
    )
    assert stat(result, 3, "fold_to_three_bet") == (1, 1)
    assert stat(result, 4, "fold_to_three_bet") == (0, 0)
    called = counts(
        [*BLINDS, (P, 3, RAISE, 300), (P, 4, RAISE, 900), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, FOLD, 0), (P, 3, CALL, 900)]
    )
    assert stat(called, 3, "fold_to_three_bet") == (0, 1)


def test_a_steal_is_a_first_in_raise_from_the_cutoff_button_or_small_blind():
    result = counts(
        [*BLINDS, (P, 3, FOLD, 0), (P, 4, FOLD, 0), (P, 5, RAISE, 300), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, FOLD, 0)]
    )
    assert stat(result, 5, "steal") == (1, 1)
    assert stat(result, 3, "steal") == (0, 0) and stat(result, 4, "steal") == (0, 0)
    assert stat(result, 0, "steal") == (0, 0)  # the button faced a raise, it was not first in


def test_a_limper_in_front_ends_the_steal():
    result = counts(
        [*BLINDS, (P, 3, CALL, 100), (P, 4, FOLD, 0), (P, 5, RAISE, 400), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, FOLD, 0), (P, 3, FOLD, 0)]
    )
    assert stat(result, 5, "steal") == (0, 0)


def test_the_button_is_the_only_steal_seat_heads_up():
    result = counts(
        [(P, 1, POST, 50), (P, 0, POST, 100), (P, 1, RAISE, 300), (P, 0, FOLD, 0)],
        dealt=[0, 1], button=1,
    )
    assert stat(result, 1, "steal") == (1, 1)
    assert stat(result, 0, "steal") == (0, 0)


# ---- flop and later -----------------------------------------------------------

OPEN_AND_CALL = [
    *BLINDS, (P, 3, RAISE, 300), (P, 4, FOLD, 0), (P, 5, FOLD, 0), (P, 0, CALL, 300),
    (P, 1, FOLD, 0), (P, 2, FOLD, 0),
]


def test_a_continuation_bet_and_the_fold_to_it():
    result = counts([*OPEN_AND_CALL, (F, 3, BET, 400), (F, 0, FOLD, 0)], board=3)
    assert stat(result, 3, "cbet") == (1, 1)
    assert stat(result, 0, "fold_to_cbet") == (1, 1)
    assert stat(result, 3, "fold_to_cbet") == (0, 0)  # the bettor does not fold to their own bet
    assert stat(result, 3, "aggression") == (1, 1)
    assert stat(result, 0, "aggression") == (0, 0)  # a fold puts no chips in


def test_checking_the_flop_as_the_raiser_is_a_missed_continuation_bet():
    result = counts(
        [*OPEN_AND_CALL, (F, 3, CHECK, 0), (F, 0, BET, 400), (F, 3, CALL, 400)], board=3
    )
    assert stat(result, 3, "cbet") == (0, 1)
    assert stat(result, 0, "cbet") == (0, 0)  # not the last preflop raiser
    assert stat(result, 0, "fold_to_cbet") == (0, 0)  # there was no c-bet to fold to
    assert stat(result, 3, "aggression") == (0, 1)  # calling is a chip-in without aggression
    assert stat(result, 0, "aggression") == (1, 1)


def test_a_raiser_who_faces_a_bet_first_had_no_chance_to_continuation_bet():
    result = counts(
        [*OPEN_AND_CALL, (F, 0, BET, 400), (F, 3, FOLD, 0)], board=3
    )
    # seat 0 acted first on this street here (a scripted order): the raiser, seat 3,
    # then faces a bet, so it never had the chance to be the one to bet first.
    assert stat(result, 3, "cbet") == (0, 0)


def test_went_to_showdown_over_saw_the_flop():
    reached = counts(
        [*OPEN_AND_CALL, (F, 3, BET, 400), (F, 0, CALL, 400), (T, 3, CHECK, 0), (T, 0, CHECK, 0),
         (R, 3, CHECK, 0), (R, 0, CHECK, 0)],
        board=5,
    )
    assert stat(reached, 3, "wtsd") == (1, 1) and stat(reached, 0, "wtsd") == (1, 1)
    assert stat(reached, 4, "wtsd") == (0, 0)  # folded before the flop
    folded = counts([*OPEN_AND_CALL, (F, 3, BET, 400), (F, 0, FOLD, 0)], board=3)
    assert stat(folded, 3, "wtsd") == (0, 1) and stat(folded, 0, "wtsd") == (0, 1)


def test_an_all_in_runout_is_a_showdown_even_with_no_postflop_action():
    result = counts(
        [*BLINDS, (P, 3, ALL_IN, 2000), (P, 4, FOLD, 0), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, ALL_IN, 2000)],
        board=5,
    )
    assert stat(result, 3, "wtsd") == (1, 1) and stat(result, 2, "wtsd") == (1, 1)


def test_nobody_saw_a_flop_that_was_never_dealt():
    result = counts(
        [*BLINDS, (P, 3, RAISE, 300), (P, 4, FOLD, 0), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
         (P, 1, FOLD, 0), (P, 2, FOLD, 0)],
        board=0,
    )
    assert all(stat(result, seat, "wtsd") == (0, 0) for seat in SIX)


# ---- the tracker ---------------------------------------------------------------

LIMP = [*BLINDS, (P, 3, CALL, 100), (P, 4, FOLD, 0), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
        (P, 1, FOLD, 0), (P, 2, CHECK, 100)]
FOLDED = [*BLINDS, (P, 3, FOLD, 0), (P, 4, FOLD, 0), (P, 5, FOLD, 0), (P, 0, FOLD, 0),
          (P, 1, FOLD, 0)]


def record(tracker: StatsTracker, steps) -> None:
    tracker.record_hand(
        hand(*steps), dealt=SIX, button_seat=0, board_cards=0,
        player_ids={seat: f"p{seat}" for seat in SIX},
    )


def test_an_unseen_player_has_no_vector_at_all():
    assert StatsTracker().vector("nobody") is None
    assert StatsTracker().vectors({0: "nobody"}) == {}


def test_the_window_forgets_the_oldest_hand():
    tracker = StatsTracker(window=3)
    for steps in (LIMP, FOLDED, FOLDED, FOLDED):
        record(tracker, steps)
    assert tracker.hands("p3") == 3
    assert tracker.rates("p3")["vpip"] == (0, 3)  # the one limp has slid out


def test_the_window_keeps_hands_inside_it():
    tracker = StatsTracker(window=3)
    for steps in (FOLDED, LIMP, FOLDED):
        record(tracker, steps)
    assert tracker.rates("p3")["vpip"] == (1, 3)


def test_the_vector_pairs_every_rate_with_how_often_it_could_have_happened():
    tracker = StatsTracker()
    for steps in (LIMP, FOLDED, LIMP):
        record(tracker, steps)
    vector = tracker.vector("p3")
    assert len(vector) == USED_SLOTS == 20
    assert vector[0] == 1.0
    assert 0.0 < vector[1] < 1.0  # three hands of a 200-hand window, log-scaled
    vpip_rate, vpip_count = vector[2 + 2 * STATS.index("vpip")], vector[3 + 2 * STATS.index("vpip")]
    assert vpip_rate == pytest.approx(2 / 3)
    assert vpip_count == vector[1]  # VPIP has an opportunity every hand
    never = STATS.index("three_bet")
    assert vector[2 + 2 * never] == 0.0 and vector[3 + 2 * never] == 0.0  # unknown reads as zeros
    assert all(0.0 <= value <= 1.0 for value in vector)


def test_the_model_has_room_for_far_more_than_is_filled_today():
    assert USED_SLOTS < STAT_SLOTS == 100 and WINDOW == 200


# ---- through the table -----------------------------------------------------------


class Probe(Player):
    def __init__(self, player_id: str, rng: random.Random, sink: list) -> None:
        super().__init__(player_id, player_id)
        self._inner = make_random_legal_bot(player_id, player_id, rng)
        self._sink = sink

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self._sink.append(observation)
        return self._inner.act(observation, legal_actions)


def play(tracker, hands=6, players=4, seed=3) -> list[Observation]:
    rng = random.Random(seed)
    sink: list[Observation] = []
    config = GameConfig(num_players=players, starting_stack=2000, small_blind=50, big_blind=100)
    table = Table(
        config, [Probe(f"p{i}", random.Random(i), sink) for i in range(players)],
        rng=rng, stats_tracker=tracker,
    )
    for _ in range(hands):
        for stack in range(players):
            table.stacks[stack] = 2000
        table.play_hand()
    return sink


def test_without_a_tracker_no_observation_carries_statistics():
    assert all(o.seat_stats == {} for o in play(None))


def test_with_a_tracker_the_second_hand_onward_sees_every_seat_it_has_met():
    sink = play(StatsTracker())
    first_hand = [o for o in sink if len(o.action_history) <= 3][:1]
    assert first_hand and first_hand[0].seat_stats == {}  # nobody has been seen yet
    later = [o for o in sink if o.seat_stats]
    assert later, "after the first hand the tracker knows the players"
    for observation in later:
        for vector in observation.seat_stats.values():
            assert len(vector) == USED_SLOTS and vector[0] == 1.0
            assert all(0.0 <= value <= 1.0 for value in vector)
        assert set(observation.seat_stats) <= {s.seat for s in observation.seats}


def test_the_tracker_sees_the_same_hands_the_table_played():
    tracker = StatsTracker()
    play(tracker, hands=6)
    assert {tracker.hands(f"p{i}") for i in range(4)} == {6}


def test_a_tracker_changes_nothing_about_how_the_hands_are_played():
    """It only watches: the same seed with and without one ends on the same stacks."""

    def final_stacks(tracker):
        rng = random.Random(11)
        config = GameConfig(num_players=5, starting_stack=2000, small_blind=50, big_blind=100)
        players = [make_random_legal_bot(f"p{i}", f"p{i}", random.Random(i)) for i in range(5)]
        table = Table(config, players, rng=rng, stats_tracker=tracker)
        table.play_session(25)
        return list(table.stacks)

    assert final_stacks(None) == final_stacks(StatsTracker())
