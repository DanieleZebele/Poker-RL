import random

import pytest

from pokerlab.cards.card import Card
from pokerlab.engine.actions import ActionType, LegalAction
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.players.base import Observation, SeatPublicInfo
from pokerlab.players.scripted import (
    BOT_CATALOG,
    get_bot_profile,
    list_bot_profiles,
    make_always_call_bot,
    make_heuristic_bot,
    make_random_legal_bot,
)


def make_observation() -> Observation:
    return Observation(
        street=Street.FLOP,
        hole_cards=(Card.parse("Ah"), Card.parse("Kh")),
        community_cards=(Card.parse("2c"), Card.parse("7d"), Card.parse("9s")),
        pot_size=30,
        current_bet_to_match=10,
        min_raise=10,
        my_seat=0,
        my_stack=190,
        my_current_bet=0,
        seats=(SeatPublicInfo(seat=0, name="Me", stack=190, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False),),
        button_seat=1,
        action_history=(),
    )


def test_always_call_bot_never_folds_when_check_or_call_available():
    bot = make_always_call_bot("p0")
    obs = make_observation()

    legal_with_call = [LegalAction(ActionType.FOLD), LegalAction(ActionType.CALL, 10, 10), LegalAction(ActionType.ALL_IN, 190, 190)]
    action = bot.act(obs, legal_with_call)
    assert action.action_type == ActionType.CALL

    legal_with_check = [LegalAction(ActionType.FOLD), LegalAction(ActionType.CHECK), LegalAction(ActionType.ALL_IN, 190, 190)]
    action = bot.act(obs, legal_with_check)
    assert action.action_type == ActionType.CHECK


def test_always_call_bot_shoves_rather_than_fold_when_it_cannot_afford_a_call():
    bot = make_always_call_bot("p0")
    obs = make_observation()
    legal = [LegalAction(ActionType.FOLD), LegalAction(ActionType.ALL_IN, 5, 5)]
    action = bot.act(obs, legal)
    assert action.action_type == ActionType.ALL_IN


def test_random_legal_bot_always_returns_a_legal_action_across_many_trials():
    obs = make_observation()
    legal = [
        LegalAction(ActionType.FOLD),
        LegalAction(ActionType.CALL, 10, 10),
        LegalAction(ActionType.RAISE, 20, 190),
        LegalAction(ActionType.ALL_IN, 190, 190),
    ]
    bot = make_random_legal_bot("p0", rng=random.Random(0))
    legal_types = {la.action_type for la in legal}
    for _ in range(500):
        action = bot.act(obs, legal)
        assert action.action_type in legal_types
        if action.action_type == ActionType.RAISE:
            assert 20 <= action.amount <= 190


def make_strong_observation() -> Observation:
    """Flopped trip aces -- a hand any reasonable heuristic should raise."""
    return Observation(
        street=Street.FLOP,
        hole_cards=(Card.parse("Ah"), Card.parse("Ac")),
        community_cards=(Card.parse("Ad"), Card.parse("Kc"), Card.parse("2h")),
        pot_size=30,
        current_bet_to_match=0,
        min_raise=10,
        my_seat=0,
        my_stack=190,
        my_current_bet=0,
        seats=(SeatPublicInfo(seat=0, name="Me", stack=190, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False),),
        button_seat=1,
        action_history=(),
    )


def test_heuristic_bot_bluffs_with_weak_hand_when_bluff_frequency_is_certain():
    bot = make_heuristic_bot(
        "p0", "Bluffer", tightness=0.9, aggression=0.9, bluff_frequency=1.0, size_variance=0.0, rng=random.Random(0)
    )
    obs = make_observation()  # ace-high only: strength (0.15) is far below tightness (0.9)
    legal = [
        LegalAction(ActionType.FOLD),
        LegalAction(ActionType.CALL, 10, 10),
        LegalAction(ActionType.RAISE, 20, 190),
        LegalAction(ActionType.ALL_IN, 190, 190),
    ]
    assert bot.act(obs, legal).action_type == ActionType.RAISE


def test_heuristic_bot_never_bluffs_when_bluff_frequency_is_zero():
    bot = make_heuristic_bot(
        "p0", "Nit", tightness=0.9, aggression=0.9, bluff_frequency=0.0, size_variance=0.0, rng=random.Random(0)
    )
    obs = make_observation()
    legal = [
        LegalAction(ActionType.FOLD),
        LegalAction(ActionType.CALL, 10, 10),
        LegalAction(ActionType.RAISE, 20, 190),
        LegalAction(ActionType.ALL_IN, 190, 190),
    ]
    for _ in range(50):
        assert bot.act(obs, legal).action_type == ActionType.FOLD


def test_heuristic_bot_bet_sizing_varies_across_calls_with_the_same_hand():
    bot = make_heuristic_bot(
        "p0", "Varied", tightness=0.1, aggression=0.8, bluff_frequency=0.0, size_variance=0.5, rng=random.Random(0)
    )
    obs = make_strong_observation()
    legal = [
        LegalAction(ActionType.FOLD),
        LegalAction(ActionType.CHECK),
        LegalAction(ActionType.BET, 10, 190),
        LegalAction(ActionType.ALL_IN, 190, 190),
    ]
    amounts = set()
    for _ in range(50):
        action = bot.act(obs, legal)
        assert action.action_type == ActionType.BET
        assert 10 <= action.amount <= 190
        amounts.add(action.amount)
    assert len(amounts) > 1, "expected varied bet sizes with size_variance > 0, got a single fixed amount"


def test_zero_size_variance_gives_a_consistent_bet_size():
    bot = make_heuristic_bot(
        "p0", "Consistent", tightness=0.1, aggression=0.8, bluff_frequency=0.0, size_variance=0.0, rng=random.Random(0)
    )
    obs = make_strong_observation()
    legal = [
        LegalAction(ActionType.FOLD),
        LegalAction(ActionType.CHECK),
        LegalAction(ActionType.BET, 10, 190),
        LegalAction(ActionType.ALL_IN, 190, 190),
    ]
    amounts = {bot.act(obs, legal).amount for _ in range(20)}
    assert len(amounts) == 1


def make_observation_with_opponents(num_opponents: int) -> Observation:
    """A flopped pair of twos (strength exactly 0.35) facing a bet, with a
    configurable number of other live (non-folded) seats -- for testing how
    the field size shifts the bot's continue/fold threshold and bluff rate."""
    seats = [SeatPublicInfo(seat=0, name="Me", stack=190, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False)]
    seats += [
        SeatPublicInfo(seat=i + 1, name=f"Opp{i}", stack=190, current_bet=10, status=PlayerStatus.ACTIVE, is_button=False)
        for i in range(num_opponents)
    ]
    return Observation(
        street=Street.FLOP,
        hole_cards=(Card.parse("2h"), Card.parse("7d")),
        community_cards=(Card.parse("2c"), Card.parse("9s"), Card.parse("Kc")),
        pot_size=30,
        current_bet_to_match=10,
        min_raise=10,
        my_seat=0,
        my_stack=190,
        my_current_bet=0,
        seats=tuple(seats),
        button_seat=1,
        action_history=(),
    )


def test_more_live_opponents_makes_the_bot_play_tighter():
    # Base tightness 0.3 with a strength-0.35 hand (a plain pair): playable
    # heads-up (1 opponent, effective tightness drops) but not against a
    # full field (7 opponents, effective tightness rises above 0.35).
    bot = make_heuristic_bot(
        "p0", "FieldAware", tightness=0.3, aggression=0.5, bluff_frequency=0.0, size_variance=0.0, rng=random.Random(0)
    )
    legal = [LegalAction(ActionType.FOLD), LegalAction(ActionType.CALL, 10, 10), LegalAction(ActionType.ALL_IN, 190, 190)]

    heads_up = bot.act(make_observation_with_opponents(1), legal)
    full_field = bot.act(make_observation_with_opponents(7), legal)

    assert heads_up.action_type == ActionType.CALL
    assert full_field.action_type == ActionType.FOLD


def test_fewer_live_opponents_increases_effective_bluff_frequency():
    trials = 400
    weak_legal = [
        LegalAction(ActionType.FOLD),
        LegalAction(ActionType.CALL, 10, 10),
        LegalAction(ActionType.RAISE, 20, 190),
        LegalAction(ActionType.ALL_IN, 190, 190),
    ]

    def bluff_rate(num_opponents: int) -> float:
        bot = make_heuristic_bot(
            "p0", "Bluffer", tightness=0.95, aggression=0.5, bluff_frequency=0.3, size_variance=0.0, rng=random.Random(42)
        )
        obs = make_observation_with_opponents(num_opponents)  # strength 0.35, tightness 0.95: never plays without bluffing
        raises = sum(1 for _ in range(trials) if bot.act(obs, weak_legal).action_type == ActionType.RAISE)
        return raises / trials

    heads_up_rate = bluff_rate(1)
    full_field_rate = bluff_rate(7)
    assert heads_up_rate > full_field_rate


def test_bot_catalog_is_sorted_weakest_to_strongest_with_unique_keys():
    profiles = list_bot_profiles()
    difficulties = [p.difficulty for p in profiles]
    assert difficulties == sorted(difficulties)
    assert len(profiles) == len({p.key for p in profiles})
    assert len(BOT_CATALOG) == len(profiles)


def test_get_bot_profile_round_trips_and_rejects_unknown_key():
    profile = get_bot_profile("shark")
    assert profile.key == "shark"
    bot = profile.factory("p0", "TestShark", random.Random(0))
    assert bot.name == "TestShark"

    with pytest.raises(KeyError):
        get_bot_profile("does_not_exist")
