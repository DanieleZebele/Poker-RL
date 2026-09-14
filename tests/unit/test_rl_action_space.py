from __future__ import annotations

import pytest

from pokerlab.cards.deck import Deck
from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.betting import (
    apply_action,
    compute_legal_actions,
    post_blinds,
    start_new_street_betting,
)
from pokerlab.engine.state import HandState, PlayerState, Street
from pokerlab.players.base import build_observation
from pokerlab.rl.action_space import (
    ACTION_DIM,
    ALL_IN_BIN,
    CHECK_CALL_BIN,
    FIRST_FRACTION_BIN,
    FOLD_BIN,
    POT_FRACTIONS,
    RAISE_MIN_BIN,
    _aggressive_legal,
    action_index_to_action,
    legal_action_mask,
    raise_to_for_fraction,
)

RAISE_BINS = (RAISE_MIN_BIN, *range(FIRST_FRACTION_BIN, FIRST_FRACTION_BIN + len(POT_FRACTIONS)))


def make_hand(stacks: dict[int, int], small_blind=1, big_blind=2, button_seat=0) -> HandState:
    deck = Deck()
    seats = []
    for seat, stack in sorted(stacks.items()):
        ps = PlayerState(seat=seat, player_id=f"p{seat}", name=f"P{seat}", stack=stack)
        ps.hole_cards = (deck.deal(1)[0], deck.deal(1)[0])
        seats.append(ps)
    return HandState(
        hand_id="h1",
        button_seat=button_seat,
        seats=seats,
        deck=deck,
        small_blind=small_blind,
        big_blind=big_blind,
    )


def obs_and_legal(hand_state: HandState, seat: int):
    return build_observation(hand_state, seat), compute_legal_actions(hand_state, seat)


def preflop_utg():
    hs = make_hand({0: 200, 1: 200, 2: 200}, button_seat=0)
    post_blinds(hs, sb_seat=1, bb_seat=2)
    return hs, 0


def big_blind_option():
    """Everyone limps to the BB: to_call == 0 but a live bet (the BB) exists."""
    hs = make_hand({0: 200, 1: 200, 2: 200}, button_seat=0)
    post_blinds(hs, sb_seat=1, bb_seat=2)
    apply_action(hs, 0, Action(ActionType.CALL))
    apply_action(hs, 1, Action(ActionType.CALL))
    return hs, 2


def flop_first_to_act():
    hs = make_hand({0: 200, 1: 200}, button_seat=0)
    post_blinds(hs, sb_seat=0, bb_seat=1)
    apply_action(hs, 0, Action(ActionType.CALL))
    apply_action(hs, 1, Action(ActionType.CHECK))
    start_new_street_betting(hs, Street.FLOP)
    return hs, 1


def flop_facing_a_bet():
    hs, _ = flop_first_to_act()
    apply_action(hs, 1, Action(ActionType.BET, amount=10))
    return hs, 0


def flop_facing_a_bet_short_stack():
    hs = make_hand({0: 26, 1: 200}, button_seat=0)
    post_blinds(hs, sb_seat=0, bb_seat=1)
    apply_action(hs, 0, Action(ActionType.CALL))
    apply_action(hs, 1, Action(ActionType.CHECK))
    start_new_street_betting(hs, Street.FLOP)
    apply_action(hs, 1, Action(ActionType.BET, amount=8))
    return hs, 0


SCENARIOS = [
    preflop_utg,
    big_blind_option,
    flop_first_to_act,
    flop_facing_a_bet,
    flop_facing_a_bet_short_stack,
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_mask_is_fixed_size_and_all_in_is_always_available(scenario):
    hs, seat = scenario()
    obs, legal = obs_and_legal(hs, seat)
    mask = legal_action_mask(obs, legal)
    assert len(mask) == ACTION_DIM
    assert mask[ALL_IN_BIN], "an actionable seat can always shove"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_every_unmasked_bin_is_accepted_by_the_engine(scenario):
    """The action-mapping analogue of chip conservation: no bin the policy is
    allowed to sample may ever produce an IllegalActionError."""
    hs, seat = scenario()
    obs, legal = obs_and_legal(hs, seat)
    mask = legal_action_mask(obs, legal)

    for index, allowed in enumerate(mask):
        if not allowed:
            continue
        fresh_hs, fresh_seat = scenario()  # apply_action mutates, so rebuild per bin
        fresh_obs, fresh_legal = obs_and_legal(fresh_hs, fresh_seat)
        action = action_index_to_action(index, fresh_obs, fresh_legal)
        apply_action(fresh_hs, fresh_seat, action)


def test_big_blind_option_maps_to_raise_never_bet():
    """Mirror of the engine's big-blind-option regression: with to_call == 0 but
    current_bet_to_match > 0, the aggressive bins must emit RAISE, not BET."""
    hs, seat = big_blind_option()
    obs, legal = obs_and_legal(hs, seat)
    assert obs.current_bet_to_match == 2
    assert obs.my_current_bet == 2  # to_call == 0, yet a live bet exists

    mask = legal_action_mask(obs, legal)
    assert any(mask[b] for b in RAISE_BINS), "the BB must keep its option to raise"
    for index in RAISE_BINS:
        if mask[index]:
            assert action_index_to_action(index, obs, legal).action_type == ActionType.RAISE


def test_fresh_street_maps_to_bet():
    hs, seat = flop_first_to_act()
    obs, legal = obs_and_legal(hs, seat)
    assert obs.current_bet_to_match == 0
    mask = legal_action_mask(obs, legal)
    for index in RAISE_BINS:
        if mask[index]:
            assert action_index_to_action(index, obs, legal).action_type == ActionType.BET


def test_fold_is_masked_when_checking_is_free():
    hs, seat = flop_first_to_act()
    obs, legal = obs_and_legal(hs, seat)
    assert ActionType.FOLD in {la.action_type for la in legal}
    assert not legal_action_mask(obs, legal)[FOLD_BIN]
    assert legal_action_mask(obs, legal, mask_dominated_folds=False)[FOLD_BIN]


def test_fold_stays_available_when_facing_a_bet():
    hs, seat = flop_facing_a_bet()
    obs, legal = obs_and_legal(hs, seat)
    assert legal_action_mask(obs, legal)[FOLD_BIN]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_unmasked_raise_bins_land_inside_the_legal_bounds(scenario):
    hs, seat = scenario()
    obs, legal = obs_and_legal(hs, seat)
    mask = legal_action_mask(obs, legal)
    aggressive = _aggressive_legal(legal)
    for index in RAISE_BINS:
        if not mask[index]:
            continue
        action = action_index_to_action(index, obs, legal)
        assert aggressive.min_amount <= action.amount < aggressive.max_amount


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_colliding_raise_levels_are_deduplicated(scenario):
    hs, seat = scenario()
    obs, legal = obs_and_legal(hs, seat)
    mask = legal_action_mask(obs, legal)
    amounts = [
        action_index_to_action(i, obs, legal).amount for i in RAISE_BINS if mask[i]
    ]
    assert len(amounts) == len(set(amounts)), "two bins must never mean the same raise"


def test_oversized_raise_bins_are_masked_not_clamped():
    """A 26-chip stack cannot make a 2x-pot raise; that bin must go dark rather
    than silently collapse onto the shove."""
    hs, seat = flop_facing_a_bet_short_stack()
    obs, legal = obs_and_legal(hs, seat)
    aggressive = _aggressive_legal(legal)
    mask = legal_action_mask(obs, legal)

    biggest_bin = FIRST_FRACTION_BIN + len(POT_FRACTIONS) - 1
    assert raise_to_for_fraction(obs, POT_FRACTIONS[-1]) > aggressive.max_amount
    assert not mask[biggest_bin]
    for index in RAISE_BINS:
        if mask[index]:
            assert action_index_to_action(index, obs, legal).amount != aggressive.max_amount


def test_check_call_bin_picks_check_or_call_from_the_legal_actions():
    hs, seat = flop_first_to_act()
    obs, legal = obs_and_legal(hs, seat)
    assert action_index_to_action(CHECK_CALL_BIN, obs, legal).action_type == ActionType.CHECK

    hs, seat = flop_facing_a_bet()
    obs, legal = obs_and_legal(hs, seat)
    action = action_index_to_action(CHECK_CALL_BIN, obs, legal)
    assert action.action_type == ActionType.CALL
    assert action.amount == 0  # the engine derives call chips; see Action's docstring


def test_pot_fraction_raise_to_is_measured_after_the_call():
    hs, seat = flop_facing_a_bet()
    obs, _ = obs_and_legal(hs, seat)
    # Preflop both put in 2, then seat 1 bet 10 on the flop: pot 14, to_call 10.
    assert obs.pot_size == 14
    assert obs.current_bet_to_match == 10
    # A pot-sized raise calls the 10 and raises by the 24-chip pot that leaves.
    assert raise_to_for_fraction(obs, 1.0) == 34
    assert raise_to_for_fraction(obs, 0.5) == 22


def test_raise_bins_are_all_masked_when_no_aggressive_action_is_legal():
    """Facing a bet a short stack cannot cover, only fold and shove remain."""
    hs = make_hand({0: 200, 1: 6}, button_seat=0)
    post_blinds(hs, sb_seat=0, bb_seat=1)
    apply_action(hs, 0, Action(ActionType.RAISE, amount=40))
    obs, legal = obs_and_legal(hs, 1)
    mask = legal_action_mask(obs, legal)

    assert not any(mask[b] for b in RAISE_BINS)
    assert mask[FOLD_BIN] and mask[ALL_IN_BIN]
    assert not mask[CHECK_CALL_BIN]
    with pytest.raises(ValueError, match="no BET/RAISE is legal"):
        action_index_to_action(RAISE_MIN_BIN, obs, legal)
