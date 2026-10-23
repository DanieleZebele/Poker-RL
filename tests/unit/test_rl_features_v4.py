"""The newest observation encoding (feature version 4): one scale for every amount, and the
hand's actions as a sequence. Version 3, still served, is pinned in `test_rl_features.py`.
"""

from __future__ import annotations

import dataclasses
import math
import random

import pytest
from support import make_random_legal_bot

from pokerlab.cards.card import Card
from pokerlab.cards.deck import Deck
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.betting import apply_action, compute_legal_actions, post_blinds
from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import ActionRecord, HandState, PlayerState, PlayerStatus, Street
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player, build_observation
from pokerlab.players.rl_agent import PolicyDecision, RLAgentPlayer
from pokerlab.rl.action_space import legal_action_mask
from pokerlab.rl.features import (
    ACTION_FEATURES,
    CARDS_DIM,
    FEATURE_VERSION,
    FIELD_SCALARS_DIM,
    HISTORY_DIM,
    MASK_DIM,
    OBS_DIM_V3,
    SEQUENCE_DIM,
    SEQUENCE_LENGTH,
    SPR_SCALE,
    STREET_DIM,
    SUPPORTED_VERSIONS,
    _amount,
    _seats_to_act,
    _squash,
    encode_observation,
    observation_dim,
)
from pokerlab.rl.features import OBS_DIM_V4 as OBS_DIM
from pokerlab.rl.features import POT_SCALARS_DIM_V4 as POT_SCALARS_DIM
from pokerlab.rl.features import SEAT_BASE_FEATURES_V4 as SEAT_BASE_FEATURES
from pokerlab.rl.features import SEAT_FEATURES_V4 as SEAT_FEATURES
from pokerlab.rl.features import SEATS_DIM_V4 as SEATS_DIM

BIG_BLIND = 2
STARTING_STACK = 200

SEQUENCE_OFFSET = OBS_DIM - MASK_DIM - SEQUENCE_DIM
SEATS_OFFSET = CARDS_DIM + STREET_DIM + POT_SCALARS_DIM + FIELD_SCALARS_DIM


def encode(observation: Observation, legal_actions: list[LegalAction], version: int = 4) -> list[float]:
    return encode_observation(
        observation,
        big_blind=BIG_BLIND,
        starting_stack=STARTING_STACK,
        legal_mask=legal_action_mask(observation, legal_actions),
        version=version,
    )


def sequence_rows(vector: list[float]) -> list[list[float]]:
    block = vector[SEQUENCE_OFFSET : SEQUENCE_OFFSET + SEQUENCE_DIM]
    return [block[i : i + ACTION_FEATURES] for i in range(0, SEQUENCE_DIM, ACTION_FEATURES)]


def make_hand(stacks: dict[int, int], button_seat: int = 0) -> HandState:
    cards = ["Ah", "Kd", "7c", "2s", "Tc", "Th", "4d", "9s", "Js", "Qc", "5h", "6d", "3c", "8h", "Ad", "Kc", "2d", "3d"]
    seats = []
    for seat, stack in sorted(stacks.items()):
        ps = PlayerState(seat=seat, player_id=f"p{seat}", name=f"P{seat}", stack=stack)
        ps.hole_cards = (Card.parse(cards[2 * seat]), Card.parse(cards[2 * seat + 1]))
        seats.append(ps)
    return HandState(
        hand_id="h1",
        button_seat=button_seat,
        seats=seats,
        deck=Deck(),
        small_blind=1,
        big_blind=BIG_BLIND,
        community_cards=[],
    )


def three_handed_raise_and_call() -> HandState:
    """Button 0, small blind 1, big blind 2. Seat 0 raises to 6, seat 1 calls; the big blind is next."""
    hs = make_hand({0: 200, 1: 200, 2: 200})
    post_blinds(hs, sb_seat=1, bb_seat=2)
    apply_action(hs, 0, Action(ActionType.RAISE, 6))
    apply_action(hs, 1, Action(ActionType.CALL))
    return hs


class Probe(Player):
    """Plays randomly but encodes every observation it is asked to act on."""

    def __init__(self, player_id: str, name: str, rng: random.Random, sink: list) -> None:
        super().__init__(player_id, name)
        self._inner = make_random_legal_bot(player_id, name, rng)
        self._sink = sink

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self._sink.append((observation, encode(observation, legal_actions)))
        return self._inner.act(observation, legal_actions)


def play_probes(num_players: int, seed: int, hands: int = 30) -> list[tuple[Observation, list[float]]]:
    rng = random.Random(seed)
    sink: list[tuple[Observation, list[float]]] = []
    config = GameConfig(
        num_players=num_players, starting_stack=STARTING_STACK, small_blind=1, big_blind=BIG_BLIND
    )
    players = [Probe(f"p{i}", f"P{i}", rng, sink) for i in range(num_players)]
    Table(config, players, rng=rng).play_session(hands)
    assert sink, "the probe never got to act"
    return sink


# ---- shape and versions --------------------------------------------------------


def test_new_models_are_built_at_version_three_and_both_are_served():
    assert FEATURE_VERSION == 3
    assert SUPPORTED_VERSIONS == (3, 4)
    assert observation_dim(3) == OBS_DIM_V3 == 660
    assert observation_dim(4) == OBS_DIM
    with pytest.raises(ValueError, match="not one this build serves"):
        observation_dim(5)
    with pytest.raises(ValueError, match="not one this build serves"):
        encode_observation(
            build_observation(three_handed_raise_and_call(), 2),
            big_blind=BIG_BLIND,
            starting_stack=STARTING_STACK,
            legal_mask=[True] * MASK_DIM,
            version=5,
        )


def test_obs_dim_is_the_sum_of_its_sections():
    assert (
        CARDS_DIM
        + STREET_DIM
        + POT_SCALARS_DIM
        + FIELD_SCALARS_DIM
        + SEATS_DIM
        + HISTORY_DIM
        + SEQUENCE_DIM
        + MASK_DIM
        == OBS_DIM
    )
    assert SEATS_DIM == 9 * SEAT_FEATURES
    assert SEAT_FEATURES == SEAT_BASE_FEATURES + 20
    assert SEQUENCE_DIM == SEQUENCE_LENGTH * ACTION_FEATURES


@pytest.mark.parametrize("num_players", range(2, 10))
def test_encoding_is_fixed_size_and_bounded_across_table_sizes(num_players):
    for observation, vector in play_probes(num_players, 4321 + num_players):
        assert len(vector) == OBS_DIM
        for i, value in enumerate(vector):
            assert math.isfinite(value), f"non-finite feature {i} on {observation.street}"
            assert 0.0 <= value <= 1.0, f"feature {i} out of range: {value}"


def test_the_cards_the_street_and_the_mask_are_the_same_in_both_versions():
    hs = three_handed_raise_and_call()
    observation, legal = build_observation(hs, 2), compute_legal_actions(hs, 2)
    old, new = encode(observation, legal, version=3), encode(observation, legal, version=4)
    assert len(old) == OBS_DIM_V3 and len(new) == OBS_DIM
    assert new[: CARDS_DIM + STREET_DIM] == old[: CARDS_DIM + STREET_DIM]
    assert new[-MASK_DIM:] == old[-MASK_DIM:]


def test_version_four_does_not_read_the_starting_stack():
    hs = three_handed_raise_and_call()
    observation, legal = build_observation(hs, 2), compute_legal_actions(hs, 2)
    mask = legal_action_mask(observation, legal)
    kwargs = {"big_blind": BIG_BLIND, "legal_mask": mask, "version": 4}
    assert encode_observation(observation, starting_stack=200, **kwargs) == encode_observation(
        observation, starting_stack=20_000, **kwargs
    )


# ---- one scale for amounts -----------------------------------------------------


def test_every_amount_is_in_big_blinds_on_one_scale():
    one_bb, hundred_bb = _amount(BIG_BLIND, BIG_BLIND), _amount(100 * BIG_BLIND, BIG_BLIND)
    assert _amount(0, BIG_BLIND) == 0.0
    assert one_bb == pytest.approx(0.100, abs=1e-3)
    assert hundred_bb == pytest.approx(0.668, abs=1e-3)
    # The unit is the blind, not the chip: the same stack in big blinds reads the same
    assert _amount(50 * 100, 100) == _amount(50 * 2, 2)
    assert _amount(5 * BIG_BLIND, BIG_BLIND) < _amount(50 * BIG_BLIND, BIG_BLIND) < hundred_bb
    # Anything above the scale's 1000 bb is the top of it, not out of range
    assert _amount(10**9, BIG_BLIND) == 1.0


def test_a_ratio_without_a_unit_does_not_saturate():
    assert _squash(1, 1) == 0.5
    assert _squash(0.5, 1) < _squash(1, 1) < _squash(2, 1) < _squash(3, 1) < 1.0
    assert _squash(3, 1) != _squash(1, 1), "a 3x-pot bet must not read as a pot-sized one"
    assert _squash(4, 1, scale=SPR_SCALE) == 0.5
    assert _squash(5, 0) == 1.0 and _squash(0, 0) == 0.0


def test_stacks_and_bets_of_the_seats_use_the_same_scale_as_the_scalars():
    hs = three_handed_raise_and_call()
    observation, legal = build_observation(hs, 2), compute_legal_actions(hs, 2)
    vector = encode(observation, legal)
    # Slot 0 is me (seat 2: the big blind, who has 198 behind and 2 in front); slot 1 seat 0
    # (raised to 6: stack 194), slot 2 seat 1 (called: stack 194).
    me = vector[SEATS_OFFSET : SEATS_OFFSET + SEAT_BASE_FEATURES]
    raiser = vector[SEATS_OFFSET + SEAT_FEATURES : SEATS_OFFSET + SEAT_FEATURES + SEAT_BASE_FEATURES]
    assert me[5] == _amount(198, BIG_BLIND) and me[6] == _amount(2, BIG_BLIND)
    assert raiser[5] == _amount(194, BIG_BLIND) and raiser[6] == _amount(6, BIG_BLIND)
    assert raiser[7] == _amount(6, BIG_BLIND), "what the raiser has in over the hand"
    scalars = vector[CARDS_DIM + STREET_DIM : CARDS_DIM + STREET_DIM + POT_SCALARS_DIM]
    assert scalars[0] == _amount(observation.pot_size, BIG_BLIND)
    assert scalars[1] == _amount(198, BIG_BLIND)
    assert scalars[4] == _amount(4, BIG_BLIND), "to call: 6 - 2"
    pot_odds = scalars[9]
    assert pot_odds == pytest.approx(4 / (observation.pot_size + 4))


# ---- who still has to respond --------------------------------------------------


def test_the_big_blind_keeps_its_option_and_a_raise_reopens_the_action():
    hs = make_hand({0: 200, 1: 200, 2: 200})
    post_blinds(hs, sb_seat=1, bb_seat=2)
    assert _seats_to_act(build_observation(hs, 0)) == {0, 1, 2}

    apply_action(hs, 0, Action(ActionType.CALL))
    assert _seats_to_act(build_observation(hs, 1)) == {1, 2}
    apply_action(hs, 1, Action(ActionType.CALL))
    assert _seats_to_act(build_observation(hs, 2)) == {2}, "the big blind has not acted yet"

    apply_action(hs, 2, Action(ActionType.RAISE, 8))
    assert _seats_to_act(build_observation(hs, 0)) == {0, 1}, "the raise reopens it for the callers"


def test_the_seat_flag_says_who_still_has_to_respond():
    hs = three_handed_raise_and_call()
    vector = encode(build_observation(hs, 2), compute_legal_actions(hs, 2))
    to_act = [vector[SEATS_OFFSET + slot * SEAT_FEATURES + 9] for slot in range(3)]
    assert to_act == [1.0, 0.0, 0.0], "only the big blind (me) still has to respond to the raise"


@pytest.mark.parametrize("num_players", [2, 3, 6, 9])
def test_whoever_is_asked_to_act_still_has_to_respond(num_players):
    for observation, _vector in play_probes(num_players, 77 + num_players, hands=60):
        assert observation.my_seat in _seats_to_act(observation)


# ---- the action sequence -------------------------------------------------------


def test_the_sequence_has_one_row_per_action_oldest_first_and_right_aligned():
    hs = three_handed_raise_and_call()
    rows = sequence_rows(encode(build_observation(hs, 2), compute_legal_actions(hs, 2)))

    assert len(rows) == SEQUENCE_LENGTH
    assert all(row == [0.0] * ACTION_FEATURES for row in rows[: SEQUENCE_LENGTH - 4])
    small, big, raised, called = rows[-4:]
    for row in (small, big, raised, called):
        assert row[ACTION_FEATURES - 1] == 1.0, "a real action says it exists"

    def kind(row):  # fold, check, call, bet, raise, all-in, post
        return row[:7].index(1.0)

    def actor(row):  # relative to me: I am seat 2, so seat 0 is slot 1 and seat 1 slot 2
        return row[7:16].index(1.0)

    assert [kind(r) for r in (small, big, raised, called)] == [6, 6, 4, 2]
    assert [actor(r) for r in (small, big, raised, called)] == [2, 0, 1, 2]
    assert all(r[16] == 1.0 for r in (small, big, raised, called)), "all on the preflop"


def test_a_row_says_what_the_action_put_in_and_what_it_faced():
    hs = three_handed_raise_and_call()
    rows = sequence_rows(encode(build_observation(hs, 2), compute_legal_actions(hs, 2)))
    raised, called = rows[-2], rows[-1]
    base = 20

    # Seat 0 raised to 6 into a pot of 3 (the blinds), facing the big blind's 2.
    assert raised[base] == _amount(6, BIG_BLIND)
    assert raised[base + 1] == _squash(6, 3)
    assert raised[base + 2] == _amount(6, BIG_BLIND) and raised[base + 3] == _squash(6, 3)
    assert raised[base + 4] == _squash(2, 3)
    assert raised[base + 5] == 0.0
    assert raised[base + 6] == _squash(194, 3 + 6, SPR_SCALE)

    # Seat 1 called: 5 more chips (it had the small blind in), to the same 6, facing 5.
    assert called[base] == _amount(5, BIG_BLIND)
    assert called[base + 1] == _squash(5, 9)
    assert called[base + 2] == _amount(6, BIG_BLIND)
    assert called[base + 4] == _squash(5, 9)


def test_a_shove_is_an_all_in_row_and_a_fold_puts_nothing_in():
    hs = make_hand({0: 30, 1: 200, 2: 200})
    post_blinds(hs, sb_seat=1, bb_seat=2)
    apply_action(hs, 0, Action(ActionType.ALL_IN))
    apply_action(hs, 1, Action(ActionType.FOLD))
    rows = sequence_rows(encode(build_observation(hs, 2), compute_legal_actions(hs, 2)))
    shove, fold = rows[-2], rows[-1]
    base = 20

    assert shove[:7].index(1.0) == 5
    assert shove[base + 5] == 1.0, "left all-in"
    assert shove[base + 6] == 0.0, "with nothing behind"
    assert shove[base] == _amount(30, BIG_BLIND)
    assert fold[:7].index(1.0) == 0
    assert fold[base] == 0.0 and fold[base + 2] == 0.0, "a fold puts nothing in and goes to nothing"
    assert fold[base + 4] == _squash(29, 33), "it faced the shove, less its own small blind"


def test_a_long_hand_keeps_the_most_recent_actions():
    hs = make_hand({0: 200, 1: 200, 2: 200})
    post_blinds(hs, sb_seat=1, bb_seat=2)
    observation = build_observation(hs, 0)
    legal = compute_legal_actions(hs, 0)

    def check(i: int) -> ActionRecord:
        return ActionRecord(Street.FLOP, i % 3, f"p{i % 3}", ActionType.CHECK, 0, 100, 100, 50, 0.0)

    long = dataclasses.replace(
        observation, street=Street.FLOP, action_history=tuple(check(i) for i in range(35))
    )
    last_twenty = dataclasses.replace(long, action_history=long.action_history[-SEQUENCE_LENGTH:])
    assert sequence_rows(encode(long, legal)) == sequence_rows(encode(last_twenty, legal))
    assert all(row[ACTION_FEATURES - 1] == 1.0 for row in sequence_rows(encode(long, legal)))


@pytest.mark.parametrize("num_players", [2, 4, 9])
def test_the_sequence_matches_the_history_on_real_hands(num_players):
    for observation, vector in play_probes(num_players, 555 + num_players, hands=40):
        records = observation.action_history
        rows = sequence_rows(vector)
        filled = min(len(records), SEQUENCE_LENGTH)
        padding, real = rows[: SEQUENCE_LENGTH - filled], rows[SEQUENCE_LENGTH - filled :]
        assert all(row == [0.0] * ACTION_FEATURES for row in padding)
        assert all(row[ACTION_FEATURES - 1] == 1.0 for row in real)
        for row, record in zip(real, records[len(records) - filled :], strict=True):
            kind = row[:7].index(1.0)
            expected = {
                ActionType.FOLD: 0,
                ActionType.CHECK: 1,
                ActionType.CALL: 2,
                ActionType.BET: 3,
                ActionType.RAISE: 4,
                ActionType.ALL_IN: 5,
                ActionType.POST_BLIND: 6,
                ActionType.POST_ANTE: 6,
            }[record.action_type]
            assert kind == expected
            slots = [seat.seat for seat in observation.seats]
            mine = slots.index(observation.my_seat)
            relative = (slots.index(record.seat) - mine) % len(slots)
            assert row[7:16].index(1.0) == relative


# ---- versions at one table -----------------------------------------------------


def test_each_player_is_fed_the_version_its_policy_was_trained_on():
    lengths: dict[str, set[int]] = {"old": set(), "new": set(), "default": set()}

    def policy_for(name: str, version: int | None):
        def policy(features: list[float], mask: list[bool], *_style) -> PolicyDecision:
            lengths[name].add(len(features))
            return PolicyDecision(action_index=next(i for i, ok in enumerate(mask) if ok and i != 0))

        if version is not None:
            policy.feature_version = version  # type: ignore[attr-defined]
        return policy

    config = GameConfig(num_players=3, starting_stack=STARTING_STACK, small_blind=1, big_blind=BIG_BLIND)
    players = [
        RLAgentPlayer("p0", "old", policy_fn=policy_for("old", 3), big_blind=BIG_BLIND, starting_stack=STARTING_STACK),
        RLAgentPlayer("p1", "new", policy_fn=policy_for("new", 4), big_blind=BIG_BLIND, starting_stack=STARTING_STACK),
        RLAgentPlayer(
            "p2", "default", policy_fn=policy_for("default", None), big_blind=BIG_BLIND, starting_stack=STARTING_STACK
        ),
    ]
    Table(config, players, rng=random.Random(5)).play_session(20)

    assert lengths == {"old": {OBS_DIM_V3}, "new": {OBS_DIM}, "default": {OBS_DIM_V3}}


def test_a_player_told_its_version_overrides_the_policys():
    seen: list[int] = []

    def policy(features: list[float], mask: list[bool], *_style) -> PolicyDecision:
        seen.append(len(features))
        return PolicyDecision(action_index=next(i for i, ok in enumerate(mask) if ok and i != 0))

    policy.feature_version = 4  # type: ignore[attr-defined]
    config = GameConfig(num_players=2, starting_stack=STARTING_STACK, small_blind=1, big_blind=BIG_BLIND)
    rng = random.Random(1)
    players = [
        RLAgentPlayer(
            "p0", "a", policy_fn=policy, big_blind=BIG_BLIND, starting_stack=STARTING_STACK, feature_version=3
        ),
        make_random_legal_bot("p1", "b", rng),
    ]
    Table(config, players, rng=rng).play_session(5)
    assert set(seen) == {OBS_DIM_V3}


def test_a_seat_folded_or_all_in_is_not_waiting_to_respond():
    hs = make_hand({0: 30, 1: 200, 2: 200})
    post_blinds(hs, sb_seat=1, bb_seat=2)
    apply_action(hs, 0, Action(ActionType.ALL_IN))
    apply_action(hs, 1, Action(ActionType.FOLD))
    observation = build_observation(hs, 2)
    status = {seat.seat: seat.status for seat in observation.seats}
    assert status[0] is PlayerStatus.ALL_IN and status[1] is PlayerStatus.FOLDED
    assert _seats_to_act(observation) == {2}
