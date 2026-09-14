from __future__ import annotations

import math
import random

import pytest

from pokerlab.cards.card import Card
from pokerlab.cards.deck import Deck
from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.betting import compute_legal_actions, post_blinds
from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import HandState, PlayerState, Street
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player, build_observation
from pokerlab.players.scripted import make_random_legal_bot
from pokerlab.rl.action_space import ACTION_DIM, legal_action_mask
from pokerlab.rl.features import (
    CARDS_DIM,
    FIELD_SCALARS_DIM,
    HISTORY_DIM,
    MASK_DIM,
    MAX_SEATS,
    OBS_DIM,
    POT_SCALARS_DIM,
    SEAT_FEATURES,
    SEATS_DIM,
    STREET_DIM,
    _card_index,
    committed_by_seat,
    encode_observation,
)

BIG_BLIND = 2
STARTING_STACK = 200

STREET_OFFSET = CARDS_DIM
FIELD_OFFSET = CARDS_DIM + STREET_DIM + POT_SCALARS_DIM
SEATS_OFFSET = CARDS_DIM + STREET_DIM + POT_SCALARS_DIM + FIELD_SCALARS_DIM
MASK_OFFSET = OBS_DIM - MASK_DIM


def encode(observation: Observation, legal_actions: list[LegalAction]) -> list[float]:
    return encode_observation(
        observation,
        big_blind=BIG_BLIND,
        starting_stack=STARTING_STACK,
        legal_mask=legal_action_mask(observation, legal_actions),
    )


def make_hand(
    stacks: dict[int, int],
    hole_cards: dict[int, tuple[Card, Card]],
    board: tuple[Card, ...] = (),
    button_seat: int = 0,
) -> HandState:
    seats = []
    for seat, stack in sorted(stacks.items()):
        ps = PlayerState(seat=seat, player_id=f"p{seat}", name=f"P{seat}", stack=stack)
        ps.hole_cards = hole_cards[seat]
        seats.append(ps)
    return HandState(
        hand_id="h1",
        button_seat=button_seat,
        seats=seats,
        deck=Deck(),
        small_blind=1,
        big_blind=BIG_BLIND,
        community_cards=list(board),
    )


class FeatureProbe(Player):
    """Plays randomly but encodes every observation it is asked to act on."""

    def __init__(self, player_id: str, name: str, rng: random.Random, sink: list) -> None:
        super().__init__(player_id, name)
        self._inner = make_random_legal_bot(player_id, name, rng)
        self._sink = sink

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self._sink.append((observation, encode(observation, legal_actions)))
        return self._inner.act(observation, legal_actions)


def test_obs_dim_is_the_sum_of_its_sections():
    assert (
        CARDS_DIM + STREET_DIM + POT_SCALARS_DIM + FIELD_SCALARS_DIM + SEATS_DIM + HISTORY_DIM + MASK_DIM
        == OBS_DIM
    )
    assert SEATS_DIM == MAX_SEATS * SEAT_FEATURES
    assert MASK_DIM == ACTION_DIM


@pytest.mark.parametrize("num_players", range(2, 10))
def test_encoding_is_fixed_size_and_bounded_across_table_sizes(num_players):
    rng = random.Random(1234 + num_players)
    sink: list[tuple[Observation, list[float]]] = []
    config = GameConfig(
        num_players=num_players, starting_stack=STARTING_STACK, small_blind=1, big_blind=BIG_BLIND
    )
    players = [FeatureProbe(f"p{i}", f"P{i}", rng, sink) for i in range(num_players)]
    Table(config, players, rng=rng).play_session(30)

    assert sink, "the probe never got to act"
    for observation, vector in sink:
        assert len(vector) == OBS_DIM
        for i, value in enumerate(vector):
            assert math.isfinite(value), f"non-finite feature {i} on {observation.street}"
            assert 0.0 <= value <= 1.0, f"feature {i} out of range: {value}"


@pytest.mark.parametrize("num_players", range(2, 10))
def test_committed_by_seat_reconstructs_the_pot(num_players):
    """SeatPublicInfo hides per-hand totals, so they are rebuilt from the action
    log; if that derivation is wrong the per-seat features are silently wrong."""
    rng = random.Random(99 + num_players)
    sink: list[tuple[Observation, list[float]]] = []
    config = GameConfig(
        num_players=num_players, starting_stack=STARTING_STACK, small_blind=1, big_blind=BIG_BLIND
    )
    players = [FeatureProbe(f"p{i}", f"P{i}", rng, sink) for i in range(num_players)]
    Table(config, players, rng=rng).play_session(30)

    for observation, _ in sink:
        assert sum(committed_by_seat(observation).values()) == observation.pot_size


def test_board_planes_are_empty_before_the_flop():
    hole = {0: (Card.parse("Ah"), Card.parse("Kd")), 1: (Card.parse("7c"), Card.parse("2s"))}
    hs = make_hand({0: 200, 1: 200}, hole)
    post_blinds(hs, sb_seat=0, bb_seat=1)
    vector = encode(build_observation(hs, 0), compute_legal_actions(hs, 0))

    assert sum(vector[0:52]) == 2  # hole cards
    assert sum(vector[52 : 4 * 52]) == 0  # flop / turn / river planes
    assert sum(vector[4 * 52 : 5 * 52]) == 0  # whole board
    assert sum(vector[5 * 52 : 6 * 52]) == 2  # hole + board


def test_board_planes_match_the_dealt_cards():
    hole = {0: (Card.parse("Ah"), Card.parse("Kd")), 1: (Card.parse("7c"), Card.parse("2s"))}
    board = tuple(Card.parse(c) for c in ("Qs", "Jh", "9d", "3c"))
    hs = make_hand({0: 200, 1: 200}, hole, board=board)
    hs.street = Street.TURN
    vector = encode(build_observation(hs, 0), compute_legal_actions(hs, 0))

    assert sum(vector[52 : 2 * 52]) == 3  # flop
    assert sum(vector[2 * 52 : 3 * 52]) == 1  # turn
    assert sum(vector[3 * 52 : 4 * 52]) == 0  # river not dealt
    assert sum(vector[4 * 52 : 5 * 52]) == 4
    assert sum(vector[5 * 52 : 6 * 52]) == 6

    assert vector[52 + _card_index(Card.parse("Qs"))] == 1.0
    assert vector[2 * 52 + _card_index(Card.parse("3c"))] == 1.0
    assert vector[_card_index(Card.parse("Ah"))] == 1.0
    assert vector[_card_index(Card.parse("Qs"))] == 0.0  # board card is not a hole card


def test_street_one_hot_tracks_the_street():
    hole = {0: (Card.parse("Ah"), Card.parse("Kd")), 1: (Card.parse("7c"), Card.parse("2s"))}
    board = tuple(Card.parse(c) for c in ("Qs", "Jh", "9d"))
    hs = make_hand({0: 200, 1: 200}, hole, board=board)
    hs.street = Street.FLOP
    vector = encode(build_observation(hs, 0), compute_legal_actions(hs, 0))
    assert vector[STREET_OFFSET : STREET_OFFSET + STREET_DIM] == [0.0, 1.0, 0.0, 0.0]


def test_seat_slots_are_relative_so_slot_zero_is_always_me():
    hole = {
        0: (Card.parse("Ah"), Card.parse("Kd")),
        1: (Card.parse("7c"), Card.parse("2s")),
        2: (Card.parse("Tc"), Card.parse("Th")),
    }
    hs = make_hand({0: 200, 1: 150, 2: 100}, hole, button_seat=0)
    post_blinds(hs, sb_seat=1, bb_seat=2)

    for seat in (0, 1, 2):
        vector = encode(build_observation(hs, seat), compute_legal_actions(hs, seat))
        slots = vector[SEATS_OFFSET : SEATS_OFFSET + SEATS_DIM]
        is_me = [slots[s * SEAT_FEATURES + 1] for s in range(MAX_SEATS)]
        assert is_me == [1.0] + [0.0] * (MAX_SEATS - 1)
        occupied = [slots[s * SEAT_FEATURES] for s in range(MAX_SEATS)]
        assert occupied == [1.0, 1.0, 1.0] + [0.0] * (MAX_SEATS - 3)


def test_position_feature_follows_postflop_action_order():
    """Postflop action opens on the small blind and closes on the button, so the
    seat *before* the button is second-best. Ranking by seating order clockwise
    from the button instead would invert this and score the small blind high."""
    ranks = ["Ah", "Kd", "7c", "2s", "Tc", "Th", "4d", "9s", "Js", "Qc", "5h", "6d"]
    hole = {s: (Card.parse(ranks[2 * s]), Card.parse(ranks[2 * s + 1])) for s in range(6)}
    hs = make_hand({s: 200 for s in range(6)}, hole, button_seat=0)
    post_blinds(hs, sb_seat=1, bb_seat=2)

    position = {
        seat: encode(build_observation(hs, seat), compute_legal_actions(hs, seat))[FIELD_OFFSET]
        for seat in range(6)
    }

    assert position[0] == 1.0  # button acts last postflop
    assert position[1] == 0.0  # small blind acts first
    assert position[5] > position[4] > position[3] > position[2] > position[1]
    assert position[5] > position[2], "the seat before the button beats the big blind"


def test_mask_echo_matches_the_legal_action_mask():
    hole = {0: (Card.parse("Ah"), Card.parse("Kd")), 1: (Card.parse("7c"), Card.parse("2s"))}
    hs = make_hand({0: 200, 1: 200}, hole)
    post_blinds(hs, sb_seat=0, bb_seat=1)
    observation = build_observation(hs, 0)
    legal = compute_legal_actions(hs, 0)
    mask = legal_action_mask(observation, legal)

    vector = encode(observation, legal)
    assert vector[MASK_OFFSET:] == [1.0 if allowed else 0.0 for allowed in mask]


def test_encoding_is_deterministic():
    hole = {0: (Card.parse("Ah"), Card.parse("Kd")), 1: (Card.parse("7c"), Card.parse("2s"))}
    hs = make_hand({0: 200, 1: 200}, hole)
    post_blinds(hs, sb_seat=0, bb_seat=1)
    observation = build_observation(hs, 0)
    legal = compute_legal_actions(hs, 0)
    assert encode(observation, legal) == encode(observation, legal)
