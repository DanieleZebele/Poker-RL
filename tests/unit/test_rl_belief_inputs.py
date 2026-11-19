"""What the belief network of feature version 5 reads, and the table's memory (`rl/belief_inputs.py`).

No torch: the question asked at a decision, the hands the memory reads and the tracker's laziness are
plain Python, and so is the layout of version 5 in `rl/features.py`.
"""

from __future__ import annotations

import math
import random

import pytest
from support import make_random_legal_bot

from pokerlab.engine.actions import ActionType
from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import PlayerStatus
from pokerlab.engine.table import Table
from pokerlab.players.base import Player
from pokerlab.rl.action_space import legal_action_mask
from pokerlab.rl.belief_inputs import (
    COMBO_CARDS,
    KEEP_WITHOUT_READERS,
    NUM_COMBOS,
    POSITIONS,
    STATIC_DIM,
    STREETS,
    MemoryTracker,
    action_tokens,
    combo_index,
    decision_question,
    memory_hand,
    positions,
)
from pokerlab.rl.features import (
    BELIEF_FEATURES,
    MAX_SEATS,
    OBS_DIM_V3,
    OBS_DIM_V5,
    SEAT_BASE_FEATURES_V3,
    SEAT_FEATURES_V3,
    SEAT_FEATURES_V5,
    card_index,
    encode_observation,
)
from pokerlab.rl.rollout import TableBank
from pokerlab.rl.table_mix import TableMix


class _Table(Table):
    """A table whose stacks are reset before every hand, as everywhere in training."""

    def play_hand(self):
        self.stacks = [2000] * len(self.stacks)
        return super().play_hand()


class _Spy(Player):
    """A random player that keeps every observation it is shown."""

    def __init__(self, player_id: str, rng: random.Random, seen: list) -> None:
        super().__init__(player_id, player_id)
        self._bot = make_random_legal_bot(player_id, rng=rng)
        self._seen = seen

    def act(self, observation, legal_actions):
        self._seen.append((observation, legal_actions))
        return self._bot.act(observation, legal_actions)


def _table(size: int, seed: int, seen: list, tracker=None) -> Table:
    rng = random.Random(seed)
    players = [_Spy(f"p{seat}", random.Random(seed * 10 + seat), seen) for seat in range(size)]
    return _Table(GameConfig(size, 2000, 10, 20), players, rng=rng, memory_tracker=tracker)


def test_every_pair_of_cards_is_one_combo():
    assert len(COMBO_CARDS) == NUM_COMBOS == len(set(COMBO_CARDS))
    assert all(combo_index(a, b) == combo_index(b, a) == i for i, (a, b) in enumerate(COMBO_CARDS))


@pytest.mark.parametrize("size", [2, 3, 6, 9])
def test_every_seat_has_one_position_and_the_button_is_named(size):
    names = positions(list(range(size)), button_seat=size - 1)
    assert sorted(names) == list(range(size))  # several middle seats share MP at a full table
    assert names[size - 1] == "BTN"
    assert set(names.values()) <= set(POSITIONS)


@pytest.mark.parametrize("size", [2, 4, 6, 9])
def test_a_decision_asks_about_the_hand_so_far_as_the_finished_hand_reads_it(size):
    """The tokens a decision is given are the first tokens of the finished hand, the statics hold
    each live opponent's stack at the start of the hand, and the observer's cards are known."""
    seen: list = []
    table = _table(size, seed=size, seen=seen)
    for _ in range(15):
        start = len(seen)
        hand = table.play_hand().hand_history
        whole = action_tokens(hand.actions, sorted(hand.starting_stacks), hand.button_seat, hand.big_blind)
        for observation, _legal in seen[start:]:
            asked = decision_question(observation, hand.big_blind)
            live = [s.seat for s in observation.seats if s.status is not PlayerStatus.FOLDED]
            if len(live) < 2:
                assert asked is None
                continue
            count = len(observation.action_history)
            assert asked["tokens"] == whole.tokens[:count]
            assert asked["actors"] == whole.actors[:count]
            assert asked["targets"] == [seat for seat in live if seat != observation.my_seat]
            names = positions(sorted(hand.starting_stacks), hand.button_seat)
            street = STREETS.index(observation.street)
            for seat, static in zip(asked["targets"], asked["static"], strict=True):
                assert len(static) == STATIC_DIM
                assert static[POSITIONS.index(names[seat])] == 1.0
                assert static[len(POSITIONS) + 2 + street] == 1.0
                expected = math.log1p(hand.starting_stacks[seat] / hand.big_blind) / math.log1p(100.0)
                assert static[len(POSITIONS) + 1] == pytest.approx(expected)
            mine = {card_index(card) for card in observation.hole_cards}
            assert mine <= asked["known"]
            assert set(asked["board"]) == {card_index(card) for card in observation.community_cards}


def test_the_memory_reads_a_players_cards_only_when_a_showdown_revealed_them():
    seen: list = []
    table = _table(4, seed=3, seen=seen)
    shown = hidden = 0
    for _ in range(80):
        hand = table.play_hand().hand_history
        read = memory_hand(hand, {seat: f"p{seat}" for seat in hand.starting_stacks})
        folded = {r.seat for r in hand.actions if r.action_type is ActionType.FOLD}
        still_in = set(hand.starting_stacks) - folded
        for player in read.players:
            if len(still_in) >= 2 and player.seat in still_in:
                shown += 1
                assert player.shown
                pair = hand.hole_cards[player.seat]
                assert player.cards == (card_index(pair[0]), card_index(pair[1]))
            else:
                hidden += 1
                assert not player.shown and player.cards == (-1, -1)
            assert player.result_bb == (hand.final_stacks[player.seat] - hand.starting_stacks[player.seat]) / 20
        assert len(read.board) == 5
    assert shown and hidden


def _counting(calls: dict, key: str):
    def update(hands, states):
        calls.setdefault(key, []).append(len(hands))
        for hand in hands:
            for player in hand.players:
                states[player.player_id] = states.get(player.player_id, 0) + 1
    return update


def test_the_tracker_reads_each_hand_once_per_reader_and_only_when_asked():
    tracker = MemoryTracker()
    seen: list = []
    table = _table(3, seed=5, seen=seen, tracker=tracker)
    calls: dict[str, list[int]] = {}
    ids = {0: "p0", 1: "p1", 2: "p2"}
    for _ in range(4):
        table.play_hand()
    assert tracker.hands == 4 and calls == {}  # nobody asked: nothing read
    view = tracker.view(ids)
    assert view.states("a", _counting(calls, "a")) == {0: 4, 1: 4, 2: 4}
    assert view.states("b", _counting(calls, "b")) == {0: 4, 1: 4, 2: 4}
    assert view.states("a", _counting(calls, "a")) == {0: 4, 1: 4, 2: 4}  # nothing new: not called
    table.play_hand()
    assert view.states("a", _counting(calls, "a")) == {0: 4, 1: 4, 2: 4}  # a view is the memory before its hand
    view = tracker.view(ids)
    assert view.states("a", _counting(calls, "a")) == {0: 5, 1: 5, 2: 5}
    assert view.states("b", _counting(calls, "b")) == {0: 5, 1: 5, 2: 5}
    assert calls == {"a": [4, 1], "b": [4, 1]}
    assert tracker.view({0: "p0", 1: "newcomer"}).states("a", _counting(calls, "a")) == {0: 5, 1: None}


def test_a_tracker_keeps_only_the_last_hands_and_those_someone_still_has_to_read():
    tracker = MemoryTracker()
    seen: list = []
    table = _table(3, seed=8, seen=seen, tracker=tracker)
    for _ in range(KEEP_WITHOUT_READERS + 15):
        table.play_hand()
    assert len(tracker._hands) == KEEP_WITHOUT_READERS  # a table nobody reads keeps a short list
    calls: dict[str, list[int]] = {}
    states = tracker.view({0: "p0"}).states("a", _counting(calls, "a"))
    assert calls == {"a": [KEEP_WITHOUT_READERS]} and states == {0: KEEP_WITHOUT_READERS}
    for _ in range(30):
        table.play_hand()
    assert len(tracker._hands) == 30  # waiting for its reader
    tracker.view({0: "p0"}).states("a", _counting(calls, "a"))
    assert calls["a"][-1] == 30 and len(tracker._hands) == KEEP_WITHOUT_READERS


def test_a_view_asked_once_its_hand_is_over_and_a_later_one_was_read_refuses():
    tracker = MemoryTracker()
    seen: list = []
    table = _table(3, seed=6, seen=seen, tracker=tracker)
    table.play_hand()
    stale = tracker.view({0: "p0"})
    table.play_hand()
    tracker.view({0: "p0"}).states("a", lambda hands, states: None)
    with pytest.raises(RuntimeError, match="that hand is over"):
        stale.states("a", lambda hands, states: None)


def test_an_observation_carries_the_view_of_the_hands_before_it():
    tracker = MemoryTracker()
    seen: list = []
    table = _table(3, seed=7, seen=seen, tracker=tracker)
    for number in range(5):
        start = len(seen)
        table.play_hand()
        for observation, _ in seen[start:]:
            assert observation.seat_memory is not None
            assert observation.seat_memory.player_ids == {0: "p0", 1: "p1", 2: "p2"}
        assert tracker.hands == number + 1


def test_a_table_bank_starts_its_memory_afresh_with_its_statistics():
    bank = TableBank(TableMix(), random.Random(0))
    for proxy in bank.seats(3):
        proxy.inner = make_random_legal_bot(proxy.player_id, rng=random.Random(1))
    bank.play_hand(3)
    table = bank._entry(3)[0]
    assert table.memory_tracker.hands == 1
    bank.reset_stats(3)
    assert table.memory_tracker.hands == 0


def _encoded(version, beliefs=None):
    seen: list = []
    _table(6, seed=11, seen=seen).play_hand()
    observation, legal = seen[-1]
    mask = legal_action_mask(observation, legal)
    return observation, encode_observation(observation, big_blind=20, starting_stack=2000, legal_mask=mask,
                                           version=version, beliefs=beliefs)


def test_version_5_puts_each_seats_belief_where_version_3_put_its_statistics():
    observation, base = _encoded(3)
    assert len(base) == OBS_DIM_V3
    others = [s.seat for s in observation.seats if s.seat != observation.my_seat]
    beliefs = {seat: [0.01 * (seat + 1)] * BELIEF_FEATURES for seat in others}
    _, features = _encoded(5, beliefs)
    assert len(features) == OBS_DIM_V5
    seats = list(observation.seats)
    me = next(i for i, s in enumerate(seats) if s.seat == observation.my_seat)
    rotated = seats[me:] + seats[:me]
    start = len(base) - (MAX_SEATS * SEAT_FEATURES_V3) - 40 - 11  # the seat block, as version 3 lays it out
    for slot, seat in enumerate(rotated):
        v3 = start + slot * SEAT_FEATURES_V3
        v5 = start + slot * SEAT_FEATURES_V5
        assert features[v5 : v5 + SEAT_BASE_FEATURES_V3] == base[v3 : v3 + SEAT_BASE_FEATURES_V3]
        held = features[v5 + SEAT_BASE_FEATURES_V3 : v5 + SEAT_FEATURES_V5]
        assert held == beliefs.get(seat.seat, [0.0] * BELIEF_FEATURES)
    # everything before and after the seats is version 3's
    assert features[:start] == base[:start]
    assert features[start + MAX_SEATS * SEAT_FEATURES_V5 :] == base[start + MAX_SEATS * SEAT_FEATURES_V3 :]


def test_version_5_reads_no_opponent_statistics_and_refuses_a_belief_of_the_wrong_size():
    _, without = _encoded(5, {})
    assert len(without) == OBS_DIM_V5
    observation, _ = _encoded(3)
    other = next(s.seat for s in observation.seats if s.seat != observation.my_seat)
    with pytest.raises(ValueError, match="belief"):
        _encoded(5, {other: [0.0] * (BELIEF_FEATURES - 1)})


def test_near_keeps_one_count_for_every_street_or_one_per_street():
    from pokerlab.rl.belief_inputs import near_combos

    assert [near_combos((64,), n) for n in (0, 3, 4, 5)] == [64, 64, 64, 64]
    assert [near_combos((200, 64, 32, 16), n) for n in (0, 3, 4, 5)] == [200, 64, 32, 16]
