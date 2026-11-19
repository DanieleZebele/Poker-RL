"""The belief network in play (`rl/belief.py`) and the models of feature version 5 built on it."""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from support import (
    TINY_BELIEF,
    fake_collector,
    make_random_legal_bot,
    tiny_belief_model,
)

from pokerlab.engine.state import PlayerStatus
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.belief import card_planes
from pokerlab.rl.belief_inputs import MemoryTracker, decision_question, memory_hand
from pokerlab.rl.features import (
    BELIEF_FEATURES,
    MAX_SEATS,
    OBS_DIM_V5,
    SEAT_BASE_FEATURES_V3,
    SEAT_FEATURES_V5,
    card_index,
    encode_observation,
)
from pokerlab.rl.policy import make_policy_fn
from pokerlab.rl.ppo import (
    IncompatibleCheckpointError,
    build_model_from_checkpoint,
    check_compatible,
    load_checkpoint,
    save_checkpoint,
)
from pokerlab.rl.rollout import OpponentPool, TableBank, policy_opponent
from pokerlab.rl.table_mix import TableMix

SEATS_START = OBS_DIM_V5 - MAX_SEATS * SEAT_FEATURES_V5 - 40 - 11  # history (40) and mask (11) follow


def _play(model, hands: int, size: int = 4, seed: int = 0, keep=None):
    """`hands` hands with the model in every seat, at a table that remembers its players."""
    mix = TableMix()
    policy = make_policy_fn(model)
    if keep is not None:
        policy = _keeping(policy, keep)
    bank = TableBank(mix, random.Random(seed))
    bank.reset_stats(size)
    for proxy in bank.seats(size):
        proxy.inner = RLAgentPlayer(proxy.player_id, "m", policy_fn=policy, big_blind=mix.big_blind,
                                    starting_stack=mix.starting_stack)
    histories = []
    for _ in range(hands):
        bank.play_hand(size)
        histories.append(bank.last_hand)
    return bank, histories


def _keeping(policy, keep):
    reader = policy.beliefs

    def beliefs(observation, big_blind, oracle=None):
        found = reader(observation, big_blind, oracle)
        keep.append((observation, big_blind, found))
        return found

    def wrapped(*args):
        return policy(*args)

    wrapped.beliefs = beliefs
    wrapped.feature_version = policy.feature_version
    return wrapped


def test_a_model_of_version_5_reads_a_belief_about_every_live_opponent_and_none_about_itself():
    torch.manual_seed(0)
    model = tiny_belief_model(hidden=16, num_layers=1)
    kept: list = []
    _play(model, hands=20, keep=kept)
    assert kept
    for observation, _bb, beliefs in kept:
        live = {s.seat for s in observation.seats if s.status is not PlayerStatus.FOLDED}
        assert set(beliefs) == live - {observation.my_seat}
        for vector in beliefs.values():
            assert len(vector) == BELIEF_FEATURES
            assert 0.0 < vector[-1] <= 1.0  # a range of more than one hand, no wider than all of them


def test_the_belief_lands_in_each_seats_slots_and_slot_0_holds_none():
    torch.manual_seed(1)
    model = tiny_belief_model(hidden=16, num_layers=1)
    policy = make_policy_fn(model)
    kept: list = []
    _play(model, hands=10, keep=kept)
    observation, big_blind, beliefs = kept[-1]
    player = RLAgentPlayer("x", "x", policy_fn=policy, big_blind=big_blind, starting_stack=100 * big_blind)
    seats = list(observation.seats)
    me = next(i for i, s in enumerate(seats) if s.seat == observation.my_seat)
    rotated = seats[me:] + seats[:me]
    features = encode_observation(observation, big_blind=big_blind, starting_stack=100 * big_blind,
                                  legal_mask=[True] * 11, version=5,
                                  beliefs=player._beliefs(observation, big_blind, None))
    for slot, seat in enumerate(rotated):
        start = SEATS_START + slot * SEAT_FEATURES_V5 + SEAT_BASE_FEATURES_V3
        held = features[start : start + BELIEF_FEATURES]
        if seat.seat in beliefs:
            assert held == pytest.approx(beliefs[seat.seat], abs=1e-6)
        else:
            assert held == [0.0] * BELIEF_FEATURES


def test_running_the_tokens_as_they_come_is_running_the_whole_hand_each_time():
    torch.manual_seed(2)
    model = tiny_belief_model(hidden=16, num_layers=1)
    kept: list = []
    _play(model, hands=15, size=5, keep=kept)
    reader = model.belief_reader()
    net = model.belief_net.belief
    for observation, big_blind, _ in kept:
        asked = decision_question(observation, big_blind)
        if asked is None:
            continue
        with torch.inference_mode():
            summary = reader._summaries(observation, asked["targets"], asked["tokens"], asked["actors"])
            tokens = torch.tensor(asked["tokens"]).unsqueeze(0).expand(len(asked["targets"]), -1, -1)
            own = torch.tensor([[float(a == t) for a in asked["actors"]] for t in asked["targets"]])
            out, _ = net.gru(torch.relu(net.token_in(torch.cat([tokens, own.unsqueeze(-1)], dim=-1))))
        assert torch.allclose(summary, out[:, -1], atol=1e-5)


def test_the_memory_read_a_hand_at_a_time_is_the_memory_read_all_at_once():
    torch.manual_seed(3)
    model = tiny_belief_model(hidden=16, num_layers=1)
    _bank, histories = _play(model, hands=12, size=3)
    reader = model.belief_reader()
    ids = {seat: f"p{seat}" for seat in range(3)}
    hands = [memory_hand(h, ids) for h in histories]
    states: dict = {}
    reader.update_memory(hands, states)
    memory = model.belief_net.memory
    with torch.inference_mode():
        for seat in range(3):
            vectors = []
            for hand in hands:
                player = next(p for p in hand.players if p.seat == seat)
                tokens = torch.tensor(hand.tokens)
                own = torch.tensor([float(a == seat) for a in hand.actors]).unsqueeze(1)
                phi = (model.equity_encoder(card_planes(torch.tensor([list(player.cards)])),
                                            card_planes(torch.tensor([list(hand.board)])))
                       if player.shown else torch.zeros(1, 32))
                vectors.append(memory.hand_vectors(
                    torch.cat([tokens, own], 1).unsqueeze(0), torch.tensor([len(hand.tokens)]),
                    torch.tensor([float(player.shown)]), phi, torch.tensor([player.result_bb])))
            out, _ = memory.player_gru(torch.cat(vectors).unsqueeze(0))
            assert torch.allclose(states[f"p{seat}"], out[0, -1], atol=1e-5)


def test_with_the_whole_oracle_the_belief_is_the_true_hand():
    torch.manual_seed(4)
    model = tiny_belief_model(hidden=16, num_layers=1)
    kept: list = []
    bank, _ = _play(model, hands=6, keep=kept)
    reader = model.belief_reader()
    observation, big_blind, plain = kept[-1]
    holes = bank.hole_cards
    true = {seat: (card_index(a), card_index(b)) for seat, (a, b) in holes.items()}
    board = card_planes(torch.tensor([[card_index(c) for c in observation.community_cards] or [-1]]))
    full = reader(observation, big_blind, (1.0, true, "mix", (64,)))
    half = reader(observation, big_blind, (0.5, true, "mix", (64,)))
    for seat, vector in full.items():
        with torch.inference_mode():
            phi = model.equity_encoder(card_planes(torch.tensor([list(true[seat])])), board)[0].tolist()
        assert vector[:-1] == pytest.approx(phi, abs=1e-5) and vector[-1] == 0.0
        mixed = [0.5 * a + 0.5 * b for a, b in zip(plain[seat], vector, strict=True)]
        assert half[seat] == pytest.approx(mixed, abs=1e-5)


def test_swap_gives_an_opponent_its_true_hand_or_its_range_for_the_whole_hand_and_near_a_narrow_range():
    torch.manual_seed(4)
    model = tiny_belief_model(hidden=16, num_layers=1)
    kept: list = []
    bank, _ = _play(model, hands=6, keep=kept)
    observation, big_blind, plain = kept[-1]
    true = {seat: (card_index(a), card_index(b)) for seat, (a, b) in bank.hole_cards.items()}
    full = model.belief_reader()(observation, big_blind, (1.0, true, "mix", (64,)))
    reader = model.belief_reader()
    assert reader(observation, big_blind, (1.0, true, "swap", (64,))) == pytest.approx(full)
    assert model.belief_reader()(observation, big_blind, (1e-9, true, "swap", (64,))) == pytest.approx(plain)
    # a half share: each opponent is one or the other, and stays so for the rest of the hand
    random.seed(1)
    reader = model.belief_reader()
    first = reader(observation, big_blind, (0.5, true, "swap", (64,)))
    for seat, vector in first.items():
        assert vector == pytest.approx(full[seat]) or vector == pytest.approx(plain[seat])
    assert reader(observation, big_blind, (0.5, true, "swap", (64,))) == first
    near = model.belief_reader()(observation, big_blind, (1.0, true, "near", (64,)))
    for seat, vector in near.items():
        assert 0.0 < vector[-1] < plain[seat][-1]  # a range, narrower than the whole one
        gap_near = sum((a - b) ** 2 for a, b in zip(vector[:-1], full[seat][:-1], strict=True))
        gap_plain = sum((a - b) ** 2 for a, b in zip(plain[seat][:-1], full[seat][:-1], strict=True))
        assert gap_near < gap_plain  # and its mean closer to the true hand


def test_a_model_of_version_5_saves_and_comes_back_with_its_belief_network(tmp_path):
    torch.manual_seed(5)
    model = tiny_belief_model(hidden=16, num_layers=1)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model)
    rebuilt, checkpoint = build_model_from_checkpoint(path)
    assert checkpoint["belief"] == model.belief == TINY_BELIEF
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, rebuilt.state_dict()[name]), name
    other = tiny_belief_model(hidden=16, num_layers=1)
    other.belief = {**other.belief, "fingerprint": "another"}
    with pytest.raises(IncompatibleCheckpointError, match="belief"):
        load_checkpoint(path, other)
    broken = torch.load(path, weights_only=True)
    broken["belief"] = None
    with pytest.raises(IncompatibleCheckpointError, match="belief"):
        check_compatible(broken)


def test_the_learner_of_version_5_trains_on_the_oracles_share_and_its_beliefs_change_with_it():
    torch.manual_seed(6)
    model = tiny_belief_model(hidden=16, num_layers=1)
    mix = TableMix()
    collector = fake_collector(mix, make_policy_fn(model), rng=random.Random(0), table_hands=20)
    plain = collector.collect(60)
    collector.oracle_share = 1.0
    told = collector.collect(60)

    def widths(trajectories):
        found = []
        for trajectory in trajectories:
            for decision in trajectory.decisions:
                assert len(decision.features) == OBS_DIM_V5
                for slot in range(1, MAX_SEATS):
                    start = SEATS_START + slot * SEAT_FEATURES_V5 + SEAT_BASE_FEATURES_V3
                    if any(decision.features[start : start + BELIEF_FEATURES]):
                        found.append(decision.features[start + BELIEF_FEATURES - 1])
        return found

    assert widths(plain) and min(widths(plain)) > 0
    assert widths(told) and max(widths(told)) == 0.0  # every opponent read as its true hand


def test_a_version_5_opponent_sits_with_version_3_players_and_reads_the_tables_memory():
    torch.manual_seed(7)
    model = tiny_belief_model(hidden=16, num_layers=1)
    mix = TableMix()
    bank = TableBank(mix, random.Random(3))
    opponent = policy_opponent("v5", make_policy_fn(model), mix)
    bank.reset_stats(3)
    proxies = bank.seats(3)
    proxies[0].inner = opponent.factory(proxies[0].player_id, "v5", random.Random(0))
    for proxy in proxies[1:]:
        proxy.inner = make_random_legal_bot(proxy.player_id, rng=random.Random(1))
    for _ in range(30):
        bank.play_hand(3)
    tracker: MemoryTracker = bank._entry(3)[0].memory_tracker
    readers = tracker._readers
    assert set(readers) == {TINY_BELIEF["fingerprint"]}
    done, states = readers[TINY_BELIEF["fingerprint"]]
    assert done >= 29 and set(states) == {p.player_id for p in proxies}
    assert len(OpponentPool([opponent])) == 1  # an ordinary opponent of the pool
    assert math.isfinite(float(states[proxies[1].player_id].sum()))


def test_a_parent_must_read_the_same_version_and_the_same_belief_network(tmp_path):
    from pokerlab.rl.train import parent_shape_mismatch, same_network_filter

    model = tiny_belief_model(hidden=16, num_layers=1)
    path = tmp_path / "parent.pt"
    save_checkpoint(path, model)
    shape = {"hidden": 16, "num_layers": 1, "head_hidden": 256, "head_layers": 0}
    fingerprint = TINY_BELIEF["fingerprint"]
    assert same_network_filter(shape, model.equity, 5, fingerprint)(path)
    assert not same_network_filter(shape, model.equity, 5, "another")(path)
    assert not same_network_filter(shape, model.equity, 3, None)(path)
    assert parent_shape_mismatch(path, shape, model.equity, 5, model.belief) is None
    assert parent_shape_mismatch(path, shape, model.equity, 5, {**model.belief, "fingerprint": "x"})["belief"]
    assert parent_shape_mismatch(path, shape, model.equity, 3, None)["feature_version"] == 5
