"""`studies/belief`: how a hand becomes what the belief network reads (tokens, samples, the
statistics every seat had), and how it is packed. Torch-free; the model's tests need torch."""

from __future__ import annotations

import random
from math import comb

import numpy as np
import pytest
from belief_data import concatenate, pack_sessions
from belief_records import (
    ACTION_CLASSES,
    COMBO_CARDS,
    NUM_COMBOS,
    POSITIONS,
    STREETS,
    TOKEN_DIM,
    blocked_combos,
    card_index,
    card_of,
    combo_index,
    hand_tokens,
    hud_before_hands,
    positions,
    samples_of,
)
from support import fixed_mix, make_random_legal_bot

from pokerlab.engine.actions import ActionType
from pokerlab.engine.history import _hand_history_from_dict, _to_jsonable
from pokerlab.rl.features import _card_index
from pokerlab.rl.rollout import TableBank


class _Recording:
    """A random bot that writes down the statistics it was shown, by hand number."""

    def __init__(self, player_id: str, seen: list, counter: list, rng: random.Random) -> None:
        self.inner = make_random_legal_bot(player_id, rng=rng)
        self.player_id, self.name = player_id, player_id
        self.seen, self.counter = seen, counter

    def act(self, observation, legal_actions):
        self.seen.append((self.counter[0], observation.my_seat, dict(observation.seat_stats)))
        return self.inner.act(observation, legal_actions)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _session(size: int, hands: int, seed: int, seen: list | None = None):
    bank = TableBank(fixed_mix(size, stack_bb=30.0), random.Random(seed))
    rng = random.Random(seed + 1)
    counter = [0]
    seen = seen if seen is not None else []
    for proxy in bank.seats(size):
        proxy.inner = _Recording(proxy.player_id, seen, counter, rng)
    played = []

    def finished(hand):
        played.append(hand)
        counter[0] += 1

    bank.play_session(size, hands, on_hand=finished)
    return played


def test_cards_are_indexed_as_everywhere_in_the_project():
    for index in range(52):
        card = card_of(index)
        assert card_index(card) == index == _card_index(card)


def test_every_pair_of_cards_is_one_combo_and_the_blocked_ones_are_counted():
    assert len(COMBO_CARDS) == NUM_COMBOS == comb(52, 2)
    assert {combo_index(a, b) for a in range(52) for b in range(52) if a != b} == set(range(NUM_COMBOS))
    assert combo_index(7, 3) == combo_index(3, 7)
    assert len(blocked_combos([5])) == 51
    assert len(blocked_combos(range(7))) == NUM_COMBOS - comb(45, 2)


@pytest.mark.parametrize("size", [2, 3, 6, 9])
def test_a_token_per_record_with_one_street_one_position_and_one_action(size):
    for hand in _session(size, 30, seed=size):
        read = hand_tokens(hand)
        assert len(read.tokens) == len(hand.actions)
        for token, record, actor in zip(read.tokens, hand.actions, read.actors, strict=True):
            assert len(token) == TOKEN_DIM and actor == record.seat
            street = token[: len(STREETS)]
            position = token[len(STREETS) : len(STREETS) + len(POSITIONS)]
            action = token[len(STREETS) + len(POSITIONS) : len(STREETS) + len(POSITIONS) + len(ACTION_CLASSES)]
            assert sum(street) == sum(position) == sum(action) == 1
            assert all(0.0 <= value <= 1.0 for value in token)
            label = ACTION_CLASSES[action.index(1.0)]
            expected = {
                ActionType.POST_BLIND: "blind", ActionType.FOLD: "fold", ActionType.CHECK: "check",
                ActionType.CALL: "call", ActionType.BET: "raise", ActionType.RAISE: "raise",
            }.get(record.action_type)
            if expected is None:
                assert label in ("call", "shove")
            else:
                assert label == expected


def test_a_sample_asks_about_another_player_still_in_the_hand_before_a_choice():
    for hand in _session(6, 40, seed=3):
        read = hand_tokens(hand)
        folded: set[int] = set()
        live_at = []
        for record in hand.actions:
            live_at.append(set(hand.starting_stacks) - folded)
            if record.action_type == ActionType.FOLD:
                folded.add(record.seat)
        samples = samples_of(read)
        voluntary = sum(read.voluntary)
        assert len(samples) == sum(len(live_at[t]) - 1 for t in range(len(read.tokens)) if read.voluntary[t])
        assert voluntary == sum(r.action_type != ActionType.POST_BLIND for r in hand.actions)
        for sample in samples:
            assert sample.target != sample.observer
            assert sample.observer == hand.actions[sample.prefix].seat
            assert sample.target in live_at[sample.prefix]
            assert sample.live == len(live_at[sample.prefix])


def test_the_statistics_replayed_are_the_ones_the_players_were_shown():
    seen: list = []
    hands = _session(6, 60, seed=11, seen=seen)
    huds = hud_before_hands(hands)
    assert not huds[0] and len(huds[-1]) == 6
    checked = 0
    for number, seat, stats in seen:
        mine = {other: vector for other, vector in huds[number].items() if other != seat}
        shown = {other: tuple(vector) for other, vector in stats.items() if other != seat}
        assert shown == {other: tuple(v) for other, v in mine.items()}
        checked += 1
    assert checked > 100


def test_packing_keeps_each_sample_on_its_hand_and_concatenating_shifts_the_indices():
    sessions = []
    for number, size in enumerate((3, 7)):
        hands = _session(size, 15, seed=20 + number)
        sessions.append({
            "session": number, "size": size, "models": [f"m{number}-{s}" for s in range(size)],
            "styled": [s == 0 for s in range(size)], "hands": [_to_jsonable(h) for h in hands],
        })
    one = pack_sessions(sessions[:1], hud=True)
    two = pack_sessions(sessions[1:], hud=True)
    both = concatenate([one, two])
    whole = pack_sessions(sessions, hud=True)
    for key in ("tokens", "token_offset", "board", "holes", "hud", "s_hand", "s_prefix", "s_target", "player",
                "seq_pos", "s_mode", "s_shows", "source"):
        np.testing.assert_array_equal(both[key], whole[key])
    assert both["models"].tolist() == whole["models"].tolist()
    np.testing.assert_array_equal(both["models"][both["seat_model"][both["seat_model"] >= 0]],
                                  whole["models"][whole["seat_model"][whole["seat_model"] >= 0]])
    offsets = whole["token_offset"]
    assert offsets[0] == 0 and offsets[-1] == len(whole["tokens"])
    lengths = offsets[1:] - offsets[:-1]
    assert (whole["s_prefix"] < lengths[whole["s_hand"]]).all()
    target_cards = whole["holes"][whole["s_hand"], whole["s_target"]]
    assert (target_cards >= 0).all()


# ---- the network (torch) ----------------------------------------------------


def _tiny_data(tmp_path):
    torch = pytest.importorskip("torch")
    from belief_train import Data

    sessions = []
    for number, size in enumerate((2, 5)):
        hands = _session(size, 12, seed=40 + number)
        sessions.append({
            "session": number, "size": size, "models": [f"m{number}-{s}" for s in range(size)],
            "styled": [False] * size, "hands": [_to_jsonable(h) for h in hands],
        })
    return torch, Data(pack_sessions(sessions), "cpu")


def _tiny_net(torch, **options):
    from belief_model import BeliefNet

    from pokerlab.rl.equity_net import EquityEncoder

    torch.manual_seed(0)
    return BeliefNet(EquityEncoder(hidden=8, latent=4, blocks=1), hidden=16, gru=8, rank=4, **options)


def test_the_range_rules_out_what_the_observer_sees_and_never_the_true_hand(tmp_path):
    torch, data = _tiny_data(tmp_path)
    net = _tiny_net(torch)
    every = torch.arange(len(data.s_hand))
    for augment in (False, True):
        batch = data.batch(every, augment=augment)
        logits, _phi, _inverse = net(batch["tokens"], batch["lengths"], batch["static"], batch["board"], batch["known"])
        assert logits.shape == (len(every), NUM_COMBOS)
        true = logits.gather(1, batch["target"].unsqueeze(1)).squeeze(1)
        assert (true > -1e8).all()
        known = batch["known"]
        for row in range(0, len(every), 37):
            cards = [c for c in known[row].tolist() if c >= 0]
            blocked = blocked_combos(cards)
            assert (logits[row, blocked] < -1e8).all()
            assert int((logits[row] > -1e8).sum()) == comb(52 - len(cards), 2)


def test_a_range_on_one_hand_is_that_hand_in_the_encoders_space(tmp_path):
    torch, data = _tiny_data(tmp_path)
    from belief_model import range_embedding

    net = _tiny_net(torch)
    batch = data.batch(torch.arange(len(data.s_hand))[:50], augment=False)
    _logits, phi, inverse = net(batch["tokens"], batch["lengths"], batch["static"], batch["board"], batch["known"])
    oracle = torch.full((50, NUM_COMBOS), -1e9)
    oracle[torch.arange(50), batch["target"]] = 0.0
    mean, spread = range_embedding(oracle, phi, inverse)
    torch.testing.assert_close(mean, phi[inverse, batch["target"]])
    assert float(spread.max()) < 1e-3


def test_without_tokens_the_actions_of_the_hand_change_nothing(tmp_path):
    torch, data = _tiny_data(tmp_path)
    net = _tiny_net(torch, use_tokens=False)
    batch = data.batch(torch.arange(len(data.s_hand))[:40], augment=False)
    args = (batch["lengths"], batch["static"], batch["board"], batch["known"])
    first, _, _ = net(batch["tokens"], *args)
    second, _, _ = net(torch.rand_like(batch["tokens"]), *args)
    torch.testing.assert_close(first, second)


def test_a_seat_shows_its_cards_only_when_still_in_at_a_showdown_and_its_result_is_its_stack():
    hands = _session(4, 40, seed=50)
    packed = pack_sessions([{
        "session": 0, "size": 4, "models": [f"m{s}" for s in range(4)], "styled": [False] * 4,
        "hands": [_to_jsonable(h) for h in hands],
    }])
    assert packed["hand_number"].tolist() == list(range(40))
    for row, hand in enumerate(hands):
        folded = {r.seat for r in hand.actions if r.action_type == ActionType.FOLD}
        still_in = set(hand.starting_stacks) - folded
        for seat in range(4):
            assert packed["shown"][row, seat] == (len(still_in) >= 2 and seat in still_in)
            expected = (hand.final_stacks[seat] - hand.starting_stacks[seat]) / hand.big_blind
            assert packed["result_bb"][row, seat] == pytest.approx(expected, abs=0.05)
        assert not packed["shown"][row, 4:].any()


def test_the_memory_used_in_a_hand_knows_nothing_of_that_hand_or_the_ones_after(tmp_path):
    torch = pytest.importorskip("torch")
    import numpy as np
    from belief_history import Model, SessionData

    from pokerlab.rl.equity_net import EquityEncoder

    sessions = [{
        "session": 1, "size": 3, "models": ["a", "b", "c"], "styled": [False] * 3,
        "hands": [_to_jsonable(h) for h in _session(3, 20, seed=60)],
    }]
    data = SessionData(pack_sessions(sessions), "cpu")
    torch.manual_seed(0)
    model = Model(EquityEncoder(hidden=8, latent=4, blocks=1), "memory", hidden=16, gru=8, rank=4, state=8).eval()
    chosen = np.array([0])
    index, local = data.samples(chosen)
    with torch.no_grad():
        before, _ = model(data, chosen, index, local, augment=False)
        changed = 10
        data.result_bb[changed] += 50.0  # what the table saw happen in hand 10, and nothing else
        after, _ = model(data, chosen, index, local, augment=False)
    hand_of = data.hand_number[data.s_hand[index]]
    same = (before - after).abs().amax(dim=1) < 1e-5
    assert same[hand_of <= changed].all()
    assert not same[hand_of > changed].any()


# ---- real hands in the PHH format (phh_convert.py) --------------------------------------------

PHH_HEADS_UP = """
[1]
variant = 'NT'
antes = [0, 0]
blinds_or_straddles = [1, 2]
starting_stacks = [200, 300]
actions = ['d dh p1 ????', 'd dh p2 ????', 'p2 cbr 6', 'p1 cc', 'd db 7cKh7s', 'p1 cc', 'p2 cbr 8', 'p1 f']
players = ['bb-player', 'button-player']
table = 'T1'
time = 10:00:00
"""

PHH_SHOWDOWN = """
variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [1, 2, 0]
starting_stacks = [100, 40, 100]
actions = ['d dh p1 ????', 'd dh p2 ????', 'd dh p3 ????', 'p3 cbr 10', 'p1 cc', 'p2 cbr 40', 'p3 cc', 'p1 cc',
           'd db 2c3d4h', 'p1 cbr 20', 'p3 cc', 'd db 9s', 'p1 cc', 'p3 cc', 'd db Td',
           'p1 cc', 'p3 cc', 'p1 sm AhAd', 'p3 sm ????', 'p2 sm KsKd']
players = ['a', 'b', 'c']
"""


def test_a_heads_up_phh_hand_has_its_blinds_reversed_and_its_button_second():
    from phh_convert import convert_hand, parse_file

    hand = convert_hand(parse_file(PHH_HEADS_UP)[0], hand_id="x", full_information=False)
    assert hand["button_seat"] == 1 and hand["big_blind"] == 100
    posts = [(r["seat"], r["amount"]) for r in hand["actions"] if r["action_type"] == "post_blind"]
    assert posts == [(1, 50), (0, 100)]  # the button posts the small blind
    kinds = [(r["seat"], r["action_type"], r["amount"]) for r in hand["actions"][2:]]
    assert kinds == [(1, "raise", 300), (0, "call", 300), (0, "check", 0), (1, "bet", 400), (0, "fold", 0)]
    assert hand["hole_cards"] == {} and hand["final_stacks"] == {"0": 10000 - 300, "1": 15000 + 300}
    hh = _hand_history_from_dict(hand)
    assert positions(hh) == {1: "BTN", 0: "BB"}
    assert len(hand_tokens(hh).tokens) == len(hand["actions"])


def test_a_showdown_without_winnings_is_settled_with_its_side_pot():
    from phh_convert import convert_hand, parse_file

    hand = convert_hand(parse_file(PHH_SHOWDOWN)[0], hand_id="x", full_information=False)
    # stacks of 100/40/100 at a big blind of 2: 5000/2000/5000 chips. p2 (seat 1) all-in for 2000:
    # the main pot, 6000, to the aces (seat 0) over the kings; the side pot (1000 each on the flop
    # from seat 0 and seat 2) to the aces too, seat 2 mucked
    assert hand["final_stacks"] == {"0": 5000 - 3000 + 6000 + 2000, "1": 0, "2": 5000 - 3000}
    assert sum(hand["final_stacks"].values()) == sum(hand["starting_stacks"].values())
    assert hand["hole_cards"] == {"0": ["Ah", "Ad"], "1": ["Ks", "Kd"]}
    assert hand["known_by"] == {"0": "shown", "1": "shown"}
    all_in = [r for r in hand["actions"] if r["seat"] == 1 and r["action_type"] == "all_in"]
    assert all_in and all_in[0]["stack_after"] == 0


def test_unknown_stacks_are_a_hundred_big_blinds_or_what_was_put_in():
    from phh_convert import UNKNOWN_STACK, convert_hand, parse_file

    raw = parse_file(PHH_SHOWDOWN)[0] | {"starting_stacks": [float("inf")] * 3}
    hand = convert_hand(raw, hand_id="x", full_information=False)
    assert all(v >= UNKNOWN_STACK for v in hand["starting_stacks"].values())


def test_what_is_not_no_limit_holdem_of_two_to_nine_is_left_out():
    from phh_convert import sessions_of_file

    base = parse_file_one = __import__("phh_convert").parse_file(PHH_SHOWDOWN)[0]
    del parse_file_one
    skipped: dict[str, int] = {}
    hands = [base | {"variant": "FT"}, base | {"antes": [1, 1, 1]}, base | {"blinds_or_straddles": [1, 2, 4]}, base]
    sessions = sessions_of_file(hands, source="test", full_information=False, skipped=skipped)
    assert sum(len(s["hands"]) for s in sessions) == 1
    assert skipped == {"variant": 1, "antes": 1, "blinds": 1}


def test_real_sessions_pack_with_players_by_name_and_samples_only_where_cards_are_known():
    from phh_convert import parse_file, sessions_of_file

    hands = [parse_file(PHH_SHOWDOWN)[0] | {"table": "T", "hand": i} for i in range(3)]
    sessions = sessions_of_file(hands, source="handhq-test", full_information=False, skipped={})
    for i, session in enumerate(sessions):
        session["session"] = 10_000_000 + i
    arrays = pack_sessions(sessions)
    assert arrays["sources"].tolist() == ["handhq-test"]
    assert set(arrays["s_target"].tolist()) <= {0, 1}  # the two whose cards were shown
    assert set(arrays["s_mode"].tolist()) == {1}
    assert arrays["seq_pos"][:, 0].tolist() == [0, 1, 2]  # the same player's first, second, third hand
    assert (arrays["player"][:, :3] == arrays["player"][0, :3]).all()
