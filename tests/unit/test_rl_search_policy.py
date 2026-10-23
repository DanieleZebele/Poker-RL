"""`make_search_policy_fn`: the policy a search plays its rollouts with must be the model's own,
only cheaper to ask (`rl/policy.py`)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from support import make_random_legal_bot, tiny_model

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.base import Player
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.policy import make_policy_fn, make_search_policy_fn


class Recorder(Player):
    def __init__(self, player_id, rng, sink):
        super().__init__(player_id, player_id)
        self._inner = make_random_legal_bot(player_id, player_id, rng)
        self._sink = sink

    def act(self, observation, legal_actions):
        self._sink.append((observation, legal_actions))
        return self._inner.act(observation, legal_actions)


def decisions(players: int = 4, hands: int = 15, seed: int = 3):
    rng = random.Random(seed)
    sink: list = []
    table = Table(GameConfig(num_players=players, starting_stack=200, small_blind=1, big_blind=2),
                  [Recorder(f"p{i}", rng, sink) for i in range(players)], rng=rng)
    for _ in range(hands):
        table.stacks = [rng.randint(10, 600) for _ in range(players)]
        table.play_hand()
    return sink


@pytest.mark.parametrize("version", [3, 4])
def test_the_search_policy_gives_the_model_s_own_probabilities(version):
    torch.manual_seed(0)
    model = tiny_model(hidden=32, num_layers=2, feature_version=version)
    exact, fast = make_policy_fn(model), make_search_policy_fn(model)
    player = RLAgentPlayer("p", "p", policy_fn=exact, big_blind=2, starting_stack=200)
    sample = decisions()
    assert sample
    for _round in range(2):  # the second time the card encoder's output comes from the cache
        for observation, legal in sample:
            features, mask = player.features_for(observation, legal)
            assert fast.distribution(features, mask) == pytest.approx(exact.distribution(features, mask), abs=1e-6)
    assert fast.feature_version == version


def test_with_a_style_the_push_and_the_temperature_are_applied_the_same_way():
    torch.manual_seed(1)
    model = tiny_model(hidden=32, num_layers=1)
    exact, fast = make_policy_fn(model), make_search_policy_fn(model)
    player = RLAgentPlayer("p", "p", policy_fn=exact, big_blind=2, starting_stack=200)
    bias = [0.5, -1.0, 2.0, 0.0, 0.3, -0.7, 1.1, 0.0, -2.0, 0.4, 0.9]
    for observation, legal in decisions(players=3, hands=6)[:30]:
        features, mask = player.features_for(observation, legal)
        assert fast.distribution(features, mask, bias, 1.7) == pytest.approx(
            exact.distribution(features, mask, bias, 1.7), abs=1e-6
        )


def test_a_tiny_cache_that_keeps_starting_afresh_changes_nothing():
    torch.manual_seed(2)
    model = tiny_model(hidden=32, num_layers=1)
    exact, fast = make_policy_fn(model), make_search_policy_fn(model, cache_size=1)
    player = RLAgentPlayer("p", "p", policy_fn=exact, big_blind=2, starting_stack=200)
    for observation, legal in decisions(players=5, hands=5)[:40]:
        features, mask = player.features_for(observation, legal)
        assert fast.distribution(features, mask) == pytest.approx(exact.distribution(features, mask), abs=1e-6)


def test_its_draws_follow_its_probabilities_and_never_a_masked_bin():
    torch.manual_seed(3)
    model = tiny_model(hidden=32, num_layers=1)
    fast = make_search_policy_fn(model)
    player = RLAgentPlayer("p", "p", policy_fn=fast, big_blind=2, starting_stack=200)
    observation, legal = decisions(players=2, hands=3)[0]
    features, mask = player.features_for(observation, legal)
    probabilities = fast.distribution(features, mask)
    draws = Counter(fast(features, mask).action_index for _ in range(6000))
    assert all(mask[index] for index in draws)
    for index, probability in enumerate(probabilities):
        assert draws[index] / 6000 == pytest.approx(probability, abs=0.03)


def test_many_decisions_in_one_pass_give_each_its_own_probabilities():
    torch.manual_seed(4)
    model = tiny_model(hidden=32, num_layers=2)
    exact, fast = make_policy_fn(model), make_search_policy_fn(model, cache_size=50)
    player = RLAgentPlayer("p", "p", policy_fn=exact, big_blind=2, starting_stack=200)
    rows = [player.features_for(observation, legal) for observation, legal in decisions(players=6, hands=10)[:120]]
    batch = fast.batch_distribution([f for f, _m in rows], [m for _f, m in rows])
    for (features, mask), probabilities in zip(rows, batch, strict=True):
        assert probabilities == pytest.approx(exact.distribution(features, mask), abs=1e-5)


def test_the_leaf_value_is_the_critic_s_estimate_in_big_blinds():
    from support import TINY_EQUITY

    from pokerlab.cards.card import Card, Rank, Suit
    from pokerlab.players.rl_agent import deal_view
    from pokerlab.rl.equity_net import EquityNet
    from pokerlab.rl.policy import make_critic_fns, make_leaf_value

    torch.manual_seed(5)
    model = tiny_model(hidden=32, num_layers=1)
    net = EquityNet(**TINY_EQUITY, arch="sets")
    net.phi.load_state_dict(model.equity_encoder.phi.state_dict())
    net = net.requires_grad_(False).eval()
    leaf = make_leaf_value(model, net, value_unit=100.0)
    critic = make_critic_fns(model, net)
    player = RLAgentPlayer("p", "p", policy_fn=make_policy_fn(model), big_blind=2, starting_stack=200)

    deck = [Card(rank, suit) for suit in Suit for rank in Rank]
    for observation, legal in decisions(players=3, hands=4)[:10]:
        seen = {*observation.hole_cards, *observation.community_cards}
        spare = iter(card for card in deck if card not in seen)
        holes = {
            seat.seat: observation.hole_cards if seat.seat == observation.my_seat else (next(spare), next(spare))
            for seat in observation.seats
        }
        features, _mask = player.features_for(observation, legal)
        extra = critic.equity([deal_view(observation, holes)])[0]
        assert leaf(features, observation, holes) == pytest.approx(critic.value([features], [extra])[0] * 100.0)
