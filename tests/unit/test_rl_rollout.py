from __future__ import annotations

import random

import pytest
from support import fake_collector, fixed_mix, make_always_call_bot

from pokerlab.engine.actions import Action, ActionType
from pokerlab.players.base import Player
from pokerlab.players.rl_agent import DecisionRecord, PolicyDecision
from pokerlab.rl.features import (
    CARDS_DIM,
    FIELD_SCALARS_DIM,
    POT_SCALARS_DIM,
    SEAT_BASE_FEATURES,
    STREET_DIM,
)
from pokerlab.rl.rollout import (
    STYLE_WINDOW,
    HandTrajectory,
    SelfPlayCollector,
    TableBank,
    compute_gae,
)
from pokerlab.rl.table_mix import TableMix

BIG_BLIND = 2
STARTING_STACK = 200


def uniform_masked_policy(rng: random.Random):
    def policy_fn(features: list[float], mask: list[bool]) -> PolicyDecision:
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    return policy_fn


def make_trajectory(values: list[float], reward: float) -> HandTrajectory:
    decisions = [
        DecisionRecord(
            player_id="p0",
            seat=0,
            features=[],
            legal_mask=[],
            action_index=0,
            log_prob=0.0,
            value=value,
            stake_bb=50.0,
        )
        for value in values
    ]
    decisions[-1].reward = reward
    return HandTrajectory(seat=0, player_id="p0", decisions=decisions, reward=reward)


def make_collector(num_players: int = 4, seed: int = 11, **kwargs) -> SelfPlayCollector:
    return fake_collector(
        fixed_mix(num_players, stack_bb=STARTING_STACK / BIG_BLIND),
        uniform_masked_policy(random.Random(seed)),
        rng=random.Random(seed),
        **kwargs,
    )


def test_undiscounted_gae_returns_the_terminal_reward_at_every_step():
    """With gamma = lam = 1 the return is the realised chip delta, whatever the
    critic said -- the sanity check that the recursion is wired correctly."""
    trajectory = make_trajectory([1.0, 2.0, 3.0], reward=5.0)
    advantages, returns = compute_gae(trajectory, gamma=1.0, lam=1.0)
    assert returns == pytest.approx([5.0, 5.0, 5.0])
    assert advantages == pytest.approx([4.0, 3.0, 2.0])


def test_gae_with_a_perfect_critic_gives_zero_advantage():
    trajectory = make_trajectory([7.0, 7.0, 7.0], reward=7.0)
    advantages, _ = compute_gae(trajectory, gamma=1.0, lam=1.0)
    assert advantages == pytest.approx([0.0, 0.0, 0.0])


def test_lambda_discounts_advantage_towards_the_start_of_the_hand():
    trajectory = make_trajectory([0.0, 0.0, 0.0], reward=5.0)
    advantages, _ = compute_gae(trajectory, gamma=1.0, lam=0.95)
    assert advantages == pytest.approx([4.5125, 4.75, 5.0])


def test_collect_produces_one_trajectory_per_seat_that_acted():
    # No all-in expectation, so the training reward is the chips that moved.
    trajectories = make_collector(allin_runouts=0).collect(20)
    assert trajectories
    for trajectory in trajectories:
        assert trajectory.decisions
        assert len(trajectory.advantages) == len(trajectory.decisions)
        assert len(trajectory.returns) == len(trajectory.decisions)
        # The reward is terminal: only the last decision of the hand carries it.
        assert [d.reward for d in trajectory.decisions[:-1]] == [0.0] * (
            len(trajectory.decisions) - 1
        )
        assert trajectory.decisions[-1].reward == trajectory.reward


def test_reward_scale_separates_the_training_signal_from_the_reported_big_blinds():
    """The critic predicts scaled rewards, but the trajectory keeps reporting
    big blinds -- otherwise value targets of +/-100 swamp the policy loss."""
    collector = fake_collector(
        fixed_mix(4, stack_bb=STARTING_STACK / BIG_BLIND),
        uniform_masked_policy(random.Random(3)),
        rng=random.Random(3),
        reward_scale=0.01,
        allin_runouts=0,
    )
    trajectories = [t for t in collector.collect(40) if t.reward != 0.0]

    assert trajectories, "no hand moved chips"
    for trajectory in trajectories:
        assert trajectory.decisions[-1].reward == pytest.approx(trajectory.reward * 0.01)
        assert abs(trajectory.decisions[-1].reward) < abs(trajectory.reward)


def test_every_hand_starts_from_freshly_drawn_stacks_so_a_session_never_runs_dry():
    """Stacks are redrawn after every hand, so a busted seat cannot shorten the
    session: the collector always plays as many hands as it was asked for."""
    collector = make_collector(num_players=3, seed=8)
    trajectories = collector.collect(200)
    assert len({t.reward for t in trajectories}) > 1
    # Every hand produced a learner decision: 200 hands, at least one trajectory each.
    assert len(trajectories) >= 200


def test_the_collector_visits_every_table_size_in_the_mixture():
    mix = TableMix(weights=(1, 1, 1, 1, 1, 1, 1, 1), stack_min_bb=5.0, stack_max_bb=100.0)
    collector = fake_collector(
        mix, uniform_masked_policy(random.Random(3)), rng=random.Random(3),
        table_hands=3, concurrent_tables=2,
    )
    sizes = {t.num_players for t in collector.collect(400)}
    assert sizes == set(range(2, 10))


def test_a_table_size_with_no_weight_is_never_played():
    mix = TableMix(weights=(0, 0, 1, 0, 0, 0, 0, 1), stack_min_bb=5.0, stack_max_bb=100.0)
    collector = fake_collector(
        mix, uniform_masked_policy(random.Random(4)), rng=random.Random(4),
        table_hands=2, concurrent_tables=2,
    )
    assert {t.num_players for t in collector.collect(160)} == {4, 9}


def test_the_bank_conserves_chips_whatever_the_stacks_are():
    """Chip conservation is the engine's single most valuable invariant; with
    uneven decimal-BB stacks it must still hold on every hand."""
    mix = TableMix(weights=(1, 1, 1, 1, 1, 1, 1, 1), stack_min_bb=1.0, stack_max_bb=100.0,
                   small_blind=50, big_blind=100)
    bank = TableBank(mix, random.Random(5))
    policy = uniform_masked_policy(random.Random(5))
    from pokerlab.players.rl_agent import RLAgentPlayer

    for num_players in range(2, 10):
        for seat, proxy in enumerate(bank.seats(num_players)):
            proxy.inner = RLAgentPlayer(
                f"s{seat}", f"S{seat}", policy_fn=policy,
                big_blind=mix.big_blind, starting_stack=mix.starting_stack,
            )
        for _ in range(10):
            before, after = bank.play_hand(num_players)
            assert sum(after) == sum(before)


def test_stacks_do_not_touch_the_card_sequence():
    """`Table` consumes its rng only to shuffle, so the stacks are drawn from a
    separate stream and a fixed seed deals the same cards whatever they are."""
    def first_hand_cards(stack_seed: int):
        mix = TableMix(weights=(0, 0, 1, 0, 0, 0, 0, 0), stack_min_bb=1.0, stack_max_bb=100.0)
        bank = TableBank(mix, random.Random(9), stack_rng=random.Random(stack_seed))
        from support import make_always_call_bot

        for seat, proxy in enumerate(bank.seats(4)):
            proxy.inner = make_always_call_bot(f"s{seat}")
        captured = {}
        table, _ = bank._entry(4)
        table._on_hand_started = lambda info: captured.update(info["hole_cards"])
        bank.play_hand(4)
        return captured

    assert first_hand_cards(1) == first_hand_cards(2)


def test_every_seat_gets_to_play_every_position_over_a_session():
    """The button rotates from hand to hand, so training data is not skewed to
    one seat's position."""
    trajectories = make_collector(num_players=4, seed=2).collect(40)
    assert {t.seat for t in trajectories} == {0, 1, 2, 3}


def test_the_collector_describes_how_the_learner_plays():
    """Every seat the learner sat in adds a hand to its style, so with no pool (all
    seats are the learner) there is a VPIP opportunity per seat per hand."""
    collector = make_collector(num_players=4)
    collector.collect(60)
    assert collector.style_hands >= 60
    events, chances = collector.style_rates["vpip"]
    assert chances == collector.style_hands and 0 <= events <= chances
    for name, (hits, opportunities) in collector.style_rates.items():
        assert 0 <= hits <= opportunities <= collector.style_hands, name


def test_the_style_window_forgets_old_hands():
    collector = make_collector(num_players=4)
    collector.collect(20)
    first = collector.style_hands
    assert 0 < first <= STYLE_WINDOW
    # A window of a few hands: it never holds more than that however many are played.
    from pokerlab.engine.stats import StatsTracker

    collector._style = StatsTracker(window=10)
    collector.collect(20)
    assert collector.style_hands == 10


def test_a_table_bank_remembers_the_hand_it_just_played():
    bank = TableBank(fixed_mix(3), random.Random(1))
    assert bank.last_hand is None
    proxies = bank.seats(3)
    for proxy in proxies:
        proxy.inner = make_always_call_bot(proxy.player_id)
    bank.play_hand(3)
    assert bank.last_hand is not None and bank.last_hand.num_players == 3


# ---- tables that last, so the statistics have something to describe ----------------

# Where the "statistics supplied" flag of the seat the decision is made from sits in
# the encoded observation (slot 0 is always me).
_SEATS_START = CARDS_DIM + STREET_DIM + POT_SCALARS_DIM + FIELD_SCALARS_DIM
_MY_STATS_FLAG = _SEATS_START + SEAT_BASE_FEATURES


def _supplied(trajectories) -> list[bool]:
    return [d.features[_MY_STATS_FLAG] == 1.0 for t in trajectories for d in t.decisions]


def test_a_table_keeps_its_players_and_its_statistics_for_table_hands_hands():
    collector = fake_collector(
        fixed_mix(4, stack_bb=100.0), uniform_masked_policy(random.Random(5)),
        rng=random.Random(5), table_hands=30, concurrent_tables=1,
    )
    first = collector.collect(1)
    later = collector.collect(29)
    assert not any(_supplied(first)), "nothing is known after no hands"
    assert any(_supplied(later)), "the statistics fill in as the table plays"
    # The table is replaced after its 30 hands, with blank statistics.
    assert not any(_supplied(collector.collect(1)))


def test_one_hand_per_table_is_the_old_behaviour_with_no_statistics_ever():
    collector = fake_collector(
        fixed_mix(4, stack_bb=100.0), uniform_masked_policy(random.Random(5)),
        rng=random.Random(5), table_hands=1, concurrent_tables=1,
    )
    assert not any(_supplied(collector.collect(40)))


def test_concurrent_tables_each_keep_their_own_statistics():
    collector = fake_collector(
        fixed_mix(4, stack_bb=100.0), uniform_masked_policy(random.Random(6)),
        rng=random.Random(6), table_hands=20, concurrent_tables=4,
    )
    first_round = collector.collect(4)  # the first hand of each of four tables
    assert not any(_supplied(first_round))
    assert any(_supplied(collector.collect(4)))  # each table is now on its second hand


def test_a_collector_needs_at_least_one_hand_and_one_table():
    for kwargs in ({"table_hands": 0}, {"concurrent_tables": 0}):
        with pytest.raises(ValueError):
            fake_collector(fixed_mix(4), uniform_masked_policy(random.Random(1)), **kwargs)


# ---- rated sessions: the same statistics, kept for one session ---------------------------


def _recording_policy(rng: random.Random, seen: list[float]):
    def policy_fn(features: list[float], mask: list[bool]) -> PolicyDecision:
        seen.append(features[_MY_STATS_FLAG])
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    return policy_fn


def _seated_bank(seen: list[float], seed: int = 3) -> TableBank:
    from pokerlab.players.rl_agent import RLAgentPlayer

    mix = fixed_mix(4, stack_bb=100.0)
    bank = TableBank(mix, random.Random(seed))
    for proxy in bank.seats(4):
        proxy.inner = RLAgentPlayer(
            proxy.player_id, proxy.name, policy_fn=_recording_policy(random.Random(seed), seen),
            big_blind=mix.big_blind, starting_stack=mix.starting_stack,
        )
    return bank


def test_a_session_gives_the_models_the_statistics_of_the_players_in_front_of_them():
    seen: list[float] = []
    bank = _seated_bank(seen)
    bank.play_session(4, 40)
    assert seen[0] == 0.0, "nothing is known before the first hand has been played"
    assert any(flag == 1.0 for flag in seen), "they fill in as the session goes"
    assert seen[-1] == 1.0


def test_every_session_starts_with_blank_statistics_whoever_sat_in_the_last_one():
    seen: list[float] = []
    bank = _seated_bank(seen)
    bank.play_session(4, 30)
    seen.clear()
    bank.play_session(4, 1)  # new players in the seats, nobody known yet
    assert seen and not any(seen)


def test_playing_hands_one_by_one_keeps_the_statistics_of_the_table_as_before():
    seen: list[float] = []
    bank = _seated_bank(seen)
    for _ in range(30):
        bank.play_hand(4)
    assert any(flag == 1.0 for flag in seen) is False  # no tracker: `play_hand` leaves it to the caller



def test_the_style_is_also_kept_per_group_of_table_sizes():
    mix = TableMix(
        weights=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),  # 2, 5 and 8 players
        stack_min_bb=50.0, stack_max_bb=50.0, small_blind=1, big_blind=2,
    )
    collector = fake_collector(
        mix, uniform_masked_policy(random.Random(4)), rng=random.Random(4),
        table_hands=1, concurrent_tables=1,
    )
    collector.collect(60)
    by_group = collector.style_by_group
    assert set(by_group) == {"2-3", "4-6", "7-9"}
    # every learner seat-hand is in the pooled style and in exactly one group
    assert sum(hands for hands, _rates in by_group.values()) == collector.style_hands
    for hands, rates in by_group.values():
        assert rates["vpip"][1] == hands


def test_a_group_never_played_has_no_style():
    collector = fake_collector(
        fixed_mix(3), uniform_masked_policy(random.Random(4)), rng=random.Random(4), table_hands=1,
    )
    collector.collect(10)
    assert set(collector.style_by_group) == {"2-3"}


def recording_policy(calls: list):
    """A policy that takes the opponent's style (a push and a temperature) and records it."""
    rng = random.Random(9)

    def policy_fn(features, mask, bias=None, temperature=1.0):
        calls.append((bias, temperature))
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    return policy_fn


def test_styled_opponents_get_a_push_and_the_learner_never_does():
    from pokerlab.rl.rollout import OpponentPool, policy_opponent
    from pokerlab.rl.styles import StyleConfig

    mix = fixed_mix(4, stack_bb=STARTING_STACK / BIG_BLIND)
    opponent_calls: list = []
    learner_calls: list = []

    def learner(features, mask):  # takes two arguments only: it would fail with a style
        learner_calls.append(1)
        return PolicyDecision(action_index=1 if mask[1] else 0)

    collector = fake_collector(
        mix, learner, rng=random.Random(2), opponent_probability=1.0, table_hands=5, concurrent_tables=1,
        opponent_pool=OpponentPool([policy_opponent("m", recording_policy(opponent_calls), mix)]),
        styles=StyleConfig(share=1.0, spread=0.5),
    )
    collector.collect(40)
    assert opponent_calls and learner_calls
    assert all(bias is not None and len(bias) == 11 for bias, _t in opponent_calls)
    assert collector.styled_seats == collector.opponent_seats > 0


def test_without_styles_an_opponent_is_the_plain_model_it_always_was():
    from pokerlab.rl.rollout import OpponentPool, policy_opponent

    mix = fixed_mix(3, stack_bb=STARTING_STACK / BIG_BLIND)
    calls: list = []
    collector = fake_collector(
        mix, uniform_masked_policy(random.Random(4)), rng=random.Random(4), opponent_probability=1.0,
        table_hands=5, concurrent_tables=1,
        opponent_pool=OpponentPool([policy_opponent("m", recording_policy(calls), mix)]),
    )
    collector.collect(30)
    assert calls and all(bias is None for bias, _t in calls)
    assert collector.styled_seats == 0 and collector.opponent_seats > 0


def test_a_table_keeps_the_style_it_was_opened_with():
    """The style is drawn when the table opens and lasts its hands, so the statistics the
    learner reads fill in over a player who stays the same."""
    from pokerlab.rl.rollout import OpponentPool, policy_opponent
    from pokerlab.rl.styles import StyleConfig

    mix = fixed_mix(2, stack_bb=STARTING_STACK / BIG_BLIND)
    calls: list = []
    collector = fake_collector(
        mix, uniform_masked_policy(random.Random(1)), rng=random.Random(1), opponent_probability=1.0,
        table_hands=40, concurrent_tables=1,
        opponent_pool=OpponentPool([policy_opponent("m", recording_policy(calls), mix)]),
        styles=StyleConfig(share=1.0, spread=0.6, jitter=0.0),
    )
    collector.collect(40)  # one table of 40 hands
    # With jitter 0 the push depends on the state only, so the temperature identifies the style.
    assert len({temperature for _bias, temperature in calls}) == 1
    assert collector.styled_seats == 1


def test_a_model_keeps_reading_in_big_blinds_when_the_blinds_go_up():
    """The same hand with every chip amount doubled -- stacks and blinds, as when the blinds
    double -- encodes to the same features: the player divides by the blind of the hand
    (`Observation.big_blind`), not by the one it was built with."""

    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.table import Table
    from pokerlab.players.rl_agent import RLAgentPlayer

    def first_features(small_blind: int, big_blind: int, stack: int):
        seen: list = []

        def policy_fn(features, mask):
            seen.append(list(features))
            return PolicyDecision(action_index=1 if mask[1] else 0)

        # Built for the session's first blinds (1/2 and 100 bb), whatever the table plays at.
        players = [
            RLAgentPlayer(f"p{i}", f"P{i}", policy_fn=policy_fn, big_blind=2, starting_stack=200)
            for i in range(3)
        ]
        config = GameConfig(num_players=3, starting_stack=stack, small_blind=small_blind, big_blind=big_blind)
        Table(config, players, rng=random.Random(7)).play_hand()
        return seen

    base = first_features(1, 2, 200)
    doubled = first_features(2, 4, 400)
    # Everything but the legal mask, which is the last block of the vector: a pot-fraction raise
    # is rounded to whole chips, so which sizes collide (and are masked) can differ with the
    # amounts -- a property of the chip grid, not of how the player normalises.
    from pokerlab.rl.features import MASK_DIM

    assert base and len(doubled) == len(base)
    for plain, scaled in zip(base, doubled):
        assert scaled[:-MASK_DIM] == pytest.approx(plain[:-MASK_DIM])


def test_a_player_reports_the_probabilities_of_the_decision_it_will_face():
    """`RLAgentPlayer.action_probabilities` is what a spectator is shown before the player
    acts: a probability for each bin, nothing on the illegal ones, and None for a policy that
    cannot give a distribution."""
    pytest.importorskip("torch")
    from support import make_always_call_bot, tiny_model

    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.table import Table
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.action_space import ACTION_DIM, legal_action_mask
    from pokerlab.rl.policy import make_policy_fn

    decisions = []

    class Recorder(Player):
        def act(self, observation, legal_actions):
            decisions.append((observation, legal_actions))
            return Action(ActionType.FOLD)

    players = [Recorder("p0", "P0"), make_always_call_bot("p1"), make_always_call_bot("p2")]
    Table(GameConfig(num_players=3, starting_stack=200, small_blind=1, big_blind=2), players,
          rng=random.Random(1)).play_hand()
    observation, legal_actions = decisions[0]

    with_model = RLAgentPlayer("m", "M", policy_fn=make_policy_fn(tiny_model()), big_blind=2, starting_stack=200)
    probabilities = with_model.action_probabilities(observation, legal_actions)
    mask = legal_action_mask(observation, legal_actions)
    assert len(probabilities) == ACTION_DIM and sum(probabilities) == pytest.approx(1.0, abs=1e-5)
    assert all(p == 0.0 for p, legal in zip(probabilities, mask) if not legal)

    without = RLAgentPlayer("u", "U", policy_fn=uniform_masked_policy(random.Random(0)), big_blind=2,
                            starting_stack=200)
    assert without.action_probabilities(observation, legal_actions) is None
