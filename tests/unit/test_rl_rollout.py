from __future__ import annotations

import random

import pytest
from support import fixed_mix, make_always_call_bot

from pokerlab.players.rl_agent import DecisionRecord, PolicyDecision
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
        )
        for value in values
    ]
    decisions[-1].reward = reward
    return HandTrajectory(seat=0, player_id="p0", decisions=decisions, reward=reward)


def make_collector(num_players: int = 4, seed: int = 11) -> SelfPlayCollector:
    return SelfPlayCollector(
        fixed_mix(num_players, stack_bb=STARTING_STACK / BIG_BLIND),
        uniform_masked_policy(random.Random(seed)),
        rng=random.Random(seed),
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
    trajectories = make_collector().collect(20)
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
    collector = SelfPlayCollector(
        fixed_mix(4, stack_bb=STARTING_STACK / BIG_BLIND),
        uniform_masked_policy(random.Random(3)),
        rng=random.Random(3),
        reward_scale=0.01,
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
    collector = SelfPlayCollector(
        mix, uniform_masked_policy(random.Random(3)), rng=random.Random(3)
    )
    sizes = {t.num_players for t in collector.collect(120)}
    assert sizes == set(range(2, 10))


def test_a_table_size_with_no_weight_is_never_played():
    mix = TableMix(weights=(0, 0, 1, 0, 0, 0, 0, 1), stack_min_bb=5.0, stack_max_bb=100.0)
    collector = SelfPlayCollector(
        mix, uniform_masked_policy(random.Random(4)), rng=random.Random(4)
    )
    assert {t.num_players for t in collector.collect(80)} == {4, 9}


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
