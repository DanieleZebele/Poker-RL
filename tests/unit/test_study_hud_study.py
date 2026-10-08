"""`studies/agents/hud_study.py`: the profiles, the counterfactual states and the report
are torch-free; one test plays and evaluates a tiny network."""

from __future__ import annotations

import random

import pytest
from hud_study import (
    ARCHETYPES,
    REFERENCE,
    UNKNOWN,
    CollectJob,
    Comparison,
    Profile,
    action_group,
    context_of,
    format_report,
    population_from_members,
    profile_vector,
    profiles_for,
    raises_this_street,
    total_variation,
    with_hud,
)
from support import make_random_legal_bot

from pokerlab.engine.config import GameConfig
from pokerlab.engine.stats import STATS, WINDOW, HandCounts, StatsTracker
from pokerlab.engine.table import Table
from pokerlab.players.base import Player
from pokerlab.rl.action_space import ACTION_DIM, ALL_IN_BIN, CHECK_CALL_BIN, FOLD_BIN, RAISE_MIN_BIN


def member(vpip: float, wtsd: float) -> dict:
    hands = 10_000
    style = {name: [int(0.3 * hands), hands] for name in STATS}
    style["vpip"] = [int(vpip * hands), hands]
    style["wtsd"] = [int(wtsd * hands * 0.25), int(hands * 0.25)]
    return {"style": style, "style_hands": hands}


POPULATION = population_from_members([member(0.2, 0.7), member(0.4, 0.8), member(0.6, 0.9)])


def test_the_population_is_read_from_the_members_styles():
    assert POPULATION.members == 3
    assert POPULATION.median["vpip"] == pytest.approx(0.4)
    assert (POPULATION.low["vpip"], POPULATION.high["vpip"]) == pytest.approx((0.2, 0.6))
    assert POPULATION.tightest["vpip"] == pytest.approx(0.2)
    assert POPULATION.loosest["vpip"] == pytest.approx(0.6)
    assert POPULATION.per_hand["wtsd"] == pytest.approx(0.25)


def test_a_member_with_too_few_hands_does_not_stand_for_the_population():
    short = member(0.9, 0.9) | {"style_hands": 100}
    assert population_from_members([member(0.4, 0.8), short]).high["vpip"] == pytest.approx(0.4)


def test_archetypes_outside_the_measured_range_are_marked():
    profiles = {p.name: p for p in profiles_for(POPULATION)}
    assert profiles[UNKNOWN].rates is None
    assert profiles[REFERENCE].out_of_range == ()
    assert "vpip" in profiles["nit"].out_of_range  # 12% against a population of 20-60%
    assert set(profiles) >= set(ARCHETYPES)


def test_a_profile_is_the_hud_a_tracker_gives_a_player_seen_for_a_full_window():
    rates = ARCHETYPES["maniaco"]
    vector = profile_vector(rates, POPULATION.per_hand)
    tracker = StatsTracker()
    tracker.add("x", HandCounts((0,) * len(STATS), (0,) * len(STATS)))
    assert len(vector) == len(tracker.vector("x"))
    assert vector[0] == 1.0  # seen
    for index, name in enumerate(STATS):
        assert vector[2 + 2 * index] == pytest.approx(rates[name])  # the exact rate, not rounded
    assert vector[1] == pytest.approx(1.0)  # a full window of WINDOW hands
    assert WINDOW == 200


def played_states(count: int = 40) -> list:
    """Real (observation, legal actions) pairs from random bots with a tracker."""
    kept = []

    class Keeper(Player):
        def __init__(self, inner):
            super().__init__(inner.player_id, inner.name)
            self.inner = inner

        def act(self, observation, legal_actions):
            kept.append((observation, list(legal_actions)))
            return self.inner.act(observation, legal_actions)

    rng = random.Random(4)
    players = [Keeper(make_random_legal_bot(f"p{i}", rng=rng)) for i in range(4)]
    table = Table(GameConfig(num_players=4, starting_stack=200, small_blind=1, big_blind=2), players,
                  rng=random.Random(4), stats_tracker=StatsTracker())
    while len(kept) < count:
        table.stacks = [200] * 4
        table.play_hand()
    return kept[:count]


def test_only_the_opponents_huds_change_and_everything_else_stays():
    observation, _legal = played_states(30)[-1]
    vector = profile_vector(ARCHETYPES["nit"], POPULATION.per_hand)
    changed = with_hud(observation, vector)
    assert changed.my_seat not in changed.seat_stats
    assert set(changed.seat_stats) == {s.seat for s in observation.seats} - {observation.my_seat}
    assert all(v == vector for v in changed.seat_stats.values())
    assert (changed.hole_cards, changed.action_history, changed.pot_size) == (
        observation.hole_cards, observation.action_history, observation.pot_size)
    assert with_hud(observation, None).seat_stats == {}


def test_the_bins_are_grouped_by_size():
    assert action_group(FOLD_BIN) == "fold"
    assert action_group(CHECK_CALL_BIN) == "check/call"
    assert action_group(RAISE_MIN_BIN) == "raise piccolo"
    assert action_group(ALL_IN_BIN) == "all-in"
    assert action_group(ALL_IN_BIN - 1) == "raise grande"  # 200% of the pot
    assert total_variation([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_raises_are_counted_on_the_street_being_played_only():
    for observation, _legal in played_states(80):
        raises = raises_this_street(observation, 2)
        assert raises >= 0
        if not any(r.street == observation.street for r in observation.action_history):
            assert raises == 0
        context = context_of(observation, 2)
        assert (context.made is None) == (observation.street.value == "preflop")


def test_the_report_holds_every_section():
    states = played_states(60)
    contexts = [context_of(observation, 2) for observation, _legal in states]
    profiles = profiles_for(POPULATION)
    rng = random.Random(1)
    distributions = {}
    for profile in profiles:
        rows = []
        for _ in states:
            weights = [rng.random() for _ in range(ACTION_DIM)]
            rows.append([w / sum(weights) for w in weights])
        distributions[profile.name] = rows
    text = "\n".join(format_report(Comparison(contexts, distributions), profiles, POPULATION,
                                   label="bot", min_count=1))
    for title in ("PROFILI", "QUANTO CAMBIANO", "COMPORTAMENTI", "DISTRIBUZIONE"):
        assert title in text


def test_a_tiny_network_is_collected_and_evaluated_under_every_profile(tmp_path):
    pytest.importorskip("torch")
    from hud_study import collect_job, evaluate
    from support import tiny_model

    from pokerlab.rl.ppo import save_checkpoint

    path = tmp_path / "agent.pt"
    save_checkpoint(path, tiny_model())
    states = collect_job(CollectJob(seed=1, decisions=40, path=str(path), weights=(0, 1, 0, 0, 0, 0, 0, 0),
                                    stack_min_bb=20.0, stack_max_bb=50.0, small_blind=50, big_blind=100,
                                    session_hands=10))
    assert len(states) == 40
    profiles = [Profile(UNKNOWN, None), Profile(REFERENCE, POPULATION.median)]
    comparison = evaluate(path, states, profiles, POPULATION, big_blind=100, starting_stack=10_000)
    for rows in comparison.distributions.values():
        assert len(rows) == 40
        assert all(sum(row) == pytest.approx(1.0, abs=1e-5) for row in rows)
