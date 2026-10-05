from __future__ import annotations

import random

import pytest

from pokerlab.players.rl_agent import DecisionRecord, PolicyDecision
from pokerlab.rl.rollout import HandTrajectory, SelfPlayCollector
from pokerlab.rl.table_mix import TableMix
from pokerlab.rl.value_diagnostics import (
    MIN_DECISIONS,
    SIZE_KIND,
    STACK_KIND,
    STACK_LABELS,
    GroupStats,
    ValueDiagnostics,
    format_value_diagnostics,
    parse_value_line,
    stack_label,
    value_diagnostics,
)


def trajectory(
    values: list[float], returns: list[float], *, num_players: int = 6, effective_bb: float = 50.0
) -> HandTrajectory:
    decisions = [
        DecisionRecord(
            player_id="p0", seat=0, features=[], legal_mask=[], action_index=0, log_prob=0.0, value=v
        )
        for v in values
    ]
    t = HandTrajectory(
        seat=0,
        player_id="p0",
        decisions=decisions,
        reward=returns[-1],
        num_players=num_players,
        effective_stack_bb=effective_bb,
    )
    t.returns = list(returns)
    return t


def test_stack_labels_follow_the_edges():
    assert [stack_label(x) for x in (0.5, 9.99, 10.0, 29.99, 30.0, 100.0)] == [
        "<10", "<10", "10-30", "10-30", "30+", "30+",
    ]


def test_decisions_are_grouped_by_table_size_and_by_stack():
    trajectories = [
        trajectory([0.0, 0.0], [1.0, 1.0], num_players=2, effective_bb=5.0),
        trajectory([0.0], [3.0], num_players=2, effective_bb=60.0),
        trajectory([0.0], [2.0], num_players=9, effective_bb=60.0),
    ]
    d = value_diagnostics(trajectories)
    assert {size: g.decisions for size, g in d.by_size.items()} == {2: 3, 9: 1}
    assert {label: g.decisions for label, g in d.by_stack.items()} == {"<10": 2, "30+": 2}


def test_target_sd_is_the_population_sd_of_the_returns():
    d = value_diagnostics([trajectory([0.0] * 4, [1.0, 3.0, 1.0, 3.0])])
    assert d.by_size[6].target_sd == pytest.approx(1.0)


def test_a_perfect_critic_explains_everything_and_a_constant_one_nothing():
    returns = [1.0, 3.0, 1.0, 3.0]
    perfect = value_diagnostics([trajectory(returns, returns)]).by_size[6]
    constant = value_diagnostics([trajectory([2.0] * 4, returns)]).by_size[6]
    assert perfect.explained_variance == pytest.approx(1.0)
    assert constant.explained_variance == pytest.approx(0.0)


def test_a_critic_that_is_worse_than_the_mean_scores_negative():
    returns = [1.0, 3.0, 1.0, 3.0]
    wrong = value_diagnostics([trajectory([3.0, 1.0, 3.0, 1.0], returns)]).by_size[6]
    assert wrong.explained_variance == pytest.approx(-3.0)


def test_a_target_with_no_spread_has_no_explained_variance():
    group = value_diagnostics([trajectory([0.5, 0.5], [2.0, 2.0])]).by_size[6]
    assert group.target_sd == 0.0
    assert group.explained_variance is None


def test_spread_is_widest_over_narrowest_among_groups_large_enough_to_compare():
    big = MIN_DECISIONS
    groups = [GroupStats(big, 0.1, 0.0), GroupStats(big, 0.4, 0.0), GroupStats(big, 0.2, 0.0)]
    assert ValueDiagnostics.spread(groups) == pytest.approx(4.0)


def test_a_small_group_is_left_out_of_the_spread():
    groups = [GroupStats(MIN_DECISIONS, 0.2, 0.0), GroupStats(MIN_DECISIONS, 0.4, 0.0),
              GroupStats(MIN_DECISIONS - 1, 0.001, 0.0)]
    assert ValueDiagnostics.spread(groups) == pytest.approx(2.0)


def test_spread_needs_two_groups_and_a_nonzero_minimum():
    assert ValueDiagnostics.spread([GroupStats(MIN_DECISIONS, 0.3, 0.0)]) is None
    assert ValueDiagnostics.spread(
        [GroupStats(MIN_DECISIONS, 0.0, None), GroupStats(MIN_DECISIONS, 0.3, 0.0)]
    ) is None


def test_the_log_lines_start_with_valore_so_the_monitor_cannot_mistake_them_for_iter():
    d = value_diagnostics([trajectory([0.0, 0.0], [1.0, 2.0], num_players=3, effective_bb=5.0)])
    lines = format_value_diagnostics(d)
    assert len(lines) == 2
    assert all(line.startswith("valore ") for line in lines)
    assert "3 sd" in lines[0] and "<10 sd" in lines[1]


def test_the_collector_records_the_effective_stack_of_every_hand():
    """Effective stack = min(own stack, deepest opponent), in big blinds."""
    mix = TableMix(weights=(0, 0, 1, 0, 0, 0, 0, 0), stack_min_bb=1.0, stack_max_bb=100.0)
    rng = random.Random(3)

    def policy(features, mask):
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    trajectories = SelfPlayCollector(mix, policy, rng=random.Random(3)).collect(40)
    assert trajectories
    for t in trajectories:
        assert t.num_players == 4
        assert 0.0 < t.effective_stack_bb <= mix.stack_max_bb
    assert {stack_label(t.effective_stack_bb) for t in trajectories} <= set(STACK_LABELS)
    assert len({round(t.effective_stack_bb, 2) for t in trajectories}) > 5


def test_what_the_worker_prints_is_what_the_monitor_reads_back():
    """The format and its parser live in one module so they cannot drift; this is
    the check that they did not already."""
    d = value_diagnostics(
        [
            trajectory([0.0] * 150, [1.0, 3.0] * 75, num_players=2, effective_bb=5.0),
            trajectory([0.1] * 120, [2.0, 6.0] * 60, num_players=9, effective_bb=60.0),
            trajectory([0.0, 0.0], [2.0, 2.0], num_players=5, effective_bb=20.0),
        ]
    )
    size_line, stack_line = format_value_diagnostics(d)
    size = parse_value_line(size_line)
    stack = parse_value_line(stack_line)
    assert (size.kind, stack.kind) == (SIZE_KIND, STACK_KIND)
    assert set(size.groups) == {"2", "5", "9"}
    assert set(stack.groups) == {"<10", "10-30", "30+"}
    for name, group in d.by_size.items():
        read = size.groups[str(name)]
        assert read.decisions == group.decisions
        assert read.target_sd == pytest.approx(group.target_sd, abs=5e-4)
    # n/a (no spread in the target) survives the round trip as None.
    assert size.groups["5"].explained_variance is None
    assert size.spread == pytest.approx(d.size_spread, abs=0.05)
    assert stack.spread == pytest.approx(d.stack_spread, abs=0.05)


def test_a_half_written_or_foreign_line_is_not_parsed():
    line = "valore per tavolo: 2 sd 0.270 ev -0.50 n 566 | 3 sd 0.426 ev -0.16 n 497  (spread 3.2x)"
    assert parse_value_line(line) is not None
    assert parse_value_line(line[:60]) is None
    assert parse_value_line("iter    1  reward   +0.54 bb") is None
    assert parse_value_line("valore per qualcosa: 2 sd 0.2 ev +0.1 n 5  (spread -)") is None


def test_a_missing_spread_reads_as_none():
    line = "valore per stack (bb): <10 sd 0.083 ev -4.12 n 281  (spread -)"
    assert parse_value_line(line).spread is None
