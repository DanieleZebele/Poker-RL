"""The distribution of tables: sizes, stacks, and the flags that carry them."""

from __future__ import annotations

import argparse
import random
from collections import Counter

import pytest

from pokerlab.rl.table_mix import (
    DEFAULT_TABLE_WEIGHTS,
    SIZES,
    TableMix,
    add_table_arguments,
    sizes_text,
    table_arguments,
    table_mix_from_args,
)


def test_the_default_weights_are_the_ones_asked_for():
    assert DEFAULT_TABLE_WEIGHTS == (25, 20, 15, 10, 10, 10, 5, 5)
    assert SIZES == tuple(range(2, 10))


def test_sizes_are_drawn_in_proportion_to_their_weights():
    mix = TableMix()
    rng = random.Random(0)
    counts = Counter(mix.draw_size(rng) for _ in range(40_000))
    total = sum(DEFAULT_TABLE_WEIGHTS)
    for size, weight in zip(SIZES, DEFAULT_TABLE_WEIGHTS):
        assert counts[size] / 40_000 == pytest.approx(weight / total, abs=0.01)


def test_a_size_with_no_weight_is_never_drawn_and_max_players_ignores_it():
    mix = TableMix(weights=(1, 1, 0, 0, 1, 0, 0, 0))
    rng = random.Random(1)
    assert {mix.draw_size(rng) for _ in range(500)} == {2, 3, 6}
    assert mix.sizes == (2, 3, 6)
    assert mix.max_players == 6


def test_stacks_are_whole_chips_with_decimal_big_blinds_inside_the_range():
    mix = TableMix(stack_min_bb=1.0, stack_max_bb=100.0, small_blind=50, big_blind=100)
    rng = random.Random(2)
    stacks = [chips for _ in range(2000) for chips in mix.draw_stacks(rng, 6)]
    assert all(isinstance(chips, int) for chips in stacks)
    assert min(stacks) >= 100 and max(stacks) <= 10_000
    # Decimals really are used: a stack is not confined to whole big blinds.
    assert any(chips % 100 for chips in stacks)
    assert mix.starting_stack == 10_000


def test_each_seat_draws_its_own_stack():
    mix = TableMix()
    stacks = mix.draw_stacks(random.Random(3), 9)
    assert len(stacks) == 9 and len(set(stacks)) > 1


def test_a_fixed_stack_is_exactly_that_many_big_blinds():
    mix = TableMix(stack_min_bb=45.6, stack_max_bb=45.6, small_blind=50, big_blind=100)
    assert mix.draw_stacks(random.Random(4), 3) == [4560, 4560, 4560]


def test_the_config_of_a_table_carries_its_size_and_the_deepest_stack():
    config = TableMix().config(7)
    assert (config.num_players, config.starting_stack) == (7, 10_000)
    assert (config.small_blind, config.big_blind) == (50, 100)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": (1, 1, 1)},
        {"weights": (0,) * 8},
        {"weights": (-1, 1, 1, 1, 1, 1, 1, 1)},
        {"stack_min_bb": 0.0},
        {"stack_min_bb": 50.0, "stack_max_bb": 10.0},
        {"small_blind": 100, "big_blind": 100},
    ],
)
def test_a_mixture_that_makes_no_sense_is_refused(kwargs):
    with pytest.raises(ValueError):
        TableMix(**kwargs)


def test_the_flags_round_trip_through_the_command_line():
    """`poker-loop` hands the mixture to every worker and shard as flags."""
    parser = argparse.ArgumentParser()
    add_table_arguments(parser)
    original = TableMix(
        weights=(3, 1, 0, 2, 1, 1, 1, 1), stack_min_bb=2.5, stack_max_bb=80.0,
        small_blind=5, big_blind=10,
    )
    rebuilt = table_mix_from_args(parser.parse_args(table_arguments(original)))
    assert rebuilt == original


def test_the_defaults_of_the_flags_are_the_default_mixture():
    parser = argparse.ArgumentParser()
    add_table_arguments(parser)
    assert table_mix_from_args(parser.parse_args([])) == TableMix()


def test_the_log_line_shows_only_the_sizes_in_play():
    assert sizes_text((1, 1, 0, 0, 0, 0, 0, 2)) == "2:25% 3:25% 9:50%"
