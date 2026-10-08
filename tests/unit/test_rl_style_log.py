from __future__ import annotations

import pytest

from pokerlab.engine.stats import STATS
from pokerlab.rl.style_log import (
    PREFIX,
    SIZE_GROUPS,
    format_style_line,
    merge_style,
    parse_style_line,
    size_group,
)

RATES = {name: (i, 100 + i) for i, name in enumerate(STATS)}


def test_what_the_worker_prints_is_what_the_monitor_reads_back():
    line = format_style_line(4380, RATES)
    assert line.startswith(PREFIX + " ")
    parsed = parse_style_line(line)
    assert parsed.hands == 4380
    assert parsed.rates == RATES
    assert list(parsed.rates) == list(STATS)  # the order is kept


def test_the_raw_counts_are_printed_not_the_percentages():
    assert "vpip 0/100" in format_style_line(10, RATES)


def test_a_half_written_or_foreign_line_is_not_parsed():
    line = format_style_line(4380, RATES)
    assert parse_style_line(line[:-3]) is None  # a cell cut mid-number
    assert parse_style_line("stile (4380 mani): [vpip 12]") is None
    # Cut inside the last number: only the closing bracket tells it from a valid line.
    assert parse_style_line("stile (4380 mani): [vpip 1043/43") is None
    assert parse_style_line("valore per tavolo: 2 sd 0.270 ev -0.50 n 566  (spread -)") is None
    assert parse_style_line("iter    1  reward   +0.54 bb") is None
    assert parse_style_line("") is None


def test_merging_adds_counts_while_they_fit_the_window():
    merged, hands = merge_style({"vpip": [10, 40]}, 40, {"vpip": [5, 20], "pfr": [1, 20]}, 20)
    assert (merged, hands) == ({"vpip": [15, 60], "pfr": [1, 20]}, 60)


def test_merging_past_the_window_fades_old_and_new_together():
    merged, hands = merge_style({"vpip": [30, 100]}, 100, {"vpip": [30, 100]}, 100, window=100)
    assert (merged, hands) == ({"vpip": [30, 100]}, 100)


def test_evidence_larger_than_the_window_replaces_what_was_there():
    merged, hands = merge_style({"vpip": [0, 100]}, 100, {"vpip": [500, 1000]}, 1000, window=100)
    assert (merged, hands) == ({"vpip": [45, 100]}, 100)


def test_a_tally_counts_each_labelled_seat_over_the_hands_it_saw():
    import random

    from support import make_always_call_bot

    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.table import Table
    from pokerlab.rl.style_log import StyleTally

    config = GameConfig(num_players=3, starting_stack=2000, small_blind=50, big_blind=100)
    table = Table(
        config, [make_always_call_bot(f"p{i}") for i in range(3)], rng=random.Random(1)
    )
    tally = StyleTally(track=["a", "b"])
    for _ in range(4):
        for seat in range(3):
            table.stacks[seat] = 2000
        history = table.play_hand().hand_history
        tally.add_hand(history, {0: "a", 1: "b", 2: "c"})

    seen = tally.export()
    assert set(seen) == {"a", "b"}  # "c" was not tracked
    assert seen["a"]["hands"] == 4
    assert list(seen["a"]["style"]) == list(STATS)
    events, chances = seen["a"]["style"]["vpip"]
    assert chances == 4 and events <= chances


def test_a_group_of_table_sizes_has_a_line_of_its_own_that_round_trips():
    line = format_style_line(1910, RATES, group="4-6")
    assert line.startswith(f"{PREFIX} 4-6 (1910 mani): [")
    parsed = parse_style_line(line)
    assert parsed is not None and parsed.group == "4-6"
    assert (parsed.hands, parsed.rates) == (1910, dict(RATES))
    # the pooled line keeps its shape and reads as no group at all
    assert parse_style_line(format_style_line(4380, RATES)).group is None
    assert parse_style_line(line[:-2]) is None  # cut short, like any other line


def test_every_table_size_falls_in_exactly_one_group():
    assert [size_group(n) for n in range(2, 10)] == ["2-3"] * 2 + ["4-6"] * 3 + ["7-9"] * 3
    assert [f"{low}-{high}" for low, high in SIZE_GROUPS] == ["2-3", "4-6", "7-9"]
    with pytest.raises(ValueError):
        size_group(10)
