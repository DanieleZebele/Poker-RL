from __future__ import annotations

from pokerlab.engine.stats import STATS
from pokerlab.rl.style_log import PREFIX, format_style_line, parse_style_line

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
