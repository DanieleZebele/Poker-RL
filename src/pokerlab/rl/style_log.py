"""How a model plays, as one line of its training log.

`SelfPlayCollector` pools every seat the learner sat in and counts the statistics
of `engine/stats.py` over its recent hands (`STYLE_WINDOW`): VPIP, PFR, 3-bet and
the rest, as *events over opportunities*. This module is the line `poker-train`
prints from them and the parser the dashboard reads it back with, in one file so
the two cannot drift apart (the same rule as `value_diagnostics` and `phases`):

    stile (4380 mani): [vpip 1043/4380 | pfr 702/4380 | three_bet 31/510 | ...]

The closing bracket is what makes a half-written line detectable: without it a line
cut inside the last number (`vpip 1043/43`) would still parse, as a valid and wrong
reading. The raw counts are printed, not the percentages: a reader can compute the rate,
and cannot recover the sample behind a rate. The line starts with `stile`, which
no parser of the `iter` lines can mistake for one.

Pure Python, no torch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

PREFIX = "stile"
_LINE = re.compile(r"^stile \((\d+) mani\): \[(.+)\]\s*$")
_CELL = re.compile(r"^(\w+) (\d+)/(\d+)$")


@dataclass(frozen=True)
class ParsedStyle:
    hands: int  # learner-seat hands behind the counts
    rates: dict[str, tuple[int, int]]  # name -> (events, opportunities)


def format_style_line(hands: int, rates: Mapping[str, tuple[int, int]]) -> str:
    cells = " | ".join(f"{name} {events}/{chances}" for name, (events, chances) in rates.items())
    return f"{PREFIX} ({hands} mani): [{cells}]"


def parse_style_line(line: str) -> ParsedStyle | None:
    """The inverse of `format_style_line`, or None for any other line.

    Anchored at both ends and every cell must parse, so a line the worker had only
    half written when the log was read is dropped rather than read as a wrong point.
    """
    matched = _LINE.match(line)
    if matched is None:
        return None
    rates: dict[str, tuple[int, int]] = {}
    for cell in matched.group(2).split(" | "):
        parts = _CELL.match(cell.strip())
        if parts is None:
            return None
        rates[parts.group(1)] = (int(parts.group(2)), int(parts.group(3)))
    return ParsedStyle(int(matched.group(1)), rates)
