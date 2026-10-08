"""How a model plays, as one line of its training log.

`SelfPlayCollector` pools every seat the learner sat in and counts the statistics
of `engine/stats.py` over its recent hands (`STYLE_WINDOW`): VPIP, PFR, 3-bet and
the rest, as *events over opportunities*. This module is the line `poker-train`
prints from them and the parser the dashboard reads it back with, in one file so
the two cannot drift apart (the same rule as `value_diagnostics` and `phases`):

    stile (4380 mani): [vpip 1043/4380 | pfr 702/4380 | three_bet 31/510 | ...]
    stile 4-6 (1910 mani): [vpip 488/1910 | ...]

The first line pools every table; the others, one per group of table sizes
(`SIZE_GROUPS`: 2-3, 4-6 and 7-9 players), count only the hands played at that size,
because a model that plays well plays differently heads-up and nine-handed, and a
pooled VPIP is mostly a statement about the mixture of table sizes.

The closing bracket is what makes a half-written line detectable: without it a line
cut inside the last number (`vpip 1043/43`) would still parse, as a valid and wrong
reading. The raw counts are printed, not the percentages: a reader can compute the rate,
and cannot recover the sample behind a rate. The line starts with `stile`, which
no parser of the `iter` lines can mistake for one.

Pure Python, no torch.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from pokerlab.engine.history import HandHistory
from pokerlab.engine.stats import STAT_COUNT, STATS, analyse_hand

# How many seat-hands a model's style is kept over, wherever it is measured: a
# training run's own window, and what a ranking member accumulates across the passes
# that seat it (`merge_style`). ~2 seat-hands per hand at the default mixture, so
# about nine thousand hands -- long enough that the rare statistics (3-bet, fold to
# c-bet) have a few hundred opportunities, short enough to follow a policy that changes.
STYLE_WINDOW = 20_000

# The groups of table sizes the style is also reported for, each its own line.
SIZE_GROUPS = ((2, 3), (4, 6), (7, 9))

PREFIX = "stile"
_LINE = re.compile(r"^stile(?: (\d+-\d+))? \((\d+) mani\): \[(.+)\]\s*$")
_CELL = re.compile(r"^(\w+) (\d+)/(\d+)$")


def size_group(num_players: int) -> str:
    """The label of the group of table sizes `num_players` falls in, e.g. "4-6"."""
    for low, high in SIZE_GROUPS:
        if low <= num_players <= high:
            return f"{low}-{high}"
    raise ValueError(f"no group of table sizes holds {num_players} players")


@dataclass(frozen=True)
class ParsedStyle:
    hands: int  # learner-seat hands behind the counts
    rates: dict[str, tuple[int, int]]  # name -> (events, opportunities)
    group: str | None = None  # the group of table sizes ("4-6"), None for every table pooled


def format_style_line(
    hands: int, rates: Mapping[str, tuple[int, int]], group: str | None = None
) -> str:
    """The style line, for every table pooled or, given `group`, for one group of sizes."""
    cells = " | ".join(f"{name} {events}/{chances}" for name, (events, chances) in rates.items())
    label = PREFIX if group is None else f"{PREFIX} {group}"
    return f"{label} ({hands} mani): [{cells}]"


def parse_style_line(line: str) -> ParsedStyle | None:
    """The inverse of `format_style_line`, or None for any other line.

    Anchored at both ends and every cell must parse, so a line the worker had only
    half written when the log was read is dropped rather than read as a wrong point.
    """
    matched = _LINE.match(line)
    if matched is None:
        return None
    rates: dict[str, tuple[int, int]] = {}
    for cell in matched.group(3).split(" | "):
        parts = _CELL.match(cell.strip())
        if parts is None:
            return None
        rates[parts.group(1)] = (int(parts.group(2)), int(parts.group(3)))
    return ParsedStyle(int(matched.group(2)), rates, matched.group(1))


def merge_style(
    old: Mapping[str, Sequence[int]],
    old_hands: int,
    new: Mapping[str, Sequence[int]],
    new_hands: int,
    window: int = STYLE_WINDOW,
) -> tuple[dict[str, list[int]], int]:
    """Fold new evidence into what is already known: `(counts, hands)` over at most
    `window` seat-hands.

    Counts add; past the window both sides are scaled down together, so the oldest
    evidence fades in proportion rather than being cut off, which needs no record of
    where each hand came from. Evidence larger than the window alone simply replaces
    what was there.
    """
    total = old_hands + new_hands
    keep = 1.0 if total <= window else window / total
    merged: dict[str, list[int]] = {}
    for name in [*old, *(n for n in new if n not in old)]:
        events = sum(side[name][0] for side in (old, new) if name in side)
        chances = sum(side[name][1] for side in (old, new) if name in side)
        merged[name] = [round(events * keep), round(chances * keep)]
    return merged, round(total * keep)


class StyleTally:
    """What each participant of some hands did, counted per label.

    Fed the finished hands of a session (`add_hand`) with who sat where, it holds
    `(events, opportunities)` per statistic and the hands behind them, ready to be
    written next to the session's results (`export`) and merged into a ranking
    member (`merge_style`). `track` limits it to some labels, for a pass that only
    wants one model's style.
    """

    def __init__(self, track: Collection[str] | None = None) -> None:
        self._track = None if track is None else set(track)
        self._hands: dict[str, int] = {}
        self._events: dict[str, list[int]] = {}
        self._chances: dict[str, list[int]] = {}

    def add_hand(self, hand: HandHistory, labels: Mapping[int, str]) -> None:
        """`labels` names the model in each seat; seats it does not name are ignored."""
        dealt = [seat for seat in hand.starting_stacks if seat in labels]
        counts = analyse_hand(
            hand.actions,
            dealt=list(hand.starting_stacks),
            button_seat=hand.button_seat,
            board_cards=len(hand.community_cards),
            player_ids={seat: labels.get(seat, f"s{seat}") for seat in hand.starting_stacks},
        )
        for seat in dealt:
            label = labels[seat]
            if self._track is not None and label not in self._track:
                continue
            hand_counts = counts[label]
            events = self._events.setdefault(label, [0] * STAT_COUNT)
            chances = self._chances.setdefault(label, [0] * STAT_COUNT)
            for i in range(STAT_COUNT):
                events[i] += hand_counts.events[i]
                chances[i] += hand_counts.opportunities[i]
            self._hands[label] = self._hands.get(label, 0) + 1

    def export(self) -> dict[str, dict]:
        """`{label: {"hands": n, "style": {statistic: [events, opportunities]}}}`."""
        return {
            label: {
                "hands": hands,
                "style": {
                    name: [self._events[label][i], self._chances[label][i]]
                    for i, name in enumerate(STATS)
                },
            }
            for label, hands in self._hands.items()
        }
