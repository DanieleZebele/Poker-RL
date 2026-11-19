"""Real hands in the PHH format (`dataset_real_hands/`) as the hands the belief study reads.

The poker hand history format (https://phh.readthedocs.io) records a hand as a list of actions:
`d dh p1 AsKd` (hole cards dealt, `????` when unknown), `d db 7c8d9h` (board), `p3 f` (fold),
`p3 cc` (check or call), `p3 cbr 300` (bet or raise *to* 300), `p3 sm AsKd` (show; `????` mucks).
The players are listed from the small blind round to the button; **heads-up the blinds are
reversed**, as in pokerkit: `p1` posts the big blind and `p2`, the button, the small one and acts
first preflop.

`convert_hand` turns one into the dict `engine/history.py` writes (`_to_jsonable`), replaying the
betting to fill in what every `ActionRecord` holds (the street total after the action, the pot
before it, the stacks before and after), every amount scaled so the big blind is 100 chips (the
engine's, `TableMix`). Only **no-limit Texas hold'em** (`variant = 'NT'`) of 2 to 9 players with
two blinds and no ante or straddle is kept; anything else returns `None` with the reason. Hole cards
are kept only where the history knows them: every player's for a full-information source (Pluribus,
the ACPC, a televised table), the ones shown at a showdown for an online one (HandHQ). `seat_names`
holds the players' names, which is who a player is across hands.

`sessions_of_file` groups a file's hands by table, in the order they were played, into sessions of
at most `SESSION_HANDS` (what the memory of the belief network reads, `belief_data.py`). Pure Python.
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
BIG_BLIND = 100
SESSION_HANDS = 200
STREETS = ("preflop", "flop", "turn", "river")


class Skip(Exception):
    """A hand this study does not read, and why."""


def _cards(text: str) -> list[str] | None:
    """`AsKd` -> ['As', 'Kd']; None for `????` (unknown)."""
    if "?" in text:
        return None
    if len(text) % 2:
        raise Skip(f"cards {text!r}")
    return [text[i] + text[i + 1] for i in range(0, len(text), 2)]


def parse_file(text: str) -> list[dict[str, Any]]:
    """The hands of a `.phh` (one hand) or `.phhs` (sections `[1]`, `[2]`, ...) file."""
    data = tomllib.loads(text)
    if "actions" in data:
        return [data]
    return [value for key, value in sorted(data.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
            if isinstance(value, dict) and "actions" in value]


UNKNOWN_STACK = 100 * BIG_BLIND


def _replay(actions: list[str], stacks: list[int], names: list[str], posts, chips) -> dict[str, Any]:
    """The betting of `actions` from `stacks`: the records, the cards known (and how), the board, who
    folded and the stacks left. Raises `Skip` on anything it cannot read."""
    count = len(stacks)
    holes: dict[int, list[str]] = {}
    known_by: dict[int, str] = {}
    board: list[str] = []
    records: list[dict[str, Any]] = []
    bets = [0] * count
    pot = 0
    street = 0
    folded: set[int] = set()
    pot_before = 0

    def record(seat: int, kind: str, before: int) -> None:
        records.append({
            "street": STREETS[street], "seat": seat, "player_id": names[seat], "action_type": kind,
            "amount": bets[seat], "stack_before": before, "stack_after": stacks[seat],
            "pot_before": pot_before, "timestamp": 0.0,
        })

    for seat, blind in posts:
        pot_before = pot + sum(bets)
        before = stacks[seat]
        paid = min(blind, stacks[seat])
        bets[seat] += paid
        stacks[seat] -= paid
        record(seat, "post_blind", before)
    level = max(bets)

    for action in actions:
        parts = action.split()
        if parts[0] == "d":
            if parts[1] == "dh":
                seat = int(parts[2][1:]) - 1
                cards = _cards(parts[3]) if len(parts) > 3 else None
                if cards is not None:
                    holes[seat] = cards
                    known_by[seat] = "dealt"
            elif parts[1] == "db":
                cards = _cards(parts[2])
                if cards is None:
                    raise Skip("unknown board")
                board += cards
                street += 1
                if street > 3 or len(board) != (3, 4, 5)[street - 1]:
                    raise Skip(f"board {board}")
                pot += sum(bets)
                bets = [0] * count
                level = 0
            else:
                raise Skip(f"dealer action {action!r}")
            continue
        seat = int(parts[0][1:]) - 1
        verb = parts[1]
        if verb == "sm":
            cards = _cards(parts[2]) if len(parts) > 2 else None
            if cards is not None:
                holes[seat] = cards
                known_by.setdefault(seat, "shown")
            continue
        if seat in folded or not 0 <= seat < count:
            raise Skip(f"action of seat {seat}")
        pot_before = pot + sum(bets)
        before = stacks[seat]
        if verb == "f":
            folded.add(seat)
            record(seat, "fold", before)
        elif verb == "cc":
            owed = level - bets[seat]
            if owed <= 0:
                record(seat, "check", before)
            else:
                paid = min(owed, stacks[seat])
                bets[seat] += paid
                stacks[seat] -= paid
                record(seat, "all_in" if stacks[seat] == 0 else "call", before)
        elif verb == "cbr":
            target = chips(float(parts[2]))
            paid = min(max(target - bets[seat], 0), stacks[seat])
            if paid == 0 or bets[seat] + paid <= level:
                raise Skip(f"raise {action!r} to no more than {level}")
            raising = level > 0
            bets[seat] += paid
            stacks[seat] -= paid
            level = bets[seat]
            record(seat, "all_in" if stacks[seat] == 0 else ("raise" if raising else "bet"), before)
        else:
            raise Skip(f"verb {verb!r}")
    return {"records": records, "holes": holes, "known_by": known_by, "board": board, "folded": folded,
            "stacks": stacks}


def convert_hand(hand: dict[str, Any], *, hand_id: str, full_information: bool) -> dict[str, Any]:
    """`hand` (a parsed PHH hand) as an `engine/history.py` dict. Raises `Skip`."""
    if hand.get("variant") != "NT":
        raise Skip(f"variant {hand.get('variant')}")
    stacks_raw = list(hand["starting_stacks"])
    count = len(stacks_raw)
    if not 2 <= count <= 9:
        raise Skip(f"{count} players")
    if any(hand.get("antes", [0] * count)):
        raise Skip("antes")
    blinds = list(hand.get("blinds_or_straddles", []))
    if len(blinds) != count or any(blinds[2:]) or not blinds[0] or not blinds[1] or blinds[0] > blinds[1]:
        raise Skip(f"blinds {blinds}")
    scale = BIG_BLIND / blinds[1]

    def chips(amount: float) -> int:
        return round(amount * scale)

    if not all(isinstance(x, (int, float)) and x > 0 for x in stacks_raw):
        raise Skip("stack")
    names = [str(x) for x in hand.get("players", [f"p{i + 1}" for i in range(count)])]
    if len(names) != count:
        raise Skip("players")
    if count == 2:  # reversed blinds: p1 the big blind, p2 the button posts the small one
        button, posts = 1, ((1, chips(blinds[0])), (0, chips(blinds[1])))
    else:
        button, posts = count - 1, ((0, chips(blinds[0])), (1, chips(blinds[1])))
    if all(x < 1e12 for x in stacks_raw):
        starting = [chips(x) for x in stacks_raw]
    else:
        # iPoker's histories give no stacks: 100 big blinds, or what the player put in if more
        played = _replay(hand["actions"], [10**12] * count, names, posts, chips)
        starting = [max(UNKNOWN_STACK, 10**12 - left) for left in played["stacks"]]
    if min(starting) <= 0:
        raise Skip("empty stack")
    played = _replay(hand["actions"], list(starting), names, posts, chips)
    records, holes, board, folded, stacks = (played[k] for k in ("records", "holes", "board", "folded", "stacks"))
    if hand.get("finishing_stacks") is not None:
        final = [chips(x) for x in hand["finishing_stacks"]]
    else:
        # HandHQ's `winnings` cannot be trusted (left out at a showdown, 0 for a pot won uncontested on
        # PokerStars): the pots are shared out here, each to the best hand shown among the players
        # still in it (one who mucked loses), rake left out
        final = _settle(starting, stacks, folded, holes, board, button)
    for cards in holes.values():
        if len(cards) != 2:
            raise Skip(f"hole cards {cards}")
    if full_information and len(holes) != count:
        raise Skip("full information without every hole card")
    known_by = played["known_by"] if full_information else dict.fromkeys(holes, "shown")
    seats = range(count)
    return {
        "schema_version": SCHEMA_VERSION,
        "hand_id": hand_id,
        "started_at": 0.0,
        "num_players": count,
        "small_blind": chips(blinds[0]),
        "big_blind": BIG_BLIND,
        "button_seat": button,
        "starting_stacks": {str(s): starting[s] for s in seats},
        "seat_names": {str(s): names[s] for s in seats},
        "community_cards": board,
        "actions": records,
        "hole_cards": {str(s): cards for s, cards in holes.items()},
        # how each known hand came to be known: `dealt` (the history saw it dealt: whatever the
        # player then did) or `shown` (only because it was shown at a showdown)
        # (an online source's cards are all `shown`: iPoker writes a showdown's cards where they are dealt)
        "known_by": {str(s): how for s, how in known_by.items()},
        "payouts": {},
        "final_stacks": {str(s): final[s] for s in seats},
    }


def _settle(starting: list[int], stacks: list[int], folded: set[int], holes: dict[int, list[str]],
            board: list[str], button: int) -> list[int]:
    """Every player's final stack, the pots (side pots included) going to the best hand shown among
    the players in each, split evenly (an odd chip to the first after the button). A pot no one
    still in it showed for goes back to the players still in it, evenly."""
    from pokerlab.cards.card import Card
    from pokerlab.evaluator.evaluator import evaluate

    count = len(starting)
    put = [starting[i] - stacks[i] for i in range(count)]
    final = list(stacks)
    strength = {}
    if len(board) == 5:
        for seat, cards in holes.items():
            if seat not in folded:
                strength[seat] = evaluate([Card.parse(c) for c in cards + board])
    order = [(button + 1 + i) % count for i in range(count)]
    levels = sorted({p for i, p in enumerate(put) if i not in folded and p > 0})
    floor = 0
    for level in levels:
        amount = sum(min(p, level) - min(p, floor) for p in put)
        eligible = [i for i in order if i not in folded and put[i] >= level]
        shown = [i for i in eligible if i in strength]
        if shown:
            best = max(strength[i] for i in shown)
            winners = [i for i in shown if strength[i] == best]
        else:
            winners = eligible
        share, odd = divmod(amount, len(winners))
        for i in winners:
            final[i] += share
        for i in winners[:odd]:
            final[i] += 1
        floor = level
    # chips over the highest level still in (an uncalled bet) go back to whoever put them in
    for i in range(count):
        if put[i] > floor:
            final[i] += put[i] - floor
    return final


def _order_key(hand: dict[str, Any], index: int) -> tuple:
    return (hand.get("year", 0), hand.get("month", 0), hand.get("day", 0), str(hand.get("time", "")),
            hand.get("hand", 0) if isinstance(hand.get("hand"), int) else 0, index)


def sessions_of_file(
    hands: Iterable[dict[str, Any]], *, source: str, full_information: bool, skipped: dict[str, int]
) -> list[dict[str, Any]]:
    """A file's hands, converted, as sessions: by table (`venue`, `table`), in the order played,
    `SESSION_HANDS` at most. `skipped` counts the reasons hands were left out."""
    by_table: dict[tuple, list[tuple[tuple, dict]]] = defaultdict(list)
    for index, hand in enumerate(hands):
        key = (hand.get("venue", source), hand.get("table", ""))
        try:
            converted = convert_hand(hand, hand_id=f"{source}:{hand.get('hand', index)}",
                                     full_information=full_information)
        except (Skip, KeyError, ValueError, TypeError) as error:
            reason = str(error).split(" ")[0] if isinstance(error, Skip) else type(error).__name__
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        by_table[key].append((_order_key(hand, index), converted))
    sessions = []
    for key, rows in by_table.items():
        rows.sort(key=lambda row: row[0])
        played = [hand for _, hand in rows]
        for start in range(0, len(played), SESSION_HANDS):
            sessions.append({"source": source, "table": "/".join(map(str, key)), "full_information": full_information,
                             "hands": played[start : start + SESSION_HANDS]})
    return sessions


def read_phh(path: Path) -> list[dict[str, Any]]:
    return parse_file(Path(path).read_text(encoding="utf-8", errors="replace"))
