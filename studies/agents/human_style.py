"""How differently do real players play from our models? The nine statistics of the HUD, compared.

    PYTHONPATH=src:studies/agents python studies/agents/human_style.py \\
        --real dataset_real_hands/unified --selfplay checkpoints/studies/belief/data-v2 --jobs 40 \\
        --out checkpoints/studies/human_style/players.json
    PYTHONPATH=src:studies/agents python studies/agents/human_style.py --report checkpoints/studies/human_style/players.json

Every player's statistics are counted the way the HUD counts them (`engine/stats.analyse_hand`: events
over opportunities, at the first chance in a hand), over all its hands:

- **real players**: HandHQ (online, 2009, by site; a player is a name within a site), the
  professionals who played Pluribus (`professionisti`) and Pluribus itself, from the unified data set
  (`real_data.py`); ACPC bots are left out;
- **our models**: the self-play sessions of the belief data (`belief_generate.py`): every seat a
  store model, half of them with a style -- `modelli` (plain) and `modelli con stile`.

The report gives, per group, the distribution of each statistic over the players with at least
`--min-hands` hands (10/25/50/75/90%), and the table sizes they played: a VPIP at six seats and at two
is not one number, so the groups are also split by table size (2, 3-6, 7-9).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from pokerlab.engine.history import _hand_history_from_dict
from pokerlab.engine.stats import STATS, analyse_hand

SIZES = (("2", 2, 2), ("3-6", 3, 6), ("7-9", 7, 9))


def size_group(players: int) -> str:
    return next(name for name, low, high in SIZES if low <= players <= high)


def count_file(path: Path, kind: str, min_stack_bb: float = 0.0) -> dict:
    """Per player (and table size): hands, and events/opportunities per statistic."""
    found: dict[str, dict] = defaultdict(lambda: {"hands": 0, "events": [0] * len(STATS), "chances": [0] * len(STATS)})
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            session = json.loads(line)
            styled = session.get("styled")
            for raw in session["hands"]:
                hand = _hand_history_from_dict(raw)
                dealt = sorted(hand.starting_stacks)
                if kind == "real":
                    source = session.get("source", "real")
                    ids = {seat: f"{source}|{hand.seat_names[seat]}" for seat in dealt}
                    groups = {seat: source for seat in dealt}
                    if source == "pluribus":
                        groups = {seat: "pluribus (bot)" if hand.seat_names[seat] == "Pluribus" else "professionisti"
                                  for seat in dealt}
                else:
                    ids = {seat: f"self|{session['session']}|{seat}" for seat in dealt}
                    groups = {seat: "modelli con stile" if styled and styled[seat] else "modelli" for seat in dealt}
                counts = analyse_hand(hand.actions, dealt=dealt, button_seat=hand.button_seat,
                                      board_cards=len(hand.community_cards), player_ids=ids)
                size = size_group(len(dealt))
                for seat in dealt:
                    if hand.starting_stacks[seat] < min_stack_bb * hand.big_blind:
                        continue
                    key = f"{groups[seat]}#{size}#{ids[seat]}"
                    row = found[key]
                    mine = counts[ids[seat]]
                    row["hands"] += 1
                    for i in range(len(STATS)):
                        row["events"][i] += mine.events[i]
                        row["chances"][i] += mine.opportunities[i]
    return dict(found)


def report(players: dict, min_hands: int) -> str:
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for key, row in players.items():
        group, size, _player = key.split("#", 2)
        if row["hands"] >= min_hands:
            groups[(group, size)].append(row)
    lines = [f"statistics per player (players with at least {min_hands} hands): median [10%-90%]", ""]
    header = f"{'gruppo':24s} {'tav.':>4s} {'giocatori':>9s} " + "".join(f"{name[:11]:>17s}" for name in STATS)
    lines.append(header)
    for (group, size), rows in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cells = []
        for i in range(len(STATS)):
            rates = sorted(r["events"][i] / r["chances"][i] for r in rows if r["chances"][i] >= 5)
            if len(rates) < 5:
                cells.append(f"{'-':>17s}")
                continue
            low, mid, high = (rates[int(f * (len(rates) - 1))] for f in (0.1, 0.5, 0.9))
            cells.append(f"{100 * mid:5.0f} [{100 * low:3.0f}-{100 * high:3.0f}] ")
        lines.append(f"{group:24s} {size:>4s} {len(rows):9d} " + "".join(f"{c:>17s}" for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from pokerlab.rl.forking import run_in_copies

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real", type=Path, default=None)
    parser.add_argument("--selfplay", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--every", type=int, default=1, help="read one file in this many")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/studies/human_style/players.json"))
    parser.add_argument("--report", type=Path, default=None, help="only report a saved --out")
    parser.add_argument("--min-hands", type=int, default=100)
    parser.add_argument("--min-stack-bb", type=float, default=0.0,
                        help="count only the hands a player started with this many big blinds or more: our models "
                        "play stacks of 1-100 bb, the online players mostly 100")
    args = parser.parse_args(argv)
    if args.report is not None:
        print(report(json.loads(args.report.read_text()), args.min_hands))
        return 0
    files = []
    if args.real is not None:
        files += [(p, "real") for p in sorted(args.real.glob("hands-*.jsonl.gz"))[:: args.every]
                  if "-acpc-" not in p.name]
    if args.selfplay is not None:
        files += [(p, "self") for p in sorted(args.selfplay.glob("hands-*.jsonl.gz"))[:: args.every]]
    merged: dict[str, dict] = {}
    for part in run_in_copies(lambda job: count_file(*files[job], args.min_stack_bb), len(files), args.jobs):
        for key, row in part.items():
            into = merged.setdefault(key, {"hands": 0, "events": [0] * len(STATS), "chances": [0] * len(STATS)})
            into["hands"] += row["hands"]
            into["events"] = [a + b for a, b in zip(into["events"], row["events"], strict=True)]
            into["chances"] = [a + b for a, b in zip(into["chances"], row["chances"], strict=True)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged))
    print(report(merged, args.min_hands))
    return 0


if __name__ == "__main__":
    sys.exit(main())
