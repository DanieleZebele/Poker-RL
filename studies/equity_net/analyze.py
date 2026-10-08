"""Where does the equity network go wrong? A closer look than `evaluate.py`.

    PYTHONPATH=src:studies/equity_net python studies/equity_net/analyze.py
    PYTHONPATH=src:studies/equity_net python studies/equity_net/analyze.py --top 25 --min-count 40
    PYTHONPATH=src:studies/equity_net python studies/equity_net/analyze.py --hard

Every seat of every validation deal is one row: its true equity (accurate, many board
completions), what the network says, the street, the number of players and the hand. The
report then answers, in order:

1. the error per street (mean error, root-mean-square, and the signed bias);
2. the error per number of players, and street by number of players;
3. which preflop hands it gets wrong most (the 169 starting hands) and which kinds of hand;
4. which made hands it gets wrong most after the flop (pair, flush, ...);
5. by how much it is off when the true equity is low or high;
6. the single worst seats, with the cards;
7. a few deals picked at random (`--samples`, 5 by default), with the true equity and the network's.

`--hard` reads the set of hard cases (two full houses, two flushes, a kicker deciding...)
instead of the ordinary deals and adds the error by kind of case.

A cell with few rows says little: counts are printed, cells under `--min-count` rows are
left out of the rankings, and the last column is the 95% interval of the signed bias.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from data import (
    DEFAULT_HARD_DEALS,
    DEFAULT_VAL_DEALS,
    DEFAULT_VAL_RUNOUTS,
    HARD_TAGS,
    hard_path,
    hard_set,
    validation_path,
    validation_set,
)
from deals import card_from_index
from evaluate import load
from metrics import STREET_NAMES, predict

from pokerlab.evaluator.evaluator import HandCategory, evaluate

RUNS = Path(__file__).parent / "runs"
RANKS = "23456789TJQKA"
SUITS = "cdhs"
CATEGORY_NAMES = {
    HandCategory.HIGH_CARD: "carta alta",
    HandCategory.PAIR: "coppia",
    HandCategory.TWO_PAIR: "doppia coppia",
    HandCategory.THREE_OF_A_KIND: "tris",
    HandCategory.STRAIGHT: "scala",
    HandCategory.FLUSH: "colore",
    HandCategory.FULL_HOUSE: "full",
    HandCategory.FOUR_OF_A_KIND: "poker",
    HandCategory.STRAIGHT_FLUSH: "scala colore",
}


def text(index: int) -> str:
    return RANKS[index % 13] + SUITS[index // 13]


def hole_class(a: int, b: int) -> str:
    """The starting hand as poker players name it: AA, AKs, T9o."""
    ranks = sorted((a % 13, b % 13), reverse=True)
    suited = a // 13 == b // 13
    name = RANKS[ranks[0]] + RANKS[ranks[1]]
    return name if ranks[0] == ranks[1] else name + ("s" if suited else "o")


def hole_group(a: int, b: int) -> str:
    hi, lo = sorted((a % 13, b % 13), reverse=True)
    suited = a // 13 == b // 13
    if hi == lo:
        return "coppie alte (TT-AA)" if hi >= 8 else "coppie medie (66-99)" if hi >= 4 else "coppie basse (22-55)"
    gap = hi - lo
    if hi >= 11 and lo >= 8:
        return "due carte alte (T+)" + (" suited" if suited else " offsuit")
    if hi == 12:
        return "asso con kicker basso" + (" suited" if suited else " offsuit")
    if gap == 1 and lo >= 3:
        return "connesse" + (" suited" if suited else " offsuit")
    return "altre" + (" suited" if suited else " offsuit")


class Rows:
    """One row per seat of the validation set, as plain arrays and lists."""

    def __init__(self, model, val, tags=None) -> None:
        holes, board, mask, equity, visible = val
        pred = predict(model, val)
        self.true, self.pred = [], []
        self.players, self.street, self.deal = [], [], []
        self.hole, self.group, self.category, self.cards = [], [], [], []
        for d in range(len(holes)):
            n = int(mask[d].sum())
            known = [int(i) for i in torch.nonzero(board[d]).flatten()]
            street = int(visible[d])
            for p in range(n):
                hole = [int(i) for i in torch.nonzero(holes[d, p]).flatten()]
                self.true.append(float(equity[d, p]))
                self.pred.append(float(pred[d, p]))
                self.players.append(n)
                self.street.append(street)
                self.deal.append(d)
                self.hole.append(hole_class(*hole))
                self.group.append(hole_group(*hole))
                if known:
                    rank = evaluate([card_from_index(c) for c in hole + known])
                    self.category.append(CATEGORY_NAMES[rank.category])
                else:
                    self.category.append("")
                self.cards.append((hole, known))
        self.true, self.pred = np.array(self.true), np.array(self.pred)
        self.players, self.street = np.array(self.players), np.array(self.street)
        self.error = self.pred - self.true
        self.tag = np.array(tags)[self.deal] if tags is not None else None

    def stats(self, pick: np.ndarray) -> tuple[int, float, float, float, float]:
        """Rows, mean error, rmse, signed bias and the half-width of the bias's 95% interval
        (seats of one deal are not independent, so it is a little optimistic)."""
        e = self.error[pick]
        half = 1.96 * float(e.std()) / max(1.0, float(np.sqrt(len(e))))
        return int(pick.sum()), float(np.abs(e).mean()), float(np.sqrt((e**2).mean())), float(e.mean()), half


def table(title: str, groups: dict[str, np.ndarray], rows: Rows, min_count: int, top: int | None = None,
          worst_first: bool | None = True) -> None:
    """`worst_first` None keeps the groups in the order given."""
    results = []
    for name, pick in groups.items():
        if pick.sum() >= min_count:
            results.append((name, *rows.stats(pick)))
    if worst_first is not None:
        results.sort(key=lambda r: r[2], reverse=worst_first)
    print(f"\n{title}")
    if not results:
        print(f"  nessun gruppo con almeno {min_count} righe: serve una validazione piu' grande "
              "o un --min-count piu' basso")
        return
    print(f"  {'':24s} {'n':>6s} {'errore':>8s} {'rmse':>8s} {'scarto medio':>13s} {'+/-':>7s}")
    for name, n, mae, rmse, bias, half in results[:top]:
        print(f"  {name:24s} {n:6d} {mae:8.4f} {rmse:8.4f} {bias:+13.4f} {half:7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=Path, default=RUNS / "model.pt")
    parser.add_argument("--val-deals", type=int, default=DEFAULT_VAL_DEALS)
    parser.add_argument("--val-runouts", type=int, default=DEFAULT_VAL_RUNOUTS, help="preflop completions per deal")
    parser.add_argument("--hard", action="store_true", help="analyse the hard cases instead of ordinary deals")
    parser.add_argument("--hard-deals", type=int, default=DEFAULT_HARD_DEALS, help="hard cases of each kind")
    parser.add_argument("--val-jobs", type=int, default=24, help="processes making the set if it is missing")
    parser.add_argument("--top", type=int, default=15, help="rows in each ranking")
    parser.add_argument("--min-count", type=int, default=30, help="fewest rows for a cell to be ranked")
    parser.add_argument("--min-count-hand", type=int, default=10,
                        help="the same for one of the 169 starting hands, each of which is rarer")
    parser.add_argument("--samples", type=int, default=5, help="random deals printed with true and predicted equity")
    parser.add_argument("--seed", type=int, default=None, help="seed of those random deals (default: different every run)")
    args = parser.parse_args()

    model = load(args.model)
    tags = None
    if args.hard:
        val, tags = hard_set(hard_path(RUNS, args.hard_deals), args.hard_deals, args.val_jobs)
    else:
        val = validation_set(
            validation_path(RUNS, args.val_deals, args.val_runouts),
            args.val_deals, args.val_runouts, args.val_jobs,
        )
    rows = Rows(model, val, tags)
    everything = np.ones(len(rows.true), bool)
    n, mae, rmse, bias, _half = rows.stats(everything)
    uniform = np.abs(1.0 / rows.players - rows.true).mean()
    print(f"modello {args.model}\n{len(val[0])} mani, {n} seggi: errore medio {mae:.4f} "
          f"(rmse {rmse:.4f}, scarto medio {bias:+.4f}); stima uniforme {uniform:.4f}")

    if tags is not None:
        table("per tipo di caso difficile", {name: rows.tag == i for i, name in enumerate(HARD_TAGS)}, rows, 1,
              worst_first=None)
    table("per strada", {STREET_NAMES[s]: rows.street == s for s in range(len(STREET_NAMES))}, rows, 1, worst_first=None)
    table("per numero di giocatori", {f"{k} giocatori": rows.players == k for k in range(2, 10)},
          rows, 1, worst_first=None)

    print("\nerrore medio per strada (righe) e numero di giocatori (colonne)")
    print("            " + "".join(f"{k:>8d}" for k in range(2, 10)))
    for s in range(len(STREET_NAMES)):
        cells = []
        for k in range(2, 10):
            pick = (rows.street == s) & (rows.players == k)
            cells.append(f"{rows.stats(pick)[1]:8.4f}" if pick.sum() >= 10 else f"{'-':>8s}")
        print(f"  {STREET_NAMES[s]:9s}" + "".join(cells))

    pre = rows.street == 0
    by_class = defaultdict(list)
    by_group = defaultdict(list)
    for i in np.nonzero(pre)[0]:
        by_class[rows.hole[i]].append(i)
        by_group[rows.group[i]].append(i)

    def masks(groups):
        out = {}
        for name, idx in groups.items():
            m = np.zeros(len(rows.true), bool)
            m[idx] = True
            out[name] = m
        return out

    table("preflop: tipi di mano di partenza", masks(by_group), rows, args.min_count)
    table(f"preflop: le {args.top} mani di partenza dove sbaglia di piu'", masks(by_class), rows,
          args.min_count_hand, top=args.top)
    table(f"preflop: le {args.top} dove sbaglia di meno", masks(by_class), rows,
          args.min_count_hand, top=args.top, worst_first=False)

    post = rows.street > 0
    categories = {c: post & np.array([x == c for x in rows.category]) for c in CATEGORY_NAMES.values()}
    table("dopo il flop: per mano fatta (quello che hai gia' sul board)", categories, rows, args.min_count)

    edges = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0001]
    table("per equity vera", {f"{lo:.2f}-{min(hi, 1.0):.2f}": (rows.true >= lo) & (rows.true < hi)
                              for lo, hi in pairwise(edges)}, rows, 1, worst_first=None)

    holes, board, mask, equity, visible = val
    shares = predict(model, val)

    def show(title: str, deal: int) -> None:
        """One deal with the cards of every player, the true equity and the network's."""
        known = " ".join(text(int(c)) for c in torch.nonzero(board[deal]).flatten()) or "(nessuna)"
        print(f"\n{title}{STREET_NAMES[int(visible[deal])]}, {int(mask[deal].sum())} giocatori, board {known}")
        for p in range(int(mask[deal].sum())):
            cards = " ".join(text(int(c)) for c in torch.nonzero(holes[deal, p]).flatten())
            mark = "  <-- sbaglia qui" if abs(float(shares[deal, p]) - float(equity[deal, p])) > 0.15 else ""
            print(f"     giocatore {p + 1}: {cards}   vera {float(equity[deal, p]):.3f}   "
                  f"rete {float(shares[deal, p]):.3f}{mark}")

    print(f"\nle {args.top} situazioni dove sbaglia di piu' (con le carte di tutti i giocatori della mano)")
    shown: set[int] = set()
    for i in np.argsort(-np.abs(rows.error)):
        deal = rows.deal[i]
        if deal in shown:  # several seats of one hand can be among the worst: show it once
            continue
        shown.add(deal)
        if len(shown) > args.top:
            break
        show(f"{len(shown)}. ", deal)

    if args.samples:
        rng = random.Random(args.seed)
        print(f"\n{args.samples} mani a caso (vera e prevista)")
        for k, deal in enumerate(rng.sample(range(len(holes)), min(args.samples, len(holes))), start=1):
            show(f"{k}. ", deal)


if __name__ == "__main__":
    main()
