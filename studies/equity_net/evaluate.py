"""Evaluates a trained equity network.

    PYTHONPATH=src:studies/equity_net python studies/equity_net/evaluate.py
    PYTHONPATH=src:studies/equity_net python studies/equity_net/evaluate.py --model other.pt

What it reports, on the fixed validation set (deals with an accurate equity, never used to
train):

1. the error overall and per street, next to the uniform guess it has to beat;
2. the error by number of players;
3. the floor the error cannot go under: only the preflop labels are estimates (from a finite
   number of board completions); from the flop on they are exact;
4. calibration: among the seats the network gives a share near x, what their accurate
   equity averages;
5. that the answer does not depend on the order the players are listed in;
6. a few matchups whose equity is known.
"""

from __future__ import annotations

import argparse
import random
from itertools import pairwise
from pathlib import Path

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
from deals import MAX_PLAYERS, SUITS
from metrics import STREET_NAMES, evaluate_model, predict
from model import EquityNet

from pokerlab.cards import Card

RUNS = Path(__file__).parent / "runs"


def load(path: Path) -> EquityNet:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = EquityNet.from_checkpoint(saved)
    model.load_state_dict(saved["state"])
    return model.eval()


def index(text: str) -> int:
    card = Card.parse(text)
    return SUITS.index(card.suit) * 13 + int(card.rank) - 2


def by_players(model: EquityNet, val) -> list[str]:
    _h, _b, mask, equity, _v = val
    pred = predict(model, val)
    sizes = mask.sum(-1)
    lines = []
    for n in range(2, MAX_PLAYERS + 1):
        pick = (sizes == n).unsqueeze(-1) & mask
        if pick.any():
            err = ((pred - equity).abs() * pick).sum() / pick.sum()
            lines.append(f"  {n} giocatori: errore medio {err:.4f}  ({int((sizes == n).sum())} mani)")
    return lines


def label_noise_floor(val, runouts: int) -> float:
    """Typical error of a preflop label itself: a share of p estimated from `runouts`
    completions has a standard deviation of about sqrt(p (1 - p) / runouts). Labels from the
    flop on enumerate every completion and have none."""
    _h, _b, mask, equity, visible = val
    live = mask & (visible == 0).unsqueeze(-1)
    sd = (equity * (1 - equity) / runouts).sqrt()
    return (sd * live).sum().item() / live.sum().item() * 0.8  # mean |error| = 0.8 sd


def hard_cases(model: EquityNet, hard, tags) -> list[str]:
    """Error on the hard cases, one line per kind (the equity is exact in all of them)."""
    _h, _b, mask, equity, _v = hard
    pred = predict(model, hard)
    lines = [f"  {'':20s} {'mani':>6s} {'errore':>8s} {'rmse':>8s} {'scarto medio':>13s}"]
    for tag, name in enumerate(HARD_TAGS):
        pick = torch.from_numpy(tags == tag).unsqueeze(-1) & mask
        if pick.any():
            e = (pred - equity)[pick]
            lines.append(f"  {name:20s} {int((tags == tag).sum()):6d} {e.abs().mean():8.4f} "
                         f"{e.pow(2).mean().sqrt():8.4f} {e.mean():+13.4f}")
    return lines


def calibration(model: EquityNet, val, bins: int = 10) -> list[str]:
    _h, _b, mask, equity, _v = val
    pred = predict(model, val)[mask]
    real = equity[mask]
    lines = ["  quota data dalla rete -> equity accurata media (n seggi)"]
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in pairwise(edges.tolist()):
        pick = (pred >= lo) & (pred < hi if hi < 1 else pred <= hi)
        if pick.any():
            lines.append(
                f"  {lo:.1f}-{hi:.1f}: rete {pred[pick].mean():.3f}  vera {real[pick].mean():.3f}  ({int(pick.sum())})"
            )
    return lines


def order_independence(model: EquityNet, val, trials: int = 200) -> float:
    """Largest change in any share when the players of a deal are listed in another order."""
    holes, board, mask, _e, _v = val
    rng = random.Random(0)
    worst = 0.0
    with torch.no_grad():
        for b in rng.sample(range(len(holes)), min(trials, len(holes))):
            n = int(mask[b].sum())
            order = list(range(n))
            rng.shuffle(order)
            h = holes[b : b + 1].clone()
            h[0, :n] = h[0, order]
            base = model(holes[b : b + 1], board[b : b + 1], mask[b : b + 1])[0, :n]
            moved = model(h, board[b : b + 1], mask[b : b + 1])[0, :n]
            worst = max(worst, (base[order] - moved).abs().max().item())
    return worst


def matchup(model: EquityNet, hands: list[tuple[str, str]], board: list[str]) -> list[float]:
    holes = torch.zeros(1, MAX_PLAYERS, 52)
    mask = torch.zeros(1, MAX_PLAYERS, dtype=torch.bool)
    for p, (a, b) in enumerate(hands):
        holes[0, p, [index(a), index(b)]] = 1.0
        mask[0, p] = True
    known = torch.zeros(1, 52)
    for card in board:
        known[0, index(card)] = 1.0
    with torch.no_grad():
        return model(holes, known, mask)[0, : len(hands)].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=Path, default=RUNS / "model.pt")
    parser.add_argument("--val-deals", type=int, default=DEFAULT_VAL_DEALS)
    parser.add_argument("--val-runouts", type=int, default=DEFAULT_VAL_RUNOUTS, help="preflop completions per deal")
    parser.add_argument("--hard-deals", type=int, default=DEFAULT_HARD_DEALS, help="hard cases of each kind")
    parser.add_argument("--val-jobs", type=int, default=24, help="processes making the sets if they are missing")
    args = parser.parse_args()

    model = load(args.model)
    val = validation_set(
        validation_path(RUNS, args.val_deals, args.val_runouts),
        args.val_deals, args.val_runouts, args.val_jobs,
    )
    hard, tags = hard_set(hard_path(RUNS, args.hard_deals), args.hard_deals, args.val_jobs)
    metrics = evaluate_model(model, val)

    print(f"modello: {args.model}  ({sum(p.numel() for p in model.parameters()):,} parametri)")
    print(f"validazione: {len(val[0])} mani\n")
    print("errore sulla quota di piatto di ogni giocatore (assoluto medio):")
    print(f"  rete      {metrics['mae']:.4f}   (rmse {metrics['rmse']:.4f})")
    print(f"  uniforme  {metrics['mae_uniform']:.4f}   (1/n per tutti: quello da battere)")
    print(f"  minimo raggiungibile, preflop ~{label_noise_floor(val, args.val_runouts):.4f} "
          f"(etichette su {args.val_runouts} estrazioni); dal flop in poi le etichette sono esatte")
    print("\nper strada:")
    for name in STREET_NAMES:
        print(f"  {name:8s} {metrics['mae_' + name]:.4f}")
    print("\nper numero di giocatori:")
    print("\n".join(by_players(model, val)))
    print(f"\ncasi difficili ({len(hard[0])} mani, equity esatta):")
    print("\n".join(hard_cases(model, hard, tags)))
    print("\ncalibrazione:")
    print("\n".join(calibration(model, val)))
    print(f"\nordine dei giocatori: variazione massima {order_independence(model, val):.2e} "
          "(deve essere ~0)")

    print("\nmatchup noti (preflop):")
    for label, hands, expected in [
        ("AsAh contro KdKc", [("As", "Ah"), ("Kd", "Kc")], "~0.813 / 0.187"),
        ("AsAh contro KsKh", [("As", "Ah"), ("Ks", "Kh")], "~0.826 / 0.174"),
        ("AKs contro 22 (testa a testa)", [("As", "Ks"), ("2d", "2c")], "~0.50 / 0.50"),
        ("72o contro AA", [("7d", "2c"), ("As", "Ah")], "~0.12 / 0.88"),
    ]:
        shares = matchup(model, hands, [])
        print(f"  {label:32s} rete {' / '.join(f'{s:.3f}' for s in shares)}   attesi {expected}")


if __name__ == "__main__":
    main()
