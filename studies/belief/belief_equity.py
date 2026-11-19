"""Is the belief worth anything to a decision? My equity against one opponent, with and without it.

    PYTHONPATH=src:studies/belief python studies/belief/belief_equity.py --belief checkpoints/studies/belief/runs/full/belief.pt

For a sample (the observer about to act, one opponent still in) the label is the observer's share
of a heads-up pot against the opponent's *real* cards on the visible board, as the equity network
the critic reads says it (`EquityNet`, an expectation over the cards to come). It is predicted by:

- `mano`: a small network on the observer's own `phi` (the 32 numbers the policy reads today): the
  equity against any hand;
- `belief`: the same with the range's mean and spread in `phi` space (`range_embedding`): the
  vector the policy is to read;
- `esatta`: no network -- the equity against every combo of the believed range, weighed by it
  (1326 equity-network calls a sample);
- `oracolo`: the observer's `phi` with the opponent's true `phi`: what the vector is with the
  privileged information, the ceiling.

Explained variance on the test sessions, by street and whether the opponent has acted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from belief_data import load
from belief_model import board_planes, load_belief, range_embedding
from belief_records import STREETS
from belief_train import Data
from torch import nn

from pokerlab.rl.equity_net import equity_net_from_checkpoint, load_equity_checkpoint


def hole_planes(cards: torch.Tensor) -> torch.Tensor:
    planes = torch.zeros(len(cards), 52, device=cards.device)
    planes.scatter_(1, cards.long(), 1.0)
    return planes


@torch.no_grad()
def features(net, equity, data: Data, index: torch.Tensor, exact: bool) -> dict:
    batch = data.batch(index, augment=False)
    logits, phi, inverse = net(batch["tokens"], batch["lengths"], batch["static"], batch["board"], batch["known"])
    mean, spread = range_embedding(logits, phi, inverse)
    hand, observer, target = data.s_hand[index], data.s_observer[index], data.s_target[index]
    mine, theirs = data.holes[hand, observer], data.holes[hand, target]
    board = board_planes(batch["board"])
    my_phi = net.encoder(hole_planes(mine), board)
    their_phi = net.encoder(hole_planes(theirs), board)
    holes = torch.zeros(len(index), 9, 52, device=index.device)
    holes[:, 0] = hole_planes(mine)
    holes[:, 1] = hole_planes(theirs)
    mask = torch.zeros(len(index), 9, dtype=torch.bool, device=index.device)
    mask[:, :2] = True
    label = equity(holes, board, mask)[:, 0]
    out = {"label": label, "mine": my_phi, "mean": mean, "spread": spread, "theirs": their_phi,
           "street": batch["street"], "acted": batch["acted"]}
    if exact:
        p = torch.softmax(logits, dim=-1)
        values = torch.zeros(len(index), device=index.device)
        combos = net.combo_cards
        for row in range(len(index)):
            allowed = (p[row] > 1e-7).nonzero().squeeze(1)
            count = len(allowed)
            h = torch.zeros(count, 9, 52, device=index.device)
            h[:, 0] = hole_planes(mine[row].expand(count, 2))
            h[:, 1] = hole_planes(combos[allowed])
            m = torch.zeros(count, 9, dtype=torch.bool, device=index.device)
            m[:, :2] = True
            shares = equity(h, board[row].expand(count, -1), m)[:, 0]
            values[row] = (shares * p[row, allowed]).sum() / p[row, allowed].sum()
        out["exact"] = values
    return out


def collect(net, equity, data, index, size, exact=False) -> dict:
    parts = [features(net, equity, data, index[i : i + size], exact) for i in range(0, len(index), size)]
    return {key: torch.cat([p[key] for p in parts]) for key in parts[0]}


def fit(inputs: torch.Tensor, label: torch.Tensor, steps: int) -> nn.Module:
    head = nn.Sequential(nn.Linear(inputs.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, 1)).to(inputs.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(steps):
        pick = torch.randint(len(label), (4096,), device=label.device)
        loss = ((head(inputs[pick]).squeeze(1) - label[pick]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head.eval()


def explained(prediction, label, mask) -> float:
    residual = ((label[mask] - prediction[mask]) ** 2).mean()
    return float(1 - residual / label[mask].var())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("checkpoints/studies/belief/data"))
    parser.add_argument("--belief", required=True)
    parser.add_argument("--train-samples", type=int, default=400000)
    parser.add_argument("--test-samples", type=int, default=100000)
    parser.add_argument("--exact-samples", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    net = load_belief(args.belief, args.device)
    equity = equity_net_from_checkpoint(load_equity_checkpoint(
        torch.load(args.belief, map_location="cpu", weights_only=False)["equity_model"])).to(args.device)
    data = Data(load(args.data), args.device)
    train_index = data.train[torch.randperm(len(data.train), device=args.device)[: args.train_samples]]
    test_index = data.test_seen[torch.randperm(len(data.test_seen), device=args.device)[: args.test_samples]]
    train = collect(net, equity, data, train_index, 2048)
    test = collect(net, equity, data, test_index, 2048)
    arms = {
        "mano": lambda d: d["mine"],
        "belief": lambda d: torch.cat([d["mine"], d["mean"], d["spread"]], dim=1),
        "oracolo": lambda d: torch.cat([d["mine"], d["theirs"]], dim=1),
    }
    predictions = {}
    for name, inputs in arms.items():
        head = fit(inputs(train), train["label"], args.steps)
        with torch.no_grad():
            predictions[name] = head(inputs(test)).squeeze(1)
    groups = {"tutti": torch.ones_like(test["label"], dtype=torch.bool)}
    for i, street in enumerate(STREETS):
        groups[street.value] = test["street"] == i
    groups["ha agito"] = test["acted"]
    groups["non ancora"] = ~test["acted"]
    print(f"varianza spiegata dell'equity contro la mano vera ({len(test['label'])} campioni di test)")
    print(f"{'gruppo':12s} " + " ".join(f"{name:>8s}" for name in predictions))
    for group, mask in groups.items():
        print(f"{group:12s} " + " ".join(f"{explained(p, test['label'], mask):8.3f}" for p in predictions.values()))
    exact = collect(net, equity, data, test_index[: args.exact_samples], 256, exact=True)
    head_mine = fit(arms["mano"](train), train["label"], args.steps)
    head_belief = fit(arms["belief"](train), train["label"], args.steps)
    with torch.no_grad():
        base = head_mine(arms["mano"](exact)).squeeze(1)
        learned = head_belief(arms["belief"](exact)).squeeze(1)
    every = torch.ones_like(exact["label"], dtype=torch.bool)
    acted = exact["acted"]
    print(f"su {len(exact['label'])} campioni, l'equity esatta contro il range creduto:")
    for name, mask in (("tutti", every), ("ha agito", acted)):
        print(f"  {name:10s} mano {explained(base, exact['label'], mask):.3f}  belief "
              f"{explained(learned, exact['label'], mask):.3f}  esatta {explained(exact['exact'], exact['label'], mask):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
