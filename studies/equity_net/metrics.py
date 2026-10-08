"""How close the network is to the accurate equity, on the fixed validation set."""

from __future__ import annotations

import torch
from model import EquityNet

STREET_NAMES = ("preflop", "flop", "turn")  # the river is not evaluated: see data.without_river


@torch.no_grad()
def predict(model: EquityNet, val) -> torch.Tensor:
    holes, board, mask, _equity, _visible = val
    model.eval()
    try:
        return model(holes, board, mask)
    finally:
        model.train()


def evaluate_model(model: EquityNet, val) -> dict[str, float]:
    """Mean absolute and root-mean-square error of each player's share against the accurate
    equity, over every seat of every validation deal; the same with a uniform guess (1/n for
    everyone), which is what the network has to beat; and the mean error per street."""
    _holes, _board, mask, equity, visible = val
    pred = predict(model, val)
    real = mask.float()
    err = (pred - equity).abs() * real
    uniform = real / real.sum(-1, keepdim=True)
    out = {
        "mae": (err.sum() / real.sum()).item(),
        "rmse": ((((pred - equity) ** 2) * real).sum() / real.sum()).sqrt().item(),
        "mae_uniform": (((uniform - equity).abs() * real).sum() / real.sum()).item(),
    }
    for street, name in enumerate(STREET_NAMES):
        pick = (visible == street).float().unsqueeze(-1) * real
        out[f"mae_{name}"] = ((pred - equity).abs() * pick).sum().item() / pick.sum().item()
    return out
