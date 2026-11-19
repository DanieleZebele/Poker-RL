"""How the clone of real players and our models play draws and the straights and flushes they make.

    PYTHONPATH=src:studies/agents:studies/human_clone python studies/human_clone/clone_draws.py \\
        --models '#1' checkpoints/studies/human_clone/run/clone.pt

Two views, on the flop and the turn, by the category of the hand the player holds (`category`:
made flush, made straight, combo draw, flush draw, open-ended, gutshot, a draw with a pair, a pair or
better, nothing; a draw only counts when one of the player's own cards is in it):

- **Pluribus** (`--pluribus`, `clone_generate.py --source pluribus`: five professionals and the bot,
  every card known): what each did in its own spots, against what every model gives in the same
  spot with the same cards;
- **HandHQ spots** (`--handhq`): real states of online players, each given every two-card hand of
  each category that fits the board (at most `--per-category` a state), every model asked; a
  counterfactual, "how does it play this kind of hand here".

Facing a bet: fold / call / raise (all-in included). Checked to: check / bet small (up to half the
pot) / bet big (three quarters and up, all-in included).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from pokerlab.rl.action_space import ALL_IN_BIN, CHECK_CALL_BIN, FOLD_BIN

CATEGORIES = ("colore fatto", "scala fatta", "doppio progetto", "progetto colore", "scala bilaterale",
              "gutshot", "progetto + coppia", "coppia o meglio", "nulla")
BOARD_PLANE = 4
SMALL_BINS = (2, 3, 4, 5)  # min-raise, 25, 33, 50% of the pot
STREET_START = 312  # the street one-hot after the card planes: preflop, flop, turn, river


def _straight(mask: int) -> bool:
    bits = (mask << 1) | ((mask >> 12) & 1)  # bit 0: the ace playing low
    run = bits & (bits >> 1) & (bits >> 2) & (bits >> 3) & (bits >> 4)
    return run != 0


def category(hole: tuple[int, int], board: list[int]) -> str:
    """The class of a two-card hand on a board (indices `suit * 13 + rank - 2`)."""
    cards = list(hole) + board
    suits = [c // 13 for c in cards]
    ranks = [c % 13 for c in cards]
    for suit in range(4):
        if suits.count(suit) >= 5 and (hole[0] // 13 == suit or hole[1] // 13 == suit):
            return "colore fatto"
    mask = 0
    for r in ranks:
        mask |= 1 << r
    board_mask = 0
    for c in board:
        board_mask |= 1 << (c % 13)
    if _straight(mask) and not _straight(board_mask):
        return "scala fatta"
    flush_draw = len(board) < 5 and any(
        suits.count(s) == 4 and (hole[0] // 13 == s or hole[1] // 13 == s) for s in range(4))
    outs = 0
    if len(board) < 5:
        for r in range(13):
            if not mask >> r & 1 and _straight(mask | 1 << r) and not _straight(board_mask | 1 << r):
                outs += 1
    board_ranks = {c % 13 for c in board}
    paired = hole[0] % 13 == hole[1] % 13 or hole[0] % 13 in board_ranks or hole[1] % 13 in board_ranks
    drawing = flush_draw or outs > 0
    if paired:
        return "progetto + coppia" if drawing else "coppia o meglio"
    if flush_draw and outs:
        return "doppio progetto"
    if flush_draw:
        return "progetto colore"
    if outs >= 2:
        return "scala bilaterale"
    if outs == 1:
        return "gutshot"
    return "nulla"


def groups(probabilities: np.ndarray, facing: np.ndarray) -> np.ndarray:
    """`(n, 3)`: fold/call/raise facing a bet, check/small/big checked to."""
    p = probabilities
    small = p[:, list(SMALL_BINS)].sum(1)
    big = p[:, 6:ALL_IN_BIN + 1].sum(1)
    facing_rows = np.stack([p[:, FOLD_BIN], p[:, CHECK_CALL_BIN], small + big], 1)
    checked_rows = np.stack([p[:, CHECK_CALL_BIN], small, big], 1)
    return np.where(facing[:, None], facing_rows, checked_rows)


def load(directory: Path) -> dict:
    from clone_train import load as load_clone

    return load_clone(directory)


def planes(features: np.ndarray, plane: int) -> list[list[int]]:
    block = features[:, plane * 52:(plane + 1) * 52]
    return [list(np.flatnonzero(row)) for row in block]


@torch.no_grad()
def probabilities(model, features: np.ndarray, masks: np.ndarray, batch: int = 4096) -> np.ndarray:
    out = []
    for start in range(0, len(features), batch):
        f = torch.from_numpy(features[start:start + batch].astype(np.float32))
        m = torch.from_numpy(masks[start:start + batch])
        out.append(torch.softmax(model.policy_logits(f, m), dim=-1).numpy())
    return np.concatenate(out)


def table(title: str, rows: dict, names: list[str]) -> list[str]:
    lines = [title]
    for facing, heads in ((True, ("fold", "call", "raise")), (False, ("check", "piccola", "grande"))):
        lines.append(("  di fronte a una puntata" if facing else "  a cui si e' fatto check")
                     + f"   {'':18s}" + "".join(f"{n[:14]:>24s}" for n in names))
        lines.append(f"  {'categoria':20s} {'n':>7s}" + "".join(f"{'  '.join(h[:5] for h in heads):>24s}" for _ in names))
        for cat in CATEGORIES:
            found = rows.get((facing, cat))
            if not found or found["n"] < 30:
                continue
            cells = "".join("{:>24s}".format("  ".join(f"{100 * v:4.0f}" for v in found[name])) for name in names)
            lines.append(f"  {cat:20s} {found['n']:7d}" + cells)
    return lines


def main(argv: list[str] | None = None) -> int:
    from duel_power import resolve_model

    from pokerlab.rl.ppo import build_model_from_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", required=True, help="#N, labels or checkpoint paths")
    parser.add_argument("--pluribus", type=Path, default=Path("checkpoints/studies/human_clone/data-pluribus"))
    parser.add_argument("--handhq", type=Path, default=Path("checkpoints/studies/human_clone/data"))
    parser.add_argument("--states", type=int, default=6000, help="HandHQ spots (flop and turn)")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    rng = np.random.default_rng(args.seed)
    models, names = [], []
    for spec in args.models:
        label, path, _ = resolve_model(spec, global_dir=Path("checkpoints/global"), root=Path("checkpoints"))
        models.append(build_model_from_checkpoint(path)[0])
        names.append("clone" if "clone" in str(path) else label.split("-agent")[0][-14:])

    # ---- Pluribus: the real actions, and every model in the same spot with the same cards
    data = load(args.pluribus)
    street = data["street"]
    keep = np.flatnonzero((street == 1) | (street == 2))
    features, masks = data["features"][keep], data["mask"][keep]
    groups_of = data["group"][keep]
    hole = data["hole"][groups_of]
    boards = planes(features, BOARD_PLANE)
    cats = [category((int(h[0]), int(h[1])), b) for h, b in zip(hole, boards, strict=True)]
    facing = masks[:, FOLD_BIN]
    actual = np.zeros((len(keep), masks.shape[1]))
    actual[np.arange(len(keep)), data["action"][keep].astype(int)] = 1.0
    who = np.array(["bot" if name.endswith("|Pluribus") else "pro" for name in data["player"][groups_of]])
    predicted = {name: groups(probabilities(m, features, masks), facing) for name, m in zip(names, models, strict=True)}
    real = groups(actual, facing)
    for kind, label in (("pro", "professionisti"), ("bot", "Pluribus (bot)")):
        rows = {}
        for f in (True, False):
            for cat in CATEGORIES:
                sel = (who == kind) & (facing == f) & (np.array(cats) == cat)
                if sel.sum():
                    rows[(f, cat)] = {"n": int(sel.sum()), label: real[sel].mean(0),
                                      **{n: predicted[n][sel].mean(0) for n in names}}
        for line in table(f"\nPLURIBUS, le decisioni di {label} (flop e turn, %): reale, poi ogni modello sugli stessi spot",
                          rows, [label, *names]):
            print(line)

    # ---- HandHQ spots, every category given to every model
    data = load(args.handhq)
    street = data["street"]
    pool = np.flatnonzero((street == 1) | (street == 2))
    chosen = rng.choice(pool, size=min(args.states, len(pool)), replace=False)
    rows_f, rows_m, rows_c, rows_facing = [], [], [], []
    for i in chosen:
        f = data["features"][i].astype(np.float32)
        board = list(np.flatnonzero(f[BOARD_PLANE * 52:(BOARD_PLANE + 1) * 52]))
        free = [c for c in range(52) if c not in board]
        by_cat: dict[str, list] = {}
        for a in range(len(free)):
            for b in range(a + 1, len(free)):
                by_cat.setdefault(category((free[a], free[b]), board), []).append((free[a], free[b]))
        for cat, hands in by_cat.items():
            for k in rng.permutation(len(hands))[: args.per_category]:
                row = f.copy()
                row[:52] = 0.0
                row[list(hands[k])] = 1.0
                rows_f.append(row)
                rows_m.append(data["mask"][i])
                rows_c.append(cat)
                rows_facing.append(bool(data["mask"][i][FOLD_BIN]))
    features, masks = np.stack(rows_f), np.stack(rows_m)
    facing, cats = np.array(rows_facing), np.array(rows_c)
    predicted = {name: groups(probabilities(m, features, masks), facing) for name, m in zip(names, models, strict=True)}
    rows = {}
    for f in (True, False):
        for cat in CATEGORIES:
            sel = (facing == f) & (cats == cat)
            if sel.sum():
                rows[(f, cat)] = {"n": int(sel.sum()), **{n: predicted[n][sel].mean(0) for n in names}}
    for line in table(f"\nSPOT REALI DI HANDHQ ({len(chosen)}, flop e turn), ogni categoria data a ogni modello (%)", rows, names):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
