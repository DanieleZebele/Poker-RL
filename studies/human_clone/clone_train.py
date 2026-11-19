"""A model of ours taught to play like the online players: behaviour cloning with hidden cards.

    PYTHONPATH=src:studies/agents:studies/human_clone python studies/human_clone/clone_train.py \\
        --data checkpoints/studies/human_clone/data --model '#1' --out checkpoints/studies/human_clone/run

The network is a store model (`--model`, the #1), its policy (trunk and head) trained to give the
actions the players took (`clone_generate.py`); the card encoder stays frozen. **Most of the cards
were never seen**: a player's two cards are known only when it showed them. Training on the shown
hands alone would teach a policy that never folds (a hand is shown because it was played to the
end). So the loss is the likelihood of everything a player did in a hand, **marginal over the cards
it could have held**: `log mean_c prod_t pi(a_t | c)` over `--candidates` two-card hands drawn
uniformly from those it could hold (not on the final board, not shown by another), the exact cards
when it showed them. That is the maximum likelihood of the observed actions, unbiased by which hands
reach a showdown (up to the sampling of the candidates).

**A clone of one profile** (`--profile K`, the players `human_imitation.py profiles` put in profile K,
`--profiles`): the same training on that profile's players only, train and test, usually from the
general clone (`--model checkpoints/studies/human_clone/run/clone.pt`) with fewer steps.

The report, on held-out sessions (every tenth): the loss per decision of the clone and of the model
it started from, overall, per street, and on the decisions whose cards are known (there the
likelihood is exact), with how often the likeliest bin is the one taken. The clone is saved as a
checkpoint every evaluation (`clone.pt`), loadable like any model of the store (never published).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HOLE_PLANE_END = 52
COMBOS = torch.tensor(list(itertools.combinations(range(52), 2)))  # (1326, 2)
HELD_OUT_EVERY = 10
STREETS = ("preflop", "flop", "turn", "river")


def load(directory: Path) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p)) for p in sorted(directory.glob("clone-*.npz"))]
    parts = [p for p in parts if len(p["action"])]
    out, group_base = {}, 0
    for p in parts:
        p["group"] = p["group"].astype(np.int64) + group_base
        group_base += len(p["hole"])
    for key in parts[0]:
        out[key] = np.concatenate([p[key] for p in parts])
    order = np.argsort(out["group"], kind="stable")
    for key in ("features", "mask", "action", "group", "street", "players"):
        out[key] = out[key][order]
    out["group_start"] = np.searchsorted(out["group"], np.arange(len(out["hole"]) + 1))
    return out


class Batcher:
    """Groups (one player in one hand) and all their decisions, with candidate cards."""

    def __init__(self, data: dict, device: str, candidates: int, keep: np.ndarray | None = None) -> None:
        self.data, self.device, self.k = data, device, candidates
        self.features = torch.from_numpy(data["features"])  # CPU, float16
        self.mask = torch.from_numpy(data["mask"])
        self.action = torch.from_numpy(data["action"].astype(np.int64))
        self.street = torch.from_numpy(data["street"].astype(np.int64))
        self.start = data["group_start"]
        self.hole = torch.from_numpy(data["hole"].astype(np.int64))
        excluded = torch.from_numpy(data["excluded"].astype(np.int64))
        self.excluded = excluded
        test = data["session"] % HELD_OUT_EVERY == 0
        keep = np.ones(len(test), dtype=bool) if keep is None else keep
        self.train_groups = np.flatnonzero(~test & keep)
        self.test_groups = np.flatnonzero(test & keep)
        self.combos = COMBOS.to(device)

    def batch(self, groups: np.ndarray, candidates: int | None = None):
        k = candidates or self.k
        lengths = self.start[groups + 1] - self.start[groups]
        index = np.concatenate([np.arange(self.start[g], self.start[g + 1]) for g in groups])
        local = np.repeat(np.arange(len(groups)), lengths)
        dev = self.device
        features = self.features[index].to(dev, non_blocking=True).float()
        mask = self.mask[index].to(dev)
        action = self.action[index].to(dev)
        street = self.street[index].to(dev)
        local_t = torch.from_numpy(local).to(dev)
        hole = self.hole[groups].to(dev)
        known = hole[:, 0] >= 0
        # the candidates: the cards shown, else k two-card hands it could hold, drawn without replacement
        blocked = torch.zeros(len(groups), 53, dtype=torch.bool, device=dev)
        blocked.scatter_(1, torch.where(self.excluded[groups] < 0, 52, self.excluded[groups]).to(dev), True)
        allowed = ~(blocked[:, :52][:, self.combos[:, 0]] | blocked[:, :52][:, self.combos[:, 1]])
        drawn = torch.multinomial(allowed.float(), k, replacement=False)  # (G, k) combo indices
        cards = self.combos[drawn]  # (G, k, 2)
        cards[known, 0] = hole[known]
        valid = torch.ones(len(groups), k, dtype=torch.bool, device=dev)
        valid[known, 1:] = False
        return {"features": features, "mask": mask, "action": action, "street": street, "local": local_t,
                "cards": cards, "valid": valid, "known": known, "groups": len(groups)}


def group_log_likelihood(model, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per group, `log mean_c prod_t pi(a_t | c)` over its valid candidates; and per decision, the
    log-probability of the action under the first candidate (the true cards where known)."""
    d, k = len(b["action"]), b["cards"].shape[1]
    features = b["features"].unsqueeze(1).expand(d, k, -1).clone()  # (D, k, F)
    hole = torch.zeros(d, k, HOLE_PLANE_END, device=features.device)
    cards = b["cards"][b["local"]]  # (D, k, 2)
    hole.scatter_(2, cards, 1.0)
    features[:, :, :HOLE_PLANE_END] = hole
    logits = model.policy_logits(features.reshape(d * k, -1), b["mask"].repeat_interleave(k, dim=0)).reshape(d, k, -1)
    log_p = torch.log_softmax(logits, dim=-1).gather(2, b["action"].view(d, 1, 1).expand(d, k, 1)).squeeze(2)
    per_group = torch.zeros(b["groups"], k, device=features.device).index_add_(0, b["local"], log_p)
    per_group = per_group.masked_fill(~b["valid"], -math.inf)
    count = b["valid"].sum(1).float()
    return torch.logsumexp(per_group, dim=1) - count.log(), log_p[:, 0]


@torch.no_grad()
def evaluate(model, batcher: Batcher, groups: np.ndarray, candidates: int, chunk: int = 256) -> dict:
    model.eval()
    total, decisions, by_street = 0.0, 0, {}
    known_lp, known_hit, known_n = 0.0, 0, 0
    for start in range(0, len(groups), chunk):
        b = batcher.batch(groups[start : start + chunk], candidates)
        group_ll, first = group_log_likelihood(model, b)
        total -= float(group_ll.sum())
        decisions += len(b["action"])
        known_d = b["known"][b["local"]]
        if known_d.any():
            known_lp -= float(first[known_d].sum())
            known_n += int(known_d.sum())
            d = len(b["action"])
            features = b["features"][known_d].clone()
            hole = torch.zeros(int(known_d.sum()), HOLE_PLANE_END, device=features.device)
            hole.scatter_(1, b["cards"][b["local"]][known_d, 0], 1.0)
            features[:, :HOLE_PLANE_END] = hole
            logits = model.policy_logits(features, b["mask"][known_d])
            known_hit += int((logits.argmax(1) == b["action"][known_d]).sum())
            del d
        for i, name in enumerate(STREETS):
            on = b["street"] == i
            by_street.setdefault(name, [0, 0])
            by_street[name][1] += int(on.sum())
        # per street: the share of the group's loss is not separable; report the known decisions' instead
    model.train()
    return {"nll_per_decision": total / max(decisions, 1), "decisions": decisions,
            "known_nll": known_lp / max(known_n, 1), "known_top1": known_hit / max(known_n, 1), "known": known_n}


def main(argv: list[str] | None = None) -> int:
    from duel_power import resolve_model

    from pokerlab.rl.ppo import build_model_from_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("checkpoints/studies/human_clone/data"))
    parser.add_argument("--model", default="#1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--groups", type=int, default=384, help="groups (a player in a hand) per step")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--eval-candidates", type=int, default=256)
    parser.add_argument("--eval-groups", type=int, default=40000)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--profile", type=int, default=None, help="only the players of this profile")
    parser.add_argument("--profiles", type=Path, default=Path("checkpoints/studies/human_imitation/profiles.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = load(args.data)
    keep = None
    if args.profile is not None:
        profile_of = json.loads(args.profiles.read_text())["profile_of"]
        keep = np.array([profile_of.get(name, -1) == args.profile for name in data["player"].tolist()])
    batcher = Batcher(data, args.device, args.candidates, keep)
    label, path, _ = resolve_model(args.model, global_dir=Path("checkpoints/global"), root=Path("checkpoints"))
    model, checkpoint = build_model_from_checkpoint(path, device=args.device)
    trainable = [*model.trunk.parameters(), *model.policy_head.parameters()]
    for p in trainable:
        p.requires_grad_(True)
    model.train()
    test = rng.choice(batcher.test_groups, size=min(args.eval_groups, len(batcher.test_groups)), replace=False)
    print(f"{label}: {len(data['action'])} decisions, {len(batcher.train_groups)} training groups, "
          f"{len(batcher.test_groups)} test ({time.time() - started:.0f}s)", flush=True)
    history = [{"step": 0, **evaluate(model, batcher, test, args.eval_candidates)}]
    print("step 0", history[-1], flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    running = None
    for step in range(1, args.steps + 1):
        groups = rng.choice(batcher.train_groups, size=args.groups, replace=False)
        b = batcher.batch(groups)
        group_ll, _ = group_log_likelihood(model, b)
        loss = -group_ll.sum() / len(b["action"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        value = loss.item()
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % 250 == 0:
            print(f"step {step:6d} loss {running:.4f} {time.time() - started:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            history.append({"step": step, **evaluate(model, batcher, test, args.eval_candidates)})
            print(f"step {step}", history[-1], flush=True)
            saved = dict(checkpoint)
            saved["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            saved["metadata"] = {**checkpoint.get("metadata", {}), "human_clone_of": label, "human_clone_step": step,
                                 "human_clone_profile": args.profile}
            torch.save(saved, args.out / "clone.pt")
            (args.out / "result.json").write_text(json.dumps({"model": label, "args": {k: str(v) for k, v in vars(args).items()},
                                                              "history": history}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
