"""Train the belief network on the arrays of `belief_data.py` and report how much it knows.

    PYTHONPATH=src:studies/belief python studies/belief/belief_train.py --data checkpoints/studies/belief/data \\
        --arm full --out checkpoints/studies/belief/runs/full

The question it answers: how much better than "any two cards" does the network place the target's
real hand, from public information alone? The loss is the cross-entropy of the true combo; the
report gives it by street and by players still in, next to the uniform range over the combos the
observer cannot rule out (`log` of their count), the gain in bits, the size of the range the
network believes (`exp` of its entropy, in combos) and where the true combo ranks in it (0 = the
network's likeliest, 0.5 = no better than chance).

Two test sets, never trained on: the hands of every tenth session (`session % 10 == 0`: players the
network has seen elsewhere), and every sample whose target is one of the held-out models (a tenth of
the models, by a hash of the label; those models are never a target in training), which is how a
player never met reads.

Arms (`--arm`): `full`; `no-tokens` (no action of the hand: what position, statistics, board and
"still in" alone give); `no-hud` (no opponent statistics). The suits of every sample are permuted at
random in training (the input is suit-symmetric, the board planes are not by themselves).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from belief_data import load
from belief_model import BeliefNet, build_encoder, visible_board
from belief_records import (
    ACTION_CLASSES,
    COMBO_CARDS,
    HUD_DIM,
    MAX_SEATS,
    POSITIONS,
    STACK_SCALE,
    STREETS,
)
from torch.nn import functional

HELD_OUT_EVERY = 10
BLIND_COLUMN = len(STREETS) + len(POSITIONS) + ACTION_CLASSES.index("blind")


def held_out_model(label: str) -> bool:
    return zlib.crc32(label.encode()) % HELD_OUT_EVERY == 0


class Data:
    """Every array on the device, and batches built there."""

    def __init__(self, arrays: dict[str, np.ndarray], device: str) -> None:
        def put(key, dtype=None):
            value = torch.from_numpy(np.ascontiguousarray(arrays[key]))
            return value.to(device=device, dtype=dtype) if dtype else value.to(device)

        self.device = device
        self.tokens = put("tokens", torch.float16)  # half the memory; made float32 per batch
        self.token_actor = put("token_actor", torch.long)
        self.offset = put("token_offset", torch.long)
        self.board = put("board", torch.long)
        self.holes = put("holes", torch.long)
        self.position = put("position", torch.long)
        self.stack_bb = put("stack_bb", torch.float32)
        self.hud = put("hud", torch.float32) if "hud" in arrays else None
        self.s_hand = put("s_hand", torch.long)
        self.s_prefix = put("s_prefix", torch.long)
        self.s_target = put("s_target", torch.long)
        self.s_observer = put("s_observer", torch.long)
        self.s_street = put("s_street", torch.long)
        self.s_live = put("s_live", torch.long)
        pair = torch.full((52, 52), -1, dtype=torch.long)
        for index, (a, b) in enumerate(COMBO_CARDS):
            pair[a, b] = pair[b, a] = index
        self.pair_to_combo = pair.to(device)
        session = torch.from_numpy(arrays["session"]).to(device)
        target_model = torch.from_numpy(arrays["seat_model"]).to(device)[self.s_hand, self.s_target]
        unseen_models = torch.tensor([held_out_model(m) for m in arrays["models"].tolist()], device=device)
        unseen = unseen_models[target_model]
        test_session = session[self.s_hand] % HELD_OUT_EVERY == 0
        everything = torch.arange(len(self.s_hand), device=device)
        self.train = everything[~unseen & ~test_session]
        self.test_seen = everything[~unseen & test_session]
        self.test_unseen = everything[unseen]

    def batch(self, index: torch.Tensor, augment: bool, condition: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """`condition` `(B,)`: 1 where the target's cards are to be read as known because it showed
        them at a showdown; the static's last column (zeros if not given)."""
        hand, target, observer = self.s_hand[index], self.s_target[index], self.s_observer[index]
        street, prefix = self.s_street[index], self.s_prefix[index]
        length = int(prefix.max())
        steps = torch.arange(length, device=self.device)
        where = (self.offset[hand].unsqueeze(1) + steps).clamp_max(len(self.tokens) - 1)
        valid = (steps.unsqueeze(0) < prefix.unsqueeze(1)).float()
        own = (self.token_actor[where] == target.unsqueeze(1)).float() * valid
        tokens = torch.cat([self.tokens[where].float() * valid.unsqueeze(-1), own.unsqueeze(-1)], dim=-1)
        players = (self.position[hand] >= 0).sum(dim=1).float()
        hud = self.hud[hand, target] if self.hud is not None else torch.zeros(len(index), HUD_DIM, device=self.device)
        if condition is None:
            condition = torch.zeros(len(index), device=self.device)
        static = torch.cat([
            functional.one_hot(self.position[hand, target], len(POSITIONS)).float(),
            (players / MAX_SEATS).unsqueeze(1),
            (torch.log1p(self.stack_bb[hand, target]) / STACK_SCALE).unsqueeze(1),
            hud,
            functional.one_hot(street, len(STREETS)).float(),
            (self.s_live[index].float() / MAX_SEATS).unsqueeze(1),
            condition.float().unsqueeze(1),
        ], dim=1)
        board = visible_board(self.board[hand], street)
        mine = self.holes[hand, observer]
        theirs = self.holes[hand, target]
        if augment:
            perm = torch.argsort(torch.rand(len(index), 4, device=self.device), dim=1)

            def remap(cards):
                suits = perm.gather(1, (cards.clamp_min(0) // 13))
                return torch.where(cards < 0, cards, suits * 13 + cards % 13)

            board, mine, theirs = remap(board), remap(mine), remap(theirs)
        return {
            "tokens": tokens, "lengths": prefix, "static": static, "board": board,
            "known": torch.cat([board, mine], dim=1),
            "target": self.pair_to_combo[theirs[:, 0], theirs[:, 1]],
            "street": street, "live": self.s_live[index],
            # the target has already chosen something in this hand (a blind is not a choice)
            "acted": ((own * (1.0 - self.tokens[where][..., BLIND_COLUMN].float())).sum(dim=1) > 0),
        }


def run(model: BeliefNet, batch: dict) -> torch.Tensor:
    logits, _phi, _inverse = model(batch["tokens"], batch["lengths"], batch["static"], batch["board"], batch["known"])
    return logits


@torch.no_grad()
def evaluate(model: BeliefNet, data: Data, index: torch.Tensor, batch_size: int, limit: int) -> dict:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(0)
    if len(index) > limit:
        index = index[torch.randperm(len(index), generator=generator)[:limit].to(index.device)]
    rows = {"loss": [], "uniform": [], "entropy": [], "rank": [], "street": [], "live": [], "acted": []}
    for start in range(0, len(index), batch_size):
        batch = data.batch(index[start : start + batch_size], augment=False)
        logits = run(model, batch)
        log_p = torch.log_softmax(logits, dim=-1)
        loss = -log_p.gather(1, batch["target"].unsqueeze(1)).squeeze(1)
        allowed = logits > -1e8
        count = allowed.sum(dim=1).float()
        p = log_p.exp()
        entropy = -(torch.where(allowed, p * log_p, torch.zeros_like(p))).sum(dim=1)
        better = ((log_p > (-loss).unsqueeze(1)) & allowed).sum(dim=1).float()
        rows["loss"].append(loss)
        rows["uniform"].append(torch.log(count))
        rows["entropy"].append(entropy)
        rows["rank"].append(better / count)
        rows["street"].append(batch["street"])
        rows["live"].append(batch["live"])
        rows["acted"].append(batch["acted"])
    model.train()
    cat = {key: torch.cat(value).float().cpu() for key, value in rows.items()}

    def summary(mask) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        loss, uniform = float(cat["loss"][mask].mean()), float(cat["uniform"][mask].mean())
        return {"n": n, "loss": loss, "uniform": uniform, "gain_bits": (uniform - loss) / math.log(2),
                "range": float(cat["entropy"][mask].exp().mean()), "rank": float(cat["rank"][mask].mean())}

    every = torch.ones_like(cat["loss"], dtype=torch.bool)
    report = {"all": summary(every)}
    for i, street in enumerate(STREETS):
        report[street.value] = summary(cat["street"] == i)
    report["ha agito"] = summary(cat["acted"] > 0)
    report["non ancora"] = summary(cat["acted"] == 0)
    for name, low, high in (("2 in", 2, 2), ("3-4 in", 3, 4), ("5+ in", 5, 9)):
        report[name] = summary((cat["live"] >= low) & (cat["live"] <= high))
    return report


def format_report(name: str, report: dict) -> str:
    lines = [f"{name}: {'gruppo':8s} {'n':>8s} {'CE':>6s} {'unif':>6s} {'bit':>5s} {'range':>6s} {'rank':>5s}"]
    for group, row in report.items():
        if row.get("n"):
            lines.append(f"{'':{len(name)}s}  {group:8s} {row['n']:8d} {row['loss']:6.3f} {row['uniform']:6.3f} "
                         f"{row['gain_bits']:5.2f} {row['range']:6.0f} {row['rank']:5.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("checkpoints/studies/belief/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("full", "no-tokens", "no-hud"), default="full")
    parser.add_argument("--equity-model", default="equity_evaluator/distill-lat32_h256_h1.pt")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gru", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    data = Data(load(args.data), args.device)
    print(f"{len(data.s_hand)} samples: train {len(data.train)}, test seen {len(data.test_seen)}, "
          f"test unseen models {len(data.test_unseen)} ({time.time() - started:.0f}s to load)", flush=True)
    model = BeliefNet(build_encoder(args.equity_model), hidden=args.hidden, gru=args.gru, rank=args.rank,
                      use_tokens=args.arm != "no-tokens", use_hud=args.arm != "no-hud").to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    print(f"arm {args.arm}: {sum(p.numel() for p in trainable)} trainable parameters", flush=True)

    history = []
    running = 0.0
    for step in range(1, args.steps + 1):
        index = data.train[torch.randint(len(data.train), (args.batch_size,), device=args.device)]
        batch = data.batch(index, augment=True)
        loss = functional.cross_entropy(run(model, batch), batch["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0:
            print(f"step {step:6d}  loss {running:.4f}  {time.time() - started:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            seen = evaluate(model, data, data.test_seen, 4096, args.eval_samples)
            history.append({"step": step, "seen": seen})
            print(format_report(f"step {step} test", seen), flush=True)
    unseen = evaluate(model, data, data.test_unseen, 4096, args.eval_samples)
    print(format_report("modelli mai visti", unseen), flush=True)
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "history": history,
              "seen": history[-1]["seen"], "unseen": unseen}
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    torch.save({"config": model.config, "state": {k: v for k, v in model.state_dict().items()
                                                  if not k.startswith("encoder.")},
                "equity_model": args.equity_model}, args.out / "belief.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
