"""Trains the equity network on endless fresh deals and measures it on a fixed set.

    PYTHONPATH=src:studies/equity_net python studies/equity_net/train.py

The defaults are for a long run on a GPU (`--device auto` uses it when there is one). The
deals are drawn by worker processes on the CPU, and that is the limit: the network is tiny
next to the cost of scoring a deal with the evaluator, so on a GPU box the CPU cores
decide how fast it goes (the run prints the rate). A run is saved at every evaluation and
`--resume` continues it, learning-rate schedule included.

The defaults are the network meant for the long run: attention across the players, two
residual blocks in the player encoder, a moving average of the weights, the suits in canonical
order. The original network is still available through `--arch sets --blocks 0 --ema 0`.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from pathlib import Path

import torch
from data import DEFAULT_VAL_DEALS, DEFAULT_VAL_RUNOUTS, stream, validation_path, validation_set
from metrics import evaluate_model
from model import EquityNet

RUNS = Path(__file__).parent / "runs"


def device_of(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def learning_rate(step: int, steps: int, base: float, warmup: int, floor: float = 0.02) -> float:
    """Linear warm-up, then a cosine down to `floor` of the base rate."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=4096, help="new deals per step")
    parser.add_argument("--lr", type=float, default=1e-3, help="peak rate, after the warm-up")
    parser.add_argument("--warmup", type=int, default=500, help="steps of linear warm-up")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient norm clip (0 = off)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decay (0 = plain Adam)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--arch", choices=("sets", "attn"), default="attn", help="sum over players, or attention")
    parser.add_argument("--blocks", type=int, default=4, help="residual blocks in the player encoder (0 = two layers)")
    parser.add_argument("--layers", type=int, default=4, help="attention layers (arch attn)")
    parser.add_argument("--heads", type=int, default=8, help="attention heads (arch attn)")
    parser.add_argument("--ema", type=float, default=0.999,
                        help="decay of a moving average of the weights, which is what is measured and saved (0 = off)")
    parser.add_argument("--loss", choices=("logscore", "brier"), default="logscore",
                        help="log-likelihood of the realised shares, or their squared error")
    parser.add_argument("--device", default="auto", help="cpu, cuda or auto (the GPU if there is one)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 6) - 6),
                        help="processes drawing new deals (default: the cores, less six)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads of the trainer")
    parser.add_argument("--street-weights", type=float, nargs=4, default=(4.0, 1.0, 1.0, 1.0))
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--val-deals", type=int, default=DEFAULT_VAL_DEALS)
    parser.add_argument("--val-runouts", type=int, default=DEFAULT_VAL_RUNOUTS, help="preflop completions per validation deal")
    parser.add_argument("--val-jobs", type=int, default=max(1, (os.cpu_count() or 6) - 6))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RUNS / "model.pt")
    parser.add_argument("--resume", action="store_true", help="continue from --out")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = device_of(args.device)
    weights = tuple(args.street_weights)

    started = time.time()
    val = validation_set(validation_path(RUNS, args.val_deals, args.val_runouts), args.val_deals, args.val_runouts, args.val_jobs)
    print(f"validazione: {len(val[0])} mani, equity esatta dal flop ({args.val_runouts} estrazioni preflop) "
          f"({time.time() - started:.0f}s)", flush=True)
    val = tuple(t.to(device) for t in val)

    model = EquityNet(
        args.hidden, args.latent, args.arch, args.blocks, args.layers, args.heads
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = copy.deepcopy(model).requires_grad_(False) if args.ema > 0 else None
    done = 0
    if args.resume and args.out.exists():
        saved = torch.load(args.out, map_location=device, weights_only=True)
        model = EquityNet.from_checkpoint(saved).to(device)
        # with a moving average, "state" is the average and "train_state" the raw weights
        model.load_state_dict(saved.get("train_state", saved["state"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(saved["optimizer"])
        if ema is not None:
            ema = copy.deepcopy(model).requires_grad_(False)
            ema.load_state_dict(saved["state"])
        done = saved["step"]
        print(f"ripreso da {args.out} al passo {done}", flush=True)
    print(f"dispositivo {device}, parametri {sum(p.numel() for p in model.parameters()):,}, "
          f"{args.workers} processi per le mani, {args.batch_size} mani a passo, "
          f"{args.steps} passi = {args.steps * args.batch_size / 1e6:.0f} milioni di mani", flush=True)

    measured = ema if ema is not None else model  # what is evaluated and saved

    def save(step: int) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial = args.out.with_suffix(".partial")
        saved = {"state": measured.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                 "config": model.config()}
        if ema is not None:
            saved["train_state"] = model.state_dict()
        torch.save(saved, partial)
        partial.replace(args.out)

    print("inizio:", {k: round(v, 4) for k, v in evaluate_model(measured, val).items()}, flush=True)
    started = time.time()
    seen = 0
    running = 0.0
    window = 0
    # A resumed run draws different deals from the first one: the seed moves with `done`.
    loader = stream(args.batch_size, args.workers, args.seed + done, weights)
    for step, batch in enumerate(loader, start=done + 1):
        holes, board, mask, shares = (t.to(device, non_blocking=True) for t in batch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup)
        if args.loss == "logscore":
            loss = -(shares * model.log_shares(holes, board, mask) * mask).sum(-1).mean()
        else:
            loss = (((model(holes, board, mask) - shares) ** 2) * mask).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if ema is not None:
            # the average starts quickly (a short memory at first) and settles at `--ema`
            decay = min(args.ema, (1 + step) / (10 + step))
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
        running += loss.item()
        window += 1
        seen += args.batch_size
        if step % args.eval_every == 0 or step >= args.steps:
            metrics = evaluate_model(measured, val)
            print(f"passo {step:7d}  loss {running / window:.4f}  {seen / (time.time() - started):.0f} mani/s  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                  + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()), flush=True)
            running, window = 0.0, 0
            save(step)
        if step >= args.steps:
            break
    print("salvato", args.out)


if __name__ == "__main__":
    main()
