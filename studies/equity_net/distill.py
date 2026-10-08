"""Trains a small equity network to copy a large one (distillation), on equities and not on wins.

    PYTHONPATH=src:studies/equity_net python studies/equity_net/distill.py \
        --teacher studies/equity_net/runs/big.pt --out studies/equity_net/runs/small.pt \
        --arch attn --hidden 128 --latent 128 --blocks 1 --layers 1

The label of a deal is what the *teacher* says each player's share is: a full probability
vector, not the single winner of one random completion of the board. Two consequences:

- The labels carry no noise from the cards still to come, so the gradient is much cleaner than
  in `train.py` and the student learns the same function in far fewer deals.
- Nothing here scores a hand with the evaluator. The deals are only cards, drawn in one tensor
  operation on the training device, so there are no worker processes and the speed is that of
  the teacher's forward pass (the evaluator, which limited `train.py`, is not involved).

The student cannot be better than its teacher: it learns the teacher's errors too. The teacher's
own error on the validation set is printed first, as the floor to compare the student with.

The student is saved in the same format as `train.py`'s, so `evaluate.py`, `analyze.py` and
`--resume` work on it unchanged. `--student-from` starts from an existing checkpoint instead
of a fresh network (its own configuration, so the network flags are then ignored).
"""

from __future__ import annotations

import argparse
import copy
import time
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
from deals import MAX_PLAYERS, MIN_PLAYERS, STREETS
from evaluate import load
from metrics import evaluate_model, predict
from model import CARDS, EquityNet
from train import device_of, learning_rate

RUNS = Path(__file__).parent / "runs"


def random_deals(
    batch: int, street_weights: torch.Tensor, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`batch` new deals as `(holes, board, mask)`: the number of players uniform in 2..9, every
    card distinct and uniform, and the number of visible board cards (0, 3, 4 or 5) drawn with
    `street_weights`. The same distribution as `deals.sample_deal`, with no result attached."""
    players = torch.randint(MIN_PLAYERS, MAX_PLAYERS + 1, (batch,), device=device, generator=generator)
    order = torch.rand(batch, CARDS, device=device, generator=generator).argsort(dim=1)  # a shuffle per row
    hole_cards = order[:, : 2 * MAX_PLAYERS].view(batch, MAX_PLAYERS, 2)
    board_cards = order[:, 2 * MAX_PLAYERS : 2 * MAX_PLAYERS + 5]
    mask = torch.arange(MAX_PLAYERS, device=device)[None] < players[:, None]
    visible = torch.tensor(STREETS, device=device)[torch.multinomial(street_weights, batch, True, generator=generator)]
    shown = torch.arange(5, device=device)[None] < visible[:, None]
    holes = torch.zeros(batch, MAX_PLAYERS, CARDS, device=device).scatter_(2, hole_cards, 1.0) * mask.unsqueeze(-1)
    board = torch.zeros(batch, CARDS, device=device).scatter_(1, board_cards, shown.float())
    return holes, board, mask


def hard_errors(model: EquityNet, hard, tags) -> list[float]:
    """Mean absolute error on each kind of hard case."""
    _h, _b, mask, equity, _v = hard
    pred = predict(model, hard)
    return [
        float((pred - equity).abs()[torch.from_numpy(tags == t).to(mask.device).unsqueeze(-1) & mask].mean())
        for t in range(len(HARD_TAGS))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--teacher", type=Path, required=True, help="the large, trained network (a checkpoint)")
    parser.add_argument("--out", type=Path, default=RUNS / "distilled.pt")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=4096, help="new deals per step")
    parser.add_argument("--lr", type=float, default=1e-3, help="peak rate, after the warm-up")
    parser.add_argument("--warmup", type=int, default=500, help="steps of linear warm-up")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient norm clip (0 = off)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decay (0 = plain Adam)")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--arch", choices=("sets", "attn"), default="attn")
    parser.add_argument("--blocks", type=int, default=1, help="residual blocks in the player encoder")
    parser.add_argument("--layers", type=int, default=1, help="attention layers (arch attn)")
    parser.add_argument("--heads", type=int, default=2, help="attention heads (arch attn)")
    parser.add_argument("--ema", type=float, default=0.9999, help="decay of the moving average of the weights (0 = off)")
    parser.add_argument("--loss", choices=("kl", "l1", "mse"), default="kl",
                        help="how the student's shares are compared with the teacher's: KL divergence, mean absolute"
                             " error (the measured quantity) or squared error")
    parser.add_argument("--street-weights", type=float, nargs=4, default=(1.0, 1.0, 1.0, 1.0),
                        help="how often preflop, flop, turn and river are drawn (evaluating costs nothing now, so even)")
    parser.add_argument("--device", default="auto", help="cpu, cuda or auto (the GPU if there is one)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--val-deals", type=int, default=DEFAULT_VAL_DEALS)
    parser.add_argument("--val-runouts", type=int, default=DEFAULT_VAL_RUNOUTS)
    parser.add_argument("--hard-deals", type=int, default=DEFAULT_HARD_DEALS)
    parser.add_argument("--val-jobs", type=int, default=24, help="processes making the sets if they are missing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-from", type=Path, help="start from this checkpoint's network instead of a new one")
    parser.add_argument("--resume", action="store_true", help="continue from --out")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = device_of(args.device)

    val = tuple(t.to(device) for t in validation_set(
        validation_path(RUNS, args.val_deals, args.val_runouts), args.val_deals, args.val_runouts, args.val_jobs))
    hard, tags = hard_set(hard_path(RUNS, args.hard_deals), args.hard_deals, args.val_jobs)
    hard = tuple(t.to(device) for t in hard)

    teacher = load(args.teacher).to(device).requires_grad_(False)
    teacher_metrics = evaluate_model(teacher, val)
    print(f"maestro {args.teacher.name}: {sum(p.numel() for p in teacher.parameters()):,} parametri, errore "
          f"{teacher_metrics['mae']:.4f} (e' il limite dell'allievo), difficili "
          f"{sum(hard_errors(teacher, hard, tags)) / len(HARD_TAGS):.4f}", flush=True)

    if args.student_from is not None:
        saved = torch.load(args.student_from, map_location=device, weights_only=True)
        student = EquityNet.from_checkpoint(saved).to(device)
        student.load_state_dict(saved.get("train_state", saved["state"]))
    else:
        student = EquityNet(
            args.hidden, args.latent, args.arch, args.blocks, args.layers, args.heads).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = copy.deepcopy(student).requires_grad_(False) if args.ema > 0 else None
    done = 0
    if args.resume and args.out.exists():
        saved = torch.load(args.out, map_location=device, weights_only=True)
        student = EquityNet.from_checkpoint(saved).to(device)
        # with a moving average, "state" is the average and "train_state" the raw weights
        student.load_state_dict(saved.get("train_state", saved["state"]))
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(saved["optimizer"])
        if ema is not None:
            ema = copy.deepcopy(student).requires_grad_(False)
            ema.load_state_dict(saved["state"])
        done = saved["step"]
        print(f"ripreso da {args.out} al passo {done}", flush=True)
    measured = ema if ema is not None else student  # what is evaluated and saved
    print(f"dispositivo {device}, allievo {sum(p.numel() for p in student.parameters()):,} parametri "
          f"({student.config()}), {args.batch_size} mani a passo, {args.steps} passi = "
          f"{args.steps * args.batch_size / 1e6:.0f} milioni di mani", flush=True)

    def save(step: int) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial = args.out.with_suffix(".partial")
        state = {"state": measured.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                 "config": student.config()}
        if ema is not None:
            state["train_state"] = student.state_dict()
        torch.save(state, partial)
        partial.replace(args.out)

    print("inizio:", {k: round(v, 4) for k, v in evaluate_model(measured, val).items()}, flush=True)
    weights = torch.tensor(args.street_weights, device=device)
    # A resumed run draws different deals from the first one: the seed moves with `done`.
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + done)
    started = time.time()
    seen = 0
    running = kl_running = 0.0
    window = 0
    for step in range(done + 1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup)
        holes, board, mask = random_deals(args.batch_size, weights, device, generator)
        with torch.inference_mode():
            target = teacher(holes, board, mask)
        target = target.clone()  # out of inference mode, so it can sit in a loss
        pred = student(holes, board, mask)
        if args.loss == "kl":
            loss = -(target * torch.log(pred.clamp_min(1e-9)) * mask).sum(-1).mean()
        elif args.loss == "l1":
            loss = ((pred - target).abs() * mask).sum(-1).mean()
        else:
            loss = (((pred - target) ** 2) * mask).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)
        optimizer.step()
        if ema is not None:
            # the average starts quickly (a short memory at first) and settles at `--ema`
            decay = min(args.ema, (1 + step) / (10 + step))
            torch._foreach_lerp_(list(ema.parameters()), list(student.parameters()), 1 - decay)
        with torch.no_grad():  # the divergence from the teacher, whatever the loss: 0 means a perfect copy
            entropy = -(target * torch.log(target.clamp_min(1e-9)) * mask).sum(-1).mean()
            kl = -(target * torch.log(pred.detach().clamp_min(1e-9)) * mask).sum(-1).mean() - entropy
        running += loss.item()
        kl_running += kl.item()
        window += 1
        seen += args.batch_size
        if step % args.eval_every == 0 or step >= args.steps:
            metrics = evaluate_model(measured, val)
            print(f"passo {step:7d}  loss {running / window:.4f}  kl {kl_running / window:.5f}  "
                  f"{seen / (time.time() - started):.0f} mani/s  lr {optimizer.param_groups[0]['lr']:.2e}  "
                  + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items() if k != "mae_uniform"), flush=True)
            running = kl_running = 0.0
            window = 0
            save(step)
    errors = hard_errors(measured, hard, tags)
    print("casi difficili (errore):", "  ".join(f"{name} {e:.4f}" for name, e in zip(HARD_TAGS, errors, strict=True)))
    print("salvato", args.out)


if __name__ == "__main__":
    main()
