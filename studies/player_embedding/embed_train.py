"""Train the player embedder on the generated hands and say whether it knows anything.

    PYTHONPATH=src:studies/agents:studies/player_embedding \\
        python studies/player_embedding/embed_train.py --data checkpoints/studies/player_embedding \\
        --sequence-models gru mean transformer

It trains the embedder with its predictor (`embed_model`) once per `--sequence-models` entry (a
GRU, the running mean of the hands, a causal transformer: if the mean does as well as the GRU, the
GRU is not earning its keep), then **the same predictor with no embedding** (zeros in its place)
for the same number of steps, and prints how much each embedding
lowers the error of predicting a player's actions on **models it was not trained on**
(`--holdout`, by the model's name). That table is the gate before any reinforcement learning is
spent on the idea:

- *by hands seen* (the error at a player's hand 0, 1-9, 10-29, 30-59, 60+): an embedding that has
  learnt styles helps more the more hands it has read, and not at all at hand 0;
- *by showdown mode* (`full`, `winners` only, `none`): the embedder is trained with the mode drawn at
  random per sequence, and judged under each. What `none` shows is what it is worth when no cards are
  ever seen -- the case of a screen that does not read them.

The loss is the cross-entropy of the action (`NUM_ACTIONS` classes) given the embedding, the public
spot and the player's own cards, plus `--strength-weight` times the squared error of the cards'
strength given the embedding, the spot and the action (the range-reading task).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import embed_data
import numpy as np
import torch
from embed_model import SEQUENCE_MODELS, PlayerEmbedder, Predictor
from embed_records import STATE_DIM
from torch import Tensor
from torch.nn import functional

BUCKETS = ((0, 1, "0"), (1, 10, "1-9"), (10, 30, "10-29"), (30, 60, "30-59"), (60, 10_000, "60+"))
TRAIN_MODE_WEIGHTS = (0.5, 0.25, 0.25)  # full, winners, none


def tensors(arrays: dict, rows: np.ndarray, modes: list[str], device: str) -> dict[str, Tensor]:
    """The batch `rows`, each sequence under its own showdown mode."""
    show = arrays["show"][rows].copy()
    shown = arrays["shown"][rows].copy()
    for index, mode in enumerate(modes):
        show[index], shown[index] = (
            value[0] for value in embed_data.apply_mode(show[index][None], arrays["won"][rows][index][None],
                                                        shown[index][None], mode)
        )
    cls = torch.from_numpy(arrays["cls"][rows].astype(np.int64)).to(device)
    return {
        "x": torch.from_numpy(arrays["x"][rows]).to(device),
        "mask": (cls >= 0),
        "cls": cls.clamp(min=0),
        "strength": torch.from_numpy(arrays["strength"][rows].astype(np.float32)).to(device),
        "show": torch.from_numpy(show.astype(np.int64)).to(device),
        "won": torch.from_numpy(arrays["won"][rows].astype(np.int64)).to(device),
        "shown": torch.from_numpy(shown.astype(np.float32)).to(device),
    }


def losses(embedder: PlayerEmbedder | None, predictor: Predictor, batch: dict, embedding_size: int):
    """Per-decision action error and strength error, with the hand index of each, over the
    decisions that exist. `embedder=None` is the ablation: a zero embedding."""
    x, mask = batch["x"], batch["mask"]
    batches, hands, slots, _ = x.shape
    if embedder is None:
        embedding = torch.zeros(batches, hands, embedding_size, device=x.device)
    else:
        embedding = embedder.embeddings(x, mask, batch["show"], batch["won"], batch["shown"])
    embedding = embedding[:, :, None, :].expand(batches, hands, slots, embedding_size)
    state = x[..., :STATE_DIM]
    action = x[..., STATE_DIM:]
    logits = predictor.action_logits(embedding[mask], state[mask], batch["strength"][mask][:, None])
    action_error = functional.cross_entropy(logits, batch["cls"][mask], reduction="none")
    guess = predictor.strength_guess(embedding[mask], state[mask], action[mask])
    strength_error = (guess - batch["strength"][mask]) ** 2
    hand_index = torch.arange(hands, device=x.device)[None, :, None].expand(batches, hands, slots)[mask]
    return action_error, strength_error, hand_index


@torch.no_grad()
def evaluate(embedder, predictor, arrays, rows, mode, embedding_size, device, batch=64):
    """Mean action error / strength error of the held-out sequences under `mode`, in total and
    by the number of hands the player had been seen for."""
    total = {"n": 0.0, "action": 0.0, "strength": 0.0}
    by_bucket = {name: {"n": 0.0, "action": 0.0, "strength": 0.0} for _, _, name in BUCKETS}
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        data = tensors(arrays, part, [mode] * len(part), device)
        action, strength, hand_index = losses(embedder, predictor, data, embedding_size)
        for low, high, name in BUCKETS:
            pick = (hand_index >= low) & (hand_index < high)
            count = float(pick.sum())
            if count:
                by_bucket[name]["n"] += count
                by_bucket[name]["action"] += float(action[pick].sum())
                by_bucket[name]["strength"] += float(strength[pick].sum())
        total["n"] += float(len(action))
        total["action"] += float(action.sum())
        total["strength"] += float(strength.sum())
    def mean(cell):
        return {"action": cell["action"] / max(cell["n"], 1), "strength": cell["strength"] / max(cell["n"], 1)}
    return mean(total), {name: mean(cell) for name, cell in by_bucket.items()}


def fit(arrays, train_rows, label, *, use_embedding, args, device, sequence_model="gru"):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    embedder = (
        PlayerEmbedder(args.hidden, args.embedding, sequence_model).to(device) if use_embedding else None
    )
    predictor = Predictor(args.embedding, args.predictor_hidden).to(device)
    parameters = list(predictor.parameters()) + (list(embedder.parameters()) if embedder else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    started = time.time()
    for step in range(1, args.steps + 1):
        rows = np.asarray(rng.sample(list(train_rows), min(args.batch, len(train_rows))))
        modes = rng.choices(embed_data.MODES, weights=TRAIN_MODE_WEIGHTS, k=len(rows))
        data = tensors(arrays, rows, modes, device)
        action, strength, _ = losses(embedder, predictor, data, args.embedding)
        loss = action.mean() + args.strength_weight * strength.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            print(f"  {label}: step {step}/{args.steps}  action {action.mean():.4f}  "
                  f"strength {strength.mean():.4f}  {time.time() - started:.0f}s", flush=True)
    return embedder, predictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="a folder of shard-*.npz")
    parser.add_argument("--out", type=Path, default=None, help="where to save the embedder (default: <data>/embedder.pt)")
    parser.add_argument("--holdout", type=float, default=0.2, help="share of the MODELS kept for validation")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embedding", type=int, default=16)
    parser.add_argument("--sequence-models", nargs="+", choices=SEQUENCE_MODELS, default=["gru"],
                        help="what reads the hands in order: gru, mean (no order, no memory), transformer")
    parser.add_argument("--predictor-hidden", type=int, default=128)
    parser.add_argument("--strength-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    shards = sorted(args.data.glob("shard-*.npz"))
    if not shards:
        print(f"no shard-*.npz in {args.data}")
        return 1
    arrays = embed_data.load_shards(shards)
    held = embed_data.holdout_mask(arrays["model"], args.holdout)
    train_rows, test_rows = np.flatnonzero(~held), np.flatnonzero(held)
    if len(train_rows) == 0 or len(test_rows) == 0:
        print("the models split leaves nothing to train or to validate on: more models or another --holdout")
        return 1
    print(f"{len(held)} sequences of {arrays['show'].shape[1]} hands; "
          f"train {len(train_rows)} / held-out models {len(test_rows)}", flush=True)

    trained = {}
    for name in args.sequence_models:
        trained[name] = fit(arrays, train_rows, name, use_embedding=True, args=args,
                            device=args.device, sequence_model=name)
    _, base_predictor = fit(arrays, train_rows, "no embedding", use_embedding=False, args=args, device=args.device)
    for embedder, predictor in trained.values():
        embedder.eval(); predictor.eval()
    base_predictor.eval()

    names = list(trained)
    print("\naction error on models never trained on: gain over no embedding "
          "(cross-entropy, positive is better)")
    print(f"{'mode':>8} {'hands seen':>10} {'no embedding':>13}" + "".join(f"{name:>13}" for name in names))
    for mode in embed_data.MODES:
        without_total, without = evaluate(None, base_predictor, arrays, test_rows, mode, args.embedding, args.device)
        results = {
            name: evaluate(embedder, predictor, arrays, test_rows, mode, args.embedding, args.device)
            for name, (embedder, predictor) in trained.items()
        }
        for _, _, bucket in BUCKETS:
            cells = "".join(f"{without[bucket]['action'] - results[n][1][bucket]['action']:+13.4f}" for n in names)
            print(f"{mode:>8} {bucket:>10} {without[bucket]['action']:13.4f}{cells}")
        cells = "".join(f"{without_total['action'] - results[n][0]['action']:+13.4f}" for n in names)
        print(f"{mode:>8} {'all':>10} {without_total['action']:13.4f}{cells}")

    for name, (embedder, predictor) in trained.items():
        out = (args.out or args.data / "embedder.pt").with_name(f"embedder-{name}.pt")
        torch.save({"embedder": embedder.state_dict(), "predictor": predictor.state_dict(),
                    "hidden": args.hidden, "embedding": args.embedding,
                    "sequence_model": name, "version": 1}, out)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
