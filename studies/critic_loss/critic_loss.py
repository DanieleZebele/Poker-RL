"""Does the critic learn better as a classifier? An offline test of "Stop Regressing" (Farebrother
et al., 2024) on this project's critic, before anything goes into PPO.

Two phases, each its own process:

    # the data: a model plays training tables (its own pool, styles, stacks, table sizes, from
    # config.toml); every decision of its seats is kept with what the critic reads and the hand's
    # result (the training reward, all-in runouts included: lambda = 1, so no critic is in it)
    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents:studies/critic_loss \\
        python studies/critic_loss/critic_loss.py generate --model '#1' --hands 100000 --jobs 30

    # the test: critics of the model's own shape trained from scratch on the same decisions, one
    # per loss, and the explained variance on hands none of them saw, by street, stack and table
    PYTHONPATH=src:studies/agents:studies/critic_loss \\
        python studies/critic_loss/critic_loss.py train --jobs 8

The losses (`LOSSES`): `mse` (the scalar critic's squared error) and two HL-Gauss classifiers --
the target is a Gaussian of width 0.75 bins over K bins, the loss the cross-entropy and the value
the mean of the bins' centres. `hl-symlog` puts its bins uniformly in sign(v) log(1 + |v| in bb),
fine near 0 and coarse in the tails; `hl-uniform` uniformly in the value itself. The bins matter
here: the target runs from -1 (a 100 bb stack lost) to +8 (eight lost to it), and a short stack's
hand is a few hundredths of that.

What is compared is the explained variance of the Monte Carlo return on held-out hands. It is
mostly luck nobody can predict (the cards still to come), so the numbers are small and the
differences are what matter; two seeds a loss say how much of a difference is noise.

How many bins, and how wide a Gaussian: `--hl-configs BINS:RATIO ...` trains every HL loss once
per pair (the Gaussian's sd is RATIO bins). The two interact -- the smoothing in the target's own
space is RATIO x range / BINS -- so a grid that wants to tell "more bins" from "less smoothing"
needs pairs that hold one fixed:

    PYTHONPATH=src:studies/agents:studies/critic_loss python studies/critic_loss/critic_loss.py \\
        train --losses hl-symlog --epochs 15 --tag bins \\
        --hl-configs 11:0.75 25:0.75 51:0.75 101:0.75 201:0.75 401:0.75 101:0.375 101:1.5 101:3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

OUT = Path("checkpoints/studies/critic_loss")
LOSSES = ("mse", "hl-symlog", "hl-uniform")
STREETS = ("preflop", "flop", "turn", "river")
STACKS = ("<10", "10-30", "30+")
SIZES = ("2-3", "4-6", "7-9")
# The value is the training reward: big blinds times big blind / deepest stack (1/100).
VALUE_RANGE = (-1.05, 8.05)
SYMLOG_BINS = 101
UNIFORM_BINS = 401
SIGMA_RATIO = 0.75   # the paper's width of the Gaussian, in bins


# ---- pure parts --------------------------------------------------------------------------------


def symlog(value: float, unit: float) -> float:
    """sign(v) log(1 + |v| / unit): `unit` is what one big blind is in the value's scale."""
    return math.copysign(math.log1p(abs(value) / unit), value)


def symexp(z: float, unit: float) -> float:
    return math.copysign(math.expm1(abs(z)) * unit, z)


def default_bins(kind: str) -> int:
    if kind == "hl-symlog":
        return SYMLOG_BINS
    if kind == "hl-uniform":
        return UNIFORM_BINS
    raise ValueError(kind)


def bin_edges(kind: str, unit: float, count: int | None = None) -> list[float]:
    """The K + 1 edges of the bins, in the space the Gaussian lives in (symlog or the value).
    `count` bins, or the kind's default."""
    count = count or default_bins(kind)
    low, high = VALUE_RANGE
    if kind == "hl-symlog":
        low, high = symlog(low, unit), symlog(high, unit)
    return [low + (high - low) * i / count for i in range(count + 1)]


def bin_values(kind: str, unit: float, count: int | None = None) -> list[float]:
    """Each bin's centre as a value: what the predicted distribution is averaged over."""
    edges = bin_edges(kind, unit, count)
    centres = [(a + b) / 2 for a, b in pairwise(edges)]
    return [symexp(c, unit) for c in centres] if kind == "hl-symlog" else centres


def hl_gauss(target: float, edges: Sequence[float], sigma: float) -> list[float]:
    """The probability of each bin under N(target, sigma), renormalised to the range (a target
    outside it is clipped to its end)."""
    target = min(max(target, edges[0]), edges[-1])
    cdf = [0.5 * (1 + math.erf((e - target) / (math.sqrt(2) * sigma))) for e in edges]
    mass = cdf[-1] - cdf[0]
    return [(b - a) / mass for a, b in pairwise(cdf)]


def hl_gauss_batch(targets, edges, sigma: float):
    """`hl_gauss` for a tensor of targets at once: `(B,)` -> `(B, K)`."""
    import torch

    clipped = targets.clamp(float(edges[0]), float(edges[-1]))
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - clipped[:, None]) / (math.sqrt(2) * sigma)))
    return (cdf[:, 1:] - cdf[:, :-1]) / (cdf[:, -1:] - cdf[:, :1])


def size_group(players: int) -> str:
    return "2-3" if players <= 3 else "4-6" if players <= 6 else "7-9"


def explained_variance(targets: Sequence[float], predictions: Sequence[float]) -> float:
    n = len(targets)
    mean = sum(targets) / n
    total = sum((t - mean) ** 2 for t in targets)
    residual = sum((t - p) ** 2 for t, p in zip(targets, predictions, strict=True))
    return 1 - residual / total if total > 0 else float("nan")


def grouped(targets, predictions, labels) -> dict[str, tuple[float, int]]:
    """Explained variance per label, with its count."""
    by: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for t, p, label in zip(targets, predictions, labels, strict=True):
        by[label][0].append(t)
        by[label][1].append(p)
    return {label: (explained_variance(t, p), len(t)) for label, (t, p) in by.items()}


def split_of(hand: int, seed: int = 0) -> str:
    """train / val / test by hand, so the decisions of one hand never sit on two sides."""
    r = random.Random(hand * 1_000_003 + seed).random()
    return "train" if r < 0.7 else "val" if r < 0.85 else "test"


def hl_config(text: str) -> tuple[int, float]:
    """"BINS:RATIO" (or "BINS", the ratio then `SIGMA_RATIO`) -> (bins, ratio)."""
    bins, _, ratio = text.partition(":")
    pair = int(bins), float(ratio) if ratio else SIGMA_RATIO
    if pair[0] < 2 or pair[1] <= 0:
        raise argparse.ArgumentTypeError(f"{text}: needs at least 2 bins and a positive ratio")
    return pair


def training_jobs(losses, hl_configs, seeds: int) -> list[tuple]:
    """(loss, bins, ratio, seed) for every run: an HL loss once per configuration (its default
    bins at `SIGMA_RATIO` when none is given), the squared errors once."""
    jobs = []
    for loss in losses:
        configs = ([(None, None)] if not loss.startswith("hl-")
                   else hl_configs or [(default_bins(loss), SIGMA_RATIO)])
        jobs += [(loss, bins, ratio, seed) for bins, ratio in configs for seed in range(seeds)]
    return jobs


def run_name(result: dict) -> str:
    config = f" {result['bins']}b s{result['sigma_ratio']:g}" if result.get("bins") else ""
    return f"{result['loss']}{config} seme {result['seed']}"


def kept(hand: int, fraction: float) -> bool:
    """Whether a training hand stays in a run that uses only `fraction` of them. The draw is the
    hand's own, so a smaller fraction keeps a subset of a larger one."""
    return fraction >= 1.0 or random.Random(hand * 7_919 + 12_345).random() < fraction


# ---- generate ----------------------------------------------------------------------------------


def generate(args: argparse.Namespace) -> None:
    import torch
    from duel_power import resolve_model

    from pokerlab.config import parse_with_config
    from pokerlab.rl import train as train_module
    from pokerlab.rl.equity_net import (
        encoder_config,
        equity_net_from_checkpoint,
        load_equity_checkpoint,
    )
    from pokerlab.rl.features import CARDS_DIM
    from pokerlab.rl.global_store import load_ranking
    from pokerlab.rl.policy import BOARD_PLANE, HOLE_PLANE, PLANE, make_critic_fns, make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import OpponentPool, SelfPlayCollector
    from pokerlab.rl.styles import style_config_from_args
    from pokerlab.rl.table_mix import table_mix_from_args
    from pokerlab.rl.training_pool import (
        available_labels,
        build_training_registry,
        draw_training_pool,
    )
    from pokerlab.rl.value_diagnostics import stack_label, street_label

    targs, _report = parse_with_config(train_module.build_parser(), [], siblings=train_module._sibling_parsers)
    mix = table_mix_from_args(targs)
    global_dir, models_dir = Path(targs.global_dir), Path(targs.models_dir)
    label, path, rating = resolve_model(args.model, global_dir=global_dir, root=Path("checkpoints"))
    model, checkpoint = build_model_from_checkpoint(path)
    settings = (checkpoint.get("metadata") or {}).get("settings") or {}
    equity_path = settings.get("equity_model") or targs.equity_model
    saved = load_equity_checkpoint(equity_path)
    if encoder_config(saved) != model.equity:
        raise SystemExit(f"{equity_path} is not the equity network of {label}")
    critic = make_critic_fns(model, equity_net_from_checkpoint(saved))

    rng = random.Random(args.seed)
    drawn = draw_training_pool(load_ranking(global_dir).members, available_labels(models_dir),
                               size=targs.pool_models, rng=rng, top_share=targs.pool_top_share,
                               top_n=targs.pool_top_n)
    opponents = train_module.registry_opponents(build_training_registry(models_dir, drawn), mix,
                                                count=targs.pool_models)
    reward_scale = mix.big_blind / mix.starting_stack
    collector = SelfPlayCollector(
        mix, make_policy_fn(model), rng=rng, opponent_pool=OpponentPool(opponents),
        opponent_probability=targs.opponent_probability, reward_scale=reward_scale,
        table_hands=targs.table_hands, concurrent_tables=targs.concurrent_tables,
        allin_runouts=targs.allin_runouts, styles=style_config_from_args(targs), critic=critic,
        child_setup=train_module._prepare_collector_copy,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": label, "path": str(path), "rating": rating, "equity_model": equity_path,
            "reward_scale": reward_scale,
            "shape": {k: getattr(model, k) for k in ("hidden", "num_layers", "head_hidden", "head_layers")},
            "opponents": [m.label for m in drawn], "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{label} (rating {rating}): {len(opponents)} avversari, scala del reward {reward_scale}", flush=True)

    shards = sorted(out.glob("shard-*.pt"))   # a rerun resumes after the shards already written
    done = sum(torch.load(s, weights_only=True)["meta"]["hands"] for s in shards)
    hand = max((int(torch.load(s, weights_only=True)["info"][:, 0].max()) + 1 for s in shards), default=0)
    started = time.time()
    while done < args.hands:
        chunk = min(args.chunk, args.hands - done)
        trajectories = collector.collect(chunk, lam=1.0, workers=args.jobs)
        rows, extras, targets, values, info = [], [], [], [], []
        holes, boards, masks, actions = [], [], [], []   # what a Q-function reads and is trained on
        for t in trajectories:
            for d, target in zip(t.decisions, t.returns, strict=True):
                rows.append(d.features[CARDS_DIM:])
                holes.append(d.features[HOLE_PLANE * PLANE:(HOLE_PLANE + 1) * PLANE])
                boards.append(d.features[BOARD_PLANE * PLANE:(BOARD_PLANE + 1) * PLANE])
                masks.append(d.legal_mask)
                actions.append(d.action_index)
                extras.append(d.critic_extra)
                targets.append(target)
                values.append(d.value)
                # one id per trajectory (a seat's hand): what the split keeps on one side
                info.append((hand, STREETS.index(street_label(d.features)),
                             STACKS.index(stack_label(t.effective_stack_bb)), t.num_players))
            hand += 1
        torch.save({
            "x": torch.tensor(rows, dtype=torch.float32),
            "extra": torch.tensor(extras, dtype=torch.float32),
            "target": torch.tensor(targets, dtype=torch.float32),
            "value": torch.tensor(values, dtype=torch.float32),
            "info": torch.tensor(info, dtype=torch.int64),
            "hole": torch.tensor(holes, dtype=torch.uint8),
            "board": torch.tensor(boards, dtype=torch.uint8),
            "mask": torch.tensor(masks, dtype=torch.bool),
            "action": torch.tensor(actions, dtype=torch.int64),
            "meta": {"hands": chunk},
        }, out / f"shard-{len(shards):04d}.pt")
        shards.append(out / f"shard-{len(shards):04d}.pt")
        done += chunk
        print(f"  {done:,}/{args.hands:,} mani, {len(rows):,} decisioni nell'ultimo blocco, "
              f"{(time.time() - started) / 60:.1f} min", flush=True)


# ---- train -------------------------------------------------------------------------------------


def load_data(out: Path):
    import torch

    shards = [torch.load(s, weights_only=True) for s in sorted(out.glob("shard-*.pt"))]
    data = {k: torch.cat([s[k] for s in shards]) for k in ("x", "extra", "target", "value", "info")}
    # hand ids are counted per shard by generate's running counter, so they are unique already
    return data


def build_critic(in_features: int, shape: dict, outputs: int):
    from torch import nn

    def block(a, b):
        return [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU()]

    layers, width = [], in_features
    for _ in range(shape["num_layers"]):
        layers += block(width, shape["hidden"])
        width = shape["hidden"]
    for _ in range(shape["head_layers"]):
        layers += block(width, shape["head_hidden"])
        width = shape["head_hidden"]
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


def critic_shape(model_shape: dict, args: argparse.Namespace) -> dict:
    """The model's critic shape, with whatever the command line overrides."""
    return {k: getattr(args, k) if getattr(args, k, None) is not None else model_shape[k]
            for k in ("hidden", "num_layers", "head_hidden", "head_layers")}


def train_one(job: tuple) -> dict:
    """One critic, one loss, one seed: trained on `train`, the epoch chosen on `val`, scored on
    `test`. Runs in a forked process that shares the data."""
    import torch

    loss_name, bins, ratio, seed, epochs, threads, shape = job
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    data, meta = _SHARED["data"], _SHARED["meta"]
    unit = meta["reward_scale"]   # one big blind in the value's scale
    inputs = data["inputs"]
    split = _SHARED["split"]
    train_idx, val_idx, test_idx = (torch.nonzero(split == k).squeeze(1) for k in range(3))
    target = data["target"]
    classes = loss_name.startswith("hl-")
    if classes:
        edges = bin_edges(loss_name, unit, bins)
        centres = torch.tensor(bin_values(loss_name, unit, bins))
        sigma = ratio * (edges[1] - edges[0])
        space = (torch.sign(target) * torch.log1p(target.abs() / unit)) if loss_name == "hl-symlog" else target
        edges_t = torch.tensor(edges)
    net = build_critic(inputs.shape[1], shape, len(centres) if classes else 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    def predict(index):
        out = []
        with torch.no_grad():
            for start in range(0, len(index), 8192):
                y = net(inputs[index[start:start + 8192]])
                out.append(torch.softmax(y, -1) @ centres if classes else y.squeeze(-1))
        return torch.cat(out)

    best, history = None, []
    started = time.time()
    for epoch in range(epochs):
        order = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(order) - 2047, 2048):
            batch = order[start:start + 2048]
            y = net(inputs[batch])
            if classes:
                loss = -(hl_gauss_batch(space[batch], edges_t, sigma) * torch.log_softmax(y, -1)).sum(-1).mean()
            else:
                loss = ((y.squeeze(-1) - target[batch]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = explained_variance(target[val_idx].tolist(), predict(val_idx).tolist())
        history.append(val)
        if best is None or val > best[0]:
            best = (val, epoch, predict(test_idx).tolist())
    return {"loss": loss_name, "bins": bins, "sigma_ratio": ratio, "seed": seed, "val": history,
            "best_epoch": best[1],
            "test_pred": best[2], "minutes": (time.time() - started) / 60}


_SHARED: dict = {}


def train(args: argparse.Namespace) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import torch

    # No thread pool in the parent: OpenMP's does not survive the fork, and a child that
    # inherits it hangs at its first parallel operation.
    torch.set_num_threads(1)
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    data = load_data(out)
    # built once here and shared with the forked children, not copied into each of them
    data["inputs"] = torch.cat([data.pop("x"), data.pop("extra")], 1)
    hands = data["info"][:, 0].tolist()
    split = torch.tensor([("train", "val", "test").index(split_of(h)) for h in hands])
    if args.train_fraction < 1.0:   # 3 = left out of this run
        split = torch.tensor([3 if k == 0 and not kept(h, args.train_fraction) else k
                              for h, k in zip(hands, split.tolist(), strict=True)])
    _SHARED.update(data=data, meta=meta, split=split)
    print(f"{meta['model']}: {len(hands):,} decisioni di {len(set(hands)):,} mani-posto "
          f"({(split == 0).sum():,} allenamento, {(split == 1).sum():,} scelta, {(split == 2).sum():,} test)",
          flush=True)
    shape = critic_shape(meta["shape"], args)
    print(f"critico {shape['hidden']}x{shape['num_layers']} + testa {shape['head_hidden']}x{shape['head_layers']}"
          f" (il modello: {meta['shape']['hidden']}x{meta['shape']['num_layers']}"
          f" + {meta['shape']['head_hidden']}x{meta['shape']['head_layers']})", flush=True)
    jobs = [(*job, args.epochs, args.threads, shape)
            for job in training_jobs(args.losses, args.hl_configs, args.seeds)]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("fork")) as pool:
        results = list(pool.map(train_one, jobs))
    test = torch.nonzero(split == 2).squeeze(1)
    report = format_report(data, test, results, meta)
    print(report)
    suffix = f"-{args.tag}" if args.tag else ""
    (out / f"report{suffix}.txt").write_text(report)
    (out / f"results{suffix}.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "test_pred"} for r in results]))


def format_report(data, test, results, meta) -> str:
    target = data["target"][test].tolist()
    info = data["info"][test].tolist()
    groups = {
        "strada": [STREETS[i[1]] for i in info],
        "stack": [STACKS[i[2]] for i in info],
        "tavolo": [size_group(i[3]) for i in info],
    }
    rows = [("critico del modello (PPO, GAE)", data["value"][test].tolist(), None)]
    rows += [(run_name(r), r["test_pred"], r) for r in results]
    lines = [(f"{meta['model']} (rating {meta['rating']:.0f}): varianza spiegata del ritorno sulle mani di test "
              f"({len(target):,} decisioni)"),
             f"{'critico':32}{'tutto':>7} | " + " ".join(f"{n:>7}" for n in STREETS) + " | "
             + " ".join(f"{n:>6}" for n in STACKS) + " | " + " ".join(f"{n:>6}" for n in SIZES) + " | epoca"]
    counts = {}
    for name, pred, result in rows:
        cells = {}
        for labels in groups.values():
            for label, (ev, n) in grouped(target, pred, labels).items():
                cells[label] = ev
                counts[label] = n
        epoch = f"{result['best_epoch'] + 1:>3}/{len(result['val'])}" if result else "  -"
        lines.append(f"{name:32}{explained_variance(target, pred):7.3f} | "
                     + " ".join(f"{cells.get(n, float('nan')):7.3f}" for n in STREETS) + " | "
                     + " ".join(f"{cells.get(n, float('nan')):6.3f}" for n in STACKS) + " | "
                     + " ".join(f"{cells.get(n, float('nan')):6.3f}" for n in SIZES) + f" | {epoch}")
    lines.append(f"{'decisioni':32}{len(target):7} | " + " ".join(f"{counts.get(n, 0):7}" for n in STREETS)
                 + " | " + " ".join(f"{counts.get(n, 0):6}" for n in STACKS) + " | "
                 + " ".join(f"{counts.get(n, 0):6}" for n in SIZES))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="phase", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", default="#1")
    g.add_argument("--hands", type=int, default=100_000)
    g.add_argument("--chunk", type=int, default=10_000)
    g.add_argument("--jobs", type=int, default=30)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=str(OUT))
    t = sub.add_parser("train")
    t.add_argument("--losses", nargs="+", default=list(LOSSES), choices=LOSSES)
    t.add_argument("--seeds", type=int, default=2)
    t.add_argument("--hl-configs", nargs="+", type=hl_config, default=None, metavar="BINS:RATIO",
                   help="the HL losses once per pair: bins, and the Gaussian's sd in bins "
                   f"(default: each kind's own bins at {SIGMA_RATIO})")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--jobs", type=int, default=8)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--out", default=str(OUT))
    t.add_argument("--hidden", type=int, help="trunk width (default: the model's)")
    t.add_argument("--num-layers", type=int, help="trunk layers (default: the model's)")
    t.add_argument("--head-hidden", type=int, help="head width (default: the model's)")
    t.add_argument("--head-layers", type=int, help="head layers (default: the model's)")
    t.add_argument("--train-fraction", type=float, default=1.0,
                   help="train on this share of the training hands (validation and test unchanged)")
    t.add_argument("--tag", default="", help="written as report-<tag>.txt / results-<tag>.json")
    args = parser.parse_args(argv)
    generate(args) if args.phase == "generate" else train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
