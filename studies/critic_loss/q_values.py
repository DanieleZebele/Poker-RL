"""Does a Q-function trained as a classifier (HL-Gauss, as in DQN of "Stop Regressing") choose
better than the PPO policy whose hands it was learned from? An offline test, before any DQN
trainer is written.

The data is `critic_loss.py generate`'s (its shards carry the cards, the legal mask and the action
taken): a model's own self-play hands on training tables. Two phases, each its own process:

    # Q-networks: the policy's inputs (the features after the cards and the equity encoder's 32
    # numbers of the player's own cards, never anyone else's), one HL-Gauss distribution over
    # symlog bins for each of the 11 actions, trained on the Monte Carlo return of the action taken
    PYTHONPATH=src:studies/agents:studies/critic_loss \\
        python studies/critic_loss/q_values.py train --jobs 4 --threads 4

    # the agents built on them sit in the hero's seat on the same hands as the model itself,
    # against the model (`self`) and against the frozen anchors (`anchors`)
    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents:studies/critic_loss \\
        python studies/critic_loss/q_values.py evaluate --jobs 16

What the Q learns is Q of the *policy that played* (Monte Carlo, no bootstrap), so acting on it
is one step of policy improvement: the most a DQN could get from these hands without playing
new ones. The agents (`agent_distribution`): `greedy` (the best Q over the legal actions),
`supported` (the best Q among actions the policy plays at least `SUPPORT` of the time: an action
it almost never plays has few examples and its Q is the noisiest, so the plain argmax favours
it), `improve` (the policy times exp(Q / tau), tau in big blinds: it moves only where the Q's
differ). Comparing them with the policy on the same hands, the cards and stacks being equal,
says whether the Q knows something the policy does not.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from critic_loss import (
    SIGMA_RATIO,
    bin_edges,
    bin_values,
    build_critic,
    explained_variance,
    hl_gauss_batch,
    size_group,
    split_of,
)

DATA = Path("checkpoints/studies/q_values")
KIND = "hl-symlog"
SUPPORT = 0.05
TAUS_BB = (1.0, 3.0)
ACTIONS = 11


# ---- pure parts ------------------------------------------------------------------------------


def agent_distribution(mode: str, q: Sequence[float], probs: Sequence[float], mask: Sequence[bool],
                       param: float = 0.0) -> list[float]:
    """The probability an agent gives each action, from the Q's and the policy's probabilities.

    `policy`: the policy's. `greedy`: all on the best legal Q. `supported`: all on the best Q
    among the legal actions the policy plays at least `param` of the time (the policy's likeliest
    if none). `improve`: the policy times exp((Q - best Q) / `param`), renormalised."""
    legal = [i for i, ok in enumerate(mask) if ok]
    if mode == "policy":
        return list(probs)
    if mode in ("greedy", "supported"):
        pool = legal if mode == "greedy" else [i for i in legal if probs[i] >= param]
        if not pool:
            pool = [max(legal, key=lambda i: probs[i])]
        best = max(pool, key=lambda i: q[i])
        return [1.0 if i == best else 0.0 for i in range(len(q))]
    if mode == "improve":
        top = max(q[i] for i in legal)
        weights = [probs[i] * math.exp((q[i] - top) / param) if mask[i] else 0.0 for i in range(len(q))]
        total = sum(weights)
        if total <= 0:
            return list(probs)
        return [w / total for w in weights]
    raise ValueError(mode)


def paired(base: dict[int, list[float]], other: dict[int, list[float]]) -> tuple[float, float]:
    """Mean of `other - base` in bb/100 over the tables both played, and its 95% half-width, the
    tables being the independent units (a table's hands share players and statistics)."""
    tables = sorted(set(base) & set(other))
    if not tables:
        return float("nan"), float("nan")
    diffs = [(sum(other[t]) - sum(base[t])) for t in tables]
    hands = [len(base[t]) for t in tables]
    mean = 100 * sum(diffs) / sum(hands)
    if len(tables) < 2:
        return mean, float("nan")
    per_hand = [d / h for d, h in zip(diffs, hands, strict=True)]
    m = sum(per_hand) / len(per_hand)
    sd = math.sqrt(sum((x - m) ** 2 for x in per_hand) / (len(per_hand) - 1))
    return mean, 100 * 1.96 * sd / math.sqrt(len(per_hand))


def arms_of(q_names: Sequence[str]) -> list[tuple[str, str, str, float]]:
    """(arm name, Q-network, mode, parameter): the policy, then four agents per Q-network."""
    arms = [("policy", "", "policy", 0.0)]
    for name in q_names:
        arms += [(f"{name} greedy", name, "greedy", 0.0),
                 (f"{name} supported", name, "supported", SUPPORT)]
        arms += [(f"{name} improve tau {tau:g}", name, "improve", tau) for tau in TAUS_BB]
    return arms


# ---- train -----------------------------------------------------------------------------------

_SHARED: dict = {}


def load_model(meta: dict):
    from pokerlab.rl.ppo import build_model_from_checkpoint

    model, _checkpoint = build_model_from_checkpoint(meta["path"])
    model.eval()
    return model


def load_inputs(out: Path, model) -> dict:
    """Every decision as the policy's trunk reads it: the features after the cards, then the
    equity encoder's output for the player's own cards and the board. The shards are read one at
    a time into one table allocated up front, so the peak is the table and one shard."""
    import torch

    paths = sorted(out.glob("shard-*.pt"))
    if not paths:
        raise SystemExit(f"{out}: no shards (generate them with critic_loss.py generate --out {out})")
    sizes = []
    for path in paths:   # the row counts first, from the small `info` table of each shard
        shard = torch.load(path, weights_only=True, mmap=True)
        if "action" not in shard:
            raise SystemExit(f"{path}: no actions in it (written before critic_loss.py stored them)")
        sizes.append(len(shard["info"]))
        width = shard["x"].shape[1] + model.equity["latent"]
        del shard
    total = sum(sizes)
    inputs = torch.empty(total, width)
    small = {"mask": [], "action": [], "target": [], "value": [], "info": []}
    row = 0
    with torch.no_grad():
        for path, n in zip(paths, sizes, strict=True):
            shard = torch.load(path, weights_only=True)
            x = shard["x"]
            inputs[row:row + n, :x.shape[1]] = x
            inputs[row:row + n, x.shape[1]:] = model.equity_encoder(shard["hole"].float(), shard["board"].float())
            for key, values in small.items():
                values.append(shard[key])
            row += n
            del shard, x
    data = {key: torch.cat(values) for key, values in small.items()}
    data["inputs"] = inputs
    return data


def warm_start(qnet, model) -> None:
    """Copy the policy's trunk and the hidden layers of its head into the Q-network's first
    layers (the output layer stays new)."""
    layers = [*model.trunk, *list(model.policy_head)[:-1]]
    for i, module in enumerate(layers):
        qnet[i].load_state_dict(module.state_dict())


def q_values(qnet, inputs, centres):
    """`(B, 11)` expected values from the Q-network's bins."""
    import torch

    logits = qnet(inputs).view(len(inputs), ACTIONS, len(centres))
    return torch.softmax(logits, -1) @ centres


def policy_probs(model, inputs, mask):
    import torch

    from pokerlab.rl.policy import MASK_FILL

    logits = model.policy_head(model.trunk(inputs)).masked_fill(~mask, MASK_FILL)
    return torch.softmax(logits, -1)


def train_one(job: tuple) -> dict:
    import torch

    name, init, seed, epochs, threads = job
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    data, meta, model, split = _SHARED["data"], _SHARED["meta"], _SHARED["model"], _SHARED["split"]
    unit = meta["reward_scale"]
    inputs, target, action, mask = data["inputs"], data["target"], data["action"], data["mask"]
    train_idx, val_idx, test_idx = (torch.nonzero(split == k).squeeze(1) for k in range(3))
    edges = bin_edges(KIND, unit)
    sigma = SIGMA_RATIO * (edges[1] - edges[0])
    edges_t = torch.tensor(edges)
    centres = torch.tensor(bin_values(KIND, unit))
    space = torch.sign(target) * torch.log1p(target.abs() / unit)
    qnet = build_critic(inputs.shape[1], meta["shape"], ACTIONS * len(centres))
    if init == "warm":
        warm_start(qnet, model)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=3e-4)

    def taken(index):
        out = []
        with torch.no_grad():
            for start in range(0, len(index), 8192):
                rows = index[start:start + 8192]
                out.append(q_values(qnet, inputs[rows], centres).gather(1, action[rows, None]).squeeze(1))
        return torch.cat(out)

    best, history, started = None, [], time.time()
    for _epoch in range(epochs):
        order = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(order) - 2047, 2048):
            batch = order[start:start + 2048]
            logits = qnet(inputs[batch]).view(len(batch), ACTIONS, len(centres))
            chosen = logits[torch.arange(len(batch)), action[batch]]
            loss = -(hl_gauss_batch(space[batch], edges_t, sigma) * torch.log_softmax(chosen, -1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = explained_variance(target[val_idx].tolist(), taken(val_idx).tolist())
        history.append(val)
        if best is None or val > best[0]:
            best = (val, len(history) - 1, {k: v.clone() for k, v in qnet.state_dict().items()})
    qnet.load_state_dict(best[2])
    torch.save(best[2], Path(_SHARED["out"]) / f"q-{name}.pt")
    return {"name": name, "init": init, "seed": seed, "val": history, "best_epoch": best[1],
            "test": diagnostics(qnet, model, inputs, target, action, mask, test_idx, centres, unit),
            "minutes": (time.time() - started) / 60}


def diagnostics(qnet, model, inputs, target, action, mask, index, centres, unit) -> dict:
    """On the test decisions: how well Q of the action taken predicts the return, and what the Q
    claims about the policy's choices (in big blinds)."""
    import torch

    q_taken, gain, differs, support, ev_policy = [], [], 0, [], []
    with torch.no_grad():
        for start in range(0, len(index), 8192):
            rows = index[start:start + 8192]
            q = q_values(qnet, inputs[rows], centres)
            p = policy_probs(model, inputs[rows], mask[rows])
            q_taken.append(q.gather(1, action[rows, None]).squeeze(1))
            legal_q = q.masked_fill(~mask[rows], -1e9)
            best = legal_q.argmax(1)
            v = (p * q).sum(1)
            ev_policy.append(v)
            gain.append((legal_q.max(1).values - v) / unit)
            differs += int((best != p.argmax(1)).sum())
            support.append(p.gather(1, best[:, None]).squeeze(1))
    gain_t, support_t = torch.cat(gain), torch.cat(support)
    targets = target[index].tolist()
    return {
        "ev_q_taken": explained_variance(targets, torch.cat(q_taken).tolist()),
        "ev_v_policy": explained_variance(targets, torch.cat(ev_policy).tolist()),
        "claimed_gain_bb": float(gain_t.mean()),
        "argmax_differs": differs / len(index),
        "policy_prob_of_greedy": float(support_t.mean()),
        "greedy_rare": float((support_t < SUPPORT).float().mean()),
    }


def train(args: argparse.Namespace) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import torch

    torch.set_num_threads(1)   # no OpenMP pool in the parent: it does not survive the fork
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    model = load_model(meta)
    data = load_inputs(out, model)
    hands = data["info"][:, 0].tolist()
    split = torch.tensor([("train", "val", "test").index(split_of(h)) for h in hands])
    _SHARED.update(data=data, meta=meta, model=model, split=split, out=str(out))
    print(f"{meta['model']}: {len(hands):,} decisioni, {(split == 0).sum():,} di allenamento", flush=True)
    jobs = [(f"{init}-s{seed}", init, seed, args.epochs, args.threads)
            for init in args.inits for seed in range(args.seeds)]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("fork")) as pool:
        results = list(pool.map(train_one, jobs))
    test = torch.nonzero(split == 2).squeeze(1)
    critic_ev = explained_variance(data["target"][test].tolist(), data["value"][test].tolist())
    lines = [f"{meta['model']}: Q con HL-Gauss ({KIND}), {len(test):,} decisioni di test",
             f"(il critico del modello, che vede le carte di tutti: varianza spiegata {critic_ev:.3f})",
             (f"{'rete':12}{'VS Q(a presa)':>14}{'VS sum pi Q':>12}{'guadagno dichiarato bb':>24}"
              f"{'argmax diverso':>16}{'pi(argmax Q)':>14}{'pi<5%':>8}{'epoca':>8}")]
    for r in results:
        t = r["test"]
        lines.append(f"{r['name']:12}{t['ev_q_taken']:14.3f}{t['ev_v_policy']:12.3f}{t['claimed_gain_bb']:24.2f}"
                     f"{t['argmax_differs']:16.1%}{t['policy_prob_of_greedy']:14.1%}{t['greedy_rare']:8.1%}"
                     f"{r['best_epoch'] + 1:>5}/{len(r['val'])}")
    report = "\n".join(lines)
    print(report)
    (out / "train-report.txt").write_text(report)
    (out / "train-results.json").write_text(json.dumps(results))


# ---- evaluate --------------------------------------------------------------------------------


def make_agent_fn(model, qnet, centres, mode: str, param: float):
    """A policy function (`RLAgentPlayer`'s) that acts as `agent_distribution` says."""
    import torch

    from pokerlab.players.rl_agent import PolicyDecision
    from pokerlab.rl.features import CARDS_DIM
    from pokerlab.rl.policy import BOARD_PLANE, HOLE_PLANE, PLANE

    def fn(features, mask, bias=None, temperature=1.0):
        with torch.no_grad():
            f = torch.tensor([features], dtype=torch.float32)
            m = torch.tensor([mask], dtype=torch.bool)
            latent = model.equity_encoder(f[:, HOLE_PLANE * PLANE:(HOLE_PLANE + 1) * PLANE],
                                          f[:, BOARD_PLANE * PLANE:(BOARD_PLANE + 1) * PLANE])
            inputs = torch.cat([f[:, CARDS_DIM:], latent], 1)
            probs = policy_probs(model, inputs, m)[0].tolist()
            q = q_values(qnet, inputs, centres)[0].tolist()
            dist = agent_distribution(mode, q, probs, mask, param)
            index = int(torch.multinomial(torch.tensor(dist), 1)) if mode == "improve" else dist.index(1.0)
        return PolicyDecision(action_index=index, log_prob=0.0, value=0.0)

    fn.feature_version = model.feature_version
    return fn


def table_setup(seed: int, field: str, mix, anchors: Sequence[str]) -> tuple[int, int, list[str]]:
    """(size, hero seat, the anchors in the other seats) of one table: the same for every arm."""
    size = mix.draw_size(random.Random(seed))
    hero = random.Random(seed + 1).randrange(size)
    others = random.Random(seed + 2).sample(list(anchors), size - 1) if field == "anchors" else []
    return size, hero, others


def play_job(job: tuple) -> tuple:
    import torch

    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.forking import torch_seeded
    from pokerlab.rl.rollout import TableBank

    field, table, arm = job
    s = _SHARED
    mix, hands = s["mix"], s["table_hands"]
    seed = s["seed"] * 1_000_003 + (0 if field == "self" else 500_009) + table
    size, hero, others = table_setup(seed, field, mix, sorted(s["anchors"]))
    torch.set_num_threads(1)
    bank = TableBank(mix, random.Random(seed), random.Random(seed + 7_777))
    proxies = bank.seats(size)
    opponents = iter(others)
    for seat, proxy in enumerate(proxies):
        fn = s["agents"][arm] if seat == hero else (s["anchors"][next(opponents)] if others else s["agents"]["policy"])
        proxy.inner = RLAgentPlayer(f"p{seat}", f"P{seat}", policy_fn=fn, big_blind=mix.big_blind,
                                    starting_stack=mix.starting_stack)
    bank.reset_stats(size)
    deltas = []
    with torch_seeded(seed):
        for _ in range(hands):
            before, after = bank.play_hand(size)
            deltas.append((after[hero] - before[hero]) / mix.big_blind)
    return field, table, arm, size, deltas


def evaluate(args: argparse.Namespace) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import torch

    from pokerlab.config import parse_with_config
    from pokerlab.rl import train as train_module
    from pokerlab.rl.global_arena import discover_benchmark_population
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.table_mix import table_mix_from_args

    torch.set_num_threads(1)
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    targs, _report = parse_with_config(train_module.build_parser(), [], siblings=train_module._sibling_parsers)
    mix = table_mix_from_args(targs)
    model = load_model(meta)
    centres = torch.tensor(bin_values(KIND, meta["reward_scale"]))
    q_names = args.q or sorted(p.stem[2:] for p in out.glob("q-*.pt"))
    agents = {"policy": make_policy_fn(model)}
    for arm, name, mode, param in arms_of(q_names)[1:]:
        qnet = build_critic(model.trunk[0].in_features, meta["shape"], ACTIONS * len(centres))
        qnet.load_state_dict(torch.load(out / f"q-{name}.pt", weights_only=True))
        qnet.eval()
        # tau is in big blinds, the Q's in the reward's unit
        agents[arm] = make_agent_fn(model, qnet, centres, mode,
                                    param * meta["reward_scale"] if mode == "improve" else param)
    anchors = {}
    for candidate in discover_benchmark_population(Path("checkpoints")):
        anchor, _ = build_model_from_checkpoint(candidate.path)
        anchors[candidate.label] = make_policy_fn(anchor)
    _SHARED.update(mix=mix, agents=agents, anchors=anchors, seed=args.seed, table_hands=args.table_hands)
    jobs = [(field, t, arm) for field in args.fields for t in range(args.tables) for arm in agents]
    print(f"{meta['model']}: {len(agents)} agenti, {len(anchors)} ancore, {len(jobs)} tavoli da "
          f"{args.table_hands} mani", flush=True)
    results, started = [], time.time()
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("fork")) as pool:
        for i, r in enumerate(pool.map(play_job, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)} tavoli, {(time.time() - started) / 60:.1f} min", flush=True)
    report = format_evaluation(results, list(agents), args.fields, meta)
    print(report)
    (out / "evaluate-report.txt").write_text(report)
    (out / "evaluate-results.json").write_text(json.dumps(results))


def format_evaluation(results, arms: Sequence[str], fields: Sequence[str], meta: dict) -> str:
    lines = [(f"{meta['model']}: l'agente nel posto dell'eroe, sulle stesse mani; differenza con la policy "
              "in bb/100 (+- 95%, tavoli come unita')")]
    for field in fields:
        by_arm = {arm: {} for arm in arms}
        sizes = {}
        for f, table, arm, size, deltas in results:
            if f == field:
                by_arm[arm][table] = deltas
                sizes[table] = size
        hands = sum(len(d) for d in by_arm["policy"].values())
        lines += ["", f"contro {'il modello stesso' if field == 'self' else 'le ancore'} ({hands:,} mani per agente)",
                  f"{'agente':28}{'bb/100':>9}{'diff':>9}{'+-':>7} | " + " ".join(f"{g:>12}" for g in ("2-3", "4-6", "7-9"))]
        for arm in arms:
            own = 100 * sum(sum(d) for d in by_arm[arm].values()) / max(1, sum(len(d) for d in by_arm[arm].values()))
            if arm == "policy":
                lines.append(f"{arm:28}{own:9.1f}")
                continue
            diff, half = paired(by_arm["policy"], by_arm[arm])
            cells = []
            for group in ("2-3", "4-6", "7-9"):
                keep = {t for t, n in sizes.items() if size_group(n) == group}
                g_diff, g_half = paired({t: d for t, d in by_arm["policy"].items() if t in keep},
                                        {t: d for t, d in by_arm[arm].items() if t in keep})
                cells.append(f"{g_diff:+6.1f}+-{g_half:<4.0f}")
            lines.append(f"{arm:28}{own:9.1f}{diff:+9.1f}{half:7.1f} | " + " ".join(f"{c:>12}" for c in cells))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="phase", required=True)
    t = sub.add_parser("train")
    t.add_argument("--inits", nargs="+", default=["scratch", "warm"], choices=("scratch", "warm"),
                   help="a new network, or one starting from the policy's trunk")
    t.add_argument("--seeds", type=int, default=2)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--jobs", type=int, default=4)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--out", default=str(DATA))
    e = sub.add_parser("evaluate")
    e.add_argument("--q", nargs="*", default=None, help="Q-networks by name (default: every q-*.pt in --out)")
    e.add_argument("--fields", nargs="+", default=["self", "anchors"], choices=("self", "anchors"))
    e.add_argument("--tables", type=int, default=100, help="tables per field (every agent plays each)")
    e.add_argument("--table-hands", type=int, default=500)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--jobs", type=int, default=16)
    e.add_argument("--out", default=str(DATA))
    args = parser.parse_args(argv)
    train(args) if args.phase == "train" else evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
