"""How different are the store's models? The study behind the pruning by diversity (`rl/diversity.py`).

    # 1. the probes: decisions of real hands, the same for every model
    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/diversity_study.py probes \\
        --sessions 1600 --count 10000 --jobs 40 --out diversity/probes-v2.pkl
    # 2. every model's signature on them (store and anchors), in the study's own directory
    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/diversity_study.py signatures \\
        --probes diversity/probes-v2.pkl --jobs 40 --out checkpoints/studies/diversity
    # 3. the report
    PYTHONPATH=src:studies/agents python studies/agents/diversity_study.py report --out checkpoints/studies/diversity

The report: how far apart the models are (`d`, the mean Hellinger distance between two models'
action distributions over the probes; 0 = they play every probe alike), by street; each model's
nearest neighbour against its rating; parents and their children against random pairs (the sweep's
observations name the parent); the top 100 among themselves; how the distance agrees with the
distance between the nine statistics of `style`; whether half the probes rank the pairs as all of
them do (enough probes?); and what a pruning by diversity would remove, against the pruning by
rating, at a few fractions.

Writes only under `--out` (and the probes file): never the shared ledger or the store.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

STREET_SHARES = {"preflop": 0.4, "flop": 0.2, "turn": 0.2, "river": 0.2}
# What share of each street's decisions a session keeps, so that a few hundred sessions give
# enough of every street (a hand is ~8 preflop decisions, ~1.2 flop, ~0.5 turn, ~0.3 river).
KEEP = {"preflop": 0.015, "flop": 0.03, "turn": 0.07, "river": 0.11}


def _population(root: Path, global_dir: Path, with_anchors: bool):
    from pokerlab.rl.global_arena import discover_benchmark_population, discover_population
    from pokerlab.rl.global_store import load_ranking

    population = list(discover_population(root))
    if with_anchors:
        population += list(discover_benchmark_population(root))
    ratings = {label: m for label, m in load_ranking(global_dir).members.items()}
    return population, ratings


def play_for_probes(seed: int, plan: dict) -> list:
    from pokerlab.players.base import Player
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.forking import torch_seeded
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank

    mix = plan["mix"]
    rng = random.Random(seed)
    size = mix.draw_size(rng)
    chosen = []
    while len(chosen) < size:
        source = plan["top"] if rng.random() < plan["top_share"] else plan["everyone"]
        pick = rng.choice(source)
        if pick not in chosen:
            chosen.append(pick)
    kept: list = []

    class Recorder(Player):
        def __init__(self, inner):
            super().__init__(inner.player_id, inner.name)
            self.inner = inner

        def act(self, observation, legal_actions):
            if rng.random() < KEEP[observation.street.value]:
                kept.append((observation, list(legal_actions)))
            return self.inner.act(observation, legal_actions)

    with torch_seeded(seed):
        bank = TableBank(mix, random.Random(seed))
        for seat, proxy in enumerate(bank.seats(size)):
            model, _ = build_model_from_checkpoint(chosen[seat], device="cpu")
            proxy.inner = Recorder(RLAgentPlayer(proxy.player_id, proxy.name, policy_fn=make_policy_fn(model),
                                                 big_blind=mix.big_blind, starting_stack=mix.starting_stack))
        bank.play_session(size, plan["hands"])
    from dataclasses import replace

    return [(replace(o, seat_memory=None), legal) for o, legal in kept]


def cmd_probes(args) -> int:
    from pokerlab.rl.diversity import save_probes
    from pokerlab.rl.forking import run_in_copies
    from pokerlab.rl.table_mix import TableMix

    population, ratings = _population(args.root, args.global_dir, with_anchors=False)
    ranked = sorted((c for c in population if c.label in ratings), key=lambda c: -ratings[c.label].rating)
    plan = {"mix": TableMix(), "hands": args.hands, "top_share": 0.7,
            "top": [str(c.path) for c in ranked[:100]], "everyone": [str(c.path) for c in population]}
    started = time.time()
    found = [probe for part in run_in_copies(lambda job: play_for_probes(args.seed + job, plan),
                                             args.sessions, args.jobs) for probe in part]
    by_street: dict[str, list] = {}
    for probe in found:
        by_street.setdefault(probe[0].street.value, []).append(probe)
    rng = random.Random(args.seed)
    probes = []
    for street, share in STREET_SHARES.items():
        pool = by_street.get(street, [])
        want = round(args.count * share)
        if len(pool) < want:
            print(f"only {len(pool)} {street} decisions for {want}: more sessions?")
        probes += rng.sample(pool, min(want, len(pool)))
    save_probes(probes, args.out)
    print(f"{args.out}: {len(probes)} probes ({ {s: len(v) for s, v in by_street.items()} } seen) "
          f"in {time.time() - started:.0f}s")
    return 0


def cmd_signatures(args) -> int:
    from pokerlab.rl.diversity import (
        load_probes,
        model_signature,
        probes_fingerprint,
        save_signature,
    )
    from pokerlab.rl.forking import run_in_copies
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.table_mix import TableMix

    population, _ratings = _population(args.root, args.global_dir, with_anchors=True)
    directory = args.out / "signatures" / probes_fingerprint(args.probes)
    todo = [c for c in population if not (directory / f"{c.label}.npy").is_file()]
    probes = load_probes(args.probes)
    mix = TableMix()

    def work(job: int) -> str:
        candidate = todo[job]
        try:
            model, _ = build_model_from_checkpoint(candidate.path, device="cpu")
        except Exception as error:  # noqa: BLE001
            return f"skip {candidate.label}: {error}"
        save_signature(directory, candidate.label,
                       model_signature(model, probes, big_blind=mix.big_blind, starting_stack=mix.starting_stack))
        return ""

    started = time.time()
    for line in run_in_copies(work, len(todo), args.jobs):
        if line:
            print(line, flush=True)
    print(f"{len(todo)} signatures in {time.time() - started:.0f}s -> {directory}")
    return 0


def _quantiles(values, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> str:
    import numpy as np

    values = np.asarray(values)
    return " ".join(f"{q:.0%}:{np.quantile(values, q):.3f}" for q in qs)


def _style_vector(member) -> list[float] | None:
    from pokerlab.engine.stats import STATS

    style = getattr(member, "style", None) or {}
    if not style:
        return None
    found = []
    for name in STATS:
        events, chances = style.get(name, (0, 0))
        found.append(events / chances if chances else math.nan)
    return found


def cmd_report(args) -> int:
    import numpy as np

    from pokerlab.rl.diversity import (
        distance_matrix,
        load_probes,
        probes_fingerprint,
        redundant_models,
    )

    population, ratings = _population(args.root, args.global_dir, with_anchors=True)
    directory = args.out / "signatures" / probes_fingerprint(args.probes)
    labels, signatures = [], []
    for candidate in population:
        path = directory / f"{candidate.label}.npy"
        if path.is_file():
            labels.append(candidate.label)
            signatures.append(np.load(path).astype(np.float32))
    anchors = {c.label for c in population if "benchmark" in str(c.path)}
    print(f"{len(labels)} models with a signature ({len(anchors & set(labels))} anchors), "
          f"{signatures[0].shape[0]} probes")
    d = distance_matrix(signatures)
    n = len(labels)
    upper = d[np.triu_indices(n, 1)]
    print(f"\ndistance between two models, every pair: {_quantiles(upper)}")
    probes = load_probes(args.probes)
    streets = np.array([o.street.value for o, _ in probes])
    for street in STREET_SHARES:
        rows = streets == street
        part = distance_matrix([s[rows] for s in signatures])
        print(f"  {street:8s} ({rows.sum():4d} probes): {_quantiles(part[np.triu_indices(n, 1)])}")

    # enough probes? two halves against each other
    rng = np.random.default_rng(0)
    order = rng.permutation(signatures[0].shape[0])
    half_a = distance_matrix([s[order[: len(order) // 2]] for s in signatures])[np.triu_indices(n, 1)]
    half_b = distance_matrix([s[order[len(order) // 2 :]] for s in signatures])[np.triu_indices(n, 1)]
    print(f"\ntwo halves of the probes: correlation of the pair distances {np.corrcoef(half_a, half_b)[0, 1]:.3f}")
    masked = d + np.eye(n) * 9
    nn_a = (half_a, half_b)
    del nn_a
    nearest = masked.argmin(axis=1)
    nearest_d = masked.min(axis=1)
    print(f"nearest neighbour's distance: {_quantiles(nearest_d)}")

    rated = [i for i, label in enumerate(labels) if label in ratings and label not in anchors]
    order_by_rating = sorted(rated, key=lambda i: -ratings[labels[i]].rating)
    top = order_by_rating[:100]
    if len(top) > 1:
        sub = d[np.ix_(top, top)]
        print(f"\ntop 100 among themselves: {_quantiles(sub[np.triu_indices(len(top), 1)])}")
    print("nearest-neighbour distance by rating band:")
    for name, part in (("1-100", order_by_rating[:100]), ("101-200", order_by_rating[100:200]),
                       ("201+", order_by_rating[200:])):
        if part:
            print(f"  {name:8s} n {len(part):4d}  {_quantiles(nearest_d[part], (0.1, 0.5, 0.9))}")
    if len(rated) > 2:
        r = np.array([ratings[labels[i]].rating for i in rated])
        pairs = rng.integers(0, len(rated), size=(20000, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        gap = np.abs(r[pairs[:, 0]] - r[pairs[:, 1]])
        dist = d[np.array(rated)[pairs[:, 0]], np.array(rated)[pairs[:, 1]]]
        print(f"correlation of a pair's distance with its rating gap: {np.corrcoef(gap, dist)[0, 1]:.3f}")

    # parents and children (the sweep's observations)
    index = {label: i for i, label in enumerate(labels)}
    sweep = args.global_dir / "sweep"
    kin = []
    for path in sweep.glob("*.json") if sweep.is_dir() else []:
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        child, parent = row.get("label") or path.stem, row.get("parent")
        if child in index and parent in index:
            kin.append(d[index[child], index[parent]])
    if kin:
        print(f"\nparent -> child ({len(kin)} pairs): {_quantiles(kin)}")
        print(f"random pairs:               {_quantiles(upper)}")

    # agreement with the nine statistics of `style`
    styles = {i: _style_vector(ratings[labels[i]]) for i in range(n) if labels[i] in ratings}
    styles = {i: v for i, v in styles.items() if v is not None and not any(math.isnan(x) for x in v)}
    if len(styles) > 2:
        keys = list(styles)
        matrix = np.array([styles[i] for i in keys])
        matrix = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-9)
        pairs = rng.integers(0, len(keys), size=(20000, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        style_d = np.linalg.norm(matrix[pairs[:, 0]] - matrix[pairs[:, 1]], axis=1)
        policy_d = d[np.array(keys)[pairs[:, 0]], np.array(keys)[pairs[:, 1]]]
        print(f"\ncorrelation with the distance between the nine statistics: {np.corrcoef(style_d, policy_d)[0, 1]:.3f} "
              f"({len(keys)} models with a style)")

    _anchor_report(labels, d, anchors, ratings, upper)

    # what a pruning would remove
    store = [i for i in rated]
    store_labels = [labels[i] for i in store]
    store_sigs = [signatures[i] for i in store]
    protected = {labels[i] for i in top}
    removable = set(store_labels) - protected
    print(f"\npruning of the store ({len(store)} rated models, top 100 kept):")
    for fraction in (0.1, 0.25, 0.5):
        count = round(len(removable) * fraction)
        gone = redundant_models(store_labels, store_sigs, removable=removable, count=count,
                                ratings={label: ratings[label].rating for label in store_labels})
        kept = [i for i, label in enumerate(store_labels) if label not in set(gone)]
        sub = distance_matrix([store_sigs[i] for i in kept]) + np.eye(len(kept)) * 9
        by_rating = sorted(removable, key=lambda label: ratings[label].rating)[:count]
        kept_r = [i for i, label in enumerate(store_labels) if label not in set(by_rating)]
        sub_r = distance_matrix([store_sigs[i] for i in kept_r]) + np.eye(len(kept_r)) * 9
        print(f"  remove {fraction:.0%} ({count}): by diversity the nearest neighbour of those left is "
              f"{_quantiles(sub.min(1), (0.1, 0.5))}, mean rating of the removed "
              f"{np.mean([ratings[g].rating for g in gone]):.0f}; by rating {_quantiles(sub_r.min(1), (0.1, 0.5))}, "
              f"mean rating {np.mean([ratings[g].rating for g in by_rating]):.0f}; "
              f"overlap {len(set(gone) & set(by_rating))}")
    nearest_pairs = sorted(range(n), key=lambda i: nearest_d[i])[:10]
    print("\nthe ten closest models and their nearest neighbour:")
    for i in nearest_pairs:
        j = nearest[i]
        ri = ratings[labels[i]].rating if labels[i] in ratings else float("nan")
        rj = ratings[labels[j]].rating if labels[j] in ratings else float("nan")
        print(f"  {nearest_d[i]:.4f}  {labels[i]} ({ri:.0f})  ~  {labels[j]} ({rj:.0f})")
    return 0


def _anchor_report(labels, d, anchors, ratings, upper) -> None:
    """The frozen anchors: how far apart they are, from each other and from the store."""
    import numpy as np

    index = [i for i, label in enumerate(labels) if label in anchors]
    if len(index) < 2:
        return
    rating = {i: ratings[labels[i]].rating if labels[i] in ratings else float("nan") for i in index}
    index.sort(key=lambda i: -rating[i])
    store = [i for i in range(len(labels)) if labels[i] not in anchors]
    sub = d[np.ix_(index, index)]
    print(f"\nanchors ({len(index)}), among themselves: {_quantiles(sub[np.triu_indices(len(index), 1)])}")
    print(f"store pairs, for comparison:  {_quantiles(upper)}")
    print(f"{'#':>3} {'rating':>7} {'nearest anchor':>15} {'d':>6} {'nearest store model':>20} {'d':>6}  label")
    for row, i in enumerate(index):
        others = [j for j in index if j != i]
        a = min(others, key=lambda j: d[i, j])
        m = min(store, key=lambda j: d[i, j]) if store else None
        print(f"{row + 1:3d} {rating[i]:7.0f} {'#' + str(index.index(a) + 1):>15} {d[i, a]:6.3f} "
              f"{'' if m is None else f'{rating_of(ratings, labels[m]):.0f}':>20} "
              f"{'' if m is None else f'{d[i, m]:.3f}':>6}  {labels[i]}")
    print("\ndistance between anchors x 100 (rows and columns in rating order):")
    print("    " + "".join(f"{c + 1:4d}" for c in range(len(index))))
    for row, i in enumerate(index):
        print(f"{row + 1:3d} " + "".join("   ." if i == j else f"{100 * d[i, j]:4.0f}" for j in index))
    gap = np.array([abs(rating[i] - rating[j]) for x, i in enumerate(index) for j in index[x + 1 :]])
    dist = np.array([d[i, j] for x, i in enumerate(index) for j in index[x + 1 :]])
    print(f"correlation of two anchors' distance with their rating gap: {np.corrcoef(gap, dist)[0, 1]:.3f}")
    for threshold in (0.03, 0.05, 0.1):
        groups: list[list[int]] = []
        for i in index:  # single linkage: an anchor joins every group it is that close to
            near = [g for g in groups if any(d[i, j] < threshold for j in g)]
            merged = [i] + [j for g in near for j in g]
            groups = [g for g in groups if g not in near] + [merged]
        sizes = sorted((len(g) for g in groups), reverse=True)
        print(f"groups of anchors closer than {threshold}: {len(groups)} (sizes {sizes})")


def rating_of(ratings, label) -> float:
    return ratings[label].rating if label in ratings else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probes", "signatures", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path("checkpoints"))
        p.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        if name == "probes":
            p.add_argument("--sessions", type=int, default=1600)
            p.add_argument("--hands", type=int, default=50, help="hands a session")
            p.add_argument("--count", type=int, default=10000, help="probes kept")
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--probes", type=Path, default=Path("diversity/probes-v2.pkl"))
            p.add_argument("--out", type=Path, default=Path("checkpoints/studies/diversity"))
    args = parser.parse_args(argv)
    return {"probes": cmd_probes, "signatures": cmd_signatures, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
