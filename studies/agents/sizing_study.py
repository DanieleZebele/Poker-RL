"""Does sizing a raise by every raise bin's probability play better than the drawn bin's own size?

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/sizing_study.py \\
        --model '#1' --sessions 4000 --jobs 40

The same model sits in one seat of a table twice, on the same hands (the table only shuffles with
its own rng, and the stacks and the seat come from the session's seed): once sizing its raises by
the bin it drew (`bin`, how it trained), once by the probability-weighted mean of the open raise
bins (`weighted`, `action_space.weighted_raise_to`). The other seats are the frozen anchors
(`--field anchors`, as the pass that rates a model) or the store's 50 best-rated (`--field top`),
who keep their bins. The difference, session by session, is the sizing's alone: bb/100 with a 95%
interval, by group of table sizes, and how the hero's raises changed (how often it raised and how
big, over the pot).

Writes nothing but its report (and `--save`, if given).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

_POLICIES: dict = {}


def _policy(path: str, sizing: str):
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint

    key = (path, sizing)
    if key not in _POLICIES:
        model, _ = build_model_from_checkpoint(path, device="cpu")
        _POLICIES[key] = make_policy_fn(model, raise_sizing=sizing)
    return _POLICIES[key]


def play_session(job: int, plan: dict) -> dict:
    from pokerlab.engine.actions import ActionType
    from pokerlab.players.base import Player
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.forking import torch_seeded
    from pokerlab.rl.rollout import TableBank

    mix = plan["mix"]
    seed = plan["seed"] + job
    rng = random.Random(seed)
    size = mix.draw_size(rng)
    hero_seat = rng.randrange(size)
    field = rng.sample(plan["field"], size - 1)
    found = {"size": size}
    for sizing in ("bin", "weighted"):
        raises: list[float] = []

        class Watch(Player):
            def __init__(self, inner, raises=raises):
                super().__init__(inner.player_id, inner.name)
                self.inner = inner
                self.raises = raises

            def act(self, observation, legal):
                action = self.inner.act(observation, legal)
                if action.action_type in (ActionType.BET, ActionType.RAISE):
                    to_call = observation.current_bet_to_match - observation.my_current_bet
                    pot = observation.pot_size + to_call
                    self.raises.append((action.amount - observation.current_bet_to_match) / max(pot, 1))
                return action

        with torch_seeded(seed):
            bank = TableBank(mix, random.Random(seed))
            others = iter(field)
            for seat, proxy in enumerate(bank.seats(size)):
                path, policy_sizing = (plan["hero"], sizing) if seat == hero_seat else (next(others), "bin")
                player = RLAgentPlayer(proxy.player_id, proxy.name, policy_fn=_policy(path, policy_sizing),
                                       big_blind=mix.big_blind, starting_stack=mix.starting_stack)
                proxy.inner = Watch(player) if seat == hero_seat else player
            deltas = bank.play_session(size, plan["hands"])
        found[sizing] = deltas[hero_seat] / mix.big_blind
        found[sizing + "_raises"] = raises
    return found


def _interval(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))
    return mean, 1.96 * sd / math.sqrt(n)


def report(rows: list[dict], hands: int) -> str:
    from pokerlab.rl.style_log import SIZE_GROUPS

    lines = [f"{len(rows)} sessions of {hands} hands, the hero's bb/100:"]
    groups = [("tutti", 2, 9)] + [(f"{low}-{high}", low, high) for low, high in SIZE_GROUPS]
    for name, low, high in groups:
        part = [r for r in rows if low <= r["size"] <= high]
        if len(part) < 2:
            continue
        per100 = 100 / hands
        diff = _interval([(r["weighted"] - r["bin"]) * per100 for r in part])
        b = _interval([r["bin"] * per100 for r in part])
        w = _interval([r["weighted"] * per100 for r in part])
        lines.append(f"  {name:6s} n {len(part):5d}  bin {b[0]:+7.1f} +- {b[1]:5.1f}  weighted {w[0]:+7.1f} +- {w[1]:5.1f}"
                     f"  weighted - bin {diff[0]:+6.1f} +- {diff[1]:5.1f}")
    for sizing in ("bin", "weighted"):
        sizes = sorted(x for r in rows for x in r[sizing + "_raises"])
        if sizes:
            q = [sizes[int(f * (len(sizes) - 1))] for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
            lines.append(f"  {sizing:8s} raises {len(sizes) / (len(rows) * hands):.3f} a hand, size over the pot "
                         f"10/25/50/75/90%: " + " ".join(f"{v:.2f}" for v in q))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="#1", help="a label, a path or #N (rank in the global ranking)")
    parser.add_argument("--field", choices=("anchors", "top"), default="anchors")
    parser.add_argument("--sessions", type=int, default=4000)
    parser.add_argument("--hands", type=int, default=100, help="hands a session")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args(argv)

    from duel_power import resolve_model

    from pokerlab.rl.forking import run_in_copies
    from pokerlab.rl.global_arena import discover_benchmark_population, discover_population
    from pokerlab.rl.global_store import load_ranking
    from pokerlab.rl.table_mix import TableMix

    _label, hero_path, _rating = resolve_model(args.model, global_dir=Path("checkpoints/global"),
                                               root=Path("checkpoints"))
    hero = str(hero_path)
    if args.field == "anchors":
        field = [str(c.path) for c in discover_benchmark_population(Path("checkpoints"))]
    else:
        ratings = load_ranking(Path("checkpoints/global")).members
        store = sorted((c for c in discover_population(Path("checkpoints")) if c.label in ratings),
                       key=lambda c: -ratings[c.label].rating)
        field = [str(c.path) for c in store[:51] if str(c.path) != hero][:50]
    plan = {"mix": TableMix(), "hero": hero, "field": field, "hands": args.hands, "seed": args.seed}
    print(f"hero {Path(hero).stem}, field {args.field} ({len(field)} models)", flush=True)
    started = time.time()
    rows = run_in_copies(lambda job: play_session(job, plan), args.sessions, args.jobs)
    print(report(rows, args.hands))
    print(f"({time.time() - started:.0f}s)")
    if args.save:
        args.save.write_text(json.dumps({"hero": hero, "field": args.field, "rows": rows}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
