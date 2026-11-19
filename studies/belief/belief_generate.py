"""Play mixed tables of the store's models and keep every hand, every hole card included.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/belief \\
        python studies/belief/belief_generate.py --sessions 2000 --jobs 44 --out checkpoints/studies/belief/data

A session is a training table's stay: one size drawn from the mixture, `--session-hands` (200)
hands, the same players throughout and a live `StatsTracker`, so the models read each other's
statistics as they do in training and the belief network sees the opponents the policy will meet.
Its seats are drawn like a training pool: `--top-share` of them from the `--top` best-rated models,
the rest from the whole store (`--with-anchors`: the frozen anchors too), and `--style-share` of them
given a random style (`rl/styles.py`). **The style is visible only at the calibrated scales**
(`--style-scales 5 8 8 4 8`, the fleet's `config.toml`): at the default 1.0 a push barely moves a
model (`style_calibration.py`), which is what the first data set (`data/`) was played with.

The output is the raw hands, one JSON line per session (`{"session", "size", "models", "styled",
"hands": [...]}`, the hands as `engine/history.py` writes them), in gzip shards of
`--shard-sessions` sessions. `belief_data.py` turns them into arrays: keeping the hands means a
change of featurisation does not need the hands played again.

Each session is a function of its number (`seed + number`), so the result does not depend on
`--jobs`; a rerun skips the shards already written.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Plan:
    top: list[tuple[str, str]]  # (label, path), best rated first
    everyone: list[tuple[str, str]]
    top_share: float
    mix: object
    session_hands: int
    seed: int
    style_share: float
    style_spread: float
    style_temperature_spread: float
    style_jitter: float
    style_scales: tuple[float, ...]


def draw_seats(plan: Plan, size: int, rng: random.Random) -> list[tuple[str, str]]:
    """`size` different models, each from the top with probability `top_share`."""
    chosen: list[tuple[str, str]] = []
    while len(chosen) < size:
        source = plan.top if rng.random() < plan.top_share else plan.everyone
        pick = rng.choice(source)
        if pick not in chosen:
            chosen.append(pick)
    return chosen


def play_session(plan: Plan, number: int, policies: dict) -> dict:
    from pokerlab.engine.history import _to_jsonable
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.forking import torch_seeded
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.styles import StyleConfig

    seed = plan.seed + number
    rng = random.Random(seed)
    size = plan.mix.draw_size(rng)
    chosen = draw_seats(plan, size, rng)
    style_config = StyleConfig(
        share=plan.style_share, spread=plan.style_spread,
        temperature_spread=plan.style_temperature_spread, jitter=plan.style_jitter,
        scales=plan.style_scales,
    )
    styles = [style_config.draw(rng) for _ in range(size)]
    hands = []
    with torch_seeded(seed):
        bank = TableBank(plan.mix, random.Random(seed))
        for seat, proxy in enumerate(bank.seats(size)):
            _label, path = chosen[seat]
            if path not in policies:
                model, _checkpoint = build_model_from_checkpoint(path, device="cpu")
                policies[path] = make_policy_fn(model, device="cpu")
            proxy.inner = RLAgentPlayer(
                proxy.player_id, proxy.name, policy_fn=policies[path],
                big_blind=plan.mix.big_blind, starting_stack=plan.mix.starting_stack,
                style=styles[seat],
            )
        bank.play_session(size, plan.session_hands, on_hand=lambda hand: hands.append(_to_jsonable(hand)))
    return {
        "session": number,
        "size": size,
        "models": [label for label, _path in chosen],
        "styled": [style is not None for style in styles],
        "hands": hands,
    }


def read_shard(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_parser() -> argparse.ArgumentParser:
    from pokerlab.rl.styles import DEFAULT_SCALES
    from pokerlab.rl.table_mix import add_table_arguments

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--shard-sessions", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/studies/belief/data"))
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    parser.add_argument("--top", type=int, default=100, help="the best-rated models a top seat is drawn from")
    parser.add_argument("--top-share", type=float, default=0.7, help="share of the seats drawn from the top")
    parser.add_argument("--with-anchors", action="store_true",
                        help="the frozen anchors join the models a seat is drawn from (they play unlike the store)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--style-share", type=float, default=0.3)
    parser.add_argument("--style-spread", type=float, default=0.75)
    parser.add_argument("--style-temperature-spread", type=float, default=0.2)
    parser.add_argument("--style-jitter", type=float, default=0.1)
    parser.add_argument("--style-scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    add_table_arguments(parser)
    parser.set_defaults(session_hands=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    from pokerlab.rl.forking import run_in_copies
    from pokerlab.rl.global_arena import discover_population
    from pokerlab.rl.global_store import load_ranking
    from pokerlab.rl.table_mix import table_mix_from_args

    args = build_parser().parse_args(argv)
    mix = table_mix_from_args(args)
    population = discover_population(args.root)
    if args.with_anchors:
        from pokerlab.rl.global_arena import discover_benchmark_population

        population += discover_benchmark_population(args.root)
    ratings = {label: member.rating for label, member in load_ranking(args.global_dir).members.items()}
    ranked = sorted((c for c in population if c.label in ratings), key=lambda c: -ratings[c.label])
    if len(population) < 2 * mix.max_players or len(ranked) < mix.max_players:
        print(f"{len(population)} models ({len(ranked)} rated): too few for a table of {mix.max_players}")
        return 1
    plan = Plan(
        top=[(c.label, str(c.path)) for c in ranked[: args.top]],
        everyone=[(c.label, str(c.path)) for c in population],
        top_share=args.top_share, mix=mix, session_hands=args.session_hands, seed=args.seed,
        style_share=args.style_share, style_spread=args.style_spread,
        style_temperature_spread=args.style_temperature_spread, style_jitter=args.style_jitter,
        style_scales=tuple(args.style_scales),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps({
        "argv": sys.argv[1:], "top": [label for label, _ in plan.top], "store": len(population),
    }, indent=1))
    print(f"{len(population)} models ({len(plan.top)} at the top), {args.sessions} sessions of "
          f"{args.session_hands} hands, {args.jobs} copies -> {args.out}", flush=True)

    policies: dict = {}
    started = time.time()
    for shard, first in enumerate(range(0, args.sessions, args.shard_sessions)):
        path = args.out / f"hands-{shard:04d}.jsonl.gz"
        if path.exists():
            continue
        count = min(args.shard_sessions, args.sessions - first)
        sessions = run_in_copies(lambda job, first=first: play_session(plan, first + job, policies), count, args.jobs)
        partial = path.with_name("." + path.name + ".partial")
        with gzip.open(partial, "wt", encoding="utf-8") as handle:
            for session in sessions:
                handle.write(json.dumps(session, separators=(",", ":")) + "\n")
        partial.rename(path)
        hands = sum(len(s["hands"]) for s in sessions)
        print(f"{path.name}: {count} sessions, {hands} hands, {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
