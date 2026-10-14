"""Play mixed tables of the store's models and write what an observer sees of every player.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents:studies/player_embedding \\
        python studies/player_embedding/embed_generate.py --sessions 1000 --jobs 24 --out checkpoints/studies/player_embedding

Each session seats `size` *different* models (drawn from the store, optionally only the best
`--top` by rating), each with its own **p_show** -- how often it shows a hand it loses -- and, for a
share of them (`--style-share`), a random style pushed on its logits (`rl/styles.py`). It plays
`--session-hands` hands (100 by default, a training table's stay) and writes, per player, the
sequence of its hands as `embed_records` reads them (`embed_data` describes the arrays). Shards of
`--shard-sessions` sessions, `shard-0000.npz`, ...

**The models do not see their opponents' statistics** (unless `--with-opponent-stats`): the
observation they play on carries zeros where the HUD would be, as at the first hand of any table.
What is measured is then how each model plays on its own, not how it reacts to the others -- the
style the embedder is to read. The hands are played one by one (`TableBank.play_hand`, which leaves
the table's tracker to the caller and so has none) instead of as a session (`play_session`, which
starts a tracker). A model trained with the statistics reads "unknown" for everyone, a state it
knows from the first hands of every training table.

**p_show is drawn per player and session**: a quarter never show a lost hand, a quarter always do,
the rest anywhere between. So the embedder cannot assume the muck rule of any client: a shown hand
is a selected sample whose selection it has to learn from the players themselves.

The sessions are independent and each is a function of its number (the models, the table, the
stacks and the cards all come from `seed + number`), so they are played by forked copies of this
process (`pokerlab.rl.forking.run_in_copies`) and the result does not depend on `--jobs`.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import embed_data
from embed_records import seat_hands

P_SHOW_NEVER, P_SHOW_ALWAYS = 0.25, 0.25  # shares of the players that never / always show a loss


def draw_p_show(rng: random.Random) -> float:
    """One player's chance of showing a hand it loses: never, always, or anywhere between."""
    pick = rng.random()
    if pick < P_SHOW_NEVER:
        return 0.0
    if pick < P_SHOW_NEVER + P_SHOW_ALWAYS:
        return 1.0
    return rng.random()


@dataclass
class Plan:
    """Everything one session needs, in picklable primitives."""

    candidates: list[tuple[str, str]]  # (label, path)
    mix: object
    session_hands: int
    seed: int
    style_share: float
    style_spread: float
    style_temperature_spread: float
    style_jitter: float
    style_scales: tuple[float, ...]
    device: str = "cpu"
    opponent_stats: bool = False  # True: the models read their opponents' statistics


def play_session(plan: Plan, number: int, policies: dict) -> dict:
    """One session, as `embed_data.pack` arrays. `policies` caches the models loaded so far."""
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.forking import torch_seeded
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.styles import StyleConfig

    seed = plan.seed + number
    rng = random.Random(seed)
    size = plan.mix.draw_size(rng)
    chosen = rng.sample(plan.candidates, size)
    style_config = StyleConfig(
        share=plan.style_share, spread=plan.style_spread,
        temperature_spread=plan.style_temperature_spread, jitter=plan.style_jitter,
        scales=plan.style_scales,
    )
    p_show = [draw_p_show(rng) for _ in range(size)]
    styles = [style_config.draw(rng) for _ in range(size)]

    with torch_seeded(seed):
        bank = TableBank(plan.mix, random.Random(seed))
        for seat, proxy in enumerate(bank.seats(size)):
            label, path = chosen[seat]
            if path not in policies:
                model, _checkpoint = build_model_from_checkpoint(path, device=plan.device)
                policies[path] = make_policy_fn(model, device=plan.device)
            proxy.inner = RLAgentPlayer(
                proxy.player_id, proxy.name, policy_fn=policies[path],
                big_blind=plan.mix.big_blind, starting_stack=plan.mix.starting_stack,
                style=styles[seat],
            )
        hands = []
        if plan.opponent_stats:
            bank.play_session(size, plan.session_hands, on_hand=hands.append)
        else:
            for _ in range(plan.session_hands):
                bank.play_hand(size)  # no tracker: the models see no statistics
                hands.append(bank.last_hand)

    show_rng = random.Random(seed ^ 0x5EED)
    shown_by = {seat: p for seat, p in enumerate(p_show)}
    sequences: list[list] = [[] for _ in range(size)]
    for hand in hands:
        seats = seat_hands(hand, shown_by, show_rng)
        for seat in range(size):
            sequences[seat].append(seats[seat])
    return embed_data.pack(
        sequences,
        models=[label for label, _path in chosen],
        p_show=p_show,
        styled=[style is not None for style in styles],
        session=number,
        hands=plan.session_hands,
    )


def build_parser() -> argparse.ArgumentParser:
    from pokerlab.rl.styles import DEFAULT_SCALES
    from pokerlab.rl.table_mix import add_table_arguments

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sessions", type=int, default=1000, help="sessions to play")
    parser.add_argument("--shard-sessions", type=int, default=200, help="sessions per shard file")
    parser.add_argument("--jobs", type=int, default=1, help="forked copies playing sessions at once")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/studies/player_embedding"))
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    parser.add_argument("--top", type=int, default=0, help="only the N best rated models (0: every model)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-opponent-stats", action="store_true",
                        help="let the models read their opponents' statistics (off by default: the "
                        "observation carries zeros there, so each model plays without reacting)")
    parser.add_argument("--style-share", type=float, default=0.3,
                        help="share of the seats given a random style (rl/styles.py)")
    parser.add_argument("--style-spread", type=float, default=0.75)
    parser.add_argument("--style-temperature-spread", type=float, default=0.2)
    parser.add_argument("--style-jitter", type=float, default=0.1)
    parser.add_argument("--style-scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    add_table_arguments(parser)
    parser.set_defaults(session_hands=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    from pokerlab.rl.forking import run_in_copies
    from pokerlab.rl.global_arena import discover_population
    from pokerlab.rl.global_store import load_ranking
    from pokerlab.rl.table_mix import table_mix_from_args

    args = build_parser().parse_args(argv)
    mix = table_mix_from_args(args)
    population = discover_population(args.root)
    if args.top > 0:
        ratings = {label: member.rating for label, member in load_ranking(args.global_dir).members.items()}
        population = sorted(population, key=lambda c: -ratings.get(c.label, 0.0))[: args.top]
    if len(population) < mix.max_players:
        print(f"{len(population)} models in {args.root}/models: a table of {mix.max_players} needs more")
        return 1
    plan = Plan(
        candidates=[(c.label, str(c.path)) for c in population], mix=mix,
        session_hands=args.session_hands, seed=args.seed, style_share=args.style_share,
        style_spread=args.style_spread, style_temperature_spread=args.style_temperature_spread,
        style_jitter=args.style_jitter, style_scales=tuple(args.style_scales),
        opponent_stats=args.with_opponent_stats,
    )
    print(f"{len(population)} models, {args.sessions} sessions of {args.session_hands} hands, "
          f"{args.jobs} copies, opponent statistics "
          f"{'read' if args.with_opponent_stats else 'zero'} -> {args.out}", flush=True)

    policies: dict = {}  # filled in each copy as it loads models; the parent's stays empty
    started = time.time()
    shard = 0
    for first in range(0, args.sessions, args.shard_sessions):
        count = min(args.shard_sessions, args.sessions - first)
        parts = run_in_copies(
            lambda job, first=first: play_session(plan, first + job, policies), count, args.jobs
        )
        arrays = embed_data.concatenate(parts)
        path = args.out / f"shard-{shard:04d}.npz"
        embed_data.save_shard(path, arrays)
        sequences = len(arrays["model"])
        showed = float((arrays["show"] == 1).mean())
        mucked = float((arrays["show"] == 2).mean())
        print(f"{path.name}: {count} sessions, {sequences} sequences, showed {showed:.1%} / "
              f"mucked {mucked:.1%} of hands, {time.time() - started:.0f}s", flush=True)
        shard += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
