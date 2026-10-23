"""Does searching a decision beat just playing the policy? (`rl/search.py`)

The same model sits in the hero's seat twice, on the very same hands: once playing its policy, and
once with the search on the streets chosen. Everything else is identical -- the stacks, the
button, the cards (`Table` touches its `random.Random` only to shuffle) and the seed of the
network's sampling -- so the two arms play the same hand until the search first speaks, and the
difference of their results, hand by hand, is the search's alone. The paired difference
removes nearly all the luck of the cards, which is the only reason a few hundred hands can say
anything (the same idea as the duplicate decks of `duel_power.py`).

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/search_study.py \\
        --model '#1' --players 2 --hands 400 --worlds 100 --streets river --jobs 20

**Always run the null control** (`--null`): the "search" arm then plays the policy too, with
another seed for the network's sampling. It must read ~0 with an interval that holds 0; it is the
noise floor the real difference has to be larger than.

What the search is allowed to know. The models in the opponents' seats are `--opponent-model`
(the same as `--model` unless said). The search *believes* the opponents are `--sim-model` (the
hero's own model unless said). Same model on both: the search is right about who it plays, the
most it can gain. Different: how much of that survives being wrong about them, which is what a
real table is. Only the hands where the search spoke can differ, so the report gives the
difference over every hand and over those.

Slow by nature: a searched decision plays `worlds x candidates` rollouts, seconds each.
`--jobs` forked copies share the loaded models; the rest is the machine's cores.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from duel_power import DEFAULT_GLOBAL_DIR, DEFAULT_ROOT, resolve_model

from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import Street
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.forking import run_in_copies, torch_seeded
from pokerlab.rl.search import BELIEFS, SearchConfig, Searcher, SearchPlayer, SimModel
from pokerlab.rl.table_mix import DEFAULT_TABLE_WEIGHTS, SIZES, TableMix

BIG_BLIND = 100
SMALL_BLIND = 50
SIZE_GROUPS = ((2, 3), (4, 6), (7, 9))  # as the training's style lines (`style_log.SIZE_GROUPS`)
STREET_NAMES = {"preflop": Street.PREFLOP, "flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}

_LOADED: dict[tuple[str, str], tuple[object, int]] = {}


def load_policy(path: str, device: str):
    """`(policy_fn, starting_stack)` of a checkpoint, once per process: the stack constant is the
    one the model was trained with, as at any table that plays it (`feature_stack_bb`)."""
    key = (path, device)
    if key not in _LOADED:
        from pokerlab.rl.policy import make_policy_fn
        from pokerlab.rl.ppo import build_model_from_checkpoint
        from pokerlab.rl.table_mix import feature_stack_bb

        model, checkpoint = build_model_from_checkpoint(path, device=device)
        _LOADED[key] = (make_policy_fn(model, device=device), round(feature_stack_bb(checkpoint) * BIG_BLIND))
    return _LOADED[key]


_SEARCH_LOADED: dict[tuple[str, str], object] = {}


def load_search_policy(path: str, device: str):
    """The same model as `load_policy`, as the faster policy a search plays its rollouts with
    (`make_search_policy_fn`: the same probabilities, cheaper to ask), once per process."""
    key = (path, device)
    if key not in _SEARCH_LOADED:
        from pokerlab.rl.policy import make_search_policy_fn
        from pokerlab.rl.ppo import build_model_from_checkpoint

        model, _checkpoint = build_model_from_checkpoint(path, device=device)
        _SEARCH_LOADED[key] = make_search_policy_fn(model, device=device)
    return _SEARCH_LOADED[key]


def make_sim_player(path: str, device: str, fast: bool):
    """Who sits in the search's simulated seats: the model of `path`, with the faster policy if
    `fast`. The real seats keep `make_player`, so a paired comparison is untouched."""
    if not fast:
        return make_player(path, device)
    policy = load_search_policy(path, device)
    stack = load_policy(path, device)[1]
    return lambda player_id, name: RLAgentPlayer(
        player_id, name, policy_fn=policy, big_blind=BIG_BLIND, starting_stack=stack
    )


_LEAVES: dict[tuple[str, str], object] = {}


def load_leaf_value(path: str, device: str):
    """The critic of the model of `path` as a search's `leaf_value` (`make_leaf_value`), with the
    equity network it was trained with (named in its metadata) and its reward scale, once per
    process."""
    key = (path, device)
    if key not in _LEAVES:
        from pokerlab.rl.equity_net import (
            encoder_config,
            equity_net_from_checkpoint,
            load_equity_checkpoint,
        )
        from pokerlab.rl.policy import make_leaf_value
        from pokerlab.rl.ppo import build_model_from_checkpoint
        from pokerlab.rl.table_mix import feature_stack_bb

        model, checkpoint = build_model_from_checkpoint(path, device=device)
        settings = (checkpoint.get("metadata") or {}).get("settings") or {}
        equity_path = settings.get("equity_model")
        if not equity_path or not Path(equity_path).is_file():
            raise SystemExit(f"{path}: the equity network its critic was trained with is not known or not on disk")
        saved = load_equity_checkpoint(equity_path)
        if encoder_config(saved) != model.equity:
            raise SystemExit(f"{equity_path} is not the equity network of {path}")
        # The training reward is the result in big blinds times big blind / deepest stack.
        _LEAVES[key] = make_leaf_value(
            model, equity_net_from_checkpoint(saved), value_unit=feature_stack_bb(checkpoint), device=device
        )
    return _LEAVES[key]


def make_searcher(
    path: str, device: str, fast: bool, threads: int, config: SearchConfig, rng: random.Random
) -> Searcher:
    """The search of a study: its simulated seats are the model of `path`; with `threads` > 1 they
    ask it in batches (`SimModel`, the faster policy's `batch_distribution`); with `config.leaf`
    the rollouts stop at the hero's next decision and ask that model's critic."""
    leaf_value = load_leaf_value(path, device) if config.leaf else None
    if threads <= 1:
        return Searcher(make_sim_player(path, device, fast), config, rng, leaf_value=leaf_value)
    policy = load_search_policy(path, device)
    model = SimModel(
        batch_distribution=policy.batch_distribution,
        feature_version=policy.feature_version,
        big_blind=BIG_BLIND,
        starting_stack=load_policy(path, device)[1],
    )
    return Searcher(None, config, rng, batched=model, threads=threads, leaf_value=leaf_value)


def add_spending_arguments(parser: argparse.ArgumentParser) -> None:
    """How the search spends less (`SearchConfig`): the same options in every study."""
    group = parser.add_argument_group("spending less")
    group.add_argument("--optimized", action="store_true",
                       help="what was measured to leave the quality as it was, 3-4x faster: "
                            "--stop-z 3 --dedup --fast-sim --threads 16")
    group.add_argument("--stop-z", type=float, default=None,
                       help="stop and drop candidates more than this many standard errors behind (0: off)")
    group.add_argument("--min-worlds", type=int, default=100)
    group.add_argument("--check-every", type=int, default=50)
    group.add_argument("--dedup", action="store_true", default=None,
                       help="a world drawn twice is played once, counted twice; the river's range hands as worlds")
    group.add_argument("--fast-sim", action="store_true", default=None,
                       help="the rollouts' policy remembers the card encoder and skips Categorical")
    group.add_argument("--branch", action="store_true",
                       help="play every answer to the candidate instead of drawing one")
    group.add_argument("--branch-min", type=float, default=0.02)
    group.add_argument("--min-pot-bb", type=float, default=0.0, help="no search in a smaller pot")
    group.add_argument("--leaf", action="store_true",
                       help="a rollout stops at the hero's next decision and the model's critic values it")
    group.add_argument("--threads", type=int, default=None,
                       help="rollouts in this many threads, the model asked in batches (one core still: "
                            "the engine is Python; the gain is the forward passes done together)")


def threads_of(args: argparse.Namespace) -> int:
    """`--threads`, or 16 with `--optimized`, else 1."""
    if args.threads is not None:
        return args.threads
    return 16 if args.optimized else 1


def spending_of(args: argparse.Namespace) -> tuple[dict, bool]:
    """`(SearchConfig keywords, fast_sim)` from the options, `--optimized` filling in the ones not given."""
    optimized = args.optimized
    stop_z = args.stop_z if args.stop_z is not None else (3.0 if optimized else 0.0)
    dedup = args.dedup if args.dedup is not None else optimized
    fast = args.fast_sim if args.fast_sim is not None else optimized
    return (
        {
            "stop_z": stop_z,
            "min_worlds": args.min_worlds,
            "check_every": args.check_every,
            "dedup": dedup,
            "branch": args.branch,
            "branch_min": args.branch_min,
            "min_pot_bb": args.min_pot_bb,
            "leaf": args.leaf,
        },
        fast,
    )


def make_player(path: str, device: str):
    policy, stack = load_policy(path, device)
    return lambda player_id, name: RLAgentPlayer(
        player_id, name, policy_fn=policy, big_blind=BIG_BLIND, starting_stack=stack
    )


@dataclass(frozen=True)
class Job:
    hero_model: str
    opponent_model: str
    sim_model: str
    players: int
    seed: int
    stack_min_bb: float
    stack_max_bb: float
    config: SearchConfig
    null: bool
    device: str
    table_weights: tuple[float, ...] | None = None  # a size drawn every hand; None: `players`
    fast_sim: bool = False  # the rollouts use the faster policy (`make_search_policy_fn`)
    threads: int = 1  # rollouts in threads, the model asked in batches
    # Opponents drawn every hand from these models instead of `opponent_model` (`--field anchors`), and
    # who the search believes sits in each seat (`--believe`): `sim` (every seat is `sim_model`),
    # `exact` (the model really there), `top` (a model drawn from `believed_pool`), `style` (the
    # model of the store whose style is nearest the real one's: `style_match`, real -> believed).
    field: tuple[str, ...] = ()
    believe: str = "sim"
    believed_pool: tuple[str, ...] = ()
    style_match: tuple[tuple[str, str], ...] = ()


@dataclass
class HandOutcome:
    hand: int
    hero_seat: int
    search_bb: float  # the hero's result with the search on
    base_bb: float  # ... playing the policy, on the same hand
    searched: int  # decisions the search answered
    changed: int  # ... that were not the policy's likeliest bin
    declined_ess: int
    declined_other: int
    worlds_failed: int
    ess_total: float
    worlds_pruned: int
    worlds_kept: int
    players: int = 0  # the table's size (0 in runs saved before the mix)


def deal_of(
    seed_base: int,
    hand: int,
    players: int,
    stack_min_bb: float,
    stack_max_bb: float,
    table_weights: tuple[float, ...] | None = None,
) -> tuple[int, list[int], int, int]:
    """The seed, the stacks, the hero's seat and the table size of a hand. One rule for every study
    that wants to play the same hands as this one (`search_distill.py`): the same `--seed` (and the
    same table choice) is the same hands.

    A fixed size rotates the hero through the seats (the first hand of a table puts the button on
    seat 0, so this rotates the position). A mix draws the size with `table_weights` (one weight a
    size, 2-9, as `config.toml`) and the hero's seat at random, from a stream of their own, so a
    fixed-size run deals exactly what it dealt before the mix existed."""
    seed = seed_base * 1_000_003 + hand
    rng = random.Random(seed)
    if table_weights is None:
        size, hero = players, hand % players
    else:
        chooser = random.Random(seed ^ 0x7AB1E5)
        size = chooser.choices(SIZES, weights=table_weights)[0]
        hero = chooser.randrange(size)
    stacks = [max(BIG_BLIND, round(rng.uniform(stack_min_bb, stack_max_bb) * BIG_BLIND)) for _ in range(size)]
    return seed, stacks, hero, size


def add_players_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--players", default="2",
                        help="a table size, 2-9, or 'mix': a size drawn every hand with the training's weights "
                             "(table_weights in --config and in this machine's config.<machine>.toml)")
    parser.add_argument("--table-weights", type=float, nargs=len(SIZES), default=None,
                        help="the mix's weights for the sizes 2-9, in place of the config's")
    parser.add_argument("--config", type=Path, default=Path("config.toml"), help="where 'mix' reads the weights")


def table_choice(args: argparse.Namespace) -> tuple[int, tuple[float, ...] | None]:
    """`(players, weights)`: a fixed size and None, or the largest size of the mix and its weights."""
    if args.players != "mix":
        players = int(args.players)
        if players not in SIZES:
            raise SystemExit(f"--players must be 2-9 or 'mix', got {args.players}")
        return players, None
    weights = args.table_weights
    if weights is None:
        from pokerlab.config import machine_name, override_path, read_config

        values = read_config(args.config) if args.config.is_file() else {}
        own = override_path(args.config, machine_name([]))
        if own is not None and own.is_file():
            values.update(read_config(own))
        weights = values.get("table_weights", DEFAULT_TABLE_WEIGHTS)
    mix = TableMix(weights=tuple(weights))  # refuses a wrong count or all zeros
    return mix.max_players, mix.weights


def players_text(players: int, weights: tuple[float, ...] | None) -> str:
    if weights is None:
        return f"{players} players"
    total = sum(weights)
    shares = " ".join(f"{size}:{w / total:.0%}" for size, w in zip(SIZES, weights, strict=True) if w > 0)
    return f"a mix of tables ({shares})"


def play_pair(job: Job, hand: int) -> HandOutcome:
    """One hand, played with and without the search."""
    seed, stacks, hero, n = deal_of(job.seed, hand, job.players, job.stack_min_bb, job.stack_max_bb,
                                    job.table_weights)
    game = GameConfig(num_players=n, starting_stack=round(job.stack_max_bb * BIG_BLIND),
                      small_blind=SMALL_BLIND, big_blind=BIG_BLIND)
    hero_player = make_player(job.hero_model, job.device)
    others = [seat for seat in range(n) if seat != hero]
    if job.field:
        drawn = random.Random(seed ^ 0xF1E1D)
        chosen = drawn.sample(job.field, n - 1) if len(job.field) >= n - 1 else drawn.choices(job.field, k=n - 1)
        paths = dict(zip(others, chosen, strict=True))
    else:
        paths = dict.fromkeys(others, job.opponent_model)
    seat_models = None
    if job.believe != "sim":
        guess = random.Random(seed ^ 0xBE11EF)
        matched = dict(job.style_match)
        believed = {
            seat: path if job.believe == "exact"
            else matched[path] if job.believe == "style"
            else guess.choice(job.believed_pool)
            for seat, path in paths.items()
        }
        seat_models = {seat: make_sim_player(path, job.device, job.fast_sim) for seat, path in believed.items()}
        seat_models[hero] = make_sim_player(job.sim_model, job.device, job.fast_sim)

    def play(with_search: bool, torch_seed: int):
        searcher = make_searcher(job.sim_model, job.device, job.fast_sim, job.threads, job.config,
                                 random.Random(seed ^ 0x5EA4C4))
        players = []
        for seat in range(n):
            if seat != hero:
                players.append(make_player(paths[seat], job.device)(f"o{seat}", f"O{seat}"))
                continue
            base = hero_player("hero", "Hero")
            players.append(SearchPlayer("hero", "Hero", base, searcher, seat_models) if with_search else base)
        table = Table(game, players, rng=random.Random(seed))
        table.stacks = list(stacks)
        with torch_seeded(torch_seed):
            table.play_hand()
        return (table.stacks[hero] - stacks[hero]) / BIG_BLIND, searcher.counts

    search_bb, counts = play(with_search=not job.null, torch_seed=seed)
    base_bb, _ = play(with_search=False, torch_seed=seed + (7 if job.null else 0))
    return HandOutcome(
        hand=hand, hero_seat=hero, search_bb=search_bb, base_bb=base_bb,
        searched=counts.searched, changed=counts.changed, declined_ess=counts.declined_ess,
        declined_other=counts.declined_other, worlds_failed=counts.worlds_failed, ess_total=counts.ess_total,
        worlds_pruned=counts.worlds_pruned, worlds_kept=counts.worlds_kept, players=n,
    )


def believed_models(job: Job, args: argparse.Namespace) -> Job:
    """`job` with the models a `top` or `style` belief draws from: the `--believe-top-n` best-rated
    models of the store with a file (anchors left out), and for `style` the one of the store models
    with a measured style (`style_hands` over 5,000) nearest each opponent's own (distance between
    the nine rates)."""
    from dataclasses import replace

    from pokerlab.rl.global_store import load_ranking

    ranking = load_ranking(args.global_dir)
    store = [member for member in ranking.ranked() if not member.frozen and Path(member.ref).is_file()]
    if args.believe == "top":
        return replace(job, believed_pool=tuple(member.ref for member in store[: args.believe_top_n]))

    def rates(member) -> list[float] | None:
        if not member.style or member.style_hands < 5_000:
            return None
        return [events / opportunities if opportunities else 0.0 for events, opportunities in member.style.values()]

    by_label = {member.label: member for member in ranking.members.values()}
    candidates = [(member.ref, rates(member)) for member in store if rates(member) is not None]
    real = job.field or (job.opponent_model,)
    pairs = []
    for path in real:
        own = rates(by_label[Path(path).stem])
        nearest = min(candidates, key=lambda c: sum((a - b) ** 2 for a, b in zip(own, c[1], strict=True)))
        pairs.append((path, nearest[0]))
        print(f"  {Path(path).stem} is believed to be {Path(nearest[0]).stem}", flush=True)
    return replace(job, style_match=tuple(pairs))


def anchor_field(directory: Path) -> tuple[str, ...]:
    from pokerlab.rl.benchmark import anchor_paths

    return tuple(str(path) for path in anchor_paths(directory))


def interval(values: list[float]) -> tuple[float, float]:
    """Mean and the half-width of its 95% confidence interval."""
    if len(values) < 2:
        return (values[0] if values else 0.0, float("inf"))
    return statistics.fmean(values), 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def header(job: Job) -> str:
    who = (
        f"{players_text(job.players, job.table_weights)}, hero {job.hero_model}, opponents "
        f"{f'{len(job.field)} anchors drawn every hand' if job.field else job.opponent_model}, search believes "
        f"{job.sim_model if job.believe == 'sim' else 'by ' + job.believe}"
    )
    how = (
        f"search: {'every world' if job.config.exhaustive and not job.config.ranges else str(job.config.worlds) + ' worlds'}"
        f"{' drawn from per-opponent ranges' if job.config.ranges else ''}, "
        f"{job.config.candidates} candidates, belief {job.config.belief}, "
        f"drop under {job.config.world_threshold:.2%}, min ess {job.config.min_ess:g}, "
        f"streets {sorted(s.name.lower() for s in job.config.streets)}"
        + (f", stop at {job.config.stop_z:g} se" if job.config.stop_z else "")
        + (", duplicates folded" if job.config.dedup else "")
        + (", answers branched" if job.config.branch else "")
        + (f", pots from {job.config.min_pot_bb:g} bb" if job.config.min_pot_bb else "")
        + (", fast rollouts" if job.fast_sim else "")
        + (f", {job.threads} threads" if job.threads > 1 else "")
        + (", critic at the next decision" if job.config.leaf else "")
        + ("   [NULL CONTROL]" if job.null else "")
    )
    return f"{who}\n{how}"


def summary(outcomes: list[HandOutcome]) -> str:
    diffs = [o.search_bb - o.base_bb for o in outcomes]
    acted = [o for o in outcomes if o.searched > 0]
    searched = sum(o.searched for o in outcomes)
    per = max(searched, 1)
    lines = [f"{len(outcomes)} hands"]
    for name, rows in (("every hand", outcomes), (f"the {len(acted)} hands where it spoke", acted)):
        if not rows:
            lines.append(f"{name}: none")
            continue
        d = [o.search_bb - o.base_bb for o in rows]
        mean, half = interval(d)
        lines.append(
            f"{name:34} search {statistics.fmean(o.search_bb for o in rows) * 100:+9.1f}  "
            f"policy {statistics.fmean(o.base_bb for o in rows) * 100:+9.1f}   "
            f"difference {mean * 100:+8.1f} +- {half * 100:6.1f} bb/100"
        )
    sizes = {o.players for o in outcomes}
    if len(sizes) > 1:
        for low, high in SIZE_GROUPS:
            rows = [o for o in outcomes if low <= o.players <= high]
            if rows:
                mean, half = interval([o.search_bb - o.base_bb for o in rows])
                lines.append(f"{f'{low}-{high} players ({len(rows)} hands)':34} "
                             f"difference {mean * 100:+8.1f} +- {half * 100:6.1f} bb/100")
    changed = sum(o.changed for o in outcomes)
    kept = sum(o.worlds_kept for o in outcomes) / per
    pruned = sum(o.worlds_pruned for o in outcomes) / per
    lines += [
        "",
        (
            f"decisions searched {searched}, changed {changed} ({100 * changed / per:.0f}%), "
            f"mean effective worlds {sum(o.ess_total for o in outcomes) / per:.1f}"
        ),
        f"worlds per searched decision: played {kept:.0f}, dropped under the threshold {pruned:.0f}",
        (
            f"declined: too few effective worlds {sum(o.declined_ess for o in outcomes)}, "
            f"nothing to choose or not rebuildable {sum(o.declined_other for o in outcomes)}; "
            f"worlds whose replay failed {sum(o.worlds_failed for o in outcomes)}"
        ),
    ]
    mean, half = interval(diffs)
    verdict = "above the noise" if abs(mean) > half else "within the noise"
    lines.append(f"over every hand the difference is {verdict} ({mean * 100:+.1f} +- {half * 100:.1f} bb/100).")
    return "\n".join(lines)


def report(outcomes: list[HandOutcome], job: Job) -> str:
    return header(job) + "\n\n" + summary(outcomes)


def combine(paths: list[Path]) -> str:
    """One report over several saved runs (other seeds, other hands): they are independent samples."""
    outcomes: list[HandOutcome] = []
    first = None
    for path in paths:
        saved = json.loads(path.read_text())
        first = first or saved["header"]
        outcomes += [HandOutcome(**row) for row in saved["hands"]]
    return f"{first}\n[{len(paths)} runs combined]\n\n{summary(outcomes)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="the hero: a path, a label or #N in the ranking")
    parser.add_argument("--opponent-model", default=None, help="who really sits opposite (default: --model)")
    parser.add_argument("--sim-model", default=None, help="who the search believes they are (default: --model)")
    add_players_arguments(parser)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--belief", choices=BELIEFS, default="weighted")
    parser.add_argument("--min-ess", type=float, default=8.0)
    parser.add_argument("--exhaustive", action="store_true",
                        help="every world there is (a heads-up river has 990), where there are few enough")
    parser.add_argument("--ranges", action="store_true",
                        help="weigh each opponent's hands on its own, then draw --worlds worlds from the ranges")
    parser.add_argument("--world-threshold", type=float, default=0.0,
                        help="drop a world once the opponents' actions replayed so far are this unlikely (0.001 = 0.1%%)")
    parser.add_argument("--relative-threshold", action="store_true",
                        help="with --ranges: drop a hand this much less likely than the range's likeliest")
    parser.add_argument("--streets", default="river", help="comma list of preflop,flop,turn,river")
    add_spending_arguments(parser)
    parser.add_argument("--stack-min-bb", type=float, default=20.0)
    parser.add_argument("--stack-max-bb", type=float, default=100.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--null", action="store_true", help="control: the search arm plays the policy too")
    parser.add_argument("--field", choices=("anchors",), default=None,
                        help="opponents drawn every hand from the frozen anchors instead of --opponent-model")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("checkpoints/benchmark"))
    parser.add_argument("--believe", choices=("sim", "exact", "top", "style"), default="sim",
                        help="who the search believes sits in each seat: --sim-model, the model really there, "
                             "a model drawn from the --believe-top-n best, or the store model of the nearest style")
    parser.add_argument("--believe-top-n", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--save", type=Path, default=None, help="write every hand's outcome here (JSON)")
    parser.add_argument("--combine", type=Path, nargs="+", default=None,
                        help="saved runs to report together, without playing anything")
    args = parser.parse_args(argv)
    if args.combine:
        print(combine(args.combine))
        return 0

    if args.model is None:
        parser.error("--model is required (unless --combine)")
    args.players, args.table_weights = table_choice(args)
    streets = frozenset(STREET_NAMES[name.strip()] for name in args.streets.split(","))
    paths = {}
    for spec in {args.model, args.opponent_model or args.model, args.sim_model or args.model}:
        label, path, rating = resolve_model(spec, global_dir=args.global_dir, root=args.root)
        paths[spec] = str(path)
        print(f"{spec}: {label}" + (f" (rating {rating:.0f})" if rating is not None else ""), flush=True)
    job = Job(
        hero_model=paths[args.model],
        opponent_model=paths[args.opponent_model or args.model],
        sim_model=paths[args.sim_model or args.model],
        players=args.players, seed=args.seed, stack_min_bb=args.stack_min_bb, stack_max_bb=args.stack_max_bb,
        config=SearchConfig(**spending_of(args)[0], worlds=args.worlds, candidates=args.candidates, belief=args.belief,
                            min_ess=args.min_ess, streets=streets, exhaustive=args.exhaustive,
                            world_threshold=args.world_threshold, ranges=args.ranges,
                            relative_threshold=args.relative_threshold),
        null=args.null, device=args.device, table_weights=args.table_weights, fast_sim=spending_of(args)[1],
        threads=threads_of(args),
        field=anchor_field(args.benchmark_dir) if args.field == "anchors" else (),
        believe=args.believe,
    )
    if args.believe in ("top", "style"):
        job = believed_models(job, args)
    for path in {*job.field, *job.believed_pool, *(b for _a, b in job.style_match)}:
        load_policy(path, job.device)
        if job.fast_sim:
            load_search_policy(path, job.device)
    load_policy(job.hero_model, job.device)  # before the fork: the copies share the loaded models
    load_policy(job.opponent_model, job.device)
    load_policy(job.sim_model, job.device)
    if job.fast_sim or job.threads > 1:
        load_search_policy(job.sim_model, job.device)
    if job.config.leaf:
        load_leaf_value(job.sim_model, job.device)

    started = time.time()

    def progress(hand: int, outcome: HandOutcome) -> None:
        if (hand + 1) % max(1, args.hands // 20) == 0:
            print(f"  {hand + 1}/{args.hands} hands, {time.time() - started:.0f} s", file=sys.stderr, flush=True)

    outcomes = run_in_copies(lambda hand: play_pair(job, hand), args.hands, args.jobs, on_result=progress)
    print()
    print(report(outcomes, job))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({"header": header(job), "hands": [asdict(o) for o in outcomes]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
