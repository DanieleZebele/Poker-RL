"""Can what the search finds be taught to the policy? (`rl/search.py`)

The search beats the policy (`search_study.py`) but costs seconds a decision. If its choices can be
*distilled* into the network, the gain costs nothing at play time, and the search can be put to
work in training. This is the cheap test before building any of that: no PPO, one model. The
streets are `--streets` (flop, turn and river by default; the earlier the street, the longer a
rollout and the more it costs: set `--minutes` to bound the generation).

Three phases, each resumable and each its own process (a CUDA context does not survive the
`fork` the other two use):

    generate   play hands with the model at the table; at each decision of the hero on those streets
               ask the search what every candidate is worth (the state and the values are kept), and play
               the *policy's* move, so the states are the ones the policy meets. One file of
               labelled decisions per block of hands, written as soon as the block is done: a
               killed run keeps what it has, and a rerun skips the blocks that exist.
    train      fine-tune a copy of the policy on those labels. Only the policy's trunk and head
               move (the equity encoder and the critic do not). The loss is the cross-entropy to
               the search's target over its candidates (`--target improve`: the original policy
               times exp(value / `--tau`), so it moves only where the search sees a difference),
               plus `--beta` times the KL to the original policy over every legal action, plus
               `--anchor-weight` times the same KL on the decisions of `--anchor-hands` self-play
               hands of every street: without it the network drifts where no label is (the
               preflop above all).
    evaluate   the original and the distilled model sit in the hero's seat on the same hands
               (same stacks, same cards, same seed for the network's sampling), against the
               original in every other seat. The paired difference, over every hand and over the
               hands that saw the flop, the turn and the river, is what the distillation is worth. Compare it with
               the search's own gain on the same hands: `search_study.py` with the same `--seed`,
               `--players` and stacks plays the same ones.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/search_distill.py \\
        --model '#1' --players 2 --hands 10000 --minutes 180 --jobs 45 --train-device cuda

The model is resolved once, at the first run, and pinned in `<out>/meta.json`: the ranking moves
while the fleet trains, and a second phase must not find another model than the first. Delete the
directory to start again.

The answer to look for. If the distilled model gains a good part of what the search did, search
labels are worth putting into training; if it gains nothing, a network of this size cannot absorb
the search from this much data (or the labels are too noisy), and that is worth knowing before
spending the compute. It does not answer how it fares against opponents the search was wrong
about: `--opponent-model` for the evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from array import array
from pathlib import Path

from duel_power import DEFAULT_GLOBAL_DIR, DEFAULT_ROOT, resolve_model
from search_study import (
    BIG_BLIND,
    SIZE_GROUPS,
    SIZES,
    SMALL_BLIND,
    STREET_NAMES,
    add_players_arguments,
    add_spending_arguments,
    interval,
    load_leaf_value,
    load_policy,
    load_search_policy,
    make_player,
    make_searcher,
    make_sim_player,
    players_text,
    spending_of,
    table_choice,
    threads_of,
)

from pokerlab.engine.config import GameConfig
from pokerlab.engine.stats import StatsTracker
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import CARDS_DIM, STREET_DIM, observation_dim
from pokerlab.rl.forking import run_in_copies, torch_seeded
from pokerlab.rl.search import BELIEFS, SearchConfig, Searcher

DEFAULT_OUT = Path("checkpoints/studies/search_distill")
MASK_FILL = -1e9


# ---- the tables -------------------------------------------------------------------------
#
# A block of work is one table, as a table of the training: one size (drawn with the table
# weights), the same players for its `--table-hands` hands, the button moving round, the stacks
# redrawn every hand, and a fresh `StatsTracker` that watches every hand, so the statistics go from
# nothing to `--table-hands` hands as they do in training. The other seats are `--opponent-model`
# (`--opponents model`) or models of the store (`--opponents pool`): the `--pool-top-n` best-rated,
# the frozen anchors left out since they are what the result is measured against, pinned in
# `meta.json`. A search that believes every opponent is the hero's own model learns to beat that
# model (measured: +10 bb/100 against it, -26 Elo against the anchors).


def game_of(args: argparse.Namespace, size: int) -> GameConfig:
    return GameConfig(
        num_players=size,
        starting_stack=round(args.stack_max_bb * BIG_BLIND),
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
    )


def pool_paths(args: argparse.Namespace) -> list[str]:
    """The `--pool-top-n` best-rated models of the store with a file on disk, then `--pool-random`
    more drawn at random from the rest of it (weak and odd models too, as the training's own pool
    mixes them in); the anchors are left out."""
    from pokerlab.rl.global_store import load_ranking

    ranked = [str(Path(member.ref)) for member in load_ranking(args.global_dir).ranked()
              if not member.frozen and Path(member.ref).is_file()]
    top, rest = ranked[: args.pool_top_n], ranked[args.pool_top_n :]
    return top + random.Random(args.seed ^ 0x9001).sample(rest, min(args.pool_random, len(rest)))


def table_plan(args: argparse.Namespace, meta: dict, job: int, seed_base: int) -> tuple[int, int, dict[int, str]]:
    """`(size, hero seat, {other seat: model path})` of block `job`'s table."""
    rng = random.Random((seed_base * 1_000_003 + job) ^ 0x7AB1E5)
    size = rng.choices(SIZES, weights=args.table_weights)[0] if args.table_weights is not None else args.players
    hero = rng.randrange(size)
    seats = [seat for seat in range(size) if seat != hero]
    if "pool" in meta:
        pool = meta["pool"]
        others = rng.sample(pool, size - 1) if len(pool) >= size - 1 else rng.choices(pool, k=size - 1)
    else:
        others = [meta["opponent"]] * (size - 1)
    return size, hero, dict(zip(seats, others, strict=True))


def table_styles(args: argparse.Namespace, job: int, seed_base: int, seats) -> dict:
    """`{seat: style function}` for the opponent seats of block `job`'s table that play with a style
    (`--style-share` of them, `rl/styles.py`, drawn as the training draws them): a push on the model's
    logits and a temperature, the same for the table's hands. The search is told the style too."""
    if args.style_share <= 0:
        return {}
    from pokerlab.rl.styles import StyleConfig

    config = StyleConfig(share=args.style_share, spread=args.style_spread, scales=tuple(args.style_scales))
    rng = random.Random((seed_base * 1_000_003 + job) ^ 0x57E1E)
    styles = {seat: config.draw(rng) for seat in sorted(seats)}
    return {seat: style for seat, style in styles.items() if style is not None}


def styled_sim_player(path: str, fast: bool, style):
    """The search's belief about a seat: its model, with its style if it has one."""
    if style is None:
        return make_sim_player(path, "cpu", fast)
    policy = load_search_policy(path, "cpu") if fast else load_policy(path, "cpu")[0]
    stack = load_policy(path, "cpu")[1]
    return lambda player_id, name: RLAgentPlayer(player_id, name, policy_fn=policy, big_blind=BIG_BLIND,
                                                 starting_stack=stack, style=style)


def block_hands(args: argparse.Namespace, job: int, total: int) -> int:
    return max(0, min(args.table_hands, total - job * args.table_hands))


def play_table(args, meta: dict, job: int, seed_base: int, make_hero, hands: int, *, record=None):
    """Play block `job`'s table: `(size, [(hero's result in big blinds, board cards) per hand])`.

    `make_hero(hero seat, {seat: path}, {seat: style})` builds the hero; `record` is the `on_decision` of the other
    seats. The cards follow the table's seed and the stacks a stream of their own, so another hero
    in the same block plays the same deals (the statistics, which watch the hero too, may differ)."""
    size, hero, paths = table_plan(args, meta, job, seed_base)
    styles = table_styles(args, job, seed_base, paths)
    players = []
    for seat in range(size):
        if seat == hero:
            players.append(make_hero(hero, paths, styles))
            continue
        policy, stack = load_policy(paths[seat], "cpu")
        name = f"{Path(paths[seat]).stem}@{seat}"
        players.append(RLAgentPlayer(name, name, policy_fn=policy, big_blind=BIG_BLIND, starting_stack=stack,
                                     on_decision=record, style=styles.get(seat)))
    base = (seed_base * 1_000_003 + job) * 1_000
    table = Table(game_of(args, size), players, rng=random.Random(base), stats_tracker=StatsTracker())
    stacks_rng = random.Random(base ^ 0x57AC)
    results = []
    for index in range(hands):
        stacks = [max(BIG_BLIND, round(stacks_rng.uniform(args.stack_min_bb, args.stack_max_bb) * BIG_BLIND))
                  for _ in range(size)]
        table.stacks = list(stacks)
        with torch_seeded(base + index + 1):
            result = table.play_hand()
        results.append(((table.stacks[hero] - stacks[hero]) / BIG_BLIND, len(result.hand_history.community_cards)))
    return size, results


def load_field(meta: dict, fast: bool) -> None:
    """Every model that can sit at a table, loaded before the fork so the copies share them. A model
    of the pool the fleet deleted under us (a prune) is dropped from this phase's pool."""
    for path in {meta["model"], meta["opponent"], meta["sim"]}:
        load_policy(path, "cpu")
        if fast:
            load_search_policy(path, "cpu")
    loaded = []
    for path in meta.get("pool", []):
        try:
            load_policy(path, "cpu")
            if fast:
                load_search_policy(path, "cpu")
        except FileNotFoundError:
            print(f"pool: {Path(path).stem} was deleted while loading, left out", flush=True)
            continue
        loaded.append(path)
    if "pool" in meta:
        meta["pool"] = loaded


# ---- generate -----------------------------------------------------------------------


class LabelPlayer(Player):
    """Plays its policy and, at the decisions the search covers, asks the search what the candidates
    are worth and keeps the question and the answer."""

    def __init__(self, player_id: str, name: str, base: RLAgentPlayer, searcher: Searcher,
                 seat_models: dict | None = None) -> None:
        super().__init__(player_id, name)
        self._base = base
        self._searcher = searcher
        self._seat_models = seat_models
        self.records: list[tuple[array, tuple[bool, ...], dict[int, float]]] = []

    def act(self, observation: Observation, legal_actions):
        choice = self._searcher.choose(observation, legal_actions, self._base, seat_models=self._seat_models)
        if choice is not None:
            features, mask = self._base.features_for(observation, legal_actions)
            self.records.append((array("f", features), tuple(mask), dict(choice.values), dict(choice.errors)))
        return self._base.act(observation, legal_actions)


def search_config_of(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        **spending_of(args)[0],
        worlds=args.worlds,
        candidates=args.candidates,
        belief=args.belief,
        min_ess=args.min_ess,
        streets=frozenset(STREET_NAMES[name.strip()] for name in args.streets.split(",")),
        ranges=args.ranges,
        world_threshold=args.world_threshold,
        relative_threshold=args.relative_threshold,
    )


def labelled_table(args, meta: dict, job: int) -> tuple[list, Searcher]:
    """Block `job`'s table with the hero labelling its decisions; the search believes each seat is
    the model that sits there (the hero's own seat, `--sim-model`)."""
    fast = spending_of(args)[1]
    searcher = make_searcher(meta["sim"], "cpu", fast, args.threads, search_config_of(args),
                             random.Random((args.seed * 1_000_003 + job) ^ 0x5EA4C4))
    labeller = []

    def make_hero(hero: int, paths: dict[int, str], styles: dict) -> LabelPlayer:
        if args.believe == "random":
            # Who sits there is not known: each seat is believed to be a model of the pool drawn for
            # the table, so the labels are what an action is worth against the population -- the one
            # thing a policy that cannot tell its opponents apart can learn.
            guess = random.Random((args.seed * 1_000_003 + job) ^ 0xB11D)
            paths = {seat: guess.choice(meta["pool"]) for seat in paths}
            styles = {}
        seat_models = {seat: styled_sim_player(path, fast, styles.get(seat)) for seat, path in paths.items()}
        seat_models[hero] = make_sim_player(meta["sim"], "cpu", fast)
        player = LabelPlayer("hero", "hero", make_player(meta["model"], "cpu")("hero", "hero"), searcher, seat_models)
        labeller.append(player)
        return player

    play_table(args, meta, job, args.seed, make_hero, block_hands(args, job, args.hands))
    return labeller[0].records, searcher


def write_shard(path: Path, records: list, version: int) -> None:
    import torch

    count = len(records)
    values = torch.zeros(count, ACTION_DIM)
    errors = torch.zeros(count, ACTION_DIM)
    candidates = torch.zeros(count, ACTION_DIM, dtype=torch.bool)
    for row, (_features, _mask, found, error) in enumerate(records):
        for index, value in found.items():
            values[row, index] = value
            errors[row, index] = error.get(index, 0.0)
            candidates[row, index] = True
    shard = {
        "features": torch.tensor([list(features) for features, *_ in records], dtype=torch.float32),
        "mask": torch.tensor([list(mask) for _f, mask, *_ in records], dtype=torch.bool),
        "candidates": candidates,
        "values": values,
        "errors": errors,
        "feature_version": version,
    }
    temporary = path.with_suffix(".partial")
    torch.save(shard, temporary)
    os.replace(temporary, path)


def phase_generate(args: argparse.Namespace, meta: dict) -> None:
    import torch

    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    jobs = -(-args.hands // args.table_hands)
    version = load_policy(meta["model"], "cpu")[0].feature_version
    load_field(meta, spending_of(args)[1])  # before the fork: the copies share the loaded models
    if args.threads > 1:
        load_search_policy(meta["sim"], "cpu")
    if args.leaf:
        load_leaf_value(meta["sim"], "cpu")
    started = time.time()
    deadline = started + 60 * args.minutes if args.minutes else None
    done = [0]

    # The smallest tables first: they have the most decisions to search (heads-up, about one a hand
    # on the flop, turn and river; nine-handed, a few a table) and take the longest, so started last
    # they would leave the other cores idle at the end.
    order = sorted(range(jobs), key=lambda job: table_plan(args, meta, job, args.seed)[0])

    def work(index: int) -> dict:
        job = order[index]
        shard = data / f"job-{job:05d}.pt"
        if shard.exists():
            return {"status": "had it"}
        if deadline is not None and time.time() > deadline:
            return {"status": "no time left, skipped"}  # a rerun takes it
        records, searcher = labelled_table(args, meta, job)
        counts = searcher.counts
        seen = {
            "decisions": counts.decisions,
            "searched": counts.searched,
            "changed": counts.changed,
            "declined": counts.declined_ess + counts.declined_other,
            "ess": counts.ess_total,
        }
        write_shard(shard, records, version)
        return {"status": f"{len(records)} labelled decisions", **seen}

    totals = dict.fromkeys(("blocks", "decisions", "searched", "changed", "declined", "ess"), 0.0)

    def progress(index: int, outcome: dict) -> None:
        job = order[index]
        done[0] += 1
        elapsed = time.time() - started
        print(f"  block {job} ({done[0]}/{jobs}): {outcome['status']}, {elapsed:.0f} s", flush=True)
        if "searched" not in outcome:
            return
        totals["blocks"] += 1
        for key in ("decisions", "searched", "changed", "declined", "ess"):
            totals[key] += outcome[key]
        if totals["blocks"] % args.summary_every == 0 and totals["searched"]:
            asked, answered = max(totals["decisions"], 1), totals["searched"]
            left = jobs - done[0]
            print(f"  -- {int(answered)} labelled so far: the search changed the policy's move in "
                  f"{totals['changed'] / answered:.0%}, declined {totals['declined'] / asked:.0%}, "
                  f"{totals['ess'] / answered:.0f} effective hands per range on average; "
                  f"~{left * elapsed / done[0] / 60:.0f} min for the {left} blocks left", flush=True)

    run_in_copies(work, jobs, args.jobs, on_result=progress)
    total = sum(torch.load(path, weights_only=True)["features"].shape[0] for path in data.glob("job-*.pt"))
    print(f"{total} labelled decisions in {data}")


# ---- train --------------------------------------------------------------------------


TARGETS = ("soft", "improve", "best")


def significant_values(values, errors, candidates, teacher_logits, min_z: float):
    """The values with every gap to the policy's likeliest candidate that the search is not sure of
    (under `min_z` of its standard errors) closed: such a candidate gets that bin's value, so the
    `improve` target leaves the policy's odds between the two as they were. An error of 0 (not
    recorded) closes nothing."""
    if min_z <= 0:
        return values
    top = teacher_logits.masked_fill(~candidates, MASK_FILL).argmax(dim=-1, keepdim=True)
    reference = values.gather(-1, top).expand_as(values)
    unsure = candidates & (errors > 0) & ((values - reference).abs() < min_z * errors)
    return values.masked_scatter(unsure, reference[unsure]) if unsure.any() else values


def soft_targets(values, candidates, tau: float, kind: str = "soft", teacher_logits=None):
    """The target distribution over the action bins, zero outside the candidates the search tried.

    `soft`: the softmax of the search's values (in big blinds, over `tau`). It ignores the policy,
    so candidates the search finds equal -- within its noise, the common case -- get equal mass,
    and the policy is pushed towards the actions it rightly plays least.
    `improve`: the original policy's own probabilities times `exp(value / tau)`, renormalised: the
    step a KL-bounded policy improvement takes. Equal values leave the policy as it was; it moves
    only where the search sees a difference.
    `best`: all the mass on the search's best candidate, which is what the search plays."""
    import torch

    if kind == "best":
        best = values.masked_fill(~candidates, MASK_FILL).argmax(dim=-1)
        return torch.nn.functional.one_hot(best, values.shape[-1]).to(values.dtype)
    scores = values / tau
    if kind == "improve":
        if teacher_logits is None:
            raise ValueError("the 'improve' target starts from the original policy: give its logits")
        scores = scores + torch.log_softmax(teacher_logits.masked_fill(~candidates, MASK_FILL), dim=-1)
    elif kind != "soft":
        raise ValueError(f"target must be one of {TARGETS}, got {kind!r}")
    return scores.masked_fill(~candidates, MASK_FILL).softmax(dim=-1)


def policy_kl(logits, teacher_logits, mask):
    """KL from the original policy to the new one over the legal actions, one number a row."""
    import torch

    log_new = torch.log_softmax(logits, dim=-1)
    log_old = torch.log_softmax(teacher_logits, dim=-1)
    return (log_old.exp() * (log_old - log_new)).masked_fill(~mask, 0.0).sum(dim=-1)


def distill_loss(logits, teacher_logits, mask, candidates, values, tau: float, beta: float, kind: str = "soft"):
    """`(loss, cross-entropy, kl)`, each a mean over the batch.

    The cross-entropy is to the search's target (`soft_targets`) *among its candidates* (the
    policy's own probabilities renormalised over them), so it only teaches the order of the
    candidates. The KL, from the original policy to the new one over every legal action, holds the
    rest."""
    import torch

    target = soft_targets(values, candidates, tau, kind, teacher_logits)
    restricted = torch.log_softmax(logits.masked_fill(~candidates, MASK_FILL), dim=-1)
    cross_entropy = -(target * restricted).masked_fill(~candidates, 0.0).sum(dim=-1)
    kl = policy_kl(logits, teacher_logits, mask)
    return (cross_entropy + beta * kl).mean(), cross_entropy.mean(), kl.mean()


def agreement(logits, candidates, values):
    """Share of decisions where the policy's likeliest candidate is the one the search liked most."""
    best_policy = logits.masked_fill(~candidates, MASK_FILL).argmax(dim=-1)
    best_search = values.masked_fill(~candidates, MASK_FILL).argmax(dim=-1)
    return (best_policy == best_search).float().mean().item()


def load_dataset(data: Path):
    import torch

    shards = [torch.load(path, weights_only=True) for path in sorted(data.glob("job-*.pt"))]
    if not shards:
        raise SystemExit(f"no labelled decisions in {data}: run the generate phase first")
    for shard in shards:  # a shard written before the errors were kept: 0, "unknown", never filtered
        shard.setdefault("errors", torch.zeros_like(shard["values"]))
    keys = ("features", "mask", "candidates", "values", "errors")
    return {key: torch.cat([shard[key] for shard in shards]) for key in keys}


def anchor_states(args: argparse.Namespace, meta: dict):
    """`(features, mask, street)` of every decision of `--anchor-hands` hands of the original model
    at the generation's tables (every seat, every street, the preflop included): where the KL holds
    the new policy to the old one. The labelled decisions alone are the hero's on the searched streets, and
    the network moved freely everywhere else -- the preflop above all, which no label covers."""
    import torch

    policy, stack = load_policy(meta["model"], "cpu")
    load_field(meta, fast=False)
    width = observation_dim(policy.feature_version)

    def work(job: int) -> list:
        kept = []

        def keep(record) -> None:
            # A seat of another encoding (a version-4 model of the pool) is a state the hero would
            # read differently: only the decisions in the hero's own encoding are kept.
            if len(record.features) == width:
                kept.append((array("f", record.features), tuple(record.legal_mask)))

        def make_hero(_hero: int, _paths, _styles) -> RLAgentPlayer:
            return RLAgentPlayer("hero", "hero", policy_fn=policy, big_blind=BIG_BLIND, starting_stack=stack,
                                 on_decision=keep)

        play_table(args, meta, job, args.anchor_seed, make_hero, block_hands(args, job, args.anchor_hands), record=keep)
        return kept

    tables = -(-args.anchor_hands // args.table_hands)
    rows = [row for table in run_in_copies(work, tables, args.jobs) for row in table]
    features = torch.tensor([list(f) for f, _m in rows], dtype=torch.float32)
    mask = torch.tensor([list(m) for _f, m in rows], dtype=torch.bool)
    return features, mask, features[:, CARDS_DIM : CARDS_DIM + STREET_DIM].argmax(dim=-1)


def phase_train(args: argparse.Namespace, meta: dict) -> None:
    import torch

    from pokerlab.rl.ppo import build_model_from_checkpoint, save_checkpoint

    # Before anything touches CUDA: the anchor states are played by forked copies.
    anchor = anchor_states(args, meta) if args.anchor_hands else None
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.train_device == "auto" else args.train_device
    parts = [load_dataset(args.out / "data"), *(load_dataset(extra) for extra in args.extra_data)]
    data = {key: torch.cat([part[key] for part in parts]).to(device) for key in parts[0]}
    errors = data.pop("errors")
    if args.extra_data:
        print(f"labels of {len(args.extra_data)} more directories of the same model: "
              f"{', '.join(str(extra) for extra in args.extra_data)}")
    if anchor is not None:
        anchor = tuple(part.to(device) for part in anchor)
        print(f"{len(anchor[0])} anchor decisions from {args.anchor_hands} self-play hands, every street")
    count = data["features"].shape[0]
    order = torch.randperm(count, generator=torch.Generator().manual_seed(args.seed)).to(device)
    held = max(1, count // 10)
    validation, training = order[:held], order[held:]

    model, checkpoint = build_model_from_checkpoint(meta["model"], device=device)
    teacher = copy.deepcopy(model)
    if args.min_z > 0:
        with torch.no_grad():
            teacher_logits = torch.cat([teacher(data["features"][i : i + 4096], data["mask"][i : i + 4096])[0]
                                        for i in range(0, len(data["features"]), 4096)])
        before = data["values"]
        data["values"] = significant_values(before, errors, data["candidates"], teacher_logits, args.min_z)
        changed = (data["values"] != before).any(dim=-1).float().mean().item()
        known = (errors > 0).any(dim=-1).float().mean().item()
        print(f"gaps under {args.min_z:g} standard errors closed: {changed:.0%} of the decisions "
              f"({known:.0%} carry errors)")
    for parameter in model.policy_parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.policy_parameters(), lr=args.lr)
    started = time.time()
    metadata = {**(checkpoint.get("metadata") or {}), "distilled_from": meta["model"],
                "distillation": {"decisions": count, "tau": args.tau, "beta": args.beta, "epochs": args.epochs,
                                 "lr": args.lr, "target": args.target, "anchor_hands": args.anchor_hands,
                                 "min_z": args.min_z,
                                 "anchor_weight": args.anchor_weight}}
    output = args.out / args.save_as
    print(f"{count} decisions ({len(training)} to learn from, {held} held out) on {device}")

    streets = data["features"][:, CARDS_DIM : CARDS_DIM + STREET_DIM].argmax(dim=-1)

    names = ("preflop", "flop", "turn", "river")

    def by_street(label: str) -> None:
        parts = []
        for index, name in enumerate(names):
            rows = validation[streets[validation] == index]
            if len(rows) == 0:
                continue
            cross_entropy, _kl, agree = evaluate(rows)
            parts.append(f"{name} {cross_entropy:.2f} / {agree:.0%} ({len(rows)})")
        print(f"    {label} by street (cross-entropy / agreement): " + "; ".join(parts), flush=True)
        if anchor is None:
            return
        features, mask, street = anchor
        with torch.no_grad():
            parts = []
            for index, name in enumerate(names):
                rows = street == index
                if not rows.any():
                    continue
                new_logits = torch.cat([model(features[rows][i : i + 4096], mask[rows][i : i + 4096])[0]
                                        for i in range(0, int(rows.sum()), 4096)])
                old_logits = torch.cat([teacher(features[rows][i : i + 4096], mask[rows][i : i + 4096])[0]
                                        for i in range(0, int(rows.sum()), 4096)])
                kl = policy_kl(new_logits, old_logits, mask[rows]).mean().item()
                moved = (new_logits.argmax(dim=-1) != old_logits.argmax(dim=-1)).float().mean().item()
                parts.append(f"{name} kl {kl:.3f}, top action changed {moved:.0%}")
        print(f"    {label} on self-play states: " + "; ".join(parts), flush=True)

    def evaluate(rows) -> tuple[float, float, float]:
        with torch.no_grad():
            batch = {key: value[rows] for key, value in data.items()}
            logits, _ = model(batch["features"], batch["mask"])
            teacher_logits, _ = teacher(batch["features"], batch["mask"])
            _, cross_entropy, kl = distill_loss(
                logits, teacher_logits, batch["mask"], batch["candidates"], batch["values"], args.tau, args.beta,
                args.target,
            )
            return cross_entropy.item(), kl.item(), agreement(logits, batch["candidates"], batch["values"])

    cross_entropy, kl, agree = evaluate(validation)
    print(f"  before: held-out cross-entropy {cross_entropy:.3f}, kl {kl:.3f}, "
          f"the policy's pick is the search's in {agree:.0%} of decisions")
    counts = ", ".join(f"{int((streets == i).sum())} {name}" for i, name in enumerate(("preflop", "flop", "turn", "river")))
    print(f"  decisions by street: {counts}")
    by_street("before")
    for epoch in range(1, args.epochs + 1):
        shuffled = training[torch.randperm(len(training), device=device)]
        for start in range(0, len(shuffled), args.batch):
            batch = {key: value[shuffled[start : start + args.batch]] for key, value in data.items()}
            logits, _ = model(batch["features"], batch["mask"])
            with torch.no_grad():
                teacher_logits, _ = teacher(batch["features"], batch["mask"])
            loss, _, _ = distill_loss(
                logits, teacher_logits, batch["mask"], batch["candidates"], batch["values"], args.tau, args.beta,
                args.target,
            )
            if anchor is not None and args.anchor_weight > 0:
                rows = torch.randint(len(anchor[0]), (args.batch,), device=device)
                anchor_logits, _ = model(anchor[0][rows], anchor[1][rows])
                with torch.no_grad():
                    anchor_teacher, _ = teacher(anchor[0][rows], anchor[1][rows])
                loss = loss + args.anchor_weight * policy_kl(anchor_logits, anchor_teacher, anchor[1][rows]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cross_entropy, kl, agree = evaluate(validation)
        print(f"  epoch {epoch}: held-out cross-entropy {cross_entropy:.3f}, kl {kl:.3f}, agreement {agree:.0%}"
              f"  ({time.time() - started:.0f} s)", flush=True)
        by_street(f"epoch {epoch}")
        if args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
            path = output.with_name(f"{output.stem}-epoch{epoch}.pt")
            save_checkpoint(path, model, None, iteration=epoch, metadata=metadata)
            print(f"    saved {path.name} (evaluate it with --checkpoint)", flush=True)

    save_checkpoint(output, model, None, iteration=args.epochs, metadata=metadata)
    print(f"saved {output}")


# ---- evaluate -----------------------------------------------------------------------


def phase_evaluate(args: argparse.Namespace, meta: dict) -> None:
    distilled = str(args.checkpoint or args.out / "distilled.pt")
    if not Path(distilled).is_file():
        raise SystemExit(f"{distilled} does not exist: run the train phase first")
    print(f"distilled model: {distilled}")
    load_field(meta, fast=False)
    load_policy(distilled, "cpu")
    hero_makers = {"original": make_player(meta["model"], "cpu"), "distilled": make_player(distilled, "cpu")}

    def work(job: int) -> list:
        """Block `job`'s table played once with each hero: the same deals, stacks and opponents."""
        outcomes = {}
        for name, maker in hero_makers.items():
            outcomes[name] = play_table(args, meta, job, args.eval_seed, lambda _h, _p, _s, maker=maker: maker("hero", "hero"),
                                        block_hands(args, job, args.eval_hands))
        size, original = outcomes["original"]
        _size, distilled_ = outcomes["distilled"]
        return [(o, d, max(ob, db), size) for (o, ob), (d, db) in zip(original, distilled_, strict=True)]

    started = time.time()
    tables = -(-args.eval_hands // args.table_hands)
    rows = [row for table in run_in_copies(work, tables, args.jobs) for row in table]
    opponents = f"a pool of {len(meta['pool'])} models of the store" if "pool" in meta else Path(meta["opponent"]).stem
    print(f"{args.eval_hands} hands in {time.time() - started:.0f} s; "
          f"{players_text(args.players, args.table_weights)}, opponents {opponents}")
    summary = {}
    for name, chosen in (
        ("every hand", rows),
        ("the hands that saw the flop", [r for r in rows if r[2] >= 3]),
        ("... the turn", [r for r in rows if r[2] >= 4]),
        ("... the river", [r for r in rows if r[2] >= 5]),
        *(
            (f"{low}-{high} players", [r for r in rows if low <= r[3] <= high])
            for low, high in SIZE_GROUPS
            if args.table_weights is not None
        ),
    ):
        if not chosen:
            continue
        difference = [distilled - original for original, distilled, *_ in chosen]
        mean, half = interval(difference)
        summary[name] = {"hands": len(chosen), "difference": mean * 100, "half": half * 100}
        original = sum(r[0] for r in chosen) / len(chosen) * 100
        distilled_bb = sum(r[1] for r in chosen) / len(chosen) * 100
        print(f"{name + f' ({len(chosen)})':38} original {original:+8.1f}  distilled {distilled_bb:+8.1f}   "
              f"difference {mean * 100:+7.1f} +- {half * 100:5.1f} bb/100")
    (Path(distilled).parent / f"{Path(distilled).stem}-evaluate.json").write_text(json.dumps(summary, indent=2))


# ---- command line -------------------------------------------------------------------


def pin_models(args: argparse.Namespace) -> dict:
    """Resolve the models once and keep them in `<out>/meta.json`."""
    meta_path = args.out / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        print(f"models pinned in {meta_path}: {Path(meta['model']).stem}")
        return meta
    meta = {}
    for key, spec in (("model", args.model), ("opponent", args.opponent_model or args.model),
                      ("sim", args.sim_model or args.model)):
        label, path, rating = resolve_model(spec, global_dir=args.global_dir, root=args.root)
        meta[key] = str(path)
        print(f"{key}: {label}" + (f" (rating {rating:.0f})" if rating is not None else ""), flush=True)
    if args.opponents == "pool":
        meta["pool"] = pool_paths(args)
        if not meta["pool"]:
            raise SystemExit("--opponents pool: no model of the store has a file on disk")
        print(f"pool: the {len(meta['pool'])} best-rated models of the store (anchors left out)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("generate", "train", "evaluate", "all"), default="all")
    parser.add_argument("--model", required=True, help="the model to distill into: a path, a label or #N")
    parser.add_argument("--opponent-model", default=None, help="who sits opposite (default: --model)")
    parser.add_argument("--sim-model", default=None, help="who the search believes they are (default: --model)")
    add_players_arguments(parser)
    parser.add_argument("--opponents", choices=("model", "pool"), default="model",
                        help="model: --opponent-model in every other seat; pool: models of the store, "
                             "drawn per table")
    parser.add_argument("--pool-top-n", type=int, default=50, help="the pool: this many best-rated models")
    parser.add_argument("--pool-random", type=int, default=0,
                        help="and this many more drawn at random from the rest of the store")
    parser.add_argument("--table-hands", type=int, default=200,
                        help="hands a table lasts (one block of work): the same players, and statistics from "
                             "nothing to this many hands, as a training table")
    parser.add_argument("--style-share", type=float, default=0.0,
                        help="share of the opponent seats that play a table with a random style (rl/styles.py)")
    parser.add_argument("--style-spread", type=float, default=0.75, help="sd of each style axis")
    parser.add_argument("--style-scales", type=float, nargs=5, default=[10.0, 16.0, 16.0, 8.0, 10.0],
                        help="per-axis scales (config.toml's style_scales)")
    parser.add_argument("--stack-min-bb", type=float, default=20.0)
    parser.add_argument("--stack-max-bb", type=float, default=100.0)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=1, help="seed of the hands the labels come from")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    gen = parser.add_argument_group("generate")
    gen.add_argument("--hands", type=int, default=10000,
                     help="a heads-up hand has about one searched decision on the flop, turn and river together")
    gen.add_argument("--summary-every", type=int, default=10, help="blocks between two running summaries")
    gen.add_argument("--minutes", type=float, default=0,
                     help="stop starting new blocks after this long (0: no limit); a rerun resumes")
    gen.add_argument("--streets", default="flop,turn,river", help="comma list of preflop,flop,turn,river")
    gen.add_argument("--worlds", type=int, default=300)
    gen.add_argument("--candidates", type=int, default=4)
    gen.add_argument("--belief", choices=BELIEFS, default="weighted")
    gen.add_argument("--min-ess", type=float, default=3.0)
    gen.add_argument("--world-threshold", type=float, default=0.01)
    gen.add_argument("--believe", choices=("exact", "random"), default="exact",
                     help="with a pool: the search knows the model in each seat, or believes it is one of the "
                          "pool drawn for the table")
    gen.add_argument("--relative-threshold", action="store_true",
                     help="the threshold drops a hand that much less likely than its range's likeliest")
    gen.add_argument("--no-ranges", dest="ranges", action="store_false",
                     help="weigh sampled worlds instead of drawing them from per-opponent ranges")
    add_spending_arguments(parser)
    train = parser.add_argument_group("train")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--tau", type=float, default=1.0, help="temperature of the search's values, in big blinds")
    train.add_argument("--beta", type=float, default=1.0, help="weight of the KL to the original policy")
    train.add_argument("--target", choices=TARGETS, default="improve",
                       help="soft: softmax of the values; improve: the policy times exp(value/tau); "
                            "best: the search's best candidate")
    train.add_argument("--anchor-hands", type=int, default=3000,
                       help="self-play hands whose decisions (every street) the KL also holds; 0: the labels only")
    train.add_argument("--anchor-weight", type=float, default=1.0, help="weight of the KL on the anchor states")
    train.add_argument("--min-z", type=float, default=0.0,
                       help="learn only the gaps the search is this many standard errors sure of (0: every gap)")
    train.add_argument("--anchor-seed", type=int, default=3)
    train.add_argument("--save-as", default="distilled.pt", help="file name of the model, in --out")
    train.add_argument("--extra-data", type=Path, nargs="*", default=[],
                       help="more labels to learn from: data directories made with the same --model")
    train.add_argument("--train-device", default="auto")
    train.add_argument("--save-every", type=int, default=5,
                       help="also keep the model every this many epochs (0: only the last)")
    ev = parser.add_argument_group("evaluate")
    ev.add_argument("--checkpoint", type=Path, default=None,
                    help="the model to evaluate (default: <out>/distilled.pt; the train phase also leaves "
                         "distilled-epochN.pt)")
    ev.add_argument("--eval-hands", type=int, default=60000)
    ev.add_argument("--eval-seed", type=int, default=2, help="seed of the hands the models are compared on")

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(arguments)
    args.players, args.table_weights = table_choice(args)
    args.threads = threads_of(args)
    if args.phase == "all":
        # One process each: the generation and the evaluation fork, and a forked copy cannot
        # use the CUDA context the training opens.
        base = []
        skip = False
        for item in arguments:
            if skip:
                skip = False
            elif item == "--phase":
                skip = True
            elif not item.startswith("--phase="):
                base.append(item)
        for phase in ("generate", "train", "evaluate"):
            print(f"=== {phase}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *base, "--phase", phase], check=True)
        return 0
    meta = pin_models(args)
    if "pool" in meta:
        # The fleet prunes the store while this runs: a model of the pool may be gone by the next
        # phase. Each phase seats the ones still on disk (a model already loaded stays in memory).
        present = [path for path in meta["pool"] if Path(path).is_file()]
        if len(present) < len(meta["pool"]):
            print(f"pool: {len(meta['pool']) - len(present)} of {len(meta['pool'])} models pruned since, left out")
        meta["pool"] = present
    {"generate": phase_generate, "train": phase_train, "evaluate": phase_evaluate}[args.phase](args, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
