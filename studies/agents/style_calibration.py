"""How far can a style push a model before it becomes a weak player? The scales of `rl/styles.py`.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/style_calibration.py --model '#1'
    ... --only larghezza tenacia    # a few cells at a time; results are kept in --out and resumed
    ... --report                    # only print the report from what --out holds

Every table has one **subject** seat and the model unpushed in every other. A cell is the
subject playing with one style: one axis of `rl/styles.py` at +1 or -1 times a strength of
the grid (`--strengths`), or a temperature alone, or -- the references -- another, weaker
model of the ranking in its seat (`--references`). The same hands are played by every cell
from the same seeds (table, stacks, size, seat and torch), and once more with the subject
unpushed: the paired difference of the subject's result is what the style **costs** it, in
bb/100. A hand that closed all-in before the river is scored at its expected result over
the runouts (`rl/allin_reward.py`, as in training), which takes most of the card luck out.

The references are what the population already spans: the cost of being the #50 instead of
the #1 in the same seat is how much weaker the field's real players are. A style that costs
no more than that makes an opponent no weaker than ones the learner already meets. The
report gives, per axis and sign, the cost and the style it buys (the subject's measured
statistics, and its mean bet size), and the strength at which the cost reaches the chosen
reference (interpolated over the grid): the scale to write in `config.toml`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pokerlab.engine.actions import ActionType
from pokerlab.engine.history import HandHistory
from pokerlab.engine.stats import STATS, analyse_hand
from pokerlab.rl.styles import AXES, Style, style_fn

DEFAULT_OUT = Path("checkpoints/studies/style_calibration.json")
DEFAULT_STRENGTHS = (0.5, 1.0, 2.0)
DEFAULT_TEMPERATURES = (0.5, 2.0)
DEFAULT_REFERENCES = ("#20", "#50", "#100")
BASELINE = "base"


@dataclass(frozen=True)
class Cell:
    """One way the subject plays: a style (`axis`, `sign`, `strength`), a `temperature`,
    or another model (`reference`); every field empty is the unpushed baseline."""

    name: str
    axis: str = ""
    sign: float = 0.0
    strength: float = 0.0
    temperature: float = 1.0
    reference: str = ""

    def style(self) -> tuple[Style, tuple[float, ...]]:
        axes = tuple(self.sign if name == self.axis else 0.0 for name in AXES)
        scales = tuple(self.strength if name == self.axis else 1.0 for name in AXES)
        return Style(axes, self.temperature), scales


def grid(strengths: Sequence[float], temperatures: Sequence[float], references: Sequence[str]) -> list[Cell]:
    cells = [Cell(BASELINE)]
    for axis in AXES:
        for sign in (1.0, -1.0):
            for strength in strengths:
                cells.append(Cell(f"{axis} {'+' if sign > 0 else '-'}{strength:g}", axis, sign, strength))
    cells += [Cell(f"temperatura {t:g}", temperature=t) for t in temperatures]
    cells += [Cell(f"modello {r}", reference=r) for r in references]
    return cells


# ---- playing -------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationJob:
    cell: Cell
    seed: int
    hands: int
    base_path: str
    subject_path: str
    weights: tuple[float, ...]
    stack_min_bb: float
    stack_max_bb: float
    small_blind: int
    big_blind: int
    session_hands: int
    runouts: int


@dataclass
class CellResult:
    """What the subject did over a cell's hands."""

    bb: list[float] = field(default_factory=list)  # one per hand, all-in hands at their expectation
    events: list[int] = field(default_factory=lambda: [0] * len(STATS))
    chances: list[int] = field(default_factory=lambda: [0] * len(STATS))
    bet_sizes: list[float] = field(default_factory=list)  # its bets and raises, street total / pot before


def subject_result(hand: HandHistory, seat: int, expected: dict[int, float] | None) -> float:
    """The subject's result in big blinds, at its expectation if the hand closed all-in."""
    if expected is not None:
        return expected[seat] / hand.big_blind
    return (hand.final_stacks[seat] - hand.starting_stacks[seat]) / hand.big_blind


def add_hand(result: CellResult, hand: HandHistory, seat: int, expected: dict[int, float] | None) -> None:
    result.bb.append(subject_result(hand, seat, expected))
    dealt = sorted(hand.starting_stacks)
    counts = analyse_hand(hand.actions, dealt=dealt, button_seat=hand.button_seat,
                          board_cards=len(hand.community_cards), player_ids={s: f"s{s}" for s in dealt})
    mine = counts[f"s{seat}"]
    for slot in range(len(STATS)):
        result.events[slot] += mine.events[slot]
        result.chances[slot] += mine.opportunities[slot]
    for record in hand.actions:
        if record.seat == seat and record.action_type in (ActionType.BET, ActionType.RAISE) and record.pot_before:
            result.bet_sizes.append(record.amount / record.pot_before)


def run_job(job: CalibrationJob) -> CellResult:
    import torch

    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.allin_reward import expected_deltas
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.table_mix import TableMix

    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    base = make_policy_fn(build_model_from_checkpoint(job.base_path)[0])
    subject = base if job.subject_path == job.base_path else make_policy_fn(build_model_from_checkpoint(job.subject_path)[0])
    style, scales = job.cell.style()
    styled = None if job.cell.name == BASELINE or job.cell.reference else style_fn(style, scales)
    mix = TableMix(weights=job.weights, stack_min_bb=job.stack_min_bb, stack_max_bb=job.stack_max_bb,
                   small_blind=job.small_blind, big_blind=job.big_blind)
    bank = TableBank(mix, random.Random(job.seed), stack_rng=random.Random(job.seed + 1))
    seat_rng, size_rng = random.Random(job.seed + 2), random.Random(job.seed + 3)
    runout_rng = random.Random(job.seed + 4)
    result = CellResult()
    while len(result.bb) < job.hands:
        size = mix.draw_size(size_rng)
        subject_seat = seat_rng.randrange(size)
        for seat, proxy in enumerate(bank.seats(size)):
            mine = seat == subject_seat
            proxy.inner = RLAgentPlayer(
                proxy.player_id, proxy.name, policy_fn=subject if mine else base,
                big_blind=mix.big_blind, starting_stack=mix.starting_stack, style=styled if mine else None,
            )

        def on_hand(hand: HandHistory, seat: int = subject_seat) -> None:
            add_hand(result, hand, seat, expected_deltas(hand, runout_rng, job.runouts))

        bank.play_session(size, min(job.session_hands, job.hands - len(result.bb)), on_hand=on_hand)
    return result


# ---- the arithmetic and the report ---------------------------------------------------------


def paired_cost(cell: Sequence[float], baseline: Sequence[float]) -> tuple[float, float]:
    """`cell - baseline` hand by hand, in bb/100, and its 95% half-interval."""
    n = min(len(cell), len(baseline))
    differences = [cell[i] - baseline[i] for i in range(n)]
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / max(n - 1, 1)
    return 100 * mean, 100 * 1.96 * math.sqrt(variance / n)


def strength_at_cost(points: Sequence[tuple[float, float]], target: float) -> float | None:
    """The strength at which the cost (a loss, so negative) reaches `target`, interpolated
    between the grid's points; None if no point of the grid costs that much."""
    ordered = sorted(points)
    previous = (0.0, 0.0)
    for strength, cost in ordered:
        if cost <= target:
            (s0, c0), (s1, c1) = previous, (strength, cost)
            if c1 == c0:
                return s1
            return s0 + (target - c0) * (s1 - s0) / (c1 - c0)
        previous = (strength, cost)
    return None


SHORT = {"vpip": "VPIP", "pfr": "PFR", "three_bet": "3bet", "fold_to_three_bet": "f3bet", "steal": "steal",
         "aggression": "aggr", "cbet": "cbet", "fold_to_cbet": "fcbet", "wtsd": "WTSD"}


def _row(name: str, data: dict, baseline: dict) -> str:
    cost, half = paired_cost(data["bb"], baseline["bb"]) if name != BASELINE else (0.0, 0.0)
    rates = "".join(
        f"{100 * e / c:6.0f}%" if c else f"{'-':>7}" for e, c in zip(data["events"], data["chances"])
    )
    sizes = data["bet_sizes"]
    size = f"{100 * sum(sizes) / len(sizes):6.0f}%" if sizes else f"{'-':>7}"
    return f"  {name:<28} {cost:+8.1f} +-{half:5.1f}{rates}{size}"


def format_report(results: dict[str, dict], cells: Sequence[Cell], *, reference: str) -> list[str]:
    if BASELINE not in results:
        return ["manca la cella di base: lanciala prima (--only base)"]
    baseline = results[BASELINE]
    header = f"  {'cella':<28} {'costo bb/100':>15}" + "".join(f"{SHORT[n]:>7}" for n in STATS) + f"{'punt.':>7}"
    lines = [f"{len(baseline['bb']):,} mani per cella, costo = risultato del soggetto meno lo stesso senza stile",
             "(punt. = puntata o rilancio medio del soggetto, in % del piatto prima)", "", header,
             _row(BASELINE, baseline, baseline)]
    for cell in cells:
        if cell.name != BASELINE and cell.name in results and (cell.reference or cell.temperature != 1.0):
            lines.append(_row(cell.name, results[cell.name], baseline))
    target = None
    if f"modello {reference}" in results:
        target = paired_cost(results[f"modello {reference}"]["bb"], baseline["bb"])[0]
    for axis in AXES:
        lines += ["", f"  {axis}:"]
        for sign in (1.0, -1.0):
            points = []
            for cell in cells:
                if cell.axis == axis and cell.sign == sign and cell.name in results:
                    lines.append(_row(cell.name, results[cell.name], baseline))
                    points.append((cell.strength, paired_cost(results[cell.name]["bb"], baseline["bb"])[0]))
            if target is not None and points:
                found = strength_at_cost(points, target)
                at = f"{found:.2f}" if found is not None else f"oltre {max(p[0] for p in points):g} (non costa tanto)"
                lines.append(f"    {'+' if sign > 0 else '-'}{axis}: costa quanto il modello {reference} ({target:+.1f}) a forza {at}")
    if target is None:
        lines += ["", f"  (manca il riferimento modello {reference}: nessuna scala suggerita)"]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    from duel_power import resolve_model

    from pokerlab.rl.table_mix import (
        DEFAULT_BIG_BLIND,
        DEFAULT_SMALL_BLIND,
        DEFAULT_STACK_MAX_BB,
        DEFAULT_STACK_MIN_BB,
        DEFAULT_TABLE_WEIGHTS,
    )

    parser = argparse.ArgumentParser(description="quanto costa uno stile: le scale di rl/styles.py")
    parser.add_argument("--model", default="#1")
    parser.add_argument("--strengths", type=float, nargs="+", default=list(DEFAULT_STRENGTHS))
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--references", nargs="+", default=list(DEFAULT_REFERENCES),
                        help="modelli (#N, etichetta o percorso) seduti al posto del soggetto")
    parser.add_argument("--reference", default="#50", help="il riferimento che fissa la scala suggerita")
    parser.add_argument("--only", nargs="+", default=None,
                        help="solo le celle il cui nome comincia cosi' (base, un asse, temperatura, modello)")
    parser.add_argument("--hands", type=int, default=60000, help="mani per cella")
    parser.add_argument("--runouts", type=int, default=20, help="runout per il valore atteso degli all-in")
    parser.add_argument("--jobs", type=int, default=30)
    parser.add_argument("--session-hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", action="store_true", help="solo il report di cio' che --out contiene")
    parser.add_argument("--restart", action="store_true",
                        help="butta via --out e ricomincia (altrimenti i modelli del file restano quelli)")
    parser.add_argument("--root", type=Path, default=Path("checkpoints"), help="la cartella dei checkpoint (per cercare un modello per etichetta)")
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    args = parser.parse_args(argv)

    cells = grid(args.strengths, args.temperatures, args.references)
    saved = json.loads(args.out.read_text()) if args.out.exists() and not args.restart else {}
    results: dict[str, dict] = saved.get("cells", {})
    # `#N` is the Nth of a ranking that moves while the fleet trains: it is resolved once, the
    # labels are kept in the file, and every later run plays those -- cells played by different
    # models cannot be paired against one baseline.
    pinned: dict[str, str] = saved.get("pinned", {})
    if not args.report:
        specs = {"model": args.model, **{f"modello {r}": r for r in args.references}}
        for key, spec in specs.items():
            if key not in pinned:
                pinned[key] = resolve_model(spec, global_dir=args.global_dir, root=args.root)[0]
        label, base_path, _rating = resolve_model(pinned["model"], global_dir=args.global_dir, root=args.root)
        workers = max(1, min(args.jobs, args.hands))
        shares = [args.hands // workers + (1 if i < args.hands % workers else 0) for i in range(workers)]
        seeds = [random.Random(args.seed).randrange(2**30) + 10 * i for i in range(workers)]
        todo = [c for c in cells if c.name not in results and (not args.only or any(c.name.startswith(o) for o in args.only))]
        for cell in todo:
            subject_path = base_path
            if cell.reference:
                subject_path = resolve_model(pinned[cell.name], global_dir=args.global_dir, root=args.root)[1]
            jobs = [CalibrationJob(cell, seed, share, str(base_path), str(subject_path),
                                   DEFAULT_TABLE_WEIGHTS, DEFAULT_STACK_MIN_BB, DEFAULT_STACK_MAX_BB,
                                   DEFAULT_SMALL_BLIND, DEFAULT_BIG_BLIND, args.session_hands, args.runouts)
                    for seed, share in zip(seeds, shares)]
            merged = CellResult()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for part in executor.map(run_job, jobs):  # in job order, so hands stay paired
                    merged.bb += part.bb
                    merged.bet_sizes += part.bet_sizes
                    merged.events = [a + b for a, b in zip(merged.events, part.events)]
                    merged.chances = [a + b for a, b in zip(merged.chances, part.chances)]
            results[cell.name] = asdict(merged)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            partial = args.out.with_suffix(".partial")
            partial.write_text(json.dumps({"model": label, "pinned": pinned, "hands": args.hands, "cells": results}))
            partial.replace(args.out)
            print(f"  {cell.name}: fatto", file=sys.stderr, flush=True)
    for line in format_report(results, cells, reference=args.reference):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
