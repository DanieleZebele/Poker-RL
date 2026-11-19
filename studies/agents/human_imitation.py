"""Styles that play like real players: the human profiles, and a model of ours bent to each.

    PYTHONPATH=src:studies/agents python studies/agents/human_imitation.py profiles \\
        --players checkpoints/studies/human_style/players-deep.json
    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/human_imitation.py fit --jobs 40
    PYTHONPATH=src:studies/agents python studies/agents/human_imitation.py report

**Profiles** (`profiles`): the online players of HandHQ at 3-6 seats with stacks of 50 bb or more
(`human_style.py --min-stack-bb 50`), at least `--min-hands` hands, as their nine HUD statistics
(`engine/stats.py`), clustered by k-means on the standardised rates; a profile is the cluster's
median, with its share of the players.

**Fit** (`fit`): for each profile a style of `rl/styles.py` (the fifteen axes and the temperature)
for the model (`--model`, the #1 by default) such that the model playing it shows the profile's
statistics. Played at **human tables** (3-6 seats, stacks 50-100 bb, a statistics tracker, sessions
of `--session-hands`): every seat a profile drawn by its share, playing that profile's current style
plus a small random perturbation (`--perturb`); the statistics a profile shows and how each moves
with each parameter are a weighted regression of every seat-session's counts on its perturbation.
Then one Gauss-Newton step per profile on the standardised gap (a ridge on the step), the styles
updated together, `--iterations` times. The axes are in units of the full calibration of each
(`--scales`, the strength where an axis alone costs what the #50 does), kept within +-`--max-axis`.
The result, every iteration, in `--out`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokerlab.engine.stats import STATS, analyse_hand
from pokerlab.rl.styles import AXES, Style, style_fn

FULL_SCALES = (5.0, 8.0, 8.0, 4.0, 8.0, 20.0, 24.0, 32.0, 20.0, 28.0, 32.0, 28.0, 24.0, 16.0, 32.0)
PARAMS = len(AXES) + 1  # the axes, then the log of the temperature
HUMAN_SIZES = {3: 0.1, 4: 0.15, 5: 0.2, 6: 0.55}
DEFAULT_OUT = Path("checkpoints/studies/human_imitation/fit.json")
DEFAULT_PROFILES = Path("checkpoints/studies/human_imitation/profiles.json")


# ---- profiles ---------------------------------------------------------------------------


def player_rates(players: dict, *, group_prefix: str, size: str, min_hands: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """`(players, stats)` rates and chances of the players of one kind and table size, and their ids."""
    rates, chances, ids = [], [], []
    for key, row in players.items():
        group, table, player = key.split("#", 2)
        if not group.startswith(group_prefix) or table != size or row["hands"] < min_hands:
            continue
        c = np.array(row["chances"], dtype=float)
        e = np.array(row["events"], dtype=float)
        rates.append(np.where(c > 0, e / np.maximum(c, 1), np.nan))
        chances.append(c)
        ids.append(player)
    return np.array(rates), np.array(chances), ids


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, restarts: int = 20, steps: int = 100) -> np.ndarray:
    """The labels of the best of `restarts` k-means runs (k-means++ starts)."""
    best, best_cost = None, math.inf
    for _ in range(restarts):
        centres = [x[rng.integers(len(x))]]
        for _ in range(k - 1):
            d = np.min([((x - c) ** 2).sum(1) for c in centres], axis=0)
            centres.append(x[rng.choice(len(x), p=d / d.sum())])
        centres = np.array(centres)
        for _ in range(steps):
            labels = np.argmin(((x[:, None] - centres[None]) ** 2).sum(2), axis=1)
            new = np.array([x[labels == j].mean(0) if (labels == j).any() else centres[j] for j in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        cost = ((x - centres[labels]) ** 2).sum()
        if cost < best_cost:
            best, best_cost = labels, cost
    return best


def make_profiles(players: dict, *, k: int, min_hands: int, seed: int) -> dict:
    rates, chances, ids = player_rates(players, group_prefix="handhq", size="3-6", min_hands=min_hands)
    # a rate with few chances is noisy: shrink it to the population's median by its chances
    median = np.nanmedian(rates, axis=0)
    weight = chances / (chances + 20.0)
    smooth = np.where(np.isnan(rates), median, weight * np.nan_to_num(rates) + (1 - weight) * median)
    mean, sd = smooth.mean(0), smooth.std(0)
    labels = kmeans((smooth - mean) / sd, k, np.random.default_rng(seed))
    profiles, profile_of = [], {}
    for rank, j in enumerate(np.argsort(-np.bincount(labels, minlength=k))):
        members = labels == j
        profile_of.update({ids[i]: rank for i in np.flatnonzero(members)})
        profiles.append({"share": float(members.mean()), "players": int(members.sum()),
                         "stats": [float(v) for v in np.nanmedian(rates[members], axis=0)]})
    for p in profiles:
        p["name"] = describe(p["stats"], median, sd)
    return {"stats": list(STATS), "population_median": [float(v) for v in median],
            "population_sd": [float(v) for v in sd], "players": len(rates), "min_hands": min_hands,
            "profiles": profiles, "profile_of": profile_of}


SHORT = ("vpip", "pfr", "3bet", "fold3b", "steal", "aggr", "cbet", "foldcb", "wtsd")


def describe(stats, median, sd) -> str:
    """Loose or tight, aggressive or passive preflop, and the statistic furthest from the median."""
    vpip, pfr = stats[0], stats[1]
    loose = "loose" if vpip > median[0] + 0.05 else "tight" if vpip < median[0] - 0.05 else "medio"
    aggressive = "aggressivo" if pfr / max(vpip, 1e-6) > 0.55 else "passivo" if pfr / max(vpip, 1e-6) < 0.35 else "bilanciato"
    z = [(s - m) / d for s, m, d in zip(stats, median, sd)]
    far = max(range(2, len(z)), key=lambda i: abs(z[i]))
    return f"{loose}-{aggressive}, {SHORT[far]}{'+' if z[far] > 0 else '-'}"


# ---- playing ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    model_path: str
    styles: tuple[tuple[float, ...], ...]  # one parameter vector per profile
    shares: tuple[float, ...]
    perturb: float
    scales: tuple[float, ...]
    hands: int
    session_hands: int
    seed: int
    max_axis: float


def to_style(params, max_axis: float) -> Style:
    axes = tuple(float(np.clip(v, -max_axis, max_axis)) for v in params[: len(AXES)])
    return Style(axes, float(math.exp(np.clip(params[len(AXES)], -math.log(3), math.log(3)))))


def play(job: Job) -> list[dict]:
    """Every seat-session: its profile, its perturbation, its events and chances."""
    import torch

    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.table_mix import TableMix

    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    rng = random.Random(job.seed)
    policy = make_policy_fn(build_model_from_checkpoint(job.model_path)[0])
    weights = tuple(HUMAN_SIZES.get(n, 0.0) for n in range(2, 10))
    mix = TableMix(weights=weights, stack_min_bb=50.0, stack_max_bb=100.0)
    bank = TableBank(mix, random.Random(job.seed + 1), stack_rng=random.Random(job.seed + 2))
    out: list[dict] = []
    played = 0
    while played < job.hands:
        size = mix.draw_size(rng)
        seats = []
        for seat, proxy in enumerate(bank.seats(size)):
            profile = rng.choices(range(len(job.shares)), weights=job.shares)[0]
            delta = [rng.gauss(0.0, job.perturb) for _ in range(PARAMS)]
            params = [a + b for a, b in zip(job.styles[profile], delta, strict=True)]
            proxy.inner = RLAgentPlayer(proxy.player_id, proxy.name, policy_fn=policy, big_blind=mix.big_blind,
                                        starting_stack=mix.starting_stack,
                                        style=style_fn(to_style(params, job.max_axis), job.scales))
            seats.append({"profile": profile, "delta": delta, "events": [0] * len(STATS),
                          "chances": [0] * len(STATS), "seat": seat})

        def on_hand(hand, seats=seats) -> None:
            dealt = sorted(hand.starting_stacks)
            counts = analyse_hand(hand.actions, dealt=dealt, button_seat=hand.button_seat,
                                  board_cards=len(hand.community_cards), player_ids={s: f"s{s}" for s in dealt})
            for row in seats:
                mine = counts[f"s{row['seat']}"]
                for i in range(len(STATS)):
                    row["events"][i] += mine.events[i]
                    row["chances"][i] += mine.opportunities[i]

        count = min(job.session_hands, job.hands - played)
        bank.play_session(size, count, on_hand=on_hand)
        played += count
        out += seats
    return out


def regress(rows: list[dict], profiles: int) -> list[dict]:
    """Per profile and statistic: the rate at the current style and its gradient on the parameters
    (weighted least squares of the counts on the perturbation), with their standard errors."""
    found = []
    for k in range(profiles):
        mine = [r for r in rows if r["profile"] == k]
        x = np.array([[1.0] + r["delta"] for r in mine])
        rate, grad, rate_se = [], [], []
        for j in range(len(STATS)):
            c = np.array([r["chances"][j] for r in mine], dtype=float)
            e = np.array([r["events"][j] for r in mine], dtype=float)
            use = c > 0
            xs, w, y = x[use], c[use], e[use] / c[use]
            a = xs.T @ (xs * w[:, None]) + np.diag([0.0] + [1e-3 * w.sum()] * PARAMS)
            beta = np.linalg.solve(a, xs.T @ (w * y))
            residual = y - xs @ beta
            sigma2 = (w * residual**2).sum() / max(1, len(y) - xs.shape[1])
            cov = sigma2 * np.linalg.inv(a)
            rate.append(float(beta[0]))
            rate_se.append(float(math.sqrt(cov[0, 0])))
            grad.append([float(v) for v in beta[1:]])
        found.append({"rate": rate, "rate_se": rate_se, "grad": grad, "seat_sessions": len(mine)})
    return found


def step(current: np.ndarray, measured: dict, target: np.ndarray, sd: np.ndarray, ridge: float,
         max_step: float, max_axis: float) -> np.ndarray:
    """One Gauss-Newton step on the standardised gap, with a ridge and a cap on its length."""
    j = np.array(measured["grad"]) / sd[:, None]
    gap = (target - np.array(measured["rate"])) / sd
    delta = np.linalg.solve(j.T @ j + ridge * np.eye(PARAMS), j.T @ gap)
    norm = np.abs(delta).max()
    if norm > max_step:
        delta *= max_step / norm
    new = current + delta
    new[: len(AXES)] = np.clip(new[: len(AXES)], -max_axis, max_axis)
    new[len(AXES)] = np.clip(new[len(AXES)], -math.log(3), math.log(3))
    return new


def run_round(args, model_path: str, styles: np.ndarray, shares: list[float], seed: int) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    per = [args.hands // args.jobs + (1 if i < args.hands % args.jobs else 0) for i in range(args.jobs)]
    jobs = [Job(model_path, tuple(tuple(float(v) for v in s) for s in styles), tuple(shares), args.perturb,
                tuple(args.scales), n, args.session_hands, seed + 1000 * i, args.max_axis)
            for i, n in enumerate(per) if n > 0]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for part in pool.map(play, jobs):
            rows += part
    return regress(rows, len(shares))


def fit(args) -> int:
    from duel_power import resolve_model

    profiles = json.loads(args.profiles.read_text())
    target = np.array([p["stats"] for p in profiles["profiles"]])
    shares = [p["share"] for p in profiles["profiles"]]
    sd = np.array(profiles["population_sd"])
    label, path, _ = resolve_model(args.model, global_dir=Path("checkpoints/global"), root=Path("checkpoints"))
    saved = json.loads(args.out.read_text()) if args.out.exists() and not args.restart else None
    if saved:
        label, path = saved["model"], saved["model_path"]
        styles = np.array(saved["rounds"][-1]["next_styles"])
        rounds = saved["rounds"]
    else:
        styles, rounds = np.zeros((len(shares), PARAMS)), []
    print(f"modello {label}", flush=True)
    for _ in range(args.iterations - len(rounds)):
        measured = run_round(args, str(path), styles, shares, args.seed + 7919 * len(rounds))
        gaps = [float(np.sqrt((((np.array(m["rate"]) - t) / sd) ** 2).mean())) for m, t in zip(measured, target)]
        nxt = np.array([step(s, m, t, sd, args.ridge, args.max_step, args.max_axis)
                        for s, m, t in zip(styles, measured, target)])
        rounds.append({"styles": styles.tolist(), "measured": measured, "gap": gaps, "next_styles": nxt.tolist()})
        print(f"giro {len(rounds)}: distanza (in sd della popolazione) " + " ".join(f"{g:.2f}" for g in gaps), flush=True)
        styles = nxt
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": label, "model_path": str(path), "scales": list(args.scales),
                                        "profiles": profiles, "rounds": rounds}, indent=1))
    print(report(json.loads(args.out.read_text())))
    return 0


def report(saved: dict) -> str:
    profiles = saved["profiles"]["profiles"]
    names = [s[:6] for s in STATS]
    best = [min(range(len(saved["rounds"])), key=lambda r: saved["rounds"][r]["gap"][k]) for k in range(len(profiles))]
    lines = [f"modello {saved['model']}; giri {len(saved['rounds'])}", "",
             f"{'profilo':34s} {'':10s}" + "".join(f"{n:>8s}" for n in names) + "   dist"]
    first = saved["rounds"][0]
    for k, p in enumerate(profiles):
        r = saved["rounds"][best[k]]
        lines.append(f"{p['name'] + f' ({100 * p['share']:.0f}%)':34s} {'umani':10s}"
                     + "".join(f"{100 * v:8.0f}" for v in p["stats"]))
        lines.append(f"{'':34s} {'senza stile':10s}" + "".join(f"{100 * v:8.0f}" for v in first["measured"][k]["rate"])
                     + f"   {first['gap'][k]:.2f}")
        lines.append(f"{'':34s} {f'stile g{best[k] + 1}':10s}" + "".join(f"{100 * v:8.0f}" for v in r["measured"][k]["rate"])
                     + f"   {r['gap'][k]:.2f}")
        style = r["styles"][k]
        axes = ", ".join(f"{a} {v:+.2f}" for a, v in zip(AXES, style) if abs(v) >= 0.1)
        lines.append(f"{'':34s} assi: {axes}; temperatura {math.exp(style[-1]):.2f}")
        lines.append("")
    return "\n".join(lines)


def measure(args) -> int:
    """The nine statistics of each model (`--models`) playing every seat of human tables."""
    profiles = json.loads(args.profiles.read_text())
    sd = np.array(profiles["population_sd"])
    median = np.array(profiles["population_median"])
    found = {}
    for spec in args.models:
        args.perturb = 0.0
        measured = run_round(args, spec, np.zeros((1, PARAMS)), [1.0], args.seed)[0]
        found[spec] = measured
        gap = float(np.sqrt((((np.array(measured["rate"]) - median) / sd) ** 2).mean()))
        print(f"{spec}: " + " ".join(f"{s[:6]} {100 * v:.0f}" for s, v in zip(STATS, measured["rate"]))
              + f"; distanza dalla mediana umana {gap:.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, indent=1))
    return 0


@dataclass(frozen=True)
class DuelJob:
    subject_path: str
    base_path: str
    style: tuple[float, ...] | None
    scales: tuple[float, ...]
    hands: int
    seed: int
    runouts: int
    max_axis: float


def duel(job: DuelJob) -> list[float]:
    """The subject's result per hand (all-in hands at their expectation), one seat, the base model in
    every other, at the training mixture; the same seeds give the same hands to every subject."""
    import torch

    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.allin_reward import expected_deltas
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.table_mix import DEFAULT_TABLE_WEIGHTS, TableMix

    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    base = make_policy_fn(build_model_from_checkpoint(job.base_path)[0])
    subject = base if job.subject_path == job.base_path else make_policy_fn(build_model_from_checkpoint(job.subject_path)[0])
    styled = None if job.style is None else style_fn(to_style(job.style, job.max_axis), job.scales)
    mix = TableMix(weights=DEFAULT_TABLE_WEIGHTS, stack_min_bb=1.0, stack_max_bb=100.0)
    bank = TableBank(mix, random.Random(job.seed), stack_rng=random.Random(job.seed + 1))
    seat_rng, size_rng, runout_rng = random.Random(job.seed + 2), random.Random(job.seed + 3), random.Random(job.seed + 4)
    results: list[float] = []
    while len(results) < job.hands:
        size = mix.draw_size(size_rng)
        me = seat_rng.randrange(size)
        for seat, proxy in enumerate(bank.seats(size)):
            proxy.inner = RLAgentPlayer(proxy.player_id, proxy.name, policy_fn=subject if seat == me else base,
                                        big_blind=mix.big_blind, starting_stack=mix.starting_stack,
                                        style=styled if seat == me else None)

        def on_hand(hand, seat: int = me) -> None:
            expected = expected_deltas(hand, runout_rng, job.runouts)
            chips = expected[seat] if expected is not None else hand.final_stacks[seat] - hand.starting_stacks[seat]
            results.append(chips / hand.big_blind)

        bank.play_session(size, min(100, job.hands - len(results)), on_hand=on_hand)
    return results


def strength(args) -> int:
    """bb/100 of each subject in one seat against the base model, paired hand by hand with the base
    model in that seat: the profiles' fitted styles on the base model, and any other model."""
    from concurrent.futures import ProcessPoolExecutor

    from duel_power import resolve_model

    fitted = json.loads(args.fit.read_text())
    base_path = fitted["model_path"]
    subjects: dict[str, tuple[str, tuple | None]] = {"base": (base_path, None)}
    profiles = fitted["profiles"]["profiles"]
    for k, profile in enumerate(profiles):
        best = min(range(len(fitted["rounds"])), key=lambda r: fitted["rounds"][r]["gap"][k])
        subjects[f"stile {profile['name']}"] = (base_path, tuple(fitted["rounds"][best]["styles"][k]))
    for spec in args.models:
        path = spec if Path(spec).exists() else str(resolve_model(spec, global_dir=Path("checkpoints/global"), root=Path("checkpoints"))[1])
        subjects[spec] = (path, None)
    saved = json.loads(args.out.read_text()) if args.out.exists() else {}
    per = [args.hands // args.jobs + (1 if i < args.hands % args.jobs else 0) for i in range(args.jobs)]
    seeds = [args.seed + 101 * i for i in range(args.jobs)]
    for name, (path, style) in subjects.items():
        if name in saved:
            continue
        jobs = [DuelJob(path, base_path, style, tuple(fitted["scales"]), n, seed, 20, args.max_axis)
                for n, seed in zip(per, seeds, strict=True)]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            saved[name] = [x for part in pool.map(duel, jobs) for x in part]
        args.out.write_text(json.dumps(saved))
        print(f"  {name}: fatto", file=sys.stderr, flush=True)
    base = np.array(saved["base"])
    print(f"contro {fitted['model']} negli altri posti, {len(base)} mani accoppiate; bb/100 meno il modello stesso nel posto")
    for name, values in saved.items():
        diff = 100 * (np.array(values) - base)
        print(f"{name:40s} {diff.mean():+7.1f} +- {1.96 * diff.std() / math.sqrt(len(diff)):.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profiles")
    p.add_argument("--players", type=Path, default=Path("checkpoints/studies/human_style/players-deep.json"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min-hands", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_PROFILES)
    f = sub.add_parser("fit")
    f.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    f.add_argument("--model", default="#1")
    f.add_argument("--iterations", type=int, default=6)
    f.add_argument("--hands", type=int, default=400000, help="hands per round")
    f.add_argument("--session-hands", type=int, default=100)
    f.add_argument("--perturb", type=float, default=0.25)
    f.add_argument("--ridge", type=float, default=0.3)
    f.add_argument("--max-step", type=float, default=0.6)
    f.add_argument("--max-axis", type=float, default=1.5)
    f.add_argument("--scales", type=float, nargs=len(AXES), default=list(FULL_SCALES))
    f.add_argument("--jobs", type=int, default=40)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--restart", action="store_true")
    f.add_argument("--out", type=Path, default=DEFAULT_OUT)
    m = sub.add_parser("measure")
    m.add_argument("--models", nargs="+", required=True, help="checkpoint paths")
    m.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    m.add_argument("--hands", type=int, default=200000)
    m.add_argument("--session-hands", type=int, default=100)
    m.add_argument("--max-axis", type=float, default=1.5)
    m.add_argument("--scales", type=float, nargs=len(AXES), default=list(FULL_SCALES))
    m.add_argument("--jobs", type=int, default=40)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", type=Path, default=Path("checkpoints/studies/human_imitation/measure.json"))
    d = sub.add_parser("strength")
    d.add_argument("--fit", type=Path, default=DEFAULT_OUT)
    d.add_argument("--models", nargs="*", default=[], help="other subjects: checkpoint paths or #N")
    d.add_argument("--hands", type=int, default=200000)
    d.add_argument("--max-axis", type=float, default=3.0, help="the limit the fit ran with")
    d.add_argument("--jobs", type=int, default=40)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", type=Path, default=Path("checkpoints/studies/human_imitation/strength.json"))
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.command == "profiles":
        made = make_profiles(json.loads(args.players.read_text()), k=args.k, min_hands=args.min_hands, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(made, indent=1))
        print(f"{made['players']} giocatori; " + " ".join(f"{s[:6]:>7s}" for s in STATS))
        print(f"{'mediana':28s}" + "".join(f"{100 * v:7.0f}" for v in made["population_median"]))
        for prof in made["profiles"]:
            print(f"{prof['name'] + f' {100 * prof['share']:.0f}%':28s}" + "".join(f"{100 * v:7.0f}" for v in prof["stats"]))
        return 0
    if args.command == "fit":
        return fit(args)
    if args.command == "measure":
        return measure(args)
    if args.command == "strength":
        return strength(args)
    print(report(json.loads(args.out.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
