"""Rounds of search and distillation, each starting from the model the round before made.

One round (`search_distill.py` and `rate_and_publish.py`, each phase its own process):

    generate   the current model labels its decisions with the search, at training-style tables
    train      the policy is distilled from the labels (`--target improve`, the anchor on)
    evaluate   the distilled model against the current one, paired, at the same tables
    rate       500 sessions against the frozen anchors from the current model's rating, then
               published at the rating earned (whatever it is: a model more never hurts)

and the distilled model is the next round's current model. A *chain* is one way of choosing who
sits opposite: `self` (the current model in every seat, and the search believes so: the model
improves against itself, and the next round against that improved self) or `pool` (the best-rated
models of the store, the search believing each seat is its own model) or `mix` (the 25 best-rated
and 25 drawn at random from the rest of the store, as the training's pool), or `styled` (the pool, with
`--style-share` of the opponents playing a random style: weaker and odder players, which the store,
pruned of its weak models, no longer has and the anchors do). Several chains take turns,
round by round, until `--until` leaves no time for another round. Everything is resumable: a phase
whose output exists is skipped, so a rerun picks up where a killed one stopped.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/search_iterate.py \\
        --start-model checkpoints/benchmark/<label>.pt --start-rating 2035 --chains self,pool --until 10:30

Every round adds a line to `<root>/rounds.jsonl`. Only by hand, on the machine with the cores.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("checkpoints/studies/search_iterate")


def now() -> datetime.datetime:
    """This machine's local time, with its zone."""
    return datetime.datetime.now().astimezone()


def run(command: list[str], log: Path) -> None:
    """One phase, its output into `log`; a failure stops the chain loudly."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with log.open("a") as out:
        out.write(f"\n$ {' '.join(command)}\n")
        out.flush()
        done = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=env, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{command[2] if len(command) > 2 else command} failed ({done.returncode}): see {log}")


def labels_in(data: Path) -> int:
    import torch

    return sum(torch.load(path, weights_only=True)["features"].shape[0] for path in data.glob("job-*.pt"))


def current_of(args, chain_dir: Path) -> tuple[int, str, float, float | None]:
    """`(next round, the model it starts from, its rating, its bb/100 against the anchors)`: the last
    round's model, or with `--keep-best` the best so far by its bb/100 against the anchors (every
    pass plays the same sessions, `--seed` 0 of `rate_and_publish.py`), the start model counting as
    `--start-bb100`. A round that came out worse is still published; the chain just does not follow it."""
    done = [json.loads(path.read_text()) for path in sorted(chain_dir.glob("r*/round.json"))]
    start = (args.start_model, args.start_rating, args.start_bb100)
    if not done:
        return (1, *start)
    following = [(r["model"], r["rating"], r["bb_per_100_vs_anchors"]) for r in done]
    if not args.keep_best:
        return (done[-1]["round"] + 1, *following[-1])
    best = max([start, *following], key=lambda item: -1e9 if item[2] is None else item[2])
    return (done[-1]["round"] + 1, *best)


def play_round(args, chain: str, index: int, model: str, rating: float, parent_bb: float | None) -> dict:
    """One round of `chain`: `self`, `pool` or `mix`, with `@<tau>` for a temperature of its own
    (`self@3`, in a directory `self-t3`)."""
    kind, _, rest = chain.partition("@")
    tau, _, min_z = rest.partition("/")
    tau = tau or str(args.tau)
    rdir = args.root / chain_dir(chain) / f"r{index:02d}"
    rdir.mkdir(parents=True, exist_ok=True)
    log = rdir / "round.log"
    python = sys.executable
    distill = [python, str(HERE / "search_distill.py")]
    common = [
        "--model", model, "--players", "mix", "--opponents", "model" if kind == "self" else "pool",
        *(["--pool-top-n", "25", "--pool-random", "25"] if kind == "mix" else []),
        *(["--pool-top-n", str(args.pool_top_n), "--pool-random", str(args.pool_random)]
          if kind in ("pool", "styled", "blind") else []),
        *(["--believe", "random", "--relative-threshold"] if kind == "blind" else []),
        *(["--style-share", str(args.style_share)] if kind == "styled" else []),
        "--jobs", str(args.jobs), "--out", str(rdir), "--table-hands", str(args.table_hands),
    ]
    timings = {}
    began = time.time()
    if not (rdir / "generated").exists():
        run([*distill, "--phase", "generate", *common, "--hands", str(args.hands), "--worlds", str(args.worlds),
             "--stop-z", "3", "--dedup", "--fast-sim", "--seed", str(1_000 * index + len(chain))], log)
        (rdir / "generated").touch()
    timings["generate"] = time.time() - began
    began = time.time()
    # The labels of every earlier round of the chain made by the same model (a round that came out
    # worse, then a new one from the same parent): more data for the same step.
    same = [path.parent / "data" for path in sorted((args.root / chain_dir(chain)).glob("r*/round.json"))
            if json.loads(path.read_text())["parent"] == model]
    if not (rdir / "distilled.pt").exists():
        run([*distill, "--phase", "train", *common, *(["--extra-data", *map(str, same)] if same else []), "--target", "improve", "--tau", tau, "--min-z", min_z or "0",
             "--anchor-hands", str(args.anchor_hands), "--save-every", "0", "--train-device", "cuda"], log)
    timings["train"] = time.time() - began
    began = time.time()
    if not (rdir / "distilled-evaluate.json").exists():
        run([*distill, "--phase", "evaluate", *common, "--eval-hands", str(args.eval_hands),
             "--eval-seed", str(500 + index)], log)
    timings["evaluate"] = time.time() - began
    began = time.time()
    if not (rdir / "rate.json").exists():
        run([python, str(HERE / "rate_and_publish.py"), "--start-rating", str(rating), "--sessions",
             str(args.sessions), "--workers", str(args.jobs), str(rdir / "distilled.pt"),
             *(["--publish", "--prefix", f"distill-{chain_dir(chain)}-r{index:02d}"] if args.publish else []),
             "--json", str(rdir / "rate.json")], log)
    timings["rate"] = time.time() - began
    rated = json.loads((rdir / "rate.json").read_text())[0]
    summary = {
        "chain": chain,
        "tau": float(tau),
        "min_z": float(min_z or 0),
        "round": index,
        "parent": model,
        "parent_rating": rating,
        "parent_bb_per_100_vs_anchors": parent_bb,
        "model": str(rdir / "distilled.pt"),
        "rating": rated["rating"],
        "bb_per_100_vs_anchors": rated["bb_per_100"],
        "by_size": rated["by_size"],
        "published": rated.get("published"),
        "labels": labels_in(rdir / "data"),
        "labels_reused": [str(path) for path in same],
        "evaluate": json.loads((rdir / "distilled-evaluate.json").read_text()),
        "seconds": timings,
        "finished": now().isoformat(timespec="seconds"),
    }
    (rdir / "round.json").write_text(json.dumps(summary, indent=2))
    with (args.root / "rounds.jsonl").open("a") as out:
        out.write(json.dumps(summary) + "\n")
    return summary


def chain_dir(chain: str) -> str:
    """`self`, `pool@3` -> `pool-t3`, `pool@3/2` (tau 3, gaps under 2 standard errors left alone) ->
    `pool-t3-z2`."""
    kind, _, rest = chain.partition("@")
    tau, _, min_z = rest.partition("/")
    return kind + (f"-t{tau}" if tau else "") + (f"-z{min_z}" if min_z else "")


def deadline_of(text: str | None) -> datetime.datetime | None:
    if not text:
        return None
    hour, minute = (int(part) for part in text.split(":"))
    current = now()
    moment = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return moment if moment > current else moment + datetime.timedelta(days=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-model", required=True)
    parser.add_argument("--start-rating", type=float, required=True)
    parser.add_argument("--start-bb100", type=float, default=None,
                        help="the start model's bb/100 against the anchors in the same pass (for --keep-best)")
    parser.add_argument("--keep-best", action="store_true",
                        help="every round starts from the chain's best model so far, not from the last")
    parser.add_argument("--pool-top-n", type=int, default=50, help="a pool chain's best-rated models")
    parser.add_argument("--pool-random", type=int, default=0, help="and models drawn at random from the rest")
    parser.add_argument("--style-share", type=float, default=0.5,
                        help="a styled chain: the share of opponent seats with a random style")
    parser.add_argument("--chains", default="self,pool")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rounds", type=int, default=100, help="at most this many rounds a chain")
    parser.add_argument("--until", default=None, help="HH:MM: start no round that would not end by then")
    parser.add_argument("--hands", type=int, default=9000, help="hands labelled a round (45 tables of 200)")
    parser.add_argument("--table-hands", type=int, default=200)
    parser.add_argument("--worlds", type=int, default=300)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--anchor-hands", type=int, default=3000)
    parser.add_argument("--eval-hands", type=int, default=45000)
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=45)
    parser.add_argument("--no-publish", dest="publish", action="store_false")
    args = parser.parse_args(argv)

    chains = [chain.strip() for chain in args.chains.split(",")]
    if not {chain.partition("@")[0] for chain in chains} <= {"self", "pool", "mix", "styled", "blind"}:
        parser.error("--chains is a comma list of self, pool, mix and styled (each with an optional @tau)")
    args.root.mkdir(parents=True, exist_ok=True)
    deadline = deadline_of(args.until)
    longest = 0.0
    while True:
        played = False
        for chain in chains:
            index, model, rating, parent_bb = current_of(args, args.root / chain_dir(chain))
            if index > args.rounds:
                continue
            if deadline is not None and now() + datetime.timedelta(seconds=longest) > deadline:
                print(f"{now():%H:%M} no time for another round before {args.until}", flush=True)
                return 0
            began = time.time()
            print(f"{now():%H:%M} {chain} round {index}: from {model} (rating {rating:.1f})",
                  flush=True)
            summary = play_round(args, chain, index, model, rating, parent_bb)
            longest = max(longest, time.time() - began)
            every = summary["evaluate"].get("every hand", {})
            print(f"{now():%H:%M} {chain} round {index}: {summary['labels']} labels, against its "
                  f"parent {every.get('difference', 0):+.1f} +- {every.get('half', 0):.1f} bb/100, rating "
                  f"{summary['rating']:.1f} ({summary['rating'] - rating:+.1f}), "
                  f"{summary['bb_per_100_vs_anchors']:+.1f} bb/100 against the anchors, "
                  f"{(time.time() - began) / 60:.0f} min", flush=True)
            played = True
        if not played:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
