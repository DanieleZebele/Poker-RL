"""Rate checkpoints against the frozen anchors -- the pass a model of the fleet gets before it is
published -- and, with `--publish`, put them in the store at the rating they earned.

Every checkpoint starts from the same rating (`--start-rating`, by default the rating of
`--reference` in the ranking) with the same session count (`--games`) and plays the same sessions
(the seats and the cards follow `--seed`), so their results are paired: run the reference itself
among them to read the others against it.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/rate_and_publish.py \\
        --reference '#1' --sessions 500 --workers 45 a.pt b.pt [--publish --prefix distill-r1]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from duel_power import DEFAULT_GLOBAL_DIR, DEFAULT_ROOT, resolve_model

from pokerlab.config import read_config
from pokerlab.rl.benchmark import rate_against_benchmark
from pokerlab.rl.global_store import load_ranking, publish_model, write_snapshot
from pokerlab.rl.pool_registry import DEFAULT_K_SCHEDULE, parse_k_schedule
from pokerlab.rl.ppo import build_model_from_checkpoint
from pokerlab.rl.table_mix import DEFAULT_SESSION_HANDS, DEFAULT_TABLE_WEIGHTS, TableMix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--reference", default=None, help="whose rating they start from (a path, a label or #N)")
    parser.add_argument("--start-rating", type=float, default=None, help="instead of --reference's rating")
    parser.add_argument("--games", type=int, default=100,
                        help="sessions counted before the pass, as a run's validation (sets the first K)")
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1, help="forked copies playing the sessions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("checkpoints/benchmark"))
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models-dir", type=Path, default=Path("checkpoints/models"))
    parser.add_argument("--publish", action="store_true", help="publish each one at the rating it earned")
    parser.add_argument("--prefix", default="distill", help="published as <machine>-<prefix>-<file stem>-agent-<time>")
    parser.add_argument("--machine", default="zebele")
    parser.add_argument("--json", type=Path, default=None, help="also write the results here")
    args = parser.parse_args(argv)

    settings = read_config(args.config) if args.config.is_file() else {}
    schedule = parse_k_schedule(settings["k_schedule"]) if "k_schedule" in settings else DEFAULT_K_SCHEDULE
    hands = settings.get("session_hands", DEFAULT_SESSION_HANDS)
    mix = TableMix(weights=tuple(settings.get("table_weights", DEFAULT_TABLE_WEIGHTS)))
    anchors = {label: member.rating for label, member in load_ranking(args.global_dir).members.items()}
    start = args.start_rating
    if start is None:
        if args.reference is None:
            parser.error("give --reference or --start-rating")
        label, _path, start = resolve_model(args.reference, global_dir=args.global_dir, root=args.root)
        if start is None:
            parser.error(f"{label} has no rating: give --start-rating")
    print(f"start rating {start:.1f}, {args.sessions} sessions of {hands} hands, seed {args.seed}", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for path in args.checkpoints:
        model, _ = build_model_from_checkpoint(str(path), device="cpu")
        began = time.time()
        rated = rate_against_benchmark(
            model, args.benchmark_dir, mix, label="candidate", rating=start, games=args.games,
            anchor_ratings=anchors, sessions=args.sessions, hands=hands, schedule=schedule,
            workers=args.workers, seed=args.seed,
        )
        if rated is None:
            print(f"{path}: not enough anchors to seat a table")
            continue
        sizes = " ".join(f"{size}:{bb:+.0f}" for size, bb in sorted(rated.bb_per_100_by_size.items()))
        print(f"{path}\n  rating {rated.rating_after:.1f} ({rated.rating_after - start:+.1f}), "
              f"{rated.bb_per_100:+.2f} bb/100 against the anchors, {time.time() - began:.0f} s | by size {sizes}",
              flush=True)
        result = {"checkpoint": str(path), "start": start, "rating": rated.rating_after,
                  "bb_per_100": rated.bb_per_100, "by_size": rated.bb_per_100_by_size, "games": rated.games_after}
        if args.publish:
            name = f"{args.machine}-{args.prefix}-{path.stem}-agent-{stamp}.pt"
            member = publish_model(path, models_dir=args.models_dir, global_dir=args.global_dir, name=name,
                                   rating=rated.rating_after, games=rated.games_after, style=rated.style,
                                   style_hands=rated.style_hands, machine=args.machine)
            print(f"  published as {name}" if member else f"  NOT published: {name} exists", flush=True)
            result["published"] = name if member else None
        results.append(result)
    if args.publish:
        write_snapshot(args.global_dir, machine=args.machine, force=True)
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
