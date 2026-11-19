"""Make chosen store models anchors, at the rating each holds: the hand-picked counterpart of the
fleet's own rule (`global_arena.add_benchmark_candidates`), for models that are references by what
they are (the clones of real players, `studies/human_clone`) rather than by being the strongest.

    PYTHONPATH=src:studies/agents python studies/agents/add_anchors.py LABEL [LABEL ...]

Publish them first (`rate_and_publish.py --publish`, which rates them against the anchors), then
this: each is moved into `checkpoints/benchmark/` and frozen (`global_arena.promote_to_benchmark`),
under the `__prune__` lock so no prune or promotion runs meanwhile, and the snapshot is refreshed.
Then settle them against the other anchors with `python -m pokerlab.rl.benchmark_arena`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from duel_power import DEFAULT_GLOBAL_DIR, DEFAULT_ROOT

from pokerlab.rl.global_arena import promote_to_benchmark
from pokerlab.rl.global_store import (
    PRUNE_LOCK,
    PRUNE_LOCK_SECONDS,
    acquire_locks,
    release_locks,
    write_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--machine", default="zebele")
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if not acquire_locks(args.global_dir, [PRUNE_LOCK], machine=args.machine, ttl=PRUNE_LOCK_SECONDS):
        print("a prune or a promotion is running: try again in a few minutes")
        return 1
    try:
        for label in args.labels:
            member = promote_to_benchmark(args.global_dir, args.root, label, machine=args.machine)
            print(f"{label}: " + (f"anchor at {member.rating:.1f} ({member.games} games)" if member
                                  else "NOT moved (being rated, unknown, already an anchor or no file)"))
    finally:
        release_locks(args.global_dir, [PRUNE_LOCK])
    write_snapshot(args.global_dir, machine=args.machine, force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
