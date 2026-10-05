"""Which CLIs share `config.toml`, and how each finds the others' parsers.

The file is one flat namespace read by five programs. Each one needs the parsers
of the others for a single reason: a key that belongs to a sibling is not a typo,
and is checked against the sibling's own type (`config.apply_config`).

The programs split by what importing them costs. `poker-elo`, `benchmark_arena`
and `poker-dashboard` are torch-free (the dashboard has to start on a box whose
disk is busy, and nothing may pull torch into them), so they see only each other
and read the file `lenient`ly. `poker-train` and `poker-loop` import torch anyway
and see everyone, which is what makes them the ones that catch a typo.
"""

from __future__ import annotations

import argparse
from importlib import import_module

TORCH_FREE = (
    "pokerlab.rl.population_arena",
    "pokerlab.rl.benchmark_arena",
    "pokerlab.rl.dashboard",
)
WITH_TORCH = (
    "pokerlab.rl.train",
    "pokerlab.rl.loop",
)


def sibling_parsers(me: str, *, with_torch: bool) -> list[argparse.ArgumentParser]:
    """The parsers of every other CLI that reads the file.

    `me` is passed as a literal module name rather than `__name__`: a tool run with
    `python -m` has `__name__ == "__main__"`, which would not exclude itself.
    Imported on demand, because most of these modules import this one's caller.
    """
    modules = TORCH_FREE + (WITH_TORCH if with_torch else ())
    return [import_module(name).build_parser() for name in modules if name != me]
