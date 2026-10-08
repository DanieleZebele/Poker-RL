"""Which device a program computes on, when it is told `auto`.

`auto` is the default of every CLI that runs a network: the GPU if torch sees one,
the CPU otherwise. torch is imported inside the function, so a torch-free program
(`poker-elo`, `benchmark_arena`) reads the flag without paying for it until it
actually has a network to run.
"""

from __future__ import annotations

AUTO = "auto"


def resolve_device(name: str) -> str:
    """`name` itself, except `auto`, which becomes `cuda` when available, else `cpu`."""
    if name != AUTO:
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
