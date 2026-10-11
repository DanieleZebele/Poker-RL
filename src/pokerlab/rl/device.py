"""Which device a program computes on, when it is told `auto`.

`auto` is the default of every CLI that runs a network: the GPU if torch sees one,
the CPU otherwise. torch is imported inside the function, so a torch-free program
(`poker-elo`, `benchmark_arena`) reads the flag without paying for it until it
actually has a network to run.
"""

from __future__ import annotations

AUTO = "auto"


SAME = "same"


def resolve_update_device(name: str, device: str) -> str:
    """Where the PPO update (and the critic) run, apart from where the hands are played:
    `same` is `device` itself and `auto` the GPU if torch sees one, else `device`; anything
    else is taken as given. The hands stay on `device` (the CPU, in practice): one decision
    at a time is no faster on a GPU, while the update works on big batches."""
    if name == SAME:
        return device
    if name == AUTO:
        import torch

        return "cuda" if torch.cuda.is_available() else device
    return name


def resolve_device(name: str) -> str:
    """`name` itself, except `auto`, which becomes `cuda` when available, else `cpu`."""
    if name != AUTO:
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
