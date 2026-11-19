"""A memory-arm run of `belief_history.py` as the belief file models of feature version 5 are built with.

    PYTHONPATH=src:studies/belief python studies/belief/belief_export.py \\
        --run checkpoints/studies/belief/history-v2/memory --out belief_evaluator/belief-memory-v2.pt

The file is what `rl/belief.py::load_belief_file` reads: `config` (`hidden`, `gru`, `rank`, `state`,
`embed`), `state` (`belief.*`, `memory.*`) and `equity_model`, the equity network it was trained on.
Only the `memory` arm can be exported: production reads no opponent statistics. The study's network
still has room for them in the target's statics (always zero in that arm); the export drops those
columns of the first layer of the context, which multiplied zeros, so the output is unchanged
(pinned by `tests/unit/test_study_belief.py`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from belief_records import HUD_DIM, POSITIONS

from pokerlab.rl.belief_inputs import STATIC_DIM

EMBED = 64  # `belief_model.PlayerMemory`'s default, which `belief_history.Model` keeps


def export(run: dict) -> dict:
    """The production file's content from a `belief_history.py` checkpoint."""
    if run["arm"] != "memory":
        raise ValueError(f"only the memory arm reads what production gives it, not {run['arm']!r}")
    config = run["config"]
    custom = {key: config[key] for key in ("hand_encoder", "memory", "head") if config.get(key, "gru" if key != "head" else "dot") not in ("gru", "dot")}
    if custom:
        raise ValueError(f"production reads a GRU and a dot-product head, not {custom}")
    gru = int(config["gru"])
    state = {}
    condition = bool(config.get("condition", False))
    for key, value in run["state"].items():
        if key == "belief.context.0.weight":
            hud = gru + len(POSITIONS) + 2  # [summary | position, players, stack | HUD | street, live | ...]
            value = torch.cat([value[:, :hud], value[:, hud + HUD_DIM :]], dim=1)
            if not condition:
                # trained without the condition: its column, always 0 in play, weighs nothing
                at = gru + STATIC_DIM - 1
                value = torch.cat([value[:, :at], torch.zeros(value.shape[0], 1), value[:, at:]], dim=1)
            assert value.shape[1] == gru + STATIC_DIM + 52 + int(config["state"])
        state[key] = value.clone()
    return {
        "config": {"hidden": int(config["hidden"]), "gru": gru, "rank": int(config["rank"]),
                   "state": int(config["state"]), "embed": EMBED},
        "state": state,
        "equity_model": config["equity_model"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="a belief_history.py --out directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    saved = export(torch.load(args.run / "belief.pt", map_location="cpu", weights_only=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.out)
    print(f"{args.out}: {sum(v.numel() for v in saved['state'].values())} parameters, {saved['config']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
