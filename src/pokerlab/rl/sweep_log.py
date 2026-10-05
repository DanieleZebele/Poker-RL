"""What a finished child says about the step it took from its parent.

The sweep optimizer (`rl/sweep_optimizer.py`) learns which way to move each
hyperparameter from one kind of evidence: *a child that was trained from a parent
with its settings moved by some step, and the rating gain it got for the compute
it cost*. Each worker that resumed from a parent writes one small file when it
publishes, so the evidence accumulates across every machine without anyone having
to open the published checkpoints (3 MB each, thousands of them, on NFS).

- **One file per child**, `<global-dir>/sweep/<label>.json`, written once and
  atomically (a dotted `.partial` renamed into place): the project's per-model
  pattern, which needs no lock and cannot interleave two writers on a shared
  volume the way appending to one file could.
- **Both ratings are the ones the models were published with**, frozen in the
  record, not looked up later. They are the same kind of number (earned against
  the pinned anchors over the benchmark pass), so their difference is a fair gain,
  and a parent that has since been pruned does not take its evidence with it.
- **`cpu_seconds` is the process's own CPU time, not wall time**: a machine
  running twenty workers makes wall time depend on its neighbours, and a setting
  would be charged for sharing a box with a slow one.
- **Readers are tolerant**: a half-written, foreign or non-finite file is skipped,
  never an error, because a supervisor must not die reading evidence.

Pure Python, no torch, like the other modules the supervisor and the dashboard
read.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SWEEP_DIRNAME = "sweep"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SweepObservation:
    label: str
    parent: str
    parent_rating: float  # what the parent was published with
    rating: float  # what the child was published with
    cpu_seconds: float  # CPU the child cost, up to its publication
    settings: dict[str, float]  # the child's axes
    parent_settings: dict[str, float]  # the parent's axes: the step is the difference
    machine: str = ""
    time: float = 0.0  # when it was written (epoch seconds)

    @property
    def gain(self) -> float:
        """Elo the child gained over its parent."""
        return self.rating - self.parent_rating


def _directory(global_dir: str | Path) -> Path:
    return Path(global_dir) / SWEEP_DIRNAME


def write_observation(global_dir: str | Path, observation: SweepObservation) -> Path | None:
    """Write `observation` once. None if that label already has one."""
    directory = _directory(global_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{observation.label}.json"
    if target.exists():
        return None
    record = {"schema": SCHEMA_VERSION, **asdict(observation)}
    if not record["time"]:
        record["time"] = time.time()
    partial = directory / f".{observation.label}.json.partial"
    partial.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    os.replace(partial, target)
    return target


def _numbers(value: object) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            return None
        out[str(key)] = float(item)
    return out


def _parse(text: str) -> SweepObservation | None:
    try:
        raw = json.loads(text)
    except ValueError:
        return None
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_VERSION:
        return None
    settings = _numbers(raw.get("settings"))
    parent_settings = _numbers(raw.get("parent_settings"))
    scalars = {}
    for key in ("parent_rating", "rating", "cpu_seconds", "time"):
        item = raw.get(key)
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            return None
        scalars[key] = float(item)
    if settings is None or parent_settings is None:
        return None
    if not isinstance(raw.get("label"), str) or not isinstance(raw.get("parent"), str):
        return None
    return SweepObservation(
        label=raw["label"],
        parent=raw["parent"],
        settings=settings,
        parent_settings=parent_settings,
        machine=str(raw.get("machine", "")),
        **scalars,
    )


def read_observations(global_dir: str | Path, limit: int | None = None) -> list[SweepObservation]:
    """The most recent `limit` observations (all if None), oldest first.

    "Recent" is by file modification time, which costs one `stat` per file and no
    read, so a store of thousands is listed cheaply and only the window is opened.
    """
    directory = _directory(global_dir)
    if not directory.is_dir():
        return []
    entries: list[tuple[float, Path]] = []
    try:
        with os.scandir(directory) as scan:
            for entry in scan:
                if entry.name.endswith(".json") and not entry.name.startswith("."):
                    try:
                        entries.append((entry.stat().st_mtime, Path(entry.path)))
                    except OSError:
                        continue
    except OSError:
        return []
    entries.sort(key=lambda pair: pair[0])
    if limit is not None and limit > 0:
        entries = entries[-limit:]
    found: list[SweepObservation] = []
    for _mtime, path in entries:
        try:
            observation = _parse(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if observation is not None:
            found.append(observation)
    return found
