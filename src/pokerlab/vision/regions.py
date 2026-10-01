"""Where on the screen the things worth reading appear.

Pure Python, no dependency: a region is four integers and the set of regions is a
small JSON file, so both can be handled and tested without the `vision` extra.

Coordinates are in **screen pixels of the virtual desktop**, the same ones `mss`
reports for a monitor (`left`/`top` can be negative on a multi-monitor setup), so
a region captured later is exactly the rectangle that was selected.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_REGIONS_PATH = Path("checkpoints/vision/regions.json")

# The names of the regions the GUI knows about.
HOLE_CARDS = "hole_cards"
BOARD = "board"
REGION_NAMES = (HOLE_CARDS, BOARD)


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"una zona deve avere dimensioni positive, non {self.width}x{self.height}")

    def as_monitor(self) -> dict[str, int]:
        """The dict `mss.grab` takes."""
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass
class RegionConfig:
    regions: dict[str, Region] = field(default_factory=dict)

    def get(self, name: str) -> Region | None:
        return self.regions.get(name)

    def set(self, name: str, region: Region) -> None:
        self.regions[name] = region

    def clear(self, name: str) -> None:
        self.regions.pop(name, None)


def load_regions(path: Path = DEFAULT_REGIONS_PATH) -> RegionConfig:
    """The saved regions, or an empty config.

    A missing, unreadable or malformed file is "no regions yet", never an error:
    the file is written by a tool and edited by hand, and a typo in it must not
    stop the screen it is read from opening. An entry that does not describe a
    valid region is skipped on its own, so one bad zone does not take the others
    with it.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = raw["regions"]
    except (OSError, ValueError, KeyError, TypeError):
        return RegionConfig()
    config = RegionConfig()
    if not isinstance(entries, dict):
        return config
    for name, values in entries.items():
        try:
            config.set(name, Region(**values))
        except (TypeError, ValueError):
            continue
    return config


def save_regions(config: RegionConfig, path: Path = DEFAULT_REGIONS_PATH) -> None:
    """Written whole through a dotted temporary and renamed into place, like every
    other file here that something may read while it is being written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"regions": {name: asdict(region) for name, region in config.regions.items()}}
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    staging.replace(path)
