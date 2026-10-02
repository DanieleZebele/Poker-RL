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

# Outside `checkpoints/` on purpose, next to the crops (`labels.CROPS_DIR`):
# `checkpoints/` is training state, and these zones belong to the vision data.
DEFAULT_REGIONS_PATH = Path("vision_data/regions.json")

# The names of the regions the GUI knows about.
HOLE_CARDS = "hole_cards"
BOARD = "board"
REGION_NAMES = (HOLE_CARDS, BOARD)

# Where the dealer button can appear, one zone per seat, and the seat positions
# depend on how many players the table seats. Seat 0 is you (the client draws
# you at the bottom), then clockwise *as seen on screen* from your left -- the
# same order as the chairs of the spot screen. Only 6-max is mapped so far.
DEALER_TABLE_SIZES = (6,)
# A dealer zone is drawn round the button alone (measured 36-41 px a side). One
# twice that is almost certainly a player box drawn in the wrong section -- it
# happened, seat 5 at 149x111 -- and the gold of that player's stack chip could
# then be read as the button. Such a zone is flagged and not read.
DEALER_ZONE_MAX_SIDE = 80


# The zone around the pot's total, and one per seat around the chips a player has
# put in this street (the number drawn in front of them). Same seat numbering.
POT = "pot"
# The zone where the client draws the countdown bar when it is *your* turn.
TURN_TIMER = "turn_timer"


def stack_region_name(players: int, seat: int) -> str:
    """`stack_6_3`: where seat 3's stack (chips behind) is written at 6-max."""
    if players not in DEALER_TABLE_SIZES or not 0 <= seat < players:
        raise ValueError(f"nessuna zona stack per il posto {seat} a {players} giocatori")
    return f"stack_{players}_{seat}"


def bet_region_name(players: int, seat: int) -> str:
    """`bet_6_3`: where seat 3's bet amount is written at a 6-max table."""
    if players not in DEALER_TABLE_SIZES or not 0 <= seat < players:
        raise ValueError(f"nessuna zona puntata per il posto {seat} a {players} giocatori")
    return f"bet_{players}_{seat}"


def player_region_name(players: int, seat: int) -> str:
    """`player_6_3`: the zone around seat 3's player box (avatar, name, stack)
    at a 6-max table, read to tell in hand / out of the hand / empty seat. Same
    seat numbering as the dealer zones."""
    if players not in DEALER_TABLE_SIZES or not 0 <= seat < players:
        raise ValueError(f"nessuna zona giocatore per il posto {seat} a {players} giocatori")
    return f"player_{players}_{seat}"


def dealer_region_name(players: int, seat: int) -> str:
    """`dealer_6_3`: the zone where seat 3 of a 6-max table shows the button."""
    if players not in DEALER_TABLE_SIZES or not 0 <= seat < players:
        raise ValueError(f"nessuna zona dealer per il posto {seat} a {players} giocatori")
    return f"dealer_{players}_{seat}"


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
