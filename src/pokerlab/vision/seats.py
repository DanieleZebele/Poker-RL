"""Tell each seat's state from its player box: in the hand, out, sitting out, empty.

What the client draws, measured on the labelled crops in `vision_data/players/`:

- **in the hand** (an opponent): two card backs, bright pink-magenta, over the
  avatar. Measured magenta share up to 0.59; every other state has exactly 0.
- **in the hand** (an opponent at a showdown): cards face up, read by their white
  (0.109; out of the hand at most 0.023). Clubs are green in this deck: a big green area
  is not the chair, whose outline is a thin line (`CHAIR_MAX`).
- **in the hand** (you, seat 0): your cards face up, their white ranks and pips
  at full brightness (white share 0.08-0.09). Once you fold they are dimmed and
  the white goes (0.00-0.02). Brightness of the white, not the cards' colours:
  a hand of two black spades has almost no vivid colour even when it is live.
- **sitting out**: the avatar dimmed under a grey "SIT OUT" pill. Matched by
  shape: the light-grey text mask of a labelled sit-out crop is the template
  (`sit_out_templates`), so this one state does need examples.
- **empty**: no avatar, a green outline of a chair (green share 0.026; 0 for
  every other state). At 8-max the client draws no chair at all: an empty seat
  is bare table (felt, or the "888 poker" logo), so it is recognised by being
  the same picture as a crop of that same zone labelled empty
  (`empty_backgrounds`) -- one labelled example per seat is enough.
- **a reaction**: an animated emoji a player sends, drawn over the box, hiding cards
  and avatar alike -- it says nothing about the seat, and the reader keeps the state
  read before it (`ScreenReader`). The one labelled so far is a big orange face:
  orange share 0.35, every other crop at most 0.07 (`REACTION_SHARE`); reactions of
  other colours (a donkey) are recognised by being close to a crop labelled
  "reazione", in any zone (`reaction_thumbnails`) -- label one of each as it shows up.
- **out** otherwise: an avatar with nothing over it (folded, or waiting).

Fixed rules plus three kinds of example, in this order: a reaction, sit-out, then (seat 0) the
white of your cards, else magenta backs, else the chair or the zone's empty
background, else out. `python -m
pokerlab.vision.seats` checks them against the labels and lists every crop where
the rule and the label disagree -- the place to look for a mislabelled example.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pokerlab.vision.labels import (
    PLAYERS_DIR,
    SEAT_EMPTY,
    SEAT_IN_HAND,
    SEAT_OUT,
    SEAT_REACTION,
    SEAT_SIT_OUT,
)

YOUR_SEAT = 0
MAGENTA_IN_HAND = 0.10  # card backs: up to 0.59 measured, 0 without them
CHAIR_EMPTY = 0.01  # the empty-seat chair outline: 0.026 measured, 0 otherwise
# ... and a thin outline, never a big area: face-up clubs (green in the four-colour deck)
# measured 0.15-0.60 and read as an empty chair; the chair itself is 0.027-0.033.
CHAIR_MAX = 0.08
# An opponent's cards face up (a showdown): their white ranks and pips, 0.109 measured;
# an opponent out of the hand at most 0.023.
OPPONENT_WHITE_IN_HAND = 0.05
YOUR_WHITE_IN_HAND = 0.05  # your live cards: 0.08-0.09; folded: 0.00-0.02
SIT_OUT_MATCH = 0.6  # normalised correlation with a "SIT OUT" template
# An empty seat the client draws as bare table (no chair): the crop is compared
# with crops of the *same zone* labelled empty, which are pixel-identical (0.0
# measured), while an occupied seat of the same zone is >= 14.1 away.
EMPTY_MATCH = 6.0
EMPTY_THUMBNAIL = (32, 24)  # width, height
# The orange of the reaction emoji (H 5-25, S >= 120, V >= 150): 0.35 of the labelled
# reaction crop, at most 0.07 of any other (an orange avatar).
REACTION_SHARE = 0.20
# Distance (as `empty_distance`) to a crop labelled "reazione" that makes one. The three
# labelled so far (an orange face, a donkey, a fish) are >= 37.7 from every other crop; 20 leaves
# room for another frame of the same animation, which is not measured.
REACTION_MATCH = 20.0


def _hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def magenta_share(image: np.ndarray) -> float:
    h, s, v = cv2.split(_hsv(image))
    return float(((h >= 145) & (h <= 175) & (s > 100) & (v > 120)).mean())


def chair_share(image: np.ndarray) -> float:
    """The chair outline's green: measured H 65, S 239-242, V 158-164. Kept
    narrow because the 8-max felt is green too (H 76-77, S 155-173) and showed
    through a player box as a "chair" when the band was H 50-85, S > 150."""
    h, s, v = cv2.split(_hsv(image))
    return float(((h >= 58) & (h <= 72) & (s >= 195) & (v > 140)).mean())


def reaction_share(image: np.ndarray) -> float:
    h, s, v = cv2.split(_hsv(image))
    return float(((h >= 5) & (h <= 25) & (s >= 120) & (v >= 150)).mean())


def white_share(image: np.ndarray) -> float:
    _h, s, v = cv2.split(_hsv(image))
    return float(((s < 40) & (v > 220)).mean())


def text_mask(image: np.ndarray) -> np.ndarray:
    """Light-grey pixels: the "SIT OUT" lettering, and little else."""
    _h, s, v = cv2.split(_hsv(image))
    return ((s < 40) & (v > 150)).astype(np.float32)


def sit_out_template(image: np.ndarray) -> np.ndarray | None:
    """The lettering of a sit-out crop, cut to its bounding box: the blobs in
    the middle band of the box (the hood's highlights above are left out)."""
    mask = text_mask(image)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    height = mask.shape[0]
    blobs = [
        stats[i][:4] for i in range(1, count)
        if stats[i][4] >= 8 and 0.3 * height <= stats[i][1] + stats[i][3] / 2 <= 0.7 * height
    ]
    if not blobs:
        return None
    x0 = min(b[0] for b in blobs)
    y0 = min(b[1] for b in blobs)
    x1 = max(b[0] + b[2] for b in blobs)
    y1 = max(b[1] + b[3] for b in blobs)
    return mask[y0:y1, x0:x1]


def sit_out_score(image: np.ndarray, templates: list[np.ndarray]) -> float:
    """Best correlation of the crop's lettering with any sit-out template."""
    mask = text_mask(image)
    best = 0.0
    for template in templates:
        if template.shape[0] > mask.shape[0] or template.shape[1] > mask.shape[1] or not template.any():
            continue
        best = max(best, float(cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED).max()))
    return best


def _thumbnail(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, EMPTY_THUMBNAIL, interpolation=cv2.INTER_AREA).astype(np.float32)


def empty_distance(image: np.ndarray, backgrounds: list[np.ndarray]) -> float:
    """How far the crop is from the nearest picture of this same zone empty:
    mean absolute difference per channel, 0-255, on a small thumbnail."""
    if not backgrounds:
        return float("inf")
    thumb = _thumbnail(image)
    return min(float(np.abs(thumb - background).mean()) for background in backgrounds)


@dataclass
class SeatReading:
    state: str
    magenta: float
    chair: float
    white: float
    sit_out: float
    empty: float = float("inf")  # distance from this zone's empty background
    reaction: float = 0.0  # orange share of a reaction emoji


def read_seat(image: np.ndarray, seat: int, templates: list[np.ndarray],
              backgrounds: list[np.ndarray] | None = None,
              reactions: list[np.ndarray] | None = None) -> SeatReading:
    """`backgrounds`: thumbnails of this same zone labelled empty
    (`empty_backgrounds`), for clients that draw no chair on an empty seat;
    `reactions`: thumbnails of crops labelled "reazione", in any zone."""
    magenta, chair, white = magenta_share(image), chair_share(image), white_share(image)
    sit_out = sit_out_score(image, templates)
    empty = empty_distance(image, backgrounds or [])
    reaction = reaction_share(image)
    if reaction >= REACTION_SHARE or empty_distance(image, reactions or []) <= REACTION_MATCH:
        state = SEAT_REACTION
    elif sit_out >= SIT_OUT_MATCH:
        state = SEAT_SIT_OUT
    elif seat == YOUR_SEAT:
        state = SEAT_IN_HAND if white >= YOUR_WHITE_IN_HAND else SEAT_OUT
    elif magenta >= MAGENTA_IN_HAND or white >= OPPONENT_WHITE_IN_HAND:
        state = SEAT_IN_HAND
    elif CHAIR_EMPTY <= chair <= CHAIR_MAX or empty <= EMPTY_MATCH:
        state = SEAT_EMPTY
    else:
        state = SEAT_OUT
    return SeatReading(state, magenta, chair, white, sit_out, empty, reaction)


@dataclass
class LabelledSeat:
    path: Path
    image: np.ndarray
    seat: int
    state: str
    zone: str = ""


def load_seats(folder: Path = PLAYERS_DIR) -> list[LabelledSeat]:
    from pokerlab.vision.labels import load_player_label

    seats = []
    for path in sorted(Path(folder).glob("*.png")):
        label = load_player_label(path)
        image = cv2.imread(str(path))
        if label is None or image is None:
            continue
        seats.append(
            LabelledSeat(path, image, int(label["zone"].rsplit("_", 1)[1]), label["state"], label["zone"])
        )
    return seats


def empty_backgrounds(seats: list[LabelledSeat], exclude: Path | None = None) -> dict[str, list[np.ndarray]]:
    """Zone -> thumbnails of that zone labelled empty. One example per seat is
    enough, but a zone redrawn since no longer matches its old examples."""
    backgrounds: dict[str, list[np.ndarray]] = {}
    for seat in seats:
        if seat.state == SEAT_EMPTY and seat.path != exclude and seat.zone:
            backgrounds.setdefault(seat.zone, []).append(_thumbnail(seat.image))
    return backgrounds


def reaction_thumbnails(seats: list[LabelledSeat], exclude: Path | None = None) -> list[np.ndarray]:
    """Thumbnails of every crop labelled "reazione", whatever its zone: the same emoji
    is drawn over any seat."""
    return [_thumbnail(seat.image) for seat in seats if seat.state == SEAT_REACTION and seat.path != exclude]


def sit_out_templates(seats: list[LabelledSeat], exclude: Path | None = None) -> list[np.ndarray]:
    templates = []
    for seat in seats:
        if seat.state == SEAT_SIT_OUT and seat.path != exclude:
            template = sit_out_template(seat.image)
            if template is not None:
                templates.append(template)
    return templates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Controlla le regole dello stato dei posti sulle etichette.")
    parser.add_argument("--crops", type=Path, default=PLAYERS_DIR)
    args = parser.parse_args(argv)
    seats = load_seats(args.crops)
    right = 0
    for seat in seats:
        # leave-one-out for the data-driven parts: never its own template or background
        backgrounds = empty_backgrounds(seats, exclude=seat.path).get(seat.zone, [])
        reading = read_seat(seat.image, seat.seat, sit_out_templates(seats, exclude=seat.path), backgrounds,
                            reaction_thumbnails(seats, exclude=seat.path))
        if reading.state == seat.state:
            right += 1
        else:
            print(f"  {seat.path.name}: etichetta {seat.state}, regola {reading.state} "
                  f"(magenta {reading.magenta:.2f}, sedia {reading.chair:.3f}, "
                  f"bianco {reading.white:.3f}, sit-out {reading.sit_out:.2f}, "
                  f"vuoto {reading.empty:.1f}, reazione {reading.reaction:.2f})")
    print(f"ritagli d'accordo con l'etichetta: {right}/{len(seats)}")
    print(json.dumps({"seats": len(seats), "right": right}))


if __name__ == "__main__":
    main()
