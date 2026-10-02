"""Tell each seat's state from its player box: in the hand, out, sitting out, empty.

What the client draws, measured on the labelled crops in `vision_data/players/`:

- **in the hand** (an opponent): two card backs, bright pink-magenta, over the
  avatar. Measured magenta share up to 0.59; every other state has exactly 0.
- **in the hand** (you, seat 0): your cards face up, their white ranks and pips
  at full brightness (white share 0.08-0.09). Once you fold they are dimmed and
  the white goes (0.00-0.02). Brightness of the white, not the cards' colours:
  a hand of two black spades has almost no vivid colour even when it is live.
- **sitting out**: the avatar dimmed under a grey "SIT OUT" pill. Matched by
  shape: the light-grey text mask of a labelled sit-out crop is the template
  (`sit_out_templates`), so this one state does need examples.
- **empty**: no avatar, a green outline of a chair (green share 0.026; 0 for
  every other state).
- **out** otherwise: an avatar with nothing over it (folded, or waiting).

Fixed rules plus one template, in this order: sit-out, then (seat 0) the white
of your cards, else magenta backs, else the chair, else out. `python -m
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
    SEAT_SIT_OUT,
)

YOUR_SEAT = 0
MAGENTA_IN_HAND = 0.10  # card backs: up to 0.59 measured, 0 without them
CHAIR_EMPTY = 0.01  # the empty-seat chair outline: 0.026 measured, 0 otherwise
YOUR_WHITE_IN_HAND = 0.05  # your live cards: 0.08-0.09; folded: 0.00-0.02
SIT_OUT_MATCH = 0.6  # normalised correlation with a "SIT OUT" template


def _hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def magenta_share(image: np.ndarray) -> float:
    h, s, v = cv2.split(_hsv(image))
    return float(((h >= 145) & (h <= 175) & (s > 100) & (v > 120)).mean())


def chair_share(image: np.ndarray) -> float:
    h, s, v = cv2.split(_hsv(image))
    return float(((h >= 50) & (h <= 85) & (s > 150) & (v > 150)).mean())


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


@dataclass
class SeatReading:
    state: str
    magenta: float
    chair: float
    white: float
    sit_out: float


def read_seat(image: np.ndarray, seat: int, templates: list[np.ndarray]) -> SeatReading:
    magenta, chair, white = magenta_share(image), chair_share(image), white_share(image)
    sit_out = sit_out_score(image, templates)
    if sit_out >= SIT_OUT_MATCH:
        state = SEAT_SIT_OUT
    elif seat == YOUR_SEAT:
        state = SEAT_IN_HAND if white >= YOUR_WHITE_IN_HAND else SEAT_OUT
    elif magenta >= MAGENTA_IN_HAND:
        state = SEAT_IN_HAND
    elif chair >= CHAIR_EMPTY:
        state = SEAT_EMPTY
    else:
        state = SEAT_OUT
    return SeatReading(state, magenta, chair, white, sit_out)


@dataclass
class LabelledSeat:
    path: Path
    image: np.ndarray
    seat: int
    state: str


def load_seats(folder: Path = PLAYERS_DIR) -> list[LabelledSeat]:
    from pokerlab.vision.labels import load_player_label

    seats = []
    for path in sorted(Path(folder).glob("*.png")):
        label = load_player_label(path)
        image = cv2.imread(str(path))
        if label is None or image is None:
            continue
        seats.append(LabelledSeat(path, image, int(label["zone"].rsplit("_", 1)[1]), label["state"]))
    return seats


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
        # leave-one-out for the one data-driven part: never its own template
        reading = read_seat(seat.image, seat.seat, sit_out_templates(seats, exclude=seat.path))
        if reading.state == seat.state:
            right += 1
        else:
            print(f"  {seat.path.name}: etichetta {seat.state}, regola {reading.state} "
                  f"(magenta {reading.magenta:.2f}, sedia {reading.chair:.3f}, "
                  f"bianco {reading.white:.3f}, sit-out {reading.sit_out:.2f})")
    print(f"ritagli d'accordo con l'etichetta: {right}/{len(seats)}")
    print(json.dumps({"seats": len(seats), "right": right}))


if __name__ == "__main__":
    main()
