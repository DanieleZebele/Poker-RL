"""Find the dealer button: a gold disc with a "D", on green felt.

A fixed colour rule, like the suits in `recognize.py`, so it needs no examples
and works for a seat that was never captured with the button on it. Each seat
has its own zone (`regions.dealer_region_name`); the share of gold pixels in it
says whether the button is there.

Measured on 24 labelled crops (`vision_data/dealer/`, 6-max): the four with the
button are **58-66 %** gold, the twenty without are **exactly 0 %** (the felt is
H 67-70, far from gold). The threshold sits at 20 %, low enough to still see a
button half covered by cards or chips and nowhere near the empty felt.

Needs the `vision` extra. `python -m pokerlab.vision.dealer` checks the rule
against the labelled crops.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pokerlab.vision.labels import DEALER_DIR, load_dealer_label

# Gold, in OpenCV HSV (H 0-179): yellow-orange hue, saturated, bright.
GOLD_HUE = (15, 35)
GOLD_MIN_SATURATION = 100
GOLD_MIN_VALUE = 120
PRESENT_FRACTION = 0.20


def gold_fraction(image: np.ndarray) -> float:
    """Share of the crop's pixels that are the button's gold."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gold = (
        (hsv[..., 0] >= GOLD_HUE[0]) & (hsv[..., 0] <= GOLD_HUE[1])
        & (hsv[..., 1] >= GOLD_MIN_SATURATION) & (hsv[..., 2] >= GOLD_MIN_VALUE)
    )
    return float(gold.mean())


def has_dealer(image: np.ndarray) -> bool:
    return gold_fraction(image) >= PRESENT_FRACTION


@dataclass
class DealerReading:
    seat: int | None  # where the button is; None if no zone shows it
    fractions: dict[int, float]  # gold share per seat, for display and doubt
    ambiguous: bool  # more than one zone looked like the button


def find_dealer(frames: dict[int, np.ndarray]) -> DealerReading:
    """The seat holding the button, from one crop per seat.

    There is one button, so if several zones pass the threshold (a zone drawn
    too wide, something gold drawn over the felt) the most gold one wins and
    the reading says it was ambiguous rather than pretending otherwise."""
    fractions = {seat: gold_fraction(frame) for seat, frame in frames.items()}
    present = [seat for seat, share in fractions.items() if share >= PRESENT_FRACTION]
    seat = max(present, key=lambda s: fractions[s]) if present else None
    return DealerReading(seat, fractions, len(present) > 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Controlla la regola del dealer sui ritagli etichettati.")
    parser.add_argument("--crops", type=Path, default=DEALER_DIR)
    args = parser.parse_args(argv)
    right = total = 0
    for path in sorted(Path(args.crops).glob("*.png")):
        label = load_dealer_label(path)
        image = cv2.imread(str(path))
        if label is None or image is None:
            continue
        share = gold_fraction(image)
        total += 1
        if (share >= PRESENT_FRACTION) == label["dealer"]:
            right += 1
        else:
            print(f"  {path.name}: etichetta {'presente' if label['dealer'] else 'assente'}, oro {share:.1%}")
    print(f"ritagli giusti: {right}/{total}")


if __name__ == "__main__":
    main()
