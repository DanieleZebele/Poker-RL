"""Is it your turn: the countdown bar the client draws for the player to act.

What the client draws (`vision_data/turn/`): a bright green bar on a dark
track, shrinking as the time runs out; when it is not your turn the zone is the
dark track alone. Measured: with the bar, 28-39 % of the zone's pixels are
saturated and bright (hue 60, pure green); without it, exactly 0 %.

So a fixed rule, like the dealer button's, that needs no examples: the bar is
there when at least `PRESENT_SHARE` of the zone is saturated and bright. Every
hue counts, not just green, because a countdown bar commonly turns yellow and
then red as it runs out; and the threshold is low (1 %) so a bar nearly spent
still reads as your turn, which an empty track (0 %) never does.

`python -m pokerlab.vision.turn` checks the rule against the labelled crops.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pokerlab.vision.labels import TURN_DIR, load_turn_label

MIN_SATURATION = 120
MIN_VALUE = 120
PRESENT_SHARE = 0.01


def bar_share(image: np.ndarray) -> float:
    """Share of the zone that is saturated and bright: the bar, of any colour."""
    _h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return float(((s > MIN_SATURATION) & (v > MIN_VALUE)).mean())


def is_my_turn(image: np.ndarray) -> bool:
    return bar_share(image) >= PRESENT_SHARE


@dataclass
class LabelledTurn:
    path: Path
    image: np.ndarray
    turn: bool


def load_turns(folder: Path = TURN_DIR) -> list[LabelledTurn]:
    found = []
    for path in sorted(Path(folder).glob("*.png")):
        label = load_turn_label(path)
        image = cv2.imread(str(path))
        if label is not None and image is not None:
            found.append(LabelledTurn(path, image, label["turn"]))
    return found


class TurnReader:
    """The rule behind the interface the screen reader uses. It takes the
    labelled examples for symmetry with the other readers, and ignores them:
    a colour rule needs none, so it is always ready."""

    def __init__(self, examples: list[LabelledTurn] = ()) -> None:
        self.ready = True

    def read(self, image: np.ndarray) -> bool:
        return is_my_turn(image)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Controlla la regola della barra del tempo.")
    parser.add_argument("--crops", type=Path, default=TURN_DIR)
    args = parser.parse_args(argv)
    crops = load_turns(args.crops)
    right = 0
    for crop in crops:
        share = bar_share(crop.image)
        if (share >= PRESENT_SHARE) == crop.turn:
            right += 1
        else:
            print(f"  {crop.path.name}: etichetta {'barra' if crop.turn else 'niente'}, colore {share:.1%}")
    shares = {kind: [bar_share(c.image) for c in crops if c.turn is kind] for kind in (True, False)}
    for kind, label in ((True, "con la barra"), (False, "senza")):
        if shares[kind]:
            print(f"{label}: {min(shares[kind]):.1%}-{max(shares[kind]):.1%} di pixel colorati")
    print(f"ritagli giusti: {right}/{len(crops)}")
    print(json.dumps({"crops": len(crops), "right": right}))


if __name__ == "__main__":
    main()
