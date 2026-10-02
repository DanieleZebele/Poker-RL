"""Read the cards in a crop by matching them against the labelled examples.

The client draws a **four-colour deck** on a fixed layout, which makes this a
pattern-matching problem rather than a learning one:

- **Slots.** A board crop is cut into 5 equal slices, a hole-cards crop into 2
  halves (`split_slots`). In the hand the left card is turned slightly left and
  partly covered by the right one, turned slightly right; the halves still hold
  one rank glyph each, which is all that is read.
- **Empty or not.** A card's rank and pips are white; the felt has none. A slice
  with almost no white pixels is an empty slot (`WHITE_FRACTION_EMPTY`).
- **Rank.** The topmost white blob of a slice is the rank glyph ("10" is two
  blobs side by side, merged). It is cut to its bounding box, padded square and
  scaled to `GLYPH_SIZE`, then compared by normalised correlation with every
  labelled glyph; the best match wins (1-nearest-neighbour). Left, right and
  board glyphs are pooled, so the slight rotations are covered by examples
  rather than modelled.
- **Suit.** The colour of the card's background, by a fixed rule
  (`SUIT_RULES`): spades black, diamonds blue, clubs bright green (not to be
  confused with the felt), hearts red; the symbols on top are white. Taken as the
  median colour of the non-white pixels around the rank glyph -- on the left hole
  card that is the part the right card does not cover. No examples are needed.

Rank and suit are recognised separately, so a card never seen as a whole (one
labelled example per rank is enough) is still read correctly.

Needs the `vision` extra (numpy, opencv). Run `python -m pokerlab.vision.recognize`
to measure it on `vision_data/crops/` with leave-one-image-out: every crop is read
by a recogniser built from all the *other* crops, so the score is honest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from pokerlab.vision.labels import CROPS_DIR, load_label
from pokerlab.vision.regions import BOARD, HOLE_CARDS

SLOT_COUNT = {HOLE_CARDS: 2, BOARD: 5}
# Where each slice ends, as a fraction of the crop's width. The board is five
# equal slices. The hand is *not* cut in the middle: the right card overlaps the
# left one, and its rank glyph starts at ~83 of 170 px -- an exact half (85) cut
# two pixels off it, which the left slice then read as its own topmost glyph and
# got wrong 11 times out of 31. 0.47 (80 px) leaves the whole glyph to the right.
SLOT_EDGES = {
    BOARD: (0.2, 0.4, 0.6, 0.8, 1.0),
    HOLE_CARDS: (0.47, 1.0),
}
GLYPH_SIZE = 32
# Below this share of white pixels a slice holds no card. Measured on 65 crops:
# empty slots up to 0.027 (the felt is 0.000, but things get drawn over it),
# every card at least 0.10 -- so the threshold sits halfway, not at the felt.
WHITE_FRACTION_EMPTY = 0.06
# "White": low saturation, high brightness.
WHITE_MAX_SATURATION = 60
WHITE_MIN_VALUE = 190
# Blobs smaller than this are anti-aliasing specks, not part of a glyph.
MIN_BLOB_AREA = 8
# The rank glyph starts in the left part of its slice (measured: x 0-12 of
# 80-97 px); anything starting further right is a pip or something drawn over.
RANK_MAX_LEFT = 0.4

# The suit is the colour of the card's background (the symbols on it are white):
# spades black, diamonds blue, clubs bright green, hearts red. OpenCV HSV, H in
# 0-179. Measured on 205 cards of 70 crops, every suit sits in a tight band:
#   spades   S 0-38             V 57-64     (any hue: it is grey)
#   hearts   H 3-4      S 186-190 V 224-227
#   clubs    H 59-60    S 199-201 V 188-192
#   diamonds H 106-108  S 195-199 V 204
# and the felt, which clubs must not be taken for, at H 66-90 S 45-165 V 33-177:
# clubs are told from it by hue *and* by saturation, each with a margin. The
# bands below are deliberately much wider than measured, to survive another
# monitor or a slightly different render; they still do not overlap each other.
SUIT_RULES = (
    ("s", lambda h, s, v: v < 110 and s < 90),                  # black / dark grey
    ("h", lambda h, s, v: (h <= 12 or h >= 168) and s >= 120 and v >= 120),  # red
    ("c", lambda h, s, v: 45 <= h <= 64 and s >= 180 and v >= 120),         # bright green
    ("d", lambda h, s, v: 95 <= h <= 125 and s >= 120 and v >= 120),        # blue
)


def split_slots(image: np.ndarray, zone: str) -> list[np.ndarray]:
    """The crop cut into vertical slices, one per card position (`SLOT_EDGES`)."""
    width = image.shape[1]
    edges = (0.0, *SLOT_EDGES[zone])
    return [image[:, round(a * width):round(b * width)] for a, b in pairwise(edges)]


def white_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < WHITE_MAX_SATURATION) & (hsv[..., 2] > WHITE_MIN_VALUE)


def has_card(slot: np.ndarray) -> bool:
    return float(white_mask(slot).mean()) >= WHITE_FRACTION_EMPTY


def rank_glyph_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """`(x, y, w, h)` of the rank glyph: the topmost blob starting in the left
    part of the slice, plus any blob touching it on the same row (the "1" and "0"
    of a ten).

    Both limits come from a real misread. The client can draw things over a card
    -- a player's animated reaction sat on a 9h, white and higher than the rank --
    and a rule of "topmost blob, plus everything on its row" took it as the rank
    and merged it with the 9, reading a ten. The rank is always in the card's
    top-left corner, and the two halves of a ten are a couple of pixels apart."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    blobs = [tuple(int(v) for v in stats[i][:4]) for i in range(1, count) if stats[i][4] >= MIN_BLOB_AREA]
    left = [b for b in blobs if b[0] < RANK_MAX_LEFT * mask.shape[1]]
    if not left:
        return None
    x0, y0, w, h = min(left, key=lambda b: b[1])
    x1, y1 = x0 + w, y0 + h
    gap = max(2, h // 4)
    grown = True
    while grown:  # absorb blobs on the same row that touch the glyph so far
        grown = False
        for bx, by, bw, bh in blobs:
            centre = by + bh / 2
            touching = bx <= x1 + gap and bx + bw >= x0 - gap
            inside = bx >= x0 and bx + bw <= x1 and by >= y0 and by + bh <= y1
            if y0 <= centre <= y1 and touching and not inside:
                x0, y0 = min(x0, bx), min(y0, by)
                x1, y1 = max(x1, bx + bw), max(y1, by + bh)
                grown = True
    return (x0, y0, x1 - x0, y1 - y0)


def normalise_glyph(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """The glyph padded to a square (keeping its aspect) and scaled to
    `GLYPH_SIZE`, zero-mean and unit-norm so a dot product is a correlation."""
    x, y, w, h = box
    glyph = mask[y:y + h, x:x + w].astype(np.float32)
    side = max(w, h)
    square = np.zeros((side, side), np.float32)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = glyph
    vector = cv2.resize(square, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA).ravel()
    vector = vector - vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def suit_colour(slot: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Median colour (BGR) of the non-white pixels around the rank glyph."""
    x, y, w, h = box
    pad = max(w, h) // 2
    y0, y1 = max(0, y - pad), min(slot.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(slot.shape[1], x + w + pad)
    pixels = slot[y0:y1, x0:x1][~mask[y0:y1, x0:x1]]
    if not len(pixels):
        return np.zeros(3, np.float32)
    return np.median(pixels.reshape(-1, 3), axis=0).astype(np.float32)


@dataclass
class SlotReading:
    """What one slice shows, before it is named."""

    glyph: np.ndarray
    colour: np.ndarray


def read_slot(slot: np.ndarray) -> SlotReading | None:
    """The features of the card in a slice, or None if it is empty."""
    if not has_card(slot):
        return None
    mask = white_mask(slot)
    box = rank_glyph_box(mask)
    if box is None:
        return None
    return SlotReading(normalise_glyph(mask, box), suit_colour(slot, mask, box))


@dataclass
class Example:
    rank: str
    suit: str
    reading: SlotReading
    source: str  # the crop it came from, so leave-one-out can exclude it


def colour_hsv(colour_bgr: np.ndarray) -> tuple[int, int, int]:
    """OpenCV HSV (H 0-179, S and V 0-255) of one BGR colour."""
    pixel = np.clip(np.asarray(colour_bgr), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return int(h), int(s), int(v)


def classify_suit(colour_bgr: np.ndarray) -> str | None:
    """The suit a card's background colour stands for, or None if it is none of
    the four (`SUIT_RULES`, in that order)."""
    h, s, v = colour_hsv(colour_bgr)
    for suit, rule in SUIT_RULES:
        if rule(h, s, v):
            return suit
    return None


@dataclass
class Match:
    card: str  # rank + suit; suit "?" if the colour is none of the four
    rank_score: float  # correlation with the best glyph, 1.0 = identical
    colour_hsv: tuple[int, int, int]  # the background the suit was read from


class CardRecognizer:
    """Rank by 1-nearest-neighbour over labelled glyphs; suit by a fixed colour
    rule (`classify_suit`), which needs no examples at all."""

    def __init__(self, examples: list[Example]) -> None:
        if not examples:
            raise ValueError("servono esempi etichettati per riconoscere le carte")
        self.examples = examples
        self._glyphs = np.stack([e.reading.glyph for e in examples])

    def match(self, reading: SlotReading) -> Match:
        scores = self._glyphs @ reading.glyph
        best_rank = int(np.argmax(scores))
        suit = classify_suit(reading.colour) or "?"
        return Match(self.examples[best_rank].rank + suit, float(scores[best_rank]), colour_hsv(reading.colour))

    def recognize(self, image: np.ndarray, zone: str) -> list[Match]:
        """The cards in a crop, left to right, empty slots left out.

        Hole cards are two or none: one readable half alone means the hand is
        not really on screen (mid-animation, say), so it reads as none."""
        matches = []
        for slot in split_slots(image, zone):
            reading = read_slot(slot)
            if reading is None:
                if zone == BOARD:
                    break  # the board fills left to right: nothing after a gap
                continue
            matches.append(self.match(reading))
        if zone == HOLE_CARDS and len(matches) != 2:
            return []
        return matches


def examples_from_crop(image: np.ndarray, zone: str, cards: list[str], source: str) -> list[Example]:
    """One example per labelled card, from the slot it sits in."""
    examples = []
    for slot, card in zip(split_slots(image, zone), cards, strict=False):
        reading = read_slot(slot)
        if reading is not None:
            examples.append(Example(card[0], card[1], reading, source))
    return examples


@dataclass
class LabelledCrop:
    path: Path
    image: np.ndarray
    zone: str
    cards: list[str]


def load_dataset(folder: Path = CROPS_DIR) -> list[LabelledCrop]:
    """Every crop in `folder` that has a valid label."""
    crops = []
    for path in sorted(Path(folder).glob("*.png")):
        label = load_label(path)
        image = cv2.imread(str(path))
        if label is None or image is None:
            continue
        crops.append(LabelledCrop(path, image, label["zone"], label["cards"]))
    return crops


def build_recognizer(crops: list[LabelledCrop], exclude: Path | None = None) -> CardRecognizer:
    examples = [
        example
        for crop in crops
        if crop.path != exclude
        for example in examples_from_crop(crop.image, crop.zone, crop.cards, crop.path.name)
    ]
    return CardRecognizer(examples)


@dataclass
class Evaluation:
    crops: int
    crops_right: int
    cards: int
    cards_right: int
    errors: list[tuple[str, list[str], list[str]]]  # (crop, expected, read)


def evaluate(crops: list[LabelledCrop]) -> Evaluation:
    """Leave-one-image-out: each crop read by a recogniser that never saw it."""
    cards = cards_right = crops_right = 0
    errors = []
    for crop in crops:
        recognizer = build_recognizer(crops, exclude=crop.path)
        read = [m.card for m in recognizer.recognize(crop.image, crop.zone)]
        cards += max(len(crop.cards), len(read))
        cards_right += sum(a == b for a, b in zip(crop.cards, read, strict=False))
        if read == crop.cards:
            crops_right += 1
        else:
            errors.append((crop.path.name, crop.cards, read))
    return Evaluation(len(crops), crops_right, cards, cards_right, errors)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Riconosce le carte nei ritagli etichettati.")
    parser.add_argument("--crops", type=Path, default=CROPS_DIR)
    parser.add_argument("--image", type=Path, help="leggi solo questo ritaglio (con tutti gli altri come esempi)")
    parser.add_argument("--zone", choices=sorted(SLOT_COUNT), help="la zona di --image, se non ha etichetta")
    args = parser.parse_args(argv)

    crops = load_dataset(args.crops)
    if not crops:
        raise SystemExit(f"nessun ritaglio etichettato in {args.crops}")
    if args.image:
        label = load_label(args.image)
        zone = args.zone or (label or {}).get("zone")
        if zone is None:
            raise SystemExit("serve --zone per un ritaglio senza etichetta")
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"immagine illeggibile: {args.image}")
        recognizer = build_recognizer(crops, exclude=Path(args.image))
        for m in recognizer.recognize(image, zone):
            h, s, v = m.colour_hsv
            print(f"{m.card}  (valore {m.rank_score:.3f}, sfondo H{h} S{s} V{v})")
        if label is not None:
            print(f"etichetta: {' '.join(label['cards']) or '(nessuna carta)'}")
        return

    result = evaluate(crops)
    print(f"ritagli letti tutti giusti: {result.crops_right}/{result.crops}")
    print(f"carte giuste: {result.cards_right}/{result.cards}")
    for name, expected, read in result.errors:
        print(f"  {name}: atteso {expected}, letto {read}")
    print(json.dumps({"crops": result.crops, "crops_right": result.crops_right,
                      "cards": result.cards, "cards_right": result.cards_right}))


if __name__ == "__main__":
    main()
