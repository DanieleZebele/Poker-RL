"""Read the bet and pot amounts: white digits on a dark pill, then " BB".

What the client draws (`vision_data/amounts/`): the amount in big blinds, white
on a dark rounded pill, always followed by "BB"; decimals use a comma ("18,5
BB"). The label holds the number only ("18,5"), and "" where nothing is shown.

Reading, in the spirit of the card ranks (`recognize.py`):

1. the white pixels, split into blobs, left to right;
2. the last two blobs are the "BB" and are dropped -- a reading whose blobs do
   not end in two B-sized blobs is not trusted;
3. a blob much shorter than the digits, sitting at their baseline, is the comma;
4. every other blob is a digit, matched by normalised correlation (1-NN) with
   the digits cut out of the labelled crops -- a crop whose blobs line up one
   to one with its label lends one example per digit.

A digit never labelled cannot be read by matching; `examples_from_crop` reports
which digits the examples cover, so a gap is visible rather than a silent
misread. `python -m pokerlab.vision.amounts` measures it leave-one-out.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pokerlab.vision.labels import AMOUNTS_DIR, STACKS_DIR, load_amount_label
from pokerlab.vision.recognize import normalise_glyph

WHITE_MAX_SATURATION = 70
WHITE_MIN_VALUE = 170
# The stacks are written in yellow (the user's description; to be measured on
# the first labelled stack crops). Hue 15-40 in OpenCV covers yellow to gold.
YELLOW_HUE = (15, 40)
YELLOW_MIN_SATURATION = 80
YELLOW_MIN_VALUE = 140
# Blobs after a stack's number: 2 if it is written "60 BB" like a bet, 0 if
# just "60". Assumed like the bets until the stack crops say otherwise.
STACK_SUFFIX_BLOBS = 2
MIN_BLOB_AREA = 6
# A comma is far shorter than a digit (measured ~0.3 of the digit height).
COMMA_MAX_HEIGHT = 0.55
DIGITS = "0123456789"
# A digit is over half the crop's height (~18 of 31-35 px); the white stripes
# of chips lying in an empty zone are 3-5 px.
DIGIT_MIN_HEIGHT = 0.35
# Trust a match against the bets' own digits from this correlation up; below,
# read the digit against the card ranks instead (see `AmountReader`).
OWN_SURE = 0.8
# A loop smaller than this (in the 32x32 glyph) is an artefact, not a loop.
MIN_HOLE_AREA = 2


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int

    @property
    def bottom(self) -> int:
        return self.y + self.h


def is_stack_zone(zone: str) -> bool:
    return zone.startswith("stack_")


def yellow_mask(image: np.ndarray) -> np.ndarray:
    """The stacks' lettering: yellow-gold, saturated and bright."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= YELLOW_HUE[0]) & (h <= YELLOW_HUE[1]) & (s > YELLOW_MIN_SATURATION) & (v > YELLOW_MIN_VALUE)


def text_mask(image: np.ndarray, zone: str = "") -> np.ndarray:
    """The lettering of a zone: white for bets and the pot, yellow for stacks.
    Only the isolating differs -- the digits are then compared as black-and-
    white shapes, so white bet digits and yellow stack digits lend each other
    examples."""
    return yellow_mask(image) if is_stack_zone(zone) else white_mask(image)


def suffix_blobs(zone: str) -> int:
    """How many blobs follow the number: the "BB" of a bet (2); for a stack,
    `STACK_SUFFIX_BLOBS`."""
    return STACK_SUFFIX_BLOBS if is_stack_zone(zone) else 2


def white_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < WHITE_MAX_SATURATION) & (hsv[..., 2] > WHITE_MIN_VALUE)


def blobs(mask: np.ndarray) -> list[Blob]:
    """White blobs left to right; specks dropped, and blobs on top of each
    other (none in this font, but a stray mark could be) kept apart."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    found = [Blob(*(int(v) for v in stats[i][:4])) for i in range(1, count) if stats[i][4] >= MIN_BLOB_AREA]
    return sorted(found, key=lambda b: b.x)


@dataclass
class Layout:
    """The blobs of one crop, split into what they are."""

    digits: list[Blob]  # in order; a comma's place is recorded in `comma_after`
    comma_after: int | None  # index of the digit the comma follows, if any
    ok: bool  # ended in "BB" and the rest made sense


def layout(mask: np.ndarray, suffix: int = 2) -> Layout:
    """`suffix` blobs at the end are the unit ("BB") and are dropped."""
    found = blobs(mask)
    if len(found) < suffix + 1:
        return Layout([], None, False)
    body, bb = (found[:-suffix], found[-suffix:]) if suffix else (found, [])
    height = max(b.h for b in body + bb)
    if any(b.h < 0.7 * height for b in bb):
        return Layout([], None, False)  # the last ones are not the "BB"
    digits, comma_after = [], None
    for blob in body:
        if blob.h < COMMA_MAX_HEIGHT * height:
            if comma_after is not None or not digits:
                return Layout([], None, False)  # two commas, or one leading
            comma_after = len(digits) - 1
        else:
            digits.append(blob)
    return Layout(digits, comma_after, bool(digits))


def glyph(mask: np.ndarray, blob: Blob) -> np.ndarray:
    return normalise_glyph(mask, (blob.x, blob.y, blob.w, blob.h))


def has_amount(image: np.ndarray, zone: str = "") -> bool:
    """Whether the zone shows an amount (an empty zone is felt)."""
    return layout(text_mask(image, zone), suffix_blobs(zone)).ok


@dataclass
class DigitExample:
    digit: str
    glyph: np.ndarray
    source: str


def examples_from_crop(image: np.ndarray, text: str, source: str, zone: str = "") -> list[DigitExample]:
    """One example per digit, when the crop's digit blobs line up one to one
    with its label (comma included); nothing otherwise."""
    if not text:
        return []
    mask = text_mask(image, zone)
    found = layout(mask, suffix_blobs(zone))
    digits = text.replace(",", "")
    if not found.ok or len(found.digits) != len(digits):
        return []
    expected_comma = text.index(",") - 1 if "," in text else None
    if found.comma_after != expected_comma:
        return []
    return [DigitExample(d, glyph(mask, b), source) for d, b in zip(digits, found.digits, strict=True)]


@dataclass
class AmountReading:
    text: str  # "" when the zone shows no amount; "?" in place of a digit not read
    score: float  # the worst digit's correlation, 1.0 = identical


def glyph_holes(vector: np.ndarray) -> int:
    """How many closed loops a normalised glyph has: 8 has two; 0, 4, 6 and 9
    one; 1, 2, 3, 5 and 7 none. Measured identical for every example of every
    digit, in the bets and in the card ranks alike."""
    side = round(len(vector) ** 0.5)
    strokes = (vector.reshape(side, side) > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(np.pad(strokes, 1), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return 0
    return sum(
        1 for contour, links in zip(contours, hierarchy[0], strict=True)
        if links[3] != -1 and cv2.contourArea(contour) >= MIN_HOLE_AREA
    )


class _Matcher:
    """1-NN restricted to the examples with the same number of loops.

    Correlation alone was a coin toss between 3 and 8 -- same right half, the
    3 only missing two thin strokes on the left: a "3" one pixel taller than
    the others scored 0.825 against the 3s and 0.830 against the 8s. A loop
    count does not move with a pixel, so a 3 is never compared with an 8."""

    def __init__(self, examples: list[DigitExample]) -> None:
        self.examples = examples
        self.glyphs = np.stack([e.glyph for e in examples]) if examples else None
        self.holes = np.array([glyph_holes(e.glyph) for e in examples])

    def best(self, vector: np.ndarray) -> tuple[str, float] | None:
        if self.glyphs is None:
            return None
        scores = self.glyphs @ vector
        same = self.holes == glyph_holes(vector)
        if same.any():  # otherwise a stray mark broke a loop: fall back to all
            scores = np.where(same, scores, -np.inf)
        index = int(np.argmax(scores))
        return self.examples[index].digit, float(scores[index])


class AmountReader:
    """1-NN over the digits cut from labelled bets, falling back to the card
    ranks (the same typeface) for a digit the bets do not match well.

    Two stages, not one pool, because measured that way the sources do not
    compete fairly: a bet digit matches another bet digit's rendering better
    than a card's, so a "6" with no bet example of its own matched a bet "5"
    (0.73) over the card "6" (0.70). Instead: when the bets' best match is
    below `OWN_SURE` -- right answers scored >= 0.865, wrong ones <= 0.73 --
    the digit is read against the card ranks alone, which by themselves read
    14 of 14 bet digits right."""

    def __init__(self, examples: list[DigitExample], fallback: list[DigitExample] = ()) -> None:
        self._own = _Matcher(list(examples))
        self._fallback = _Matcher(list(fallback))
        self.examples = list(examples)
        self.covered = sorted({e.digit for e in [*examples, *fallback]})
        self._glyphs = self._own.glyphs if examples else self._fallback.glyphs

    def _digit(self, vector: np.ndarray) -> tuple[str, float]:
        own = self._own.best(vector)
        if own is not None and own[1] >= OWN_SURE:
            return own
        fallback = self._fallback.best(vector)
        if fallback is None:
            return own  # nothing better to ask
        return fallback if own is None or fallback[1] > 0 else own

    def read(self, image: np.ndarray, zone: str = "") -> AmountReading | None:
        """The amount, or None when the zone is not readable. `zone` picks the
        lettering's colour and unit (a stack is yellow)."""
        mask = text_mask(image, zone)
        found = layout(mask, suffix_blobs(zone))
        if not found.ok:
            # No blob as tall as a digit: no pill there (chips have white
            # stripes, 3-5 px against digits of ~18), so no amount.
            tall = [b for b in blobs(mask) if b.h >= DIGIT_MIN_HEIGHT * mask.shape[0]]
            return AmountReading("", 1.0) if not tall else None
        if self._glyphs is None:
            return None
        text, worst = "", 1.0
        for index, blob in enumerate(found.digits):
            digit, score = self._digit(glyph(mask, blob))
            text += digit
            worst = min(worst, score)
            if found.comma_after == index:
                text += ","
        return AmountReading(text, worst)


def amount_value(text: str) -> float | None:
    """ "18,5" -> 18.5 (big blinds); None for "" or an unreadable text."""
    try:
        return float(text.replace(",", ".")) if text else None
    except ValueError:
        return None


@dataclass
class LabelledAmount:
    path: Path
    image: np.ndarray
    zone: str
    text: str


def load_amounts(folder: Path = AMOUNTS_DIR) -> list[LabelledAmount]:
    found = []
    for path in sorted(Path(folder).glob("*.png")):
        label = load_amount_label(path)
        image = cv2.imread(str(path))
        if label is not None and image is not None:
            found.append(LabelledAmount(path, image, label["zone"], label["text"]))
    return found


def card_digit_examples(card_crops=None) -> list[DigitExample]:
    """The digits 2-9 cut from the labelled card crops (a ten is one merged
    glyph and is left out), as a fallback for digits never seen in a bet."""
    from pokerlab.vision import recognize

    crops = recognize.load_dataset() if card_crops is None else card_crops
    found = []
    for crop in crops:
        for example in recognize.examples_from_crop(crop.image, crop.zone, crop.cards, crop.path.name):
            if example.rank in "23456789":
                found.append(DigitExample(example.rank, example.reading.glyph, example.source))
    return found


def build_reader(crops: list[LabelledAmount], exclude: Path | None = None,
                 fallback: list[DigitExample] | None = None) -> AmountReader:
    examples = []
    for crop in crops:
        if crop.path != exclude:
            examples += examples_from_crop(crop.image, crop.text, crop.path.name, crop.zone)
    return AmountReader(examples, card_digit_examples() if fallback is None else fallback)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Legge le puntate etichettate, leave-one-out.")
    parser.add_argument("--crops", type=Path, nargs="+", default=[AMOUNTS_DIR, STACKS_DIR],
                        help="cartelle dei ritagli (di default puntate/piatto e stack)")
    args = parser.parse_args(argv)
    crops = [crop for folder in args.crops for crop in load_amounts(folder)]
    cards = card_digit_examples()
    own = build_reader(crops, fallback=[])
    reader = build_reader(crops, fallback=cards)
    print(f"cifre dalle puntate: {' '.join(own.covered) or '-'}; "
          f"dalle carte: {' '.join(d for d in reader.covered if d not in own.covered) or '-'}; "
          f"mancano: {' '.join(d for d in DIGITS if d not in reader.covered) or '-'}")
    right = 0
    for crop in crops:
        got = build_reader(crops, exclude=crop.path, fallback=cards).read(crop.image, crop.zone)
        text = None if got is None else got.text
        if text == crop.text:
            right += 1
        else:
            print(f"  {crop.path.name}: etichetta {crop.text!r}, letto {text!r}")
    print(f"ritagli giusti: {right}/{len(crops)}")
    print(json.dumps({"crops": len(crops), "right": right}))


if __name__ == "__main__":
    main()
