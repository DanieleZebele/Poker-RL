"""What a saved crop actually shows: the ground truth the recognition learns from.

Every PNG under `vision_data/crops/` (`CROPS_DIR`) has a JSON label next to it with
the same name (`hole_cards-20261001-101500.png` -> `...-101500.json`), saying
which cards are visible in it, **in order from left to right** as they appear on
screen. An empty list is a real answer -- "no card visible here" (folded, between
hands, a street not dealt yet) -- and is as useful an example as a full one. The
GUI writes the PNG and its label together, on confirmation, so a crop without a
label is one written some other way (or before labelling existed).

Pure Python, no dependency, like `regions.py`: cards are kept as the two-character
strings the GUI prints (`"Ah"`, `"Td"`), so the files read and diff by eye.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pokerlab.vision.regions import BOARD, DEFAULT_REGIONS_PATH, HOLE_CARDS

LABEL_VERSION = 1

# Where the labelled crops live: outside `checkpoints/` on purpose, which holds
# training state; these are a dataset of their own. Next to `regions.json`.
CROPS_DIR = DEFAULT_REGIONS_PATH.parent / "crops"
# The dealer-button crops: a dataset of their own, labelled present / absent.
DEALER_DIR = DEFAULT_REGIONS_PATH.parent / "dealer"
# The player-box crops, labelled with the seat's state.
PLAYERS_DIR = DEFAULT_REGIONS_PATH.parent / "players"
SEAT_IN_HAND = "in_gioco"  # dealt in and not folded
SEAT_OUT = "fuori"  # seated but not in this hand: folded, or just joined and waiting
SEAT_SIT_OUT = "sit_out"  # seated but sitting out: not dealt in until they come back
SEAT_EMPTY = "libero"  # nobody sits there
SEAT_STATES = (SEAT_IN_HAND, SEAT_OUT, SEAT_SIT_OUT, SEAT_EMPTY)
# The turn-timer crops, labelled bar present (your turn) / absent.
TURN_DIR = DEFAULT_REGIONS_PATH.parent / "turn"
# The bet and pot crops, labelled with the amount as written on screen.
AMOUNTS_DIR = DEFAULT_REGIONS_PATH.parent / "amounts"
# The stack crops: same label shape as the amounts, a folder of their own.
STACKS_DIR = DEFAULT_REGIONS_PATH.parent / "stacks"
AMOUNT_PATTERN = re.compile(r"[0-9][0-9.,]*\s?[kKmM]?")

# How many cards a zone can legitimately show. Two hole cards or none; a board
# is dealt a flop at a time, so 1 or 2 board cards would be a mislabel.
VALID_COUNTS: dict[str, tuple[int, ...]] = {
    HOLE_CARDS: (0, 2),
    BOARD: (0, 3, 4, 5),
}
# The slots the labelling window offers for each zone.
SLOTS: dict[str, int] = {HOLE_CARDS: 2, BOARD: 5}

_RANKS = "23456789TJQKA"
_SUITS = "shdc"


def label_path(png_path: Path) -> Path:
    """The label file that goes with a crop."""
    return Path(png_path).with_suffix(".json")


def label_error(zone: str, cards: list[str]) -> str | None:
    """Why `cards` is not a valid label for `zone`, or None if it is."""
    if zone not in VALID_COUNTS:
        return f"zona sconosciuta: {zone!r}"
    if not isinstance(cards, list):
        return "le carte devono essere una lista"
    for card in cards:
        if not isinstance(card, str) or len(card) != 2 or card[0] not in _RANKS or card[1] not in _SUITS:
            return f"carta non valida: {card!r}"
    if len(set(cards)) != len(cards):
        return "la stessa carta compare due volte"
    allowed = VALID_COUNTS[zone]
    if len(cards) not in allowed:
        return "servono " + " o ".join(str(n) for n in allowed) + f" carte, non {len(cards)}"
    return None


def save_label(png_path: Path, zone: str, cards: list[str]) -> Path:
    """Write the label for a crop, refusing an invalid one. Atomic, like
    `save_regions`."""
    error = label_error(zone, cards)
    if error is not None:
        raise ValueError(error)
    path = label_path(png_path)
    payload = {
        "version": LABEL_VERSION,
        "image": Path(png_path).name,
        "zone": zone,
        "cards": list(cards),  # left to right, as on screen
    }
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    staging.replace(path)
    return path


def save_dealer_label(png_path: Path, zone: str, present: bool) -> Path:
    """Label a dealer-zone crop: is the button there or not. Kept apart from the
    card labels (no `cards` key, a folder of its own) so neither dataset can be
    mistaken for the other: `load_label` rejects these, `load_dealer_label`
    rejects card labels."""
    if not zone.startswith("dealer_"):
        raise ValueError(f"non è una zona dealer: {zone!r}")
    path = label_path(png_path)
    payload = {"version": LABEL_VERSION, "image": Path(png_path).name, "zone": zone, "dealer": bool(present)}
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    staging.replace(path)
    return path


def load_dealer_label(png_path: Path) -> dict | None:
    """The label of a dealer crop, or None if it has none or it is not one."""
    try:
        raw = json.loads(label_path(png_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(raw, dict) or not str(raw.get("zone", "")).startswith("dealer_")
            or not isinstance(raw.get("dealer"), bool)):
        return None
    return raw


def save_player_label(png_path: Path, zone: str, state: str) -> Path:
    """Label a player-zone crop with the seat's state (`SEAT_STATES`). A folder
    and a label shape (`state`) of its own, like the dealer's."""
    if not zone.startswith("player_"):
        raise ValueError(f"non è una zona giocatore: {zone!r}")
    if state not in SEAT_STATES:
        raise ValueError(f"stato sconosciuto: {state!r}")
    path = label_path(png_path)
    payload = {"version": LABEL_VERSION, "image": Path(png_path).name, "zone": zone, "state": state}
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    staging.replace(path)
    return path


def load_player_label(png_path: Path) -> dict | None:
    """The label of a player crop, or None if it has none or it is not one."""
    try:
        raw = json.loads(label_path(png_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(raw, dict) or not str(raw.get("zone", "")).startswith("player_")
            or raw.get("state") not in SEAT_STATES):
        return None
    return raw


def amount_error(text: str) -> str | None:
    """Why `text` is not a valid amount label, or None. Empty means "no amount
    shown"; otherwise digits plus the separators and suffixes clients use."""
    if not isinstance(text, str):
        return "il valore deve essere un testo"
    if text and not AMOUNT_PATTERN.fullmatch(text):
        return f"valore non valido: {text!r} (solo cifre, punto, virgola, K, M)"
    return None


def save_amount_label(png_path: Path, zone: str, text: str) -> Path:
    """Label a bet or pot crop with the amount written in it, *as shown on
    screen* (separators and suffixes kept, "" for none): how the client writes
    numbers is exactly what the reader has to learn. Own folder, own shape."""
    if not (zone == "pot" or zone.startswith(("bet_", "stack_"))):
        raise ValueError(f"non è una zona di puntata, piatto o stack: {zone!r}")
    text = text.strip()
    error = amount_error(text)
    if error is not None:
        raise ValueError(error)
    path = label_path(png_path)
    payload = {"version": LABEL_VERSION, "image": Path(png_path).name, "zone": zone, "text": text}
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    staging.replace(path)
    return path


def load_amount_label(png_path: Path) -> dict | None:
    """The label of a bet/pot crop, or None if it has none or it is not one."""
    try:
        raw = json.loads(label_path(png_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    zone = str(raw.get("zone", "")) if isinstance(raw, dict) else ""
    if not (zone == "pot" or zone.startswith(("bet_", "stack_"))) or amount_error(raw.get("text")):
        return None
    return raw


def save_turn_label(png_path: Path, zone: str, present: bool) -> Path:
    """Label a turn-timer crop: is the countdown bar (your turn) showing."""
    if zone != "turn_timer":
        raise ValueError(f"non è la zona della barra del tempo: {zone!r}")
    path = label_path(png_path)
    payload = {"version": LABEL_VERSION, "image": Path(png_path).name, "zone": zone, "turn": bool(present)}
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    staging.replace(path)
    return path


def load_turn_label(png_path: Path) -> dict | None:
    """The label of a turn-timer crop, or None if it has none or it is not one."""
    try:
        raw = json.loads(label_path(png_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("zone") != "turn_timer" or not isinstance(raw.get("turn"), bool):
        return None
    return raw


def load_label(png_path: Path) -> dict | None:
    """The label of a crop, or None if it has none or it does not parse."""
    try:
        raw = json.loads(label_path(png_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or label_error(raw.get("zone", ""), raw.get("cards")):
        return None
    return raw
