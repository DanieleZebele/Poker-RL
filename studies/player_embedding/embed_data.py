"""The dataset of the player embedder, as arrays: packing, shards on disk, splits and showdown modes.

A **sequence** is one player over one session: `T` hands in order, each with up to
`MAX_DECISIONS` decisions. It is the unit the embedder reads (hands 0..t-1 in, an embedding for hand
t out), so the arrays are one row per sequence:

    x          (N, T, A, DECISION_DIM)  float16   what it faced and did (`embed_records`)
    cls        (N, T, A)                int8      the action class, -1 where there is no decision
    strength   (N, T, A)                float16   the strength of its own cards (a target, never an input)
    show       (N, T)                   int8      SHOW_NONE / SHOWED / MUCKED
    won        (N, T)                   int8
    shown      (N, T)                   float16   the strength of the cards it showed
    model      (N,)                     str       the model that played the seat
    p_show     (N,)                     float32   how often it showed a losing hand
    styled     (N,)                     int8      1 if a random style was pushed on it
    session    (N,)                     int32

numpy only (no torch), so the splits and the modes are tested in the ordinary suite.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from embed_records import (
    DECISION_DIM,
    MAX_DECISIONS,
    SHOW_MUCKED,
    SHOW_NONE,
    SHOW_SHOWED,
    SeatHand,
)

# What an observer can see of the showdowns, as a mode to train and evaluate under:
# `full` as generated (winners always show, losers by their own p_show); `winners` only the
# winners' hands are seen (every loser mucks); `none` no cards are ever seen.
MODES = ("full", "winners", "none")

ARRAY_KEYS = ("x", "cls", "strength", "show", "won", "shown", "model", "p_show", "styled", "session")


def pack(
    sequences: Sequence[Sequence[SeatHand]],
    *,
    models: Sequence[str],
    p_show: Sequence[float],
    styled: Sequence[bool],
    session: int,
    hands: int,
) -> dict[str, np.ndarray]:
    """The sequences of one session as arrays. Every sequence has `hands` hands."""
    count = len(sequences)
    x = np.zeros((count, hands, MAX_DECISIONS, DECISION_DIM), dtype=np.float16)
    cls = np.full((count, hands, MAX_DECISIONS), -1, dtype=np.int8)
    strength = np.zeros((count, hands, MAX_DECISIONS), dtype=np.float16)
    show = np.zeros((count, hands), dtype=np.int8)
    won = np.zeros((count, hands), dtype=np.int8)
    shown = np.zeros((count, hands), dtype=np.float16)
    for row, sequence in enumerate(sequences):
        if len(sequence) != hands:
            raise ValueError(f"a sequence has {len(sequence)} hands, expected {hands}")
        for t, seat_hand in enumerate(sequence):
            used = len(seat_hand.vectors)
            if used:
                x[row, t, :used] = np.asarray(seat_hand.vectors, dtype=np.float32)
                cls[row, t, :used] = seat_hand.classes
                strength[row, t, :used] = seat_hand.strengths
            show[row, t] = seat_hand.show
            won[row, t] = seat_hand.won
            shown[row, t] = seat_hand.shown_strength
    return {
        "x": x, "cls": cls, "strength": strength, "show": show, "won": won, "shown": shown,
        "model": np.asarray(list(models), dtype=object),
        "p_show": np.asarray(list(p_show), dtype=np.float32),
        "styled": np.asarray(list(styled), dtype=np.int8),
        "session": np.full(count, session, dtype=np.int32),
    }


def concatenate(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in ARRAY_KEYS}


def save_shard(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: arrays[key].astype(str) if key == "model" else arrays[key]
                                 for key in ARRAY_KEYS})


def load_shards(paths: Sequence[str | Path]) -> dict[str, np.ndarray]:
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            part = {key: data[key] for key in ARRAY_KEYS}
        part["model"] = part["model"].astype(object)
        parts.append(part)
    return concatenate(parts)


def holdout_mask(models: Sequence[str], fraction: float) -> np.ndarray:
    """True for the sequences of the models set aside for validation. By the model's name, so the
    same model is on the same side in every shard and a model never trains and validates."""
    cut = int(round(fraction * 1000))
    return np.asarray([zlib.crc32(str(label).encode()) % 1000 < cut for label in models], dtype=bool)


def apply_mode(
    show: np.ndarray, won: np.ndarray, shown: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """The showdown arrays as seen under `mode` (see `MODES`), copies: the data is not touched."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}: {MODES}")
    show = show.copy()
    shown = shown.copy()
    if mode == "winners":
        lost = (show == SHOW_SHOWED) & (won == 0)
        show[lost] = SHOW_MUCKED
        shown[lost] = 0
    elif mode == "none":
        show[:] = SHOW_NONE
        shown[:] = 0
    return show, shown
