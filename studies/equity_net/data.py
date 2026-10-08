"""Fresh deals as tensors, drawn continuously by worker processes, and two fixed sets the
network is measured on (never trained on), labelled with an accurate equity: a stratified
sample of ordinary deals and a set of hard cases."""

from __future__ import annotations

import random
from collections.abc import Iterator
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from deals import MAX_PLAYERS, MIN_PLAYERS, STREETS, Deal, card_from_index, sample_deal
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from pokerlab.evaluator.evaluator import HandCategory, evaluate

CARDS = 52
CARD_OBJECTS = [card_from_index(c) for c in range(CARDS)]
# The fixed sets the network is measured on. The file name carries their size, so a set made
# with other numbers is never mistaken for this one.
#
# Ordinary deals: DEFAULT_VAL_DEALS split equally over the 32 cells (2-9 players x 4
# streets). With the board still to come the label is *exact* from the flop on (every
# completion is enumerated: at most 990 on the flop, 44 on the turn, one on the river); only
# the preflop, with 1.7 million boards, is sampled, from DEFAULT_VAL_RUNOUTS completions.
DEFAULT_VAL_DEALS = 24000
DEFAULT_VAL_RUNOUTS = 10000
# Hard cases: this many deals for each tag of `HARD_TAGS`.
DEFAULT_HARD_DEALS = 400


# The river is not evaluated: once the board is complete the winner is read off the hand
# evaluator (exact, ties included), so no network has to estimate it. The files still hold the
# river deals; every reader below gets the sets without them. `visible` is an index into
# `STREETS`, and the river is the last one.
EVALUATED_STREETS = 3  # preflop, flop, turn


def without_river(tensors: tuple[torch.Tensor, ...], tags: np.ndarray | None = None):
    """The same arrays with the river deals taken out (and the tags of the kept ones, if given)."""
    keep = tensors[4] < EVALUATED_STREETS  # `tensors[4]` is `visible`
    kept = tuple(t[keep] for t in tensors)
    return kept if tags is None else (kept, tags[keep.numpy()])


def validation_path(runs: Path, deals: int, runouts: int) -> Path:
    return runs / f"validation_exact_{deals}x{runouts}.npz"


def hard_path(runs: Path, per_tag: int) -> Path:
    return runs / f"hard_{per_tag}.npz"


def encode(deals: list[Deal]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(holes, board, mask, shares)`: (B, 9, 52), (B, 52), (B, 9) bool and (B, 9)."""
    count = len(deals)
    holes = np.zeros((count, MAX_PLAYERS, CARDS), np.float32)
    board = np.zeros((count, CARDS), np.float32)
    mask = np.zeros((count, MAX_PLAYERS), bool)
    shares = np.zeros((count, MAX_PLAYERS), np.float32)
    for b, deal in enumerate(deals):
        for p, hole in enumerate(deal.holes):
            holes[b, p, list(hole)] = 1.0
        board[b, list(deal.known_board)] = 1.0
        mask[b, : deal.players] = True
        shares[b, : deal.players] = deal.shares
    return holes, board, mask, shares


class DealStream(IterableDataset):
    """An endless stream of batches of new deals; each worker has its own seed."""

    def __init__(self, batch_size: int, seed: int, street_weights: tuple[float, ...]) -> None:
        self.batch_size, self.seed, self.street_weights = batch_size, seed, street_weights

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        info = get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        while True:
            batch = [sample_deal(rng, self.street_weights) for _ in range(self.batch_size)]
            yield tuple(torch.from_numpy(a) for a in encode(batch))


def stream(batch_size: int, workers: int, seed: int, street_weights: tuple[float, ...]) -> DataLoader:
    return DataLoader(
        DealStream(batch_size, seed, street_weights),
        batch_size=None,
        num_workers=workers,
        prefetch_factor=4 if workers else None,
    )


# ---- fixed sets with an accurate equity ------------------------------------------------

PREFLOP_CHUNK = 20  # deals per task: a preflop deal costs seconds, the others fractions of one
OTHER_CHUNK = 100


def accurate_equity(deal: Deal, runouts: int, rng: random.Random) -> tuple[float, ...]:
    """Expected pot share of each player. Exact when the flop is out (every completion of
    the board is counted, a tie splitting the pot); preflop, the mean of `runouts` random
    completions."""
    used = {c for hole in deal.holes for c in hole} | set(deal.known_board)
    rest = [c for c in range(52) if c not in used]
    need = 5 - deal.visible
    known = [CARD_OBJECTS[c] for c in deal.known_board]
    cards = [[CARD_OBJECTS[c] for c in hole] for hole in deal.holes]
    if deal.visible:
        completions = list(combinations(rest, need))
    else:
        completions = [rng.sample(rest, need) for _ in range(runouts)]
    totals = [0.0] * deal.players
    for completion in completions:
        full = known + [CARD_OBJECTS[c] for c in completion]
        ranks = [evaluate(hole + full) for hole in cards]
        best = max(ranks)
        winners = sum(1 for rank in ranks if rank == best)
        for i, rank in enumerate(ranks):
            if rank == best:
                totals[i] += 1.0 / winners
    return tuple(t / len(completions) for t in totals)


def pack(pairs: list[tuple[Deal, tuple[float, ...]]]):
    """`(holes, board, mask, equity, visible)` as tensors, from deals with their equity."""
    holes, board, mask, _ = encode([deal for deal, _ in pairs])
    equity = np.zeros((len(pairs), MAX_PLAYERS), np.float32)
    for b, (deal, eq) in enumerate(pairs):
        equity[b, : deal.players] = eq
    visible = np.array([STREETS.index(deal.visible) for deal, _ in pairs], np.int64)
    return holes, board, mask, equity, visible


def _cell_task(args: tuple[int, int, int, int, int]) -> list[tuple[Deal, tuple[float, ...]]]:
    seed, players, visible, count, runouts = args
    rng = random.Random(seed)
    weights = tuple(1.0 if s == visible else 0.0 for s in STREETS)
    out = []
    for _ in range(count):
        deal = sample_deal(rng, weights, players, players)
        out.append((deal, accurate_equity(deal, runouts, rng)))
    return out


def validation_set(path: Path, count: int, runouts: int, jobs: int):
    """About `count` ordinary deals, the same number in each cell of players x street, with
    their accurate equity, cached in `path`. `runouts` only matters preflop. The river deals are
    left out (see `without_river`)."""
    if path.exists():
        z = np.load(path)
        return without_river(tuple(torch.from_numpy(z[k]) for k in ("holes", "board", "mask", "equity", "visible")))
    per_cell = max(1, count // (len(STREETS) * (MAX_PLAYERS - MIN_PLAYERS + 1)))
    tasks = []
    for visible in STREETS:  # preflop first: the long tasks should not be left to the end
        chunk = PREFLOP_CHUNK if visible == 0 else OTHER_CHUNK
        for players in range(MIN_PLAYERS, MAX_PLAYERS + 1):
            for start in range(0, per_cell, chunk):
                tasks.append((10_000 + len(tasks), players, visible, min(chunk, per_cell - start), runouts))
    with Pool(jobs) as pool:
        chunks = list(pool.imap(_cell_task, tasks, chunksize=1))  # in task order: the file is reproducible
    arrays = pack([pair for chunk in chunks for pair in chunk])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, holes=arrays[0], board=arrays[1], mask=arrays[2], equity=arrays[3], visible=arrays[4])
    return without_river(tuple(torch.from_numpy(a) for a in arrays))


# ---- hard cases -----------------------------------------------------------------------
#
# Where the network goes wrong is not where the ordinary deals are: two full houses with the
# same trips, two flushes, a pair on the board. Each tag below is a kind of deal that is
# found by drawing random deals and keeping the ones that fit, so the share of each kind is
# the same however rare it is among ordinary deals. They are all flop or later, so every
# label is exact.

HARD_TAGS = (
    "full contro full",  # a paired board (no trips) and two players with a full house or better
    "scale e colori",  # two players with a straight or better
    "stessa categoria",  # the two best hands are of one category (two pair or better): a kicker decides
    "board accoppiato",  # a paired board (no trips) and two players with two pair or better
    "equity contesa",  # the second player has at least a quarter of the pot
)
_STREET_CHOICES = (3, 4, 5)
MAX_TRIES_PER_DEAL = 5000  # a tag found too rarely stops here instead of running for ever


def _made_categories(deal: Deal) -> list[HandCategory]:
    known = [CARD_OBJECTS[c] for c in deal.known_board]
    return [evaluate([CARD_OBJECTS[c] for c in hole] + known).category for hole in deal.holes]


def _paired(deal: Deal) -> bool:
    """A pair or two on the visible board, but no trips: with trips on the board everyone
    plays it and there is little to tell between the hands."""
    ranks = [c % 13 for c in deal.known_board]
    return max(ranks.count(r) for r in ranks) == 2


def _fits(tag: int, deal: Deal, rng: random.Random) -> tuple[float, ...] | None:
    """The deal's equity if it is a case of `HARD_TAGS[tag]`, else None."""
    if tag == 4:  # needs the equity itself
        equity = accurate_equity(deal, 0, rng)
        return equity if sorted(equity)[-2] >= 0.25 else None
    categories = sorted(_made_categories(deal), reverse=True)
    if tag == 0:
        ok = categories[1] >= HandCategory.FULL_HOUSE
    elif tag == 1:
        ok = categories[1] >= HandCategory.STRAIGHT
    elif tag == 2:
        ok = categories[0] == categories[1] >= HandCategory.TWO_PAIR
    else:
        ok = _paired(deal) and categories[1] >= HandCategory.TWO_PAIR
    return accurate_equity(deal, 0, rng) if ok else None


def _hard_task(args: tuple[int, int, int]) -> list[tuple[int, Deal, tuple[float, ...]]]:
    seed, tag, count = args
    rng = random.Random(seed)
    # The rare tags need a crowded table and a pair on the board to be found at all.
    players = (4, MAX_PLAYERS) if tag in (0, 1, 2, 3) else (MIN_PLAYERS, MAX_PLAYERS)
    found, tries = [], 0
    while len(found) < count and tries < MAX_TRIES_PER_DEAL * count:
        tries += 1
        visible = rng.choice(_STREET_CHOICES)
        weights = tuple(1.0 if s == visible else 0.0 for s in STREETS)
        deal = sample_deal(rng, weights, *players)
        if tag in (0, 3) and not _paired(deal):
            continue
        equity = _fits(tag, deal, rng)
        if equity is not None:
            found.append((tag, deal, equity))
    return found


def hard_set(path: Path, per_tag: int, jobs: int):
    """`per_tag` deals of each kind in `HARD_TAGS` (fewer if a kind is too rare to find), with
    their exact equity: `((holes, board, mask, equity, visible), tags)`, cached in `path`. The
    river deals are left out (see `without_river`)."""
    if path.exists():
        z = np.load(path)
        return without_river(
            tuple(torch.from_numpy(z[k]) for k in ("holes", "board", "mask", "equity", "visible")), z["tags"])
    share = max(1, per_tag // jobs)
    tasks = []
    for tag in range(len(HARD_TAGS)):
        for _ in range(-(-per_tag // share)):
            tasks.append((20_000 + len(tasks), tag, share))
    with Pool(jobs) as pool:
        chunks = list(pool.imap(_hard_task, tasks, chunksize=1))
    found = [item for chunk in chunks for item in chunk]
    kept, seen = [], [0] * len(HARD_TAGS)
    for tag, deal, equity in found:  # at most `per_tag` of each: the last chunk can overshoot
        if seen[tag] < per_tag:
            seen[tag] += 1
            kept.append((tag, deal, equity))
    arrays = pack([(deal, equity) for _, deal, equity in kept])
    tags = np.array([tag for tag, _, _ in kept], np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, holes=arrays[0], board=arrays[1], mask=arrays[2], equity=arrays[3], visible=arrays[4], tags=tags)
    return without_river(tuple(torch.from_numpy(a) for a in arrays), tags)
