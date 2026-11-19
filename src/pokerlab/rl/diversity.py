"""How differently models play, and which ones the pruning can do without (`global_arena._eliminate`).

**A model's signature** is what it would do at a fixed set of decisions, the *probes*: for each,
the square root of its probability of every action bin (`ACTION_DIM`), so two models are as far
apart as their policies are. The distance between two models is the mean over the probes of the
Hellinger distance between their two distributions,

    d(a, b) = mean_probe ( 1 - sum_bin sqrt(p_a * p_b) )            in [0, 1],

0 for two models that play every probe alike. With the signatures as rows it is
`1 - <sig_a, sig_b> / probes`: a matrix product, so the distances of thousands of models are cheap.

- **The probes** are decisions of real hands (`Observation`, legal actions), drawn from sessions of
  the store's models at the training mixture with their statistics on (`studies/agents/
  diversity_study.py probes`), kept in one file for the whole fleet (`[pruning] diversity_probes`).
  A signature means something only against the probes it was taken on, so signatures are kept per
  probe file (`signature_dir`: its fingerprint names the directory). Every model reads a probe at
  its own feature version; one of version 5 with the table's memory empty (every player unseen).
- **Signatures are written once** per model (a published model never changes), as `.npy` files
  under the global directory, and computed where a model is at hand: the supervisor fills in the
  missing ones a few at a time (`backfill_signatures`), the study all at once.
- **Many models**: above `MAX_NUMBERS` the selection reads every model on the same fixed subset of
  the probes (`probe_subset`), so a worker never holds more than ~200 MB of signatures.
- **Who goes** (`redundant_models`): the pruning keeps the `protect_top` best-rated models whatever
  they play, and among the rest removes, one at a time, the model nearest to any other still
  standing (protected or not), so what is left is as spread out as the count allows; when the
  nearest two are both removable the lower-rated goes. A near-copy of a model in the top 100 is the
  first to go, a model unlike anything else the last.

Pure Python at import: numpy and torch are imported where they are used, so `global_arena` can
import this without either (the torch-free CLIs and the ordinary test suite).
"""

from __future__ import annotations

import hashlib
import pickle
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

SIGNATURES_DIRNAME = "signatures"
DEFAULT_PROBES = "diversity/probes-v2.pkl"
DEFAULT_PROTECT_TOP = 100
# Signatures the supervisor computes per generation (`backfill_signatures`): each is a model
# loaded and asked the probes, ~0.3 s.
BACKFILL_PER_GENERATION = 100


# ---- probes ------------------------------------------------------------------------------


def save_probes(probes: Sequence[tuple[Any, list]], path: str | Path) -> None:
    """`(Observation, legal actions)` pairs, the table's memory left out (it is the table's)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [(replace(observation, seat_memory=None), list(legal)) for observation, legal in probes]
    partial = path.with_name("." + path.name + ".partial")
    partial.write_bytes(pickle.dumps(cleaned, protocol=pickle.HIGHEST_PROTOCOL))
    partial.rename(path)


def load_probes(path: str | Path) -> list[tuple[Any, list]]:
    return pickle.loads(Path(path).read_bytes())


def probes_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def signature_dir(global_dir: str | Path, probes_path: str | Path) -> Path:
    return Path(global_dir) / SIGNATURES_DIRNAME / probes_fingerprint(probes_path)


# ---- signatures (torch) ---------------------------------------------------------------------


def model_signature(model: Any, probes: Sequence[tuple[Any, list]], *, big_blind: int,
                    starting_stack: int, batch: int = 512):
    """The model's signature on `probes`: `(len(probes), ACTION_DIM)` float32, the square roots of
    its action probabilities, as it reads each probe at its own feature version."""
    import numpy as np
    import torch

    from pokerlab.rl.action_space import legal_action_mask
    from pokerlab.rl.features import encode_observation

    reader = model.belief_reader()
    rows, masks = [], []
    for observation, legal in probes:
        mask = legal_action_mask(observation, legal)
        beliefs = reader(observation, big_blind) if reader is not None else None
        rows.append(encode_observation(observation, big_blind=big_blind, starting_stack=starting_stack,
                                       legal_mask=mask, version=model.feature_version, beliefs=beliefs))
        masks.append(mask)
    found = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            logits = model.policy_logits(torch.tensor(rows[start : start + batch], dtype=torch.float32),
                                         torch.tensor(masks[start : start + batch], dtype=torch.bool))
            found.append(torch.softmax(logits, dim=-1).sqrt())
    return np.asarray(torch.cat(found).numpy(), dtype=np.float32)


def save_signature(directory: Path, label: str, signature) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / f".{label}.partial.npy"
    np.save(partial, signature.astype(np.float16))
    partial.rename(directory / f"{label}.npy")


def load_signatures(directory: Path, labels: Sequence[str], subset=None) -> dict[str, Any]:
    """The signatures of `labels` that exist, as float32, cut to the probes of `subset` if given
    (`probe_subset`), so only the subset is ever held."""
    import numpy as np

    found = {}
    for label in labels:
        path = directory / f"{label}.npy"
        if path.is_file():
            try:
                signature = np.load(path, mmap_mode="r")
                found[label] = np.asarray(signature if subset is None else signature[subset], dtype=np.float32)
            except (OSError, ValueError):
                continue  # half written by a reader that died: computed again next time
    return found


def backfill_signatures(
    directory: Path,
    probes_path: str | Path,
    candidates: Sequence[tuple[str, Path]],
    *,
    big_blind: int,
    starting_stack: int,
    limit: int = BACKFILL_PER_GENERATION,
    on_skip: Callable[[str, str], None] | None = None,
) -> int:
    """Compute and save the signatures `candidates` `(label, checkpoint)` lack, at most `limit`.
    A checkpoint that cannot be loaded is reported and skipped. Returns how many were written."""
    missing = [(label, path) for label, path in candidates if not (directory / f"{label}.npy").is_file()]
    if not missing or limit <= 0:
        return 0
    from pokerlab.rl.ppo import build_model_from_checkpoint

    probes = load_probes(probes_path)
    written = 0
    for label, path in missing[:limit]:
        try:
            model, _checkpoint = build_model_from_checkpoint(path, device="cpu")
        except Exception as error:  # noqa: BLE001 - a broken or foreign file is skipped, not fatal
            if on_skip is not None:
                on_skip(label, str(error))
            continue
        save_signature(directory, label, model_signature(model, probes, big_blind=big_blind,
                                                         starting_stack=starting_stack))
        written += 1
    return written


# ---- distances and selection (numpy) --------------------------------------------------------


def distance_matrix(signatures: Sequence[Any]):
    """`d(a, b)` for every pair of `signatures` (each `(probes, ACTION_DIM)`): `(n, n)` float32."""
    import numpy as np

    stacked = np.stack([s.reshape(-1) for s in signatures]).astype(np.float32)
    probes = signatures[0].shape[0]
    found = 1.0 - (stacked @ stacked.T) / probes
    np.fill_diagonal(found, 0.0)
    return np.clip(found, 0.0, 1.0)


# The most numbers (models x probes x bins) the selection holds at once: above it each model is
# read on a fixed random subset of the probes, as many as fit (10,000 models of 10,000 probes would
# otherwise be 4.4 GB of float32). The distance is a mean over probes, and on the first study two
# halves of 2,000 probes ranked the pairs alike (correlation 0.988), so a subset loses little.
MAX_NUMBERS = 50_000_000


def probe_subset(models: int, probes: int, bins: int, limit: int = MAX_NUMBERS):
    """The probes the selection reads when `models` signatures of `probes` x `bins` are too many:
    None (all of them) when they fit, else a fixed random subset, sorted."""
    import numpy as np

    keep = limit // max(models * bins, 1)
    if keep >= probes:
        return None
    return np.sort(np.random.default_rng(0).choice(probes, size=max(keep, 1), replace=False))


def redundant_models(
    labels: Sequence[str],
    signatures: Sequence[Any],
    *,
    removable: set[str],
    count: int,
    ratings: dict[str, float],
) -> list[str]:
    """Up to `count` of the `removable` labels, in the order they would go: each time, the
    removable model nearest to any other model still standing (protected or not); when that
    nearest model is removable too, the lower-rated of the two (then the label, for a fixed order).

    `labels[i]` has signature `signatures[i]` (`(probes, bins)`, or already cut to the probes of
    `probe_subset`); a label not in `removable` is kept whatever happens and only counts as a
    neighbour. No distance matrix is held: each model's nearest neighbour is found in blocks and
    found again only when its neighbour goes."""
    import numpy as np

    if count <= 0 or not removable or len(labels) < 2:
        return []
    probes = signatures[0].shape[0]
    stacked = np.stack([s.reshape(-1).astype(np.float32) for s in signatures])
    squares = (stacked * stacked).sum(axis=1)
    n = len(labels)
    alive = np.ones(n, dtype=bool)
    can_go = np.array([label in removable for label in labels])

    def nearest_of(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # d = |a - b|^2 / (2 * probes) = 1 - <a, b> / probes (each probe's row has norm 1)
        found_index = np.empty(len(rows), dtype=np.int64)
        found_distance = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), 1024):
            block = rows[start : start + 1024]
            d = (squares[block, None] + squares[None, :] - 2 * stacked[block] @ stacked.T) / (2 * probes)
            d[np.arange(len(block)), block] = np.inf
            d[:, ~alive] = np.inf
            found_index[start : start + len(block)] = d.argmin(axis=1)
            found_distance[start : start + len(block)] = d.min(axis=1)
        return found_index, found_distance

    nearest, distance = nearest_of(np.arange(n))
    gone: list[str] = []

    def key(i: int) -> tuple[float, str]:
        return (ratings.get(labels[i], 0.0), labels[i])

    while len(gone) < count and (alive & can_go).any() and alive.sum() > 1:
        candidates = np.flatnonzero(alive & can_go)
        best = int(candidates[int(distance[candidates].argmin())])
        other = int(nearest[best])
        victim = other if can_go[other] and key(other) < key(best) else best
        alive[victim] = False
        gone.append(labels[victim])
        stale = np.flatnonzero(alive & (nearest == victim))
        if stale.size:
            nearest[stale], distance[stale] = nearest_of(stale)
    return gone


def probe_sample(probes: Sequence[Any], count: int, seed: int = 0) -> list[Any]:
    """`count` of `probes` (all if fewer), the same ones for a given seed."""
    if count >= len(probes):
        return list(probes)
    return random.Random(seed).sample(list(probes), count)
