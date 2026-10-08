"""Who a training run plays against: a fresh, diverse draw from the shared store.

Every `poker-train` run -- so every worker of every generation -- draws its own
opponents from `checkpoints/models/` instead of inheriting a pool someone else
ranked. That is the point: workers must not all train against the same twenty
models, or the population converges on beating one fixed field. The draw mixes
two sources:

  * **top**: `top_share` of the seats, uniformly from the `top_n` best-rated
    models, so there is always a strong field to learn from;
  * **random**: every remaining seat, uniformly from the whole store, which is
    what keeps weak and odd opponents (never-rated newcomers included) in the
    mix and stops the field being only the current elite.

The random source is also what makes a short top source harmless: with nothing
rated yet it has nobody to draw, and the pool still comes out full.

Pure Python -- no torch -- and given a seeded `random.Random` the draw is
reproducible.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from pokerlab.rl.pool_registry import (
    DEFAULT_POOL_SIZE,
    PoolMember,
    PoolRegistry,
)

DEFAULT_TOP_SHARE = 0.5
DEFAULT_TOP_N = 100


def available_labels(models_dir: str | Path) -> list[str]:
    """Every model in the shared store (a label is the file name without `.pt`)."""
    directory = Path(models_dir)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.pt") if not p.name.startswith("."))


def _known(label: str, ranking: Mapping[str, PoolMember]) -> PoolMember:
    """The ranking's view of a model, or a blank never-rated one if it has none."""
    member = ranking.get(label)
    if member is None:
        return PoolMember(label=label, ref=f"{label}.pt")
    return member


def draw_training_pool(
    ranking: Mapping[str, PoolMember],
    labels: Sequence[str],
    *,
    size: int = DEFAULT_POOL_SIZE,
    rng: random.Random,
    top_share: float = DEFAULT_TOP_SHARE,
    top_n: int = DEFAULT_TOP_N,
) -> list[PoolMember]:
    """Pick up to `size` opponents from `labels`.

    `top_share` of the seats come from the `top_n` best-rated models and every
    remaining seat is drawn uniformly from the whole store. The returned members
    are *copies*, frozen and with `ref` set to the file name inside the models
    directory: they are reference points for scoring one run's learner and are
    never written back to the global ranking.
    """
    if size <= 0 or not labels:
        return []
    members = [_known(label, ranking) for label in labels]
    by_label = {member.label: member for member in members}

    rated = sorted((m for m in members if m.games > 0), key=lambda m: (-m.rating, m.label))
    top_pool = [m.label for m in rated[:top_n]]
    everything = [m.label for m in members]

    chosen: list[str] = []
    taken: set[str] = set()

    def take(pool: Sequence[str], count: int) -> None:
        candidates = [label for label in pool if label not in taken]
        for label in rng.sample(candidates, min(count, len(candidates))):
            chosen.append(label)
            taken.add(label)

    take(top_pool, round(size * top_share))
    # Every remaining seat, which is also what a short top source falls back on.
    take(everything, size - len(chosen))

    return [
        PoolMember(
            label=label,
            ref=f"{label}.pt",
            rating=by_label[label].rating,
            games=by_label[label].games,
            frozen=True,
        )
        for label in chosen
    ]


def build_training_registry(models_dir: str | Path, drawn: Sequence[PoolMember]) -> PoolRegistry:
    """An in-memory registry over the drawn opponents, for scoring a learner.

    It is never saved: the members are frozen copies, so evaluating a learner
    against them cannot move anyone's rating, and nothing here reaches disk.
    """
    registry = PoolRegistry(directory=Path(models_dir), max_models=max(len(drawn), 1))
    for member in drawn:
        registry.members[member.label] = member
    return registry


# Where a parent comes from: each slot picks one of these bands with equal
# probability, then a model uniformly inside it. `None` is the whole store.
PARENT_TIERS: tuple[int | None, ...] = (10, 100, 1000, None)


def format_parent_tiers(tiers: Sequence[int | None]) -> str:
    """`tiers` as the `--parent-tiers` flag and `config.toml` write them:
    `"10, 100, 1000, all"`."""
    return ", ".join("all" if band is None else str(band) for band in tiers)


def parse_parent_tiers(text: str) -> tuple[int | None, ...]:
    """The inverse of `format_parent_tiers`; `ValueError` says what is wrong.

    Each entry is a band size (the best N rated models) or `all`. A band listed
    twice is drawn twice as often, which is how the draw is weighted.
    """
    tiers: list[int | None] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "all":
            tiers.append(None)
            continue
        try:
            band = int(part)
        except ValueError:
            raise ValueError(f"'{part}' is neither a number nor 'all'") from None
        if band < 1:
            raise ValueError(f"a band must hold at least one model, not {band}")
        tiers.append(band)
    if not tiers:
        raise ValueError("at least one band is needed")
    return tuple(tiers)


def parent_tiers_text(text: str) -> str:
    """argparse `type=` for the parent bands: validates, returns canonical text."""
    try:
        return format_parent_tiers(parse_parent_tiers(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def pick_parents(
    ranking: Mapping[str, PoolMember],
    labels: Sequence[str],
    count: int,
    *,
    rng: random.Random,
    tiers: Sequence[int | None] = PARENT_TIERS,
) -> list[str]:
    """`count` distinct parents to inherit weights from; cycling only if there are
    fewer rated models than `count`.

    Every parent first draws a band of the ranking (the top 10, 100, 1000 or all
    rated models, equally likely) and then a model uniformly inside it, so the
    top band is deep but the lineage is not confined to it. Distinct so a
    generation is competing lineages rather than copies of
    one model; a band already exhausted by earlier picks falls back to the
    unused models of the whole ranking.
    """
    if count <= 0:
        return []
    rated = sorted(
        (m for m in (_known(label, ranking) for label in labels) if m.games > 0 and not m.frozen),
        key=lambda m: (-m.rating, m.label),
    )
    ranked = [m.label for m in rated]
    if not ranked or not tiers:
        return []
    picked: list[str] = []
    taken: set[str] = set()
    for _ in range(min(count, len(ranked))):
        band = rng.choice(list(tiers))
        candidates = [x for x in ranked[:band] if x not in taken]
        if not candidates:
            candidates = [x for x in ranked if x not in taken]
        choice = rng.choice(candidates)
        picked.append(choice)
        taken.add(choice)
    return [picked[index % len(picked)] for index in range(count)]
