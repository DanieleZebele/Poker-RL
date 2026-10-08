from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from pokerlab.engine.config import GameConfig
from pokerlab.players.base import Player
from pokerlab.rl.global_arena import (
    DEFAULT_GLOBAL_DIR,
    discover_benchmark_population,
    discover_population,
)
from pokerlab.rl.global_store import load_ranking

# A trained checkpoint as an opponent, given by path. Every non-human seat is a
# `model:<path>` spec, so a bad path only ever fails for whoever asked for that
# seat, never for anyone else at the table.
MODEL_PREFIX = "model:"

DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")
# Where `rl/push_top_models.py` publishes the best models into git; the fallback
# when no checkpoint store is reachable (same names as there, kept in step by a test).
TOP_MODELS_DIR = Path("top_models")
TOP_MODELS_RATINGS = "ratings.json"
DEFAULT_RATING = 1500.0


def discover_trained_models(
    root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    *,
    limit: int = 20,
    fallback_dir: str | Path | None = TOP_MODELS_DIR,
) -> list[tuple[str, Path, float]]:
    """The best trained checkpoints available, `(label, path, rating)`, best first.

    Read from the global ranking under `<root>/global`: every model lives in the
    one shared store and has one rating on one scale, so "the best" means the
    same thing whichever machine you are sitting at. Pure Python -- the ranking
    code carries no torch -- so the GUI can list the models on a machine where
    the `rl` extra was never installed, and only fails if someone picks one.
    """
    return discover_global_top_models(Path(root) / "global", limit=limit, fallback_dir=fallback_dir)


def discover_top_models_folder(
    folder: str | Path = TOP_MODELS_DIR, *, limit: int = 6
) -> list[tuple[str, Path, float]]:
    """The models in the git-tracked `top_models/` folder, best first.

    `rl/push_top_models.py` copies the fleet's current best there as `<label>.pt`
    plus a `ratings.json`, so a machine that has the repository but not the
    training volume -- the Windows PC the GUI runs on -- still has models to
    seat. A `.pt` missing from `ratings.json` is still offered, after the rated
    ones, at the default rating; a rating whose file is gone is skipped.
    """
    folder = Path(folder)
    try:
        ratings = json.loads((folder / TOP_MODELS_RATINGS).read_text(encoding="utf-8"))["ratings"]
        if not isinstance(ratings, dict):
            ratings = {}
    except (OSError, ValueError, KeyError, TypeError):
        ratings = {}
    found = []
    for path in folder.glob("*.pt"):
        try:
            rating = float(ratings.get(path.stem, DEFAULT_RATING))
        except (TypeError, ValueError):
            rating = DEFAULT_RATING
        found.append((path.stem, path, rating, path.stem in ratings))
    found.sort(key=lambda item: (not item[3], -item[2], item[0]))
    return [(label, path, rating) for label, path, rating, _rated in found[:limit]]


def discover_global_top_models(
    global_dir: str | Path = DEFAULT_GLOBAL_DIR,
    *,
    limit: int = 6,
    fallback_dir: str | Path | None = TOP_MODELS_DIR,
) -> list[tuple[str, Path, float]]:
    """The top `limit` models by rating in the cross-machine global registry.

    Unlike `discover_trained_models`, which approximates "best" from each
    machine's own local pool rating (a scale that is not comparable across
    machines), this reads the one shared Elo scale that `rl/global_arena.py` maintains, so
    "the 6 best" actually means something across the whole fleet.

    Read from the `registry.json` snapshot (a few minutes behind at worst), or
    from the per-model files when no snapshot exists yet. A member's `ref` is
    kept current by the rounds, but if one is stale anyway (the file was moved
    since) the checkpoint is looked up by label across the volume instead of
    being skipped, so a top model is never dropped just because it changed
    directory. A model with no file anywhere is skipped, so one missing file
    cannot break the whole list.

    **When nothing is found that way** -- no `checkpoints/` at all, as on a PC
    that only cloned the repository -- the models come from `fallback_dir`
    (`top_models/`, see `discover_top_models_folder`). `None` turns that off,
    which `push_top_models` needs: it must never "find" the very copies it is
    about to replace.
    """
    found = _registry_top_models(global_dir, limit)
    if not found and fallback_dir is not None:
        found = discover_top_models_folder(fallback_dir, limit=limit)
    return found


def _registry_top_models(global_dir: str | Path, limit: int) -> list[tuple[str, Path, float]]:
    registry = load_ranking(global_dir)
    on_disk: dict[str, Path] | None = None
    found: list[tuple[str, Path, float]] = []
    for member in registry.ranked():
        if len(found) >= limit:
            break
        path = Path(member.ref)
        if not path.is_file():
            if on_disk is None:
                root = Path(global_dir).parent
                on_disk = {
                    c.label: c.path
                    for c in [*discover_population(root), *discover_benchmark_population(root)]
                }
            path = on_disk.get(member.label, path)
            if not path.is_file():
                continue
        found.append((member.label, path, member.rating))
    return found


def make_model_bot(path: str | Path, player_id: str, name: str, game: GameConfig) -> Player:
    """Seat a trained checkpoint as an opponent.

    torch is imported here and nowhere else in this module, so the GUI keeps
    working without the `rl` extra installed for as long as nobody actually
    asks for a model. The policy samples rather than taking the argmax,
    which is how the agent played while it was being trained.
    """
    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint

    model, _checkpoint = build_model_from_checkpoint(path)
    return RLAgentPlayer(
        player_id,
        name,
        policy_fn=make_policy_fn(model),
        big_blind=game.big_blind,
        starting_stack=game.starting_stack,
    )


def build_players(
    num_players: int,
    human_seats: int,
    rng: random.Random,
    bot_keys: list[str] | None = None,
    human_player_factory: Callable[[str, str], Player] | None = None,
    game: GameConfig | None = None,
) -> list[Player]:
    """Build the seat list for a session. The first `human_seats` seats are built by
    `human_player_factory` (the GUI passes one that builds a GuiPlayer), which is
    therefore required whenever there is a human seat.

    With no `bot_keys`, the non-human seats cycle through the best-rated
    trained models found across every machine's pool (see
    `discover_trained_models`) -- there is nothing else to fall back to. Raises a clear `ValueError` if there are seats to fill
    and no trained model can be found for them, rather than silently seating
    nothing.
    """
    if human_seats and human_player_factory is None:
        raise ValueError("seating a human needs a human_player_factory")
    if bot_keys:
        keys = bot_keys
    else:
        models = discover_trained_models()
        if not models and num_players > human_seats:
            raise ValueError(
                "no trained models found under checkpoints/ to fill the non-human "
                "seats, and no --bots spec was given; train a model first "
                "(see poker-train) or pass model:<path> explicitly"
            )
        keys = [f"{MODEL_PREFIX}{path}" for _label, path, _rating in models]
    players: list[Player] = []
    for seat in range(num_players):
        player_id = f"p{seat}"
        if seat < human_seats:
            assert human_player_factory is not None
            players.append(human_player_factory(player_id, f"Human{seat}"))
        else:
            key = keys[(seat - human_seats) % len(keys)]
            if game is None:
                raise ValueError(
                    "seating a trained model needs the table's GameConfig: "
                    "RLAgentPlayer normalises its features by big_blind and "
                    "starting_stack, which an Observation does not carry"
                )
            path = key[len(MODEL_PREFIX) :]
            players.append(
                make_model_bot(path, player_id, f"{Path(path).stem[:18]}{seat}", game)
            )
    return players


def validate_bot_key(key: str) -> None:
    """Raise ValueError with a clear message if `key` isn't a usable bot
    spec -- a 'model:<path>' pointing at an existing checkpoint. Used by the
    GUI's form validation."""
    if not key.startswith(MODEL_PREFIX):
        raise ValueError(
            f"unknown bot spec {key!r}; only 'model:<path>' is supported"
        )
    path = Path(key[len(MODEL_PREFIX) :])
    if not path.is_file():
        raise ValueError(f"no checkpoint at {path}")
