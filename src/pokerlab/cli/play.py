from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path

from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import HandHistoryWriter
from pokerlab.engine.table import Table
from pokerlab.players.base import Player
from pokerlab.players.manual import ManualPlayer
from pokerlab.players.scripted import get_bot_profile, list_bot_profiles, make_heuristic_bot

CUSTOM_PREFIX = "custom:"

CUSTOM_PARAM_DEFAULTS = {
    "tightness": 0.3,
    "aggression": 0.5,
    "bluff_frequency": 0.1,
    "size_variance": 0.2,
}


def parse_custom_bot_spec(spec: str) -> dict[str, float]:
    """Parse a 'custom:tightness=0.2;aggression=0.8' style spec into a
    params dict for make_heuristic_bot, filling in any axis left unspecified
    with a moderate default. Raises ValueError on anything malformed."""
    body = spec[len(CUSTOM_PREFIX) :].strip()
    params = dict(CUSTOM_PARAM_DEFAULTS)
    if not body:
        return params
    for pair in body.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"invalid custom bot parameter {pair!r} in {spec!r}, expected key=value")
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if key not in CUSTOM_PARAM_DEFAULTS:
            valid = ", ".join(CUSTOM_PARAM_DEFAULTS)
            raise ValueError(f"unknown custom bot parameter {key!r} in {spec!r}; valid: {valid}")
        try:
            value = float(raw_value)
        except ValueError:
            raise ValueError(f"parameter {key!r} must be a number, got {raw_value!r} in {spec!r}") from None
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"parameter {key!r} must be between 0 and 1, got {value} in {spec!r}")
        params[key] = value
    return params


def build_players(
    num_players: int,
    human_seats: int,
    rng: random.Random,
    bot_keys: list[str] | None = None,
    human_player_factory: Callable[[str, str], Player] = ManualPlayer,
) -> list[Player]:
    """Build the seat list for a session. `human_player_factory` defaults to
    ManualPlayer (terminal input); the GUI passes a factory that builds a
    GuiPlayer instead, reusing all of this function's bot-cycling and
    custom-spec parsing for free."""
    keys = bot_keys if bot_keys else [profile.key for profile in list_bot_profiles()]
    players: list[Player] = []
    for seat in range(num_players):
        player_id = f"p{seat}"
        if seat < human_seats:
            players.append(human_player_factory(player_id, f"Human{seat}"))
        else:
            key = keys[(seat - human_seats) % len(keys)]
            bot_rng = random.Random(rng.random())
            if key.startswith(CUSTOM_PREFIX):
                params = parse_custom_bot_spec(key)
                players.append(make_heuristic_bot(player_id, f"Custom{seat}", rng=bot_rng, **params))
            else:
                profile = get_bot_profile(key)
                players.append(profile.factory(player_id, f"{profile.label}{seat}", bot_rng))
    return players


def print_bot_catalog() -> None:
    print("Available bots (difficulty: 1 = weakest/most predictable .. 5 = strongest):")
    for profile in list_bot_profiles():
        print(f"  [{profile.difficulty}] {profile.key:<16} {profile.label:<14} {profile.description}")
    print()
    print("Custom bot: 'custom:tightness=0.2;aggression=0.8;bluff_frequency=0.3;size_variance=0.5'")
    print(f"  any parameter you omit defaults to: {CUSTOM_PARAM_DEFAULTS}")


def validate_bot_key(key: str) -> None:
    """Raise ValueError with a clear message if `key` isn't a usable bot
    spec -- either a BOT_CATALOG key or a well-formed 'custom:...' spec.
    Shared by the CLI's argparse validation and the GUI's form validation."""
    if key.startswith(CUSTOM_PREFIX):
        parse_custom_bot_spec(key)  # raises ValueError on its own
    else:
        try:
            get_bot_profile(key)
        except KeyError as e:
            raise ValueError(str(e)) from None


def _validate_bot_key(parser: argparse.ArgumentParser, key: str) -> None:
    try:
        validate_bot_key(key)
    except ValueError as e:
        parser.error(str(e))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play (or watch bots play) a No-Limit Hold'em session.")
    parser.add_argument("--players", type=int, default=6, help="number of seats (2-9)")
    parser.add_argument("--stack", type=int, default=200, help="starting stack per player")
    parser.add_argument("--sb", type=int, default=1, help="small blind")
    parser.add_argument("--bb", type=int, default=2, help="big blind")
    parser.add_argument("--hands", type=int, default=10, help="number of hands to play")
    parser.add_argument("--human-seats", type=int, default=0, help="how many of the seats are manual/human")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sessions")
    parser.add_argument(
        "--bots",
        type=str,
        default=None,
        help="comma-separated bot keys to cycle through for the non-human seats "
        "(e.g. 'shark,rock,maniac'); see --list-bots for the catalog. Also accepts "
        "custom specs like 'custom:tightness=0.2;aggression=0.8;bluff_frequency=0.3;size_variance=0.5' "
        "(any omitted parameter gets a moderate default). "
        "Default: cycle through every catalog bot, weakest to strongest.",
    )
    parser.add_argument(
        "--list-bots", action="store_true", help="print the available bot types and difficulty ranking, then exit"
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("hand_histories"),
        help="directory to write the session's hand history JSONL file into",
    )
    args = parser.parse_args(argv)

    if args.bots:
        for key in args.bots.split(","):
            _validate_bot_key(parser, key.strip())

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_bots:
        print_bot_catalog()
        return

    rng = random.Random(args.seed)
    bot_keys = [key.strip() for key in args.bots.split(",")] if args.bots else None

    config = GameConfig(num_players=args.players, starting_stack=args.stack, small_blind=args.sb, big_blind=args.bb)
    players = build_players(args.players, args.human_seats, rng, bot_keys)

    history_path = args.history_dir / f"session_{int(rng.random() * 1_000_000):06d}.jsonl"
    with HandHistoryWriter(history_path) as writer:
        table = Table(config, players, rng=rng, history_writer=writer)
        for i in range(args.hands):
            if sum(1 for s in table.stacks if s > 0) < 2:
                print(f"Session ended early after {i} hands: fewer than 2 players have chips left.")
                break
            result = table.play_hand()
            print(f"Hand {result.hand_id}: payouts={result.payouts}  stacks={result.final_stacks}")

    print(f"Hand history written to {history_path}")


if __name__ == "__main__":
    main()
