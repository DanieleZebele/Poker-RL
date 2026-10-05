"""The one file the fleet's parameters can be set in: `config.toml`.

Every parameter that matters lives in three places -- a `DEFAULT_*`
constant, an argparse flag in each CLI that touches it, and a string in the
command line `loop.py` builds for `poker-train`. This module adds the one place
a person edits, **without moving any of the reasoning**: the constants stay where
they are, comments included, as the *defaults*, and the file only *overrides*
them. An absent key means "use the documented default".

Precedence, highest first: an explicit command-line flag, then `config.toml`,
then the default in the code. It is implemented as `parser.set_defaults(...)`
before parsing, so argparse itself guarantees the flag wins and nothing else has
to know a file exists.

Rules worth knowing:

- **The file is data, not code**, which is why it is TOML and not a Python module:
  a supervisor re-reads it every generation for weeks, and a half-saved or
  mistyped file must cost a warning and the previous values, never run inside a
  live process. `tomllib` is in the standard library from 3.11.
- **A key is the long flag name with underscores** (`--ppo-epochs` is
  `ppo_epochs`). Section headers (`[game]`, `[evaluation]`...) are only for
  reading; the keys are flattened, so a name may appear once in the whole file.
- **A typo is an error, not a silent no-op**: a key no CLI knows is rejected,
  with the closest real name suggested.
- **Per-machine settings are rejected** (`LOCAL_ONLY`, and every path): the file
  is shared by the whole fleet through the project directory, so a key that is
  true of one host (`machine`, `workers`, `device`, any directory) would be
  silently wrong on the others. They stay flags, and `run.sh` derives them.
- **Five programs read it** (`poker-train`, `poker-loop`, `poker-elo`,
  `benchmark_arena`, `poker-dashboard`), so a key means one thing in all of them;
  `rl/siblings.py` says who can see whose parser, and the torch-free ones read
  `lenient`ly (see `apply_config`).
- **Pure Python, no torch**, so it is importable from every CLI at no cost.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config.toml")

# Settings that describe *this process or this machine*, not the fleet. A shared
# file cannot hold them truthfully. Any argument typed as a `Path` is local too
# (checked on the action, not listed here): directories differ per host.
LOCAL_ONLY = frozenset(
    {
        "machine",
        "workers",
        "device",
        "seed",
        "seed_base",
        "resume",
        "hp_arm",
        "parent_label",
        "archive_prefix",
        "fill_stop_file",
        "status",
        "watch",
        "generations",
        "keep_work",
        "top",
        # Per-invocation choices of the hand-run tools: whether this run may delete
        # checkpoints, whether it writes anything, and
        # which interface the dashboard listens on. None of them is a property of
        # the fleet, and `prune` in particular is a safety switch that a shared
        # file must never be able to flip for everyone.
        "prune",
        "dry_run",
        "host",
        "config",
        "print_config",
    }
)


class ConfigError(Exception):
    """The file cannot be used as it stands; the message says why, in one line."""


@dataclass(frozen=True)
class ConfigReport:
    """What reading the file did, for the caller to print."""

    path: Path | None = None
    applied: dict[str, Any] = field(default_factory=dict)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """The two flags every CLI that reads the file carries."""
    parser.add_argument(
        "--config",
        default=None,
        help=f"parameter file (TOML); default {DEFAULT_CONFIG_PATH} when it exists. "
        "A flag given here beats the file, the file beats the code's default. "
        'Pass "" to read no file at all',
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print every parameter as it resolved (file and flags applied) in "
        "the file's own format, and exit",
    )


def read_config(path: Path) -> dict[str, Any]:
    """Parse `path` into a flat `{key: value}`, or raise `ConfigError`."""
    try:
        data = tomllib.loads(Path(path).read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path}: cannot be read ({error})") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML ({error})") from None

    flat: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        if isinstance(value, dict):
            raise ConfigError(f"{path}: '{key}' is nested too deeply (one level of [section] only)")
        name = key.replace("-", "_")
        if name in flat:
            raise ConfigError(f"{path}: '{name}' is set more than once")
        flat[name] = value

    for key, value in data.items():
        if isinstance(value, dict):
            for inner_key, inner in value.items():
                put(inner_key, inner)
        else:
            put(key, value)
    return flat


def _is_local(key: str, action: argparse.Action | None) -> bool:
    return key in LOCAL_ONLY or (action is not None and action.type is Path)


def _coerce(key: str, action: argparse.Action, value: Any) -> Any:
    """`value` as `action` would have parsed it, or `ConfigError`.

    argparse applies `type=` only to *strings*, and a TOML value is already
    typed, so the check is spelled out here: an integer where a float is wanted
    is fine (`lr = 1`), a float where an integer is wanted is not, and a bool is
    never a number.
    """
    if isinstance(action, argparse._StoreTrueAction | argparse._StoreFalseAction):
        if not isinstance(value, bool):
            raise ConfigError(f"'{key}' must be true or false, not {value!r}")
        return value
    if action.nargs in ("+", "*"):
        if not isinstance(value, list) or (action.nargs == "+" and not value):
            raise ConfigError(f"'{key}' must be a non-empty list, not {value!r}")
        return [_coerce(key, _element_of(action), item) for item in value]
    wanted = action.type
    if wanted is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{key}' must be an integer, not {value!r}")
        result: Any = value
    elif wanted is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"'{key}' must be a number, not {value!r}")
        result = float(value)
        if not math.isfinite(result):
            raise ConfigError(f"'{key}' must be finite, not {value!r}")
    else:
        if not isinstance(value, str):
            raise ConfigError(f"'{key}' must be a string, not {value!r}")
        result = value
        if wanted is not None and wanted is not str:
            # A validating `type=` (a K schedule, the parent bands): the file
            # gets the same check as the flag, and its error, not a later one.
            try:
                wanted(value)
            except (argparse.ArgumentTypeError, ValueError) as error:
                raise ConfigError(f"'{key}': {error}") from None
    if action.choices is not None and result not in action.choices:
        raise ConfigError(f"'{key}' must be one of {sorted(action.choices)}, not {value!r}")
    return result


def _element_of(action: argparse.Action) -> argparse.Action:
    """`action` as it checks one item of its list."""
    return argparse.Action(option_strings=[], dest=action.dest, type=action.type, choices=action.choices)


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    # `_actions` is private but it is the only complete list argparse keeps.
    return {
        action.dest: action
        for action in parser._actions
        if action.dest != "help"
    }


def apply_config(
    parser: argparse.ArgumentParser,
    values: Mapping[str, Any],
    *,
    siblings: Sequence[argparse.ArgumentParser] = (),
    lenient: bool = False,
) -> ConfigReport:
    """Turn `values` into this parser's defaults.

    `siblings` are the other CLIs that read the same file: a key `parser` does
    not know but a sibling does is checked against the sibling's type and then
    left alone, because the file is shared and each program takes its own share.
    A key none of them knows is a typo.

    `lenient` is for a CLI that cannot import every sibling (the torch-free ones
    cannot reach `poker-train`'s parser without importing torch): a key nobody it
    can see knows is then skipped instead of refused, since it may well belong to
    a program it cannot see. The typo is still caught, by the first `poker-train`
    or `poker-loop` that reads the file.
    """
    own = _actions(parser)
    elsewhere: dict[str, argparse.Action] = {}
    for sibling in siblings:
        for dest, action in _actions(sibling).items():
            elsewhere.setdefault(dest, action)
    known = {**elsewhere, **own}

    applied: dict[str, Any] = {}
    for key, value in values.items():
        action = known.get(key)
        if action is None and lenient:
            continue
        if action is None:
            close = difflib.get_close_matches(key, [k for k in known if not _is_local(k, known[k])], 1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            raise ConfigError(f"unknown parameter '{key}'{hint}")
        if _is_local(key, action):
            raise ConfigError(
                f"'{key}' is per-machine and cannot be set in the shared file; "
                "pass it as a flag"
            )
        coerced = _coerce(key, action, value)
        if key not in own:
            continue
        applied[key] = coerced
    parser.set_defaults(**applied)
    return ConfigReport(applied=applied)


def _config_path(argv: Sequence[str]) -> tuple[Path | None, bool]:
    """The file to read, and whether it was asked for by name."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", default=None)
    named, _ = pre.parse_known_args(list(argv))
    if named.config is None:
        return (DEFAULT_CONFIG_PATH, False)
    if named.config == "":
        return (None, True)
    return (Path(named.config), True)


def parse_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    siblings: Callable[[], Sequence[argparse.ArgumentParser]] | None = None,
    lenient: bool = False,
) -> tuple[argparse.Namespace, ConfigReport]:
    """`parser.parse_args(argv)` with `config.toml` underneath the flags.

    A missing *default* file is no file; a missing file that was named with
    `--config` is an error, since the person asked for it. `siblings` is a
    callable so the other parsers (and what they import) are built only when a
    file is actually read. Raises `ConfigError` for a file that cannot be used.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    path, named = _config_path(arguments)
    report = ConfigReport()
    if path is not None and (named or path.is_file()):
        report = apply_config(
            parser,
            read_config(path),
            siblings=siblings() if siblings is not None else (),
            lenient=lenient,
        )
        report = ConfigReport(path=path, applied=report.applied)
    return parser.parse_args(arguments), report


def resolve_cli(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    siblings: Callable[[], Sequence[argparse.ArgumentParser]] | None = None,
    lenient: bool = False,
) -> argparse.Namespace | None:
    """The whole startup of a CLI that reads the file, for the ones with no extra steps.

    A file that cannot be used ends the program with exit code 2. `--print-config`
    prints what everything resolved to and returns `None`, which means "nothing
    left to do"; otherwise the namespace comes back with the file already under it.
    """
    try:
        args, report = parse_with_config(parser, argv, siblings=siblings, lenient=lenient)
    except ConfigError as error:
        parser.exit(2, f"{parser.prog}: config: {error}\n")
    if args.print_config:
        print(format_config(vars(args), [k for k, v in report.applied.items() if vars(args)[k] == v]))
        return None
    if report.path is not None:
        print(f"config: {report.path}, {len(report.applied)} valori", flush=True)
    return args


# ---- what a run resolved to --------------------------------------------------


def resolved_settings(values: Mapping[str, Any]) -> dict[str, Any]:
    """Every fleet-wide setting in `values`, as plain JSON-able data.

    This is what a run records about itself. With a file in the picture a run's
    parameters are no longer recoverable from its command line, and a rating
    earned under unknown settings is worth much less. Per-machine settings and
    paths are left out: they say where a run happened, not how it was set.
    """
    settings: dict[str, Any] = {}
    for key, value in values.items():
        if key in LOCAL_ONLY or key.startswith("_") or value is None or isinstance(value, Path):
            continue
        if isinstance(value, bool | int | float | str):
            settings[key] = value
        elif isinstance(value, list) and all(isinstance(v, int | float) for v in value):
            settings[key] = list(value)
    return settings


def diff_settings(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """One `key: old -> new` line per fleet-wide setting that changed."""
    before, after = resolved_settings(old), resolved_settings(new)
    return [
        f"{key}: {before.get(key, '-')!r} -> {after.get(key, '-')!r}"
        for key in after
        if before.get(key) != after[key]
    ]


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def format_config(values: Mapping[str, Any], from_file: Sequence[str] = ()) -> str:
    """`values` as `key = value` lines, ready to paste into the file.

    The keys the file set are marked, so what is a default and what was chosen
    can be told apart at a glance.
    """
    lines = []
    for key, value in resolved_settings(values).items():
        mark = "  # config.toml" if key in from_file else ""
        lines.append(f"{key} = {_toml_value(value)}{mark}")
    return "\n".join(lines)
