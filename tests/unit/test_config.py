"""`config.toml`: one file of overrides under the flags and over the defaults.

Pure Python, no torch: the precedence and the validation are what decide what a
whole fleet runs with, so they belong in the ordinary suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pokerlab.config import (
    ConfigError,
    add_config_arguments,
    apply_config,
    diff_settings,
    format_config,
    parse_with_config,
    read_config,
    resolved_settings,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--name", default="x")
    parser.add_argument("--global-round", dest="global_round", action="store_true", default=True)
    parser.add_argument("--no-global-round", dest="global_round", action="store_false")
    parser.add_argument("--machine", default="host")
    parser.add_argument("--models-dir", type=Path, default=Path("m"))
    add_config_arguments(parser)
    return parser


def write(tmp_path, text, name="config.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---- reading -----------------------------------------------------------------


def test_sections_are_flattened_and_dashes_read_as_underscores(tmp_path):
    path = write(tmp_path, "top = 1\n[a]\nppo-epochs = 3\n[b]\nlr = 0.1\n")
    assert read_config(path) == {"top": 1, "ppo_epochs": 3, "lr": 0.1}


def test_a_key_set_twice_is_an_error(tmp_path):
    path = write(tmp_path, "[a]\nlr = 1\n[b]\nlr = 2\n")
    with pytest.raises(ConfigError, match="more than once"):
        read_config(path)


def test_a_missing_or_broken_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        read_config(tmp_path / "nope.toml")
    with pytest.raises(ConfigError, match="not valid TOML"):
        read_config(write(tmp_path, "lr = = 1\n"))


# ---- applying ----------------------------------------------------------------


def test_the_file_overrides_the_default_and_a_flag_overrides_the_file(tmp_path):
    path = write(tmp_path, "hands = 512\nname = 'from-file'\n")
    args, report = parse_with_config(make_parser(), ["--config", str(path), "--hands", "64"])
    assert args.hands == 64  # flag > file
    assert args.name == "from-file"  # file > default
    assert args.lr == 3e-4  # absent key = default
    assert report.path == path and report.applied == {"hands": 512, "name": "from-file"}


def test_a_typo_is_rejected_with_the_closest_name(tmp_path):
    path = write(tmp_path, "hand = 5\n")
    with pytest.raises(ConfigError, match="did you mean 'hands'"):
        parse_with_config(make_parser(), ["--config", str(path)])


def test_per_machine_settings_and_paths_are_rejected(tmp_path):
    for key in ("machine = 'x'", "models_dir = '/tmp'"):
        with pytest.raises(ConfigError, match="per-machine"):
            parse_with_config(make_parser(), ["--config", str(write(tmp_path, key + "\n"))])


def test_values_are_checked_against_the_flag_they_stand_for(tmp_path):
    cases = [
        "hands = 1.5",  # a float where an integer is wanted
        "hands = true",  # a bool is never a number
        "lr = 'fast'",
        "name = 3",
        "global_round = 1",  # a toggle takes true/false
    ]
    for line in cases:
        with pytest.raises(ConfigError):
            parse_with_config(make_parser(), ["--config", str(write(tmp_path, line + "\n"))])
    # An integer is a fine float, and both spellings of the toggle's flag resolve.
    args, _ = parse_with_config(
        make_parser(), ["--config", str(write(tmp_path, "lr = 1\nglobal_round = false\n"))]
    )
    assert args.lr == 1.0 and isinstance(args.lr, float) and args.global_round is False
    args, _ = parse_with_config(
        make_parser(),
        ["--config", str(write(tmp_path, "global_round = false\n")), "--global-round"],
    )
    assert args.global_round is True


def test_a_key_another_cli_owns_is_not_a_typo_but_is_still_type_checked(tmp_path):
    other = argparse.ArgumentParser()
    other.add_argument("--only-there", type=int, default=1)
    sibling = lambda: [other]
    ok = write(tmp_path, "only_there = 7\nhands = 9\n")
    args, report = parse_with_config(make_parser(), ["--config", str(ok)], siblings=sibling)
    assert args.hands == 9 and not hasattr(args, "only_there")
    assert "only_there" not in report.applied
    with pytest.raises(ConfigError, match="integer"):
        parse_with_config(
            make_parser(), ["--config", str(write(tmp_path, "only_there = 'a'\n", "b.toml"))],
            siblings=sibling,
        )


def test_a_lenient_cli_skips_keys_it_cannot_see_but_checks_its_own(tmp_path):
    """The torch-free CLIs cannot import `poker-train`'s parser, so a key they do
    not know may belong to it: skipped, not refused. What they do own is still
    checked, and a per-machine key is still refused when someone they can see
    owns it."""
    path = write(tmp_path, "somebody_elses = 1\nhands = 9\n")
    args, report = parse_with_config(make_parser(), ["--config", str(path)], lenient=True)
    assert args.hands == 9 and report.applied == {"hands": 9}
    with pytest.raises(ConfigError, match="unknown parameter"):
        parse_with_config(make_parser(), ["--config", str(path)])
    with pytest.raises(ConfigError, match="integer"):
        parse_with_config(
            make_parser(), ["--config", str(write(tmp_path, "hands = 'x'\n", "b.toml"))],
            lenient=True,
        )
    with pytest.raises(ConfigError, match="per-machine"):
        parse_with_config(
            make_parser(), ["--config", str(write(tmp_path, "machine = 'x'\n", "c.toml"))],
            lenient=True,
        )


def test_resolve_cli_exits_2_on_a_bad_file_and_prints_on_request(tmp_path, capsys):
    from pokerlab.config import resolve_cli

    bad = write(tmp_path, "hands = 'x'\n")
    with pytest.raises(SystemExit) as raised:
        resolve_cli(make_parser(), ["--config", str(bad)])
    assert raised.value.code == 2

    good = write(tmp_path, "hands = 9\n", "ok.toml")
    assert resolve_cli(make_parser(), ["--config", str(good), "--print-config"]) is None
    assert "hands = 9  # config.toml" in capsys.readouterr().out
    assert resolve_cli(make_parser(), ["--config", str(good)]).hands == 9


def make_list_parser() -> argparse.ArgumentParser:
    parser = make_parser()
    parser.add_argument("--factors", type=float, nargs="+", default=[1.0])
    return parser


def test_a_list_is_checked_item_by_item_and_may_not_be_empty(tmp_path):
    args, _ = parse_with_config(
        make_list_parser(), ["--config", str(write(tmp_path, "factors = [0.5, 1, 2]\n"))]
    )
    assert args.factors == [0.5, 1.0, 2.0]
    for bad in ("factors = []", "factors = 2", "factors = [1, 'a']"):
        with pytest.raises(ConfigError, match="factors"):
            parse_with_config(
                make_list_parser(), ["--config", str(write(tmp_path, bad + "\n", "bad.toml"))]
            )


def test_a_list_setting_is_recorded_and_printed_as_a_list():
    values = {"factors": [0.8, 1.0], "name": "x"}
    assert resolved_settings(values)["factors"] == [0.8, 1.0]
    assert "factors = [0.8, 1.0]" in format_config(values)


# ---- which file --------------------------------------------------------------


def test_the_default_file_is_optional_but_a_named_one_is_not(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args, report = parse_with_config(make_parser(), [])
    assert report.path is None and args.hands == 256
    with pytest.raises(ConfigError, match="not found"):
        parse_with_config(make_parser(), ["--config", "missing.toml"])
    write(tmp_path, "hands = 11\n")
    assert parse_with_config(make_parser(), [])[0].hands == 11
    # The empty string reads no file at all: how a supervisor tells its workers
    # that everything they need is already on their command line.
    assert parse_with_config(make_parser(), ["--config", ""])[0].hands == 256


# ---- what a run records ------------------------------------------------------


def test_resolved_settings_leave_out_what_belongs_to_the_machine():
    args = make_parser().parse_args(["--hands", "5"])
    settings = resolved_settings(vars(args))
    assert settings == {"hands": 5, "lr": 3e-4, "name": "x", "global_round": True}


def test_a_changed_setting_is_reported_and_an_unchanged_one_is_not():
    old = {"hands": 1, "lr": 0.1, "machine": "a"}
    new = {"hands": 2, "lr": 0.1, "machine": "b"}
    assert diff_settings(old, new) == ["hands: 1 -> 2"]


def test_print_config_marks_what_the_file_set():
    text = format_config({"hands": 5, "lr": 0.1, "global_round": True, "name": "x"}, ["hands"])
    assert text.splitlines() == [
        "hands = 5  # config.toml", "lr = 0.1", "global_round = true", 'name = "x"',
    ]


def test_apply_config_sets_defaults_on_the_parser_itself(tmp_path):
    parser = make_parser()
    apply_config(parser, {"hands": 3})
    assert parser.parse_args([]).hands == 3


def test_a_value_the_flags_own_type_refuses_is_refused_in_the_file(tmp_path):
    def even_text(text):
        if len(text) % 2:
            raise argparse.ArgumentTypeError("odd length")
        return text

    parser = make_parser()
    parser.add_argument("--code", type=even_text, default="ab")
    apply_config(parser, {"code": "abcd"})
    assert parser.parse_args([]).code == "abcd"
    with pytest.raises(ConfigError, match="odd length"):
        apply_config(make_parser_with(even_text), {"code": "abc"})


def make_parser_with(kind):
    parser = make_parser()
    parser.add_argument("--code", type=kind, default="ab")
    return parser
