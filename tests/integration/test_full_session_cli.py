import json

import pytest

from pokerlab.cli.play import main, parse_custom_bot_spec


def test_bot_only_session_runs_and_writes_history(tmp_path, capsys):
    history_dir = tmp_path / "hh"
    main(
        [
            "--players",
            "5",
            "--stack",
            "100",
            "--sb",
            "1",
            "--bb",
            "2",
            "--hands",
            "8",
            "--human-seats",
            "0",
            "--seed",
            "123",
            "--history-dir",
            str(history_dir),
        ]
    )

    out = capsys.readouterr().out
    assert "Hand history written to" in out

    files = list(history_dir.glob("session_*.jsonl"))
    assert len(files) == 1

    lines = files[0].read_text(encoding="utf-8").strip().split("\n")
    assert 1 <= len(lines) <= 8  # may end early if players bust down to one
    for line in lines:
        hand = json.loads(line)
        assert hand["schema_version"] == 1
        assert sum(hand["final_stacks"].values()) == sum(hand["starting_stacks"].values())


def test_session_ends_early_message_when_players_bust(tmp_path, capsys):
    history_dir = tmp_path / "hh"
    # Tiny stacks relative to blinds make a quick bust-out likely; either
    # outcome (finishes all hands, or ends early) must be handled cleanly.
    main(
        [
            "--players",
            "2",
            "--stack",
            "10",
            "--sb",
            "1",
            "--bb",
            "2",
            "--hands",
            "50",
            "--human-seats",
            "0",
            "--seed",
            "1",
            "--history-dir",
            str(history_dir),
        ]
    )
    out = capsys.readouterr().out
    assert "Hand history written to" in out


def test_list_bots_prints_catalog_and_does_not_play(capsys):
    main(["--list-bots"])
    out = capsys.readouterr().out
    assert "random" in out
    assert "shark" in out
    assert "Hand history written to" not in out


def test_explicit_bot_selection_is_honored(tmp_path):
    history_dir = tmp_path / "hh"
    main(
        [
            "--players",
            "3",
            "--stack",
            "300",
            "--hands",
            "3",
            "--human-seats",
            "0",
            "--bots",
            "rock,shark",
            "--seed",
            "9",
            "--history-dir",
            str(history_dir),
        ]
    )
    hand = json.loads(next(history_dir.glob("session_*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    # seat 0 -> rock, seat 1 -> shark, seat 2 -> rock again (cycles through --bots)
    assert hand["seat_names"] == {"0": "Rock0", "1": "Shark1", "2": "Rock2"}


def test_unknown_bot_key_is_rejected_with_a_clear_error(capsys):
    with pytest.raises(SystemExit):
        main(["--players", "2", "--bots", "not_a_real_bot"])
    err = capsys.readouterr().err
    assert "not_a_real_bot" in err


def test_parse_custom_bot_spec_applies_defaults_and_overrides():
    params = parse_custom_bot_spec("custom:tightness=0.2;aggression=0.9")
    assert params["tightness"] == 0.2
    assert params["aggression"] == 0.9
    assert params["bluff_frequency"] == 0.1  # default, not overridden
    assert params["size_variance"] == 0.2  # default, not overridden


def test_parse_custom_bot_spec_with_no_params_returns_all_defaults():
    params = parse_custom_bot_spec("custom:")
    assert params == {"tightness": 0.3, "aggression": 0.5, "bluff_frequency": 0.1, "size_variance": 0.2}


@pytest.mark.parametrize(
    "bad_spec",
    [
        "custom:not_a_param=0.5",  # unknown parameter name
        "custom:tightness=notanumber",  # unparseable value
        "custom:tightness=1.5",  # out of 0-1 range
        "custom:tightness",  # missing '='
    ],
)
def test_parse_custom_bot_spec_rejects_malformed_specs(bad_spec):
    with pytest.raises(ValueError):
        parse_custom_bot_spec(bad_spec)


def test_custom_bot_spec_is_usable_from_the_cli(tmp_path):
    history_dir = tmp_path / "hh"
    main(
        [
            "--players",
            "2",
            "--stack",
            "300",
            "--hands",
            "2",
            "--human-seats",
            "0",
            "--bots",
            "custom:tightness=0.05;aggression=0.95;bluff_frequency=0.5;size_variance=0.8",
            "--seed",
            "3",
            "--history-dir",
            str(history_dir),
        ]
    )
    hand = json.loads(next(history_dir.glob("session_*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert hand["seat_names"] == {"0": "Custom0", "1": "Custom1"}


def test_invalid_custom_bot_spec_from_cli_is_rejected_with_a_clear_error(capsys):
    with pytest.raises(SystemExit):
        main(["--players", "2", "--bots", "custom:tightness=5"])
    err = capsys.readouterr().err
    assert "tightness" in err
