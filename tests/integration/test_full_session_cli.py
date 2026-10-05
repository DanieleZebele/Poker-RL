import json
from pathlib import Path

import pytest

from pokerlab.cli.play import main

torch = pytest.importorskip("torch")

from pokerlab.rl.policy import PokerActorCritic
from pokerlab.rl.ppo import save_checkpoint


def make_checkpoint(path: Path) -> Path:
    """A minimal, fast-to-load checkpoint: every non-human seat in these CLI
    tests needs a real `model:<path>` to seat."""
    save_checkpoint(path, PokerActorCritic(hidden=16, num_layers=1))
    return path


def test_bot_only_session_runs_and_writes_history(tmp_path, capsys):
    model = make_checkpoint(tmp_path / "agent.pt")
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
            "--bots",
            f"model:{model}",
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
    model = make_checkpoint(tmp_path / "agent.pt")
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
            "--bots",
            f"model:{model}",
            "--history-dir",
            str(history_dir),
        ]
    )
    out = capsys.readouterr().out
    assert "Hand history written to" in out


def test_list_bots_does_not_play_a_session(capsys):
    """`--list-bots` lists discovered trained models, so its exact output
    depends on whatever is under checkpoints/ on
    the machine running the tests -- the one thing every environment shares
    is that it must not play a session."""
    main(["--list-bots"])
    out = capsys.readouterr().out
    assert "Hand history written to" not in out


def test_explicit_bot_selection_is_honored(tmp_path):
    rock = make_checkpoint(tmp_path / "rock_v1.pt")
    shark = make_checkpoint(tmp_path / "shark_v1.pt")
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
            f"model:{rock},model:{shark}",
            "--seed",
            "9",
            "--history-dir",
            str(history_dir),
        ]
    )
    hand = json.loads(next(history_dir.glob("session_*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    # seat 0 -> rock, seat 1 -> shark, seat 2 -> rock again (cycles through --bots)
    assert hand["seat_names"] == {"0": "rock_v10", "1": "shark_v11", "2": "rock_v12"}


def test_unknown_bot_spec_is_rejected_with_a_clear_error(capsys):
    with pytest.raises(SystemExit):
        main(["--players", "2", "--bots", "not_a_real_bot"])
    err = capsys.readouterr().err
    assert "model:" in err


def test_missing_model_path_is_rejected_with_a_clear_error(tmp_path, capsys):
    missing = tmp_path / "nope.pt"
    with pytest.raises(SystemExit):
        main(["--players", "2", "--bots", f"model:{missing}"])
    err = capsys.readouterr().err
    assert str(missing) in err
