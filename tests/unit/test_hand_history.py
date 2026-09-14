import random

from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import HandHistoryReader, HandHistoryWriter
from pokerlab.engine.table import Table
from pokerlab.players.scripted import make_always_call_bot


def test_hand_history_roundtrips_through_jsonl(tmp_path):
    path = tmp_path / "session.jsonl"
    # All always-call bots never voluntarily bet/raise, so nobody busts
    # within 5 hands -- keeps this test about the JSONL round-trip, not
    # about surviving elimination variance (that's covered by the fuzz
    # test in test_full_hand_flow.py).
    config = GameConfig(num_players=3, starting_stack=200, small_blind=1, big_blind=2)
    players = [
        make_always_call_bot("p0", "P0"),
        make_always_call_bot("p1", "P1"),
        make_always_call_bot("p2", "P2"),
    ]
    with HandHistoryWriter(path) as writer:
        table = Table(config, players, rng=random.Random(7), history_writer=writer)
        played = table.play_session(5)

    assert len(played) == 5
    read_back = list(HandHistoryReader(path).iter_hands())
    assert len(read_back) == 5

    for original, reloaded in zip(played, read_back):
        assert reloaded.hand_id == original.hand_history.hand_id
        assert reloaded.schema_version == 1
        assert reloaded.payouts == original.hand_history.payouts
        assert reloaded.final_stacks == original.hand_history.final_stacks
        assert reloaded.hole_cards == original.hand_history.hole_cards
        assert reloaded.community_cards == original.hand_history.community_cards
        assert len(reloaded.actions) == len(original.hand_history.actions)
        for a, b in zip(reloaded.actions, original.hand_history.actions):
            assert a.action_type == b.action_type
            assert a.seat == b.seat
            assert a.amount == b.amount


def test_hand_history_file_is_valid_json_lines(tmp_path):
    import json

    path = tmp_path / "session.jsonl"
    config = GameConfig(num_players=2, starting_stack=100, small_blind=1, big_blind=2)
    players = [make_always_call_bot("p0", "P0"), make_always_call_bot("p1", "P1")]
    with HandHistoryWriter(path) as writer:
        table = Table(config, players, rng=random.Random(3), history_writer=writer)
        table.play_session(3)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert parsed["schema_version"] == 1
        assert "actions" in parsed
        assert all(isinstance(c, str) for c in parsed["community_cards"])
