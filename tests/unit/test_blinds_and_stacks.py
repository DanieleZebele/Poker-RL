import random

from pokerlab.engine.actions import ActionType
from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.scripted import make_always_call_bot


def make_table(num_players: int, starting_stack=200, small_blind=1, big_blind=2, seed=0) -> Table:
    config = GameConfig(num_players=num_players, starting_stack=starting_stack, small_blind=small_blind, big_blind=big_blind)
    players = [make_always_call_bot(f"p{i}", f"P{i}") for i in range(num_players)]
    return Table(config, players, rng=random.Random(seed))


def blind_posts(hand_history):
    return [a for a in hand_history.actions if a.action_type == ActionType.POST_BLIND]


def test_heads_up_button_is_small_blind():
    table = make_table(2)
    result = table.play_hand()
    hh = result.hand_history
    posts = blind_posts(hh)
    assert len(posts) == 2
    sb_post = min(posts, key=lambda a: a.amount)
    # heads-up: the button seat posts the small blind.
    assert sb_post.seat == hh.button_seat
    assert sb_post.amount == 1
    bb_post = next(a for a in posts if a.seat != hh.button_seat)
    assert bb_post.amount == 2


def test_three_handed_blinds_are_left_of_button():
    table = make_table(3)
    result = table.play_hand()
    hh = result.hand_history
    posts = {a.seat: a.amount for a in blind_posts(hh)}
    button = hh.button_seat
    sb_seat = (button + 1) % 3
    bb_seat = (button + 2) % 3
    assert posts[sb_seat] == 1
    assert posts[bb_seat] == 2
    assert button not in posts


def test_button_rotates_across_hands():
    table = make_table(4, starting_stack=1000)
    buttons = []
    for _ in range(4):
        result = table.play_hand()
        buttons.append(result.hand_history.button_seat)
    # With everyone always-calling and deep stacks, nobody should bust in 4
    # hands, so the button must visit 4 distinct seats via simple rotation.
    assert len(set(buttons)) == 4
    for i in range(1, 4):
        assert buttons[i] == (buttons[i - 1] + 1) % 4


def test_short_stack_posts_all_in_blind():
    config = GameConfig(num_players=2, starting_stack=200, small_blind=5, big_blind=10)
    players = [make_always_call_bot("p0", "P0"), make_always_call_bot("p1", "P1")]
    table = Table(config, players, rng=random.Random(0))
    table.stacks[1] = 3  # seat 1 can't cover a full blind of either size
    table._button_seat = 0  # force seat 0 as button -> heads-up SB
    result = table.play_hand()
    hh = result.hand_history
    posts = {a.seat: a.amount for a in blind_posts(hh)}
    # seat1 is BB in heads-up (button=0 is SB); can only post its remaining 3 chips.
    assert posts[1] == 3
    assert hh.starting_stacks[1] == 3


def test_busted_player_is_skipped_in_later_hands():
    table = make_table(3, starting_stack=1000)
    table.stacks[1] = 0  # seat 1 already busted before this session
    result = table.play_hand()
    assert 1 not in result.hand_history.starting_stacks
    assert 1 not in result.hand_history.seat_names
    assert set(result.hand_history.starting_stacks.keys()) == {0, 2}


def test_on_hand_started_hook_fires_once_with_blinds_and_hole_cards():
    calls = []
    config = GameConfig(num_players=3, starting_stack=200, small_blind=1, big_blind=2)
    players = [make_always_call_bot(f"p{i}", f"P{i}") for i in range(3)]
    table = Table(config, players, rng=random.Random(0), on_hand_started=calls.append)

    result = table.play_hand()

    assert len(calls) == 1
    info = calls[0]
    assert info["hand_id"] == result.hand_id
    assert info["button_seat"] == result.hand_history.button_seat
    assert info["small_blind"] == 1 and info["big_blind"] == 2
    assert {info["sb_seat"], info["bb_seat"]} == {(info["button_seat"] + 1) % 3, (info["button_seat"] + 2) % 3}
    assert set(info["hole_cards"].keys()) == {0, 1, 2}
    for hole in info["hole_cards"].values():
        assert len(hole) == 2
