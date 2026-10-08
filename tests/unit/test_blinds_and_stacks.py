import random

import pytest
from support import make_always_call_bot

from pokerlab.engine.actions import ActionType
from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table


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


class ShoveBot:
    """Goes all-in the moment it legally can, so every hand it plays runs
    out with nobody left to act."""

    def __init__(self, player_id: str, name: str) -> None:
        self.player_id = player_id
        self.name = name

    def act(self, observation, legal_actions):
        from pokerlab.engine.actions import Action

        for la in legal_actions:
            if la.action_type == ActionType.ALL_IN:
                return Action(ActionType.ALL_IN)
        return Action(legal_actions[0].action_type)


def test_on_street_dealt_hook_reports_each_board_as_it_is_dealt():
    calls = []
    config = GameConfig(num_players=3, starting_stack=200, small_blind=1, big_blind=2)
    players = [make_always_call_bot(f"p{i}", f"P{i}") for i in range(3)]
    table = Table(config, players, rng=random.Random(0), on_street_dealt=calls.append)

    result = table.play_hand()

    assert [c["street"].value for c in calls] == ["flop", "turn", "river"]
    assert [len(c["community_cards"]) for c in calls] == [3, 4, 5]
    assert calls[-1]["community_cards"] == result.hand_history.community_cards
    assert all(c["hand_id"] == result.hand_id for c in calls)
    # Everyone can still act on every street, so nothing is a forced runout.
    assert [c["betting_closed"] for c in calls] == [False, False, False]


def test_on_street_dealt_flags_an_all_in_runout_as_betting_closed():
    """The one case a spectator cannot see any other way: with every
    remaining player all-in, Table calls no Player at all between the last
    bet and the payouts, so without this flag a GUI has nothing to show."""
    calls = []
    config = GameConfig(num_players=3, starting_stack=200, small_blind=1, big_blind=2)
    players = [ShoveBot(f"p{i}", f"P{i}") for i in range(3)]
    table = Table(config, players, rng=random.Random(7), on_street_dealt=calls.append)

    table.play_hand()

    assert [c["street"].value for c in calls] == ["flop", "turn", "river"]
    assert all(c["betting_closed"] for c in calls)


def test_a_table_without_the_street_hook_still_plays_normally():
    table = make_table(3, seed=3)
    result = table.play_hand()
    assert sum(result.final_stacks.values()) == 600


# ---- blinds that go up -------------------------------------------------------------


def _blind_table(schedule, *, small=50, big=100, players=3, seed=1):
    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.table import Table

    bots = [make_always_call_bot(f"p{i}", f"B{i}") for i in range(players)]
    config = GameConfig(num_players=players, starting_stack=100_000, small_blind=small, big_blind=big)
    return Table(config, bots, rng=random.Random(seed), blind_schedule=schedule)


def test_the_blinds_go_up_by_the_factor_every_n_hands_rounded_up():
    from pokerlab.engine.config import BlindSchedule

    table = _blind_table(BlindSchedule(every=5, factor=1.2))
    seen, next_up = [], []
    for _ in range(16):
        table.stacks = [100_000] * 3
        seen.append(table.play_hand().hand_history.small_blind)
        next_up.append(table.current_blinds())
    assert seen[:5] == [50] * 5 and seen[5:10] == [60] * 5 and seen[10:15] == [72] * 5 and seen[15] == 87
    # 50 * 1.2 ** 3 = 86.4 -> 87; after the fifth hand the next one is already at the new level
    assert next_up[4] == (60, 120) and next_up[3] == (50, 100)


def test_each_level_is_computed_from_the_initial_blinds_not_from_the_last_rounded_ones():
    from pokerlab.engine.config import BlindSchedule

    schedule = BlindSchedule(every=1, factor=1.2)
    assert [schedule.blinds(1, 2, hands) for hands in range(5)] == [(1, 2), (2, 3), (2, 3), (2, 4), (3, 5)]
    # floating point: 100 * 1.2 is 120.00000000000001, which must not round up to 121
    assert BlindSchedule(every=1, factor=1.2).blinds(50, 100, 1) == (60, 120)


def test_the_blinds_stay_in_order_at_every_level():
    from pokerlab.engine.config import BlindSchedule

    for factor in (1.0, 1.05, 1.2, 1.5, 2.0):
        schedule = BlindSchedule(every=1, factor=factor)
        for small, big in ((1, 2), (50, 100), (3, 4)):
            for hands in range(40):
                low, high = schedule.blinds(small, big, hands)
                assert 0 < low < high


def test_a_schedule_refuses_nonsense():
    from pokerlab.engine.config import BlindSchedule

    for every, factor in ((0, 1.2), (5, 0.9), (5, float("nan"))):
        with pytest.raises(ValueError):
            BlindSchedule(every=every, factor=factor)
    assert BlindSchedule(every=5, factor=1.0).blinds(50, 100, 1000) == (50, 100)


def test_every_hand_records_the_blinds_it_was_played_at_and_the_observation_carries_them():
    from pokerlab.engine.config import BlindSchedule

    table = _blind_table(BlindSchedule(every=2, factor=2.0), players=2)
    started = []
    table._on_hand_started = started.append
    seen_by_bots = []
    for bot in table.players:
        inner = bot.act

        def act(observation, legal, inner=inner):
            seen_by_bots.append(observation.big_blind)
            return inner(observation, legal)

        bot.act = act
    histories = []
    for _ in range(5):
        table.stacks = [100_000] * 2
        histories.append(table.play_hand().hand_history)
    assert [h.big_blind for h in histories] == [100, 100, 200, 200, 400]
    assert [info["big_blind"] for info in started] == [100, 100, 200, 200, 400]
    assert set(seen_by_bots) == {100, 200, 400}
    # the blinds were really posted at the level of the hand, not just recorded
    for h in histories:
        posted = [r.amount for r in h.actions if r.action_type.value == "post_blind"]
        assert sorted(posted) == [h.small_blind, h.big_blind]


def test_chips_are_conserved_while_the_blinds_go_up_even_when_stacks_run_short():
    from pokerlab.engine.config import BlindSchedule

    table = _blind_table(BlindSchedule(every=3, factor=1.5), small=10, big=20, players=5, seed=4)
    table.stacks = [400, 250, 90, 700, 160]
    total = sum(table.stacks)
    for _ in range(40):
        if sum(1 for stack in table.stacks if stack > 0) < 2:
            break
        table.play_hand()
        assert sum(table.stacks) == total


def test_without_a_schedule_the_blinds_never_move():
    table = _blind_table(None)
    for _ in range(12):
        table.stacks = [100_000] * 3
        assert table.play_hand().hand_history.big_blind == 100
