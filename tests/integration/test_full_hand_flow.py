import random

import pytest
from support import make_always_call_bot, make_random_legal_bot

from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player


def build_mixed_table(num_players: int, seed: int, starting_stack: int = 500) -> Table:
    config = GameConfig(num_players=num_players, starting_stack=starting_stack, small_blind=1, big_blind=2)
    players = []
    for i in range(num_players):
        if i % 2 == 0:
            players.append(make_always_call_bot(f"p{i}", f"AC{i}"))
        else:
            # Picks uniformly among whatever is legal, BET/RAISE included, so
            # this alone already exercises every action type the fuzz needs.
            players.append(make_random_legal_bot(f"p{i}", f"R{i}", rng=random.Random(seed * 31 + i)))
    return Table(config, players, rng=random.Random(seed))


@pytest.mark.parametrize("num_players", range(2, 10))
@pytest.mark.parametrize("seed", range(5))
def test_chip_conservation_holds_across_a_session(num_players, seed):
    table = build_mixed_table(num_players, seed)
    for _ in range(200):
        if sum(1 for s in table.stacks if s > 0) < 2:
            break
        total_before = sum(table.stacks)
        table.play_hand()
        assert sum(table.stacks) == total_before, "chip conservation invariant violated"


def test_single_hand_produces_a_balanced_result():
    table = build_mixed_table(num_players=6, seed=99)
    result = table.play_hand()
    hh = result.hand_history
    assert sum(hh.final_stacks.values()) == sum(hh.starting_stacks.values())
    assert sum(hh.payouts.values()) > 0
    assert set(hh.payouts.keys()) <= set(hh.starting_stacks.keys())


def test_showdown_reveals_hole_cards_for_every_dealt_seat():
    table = build_mixed_table(num_players=4, seed=5)
    result = table.play_hand()
    assert set(result.hand_history.hole_cards.keys()) == set(result.hand_history.starting_stacks.keys())
    for hole in result.hand_history.hole_cards.values():
        assert len(hole) == 2


@pytest.mark.parametrize("seed", range(5))
def test_chip_conservation_holds_with_random_players_and_uneven_stacks(seed):
    """Every seat a random player (folds for free included) and every stack drawn anew each
    hand: side pots of every shape, which the session above, half always-call bots on
    stacks that drift, rarely builds."""
    rng = random.Random(seed)
    for _ in range(300):
        num_players = rng.randint(2, 9)
        config = GameConfig(num_players=num_players, starting_stack=300, small_blind=1, big_blind=2)
        players = [make_random_legal_bot(f"p{i}", rng=random.Random(rng.random())) for i in range(num_players)]
        table = Table(config, players, rng=random.Random(rng.random()))
        table.stacks = [rng.randint(1, 300) for _ in range(num_players)]
        total_before = sum(table.stacks)
        table.play_hand()
        assert sum(table.stacks) == total_before, "chip conservation invariant violated"


class _ScriptedPlayer(Player):
    """Plays the actions it is given, in order, and fails if asked for one more."""

    def __init__(self, player_id: str, actions: list[Action]) -> None:
        super().__init__(player_id, player_id)
        self.actions = list(actions)

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        assert self.actions, f"{self.player_id} was asked to act with nothing left to decide"
        return self.actions.pop(0)


def test_the_last_player_with_chips_is_not_asked_to_act_with_nothing_to_call():
    """A hand that lost 60 chips. Two all-in players (0 and 1) and two with chips behind (2 and
    4), who both put in 101. On the flop 2 folds with nothing to call: 4 is now the only
    player with chips, nobody is left to bet against, so the betting is over. Before the fix
    4 was still asked to act, a random player folded for free too, and the layer only 2 and 4
    had paid into (71 -> 101, 60 chips) had nobody left to win it and vanished."""
    players = [
        _ScriptedPlayer("p0", [Action(ActionType.ALL_IN)]),  # the button, 71
        _ScriptedPlayer("p1", [Action(ActionType.ALL_IN)]),  # small blind, 58
        _ScriptedPlayer("p2", [Action(ActionType.CALL), Action(ActionType.FOLD)]),  # big blind
        _ScriptedPlayer("p3", [Action(ActionType.RAISE, amount=49), Action(ActionType.FOLD)]),
        _ScriptedPlayer("p4", [Action(ActionType.RAISE, amount=101)]),
    ]
    table = Table(GameConfig(num_players=5, starting_stack=200, small_blind=1, big_blind=2), players,
                  rng=random.Random(7))
    table.stacks = [71, 58, 156, 136, 118]
    hand = table.play_hand().hand_history

    assert sum(hand.final_stacks.values()) == sum(hand.starting_stacks.values())
    assert players[4].actions == []  # it played its one action, and was never asked again
    # 4 is still in the hand: the 71 -> 101 layer is its own whatever the cards.
    assert hand.final_stacks[4] >= 118 - 101 + 60


class _Announcements:
    """What a spectator was last told about who is due to act, and the seats actually asked."""

    def __init__(self) -> None:
        self.due: int | None = None
        self.view: tuple | None = None  # (observation, legal actions) announced for the player due
        self.asked: list[int] = []

    def asked_seat(self, seat: int, observation, legal_actions) -> None:
        assert self.due == seat, f"asked {seat}, announced {self.due}"
        # the decision announced ahead is exactly the one handed over
        assert self.view is not None and self.view[0] == observation and self.view[1] == legal_actions
        self.asked.append(seat)
        self.due = self.view = None

    def _announce(self, seat, view) -> None:
        assert (seat is None) == (view is None)
        if view is not None:
            assert view[0].my_seat == seat
        self.due, self.view = seat, view

    def hand_started(self, info: dict) -> None:
        self._announce(info["first_actor"], info["first_actor_view"])

    def street_dealt(self, info: dict) -> None:
        assert self.due is None, "a street was dealt with a player still due"
        self._announce(info["first_actor"], info["first_actor_view"])

    def action_applied(self, info: dict) -> None:
        self._announce(info["next_seat"], info["next_view"])


class _WatchedPlayer(Player):
    def __init__(self, inner: Player, announcements: _Announcements) -> None:
        super().__init__(inner.player_id, inner.name)
        self.inner, self.announcements, self.seat = inner, announcements, int(inner.player_id[1:])

    def act(self, observation, legal_actions):
        self.announcements.asked_seat(self.seat, observation, legal_actions)
        return self.inner.act(observation, legal_actions)


@pytest.mark.parametrize("seed", range(6))
def test_the_player_announced_as_next_is_the_one_asked_next(seed):
    """`first_actor` (hand started, street dealt) and `next_seat` (action applied) tell a
    spectator who will be asked before they are: over random hands of every table size, with
    all-ins and folds for free, every `act` call is to exactly the seat last announced, and
    nobody is asked after a `None`."""
    rng = random.Random(seed)
    decisions = 0
    for _ in range(150):
        num_players = rng.randint(2, 9)
        config = GameConfig(num_players=num_players, starting_stack=300, small_blind=1, big_blind=2)
        seen = _Announcements()
        players = [
            _WatchedPlayer(make_random_legal_bot(f"p{i}", rng=random.Random(rng.random())), seen)
            for i in range(num_players)
        ]
        table = Table(config, players, rng=random.Random(rng.random()), on_hand_started=seen.hand_started,
                      on_street_dealt=seen.street_dealt, on_action_applied=seen.action_applied)
        table.stacks = [rng.randint(1, 300) for _ in range(num_players)]
        table.play_hand()
        assert seen.due is None, "the hand ended with a player still announced"
        decisions += len(seen.asked)
    assert decisions > 150  # the check was not vacuous
