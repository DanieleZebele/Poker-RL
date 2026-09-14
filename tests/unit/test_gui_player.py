import queue
import threading

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.players.base import Observation, SeatPublicInfo
from pokerlab.players.gui import GuiEvent, GuiPlayer, SteppingPlayer
from pokerlab.players.scripted import make_always_call_bot


def make_observation() -> Observation:
    return Observation(
        street=Street.FLOP,
        hole_cards=(Card.parse("Ah"), Card.parse("Kh")),
        community_cards=(),
        pot_size=10,
        current_bet_to_match=0,
        min_raise=2,
        my_seat=0,
        my_stack=200,
        my_current_bet=0,
        seats=(SeatPublicInfo(seat=0, name="Me", stack=200, current_bet=0, status=PlayerStatus.ACTIVE, is_button=False),),
        button_seat=0,
        action_history=(),
    )


def test_act_publishes_a_your_turn_event_with_the_observation_and_legal_actions():
    event_queue: queue.Queue[GuiEvent] = queue.Queue()
    player = GuiPlayer("p0", "Human0", event_queue)
    obs = make_observation()
    legal = [LegalAction(ActionType.FOLD), LegalAction(ActionType.CHECK)]

    player.decisions.put(Action(ActionType.CHECK))  # pre-answer so act() doesn't block the test thread
    result = player.act(obs, legal)

    assert result.action_type == ActionType.CHECK
    your_turn_event = event_queue.get_nowait()
    assert your_turn_event.kind == "your_turn"
    assert your_turn_event.payload == (obs, legal)

    action_taken_event = event_queue.get_nowait()
    assert action_taken_event.kind == "action_taken"
    assert action_taken_event.payload == ("p0", "Human0", obs, Action(ActionType.CHECK))


def test_act_blocks_until_a_decision_is_pushed_from_another_thread():
    event_queue: queue.Queue[GuiEvent] = queue.Queue()
    player = GuiPlayer("p0", "Human0", event_queue)
    obs = make_observation()
    legal = [LegalAction(ActionType.FOLD), LegalAction(ActionType.CALL, 10, 10)]

    results: list[Action] = []

    def act_in_background() -> None:
        results.append(player.act(obs, legal))

    worker = threading.Thread(target=act_in_background)
    worker.start()

    # act() must not have returned yet -- nobody has answered the "your_turn" event.
    worker.join(timeout=0.2)
    assert worker.is_alive()
    assert results == []

    player.decisions.put(Action(ActionType.CALL))
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results == [Action(ActionType.CALL)]


def test_stepping_player_reports_the_action_without_blocking_when_step_mode_is_off():
    event_queue: queue.Queue[GuiEvent] = queue.Queue()
    step_gate: queue.Queue[None] = queue.Queue()
    bot = make_always_call_bot("p1", "Bot1")
    player = SteppingPlayer(bot, event_queue, step_gate, step_mode=lambda: False)
    obs = make_observation()
    legal = [LegalAction(ActionType.FOLD), LegalAction(ActionType.CHECK)]

    action = player.act(obs, legal)

    assert action.action_type == ActionType.CHECK  # always-call bot checks when it can
    event = event_queue.get_nowait()
    assert event.kind == "action_taken"
    assert event.payload == ("p1", "Bot1", obs, action)


def test_stepping_player_blocks_until_step_gate_is_released_when_step_mode_is_on():
    event_queue: queue.Queue[GuiEvent] = queue.Queue()
    step_gate: queue.Queue[None] = queue.Queue()
    bot = make_always_call_bot("p1", "Bot1")
    player = SteppingPlayer(bot, event_queue, step_gate, step_mode=lambda: True)
    obs = make_observation()
    legal = [LegalAction(ActionType.FOLD), LegalAction(ActionType.CHECK)]

    results: list[Action] = []
    worker = threading.Thread(target=lambda: results.append(player.act(obs, legal)))
    worker.start()

    worker.join(timeout=0.2)
    assert worker.is_alive(), "should be blocked on step_gate while step mode is on"
    assert results == []

    step_gate.put(None)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results == [Action(ActionType.CHECK)]
