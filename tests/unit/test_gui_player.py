import queue
import threading

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.players.base import Observation, SeatPublicInfo
from pokerlab.players.gui import GuiEvent, GuiPlayer


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

    # The action itself is reported by Table's on_action_applied hook (see
    # ActionReporter), not from here -- a Player cannot see the state after
    # its own action, and the human's move must be drawn the same way a
    # bot's is.
    assert event_queue.empty()


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
