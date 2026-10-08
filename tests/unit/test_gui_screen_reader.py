"""Reading the hand and the board off the screen into the spot screen."""

import gc

import pytest

from pokerlab.engine.actions import Action, ActionType
from pokerlab.gui.screen_reader import ScreenReading, diff_reading


def reading(hole=None, board=None):
    return ScreenReading(hole=hole, board=board)


def test_an_unchanged_screen_changes_nothing():
    first = reading(["Ah", "Kd"], ["2c", "7d", "Th"])
    change = diff_reading(first, reading(["Ah", "Kd"], ["2c", "7d", "Th"]), ["Ah", "Kd"])
    assert not change.any


def test_a_zone_not_known_now_changes_nothing():
    change = diff_reading(reading(["Ah", "Kd"], []), reading(None, None), ["Ah", "Kd"])
    assert not change.any


def test_a_new_street_changes_only_the_board():
    change = diff_reading(reading(["Ah", "Kd"], ["2c", "7d", "Th"]),
                          reading(["Ah", "Kd"], ["2c", "7d", "Th", "Js"]), ["Ah", "Kd"])
    assert change.board == ["2c", "7d", "Th", "Js"] and change.hole is None and not change.new_hand


def test_a_new_hand_is_seen_across_a_fold_but_a_flicker_is_not_one():
    folded = diff_reading(reading(["Ah", "Kd"], []), reading([], []), ["Ah", "Kd"])
    assert folded.hole == [] and not folded.new_hand
    dealt = diff_reading(reading([], []), reading(["9s", "9c"], []), ["Ah", "Kd"])
    assert dealt.hole == ["9s", "9c"] and dealt.new_hand
    flicker = diff_reading(reading([], []), reading(["Ah", "Kd"], []), ["Ah", "Kd"])
    assert flicker.hole == ["Ah", "Kd"] and not flicker.new_hand


@pytest.fixture
def frame(app, monkeypatch):
    from pokerlab.gui.spot_view import SpotFrame

    spot = SpotFrame(app)
    spot.consulted = []
    monkeypatch.setattr(spot, "_consult", lambda state, s: spot.consulted.append(s.hole_cards))
    spot.add_player(3)
    spot.add_player(6)
    yield spot
    spot.destroy()
    gc.collect()


def cards(frame_cards):
    from pokerlab.gui.spot import format_card

    return [format_card(c) if c else None for c in frame_cards]


def test_the_reading_fills_the_hand_and_the_board_and_the_models_answer(frame):
    assert frame.apply_reading(reading(["Ah", "Kd"], []))
    assert cards(frame.hole) == ["Ah", "Kd"] and frame.board == []
    assert frame.consulted  # your turn, both cards in: the models were asked
    assert "A♥" in frame.screen_var.get()


def test_a_manual_correction_sticks_while_the_screen_does_not_change(frame):
    from pokerlab.gui.spot import parse_card

    frame.apply_reading(reading(["Ah", "Kd"], []))
    frame.hole[1] = parse_card("Ks")  # the user fixes a misread
    assert not frame.apply_reading(reading(["Ah", "Kd"], []))
    assert cards(frame.hole) == ["Ah", "Ks"]


def test_a_new_hand_clears_the_old_actions_but_a_new_street_does_not(frame):
    frame.apply_reading(reading(["Ah", "Kd"], []))
    frame._append(Action(ActionType.CALL))
    assert frame.script
    frame.apply_reading(reading(["Ah", "Kd"], ["2c", "7d", "Th"]))
    assert frame.script and cards(frame.board) == ["2c", "7d", "Th"]
    frame.apply_reading(reading(["9s", "9c"], []))
    assert frame.script == [] and frame.board == []


def test_problems_are_shown_and_nothing_unknown_is_applied(frame):
    frame.apply_reading(reading(["Ah", "Kd"], []))
    odd = ScreenReading(hole=None, board=None, problems=["seme non riconosciuto in mano"])
    assert not frame.apply_reading(odd)
    assert cards(frame.hole) == ["Ah", "Kd"]
    assert "seme non riconosciuto" in frame.screen_var.get()


def test_the_reader_reads_both_zones_and_refuses_a_card_in_both(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from pathlib import Path

    from pokerlab.gui import screen_reader
    from pokerlab.vision import capture, recognize
    from pokerlab.vision.regions import BOARD, HOLE_CARDS, Region, RegionConfig

    colours = {"s": (49, 53, 52), "h": (52, 68, 216), "d": (204, 106, 51), "c": (41, 179, 46)}

    def crop(hand, width, height, step):
        image = np.full((height, width, 3), (65, 174, 116), np.uint8)
        for i, card in enumerate(hand):
            x = 3 + round(i * step)
            cv2.rectangle(image, (x, 4), (x + 86, height - 4), colours[card[1]], -1)
            cv2.putText(image, card[0], (x + 6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.circle(image, (x + 53, height - 30), 16, (255, 255, 255), -1)
        return image

    regions = RegionConfig()
    regions.set(BOARD, Region(0, 0, 486, 138))
    regions.set(HOLE_CARDS, Region(0, 200, 170, 98))
    screen = {}
    examples = [recognize.LabelledCrop(Path("b.png"), crop(["As", "Kh", "Qd"], 486, 138, 97.2), BOARD,
                                       ["As", "Kh", "Qd"])]
    monkeypatch.setattr(screen_reader, "load_regions", lambda: regions)
    monkeypatch.setattr(recognize, "load_dataset", lambda folder=None: examples)
    monkeypatch.setattr(capture, "grab_regions", lambda regions: [screen[r] for r in regions])
    reader = screen_reader.ScreenReader(crops_dir=tmp_path)

    screen[Region(0, 0, 486, 138)] = crop(["Qc", "As"], 486, 138, 97.2)
    screen[Region(0, 200, 170, 98)] = crop(["Kd", "Ah"], 170, 98, 80)
    half = reader.read()  # two board cards: the flop half dealt, not a board
    assert half.board is None and half.hole == ["Kd", "Ah"] and "a metà" in half.problems[0]
    screen[Region(0, 0, 486, 138)] = crop(["Qc", "As", "Kh"], 486, 138, 97.2)
    got = reader.read()
    assert got.hole == ["Kd", "Ah"] and got.board == ["Qc", "As", "Kh"] and not got.problems

    screen[Region(0, 200, 170, 98)] = crop(["Kh", "Ah"], 170, 98, 80)  # Kh in both
    got = reader.read()
    assert got.hole is None and got.board is None and "stessa carta" in got.problems[0]


def dealer_reading(seat, hole=None):
    return ScreenReading(hole=hole, dealer=seat)


def test_the_dealer_changes_only_when_the_button_moves():
    assert diff_reading(None, dealer_reading(3), None, None).dealer == 3
    assert diff_reading(dealer_reading(3), dealer_reading(3), None, 3).dealer is None
    # gone for a moment between hands, back on the same seat: not a move
    assert diff_reading(dealer_reading(None), dealer_reading(3), None, 3).dealer is None
    assert diff_reading(dealer_reading(3), dealer_reading(None), None, 3).dealer is None
    assert diff_reading(dealer_reading(3), dealer_reading(4), None, 3).dealer == 4


def test_client_seats_map_to_the_nearest_chairs():
    from pokerlab.gui.spot_table import CHAIRS, chair_for_client_seat

    chairs = [chair_for_client_seat(6, seat) for seat in range(6)]
    assert chairs[0] == 0 and len(set(chairs)) == 6
    for seat, chair in enumerate(chairs):  # within half a chair of the seat's angle
        assert abs(chair * 360 / CHAIRS - seat * 60) <= 20


def test_the_button_is_moved_to_its_chair_seating_a_player_if_needed(frame):
    from pokerlab.gui.spot_table import chair_for_client_seat

    frame._append(Action(ActionType.CALL))
    chair = chair_for_client_seat(6, 4)  # chair 6, already seated by the fixture
    assert frame.apply_reading(dealer_reading(4))
    assert frame.layout.dealer == chair and frame.script == []  # a new hand
    assert "dealer posto 4" in frame.screen_var.get()

    empty = chair_for_client_seat(6, 1)  # chair 2: nobody there yet
    assert empty not in frame.layout.chairs
    frame.apply_reading(dealer_reading(1))
    assert empty in frame.layout.chairs and frame.layout.dealer == empty

    frame.set_dealer(0)  # a manual correction sticks while the button stays put
    assert not frame.apply_reading(dealer_reading(1))
    assert frame.layout.dealer == 0


def seats_reading(seats, dealer=None, hole=None):
    return ScreenReading(hole=hole, dealer=dealer, seats=seats)


def test_who_is_seated_leaves_out_empty_seats_and_players_sitting_out():
    from pokerlab.gui.screen_reader import seated_seats

    states = {0: "in_gioco", 1: "libero", 2: "fuori", 3: "sit_out", 4: "in_gioco"}
    assert seated_seats(states) == {0, 2, 4}


def test_the_screen_seats_the_table_and_marks_who_folded(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import OUT_OF_HAND, SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        states = {0: "in_gioco", 1: "libero", 2: "in_gioco", 3: "fuori", 4: "sit_out", 5: "in_gioco"}
        assert frame.apply_reading(seats_reading(states, dealer=2))
        assert frame.layout.chairs == {chair(6, s) for s in (0, 2, 3, 5)}
        assert frame.layout.dealer == chair(6, 2)
        folded = frame.chair_ui[chair(6, 3)]
        assert "fold" in folded.info.cget("text") and folded.box.cget("highlightbackground") == OUT_OF_HAND
        assert "2 fold" not in frame.screen_var.get() and "1 fold" in frame.screen_var.get()
    finally:
        frame.destroy()
        gc.collect()


def test_seating_changes_only_between_hands_but_folds_show_at_once(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        start = {0: "in_gioco", 1: "libero", 2: "in_gioco", 3: "in_gioco", 4: "libero", 5: "in_gioco"}
        frame.apply_reading(seats_reading(start, dealer=0))
        frame._append(Action(ActionType.CALL))
        script = list(frame.script)

        mid = {**start, 1: "fuori", 3: "fuori"}  # someone sat down, someone folded
        frame.apply_reading(seats_reading(mid, dealer=0))
        assert chair(6, 1) not in frame.layout.chairs  # not seated mid-hand
        assert frame.script == script  # the actions being entered survive
        assert "fold" in frame.chair_ui[chair(6, 3)].info.cget("text")

        frame.apply_reading(seats_reading(mid, dealer=2))  # the button moved: a new hand
        assert chair(6, 1) in frame.layout.chairs and frame.layout.dealer == chair(6, 2)
        assert frame.script == []
    finally:
        frame.destroy()
        gc.collect()


def test_the_reader_ignores_a_dealer_zone_too_big_to_be_the_button(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from pokerlab.gui import screen_reader
    from pokerlab.vision import capture
    from pokerlab.vision.regions import Region, RegionConfig, dealer_region_name

    def gold(width, height):
        image = np.full((height, width, 3), (65, 120, 40), np.uint8)
        cv2.circle(image, (width // 2, height // 2), min(width, height) // 2 - 2, (40, 200, 235), -1)
        return image

    regions = RegionConfig()
    regions.set(dealer_region_name(6, 1), Region(0, 0, 38, 37))
    regions.set(dealer_region_name(6, 5), Region(100, 0, 149, 111))  # a player box with a gold chip
    screen = {Region(0, 0, 38, 37): np.full((37, 38, 3), (65, 120, 40), np.uint8),
              Region(100, 0, 149, 111): gold(149, 111)}
    monkeypatch.setattr(screen_reader, "load_regions", lambda: regions)
    monkeypatch.setattr(capture, "grab_regions", lambda rs: [screen[r] for r in rs])
    got = screen_reader.ScreenReader(crops_dir=tmp_path).read()
    assert got.dealer is None  # not the gold of the oversized zone
    assert any("troppo grande" in p for p in got.problems)


def full_reading(bets, pot=0.0, out=(), board=None, dealer=0):
    seats = {s: ("fuori" if s in out else "in_gioco") for s in range(6)}
    return ScreenReading(hole=None, board=board, dealer=dealer, seats=seats, bets=bets, pot=pot)


def test_the_models_page_rebuilds_the_actions_from_the_bets(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        # client seat 0 (you) is the button; 1 SB, 2 BB, 3 UTG, 4, 5
        blinds = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
        frame.apply_reading(full_reading(blinds, board=[]))
        assert frame.script == []
        raised = {**blinds, 3: 3.0}  # UTG raises to 3 BB (6 chips), seat 4 folds
        assert frame.apply_reading(full_reading(raised, out={4}, board=[]))
        assert [(a.action_type, a.amount) for a in frame.script] == [(ActionType.RAISE, 6), (ActionType.FOLD, 0)]
        status = frame.screen_var.get()
        assert "+ UTG raise" in status and "tocca a" in status and "ATTENZIONE" not in status

        wrong_pot = full_reading(raised, pot=5.0, out={4}, board=[])
        frame.apply_reading(wrong_pot)
        assert "ATTENZIONE piatto" in frame.screen_var.get()
    finally:
        frame.destroy()
        gc.collect()


def test_stacks_are_set_at_a_new_hand_counting_the_blinds_in_front(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        blinds = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
        stacks = {0: 100.0, 1: 59.5, 2: 13.0, 3: 87.5, 4: 40.0, 5: 25.0}
        reading = full_reading(blinds, board=[])
        reading.stacks = stacks
        frame.apply_reading(reading)
        assert frame.stack_vars[chair(6, 1)].get() == "60"  # 59,5 behind + 0,5 posted
        assert frame.stack_vars[chair(6, 2)].get() == "14"  # 13 + the big blind
        assert frame.stack_vars[chair(6, 3)].get() == "87,5"
        later = full_reading({**blinds, 3: 3.0}, board=[])
        later.stacks = {**stacks, 3: 84.5}  # mid-hand: UTG raised, its stack fell
        frame.apply_reading(later)
        assert frame.stack_vars[chair(6, 3)].get() == "87,5"  # not rewritten mid-hand
    finally:
        frame.destroy()
        gc.collect()


def test_each_chair_shows_what_is_left_and_flags_a_screen_that_disagrees(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        blinds = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
        stacks = {0: 100.0, 1: 59.5, 2: 13.0, 3: 87.5, 4: 40.0, 5: 25.0}
        start = full_reading(blinds, board=[])
        start.stacks = stacks
        frame.apply_reading(start)
        raised = full_reading({**blinds, 3: 3.0}, board=[])  # UTG raises to 3 BB
        raised.stacks = {**stacks, 3: 84.5}
        frame.apply_reading(raised)
        utg = frame.chair_ui[chair(6, 3)].info.cget("text")
        assert "resta 84,5 BB" in utg and "schermo" not in utg
        assert frame.stack_vars[chair(6, 3)].get() == "87,5"  # the starting stack stays
        wrong = full_reading({**blinds, 3: 3.0}, board=[])
        wrong.stacks = {**stacks, 3: 80.0}  # the screen says otherwise
        frame.apply_reading(wrong)
        assert "≠ schermo 80 BB" in frame.chair_ui[chair(6, 3)].info.cget("text")
    finally:
        frame.destroy()
        gc.collect()


def _bets(**raised):
    blinds = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    return {**blinds, **{int(seat[1:]): amount for seat, amount in raised.items()}}


def _play_a_hand_then_move_the_button(frame):
    """UTG (client seat 3) raises, seat 4 folds; then the button moves: a new hand."""
    frame.apply_reading(full_reading(_bets(), board=[]))
    frame.apply_reading(full_reading(_bets(s3=3.0), out={4}, board=[]))
    assert len(frame.script) == 2
    frame.apply_reading(full_reading(_bets(), board=[], dealer=1))


def test_a_hand_read_off_the_screen_feeds_the_statistics_when_the_next_one_starts(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        _play_a_hand_then_move_the_button(frame)

        utg, folder = f"chair{chair(6, 3)}", f"chair{chair(6, 4)}"
        assert frame.stats.hands(utg) == 1
        assert frame.stats.rates(utg)["vpip"] == (1, 1) and frame.stats.rates(utg)["pfr"] == (1, 1)
        assert frame.stats.rates(folder)["vpip"] == (0, 1)
        assert frame.stats.rates(utg)["wtsd"] == (0, 0)  # nobody saw a flop
        assert "statistiche" in frame.screen_var.get()
        assert "VPIP 100% PFR 100% (1 mani)" in frame.chair_ui[chair(6, 3)].info.cget("text")
    finally:
        frame.destroy()
        gc.collect()


def test_the_models_read_those_statistics_in_the_next_hand(app):
    from pokerlab.gui.spot import replay
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        _play_a_hand_then_move_the_button(frame)

        state = replay(frame.build_spot())
        seen = dict(state.observation.seat_stats)
        raiser = frame.layout.seat_of(chair(6, 3))
        assert raiser in seen and seen[raiser][0] == 1.0  # "statistics supplied"
        assert frame.stats.hands(f"chair{chair(6, 3)}") == 1  # the replay itself recorded nothing
    finally:
        frame.destroy()
        gc.collect()


def test_a_hand_built_by_hand_is_not_counted(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        frame.add_player(3)
        frame._append(Action(ActionType.RAISE, 6))
        frame.set_dealer(3)  # editing the table clears the script without any hand having been seen
        frame._clear()
        assert all(frame.stats.hands(f"chair{c}") == 0 for c in range(9))
    finally:
        frame.destroy()
        gc.collect()


def test_a_seat_that_empties_is_forgotten_and_the_button_clears_the_rest(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        _play_a_hand_then_move_the_button(frame)
        utg = f"chair{chair(6, 3)}"
        assert frame.stats.hands(utg) == 1

        seats = {s: "in_gioco" for s in range(6)}
        seats[3] = "libero"  # the player left; the next hand starts with another dealer
        frame.apply_reading(seats_reading(seats, dealer=2))
        assert frame.stats.hands(utg) == 0
        assert frame.stats.hands(f"chair{chair(6, 4)}") == 1  # the others keep their history

        frame.reset_stats()
        assert frame.stats.hands(f"chair{chair(6, 4)}") == 0
    finally:
        frame.destroy()
        gc.collect()


def _three_handed(frame):
    frame.add_player(3)
    frame.add_player(6)  # chairs 0, 3, 6: seats 0 (BTN), 1 (SB), 2 (BB)
    return [f"chair{c}" for c in (0, 3, 6)]


def test_a_player_who_saw_the_flop_and_never_folded_reached_the_showdown(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        ids = _three_handed(frame)
        call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
        frame.script = [call, call, check] + [check] * 3 * 3  # preflop, then checked down three streets
        frame._record_finished_hand()
        assert [frame.stats.rates(i)["wtsd"] for i in ids] == [(1, 1)] * 3
    finally:
        frame.destroy()
        gc.collect()


def test_a_hand_won_by_a_bet_on_the_flop_is_no_showdown_for_anyone(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        ids = _three_handed(frame)
        call, check, fold = Action(ActionType.CALL), Action(ActionType.CHECK), Action(ActionType.FOLD)
        frame.script = [call, call, check, Action(ActionType.BET, 4), fold, fold]
        frame._record_finished_hand()
        # the flop was seen by all three (the board is only implied by the flop actions)
        assert [frame.stats.rates(i)["wtsd"] for i in ids] == [(0, 1)] * 3
    finally:
        frame.destroy()
        gc.collect()
