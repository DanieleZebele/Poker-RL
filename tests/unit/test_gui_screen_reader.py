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


def test_a_seat_under_a_reaction_keeps_the_state_it_had():
    from pokerlab.gui.screen_reader import keep_through_reactions

    last = {}
    assert keep_through_reactions({0: "in_gioco", 3: "in_gioco", 4: "fuori"}, last) == {
        0: "in_gioco", 3: "in_gioco", 4: "fuori"}
    # a reaction over seat 3 (in the hand) and over seat 4 (folded): nothing changes
    assert keep_through_reactions({0: "in_gioco", 3: "reazione", 4: "reazione"}, last) == {
        0: "in_gioco", 3: "in_gioco", 4: "fuori"}
    assert keep_through_reactions({3: "fuori"}, last) == {3: "fuori"}  # gone: now it folded
    assert keep_through_reactions({5: "reazione"}, last) == {5: "fuori"}  # never seen: seated


def test_an_opponent_out_with_no_stack_written_is_an_empty_seat():
    from pokerlab.gui.screen_reader import empty_by_stack

    seats = {0: "fuori", 1: "fuori", 2: "fuori", 3: "in_gioco", 4: "fuori", 5: "sit_out"}
    stacks = {0: 0.0, 1: 0.0, 2: 45.5, 3: 0.0, 5: 0.0}  # 4: not readable
    assert empty_by_stack(seats, stacks) == {
        0: "fuori",  # you are always at the table
        1: "libero",  # bare table, no stack: nobody there
        2: "fuori",  # folded, stack written
        3: "in_gioco",  # all-in, cards in front
        4: "fuori",  # stack not readable: nothing changes
        5: "sit_out",
    }


def test_a_card_misread_for_a_moment_is_not_a_new_hand():
    misread = diff_reading(reading(["Qh", "Kd"], []), reading(["Th", "Kd"], []), ["Qh", "Kd"])
    assert misread.hole == ["Th", "Kd"] and not misread.new_hand
    back = diff_reading(reading(["Th", "Kd"], []), reading(["Qh", "Kd"], []), ["Th", "Kd"])
    assert back.hole == ["Qh", "Kd"] and not back.new_hand


def test_a_misread_card_moves_no_statistics(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        _play_a_hand_then_move_the_button(frame)
        utg = f"chair{chair(6, 3)}"
        assert frame.stats.hands(utg) == 1
        for hole in (["Qh", "Kd"], ["Qh", "Kd"], ["Th", "Kd"], ["Qh", "Kd"], ["Th", "Kd"], ["Qh", "Kd"]):
            seen = full_reading(_bets(), board=[], dealer=1, out={4})
            seen.hole = hole
            frame.apply_reading(seen)
        assert frame.stats.hands(utg) == 1
    finally:
        frame.destroy()
        gc.collect()


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
        assert abs(chair * 360 / CHAIRS - seat * 60) <= 360 / CHAIRS / 2


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
        raised = {**blinds, 3: 3.0}  # UTG raises to 3 BB (300 chips), seat 4 folds
        assert frame.apply_reading(full_reading(raised, out={4}, board=[]))
        assert [(a.action_type, a.amount) for a in frame.script] == [(ActionType.RAISE, 300), (ActionType.FOLD, 0)]
        status = frame.screen_var.get()
        assert "+ UTG raise" in status and "tocca a" in status and "ATTENZIONE" not in status
        from pokerlab.gui.spot_table import chair_for_client_seat as chair
        from pokerlab.gui.spot_view import BOX_BG, FOLDED_BG
        assert frame.chair_ui[chair(6, 4)].box.cget("bg") == FOLDED_BG  # folded: dark
        assert frame.chair_ui[chair(6, 4)].info.cget("bg") == FOLDED_BG
        assert frame.chair_ui[chair(6, 3)].box.cget("bg") == BOX_BG

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


def test_a_fold_the_screen_keeps_contradicting_is_taken_back(app):
    """Seat 4 is read out for one frame and gets a FOLD; then it is shown holding its cards
    again. After `HEAL_READINGS` readings the fold goes, and the rebuild waits on it."""
    from pokerlab.gui.spot_view import HEAL_READINGS, SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        frame.apply_reading(full_reading(_bets(), board=[]))
        frame.apply_reading(full_reading(_bets(s3=3.0), out={4}, board=[]))  # a misread frame
        assert [a.action_type for a in frame.script] == [ActionType.RAISE, ActionType.FOLD]
        for _ in range(HEAL_READINGS - 1):
            frame.apply_reading(full_reading(_bets(s3=3.0), board=[]))
            assert len(frame.script) == 2  # not yet: one frame proves nothing either
        frame.apply_reading(full_reading(_bets(s3=3.0), board=[]))
        assert [a.action_type for a in frame.script] == [ActionType.RAISE]
        assert "tolto" in frame.screen_var.get() and "ha ancora le carte" in frame.screen_var.get()
        assert frame.state.to_act == frame.layout.seat_of(chair_of_client(4))  # lit up again
    finally:
        frame.destroy()
        gc.collect()


def test_a_hand_rebuilt_as_over_says_so(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        frame.apply_reading(full_reading(_bets(), board=[]))
        frame.apply_reading(full_reading(_bets(), out={0, 1, 3, 4, 5}, board=[]))  # all fold to the BB
        assert frame.state.finished and "mano conclusa" in frame.screen_var.get()
    finally:
        frame.destroy()
        gc.collect()


def test_while_the_screen_is_read_every_change_is_logged(app, monkeypatch, tmp_path):
    from pokerlab.gui import spot_view

    log = tmp_path / "spot_screen.log"
    monkeypatch.setattr(spot_view, "SCREEN_LOG", log)
    frame = spot_view.SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        frame.apply_reading(full_reading(_bets(), board=[]))
        assert not log.exists()  # not reading the screen (tests): nothing written
        frame._reader = object()
        frame.apply_reading(full_reading(_bets(s3=3.0), board=[]))
        frame.apply_reading(full_reading(_bets(s3=3.0), board=[]))  # nothing changed
        lines = log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 and "script[1]=raise300" in lines[0] and "tocca seat" in lines[0]
    finally:
        frame._reader = None
        frame.destroy()
        gc.collect()


def chair_of_client(seat):
    from pokerlab.gui.spot_table import chair_for_client_seat

    return chair_for_client_seat(6, seat)


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
        info = frame.chair_ui[chair(6, 3)].info.cget("text")
        assert "statistiche su 1 mani" in info and "VPIP 100% (1)" in info and "PFR 100% (1)" in info
        assert "WTSD -" in info  # no chance yet: no rate
    finally:
        frame.destroy()
        gc.collect()


def test_the_button_and_your_new_cards_read_apart_are_one_hand(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        _play_a_hand_then_move_the_button(frame)
        utg = f"chair{chair(6, 3)}"
        assert frame.stats.hands(utg) == 1
        # Dealing: the players read "out" before their cards arrive, then yours come.
        frame.apply_reading(full_reading(_bets(), board=[], dealer=1, out={3, 4, 5}))
        dealt = full_reading(_bets(), board=[], dealer=1)
        dealt.hole = ["Ah", "Kd"]
        frame.apply_reading(dealt)
        assert frame.stats.hands(utg) == 1  # the same deal, not a second hand
        assert frame.script == []

        frame.apply_reading(full_reading(_bets(), board=[], dealer=1, out={4}))
        frame.apply_reading(full_reading(_bets(), board=[], dealer=2))  # the next hand
        assert frame.stats.hands(utg) == 2
    finally:
        frame.destroy()
        gc.collect()


def test_the_policy_can_be_shown_every_opponent_as_new(app):
    from pokerlab.gui.spot import replay
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        _play_a_hand_then_move_the_button(frame)
        assert frame.build_spot().seat_stats
        frame.fresh_opponents_var.set(True)
        frame._fresh_opponents_toggled()
        assert frame.build_spot().seat_stats == {}
        assert not dict(replay(frame.build_spot()).observation.seat_stats or {})
        assert frame.stats.hands(f"chair{chair(6, 3)}") == 1  # still kept, and shown
        assert "VPIP" in frame.chair_ui[chair(6, 3)].info.cget("text")
        assert app.spot_fresh_opponents  # the screen reopens with it
    finally:
        app.spot_fresh_opponents = False
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
        assert all(frame.stats.hands(f"chair{c}") == 0 for c in range(8))
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
        frame.script = [call, call, check, Action(ActionType.BET, 200), fold, fold]
        frame._record_finished_hand()
        # the flop was seen by all three (the board is only implied by the flop actions)
        assert [frame.stats.rates(i)["wtsd"] for i in ids] == [(0, 1)] * 3
    finally:
        frame.destroy()
        gc.collect()


def test_at_eight_max_each_client_seat_is_its_own_chair():
    from pokerlab.gui.spot_table import CHAIRS, chair_for_client_seat

    assert [chair_for_client_seat(8, seat) for seat in range(8)] == list(range(CHAIRS))


def test_the_reader_reads_the_zones_of_the_table_size_it_is_given(tmp_path):
    pytest.importorskip("numpy")
    from pokerlab.gui import screen_reader

    reader = screen_reader.ScreenReader(crops_dir=tmp_path)
    assert reader.players == 6
    zones = reader._zones_read()
    assert "player_6_5" in zones and not any(name.endswith("_8_0") for name in zones)
    reader.players = 8
    zones = reader._zones_read()
    assert {"dealer_8_7", "player_8_7", "bet_8_7", "stack_8_7"} <= zones
    assert not any("_6_" in name for name in zones)


def test_the_spot_screen_can_be_switched_to_an_eight_seat_client(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        assert frame.table_size == 6
        states = {seat: "in_gioco" for seat in range(8)}
        assert frame.set_table_size(8) and not frame.set_table_size(8)
        assert app.spot_table_size == 8
        frame.apply_reading(seats_reading(states, dealer=7))
        assert frame.layout.chairs == set(range(8)) and frame.layout.dealer == 7
        with pytest.raises(ValueError):
            SpotFrame(app, table_size=9)
        reopened = SpotFrame(app)  # the choice is remembered by the app
        try:
            assert reopened.table_size == 8
        finally:
            reopened.destroy()
    finally:
        frame.destroy()
        if hasattr(app, "spot_table_size"):
            del app.spot_table_size
        gc.collect()


def test_a_player_sitting_out_is_seated_when_a_blind_is_in_front_of_them():
    from pokerlab.gui.screen_reader import seated_seats, sit_out_blinds

    states = {0: "in_gioco", 1: "sit_out", 2: "in_gioco", 3: "sit_out", 4: "libero"}
    bets = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.0}
    assert sit_out_blinds(states, bets) == {1}
    assert seated_seats(states, bets) == {0, 1, 2}
    assert seated_seats(states) == {0, 2}  # no bets read: sitting out stays out


def test_a_sit_out_small_blind_takes_the_blind_and_then_folds(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        # client seat 0 (you) is the button, seat 1 sits out but posts the SB
        states = {0: "in_gioco", 1: "sit_out", 2: "in_gioco", 3: "in_gioco", 4: "libero", 5: "libero"}
        frame.apply_reading(ScreenReading(dealer=0, seats=states, bets={s: 0.0 for s in range(6)}, board=[]))
        assert chair(6, 1) not in frame.layout.chairs  # no blind seen yet

        blinds = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
        frame.apply_reading(ScreenReading(dealer=0, seats=states, bets=blinds, board=[]))
        assert chair(6, 1) in frame.layout.chairs  # seated as soon as the blind shows
        assert frame.layout.seat_of(chair(6, 1)) == 1 and frame.layout.seat_of(chair(6, 2)) == 2

        called = {**blinds, 3: 1.0, 0: 1.0}  # UTG and the button call; the SB is out
        frame.apply_reading(ScreenReading(dealer=0, seats=states, bets=called, board=[]))
        assert [a.action_type for a in frame.script] == [ActionType.CALL, ActionType.CALL, ActionType.FOLD]
    finally:
        frame.destroy()
        gc.collect()


def test_the_ante_is_the_preflop_pot_shared_by_every_seat_dealt_in():
    from pokerlab.gui.screen_reader import ante_from_pot

    seats = {0: "in_gioco", 1: "in_gioco", 2: "sit_out", 3: "fuori", 4: "libero", 5: "in_gioco"}
    assert ante_from_pot(seats, 0.5, []) == pytest.approx(0.1)  # five seats not empty
    assert ante_from_pot(seats, 0.0, []) == 0.0  # no pot written: no ante
    assert ante_from_pot(seats, 0.5, ["Ah", "Kd", "2c"]) is None  # after the flop the pot holds bets too
    assert ante_from_pot(seats, None, []) is None and ante_from_pot(seats, 0.5, None) is None


def test_with_an_ante_a_player_sitting_out_is_dealt_in():
    from pokerlab.gui.screen_reader import seated_seats

    states = {0: "in_gioco", 1: "sit_out", 2: "in_gioco", 3: "libero"}
    assert seated_seats(states) == {0, 2}
    assert seated_seats(states, antes=True) == {0, 1, 2}


def test_the_spot_reads_the_ante_off_the_pot_and_counts_it_everywhere(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        # six players, an ante of 0,1 BB each: 0,6 BB in the pot before anyone acts
        blinds = {0: 0.0, 1: 0.5, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
        reading = full_reading(blinds, pot=0.6, board=[])
        reading.stacks = {0: 99.9, 1: 59.4, 2: 12.9, 3: 87.4, 4: 40.0, 5: 24.9}
        frame.apply_reading(reading)
        assert frame.ante_var.get() == "0,1"
        assert frame.build_spot().ante == 10  # chips at a big blind of 100
        # each stack at the start of the hand: what is shown, plus what is in front, plus the ante
        assert frame.stack_vars[chair(6, 0)].get() == "100"
        assert frame.stack_vars[chair(6, 1)].get() == "60"
        assert frame.stack_vars[chair(6, 2)].get() == "14"
        assert frame.state.pot == 60 + 50 + 100  # the engine's pot holds the antes

        raised = {**blinds, 3: 3.0}
        frame.apply_reading(full_reading(raised, pot=0.6, out={4}, board=[]))
        assert [(a.action_type, a.amount) for a in frame.script] == [(ActionType.RAISE, 300), (ActionType.FOLD, 0)]
        assert "ATTENZIONE" not in frame.screen_var.get()  # pot read = antes + bets in front
        assert frame.ante_var.get() == "0,1"  # mid-hand the ante is left alone
    finally:
        frame.destroy()
        gc.collect()


def test_a_player_replaced_mid_hand_leaves_nothing_to_the_newcomer(app):
    """The seat is read empty only between two deals: the player stood up during a hand
    and someone else sat down before the next. The newcomer starts blank -- neither the
    old player's numbers nor the hand the old player was in are theirs."""
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        _play_a_hand_then_move_the_button(frame)
        utg = f"chair{chair(6, 3)}"
        assert frame.stats.hands(utg) == 1

        # the second hand (button on seat 1: blinds on 2 and 3): seat 4 raises, then seat 3,
        # the big blind, leaves and is replaced, all before the next deal
        blinds = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0, 4: 0.0, 5: 0.0}
        raised = {**blinds, 4: 3.0}
        frame.apply_reading(full_reading(raised, board=[], dealer=1))
        assert frame.script  # the raise was rebuilt
        empty = full_reading(raised, board=[], dealer=1)
        empty.seats[3] = "libero"
        frame.apply_reading(empty)
        assert frame.stats.hands(utg) == 1  # one empty reading is not enough: a misread frame
        frame.apply_reading(empty)
        assert frame.stats.hands(utg) == 0  # two in a row: gone
        frame.apply_reading(full_reading(raised, board=[], dealer=1))  # the newcomer
        frame.apply_reading(full_reading(_bets(), board=[], dealer=2))  # the next deal
        assert frame.stats.hands(utg) == 0  # the hand the old player was in is not the newcomer's
        assert frame.stats.hands(f"chair{chair(6, 4)}") == 2  # the others count it
    finally:
        frame.destroy()
        gc.collect()


def test_a_player_who_vanishes_mid_hand_has_folded(app):
    """Seat 4 is read empty in the middle of the hand (left the table, disconnected):
    when its turn comes it folds, and the rebuild goes on to the next player."""
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        frame.apply_reading(full_reading(_bets(), board=[]))
        raised = full_reading(_bets(s3=3.0), board=[])
        raised.seats[4] = "libero"  # gone, after UTG raised
        frame.apply_reading(raised)
        assert [(a.action_type, a.amount) for a in frame.script] == [(ActionType.RAISE, 300), (ActionType.FOLD, 0)]
        assert chair(6, 4) in frame.layout.chairs  # still seated until the next hand
        assert "uscito" in frame.chair_ui[chair(6, 4)].info.cget("text")
    finally:
        frame.destroy()
        gc.collect()


def test_when_the_dealer_has_gone_the_button_passes_to_the_player_before(app):
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        seats = {s: "in_gioco" for s in range(6)}
        frame.apply_reading(seats_reading(seats, dealer=3))
        assert frame.layout.dealer == chair(6, 3)

        # the dealer leaves; the next hand starts with the button still on that empty seat
        gone = {**seats, 3: "libero"}
        frame.apply_reading(seats_reading(gone, dealer=3, hole=["Ah", "Kd"]))
        assert chair(6, 3) not in frame.layout.chairs  # nobody seated there for the button
        assert frame.layout.dealer == chair(6, 2)  # the player before has it

        # a reading that first sees the button on an empty seat does the same
        frame.apply_reading(seats_reading({**gone, 4: "libero"}, dealer=4, hole=["2c", "7d"]))
        assert chair(6, 4) not in frame.layout.chairs and frame.layout.dealer == chair(6, 2)
    finally:
        frame.destroy()
        gc.collect()


def test_your_check_is_seen_when_your_bar_goes_with_nothing_in_front(app):
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        frame.apply_reading(full_reading(_bets(), board=[]))  # the table and the button (you)
        call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
        frame.script = [call] * 5 + [check] + [check] * 5  # limped, then checked round to you
        flop = ["Ks", "8d", "3h"]
        nothing = {s: 0.0 for s in range(6)}
        waiting = full_reading(nothing, board=flop)
        waiting.my_turn = True
        frame.apply_reading(waiting)
        assert len(frame.script) == 11  # your turn: nothing to add yet
        gone = full_reading(nothing, board=flop)
        gone.my_turn = False
        frame.apply_reading(gone)
        assert frame.script[11:] == [check]  # your bar went, nothing in front: you checked
    finally:
        frame.destroy()
        gc.collect()


def test_the_opponents_are_followed_after_you_fold(app):
    """You fold under the gun; the others play on to the flop and are still rebuilt --
    the big blind's call that closes the preflop is not read again as a flop bet."""
    from pokerlab.gui.spot_view import SpotFrame

    def reading(bets, out=(), board=()):
        seats = {s: ("fuori" if s in out else "in_gioco") for s in range(6)}
        return ScreenReading(hole=[], board=list(board), dealer=3, seats=seats,
                             bets={s: bets.get(s, 0.0) for s in range(6)}, pot=0.0)

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        blinds = {4: 0.5, 5: 1.0}  # button on client seat 3: you (0) are under the gun
        frame.apply_reading(reading(blinds))
        frame.apply_reading(reading(blinds, out={0}))
        frame.apply_reading(reading({**blinds, 1: 3.0}, out={0}))
        frame.apply_reading(reading({**blinds, 1: 3.0}, out={0, 2, 3, 4}))
        frame.apply_reading(reading({1: 3.0, 5: 3.0}, out={0, 2, 3, 4}))  # the BB calls
        flop = ["Ks", "8d", "3h"]
        frame.apply_reading(reading({}, out={0, 2, 3, 4}, board=flop))
        frame.apply_reading(reading({5: 2.0}, out={0, 2, 3, 4}, board=flop))
        assert [(a.action_type, a.amount) for a in frame.script] == [
            (ActionType.FOLD, 0), (ActionType.RAISE, 300), (ActionType.FOLD, 0), (ActionType.FOLD, 0),
            (ActionType.FOLD, 0), (ActionType.CALL, 0), (ActionType.BET, 200),
        ]
    finally:
        frame.destroy()
        gc.collect()


def test_every_player_s_statistics_are_always_on_screen(app):
    """Before the first hand, on your own chair, and with the rebuilt hand over: the
    statistics stay under every seated chair."""
    from pokerlab.gui.spot_table import USER_CHAIR
    from pokerlab.gui.spot_table import chair_for_client_seat as chair
    from pokerlab.gui.spot_view import SpotFrame

    frame = SpotFrame(app)
    frame._consult = lambda state, spot: None
    try:
        frame.apply_reading(full_reading(_bets(), board=[]))
        for c in frame.layout.chairs:
            assert "nessuna mano registrata" in frame.chair_ui[c].info.cget("text")

        _play_a_hand_then_move_the_button(frame)
        utg = frame.chair_ui[chair(6, 3)].info.cget("text")
        assert "statistiche su 1 mani" in utg
        assert "statistiche su 1 mani" in frame.chair_ui[USER_CHAIR].info.cget("text")  # yours too

        fold = Action(ActionType.FOLD)
        frame.script = [fold] * 5  # everyone folds to the big blind: the hand is over
        frame.refresh()
        assert frame.state.finished
        assert "statistiche su 1 mani" in frame.chair_ui[chair(6, 3)].info.cget("text")
    finally:
        frame.destroy()
        gc.collect()
