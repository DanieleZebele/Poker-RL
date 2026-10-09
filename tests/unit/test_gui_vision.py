import gc

import pytest


def test_each_zone_can_be_saved_on_its_own(app, monkeypatch, tmp_path):
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture
    from pokerlab.vision.regions import BOARD, HOLE_CARDS, Region, RegionConfig

    grabbed = []
    monkeypatch.setattr(capture, "grab_region", lambda region: grabbed.append(region) or region)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(HOLE_CARDS, Region(0, 0, 40, 30))
        frame.labeler = None
        frame._save_crops([BOARD])
        assert frame.labeler is None and "board" in frame.vision_var.get()

        frame.regions.set(BOARD, Region(100, 0, 200, 60))
        frame._save_crops([HOLE_CARDS])
        assert frame.labeler.zone == HOLE_CARDS and grabbed == [Region(0, 0, 40, 30)]
    finally:
        frame.destroy()
        gc.collect()


def test_a_crop_is_labelled_left_to_right_and_written_only_on_confirm(app, monkeypatch, tmp_path):
    from pokerlab.gui.spot import parse_card
    from pokerlab.gui.vision_view import CropLabeler
    from pokerlab.vision import capture
    from pokerlab.vision.labels import load_label
    from pokerlab.vision.regions import BOARD

    pngs = []
    monkeypatch.setattr(capture, "save_png", lambda frame, path: pngs.append(path) or path)
    png = tmp_path / "board-x.png"
    finished = []
    labeler = CropLabeler(app, png_path=png, zone=BOARD, frame=object(), on_done=finished.append)
    try:
        assert [str(b["state"]) for b in labeler.slot_buttons] == ["normal"] + ["disabled"] * 4
        assert str(labeler.confirm_button["state"]) == "disabled"
        assert labeler._open(2) is None  # filled in order
        for slot, text in enumerate(["Th", "2c"]):
            labeler.pick(slot, parse_card(text))
        assert str(labeler.confirm_button["state"]) == "disabled"  # two cards is no board
        labeler.confirm()
        assert pngs == [] and finished == []  # nothing written yet
        labeler.pick(2, parse_card("7d"))
        picker = labeler._open(0)
        assert parse_card("2c") in picker._taken and parse_card("Th") not in picker._taken
        picker.destroy()
        assert str(labeler.confirm_button["state"]) == "normal"
        labeler.confirm()
        assert pngs == [png]
        assert load_label(png)["cards"] == ["Th", "2c", "7d"]
        assert finished == [labeler] and labeler.result == ["Th", "2c", "7d"]
    finally:
        if labeler.winfo_exists():
            labeler.destroy()
        gc.collect()


def test_no_cards_visible_is_confirmed_and_cancel_writes_nothing(app, monkeypatch, tmp_path):
    from pokerlab.gui.spot import parse_card
    from pokerlab.gui.vision_view import CropLabeler
    from pokerlab.vision import capture
    from pokerlab.vision.labels import label_path, load_label
    from pokerlab.vision.regions import HOLE_CARDS

    pngs = []
    monkeypatch.setattr(capture, "save_png", lambda frame, path: pngs.append(path) or path)
    empty, cancelled = tmp_path / "hole_cards-a.png", tmp_path / "hole_cards-b.png"
    labeler = CropLabeler(app, png_path=empty, zone=HOLE_CARDS, frame=object())
    labeler.pick(0, parse_card("Ah"))
    labeler.set_empty(True)
    assert labeler.cards == [] and all(str(b["state"]) == "disabled" for b in labeler.slot_buttons)
    labeler.confirm()
    assert load_label(empty)["cards"] == []

    CropLabeler(app, png_path=cancelled, zone=HOLE_CARDS, frame=object()).cancel()
    assert pngs == [empty] and not label_path(cancelled).exists()
    gc.collect()


def test_saving_every_zone_asks_for_each_label_in_turn(app, monkeypatch, tmp_path):
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture
    from pokerlab.vision.labels import CROPS_DIR
    from pokerlab.vision.regions import BOARD, HOLE_CARDS, Region, RegionConfig

    pngs = []
    monkeypatch.setattr(capture, "grab_region", lambda region: region)
    monkeypatch.setattr(capture, "save_png", lambda frame, path: pngs.append(path) or path)
    monkeypatch.setattr("pokerlab.vision.labels.save_label", lambda path, zone, cards: path)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(HOLE_CARDS, Region(0, 0, 40, 30))
        frame.regions.set(BOARD, Region(100, 0, 200, 60))
        frame._save_crops()
        first = frame.labeler
        assert first.png_path.parent == CROPS_DIR and "checkpoints" not in str(first.png_path)
        first.cancel()
        second = frame.labeler
        assert second is not first and {first.zone, second.zone} == {HOLE_CARDS, BOARD}
        second.set_empty(True)
        second.confirm()
        assert pngs == [second.png_path]
        assert "salvati: 1" in frame.vision_var.get()
    finally:
        frame.destroy()
        gc.collect()


def test_the_main_menu_opens_the_vision_screen(app):
    from pokerlab.gui.spot_view import SpotFrame
    from pokerlab.gui.vision_view import VisionFrame

    app.show_vision()
    try:
        assert isinstance(app._content, VisionFrame)
        app.show_spot(read_screen=False)
        assert isinstance(app._content, SpotFrame) and not hasattr(app._content, "_save_crops")
    finally:
        app.show_setup()
        gc.collect()


def test_dealer_zones_are_named_per_table_size_and_seat():
    from pokerlab.vision.regions import dealer_region_name

    assert dealer_region_name(6, 0) == "dealer_6_0" and dealer_region_name(6, 5) == "dealer_6_5"
    for players, seat in ((6, 6), (6, -1), (8, 8), (9, 0), (7, 0)):
        with pytest.raises(ValueError):
            dealer_region_name(players, seat)


def test_saving_the_dealer_writes_every_zone_with_present_or_absent(app, monkeypatch, tmp_path):
    from pokerlab.gui import vision_view
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture, labels
    from pokerlab.vision.labels import load_dealer_label, load_label
    from pokerlab.vision.regions import HOLE_CARDS, Region, RegionConfig, dealer_region_name

    written = []
    monkeypatch.setattr(capture, "grab_region", lambda region: region)
    monkeypatch.setattr(capture, "save_png", lambda frame, path: written.append(path) or path)
    monkeypatch.setattr(labels, "DEALER_DIR", tmp_path)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(HOLE_CARDS, Region(0, 0, 170, 98))
        for seat in (0, 2, 4):
            frame.regions.set(dealer_region_name(6, seat), Region(10 * seat, 300, 30, 30))
        frame._update_dealer_text()
        assert "posti 1, 3, 5" in frame.dealer_var.get()

        labeler = frame._save_dealer()
        assert [seat for seat, *_ in labeler.shots] == [0, 2, 4]  # only dealer zones, in seat order
        labeler.set_present("dealer_6_2", True)
        labeler.set_present("dealer_6_4", True)
        assert str(labeler.confirm_button["state"]) == "disabled"  # one dealer at most
        labeler.confirm()
        assert written == []
        labeler.set_present("dealer_6_4", False)
        labeler.confirm()
        assert len(written) == 3 and all(p.parent == tmp_path for p in written)
        got = {load_dealer_label(p)["zone"]: load_dealer_label(p)["dealer"] for p in written}
        assert got == {"dealer_6_0": False, "dealer_6_2": True, "dealer_6_4": False}
        assert all(load_label(p) is None for p in written)  # never mistaken for a card label
        assert "salvati 3" in frame.dealer_var.get()

        frame._save_dealer().cancel()  # cancel writes nothing
        assert len(written) == 3
        assert vision_view.DEALER_PLAYERS == 6
    finally:
        frame.destroy()
        gc.collect()


def test_card_saves_never_capture_the_dealer_zones(app, monkeypatch):
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture
    from pokerlab.vision.regions import HOLE_CARDS, Region, RegionConfig, dealer_region_name

    grabbed = []
    monkeypatch.setattr(capture, "grab_region", lambda region: grabbed.append(region) or region)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(HOLE_CARDS, Region(0, 0, 170, 98))
        frame.regions.set(dealer_region_name(6, 1), Region(5, 5, 30, 30))
        frame._save_crops()
        assert grabbed == [Region(0, 0, 170, 98)]
    finally:
        frame.destroy()
        gc.collect()


def test_player_zones_and_labels_are_their_own_kind(tmp_path):
    from pokerlab.vision.labels import (
        SEAT_STATES,
        load_dealer_label,
        load_label,
        load_player_label,
        save_player_label,
    )
    from pokerlab.vision.regions import player_region_name

    assert player_region_name(6, 4) == "player_6_4"
    with pytest.raises(ValueError):
        player_region_name(6, 6)
    png = tmp_path / "player_6_4-x.png"
    save_player_label(png, "player_6_4", "fuori")
    assert load_player_label(png)["state"] == "fuori"
    assert load_label(png) is None and load_dealer_label(png) is None
    assert set(SEAT_STATES) == {"in_gioco", "fuori", "sit_out", "libero", "reazione"}
    for zone, state in (("dealer_6_1", "fuori"), ("player_6_1", "seduto")):
        with pytest.raises(ValueError):
            save_player_label(png, zone, state)


def test_saving_the_players_labels_every_seat_and_remembers_the_states(app, monkeypatch, tmp_path):
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture, labels
    from pokerlab.vision.labels import load_player_label
    from pokerlab.vision.regions import Region, RegionConfig, dealer_region_name, player_region_name

    written = []
    monkeypatch.setattr(capture, "grab_region", lambda region: region)
    monkeypatch.setattr(capture, "save_png", lambda frame, path: written.append(path) or path)
    monkeypatch.setattr(labels, "PLAYERS_DIR", tmp_path)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(dealer_region_name(6, 0), Region(0, 0, 30, 30))  # not a player zone
        for seat in (0, 1, 3):
            frame.regions.set(player_region_name(6, seat), Region(100 * seat, 0, 120, 80))
        labeler = frame._save_players()
        assert [seat for seat, *_ in labeler.shots] == [0, 1, 3]
        assert str(labeler.confirm_button["state"]) == "disabled"  # first time: nothing chosen
        labeler.choose("player_6_0", "in_gioco")
        labeler.choose("player_6_1", "fuori")
        assert str(labeler.confirm_button["state"]) == "disabled"  # seat 3 still unset
        labeler.choose("player_6_3", "sit_out")
        labeler.confirm()
        got = {load_player_label(p)["zone"]: load_player_label(p)["state"] for p in written}
        assert got == {"player_6_0": "in_gioco", "player_6_1": "fuori", "player_6_3": "sit_out"}

        again = frame._save_players()  # the next capture starts from the last states
        assert {z: v.get() for z, v in again.choices.items()} == got
        assert str(again.confirm_button["state"]) == "normal"
        again.cancel()
        assert len(written) == 3
    finally:
        frame.destroy()
        gc.collect()


def test_a_dealer_zone_the_size_of_a_player_box_is_flagged(app):
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision.regions import Region, RegionConfig, dealer_region_name

    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(dealer_region_name(6, 4), Region(0, 0, 41, 37))
        frame.regions.set(dealer_region_name(6, 5), Region(0, 0, 149, 111))  # a player box, by mistake
        frame._update_dealer_text()
        frame._update_players_text()
        text = frame.dealer_var.get()
        assert "troppo grande" in text and "5 (149x111)" in text and "4 (" not in text
        assert frame.dealer_zone_buttons[5].cget("text").startswith("✓ Gettone 5")
        assert frame.player_zone_buttons[5].cget("text") == "Giocatore 5"
    finally:
        frame.destroy()
        gc.collect()


def test_each_section_previews_only_its_own_zones_with_the_reading(app, monkeypatch):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture
    from pokerlab.vision.regions import (
        HOLE_CARDS,
        Region,
        RegionConfig,
        dealer_region_name,
        player_region_name,
    )

    def felt(width, height, gold=False):
        image = np.full((height, width, 3), (65, 120, 40), np.uint8)
        if gold:
            cv2.circle(image, (width // 2, height // 2), min(width, height) // 2 - 2, (40, 200, 235), -1)
        return image

    grabbed = []

    def grab(region):
        grabbed.append(region)
        image = felt(region.width, region.height, gold=(region.left == 40))
        if region.left == 300:  # the player zone: a reaction (an orange emoji) over the box
            cv2.circle(image, (region.width // 2, region.height // 2), 50, (30, 140, 245), -1)
        return image

    monkeypatch.setattr(capture, "grab_region", grab)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(HOLE_CARDS, Region(0, 500, 170, 98))
        for seat in (0, 1):
            frame.regions.set(dealer_region_name(6, seat), Region(40 * seat, 0, 38, 37))
        frame.regions.set(player_region_name(6, 3), Region(300, 0, 150, 110))

        def captions(window):
            return [w.cget("text") for tile in window.winfo_children() for w in tile.winfo_children()
                    if w.winfo_class() == "TLabel" and w.cget("text")]

        window = frame._preview_dealer()
        assert captions(window) == ["Gettone 0", "vuoto (oro 0%)", "Gettone 1", captions(window)[3]]
        assert captions(window)[3].startswith("GETTONE")
        assert {r.width for r in grabbed} == {38}  # dealer zones only
        window.destroy()

        grabbed.clear()
        window = frame._preview_players()
        assert captions(window) == ["Giocatore 3", "reazione"]
        assert grabbed == [Region(300, 0, 150, 110)]
        window.destroy()

        grabbed.clear()
        window = frame._preview()
        assert captions(window) == ["Mano"] and grabbed == [Region(0, 500, 170, 98)]
        window.destroy()
    finally:
        frame.destroy()
        gc.collect()


def test_amount_labels_keep_the_text_as_shown_and_are_their_own_kind(tmp_path):
    from pokerlab.vision.labels import (
        amount_error,
        load_amount_label,
        load_dealer_label,
        load_label,
        load_player_label,
        save_amount_label,
    )
    from pokerlab.vision.regions import bet_region_name

    assert bet_region_name(6, 2) == "bet_6_2"
    for ok in ("", "40", "1,250", "1.250", "2.5K", "12 k", "0.50"):
        assert amount_error(ok) is None, ok
    for bad in ("abc", "$40", "-5", "1K5", "K"):
        assert amount_error(bad), bad
    png = tmp_path / "pot-x.png"
    save_amount_label(png, "pot", " 1,250 ")
    assert load_amount_label(png)["text"] == "1,250"  # trimmed, separators kept
    assert load_label(png) is None and load_dealer_label(png) is None and load_player_label(png) is None
    with pytest.raises(ValueError):
        save_amount_label(png, "player_6_1", "40")
    with pytest.raises(ValueError):
        save_amount_label(png, "bet_6_1", "quaranta")


def test_saving_the_amounts_writes_the_pot_and_every_bet(app, monkeypatch, tmp_path):
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture, labels
    from pokerlab.vision.labels import load_amount_label
    from pokerlab.vision.regions import (
        POT,
        Region,
        RegionConfig,
        bet_region_name,
        player_region_name,
    )

    written = []
    monkeypatch.setattr(capture, "grab_region", lambda region: region)
    monkeypatch.setattr(capture, "save_png", lambda frame, path: written.append(path) or path)
    monkeypatch.setattr(labels, "AMOUNTS_DIR", tmp_path)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(player_region_name(6, 1), Region(0, 0, 150, 110))  # not an amount zone
        frame.regions.set(POT, Region(700, 300, 120, 30))
        for seat in (1, 4):
            frame.regions.set(bet_region_name(6, seat), Region(100 * seat, 400, 60, 22))
        frame._update_amounts_text()
        assert "Puntata 0" in frame.amounts_var.get() and "Piatto" not in frame.amounts_var.get()

        labeler = frame._save_amounts()
        assert [zone for _c, zone, _f, _p in labeler.shots] == ["pot", "bet_6_1", "bet_6_4"]  # pot first
        labeler.set_value("pot", "1,250")
        labeler.set_value("bet_6_1", "abc")
        assert str(labeler.confirm_button["state"]) == "disabled"
        labeler.confirm()
        assert written == []
        labeler.set_value("bet_6_1", "200")  # bet_6_4 left empty: no bet there
        labeler.confirm()
        got = {load_amount_label(p)["zone"]: load_amount_label(p)["text"] for p in written}
        assert got == {"pot": "1,250", "bet_6_1": "200", "bet_6_4": ""}
        assert all(p.parent == tmp_path for p in written)
    finally:
        frame.destroy()
        gc.collect()


def test_the_wheel_scrolls_the_sections_and_is_released_on_close(app):
    from types import SimpleNamespace

    from pokerlab.gui.vision_view import VisionFrame

    frame = VisionFrame(app)
    frame.pack(fill="both", expand=True)
    app.deiconify()
    try:
        app.geometry("900x300")  # shorter than the sections: there is something to scroll
        app.update()
        button = frame.dealer_zone_buttons[0]  # the wheel lands on whatever is under the pointer
        assert frame.scroll_canvas.yview()[0] == 0.0
        assert frame._on_wheel(SimpleNamespace(widget=button, delta=-120, num=0)) == "break"
        app.update()
        assert frame.scroll_canvas.yview()[0] > 0.0
        frame._on_wheel(SimpleNamespace(widget=button, delta=0, num=4))  # X11 wheel up
        frame._on_wheel(SimpleNamespace(widget=button, delta=0, num=4))
        app.update()
        assert frame.scroll_canvas.yview()[0] == 0.0
        assert frame._on_wheel(SimpleNamespace(widget=app, delta=-120, num=0)) is None  # not ours
        assert "".join(str(app.bind_all(s)) for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"))
    finally:
        app.withdraw()
        frame.destroy()
        gc.collect()
    assert not any(app.bind_all(s) for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"))


def test_the_turn_timer_is_collected_and_read_from_examples(app, monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from pathlib import Path

    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision import capture, labels
    from pokerlab.vision.labels import load_turn_label, save_turn_label
    from pokerlab.vision.regions import TURN_TIMER, Region, RegionConfig
    from pokerlab.vision.turn import LabelledTurn, TurnReader

    def timer(bar, length=1.0):
        image = np.full((20, 120, 3), (40, 50, 35), np.uint8)
        if bar:
            cv2.rectangle(image, (2, 7), (2 + int(115 * length), 13), (60, 200, 240), -1)
        return image

    from pokerlab.vision.turn import bar_share

    reader = TurnReader()  # a colour rule: no examples needed
    assert reader.read(timer(True, 0.8)) is True and reader.read(timer(False)) is False
    assert reader.read(timer(True, 0.05)) is True  # a bar nearly spent is still your turn
    red = timer(False)
    cv2.rectangle(red, (2, 7), (30, 13), (40, 40, 230), -1)  # the bar turned red near the end
    assert reader.read(red) is True and bar_share(timer(False)) == 0.0
    assert LabelledTurn(Path("x.png"), red, True).turn  # the labelled-crop type still exists

    written = []
    monkeypatch.setattr(capture, "grab_region", lambda region: timer(True))
    monkeypatch.setattr(capture, "save_png", lambda frame, path: written.append(path) or path)
    monkeypatch.setattr(labels, "TURN_DIR", tmp_path)
    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(TURN_TIMER, Region(0, 0, 120, 20))
        labeler = frame._save_turn()
        assert str(labeler.confirm_button["state"]) == "disabled"  # it has to be chosen
        labeler.choose(TURN_TIMER, "si")
        labeler.confirm()
        assert load_turn_label(written[0])["turn"] is True
    finally:
        frame.destroy()
        gc.collect()
    save_turn_label(tmp_path / "x.png", TURN_TIMER, False)
    with pytest.raises(ValueError):
        save_turn_label(tmp_path / "y.png", "pot", True)


def test_the_card_section_looks_like_the_others(app):
    from pokerlab.gui.vision_view import VisionFrame
    from pokerlab.vision.regions import BOARD, HOLE_CARDS, Region, RegionConfig

    frame = VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        frame.regions.set(HOLE_CARDS, Region(0, 0, 170, 98))
        frame._update_vision_text("Zona salvata.")
        assert frame.card_zone_buttons[HOLE_CARDS][1].cget("text") == "✓ Mie carte"
        assert frame.card_zone_buttons[BOARD][1].cget("text") == "Board"
        assert frame.vision_var.get() == "zone mancanti: Board | Zona salvata."
    finally:
        frame.destroy()
        gc.collect()


def test_the_seat_sections_switch_to_an_eight_seat_table(app, monkeypatch, tmp_path):
    from pokerlab.gui import vision_view
    from pokerlab.vision import capture, labels
    from pokerlab.vision.regions import Region, RegionConfig, dealer_region_name, player_region_name

    monkeypatch.setattr(capture, "grab_region", lambda region: region)
    monkeypatch.setattr(labels, "DEALER_DIR", tmp_path)
    frame = vision_view.VisionFrame(app)
    try:
        frame.regions = RegionConfig()
        assert len(frame.dealer_zone_buttons) == 6 and len(frame.player_zone_buttons) == 6
        assert frame.set_players(8) and not frame.set_players(8)
        assert len(frame.dealer_zone_buttons) == 8 and len(frame.player_zone_buttons) == 8
        assert len(frame.stack_zone_buttons) == 8 and len(frame.amount_zone_buttons) == 9  # pot + 8
        assert "8 giocatori" in str(frame.dealer_zone_buttons[0].master.master.cget("text"))
        frame.regions.set(dealer_region_name(6, 1), Region(5, 5, 30, 30))  # a 6-max zone: not ours now
        frame.regions.set(dealer_region_name(8, 7), Region(50, 5, 30, 30))
        labeler = frame._save_dealer()
        assert [zone for _seat, zone, *_ in labeler.shots] == ["dealer_8_7"]
        labeler.cancel()
        assert "zone mancanti" in frame.players_var.get()
        assert player_region_name(8, 7) == "player_8_7"
        assert frame.set_players(6) and len(frame.dealer_zone_buttons) == 6
    finally:
        frame.destroy()
        gc.collect()
