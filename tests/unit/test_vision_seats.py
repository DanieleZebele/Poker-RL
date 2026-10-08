"""The seat-state rules, on crops drawn like the client's player boxes."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from pokerlab.vision.seats import read_seat, sit_out_template

BACKGROUND = (35, 40, 30)
AVATAR = (120, 110, 100)


def box(*, backs=False, chair=False, my_cards=None, sit_out=False):
    image = np.full((110, 150, 3), BACKGROUND, np.uint8)
    if not chair:
        cv2.circle(image, (75, 60), 40, AVATAR, -1)
    if backs:
        cv2.rectangle(image, (35, 15), (115, 100), (150, 40, 230), -1)  # pink-magenta backs
    if chair:
        cv2.rectangle(image, (60, 70), (90, 100), (40, 200, 40), 3)  # green chair outline
    if my_cards is not None:
        cv2.rectangle(image, (20, 10), (130, 105), (60, 60, 60), -1)
        if my_cards == "live":
            cv2.putText(image, "J 7", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.circle(image, (60, 80), 12, (255, 255, 255), -1)
            cv2.circle(image, (105, 80), 12, (255, 255, 255), -1)  # ~0.08 white, like real live cards
        else:  # folded: the same, dimmed
            cv2.putText(image, "J 7", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 3)
    if sit_out:
        cv2.rectangle(image, (20, 45), (130, 75), (30, 30, 30), -1)
        cv2.putText(image, "SIT OUT", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return image


@pytest.fixture
def templates():
    return [sit_out_template(box(sit_out=True))]


def test_an_opponent_with_card_backs_is_in_the_hand(templates):
    assert read_seat(box(backs=True), 3, templates).state == "in_gioco"


def test_an_opponent_showing_only_the_avatar_is_out(templates):
    assert read_seat(box(), 3, templates).state == "fuori"


def test_a_green_chair_is_an_empty_seat(templates):
    assert read_seat(box(chair=True), 5, templates).state == "libero"


def test_the_sit_out_pill_wins(templates):
    assert read_seat(box(sit_out=True), 1, templates).state == "sit_out"
    assert read_seat(box(sit_out=True), 1, []).state == "fuori"  # without an example it cannot know


def test_your_seat_is_read_from_the_white_of_your_cards(templates):
    assert read_seat(box(my_cards="live"), 0, templates).state == "in_gioco"
    assert read_seat(box(my_cards="folded"), 0, templates).state == "fuori"


def test_an_empty_seat_drawn_as_bare_table_is_told_by_its_own_zone(tmp_path, templates):
    """At 8-max the client draws no chair: an empty seat is just table (here a
    grey logo). It is recognised by matching that zone's labelled empty crop."""
    from pokerlab.vision.labels import save_player_label
    from pokerlab.vision.seats import empty_backgrounds, load_seats

    table = np.full((110, 150, 3), (140, 140, 140), np.uint8)
    cv2.putText(table, "888", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (210, 210, 210), 4)
    for name, image, state in (("player_8_4-a", table, "libero"), ("player_8_3-a", box(), "fuori")):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), image)
        save_player_label(path, name.split("-")[0], state)
    backgrounds = empty_backgrounds(load_seats(tmp_path))
    assert set(backgrounds) == {"player_8_4"}

    assert read_seat(table, 4, templates).state == "fuori"  # without the example: no chair, "out"
    assert read_seat(table, 4, templates, backgrounds["player_8_4"]).state == "libero"
    # someone sitting down in that seat is no longer the empty background
    assert read_seat(box(), 4, templates, backgrounds["player_8_4"]).state == "fuori"
    assert read_seat(box(backs=True), 4, templates, backgrounds["player_8_4"]).state == "in_gioco"
