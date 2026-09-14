import itertools

import pytest

from pokerlab.cards.card import Card
from pokerlab.evaluator.evaluator import HandCategory, HandEvaluator, evaluate, rank_five


def cards(text: str) -> list[Card]:
    """Parse a space-separated string of cards, e.g. 'Ah Kh Qh Jh Th'."""
    return [Card.parse(tok) for tok in text.split()]


def test_category_ordering_full_chain():
    hands = {
        HandCategory.HIGH_CARD: cards("2c 5d 9h Jh Ah"),
        HandCategory.PAIR: cards("2c 2d 9h Jh Ah"),
        HandCategory.TWO_PAIR: cards("2c 2d 9h 9c Ah"),
        HandCategory.THREE_OF_A_KIND: cards("2c 2d 2h 9c Ah"),
        HandCategory.STRAIGHT: cards("5c 6d 7h 8c 9h"),
        HandCategory.FLUSH: cards("2h 5h 9h Jh Ah"),
        HandCategory.FULL_HOUSE: cards("2c 2d 2h 9c 9h"),
        HandCategory.FOUR_OF_A_KIND: cards("2c 2d 2h 2s 9h"),
        HandCategory.STRAIGHT_FLUSH: cards("5h 6h 7h 8h 9h"),
    }
    ranked = {cat: rank_five(hand) for cat, hand in hands.items()}
    ordered_cats = sorted(hands.keys())
    for weaker, stronger in itertools.pairwise(ordered_cats):
        assert ranked[weaker] < ranked[stronger], f"{weaker} should be weaker than {stronger}"
        assert ranked[weaker].category == weaker
        assert ranked[stronger].category == stronger


def test_wheel_straight_ranks_as_five_high():
    wheel = rank_five(cards("Ah 2d 3h 4c 5s"))
    six_high = rank_five(cards("2h 3d 4h 5c 6s"))
    assert wheel.category == HandCategory.STRAIGHT
    assert wheel.tiebreakers == (5,)
    assert six_high.category == HandCategory.STRAIGHT
    assert wheel < six_high, "wheel (5-high) must lose to a 6-high straight"


def test_wheel_is_not_confused_with_ace_high_no_straight():
    # A, 2, 3, 4, 6 is NOT a straight (gap at 5) despite containing an ace and low cards.
    not_a_straight = rank_five(cards("Ah 2d 3h 4c 6s"))
    assert not_a_straight.category == HandCategory.HIGH_CARD


def test_royal_flush_is_ace_high_straight_flush():
    royal = rank_five(cards("Th Jh Qh Kh Ah"))
    assert royal.category == HandCategory.STRAIGHT_FLUSH
    assert royal.tiebreakers == (14,)


def test_straight_flush_beats_four_of_a_kind():
    straight_flush = rank_five(cards("5h 6h 7h 8h 9h"))
    quads = rank_five(cards("Ac Ad Ah As Kc"))
    assert straight_flush > quads


def test_four_of_a_kind_tiebreak_by_kicker():
    quads_ace_king_kicker = rank_five(cards("Ac Ad Ah As Kc"))
    quads_ace_queen_kicker = rank_five(cards("Ac Ad Ah As Qc"))
    assert quads_ace_king_kicker > quads_ace_queen_kicker


def test_full_house_tiebreak_trips_rank_first_then_pair_rank():
    aces_full_of_twos = rank_five(cards("Ac Ad Ah 2s 2c"))
    kings_full_of_queens = rank_five(cards("Kc Kd Kh Qs Qc"))
    assert aces_full_of_twos > kings_full_of_queens

    aces_full_of_kings = rank_five(cards("Ac Ad Ah Ks Kc"))
    aces_full_of_twos_2 = rank_five(cards("Ac Ad Ah 2s 2c"))
    assert aces_full_of_kings > aces_full_of_twos_2


def test_flush_tiebreak_by_highest_cards_in_order():
    ace_high_flush = rank_five(cards("Ah 2h 5h 9h Jh"))
    king_high_flush = rank_five(cards("Kh 2h 5h 9h Jh"))
    assert ace_high_flush > king_high_flush


def test_three_of_a_kind_tiebreak_by_kickers():
    trips_with_ace_kicker = rank_five(cards("2c 2d 2h Ac 9h"))
    trips_with_king_kicker = rank_five(cards("2c 2d 2h Kc 9h"))
    assert trips_with_ace_kicker > trips_with_king_kicker


def test_two_pair_tiebreak_high_pair_then_low_pair_then_kicker():
    aces_and_kings = rank_five(cards("Ac Ad Kh Kc 2h"))
    aces_and_queens = rank_five(cards("Ac Ad Qh Qc 2h"))
    assert aces_and_kings > aces_and_queens

    aces_and_kings_ace_kicker = rank_five(cards("Ac Ad Kh Kc 3h"))
    aces_and_kings_two_kicker = rank_five(cards("Ac Ad Kh Kc 2h"))
    assert aces_and_kings_ace_kicker > aces_and_kings_two_kicker


def test_pair_tiebreak_by_kickers_in_order():
    pair_with_better_kickers = rank_five(cards("2c 2d Ah Kc 9h"))
    pair_with_worse_kickers = rank_five(cards("2c 2d Ah Kc 8h"))
    assert pair_with_better_kickers > pair_with_worse_kickers


def test_high_card_tiebreak_by_all_five_ranks_in_order():
    better = rank_five(cards("2c 5d 9h Jh Ah"))
    worse = rank_five(cards("2c 5d 9h Jh Kh"))
    assert better > worse


def test_identical_five_card_hands_are_equal_for_split_pots():
    a = rank_five(cards("Ah Kd Qc Js 9h"))
    b = rank_five(cards("Ac Kh Qs Jd 9c"))
    assert a == b


def test_rank_five_rejects_wrong_card_count():
    with pytest.raises(ValueError):
        rank_five(cards("Ah Kd Qc Js"))
    with pytest.raises(ValueError):
        rank_five(cards("Ah Kd Qc Js 9h 8h"))


def test_evaluate_rejects_fewer_than_five_cards():
    with pytest.raises(ValueError):
        evaluate(cards("Ah Kd Qc Js"))


def test_evaluate_picks_best_five_of_seven():
    # Board makes a straight flush; hole cards are irrelevant low pair.
    seven = cards("5h 6h 7h 8h 9h 2c 2d")
    result = evaluate(seven)
    assert result.category == HandCategory.STRAIGHT_FLUSH
    assert result.tiebreakers == (9,)


def test_evaluate_does_not_miss_a_better_five_card_combo_hidden_in_seven():
    # Two separate trips among 7 cards (2c2d2h and 9c9d9h) plus a kicker:
    # best hand is trips 9s full of 2s (using one of the pair-of-2 cards as the pair).
    seven = cards("2c 2d 2h 9c 9d 9h 4s")
    result = evaluate(seven)
    assert result.category == HandCategory.FULL_HOUSE
    assert result.tiebreakers == (9, 2)


def test_hand_evaluator_facade_matches_module_functions():
    hole = (Card.parse("Ah"), Card.parse("Kh"))
    board = cards("Qh Jh Th 2c 3d")
    via_facade = HandEvaluator.evaluate(hole, board)
    via_function = evaluate((*hole, *board))
    assert via_facade == via_function
    assert via_facade.category == HandCategory.STRAIGHT_FLUSH

    weaker = rank_five(cards("2c 3d 4h 5s 7c"))
    assert HandEvaluator.compare(via_facade, weaker) == 1
    assert HandEvaluator.compare(weaker, via_facade) == -1
    assert HandEvaluator.compare(via_facade, via_function) == 0
