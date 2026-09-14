import random

import pytest

from pokerlab.cards.card import Card, Rank, Suit
from pokerlab.cards.deck import Deck


def test_card_str_and_parse_roundtrip():
    card = Card(Rank.ACE, Suit.HEARTS)
    assert str(card) == "Ah"
    assert Card.parse("Ah") == card
    assert Card.parse("Td") == Card(Rank.TEN, Suit.DIAMONDS)


def test_card_parse_rejects_invalid_strings():
    with pytest.raises(ValueError):
        Card.parse("Zh")
    with pytest.raises(ValueError):
        Card.parse("Ax")
    with pytest.raises(ValueError):
        Card.parse("A")


def test_deck_has_52_unique_cards():
    deck = Deck()
    dealt = deck.deal(52)
    assert len(dealt) == 52
    assert len(set(dealt)) == 52
    assert len(deck) == 0


def test_deck_deal_too_many_raises():
    deck = Deck()
    with pytest.raises(ValueError):
        deck.deal(53)


def test_deck_shuffle_is_deterministic_with_seeded_rng():
    deck_a = Deck(rng=random.Random(123))
    deck_a.shuffle()
    hand_a = deck_a.deal(5)

    deck_b = Deck(rng=random.Random(123))
    deck_b.shuffle()
    hand_b = deck_b.deal(5)

    assert hand_a == hand_b


def test_deck_shuffle_changes_order_with_different_seeds():
    deck_a = Deck(rng=random.Random(1))
    deck_a.shuffle()

    deck_b = Deck(rng=random.Random(2))
    deck_b.shuffle()

    assert deck_a.deal(52) != deck_b.deal(52)
