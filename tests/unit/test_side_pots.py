from pokerlab.cards.card import Card
from pokerlab.engine.pots import compute_pots, distribute_pots
from pokerlab.engine.state import PlayerState, PlayerStatus


def seat(seat_num, stack, total_committed, status=PlayerStatus.ACTIVE, hole=None):
    ps = PlayerState(
        seat=seat_num,
        player_id=f"p{seat_num}",
        name=f"P{seat_num}",
        stack=stack,
        hole_cards=hole,
        status=status,
    )
    ps.total_committed = total_committed
    return ps


def test_three_way_all_in_with_folded_dead_money_builds_correct_layers():
    a = seat(0, stack=0, total_committed=50, status=PlayerStatus.ALL_IN)
    b = seat(1, stack=0, total_committed=100, status=PlayerStatus.ALL_IN)
    c = seat(2, stack=50, total_committed=100, status=PlayerStatus.ACTIVE)
    d = seat(3, stack=0, total_committed=20, status=PlayerStatus.FOLDED)

    pots = compute_pots([a, b, c, d])

    assert len(pots) == 3
    assert pots[0].amount == 80 and pots[0].eligible_seats == frozenset({0, 1, 2})
    assert pots[1].amount == 90 and pots[1].eligible_seats == frozenset({0, 1, 2})
    assert pots[2].amount == 100 and pots[2].eligible_seats == frozenset({1, 2})
    assert sum(p.amount for p in pots) == 50 + 100 + 100 + 20


def test_distribute_pots_pays_short_stack_winner_only_from_pots_they_are_eligible_for():
    a = seat(0, stack=0, total_committed=50, status=PlayerStatus.ALL_IN, hole=(Card.parse("Ah"), Card.parse("Ad")))
    b = seat(1, stack=0, total_committed=100, status=PlayerStatus.ALL_IN, hole=(Card.parse("Kc"), Card.parse("Kd")))
    c = seat(2, stack=50, total_committed=100, status=PlayerStatus.ACTIVE, hole=(Card.parse("2c"), Card.parse("2d")))
    d = seat(3, stack=0, total_committed=20, status=PlayerStatus.FOLDED)
    board = [Card.parse(t) for t in ["5h", "7s", "9c", "Jd", "3h"]]

    pots = compute_pots([a, b, c, d])
    payouts = distribute_pots(pots, [a, b, c, d], board, button_seat=3)

    # A (pocket aces) wins the main pot + side pot it's eligible for (80 + 90 = 170)
    # but is not eligible for the third-level pot (100), which goes to B (pocket kings > C's twos).
    assert payouts == {0: 170, 1: 100}
    assert a.stack == 170
    assert b.stack == 100
    assert c.stack == 50  # unchanged: eligible for the last pot but lost it
    assert d.stack == 0


def test_distribute_pots_splits_ties_and_assigns_odd_chip_left_of_button():
    # Both players hold the exact same rank (split pot), pot has an odd chip.
    a = seat(0, stack=0, total_committed=101, hole=(Card.parse("Ah"), Card.parse("Kd")))
    b = seat(1, stack=0, total_committed=101, hole=(Card.parse("As"), Card.parse("Kc")))
    board = [Card.parse(t) for t in ["2h", "5s", "9c", "Jd", "3h"]]

    pots = compute_pots([a, b])
    assert len(pots) == 1 and pots[0].amount == 202

    payouts = distribute_pots(pots, [a, b], board, button_seat=1)

    # Seat 0 is immediately left of the button (seat 1) -> gets the odd chip.
    assert payouts == {0: 101, 1: 101}


def test_refund_uncalled_bet_returns_unmatched_excess_to_sole_top_contributor():
    a = seat(0, stack=500, total_committed=500, status=PlayerStatus.ACTIVE)
    b = seat(1, stack=0, total_committed=100, status=PlayerStatus.FOLDED)

    pots = compute_pots([a, b])

    assert a.total_committed == 100  # refunded 400 back
    assert a.stack == 900  # 500 leftover stack + 400 refund
    assert len(pots) == 1
    assert pots[0].amount == 200  # 100 (a) + 100 (b, dead money)
    assert pots[0].eligible_seats == frozenset({0})  # b folded, not eligible


def test_no_refund_when_top_commitment_is_shared_by_multiple_players():
    a = seat(0, stack=0, total_committed=200, status=PlayerStatus.ALL_IN)
    b = seat(1, stack=0, total_committed=200, status=PlayerStatus.ALL_IN)

    compute_pots([a, b])

    assert a.total_committed == 200
    assert b.total_committed == 200
