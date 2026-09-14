from __future__ import annotations

from collections.abc import Callable, Sequence

from pokerlab.cards.card import Card
from pokerlab.engine.state import PlayerState, PlayerStatus, Pot
from pokerlab.evaluator.evaluator import HandRank
from pokerlab.evaluator.evaluator import evaluate as default_evaluate


def _refund_uncalled_bet(seats: list[PlayerState]) -> None:
    """If the single largest total commitment this hand was never matched by
    anyone else (folded or not), return the unmatched excess to that
    player's stack before pots are built. Standard NLHE "uncalled bet" rule.
    """
    committed = sorted((ps.total_committed for ps in seats if ps.total_committed > 0), reverse=True)
    if len(committed) < 2:
        return
    highest, second_highest = committed[0], committed[1]
    if highest <= second_highest:
        return
    top_contributors = [ps for ps in seats if ps.total_committed == highest]
    if len(top_contributors) != 1:
        return
    refund = highest - second_highest
    ps = top_contributors[0]
    ps.stack += refund
    ps.total_committed -= refund


def compute_pots(seats: list[PlayerState]) -> list[Pot]:
    """Build the main pot and any side pots from each seat's total
    commitment this hand. Also refunds an unmatched excess bet in place
    (mutates `seats`) before laying out the pots, since that chip amount was
    never actually contested."""
    _refund_uncalled_bet(seats)

    contributions = {ps.seat: ps.total_committed for ps in seats if ps.total_committed > 0}
    if not contributions:
        return []
    non_folded = {ps.seat for ps in seats if ps.status != PlayerStatus.FOLDED}

    pots: list[Pot] = []
    prev_level = 0
    for level in sorted(set(contributions.values())):
        per_player = level - prev_level
        contributors = [seat for seat, committed in contributions.items() if committed >= level]
        layer_total = per_player * len(contributors)
        eligible = frozenset(seat for seat in contributors if seat in non_folded)
        if layer_total > 0 and eligible:
            pots.append(Pot(amount=layer_total, eligible_seats=eligible))
        prev_level = level
    return pots


def _order_from_seat_after(seats: Sequence[int], after_seat: int) -> list[int]:
    seats_sorted = sorted(seats)
    start = after_seat + 1
    idx = 0
    for i, s in enumerate(seats_sorted):
        if s >= start:
            idx = i
            break
    else:
        idx = 0
    return seats_sorted[idx:] + seats_sorted[:idx]


def distribute_pots(
    pots: list[Pot],
    seats: list[PlayerState],
    board: list[Card],
    button_seat: int,
    evaluate_fn: Callable[[Sequence[Card]], HandRank] = default_evaluate,
) -> dict[int, int]:
    """Evaluate each pot's eligible hands and pay out winners, splitting ties
    with any odd chip going to the first eligible winner left of the button.
    Mutates each winning seat's stack in place; also returns the payouts.
    """
    seats_by_num = {ps.seat: ps for ps in seats}
    payouts: dict[int, int] = {}

    for pot in pots:
        eligible = list(pot.eligible_seats)
        if len(eligible) == 1:
            winners = eligible
        else:
            ranks: dict[int, HandRank] = {}
            for seat in eligible:
                ps = seats_by_num[seat]
                assert ps.hole_cards is not None
                ranks[seat] = evaluate_fn((*ps.hole_cards, *board))
            best = max(ranks.values())
            winners = [seat for seat, r in ranks.items() if r == best]

        winners_in_payout_order = _order_from_seat_after(winners, button_seat)
        share, remainder = divmod(pot.amount, len(winners_in_payout_order))
        for i, seat in enumerate(winners_in_payout_order):
            payouts[seat] = payouts.get(seat, 0) + share + (1 if i < remainder else 0)

    for seat, amount in payouts.items():
        seats_by_num[seat].stack += amount

    return payouts
