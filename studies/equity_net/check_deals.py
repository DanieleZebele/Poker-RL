"""Checks the deal generator: the labels are well formed, they average to the equity a
poker table gives for a known matchup, and how fast it is.

    PYTHONPATH=src python studies/equity_net/check_deals.py
"""

import random
import time

from deals import pot_shares, sample_deal


def idx(text: str) -> int:
    from deals import SUITS

    from pokerlab.cards import Card

    card = Card.parse(text)
    return SUITS.index(card.suit) * 13 + int(card.rank) - 2


def main() -> None:
    rng = random.Random(1)

    # 1. Labels are shares of one pot: non-negative, summing to 1, half-steps for ties.
    deals = [sample_deal(rng) for _ in range(5000)]
    assert all(abs(sum(d.shares) - 1.0) < 1e-9 for d in deals)
    assert all(len(set(d.shares) | {0.0, 1.0}) >= 2 for d in deals)
    mean_share = sum(sum(d.shares) / d.players for d in deals) / len(deals)
    print(f"quota media per giocatore {mean_share:.3f} (con n giocatori vale 1/n, media su 2-9: ~0.22)")
    sizes = [d.players for d in deals]
    print("giocatori:", {n: sizes.count(n) for n in range(2, 10)})
    streets = [d.visible for d in deals]
    print("board visibile:", {v: streets.count(v) for v in (0, 3, 4, 5)})

    # 2. A known matchup: AA against KK preflop is 81-83% depending on the suits (flush
    #    chances); AsAh against KdKc is about 81.3% (150,000 runouts: 0.8129 +/- 0.001).
    holes = ((idx("As"), idx("Ah")), (idx("Kd"), idx("Kc")))
    rest = [c for c in range(52) if c not in {x for h in holes for x in h}]
    wins = [0.0, 0.0]
    runs = 20000
    for _ in range(runs):
        shares = pot_shares(holes, tuple(rng.sample(rest, 5)))
        wins[0] += shares[0]
        wins[1] += shares[1]
    print(f"AA contro KK: {wins[0] / runs:.3f} / {wins[1] / runs:.3f} (attesi ~0.813 / ~0.187 per questi semi)")

    # 3. Speed, one core.
    start = time.time()
    count = 2000
    for _ in range(count):
        sample_deal(rng)
    print(f"{count / (time.time() - start):.0f} mani nuove al secondo per processo")


if __name__ == "__main__":
    main()
