def find_best_combination(cards):
    if not 5 <= len(cards) <= 7:
        raise ValueError(f"There must be between 5 and 7 cards to find the best combination. Obtained: {len(cards)}")

    normalized_cards = [(14 if rank == 1 else rank, suit) for rank, suit in cards]
    sorted_cards = sorted(normalized_cards, key=lambda x: (x[0], x[1]), reverse=True)

    def straight_high(ranks):
        unique_ranks = set(ranks)
        if 14 in unique_ranks:
            unique_ranks.add(1)

        for high in range(14, 4, -1):
            if all((high - offset) in unique_ranks for offset in range(5)):
                return high
        return None

    def straight_flush_high(cards_for_suit):
        return straight_high([rank for rank, _ in cards_for_suit])

    rank_counts = {}
    suit_groups = {}
    for rank, suit in sorted_cards:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        suit_groups.setdefault(suit, []).append((rank, suit))

    ranks_desc = sorted(rank_counts.keys(), reverse=True)

    # Straight flush
    best_sf = None
    for suit_cards in suit_groups.values():
        if len(suit_cards) >= 5:
            sf_high = straight_flush_high(suit_cards)
            if sf_high is not None:
                candidate = (8, list(range(sf_high, sf_high - 5, -1)))
                if best_sf is None or candidate > best_sf:
                    best_sf = candidate
    if best_sf is not None:
        return best_sf

    # Four of a kind
    quads = sorted([rank for rank, count in rank_counts.items() if count == 4], reverse=True)
    if quads:
        quad_rank = quads[0]
        kicker = max(rank for rank in ranks_desc if rank != quad_rank)
        return (7, [quad_rank] * 4 + [kicker])

    # Full house
    trips = sorted([rank for rank, count in rank_counts.items() if count >= 3], reverse=True)
    pairs = sorted([rank for rank, count in rank_counts.items() if count >= 2], reverse=True)
    if trips:
        trip_rank = trips[0]
        pair_candidates = [rank for rank in pairs if rank != trip_rank]
        if len(trips) >= 2:
            pair_candidates = [rank for rank in trips[1:]] + pair_candidates
        if pair_candidates:
            return (6, [trip_rank] * 3 + [pair_candidates[0]] * 2)

    # Flush
    flush_candidate = None
    for suit_cards in suit_groups.values():
        if len(suit_cards) >= 5:
            flush_ranks = sorted([rank for rank, _ in suit_cards], reverse=True)[:5]
            candidate = (5, flush_ranks)
            if flush_candidate is None or candidate > flush_candidate:
                flush_candidate = candidate
    if flush_candidate is not None:
        return flush_candidate

    # Straight
    straight = straight_high([rank for rank, _ in sorted_cards])
    if straight is not None:
        return (4, list(range(straight, straight - 5, -1)))

    # Three of a kind
    if trips:
        trip_rank = trips[0]
        kickers = [rank for rank in ranks_desc if rank != trip_rank][:2]
        return (3, [trip_rank] * 3 + kickers)

    # Two pair
    if len(pairs) >= 2:
        top_pair, second_pair = pairs[:2]
        kicker = max(rank for rank in ranks_desc if rank not in (top_pair, second_pair))
        return (2, [top_pair] * 2 + [second_pair] * 2 + [kicker])

    # One pair
    if len(pairs) == 1:
        pair_rank = pairs[0]
        kickers = [rank for rank in ranks_desc if rank != pair_rank][:3]
        return (1, [pair_rank] * 2 + kickers)

    # High card
    return (0, ranks_desc[:5])


if __name__ == "__main__":
    cards = [(12, 1), (13, 2), (11, 1), (5, 3), (1, 1), (1, 2), (12, 2)]
    print(find_best_combination(cards))