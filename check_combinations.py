

def find_best_combination(cards):

    if len(cards) != 7:
        raise ValueError(f"There must be exactly 7 cards to find the best combination. Obtained: {len(cards)}")
    
    cards_aces_high = [(14, card[1]) if card[0] == 1 else card for card in cards]
    sorted_cards = sorted(cards_aces_high, key=lambda x: (x[0], x[1]), reverse=True)

    # no combinations
    best_combination = (0, [card[0] for card in sorted_cards[:5]])

    # count the number of cards of each rank
    card_counters = {}
    for card in sorted_cards:
        if card[0] not in card_counters:
            card_counters[card[0]] = 1
        else:
            card_counters[card[0]] += 1
    
    card_counters_list = sorted(card_counters.items(), key=lambda x: x[1], reverse=True)
    
    # check for pairs, three of a kind, four of a kind, full house
    if card_counters_list[0][1] == 4:
        best_combination = (7, [card_counters_list[0][0]] * 4 + [card_counters_list[1][0]]) # four of a kind
    elif card_counters_list[0][1] == 3:
        if card_counters_list[1][1] >= 2:
            best_combination = (6, [card_counters_list[0][0]] * 3 + [card_counters_list[1][0]] * 2) # full house
        else:
            best_combination = (3, [card_counters_list[0][0]] * 3 + [card_counters_list[1][0]] + [card_counters_list[2][0]]) # three of a kind
    elif card_counters_list[0][1] == 2:
        if card_counters_list[1][1] == 2:
            best_combination = (2, [card_counters_list[0][0]] * 2 + [card_counters_list[1][0]] * 2 + [card_counters_list[2][0]]) # two pair
        else:
            best_combination = (1, [card_counters_list[0][0]] * 2 + [card_counters_list[1][0]] + [card_counters_list[2][0]] + [card_counters_list[3][0]]) # one pair

    # check for straight
    straight = False
    if best_combination[0] < 4:
        if 14 in card_counters:
            sorted_cards.append((1, -1))
        
        straight_counter = 0
        index_first_card = -1

        for i in range(len(sorted_cards) - 1):
            if sorted_cards[i][0] - 1 == sorted_cards[i + 1][0]:
                straight_counter += 1
                if index_first_card == -1:
                    index_first_card = i
            elif sorted_cards[i][0] == sorted_cards[i + 1][0]:
                continue
            else:
                straight_counter = 0
                index_first_card = -1

            if len(sorted_cards) - i + straight_counter < 4 or straight_counter == 4:
                break
        
        if straight_counter == 4:
            best_combination = (4, list(range(sorted_cards[index_first_card][0], sorted_cards[index_first_card][0] - 5, -1))) # straight
            straight = True

    # check for flush
    flush = False
    flush_type = -1
    if best_combination[0] < 5:
        suit_counters = {}
        for card in sorted_cards:
            if card[1] not in suit_counters:
                suit_counters[card[1]] = 1
            else:
                suit_counters[card[1]] += 1
        
        suit_counters_list = sorted(suit_counters.items(), key=lambda x: x[1], reverse=True)
        flush_type = suit_counters_list[0][0]

        if suit_counters_list[0][1] >= 5:
            best_combination = (5, [card[0] for card in sorted_cards if card[1] == suit_counters_list[0][0]][:5])
            flush = True
    
    # check for straight flush
    if straight and flush:
        sorted_cards = [card for card in sorted_cards if card[1] == flush_type]

        for e in sorted_cards:
            if e[0] == 14:
                sorted_cards.append((1, flush_type))
        
        straight_counter = 0
        index_first_card = -1

        for i in range(len(sorted_cards) - 1):
            if sorted_cards[i][0] - 1 == sorted_cards[i + 1][0]:
                straight_counter += 1
                if index_first_card == -1:
                    index_first_card = i
            else:
                straight_counter = 0
                index_first_card = -1

            if len(sorted_cards) - i + straight_counter < 4 or straight_counter == 4:
                break
        
        if straight_counter == 4:
            best_combination = (8, list(range(sorted_cards[index_first_card][0], sorted_cards[index_first_card][0] - 5, -1))) # straight flush
            straight = True

    return(best_combination)



if __name__ == "__main__":
    # Example usage
    cards = [(12, 1), (13, 1), (11, 1), (5, 3), (1, 1), (1, 2), (10, 1)]
    print(find_best_combination(cards))