import random
import check_combinations
import os
import policy
terminal_width = os.get_terminal_size().columns

class Poker_Player:

    def __init__(self, name, policy, chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.current_bet = 0
        self.all_in = False # managed in the game class

        self.policy = policy

    def take_action(self, higher_bet, small_blind) -> tuple[int, bool, bool]: # bet, fold, all_in
        while True:
            if self.policy == 'manual':
                action = input(f"{self.name} turn. chips: {self.chips}, played: {self.current_bet}, {higher_bet - self.current_bet} chips to call\n").strip().lower()
            else:
                action = self.policy.decide_action(self.chips)
            
            if action == 'f':
                return 0, True, False  # Fold: bet = 0, fold = True, all_in = False
            elif action == 'c':
                if higher_bet - self.current_bet < self.chips:
                    self.chips -= higher_bet - self.current_bet
                    self.current_bet = higher_bet
                    return higher_bet, False, self.chips == 0  # Call: bet = higher_bet, fold = False, all_in = False
                else:
                    self.current_bet += self.chips
                    self.chips = 0
                    return self.current_bet, False, True  # All-in: bet = chips, fold = False, all_in = True 
            else:
                try:
                    bet = int(action)
                    if bet + self.current_bet > higher_bet and bet <= self.chips and (bet + self.current_bet >= small_blind * 2 or bet == self.chips):
                        self.chips -= bet
                        self.current_bet += bet
                        return self.current_bet, False, self.chips == 0  # Raise: bet = entered amount, fold = False, all_in = False
                    elif bet + self.current_bet <= higher_bet:
                        print(f"Your raise must be greater than the current higher bet ({higher_bet}).")
                    elif bet > self.chips:
                        print("You don't have enough chips to raise that amount.")
                    elif bet + self.current_bet < small_blind * 2:
                        print(f"Your bet must be at least equal to the blind ({small_blind * 2}).")
                except ValueError:
                    print("Invalid input. Please enter 'f', 'c', or a number.")



class Poker_Game:

    def __init__(self, num_players = 2, verboose = 0, starting_small_blind = 2, blind_increment_games = 5, blind_increment = 1.5):
        if num_players < 2 or num_players > 9:
            raise ValueError("Number of players must be between 2 and 9")
        
        if verboose == 1:
            print(f"\nStarting a new game with {num_players} players")
            print("COMMANDS: F=fold, C=call/check/all-in, N=raise: \n")

        self.verboose = verboose
        self.num_players = num_players
        self.deck = self.get_new_deck()
        self.players = []

        for i in range(1, num_players + 1):
            print('define poker policy for player', i)
            player_id = i
            #policy_choice = input(f"Player {player_id} policy (m/r): ").strip().lower()
            policy_choice = 'r'
            if policy_choice == 'm':
                player_policy = 'manual'
            elif policy_choice == 'r':
                player_policy = policy.RandomPolicy()
            else:
                raise ValueError("Invalid policy choice. Use 'manual' or 'random'.")
            
            self.players.append(Poker_Player(name=f"Player {player_id}", policy=player_policy, chips=1000))

        self.common_cards = []
        self.active_players = self.players.copy()
        self.first_player_idx = -1 # in the first game, first_player will be set to 0
        self.pots = [] # (pot, players allowed to win it)
        self.current_pot = 0
        self.small_blind = starting_small_blind
        self.blind_increment = blind_increment
        self.games = 0
        self.blind_increment_games = blind_increment_games


    def reset_game(self):
        self.games += 1
        if self.games % self.blind_increment_games == 0:
            self.small_blind = int(self.small_blind * self.blind_increment)
            if self.verboose == 1:
                print(f"Blind increased to {self.small_blind}/{self.small_blind * 2}")

        self.deck = self.get_new_deck()
        self.common_cards = []
        i = 0
        while i < len(self.players):
            if self.players[i].chips == 0:
                if self.verboose == 1:
                    print(f"{self.players[i].name} is out of the game")
                self.players.pop(i)
                if i <= self.first_player_idx:
                    self.first_player_idx -= 1
            else:
                i += 1
        self.first_player_idx = (self.first_player_idx + 1) % len(self.players) 
        self.active_players = [player for player in self.players if player.chips > 0]
        self.active_players = self.active_players[self.first_player_idx:] + self.active_players[:self.first_player_idx] # dealer last
        self.current_pot = 0
        self.pots = []
        self.reset_hands()

        for player in self.players:
            player.hand = []
            player.all_in = False

        if self.verboose == 1:
            print(f"Game starts with {self.players[self.first_player_idx].name} as first player")
            print(f"Current blinds: {self.small_blind}/{self.small_blind * 2}")
    
    def reset_hands(self): 
        for player in self.players:
            player.hand = []

    def reset_chips(self):
        for player in self.players:
            player.chips = 1000

    def get_new_deck(self):
        cards = []
        for i in range(1, 14):
            for j in range(1, 5):
                cards.append((i,j))
        random.shuffle(cards)
        return cards

    def get_cards(self, num_cards: int) -> list[tuple[int, int]]:
        if len(self.deck) < num_cards:
            raise ValueError("Not enough cards left in the deck")
        return [self.deck.pop() for _ in range(num_cards)]

    def deal_cards(self):
        for player in self.players:
            cards = self.get_cards(2)
            player.hand = cards

    def check_winner(self, players) -> tuple[list[Poker_Player], tuple[int, list[int]]]:
        best_combination = (-1, [])
        winners = []
        for player in players:
            cards = player.hand + self.common_cards
            combination = check_combinations.find_best_combination(cards)
            if combination[0] > best_combination[0]:
                best_combination = combination
                winners = [player]
            elif combination[0] == best_combination[0]:
                for i in range(len(combination[1])):
                    if combination[1][i] > best_combination[1][i]:
                        best_combination = combination
                        winners = [player]
                        break
                    elif combination[1][i] < best_combination[1][i]:
                        break
                    elif i == len(combination[1]) - 1 and combination[1][i] == best_combination[1][i]:
                        winners.append(player)
        
        return winners, best_combination

    def run_betting_round(self, first_round, num_players) -> bool: # returns True if no players left, False otherwise
        higher_bet = 0
        current_player = self.active_players[0]
        bet_player = current_player
        all_in_players = [] # (bet, player)
        for player in self.active_players:
            player.current_bet = 0
        
        blinds = 2 if first_round else 0

        while True:

            # manage blinds
            if blinds == 2:
                if current_player.chips >= self.small_blind:
                    current_player.chips -= self.small_blind
                    current_player.current_bet += self.small_blind
                    tot_bet = self.small_blind
                    all_in = False
                    fold = False
                else:
                    current_player.current_bet += current_player.chips
                    tot_bet = current_player.chips
                    current_player.chips = 0
                    all_in = True
                    fold = False
                blinds -= 1
            elif blinds == 1:
                if current_player.chips >= self.small_blind * 2:
                    current_player.chips -= self.small_blind * 2
                    current_player.current_bet += self.small_blind * 2
                    tot_bet = self.small_blind * 2
                    all_in = False
                    fold = False
                else:
                    current_player.current_bet += current_player.chips
                    tot_bet = current_player.chips
                    current_player.chips = 0
                    all_in = True
                    fold = False
                blinds -= 1
            
            else:
                # no blinds
                tot_bet, fold, all_in = current_player.take_action(higher_bet, self.small_blind) # bet = amount used to bet in this round or all-in

            if fold:
                if self.verboose == 1:
                    print(f"{current_player.name} folds")

                next_player = self.active_players[(self.active_players.index(current_player) + 1) % len(self.active_players)]
                self.active_players.remove(current_player)
                current_player = next_player
                if len(self.active_players) == 1:
                    break

            elif all_in:
                if self.verboose == 1:
                    print(f"{current_player.name} all-ins {tot_bet} chips")

                self.current_pot += tot_bet
                current_player.all_in = True

                if tot_bet > higher_bet: # all-in raise
                    higher_bet = tot_bet
                    bet_player = current_player

                all_in_players.append((tot_bet, current_player))
                next_player = self.active_players[(self.active_players.index(current_player) + 1) % len(self.active_players)]
                self.active_players.remove(current_player)
                current_player = next_player
                
            else:
                if tot_bet > higher_bet:
                    if self.verboose == 1:
                        if higher_bet == 0:
                            print(f"{current_player.name} bet to {tot_bet} chips")
                        else:
                            print(f"{current_player.name} raises to {tot_bet} chips")   
                    higher_bet = tot_bet
                    if blinds == 0:
                        bet_player = current_player
                else:
                    if self.verboose == 1:
                        if higher_bet == 0:
                            print(f"{current_player.name} checks")
                        else:
                            print(f"{current_player.name} calls to {higher_bet} chips")

                self.current_pot += tot_bet
                next_player = self.active_players[(self.active_players.index(current_player) + 1) % len(self.active_players)]
                current_player = next_player

            if first_round and tot_bet < self.small_blind * 2:
                higher_bet = self.small_blind * 2

            if first_round and num_players == 2 and blinds == 0 and len(self.active_players) == 2 and self.active_players[-1].current_bet < self.small_blind * 2:
                current_player = self.active_players[-1]
            else:
                    # termination condition
                if bet_player == current_player or len(self.active_players) < 1: # second condition is for all-in players
                    break

        # manage all-in players and pots 
        used_chips = 0
        if len(all_in_players) > 0:
            all_in_players.sort(key=lambda x: x[0])

            for i in range(len(all_in_players)):
                if all_in_players[i][0] - used_chips > 0:
                    all_in_pot = (all_in_players[i][0] - used_chips) * (len(self.active_players) + len(all_in_players) - i)
                    added_players = []
                    for j in range(i, len(all_in_players)):
                        added_players.append(all_in_players[j][1])
                    self.pots.append((all_in_pot, added_players)) # remaining players added in the last round
                    used_chips += all_in_players[i][0]
                    self.current_pot -= all_in_pot
        
        return len(self.active_players) <= 1

    def run_game(self):
        
        self.reset_game()
        self.deal_cards()

        self.show_status()

        # round 1
        end_game = self.run_betting_round(first_round = self.small_blind != 0, num_players=len(self.active_players))

        self.common_cards = self.get_cards(3)
        if not end_game:
            # round 2
            self.show_status()
            end_game = self.run_betting_round(first_round=False, num_players=len(self.active_players))
        
        self.common_cards.extend(self.get_cards(1))
        if not end_game:
            # round 3
            self.show_status()
            end_game = self.run_betting_round(first_round=False, num_players=len(self.active_players))

        self.common_cards.extend(self.get_cards(1))
        self.show_status()
        if not end_game:
            # round 4
            end_game = self.run_betting_round(first_round=False, num_players=len(self.active_players))


        # add remaining players as possible winner to each pot
        # re-activate all-in players that are not in deficit with chips
        
        if self.current_pot != 0:
            self.pots.append((self.current_pot, []))

        if len(self.active_players) == 0:
            max_bet = 0
            for player in self.players:
                if player.current_bet > max_bet:
                    max_bet = player.current_bet
            for player in self.players:
                if player.current_bet == max_bet:
                    self.active_players.append(player)
            
        for pot in self.pots:
            pot[1].extend(self.active_players)

        # check for winners and distribute pots
        for pot in self.pots:
            winners, best_combination = self.check_winner(pot[1])
            for player in winners:
                player.chips += pot[0] // len(winners)
                if self.verboose == 1:
                    s = f"#   {player.name} wins {pot[0] // len(winners)} chips   #"
                    print()
                    print("#" * len(s))
                    print(s)
                    print("#" * len(s))
                    print()
        
        tot_players = 0
        idx_winner = -1
        for player in self.players: # no active players to count also all-in players
            if player.chips > 0:
                tot_players += 1
                if self.verboose == 1:
                    idx_winner = self.players.index(player)

        if tot_players == 1:
            if self.verboose == 1:
                print(f"{self.players[idx_winner].name} wins the game!")
            return True
        else:
            return False

    def show_status(self):
        print("-" * terminal_width)

        pots_str = ""
        for pot in self.pots:
            pots_str += f"{pot[0]}"
        pots_str += f" {self.current_pot}"

        print(f"TOTAL POT: {pots_str}" )

        suit_symbols = {1: '♥', 2: '♦', 3: '♣', 4: '♠'}
        number_symbols = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}
        for player in self.players:
            hand = player.hand
            card_1 = f"{number_symbols[hand[0][0]]}{suit_symbols[hand[0][1]]}"
            card_2 = f"{number_symbols[hand[1][0]]}{suit_symbols[hand[1][1]]}"
            print(f"{player.name} ({player.chips}): {card_1} {card_2}")
        
        common_card_strs = ""
        for card in self.common_cards:
            card_str = f"{number_symbols[card[0]]}{suit_symbols[card[1]]}"
            common_card_strs += f"{card_str} "
        
        if common_card_strs != "":
            print(f"Common Cards: {common_card_strs.strip()}")
        
        print("-" * terminal_width)


if __name__ == '__main__':

    game = Poker_Game(num_players=2, starting_small_blind=5, verboose=1, blind_increment_games = 5, blind_increment = 1.5)

    while not game.run_game():
        pass
    