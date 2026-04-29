import os
import random

import check_combinations
import policy
from poker_player import Poker_Player

try:
    terminal_width = os.get_terminal_size().columns
except OSError:
    terminal_width = 100

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
        self.first_player_idx = -1

        for i in range(1, num_players + 1):
            print('define poker policy for player', i)
            player_id = i
            #policy_choice = input(f"Player {player_id} policy (m/r): ").strip().lower()
            policy_choice = 'r'
            if policy_choice == 'm':
                player_policy = 'manual'
            elif policy_choice == 'r':
                player_policy = policy.RandomPolicy()
            elif policy_choice == 'c':
                player_policy = policy.CustomPolicy()
            else:
                raise ValueError("Invalid policy choice. Use 'manual' or 'random'.")
            
            self.players.append(Poker_Player(name=f"Player {player_id}", policy=player_policy, chips=1000))

        self.common_cards = []
        self.active_players = self.players.copy()
        self.pots = [] # (pot, players allowed to win it)
        self.current_pot = 0
        self.small_blind = starting_small_blind
        self.blind_increment = blind_increment
        self.games = 0
        self.blind_increment_games = blind_increment_games

    def _print_banner(self, title):
        if self.verboose == 1:
            print()
            print("=" * terminal_width)
            print(title.center(terminal_width))
            print("=" * terminal_width)

    def _street_name(self, community_cards_count):
        if community_cards_count == 0:
            return "PRE-FLOP"
        if community_cards_count == 3:
            return "FLOP"
        if community_cards_count == 4:
            return "TURN"
        if community_cards_count == 5:
            return "RIVER"
        return "SHOWDOWN"

    def _describe_player_action(self, player, action_name, amount=0, highest_bet=0):
        if self.verboose != 1:
            return

        if action_name == "fold":
            print(f"- {player.name} folds")
        elif action_name == "check":
            print(f"- {player.name} checks")
        elif action_name == "call":
            print(f"- {player.name} calls {amount} chips")
        elif action_name == "raise":
            print(f"- {player.name} raises to {highest_bet} chips")
        elif action_name == "all-in-call":
            print(f"- {player.name} calls all-in for {amount} chips")
        elif action_name == "all-in-raise":
            print(f"- {player.name} goes all-in to {highest_bet} chips")
        elif action_name == "sb":
            print(f"- {player.name} posts small blind of {amount} chips")
        elif action_name == "bb":
            print(f"- {player.name} posts big blind of {amount} chips")

    def _combination_name(self, rank):
        combinations = {
            0: "High Card",
            1: "One Pair",
            2: "Two Pair",
            3: "Three of a Kind",
            4: "Straight",
            5: "Flush",
            6: "Full House",
            7: "Four of a Kind",
            8: "Straight Flush"
        }
        return combinations.get(rank, "Unknown")


    def _table_order(self):
        if len(self.players) == 0:
            return []
        return self.players[self.first_player_idx:] + self.players[:self.first_player_idx]

    def _build_side_pots(self):
        contributions = [(player.total_bet, player) for player in self.players if player.total_bet > 0]
        contributions.sort(key=lambda x: x[0])

        pots = []
        previous_level = 0
        remaining = contributions

        while remaining:
            level = remaining[0][0]
            pot_amount = (level - previous_level) * len(remaining)
            eligible_players = [player for _, player in remaining if not player.folded]
            pots.append((pot_amount, eligible_players))

            previous_level = level
            remaining = [(amount, player) for amount, player in remaining if amount > level]

        return pots


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
                if len(self.players) > 0 and i <= self.first_player_idx:
                    self.first_player_idx -= 1
            else:
                i += 1
        if len(self.players) == 0:
            self.active_players = []
            return

        self.first_player_idx = (self.first_player_idx + 1) % len(self.players)
        self.active_players = self._table_order()
        self.current_pot = 0
        self.pots = []
        self.reset_hands()

        for player in self.players:
            player.hand = []
            player.all_in = False
            player.folded = False
            player.current_bet = 0
            player.total_bet = 0

        if self.verboose == 1:
            self._print_banner(
                f"HAND {self.games}  |  Dealer/Button: {self.players[self.first_player_idx].name}  |  Blinds: {self.small_blind}/{self.small_blind * 2}"
            )
            table_info = " | ".join(f"{player.name}: {player.chips}" for player in self.players)
            print(table_info)
            print()
    
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
            if combination > best_combination:
                best_combination = combination
                winners = [player]
            elif combination == best_combination:
                winners.append(player)
        
        return winners, best_combination

    def run_betting_round(self, first_round, num_players) -> bool: # returns True if no players left, False otherwise
        if len(self.players) < 2:
            return True

        table_order = self.active_players.copy()
        for player in table_order:
            player.current_bet = 0

        highest_bet = 0

        if first_round:
            if len(table_order) == 2:
                small_blind_idx = 0
                big_blind_idx = 1
                start_idx = 0
            else:
                small_blind_idx = 1
                big_blind_idx = 2
                start_idx = 3 % len(table_order)

            for blind_idx, blind_amount in ((small_blind_idx, self.small_blind), (big_blind_idx, self.small_blind * 2)):
                blind_player = table_order[blind_idx]
                amount = min(blind_player.chips, blind_amount)
                blind_player.chips -= amount
                blind_player.current_bet += amount
                blind_player.total_bet += amount
                self.current_pot += amount
                if amount < blind_amount:
                    blind_player.all_in = True
                self._describe_player_action(
                    blind_player,
                    "sb" if blind_amount == self.small_blind else "bb",
                    amount=amount,
                )

            highest_bet = table_order[big_blind_idx].current_bet
            if self.verboose == 1:
                print(f"  Pot is now {self.current_pot} chips")
        else:
            start_idx = 1 % len(table_order)

        round_order = table_order[start_idx:] + table_order[:start_idx]
        pending = [player for player in round_order if not player.folded and not player.all_in]
        self.active_players = [player for player in table_order if not player.folded]

        while pending:
            current_player = pending.pop(0)
            if current_player.folded or current_player.all_in:
                continue

            table_extras = {
                "active_players": [player.name for player in self.players if not player.folded],
                "all_in_players": [player.name for player in self.players if player.all_in and not player.folded],
                "players_total": len(self.players),
            }
            tot_bet, fold, all_in = current_player.take_action(
                highest_bet,
                self.small_blind,
                self.common_cards,
                self.current_pot,
                table_extras,
            )

            if fold:
                current_player.folded = True
                self._describe_player_action(current_player, "fold")
                self.active_players = [player for player in self.active_players if not player.folded]
                pending = [player for player in pending if not player.folded]
                if len([player for player in self.players if not player.folded]) <= 1:
                    break
                continue

            self.current_pot += tot_bet

            if all_in:
                current_player.all_in = True
                if current_player.current_bet > highest_bet:
                    self._describe_player_action(current_player, "all-in-raise", highest_bet=current_player.current_bet)
                else:
                    self._describe_player_action(current_player, "all-in-call", amount=tot_bet)

            if current_player.current_bet > highest_bet:
                highest_bet = current_player.current_bet
                if not all_in:
                    self._describe_player_action(current_player, "raise", highest_bet=highest_bet)

                if current_player in table_order:
                    current_idx = table_order.index(current_player)
                    # Rebuild pending: start from next player, but include anyone who hasn't matched highest_bet yet
                    pending = [
                        player for player in (table_order[current_idx + 1:] + table_order[:current_idx])
                        if not player.folded and not player.all_in
                    ]
                    # Also re-add players who have already acted but haven't reached the new highest_bet
                    for player in table_order:
                        if not player.folded and not player.all_in and player.current_bet < highest_bet and player not in pending:
                            pending.append(player)
            elif not all_in:
                if highest_bet == 0:
                    self._describe_player_action(current_player, "check")
                else:
                    self._describe_player_action(current_player, "call", amount=tot_bet)

            self.active_players = [player for player in table_order if not player.folded]

            if self.verboose == 1:
                print(f"  Pot: {self.current_pot} chips | To call: {highest_bet}")

            if not pending:
                pending = [
                    player for player in table_order
                    if not player.folded and not player.all_in and player.current_bet < highest_bet
                ]

            non_all_in_players = [player for player in self.players if not player.folded and not player.all_in]
            unmatched_non_all_in = any(player.current_bet < highest_bet for player in non_all_in_players)
            if len(non_all_in_players) <= 1 and not unmatched_non_all_in:
                break

        self.active_players = [player for player in table_order if not player.folded]
        return len([player for player in self.players if not player.folded and not player.all_in]) <= 1

    def run_game(self):
        
        self.reset_game()
        if len(self.players) < 2:
            return True

        self.deal_cards()

        if self.verboose == 1:
            self._print_banner("CARDS DEALT")
        self.show_status()

        def can_bet_again():
            return len([player for player in self.players if not player.folded and not player.all_in]) > 1

        if self.verboose == 1:
            self._print_banner("PRE-FLOP BETTING ROUND")
        self.run_betting_round(first_round=True, num_players=len(self.active_players))

        survivors = [player for player in self.players if not player.folded]
        if len(survivors) == 1:
            self.pots = self._build_side_pots()
        else:
            self.common_cards = self.get_cards(3)
            if self.verboose == 1:
                self._print_banner("FLOP")
            self.show_status()
            if can_bet_again():
                if self.verboose == 1:
                    self._print_banner("FLOP BETTING ROUND")
                self.run_betting_round(first_round=False, num_players=len(self.active_players))

            self.common_cards.extend(self.get_cards(1))
            if self.verboose == 1:
                self._print_banner("TURN")
            self.show_status()
            if can_bet_again():
                if self.verboose == 1:
                    self._print_banner("TURN BETTING ROUND")
                self.run_betting_round(first_round=False, num_players=len(self.active_players))

            self.common_cards.extend(self.get_cards(1))
            if self.verboose == 1:
                self._print_banner("RIVER")
            self.show_status()
            if can_bet_again():
                if self.verboose == 1:
                    self._print_banner("RIVER BETTING ROUND")
                self.run_betting_round(first_round=False, num_players=len(self.active_players))

            self.pots = self._build_side_pots()

        if not self.pots and self.current_pot > 0:
            self.pots = [(self.current_pot, [player for player in self.players if not player.folded])]

        # Ensure all players with 0 chips are marked as all-in
        for player in self.players:
            if player.chips == 0 and not player.folded:
                player.all_in = True

        showdown_players = [player for player in self.players if not player.folded]
        if len(showdown_players) == 1:
            showdown_winner = showdown_players[0]
            showdown_winner.chips += self.current_pot
            if self.verboose == 1:
                print()
                print(f"{showdown_winner.name} wins the pot uncontested")
                print()
        else:
            # Check absolute best hand once at showdown
            abs_winners, abs_best_combo = self.check_winner(showdown_players)
            combo_name = self._combination_name(abs_best_combo[0])
            
            for pot in self.pots:
                eligible_players = pot[1] if len(pot[1]) > 0 else showdown_players
                # Filter to only winners who are eligible for this pot
                pot_winners = [w for w in abs_winners if w in eligible_players]
                
                if pot_winners:
                    share = pot[0] // len(pot_winners)
                    for player in pot_winners:
                        player.chips += share
                        if self.verboose == 1:
                            s = f"#   {player.name} wins {share} chips with {combo_name}   #"
                            print()
                            print("#" * len(s))
                            print(s)
                            print("#" * len(s))
                            print()
                    remainder = pot[0] - share * len(pot_winners)
                    if remainder > 0 and pot_winners:
                        pot_winners[0].chips += remainder
                else:
                    # Fallback: find winner among eligible players for this pot
                    winners, combo = self.check_winner(eligible_players)
                    combo_name_fallback = self._combination_name(combo[0])
                    share = pot[0] // len(winners)
                    for player in winners:
                        player.chips += share
                        if self.verboose == 1:
                            s = f"#   {player.name} wins {share} chips with {combo_name_fallback}   #"
                            print()
                            print("#" * len(s))
                            print(s)
                            print("#" * len(s))
                            print()
                    remainder = pot[0] - share * len(winners)
                    if remainder > 0 and winners:
                        winners[0].chips += remainder
        
        tot_players = 0
        idx_winner = -1
        for player in self.players:
            if player.chips > 0:
                tot_players += 1
                idx_winner = self.players.index(player)

        if tot_players == 1:
            if self.verboose == 1:
                print(f"{self.players[idx_winner].name} wins the game!")
            return True
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
            if player.folded:
                status = "FOLDED"
            elif player.all_in:
                status = "ALL-IN"
            else:
                status = "IN GAME"
            print(f"{player.name} ({player.chips}) [{status}]: {card_1} {card_2}")
        
        common_card_strs = ""
        for card in self.common_cards:
            card_str = f"{number_symbols[card[0]]}{suit_symbols[card[1]]}"
            common_card_strs += f"{card_str} "
        
        if common_card_strs != "":
            print(f"Common Cards: {common_card_strs.strip()}")

        in_game_players = [player.name for player in self.players if not player.folded]
        folded_players = [player.name for player in self.players if player.folded]
        print(f"Still in hand: {', '.join(in_game_players) if in_game_players else '-'}")
        print(f"Folded: {', '.join(folded_players) if folded_players else '-'}")
        
        print("-" * terminal_width)


if __name__ == '__main__':

    game = Poker_Game(num_players=2, starting_small_blind=5, verboose=1, blind_increment_games = 5, blind_increment = 1.5)

    game.players[0].policy = policy.CustomPolicy()

    while not game.run_game():
        pass
    