from policy import PolicyContext


class Poker_Player:

    def __init__(self, name, policy, chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.current_bet = 0
        self.total_bet = 0
        self.all_in = False # managed in the game class
        self.folded = False

        self.policy = policy

    def take_action(self, higher_bet, small_blind, common_cards, current_pot, table_extras=None) -> tuple[int, bool, bool]: # bet, fold, all_in
        while True:
            to_call = max(0, higher_bet - self.current_bet)
            if self.policy == 'manual':
                action = input(
                    f"{self.name} turn. chips: {self.chips}, this street: {self.current_bet}, to call: {to_call}\n"
                ).strip().lower()
            else:
                min_raise_to = higher_bet + max(small_blind * 2, 1)
                policy_context = PolicyContext(
                    hand=self.hand.copy(),
                    common_cards=common_cards.copy(),
                    chips=self.chips,
                    to_call=to_call,
                    player_street_bet=self.current_bet,
                    highest_bet=higher_bet,
                    small_blind=small_blind,
                    big_blind=small_blind * 2,
                    pot=current_pot,
                    min_raise_to=min_raise_to,
                    can_check=to_call == 0,
                    can_raise=(self.current_bet + self.chips) > higher_bet,
                    extras=table_extras if table_extras is not None else {},
                )
                action = self.policy.decide_action(policy_context)
            
            if action == 'f':
                return 0, True, False
            elif action == 'c':
                if to_call == 0:
                    return 0, False, False

                amount = min(to_call, self.chips)
                self.chips -= amount
                self.current_bet += amount
                self.total_bet += amount
                return amount, False, self.chips == 0
            else:
                try:
                    bet = int(action)
                    if bet <= 0:
                        print("Invalid input. Raise must be a positive amount.")
                    elif bet > self.chips:
                        print("You don't have enough chips to raise that amount.")
                    elif bet == self.chips and bet + self.current_bet <= higher_bet:
                        self.chips -= bet
                        self.current_bet += bet
                        self.total_bet += bet
                        return bet, False, self.chips == 0
                    elif bet + self.current_bet <= higher_bet:
                        minimum_raise = higher_bet + max(small_blind * 2, 1) - self.current_bet
                        print(
                            f"Your raise must add at least {minimum_raise} chips, "
                            f"so your total bet becomes greater than {higher_bet}."
                        )
                    elif bet + self.current_bet < higher_bet + max(small_blind * 2, 1) and bet != self.chips:
                        print(f"Your raise must bring the total bet to at least {higher_bet + max(small_blind * 2, 1)} chips.")
                    else:
                        self.chips -= bet
                        self.current_bet += bet
                        self.total_bet += bet
                        return bet, False, self.chips == 0
                except ValueError:
                    print("Invalid input. Please enter 'f', 'c', or a number.")
