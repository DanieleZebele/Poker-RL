import random
from dataclasses import dataclass, field
from typing import Any

from check_combinations import find_best_combination


@dataclass(frozen=True)
class PolicyContext:
    hand: list[tuple[int, int]]
    common_cards: list[tuple[int, int]]
    chips: int
    to_call: int
    player_street_bet: int
    highest_bet: int
    small_blind: int
    big_blind: int
    pot: int
    min_raise_to: int
    can_check: bool
    can_raise: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def minimum_raise_amount(self) -> int:
        return max(0, self.min_raise_to - self.player_street_bet)



class Policy:
    def __init__(self):
        pass

    def decide_action(self, context: PolicyContext):
        raise NotImplementedError("This method should be overridden by subclasses.")


class RandomPolicy(Policy):
    def __init__(self):
        pass

    def decide_action(self, context: PolicyContext):
        random_prob = random.random()
        if random_prob < 0.1:
            action = 'f'
        elif random_prob < 0.8:
            action = 'c'
        else:
            if context.minimum_raise_amount() < context.chips:
                action = str(random.randint(context.minimum_raise_amount(), context.chips))
            else:
                action = str(context.chips)
        
        return action


class CustomPolicy(Policy):
    def __init__(self):
        pass

    def decide_action(self, context: PolicyContext):

        if len(context.common_cards) < 3:
            if context.to_call < 2 * context.big_blind:
                return 'c'
            else:
                return 'f'

        best_combination = find_best_combination(context.hand + context.common_cards)
        rank = best_combination[0]

        if rank == 0:
            if context.to_call == 0:
                return 'c'
            else:
                return 'f'
        elif rank == 1:
            if context.to_call <= 5 * context.big_blind:
                return 'c'
            elif context.to_call == 0:
                action = str(2 * context.minimum_raise_amount())
            else:
                return 'f'
        else:
            return str(context.chips)

# Example usage
if __name__ == "__main__":
    pass