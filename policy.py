
import random
class RandomPolicy:
    def __init__(self):
        pass

    def decide_action(self, chips):
        random_prob = random.random()
        if random_prob < 0.4:
            action = 'f'
        elif random_prob < 0.9:
            action = 'c'
        else:
            action = random.choice(['f', 'c', str(random.randint(0, chips))])
        
        return action

# Example usage
if __name__ == "__main__":
    pass