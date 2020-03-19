import random

class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon # probability of explore
        self.counts = counts # number of pulls for each arms
        self.values = values # average amount of reward we've gotten from each arms
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def ind_max(self, x):
        m = max(x)
        return x.index(m)

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.ind_max(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward # weighted average of the previously estimated value and the reward we just received
        self.values[chosen_arm] = new_value
        return
