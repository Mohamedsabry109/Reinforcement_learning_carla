import random
import numpy as np


class EpsilonGreedy():
    def __init__(self, epsilon, linear_schedule = None):
        self.epsilon = epsilon
        self.linear_schedule = linear_schedule
        self.epsilon_counter = 0

    def choose_action(self,values):
        if (self.linear_schedule):
                self.update_epsilon()
        if random.randint(0,1) > self.epsilon:
            return np.argmax(values)
        else:
            return random.randrange(len(values))

    def update_epsilon(self):
        if self.epsilon <= self.linear_schedule[2]:  
            self.epsilon_counter += 1
            self.epsilon -= (self.linear_schedule[0] - self.linear_schedule[1]) / self.linear_schedule[2] 

