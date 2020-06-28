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
        if random.random() > self.epsilon:
            max_action_value = np.argmax(values)
            return  max_action_value # exploitation 
        else:
            action_number = random.randrange(len(values[0]))
            return  action_number# exploration

    def update_epsilon(self):
        if self.epsilon >= self.linear_schedule[1]:  
            self.epsilon_counter += 1
            self.epsilon -= (self.linear_schedule[0] - self.linear_schedule[1]) / self.linear_schedule[2] 

    # def update_epsilon(self):
    #     if self.epsilon <= self.linear_schedule[2]:  
    #         self.epsilon_counter += 1
    #         self.epsilon -= (self.linear_schedule[0] - self.linear_schedule[1]) / self.linear_schedule[2] 
    #         print("\nepsilon",self.epsilon)


# exp_policy = EpsilonGreedy(epsilon = 0.5, linear_schedule = [0.5,0.05,10])
