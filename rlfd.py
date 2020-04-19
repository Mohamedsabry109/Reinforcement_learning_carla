import random
import sys
from os import path, environ
from typing import Union, Dict
import time
from  collections import namedtuple
import os
import signal
import logging
import subprocess
import numpy as np
from enum import Enum
from typing import List, Union
import pygame
from Environments import carla_environment
from DataHandler import data_handler
from Agent import dqn
from Memory.memory_buffer import MemoryBuffer,OfflineMemoryBuffer, OnlineMemoryBuffer
from ExplorationPolicy.epsilon_greedy import EpsilonGreedy

#declare some paths
data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
interaction_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/rl_data'
imitation_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
validation_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'



agent = dqn.DDQN(imitation_data_directory=imitation_data_directory)
# #target_agent = dqn.DDQN()
# exp_policy = EpsilonGreedy(epsilon = 0.5, linear_schedule = [0.5,0.05,10])
#agent.initialize_buffers()
agent.train_agent()
#first we will do imitation while preparing offline buffers of imitation
#when starts interaction initialize online buffer for rl
#prepare and initialize online  and offline buffers for rl

#agent.get_branched_network(input_shape=(3,88,200))

handler = data_handler.handler(train_data_directory = imitation_data_directory, validation_data_directory = imitation_data_directory)
print(handler.map_outputs(throttle =0 , brake = 0.9 , steer = -0.99))


# env = carla_environment.CarlaEnvironment()
# print("number of available poses are : ",env.num_positions)
# for i in range(1):
#     print("iteration ",i)
#     episode_data = []
#     while env.done == False:
#         env.step([0,1,0])
#         episode_data.append(env.state)

#     env.reset(True)
# print(env.action_space)
# print(env.state_space)
# print("number of frames",len(episode_data))
# env.close_server()

# print("number of available poses are : ",env.num_positions)
# print("follow poses are : ",env.follow_poses)
# print("left poses are : ",env.left_poses)
# print("right poses are : ", env.right_poses)
# print("straight poses are : ", env.Straight_poses)
