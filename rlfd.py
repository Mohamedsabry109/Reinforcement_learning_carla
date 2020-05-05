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



agent = dqn.DDQN(imitation_data_directory=imitation_data_directory,rl_data_directory = interaction_data_directory)


#actions = agent.model.predict()
# exp_policy = EpsilonGreedy(epsilon = 0.5, linear_schedule = [0.5,0.05,10])
#agent.train_agent()
#first we will do imitation while preparing offline buffers of imitation
#when starts interaction initialize online buffer for rl
#prepare and initialize online  and offline buffers for rl

#handler = data_handler.handler(train_data_directory = imitation_data_directory, validation_data_directory = imitation_data_directory)
#print(handler.map_outputs(throttle =0 , brake = 0.9 , steer = -0.99))



######################### ENV INFO ###########################
#state data return by the environment is a dict
# episode data is a list of dicts
# iamges shape is -> 88 , 200 , 3
# measurements shape is -> (4,)
# high level command is one hot code
# episdoe data is sent to the buffer to be saved
action  = 0 # action number 
#filtered_i = 0.0
reward  = 0 # reward
env = carla_environment.CarlaEnvironment()
print("number of available poses are : ",env.num_positions)
for i in range(1):
    print("iteration ",i)
    episode_data = []
    while env.done == False:
        env.step([0,1,0])
        episode_data.append(env.state)
        actions = agent.model.predict([ np.expand_dims(env.state['segmentation'],axis = 0),np.array([env.state['measurements'][0]])])
        print(type(actions))
        print(actions)
        print(np.array(actions))
        #episode_data.append((env.state,env.done))

    env.reset(True)
print(env.action_space)
print(env.state_space)
print("number of frames",len(episode_data))
# print("number of available poses are : ",env.num_positions)
# print("follow poses are : ",env.follow_poses)
# print("left poses are : ",env.left_poses)
# print("right poses are : ", env.right_poses)
# print("straight poses are : ", env.Straight_poses)
# print(episode_data[0].keys())
# print(episode_data[0]['segmentation'])
# print(episode_data[0]['measurements'])
# print(episode_data[0]['high_level_command'])
env.close_server()
