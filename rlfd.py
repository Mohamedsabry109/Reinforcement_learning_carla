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
from Utils.utils import *
from Utils.config import *
#declare some paths

data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
interaction_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/rl_data'
imitation_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
validation_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'


agent = dqn.DDQN(imitation_data_directory=imitation_data_directory,rl_data_directory = interaction_data_directory)
exp_policy = EpsilonGreedy(epsilon = 0.1, linear_schedule = [0.1,0.001,1000000]) # 1M -> exploration ends after 10K steps
#TODO Exploration for each high level command for effective exploration 
########### supervised training #############
agent.train_agent_supervised(iterations = 3000)

########### Open the environment #############
env = carla_environment.CarlaEnvironment()

###########  Heatup for RL #############
print("Starting Heatup")
heatup(env,agent,exp_policy,ACTION_NUMBER_TO_VALUES)

########### supervised + rl training #############

print("supervised + rl training ")
number_initial_non_used_frames = 50 #number of frames the car is falling from the sky 
hlc_to_network_output = {0:2,1:0,2:1,3:3} # dictionary maps hlc from carla to the proper output branch 
freq_update_target_network = 2 # freq of updating the target network
freq_update_offline_buffers = 2 # freq for updating the offline buffers
interaction_steps = 20
for i in range(1,interaction_steps):
    episode_data = {'states':[],'actions':[],'reward':[],'done':[]}
    while env.done == False:
        actions = agent.model.predict([ np.expand_dims(env.state['forward_camera'],axis = 0),np.array([env.state['measurements'][0]])])
        high_level_command = env.state['high_level_command']
        action = exp_policy.choose_action(actions[hlc_to_network_output[high_level_command]])
        #we need to map action to acc and steer
        steer, acc_brake = ACTION_NUMBER_TO_VALUES[action]
        if acc_brake >= 0 :
            acc = acc_brake
            brake  = 0
        else:
            brake = abs(acc_brake)
            acc = 0
        #steer, throttle, brake
        env.step([steer,acc,brake])
        
        if number_initial_non_used_frames > 0:
            number_initial_non_used_frames -= 1
        else:
            #actions = agent.model.predict([ np.expand_dims(env.state['forward_camera'],axis = 0),np.array([env.state['measurements'][0]])])
            episode_data['states'].append(env.state)
            episode_data['actions'].append(actions)
            episode_data['reward'].append(env.reward)
            episode_data['done'].append(env.done)
            agent.train_agent_rl_supervised(iterations = 1)

    #update offline buffers and reload
    if (i % freq_update_offline_buffers == 0):
        agent.update_offline_buffer()
        agent.update_rl_offline_buffer()
        agent.reload_imitation_online_buffers()
        agent.reload_rl_online_buffers()

    if (i % freq_update_target_network == 0):
        agent.update_target_model()

    save_interaction_data(episode_data,agent)
    env.reset(True)

env.close_server()





######################### ENV INFO ###########################
#state data return by the environment is a dict
# episode data is a list of dicts
# iamges shape is -> 88 , 200 , 3
# measurements shape is -> (4,)
# high level command is one hot code
# actions -> list of numpy arrays
# episdoe data is sent to the buffer to be saved
action  = 0 # action number 
#filtered_i = 0.0
reward  = 0 # reward
#print("number of available poses are : ",env.num_positions)

#heatup(env,agent,exp_policy,ACTION_NUMBER_TO_VALUES)

# for i in range(1):
#     print("iteration ",i)
#     episode_data = {'states':[],'actions':[],'reward':[],'done':[]}
#     while env.done == False:
#         env.step([0,1,0])
#         episode_data['states'].append(env.state)
#         actions = agent.model.predict([ np.expand_dims(env.state['segmentation'],axis = 0),np.array([env.state['measurements'][0]])])
#         episode_data['actions'].append(actions)
#         episode_data['reward'].append(env.reward)
#         episode_data['done'].append(env.done)
#         #episode_data.append((env.state,env.done))

#     env.reset(True)

# print(env.action_space)
# print(env.state_space)
# print("number of frames",len(episode_data))
# print("number of available poses are : ",env.num_positions)
# print("follow poses are : ",env.follow_poses)
# print("left poses are : ",env.left_poses)
# print("right poses are : ", env.right_poses)
# print("straight poses are : ", env.Straight_poses)
# print(episode_data[0].keys())
# print(episode_data[0]['segmentation'])
# print(episode_data[0]['measurements'])
# print(episode_data[0]['high_level_command'])

