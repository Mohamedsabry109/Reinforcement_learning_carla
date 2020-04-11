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

from Agent import dqn
from Memory.memory_buffer import MemoryBuffer,OfflineMemoryBuffer
from ExplorationPolicy.epsilon_greedy import EpsilonGreedy

#declare some paths
data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
interaction_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/rl_data'
imitation_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
validation_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'

agent = dqn.DDQN(imitation_data_directory=imitation_data_directory)
#target_agent = dqn.DDQN()
exp_policy = EpsilonGreedy(epsilon = 0.5, linear_schedule = [0.5,0.05,10])
#agent.initialize_buffers()
#agent.train_agent()
#first we will do imitation while preparing offline buffers of imitation
#when starts interaction initialize online buffer for rl
#prepare and initialize online  and offline buffers for rl

#agent.get_branched_network(input_shape=(3,88,200))
