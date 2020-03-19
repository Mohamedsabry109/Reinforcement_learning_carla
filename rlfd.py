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
from Memory.memory_buffer import MemoryBuffer

agent = dqn.DDQN()
target_agent = dqn.DDQN()
memory_buffer = MemoryBuffer(buffer_size = 1000, with_per = True)
agent.get_branched_network(input_shape=(3,88,200))