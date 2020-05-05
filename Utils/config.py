import numpy as np 
from .utils import *




QUANTIZATION_BINS = 21

STEER_RANGE = [-1, 1]
STEER_VALUES = np.linspace(STEER_RANGE[0], STEER_RANGE[1], num=QUANTIZATION_BINS)

THROTTLE_BRAKE_RANGE = [-1, 1]
THROTTLE_BRAKE_VALUES = np.linspace(THROTTLE_BRAKE_RANGE[0], THROTTLE_BRAKE_RANGE[1], num=QUANTIZATION_BINS)

ACTION_DICT = get_comb(STEER_VALUES, THROTTLE_BRAKE_VALUES)

#print(ACTION_DICT)

