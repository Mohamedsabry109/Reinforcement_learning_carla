import numpy as np 
from .utils import get_comb




QUANTIZATION_BINS = 21

STEER_RANGE = [-1, 1]
STEER_VALUES = np.linspace(STEER_RANGE[0], STEER_RANGE[1], num=QUANTIZATION_BINS)

THROTTLE_BRAKE_RANGE = [-1, 1]
THROTTLE_BRAKE_VALUES = np.linspace(THROTTLE_BRAKE_RANGE[0], THROTTLE_BRAKE_RANGE[1], num=QUANTIZATION_BINS)

ACTION_DICT, ACTION_NUMBER_TO_VALUES = get_comb(STEER_VALUES, THROTTLE_BRAKE_VALUES)
# print("\nAction dictionary ", ACTION_DICT)
# print("\nAction values ", ACTION_NUMBER_TO_VALUES)
command_dict = {0:'follow',1:'left',2:'right',3:'straight'}

