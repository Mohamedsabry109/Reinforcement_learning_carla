import numpy as np
from .utils import *


def save_stats():
	"""
		This function takes some states and saves them in some directory
		I/P:
		
	"""
	pass

def get_comb(steer_range, throttle_range):
    c = {}
    action_number = 0
    for i in steer_range:
        for j in throttle_range:
            filtered_i = i
            filtered_j = j
            if abs(filtered_i) == 0:
           	 filtered_i = 0.0
            if abs(filtered_j) == 0:
                filtered_j = 0.0
            key = str(10*np.round(filtered_i,1)) +'_'+  str(10*np.round(filtered_j,1))
            c[key] = action_number
            action_number += 1  
    return c

 def save_interaction_data(data, agent):
 	"""
 		 Saving Step's data in the interaction data directory while keep tracking the last file 
 		 Also we need to save done flag and use it while loading to avoid stacking two nonconsecutive states
		Args: data dict -> {'state':,'high_level_command','measurments'} and done flag
			  directions or high level command is -> 0: follow , 1: left, 2: right, 3: straight

		Returns: None
 	"""

 	

 	pass
