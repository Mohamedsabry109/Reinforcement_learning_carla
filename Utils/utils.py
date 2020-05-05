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

def save():
    pass

def save_interaction_data(data, agent):
    """
         Saving Step's data in the interaction data directory while keep tracking the last file 
         Also we need to save done flag and use it while loading to avoid stacking two nonconsecutive states
        Args: data dict -> {'states':[],'actions':[],'reward':[],'done':[]}
            states -> {'state':,'high_level_command','measurments'} and done flag
            directions or high level command is -> 0: follow , 1: left, 2: right, 3: straight

        Returns: None
    """


    for i in range(len(data['done'])):#loop on all steps' data
        #saving data
        command_dict = [0:'follow',1:'left',2:'right',3:'straight']
        files_tracker = 
        current_high_level_command = data[i]['states']['high_level_command']
        state = data[i]['states']['segmentation']
        velocity = data[i]['states']['measurments'][0]
        reward = data[i]['reward']
        done = data[i]['done']
        targets = {'velocity':velocity,'reward':reward,'done':done}
        #follow data
        if current_high_level_command == 0:
            action = np.argmax(data[i]['actions']['high_level_command'][2])
            targets[actions] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory'/'+command_dict[current_high_level_command]+'/'+'data_'+current_file_number
            files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(filename, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                hdf.create_dataset('targets', data=targets)

            pass
        #left data
        elif current_high_level_command == 1:
            action = np.argmax(data[i]['actions']['high_level_command'][0])
            targets[actions] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory'/'+command_dict[current_high_level_command]+'/'+'data_'+current_file_number
            files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(filename, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                hdf.create_dataset('targets', data=targets)
            pass
        #right data
        elif current_high_level_command == 2:
            targets[actions] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory'/'+command_dict[current_high_level_command]+'/'+'data_'+current_file_number
            files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(filename, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                hdf.create_dataset('targets', data=targets)
            action = np.argmax(data[i]['actions']['high_level_command'][1])
            pass
        # straight data
        elif current_high_level_command == 4:
            targets[actions] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory'/'+command_dict[current_high_level_command]+'/'+'data_'+current_file_number
            files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(filename, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                hdf.create_dataset('targets', data=targets)
            action = np.argmax(data[i]['actions']['high_level_command'][3])

        else:
            continue

    pass
