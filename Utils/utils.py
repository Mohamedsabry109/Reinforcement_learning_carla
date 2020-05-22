import numpy as np
import h5py

#from .utils import *
#from .config import *
#rom .config import ACTION_NUMBER_TO_VALUES


def save_stats():
    """
        This function takes some states and saves them in some directory
        I/P:
        
    """
    pass

def get_comb(steer_range, throttle_range):
    c = {}
    action_number_to_values = {}
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
            action_number_to_values[action_number] = [filtered_i, filtered_j]
            c[key] = action_number
            action_number += 1  
    return c, action_number_to_values

def save():
    pass

def heatup(env,agent,exp_policy,ACTION_NUMBER_TO_VALUES,num_episodes = 1):

    for i in range(num_episodes+1):
        print("Heatup Iteration ",i)
        episode_data = {'states':[],'actions':[],'reward':[],'done':[]}
        env.reset()
        high_level_command = env.state['high_level_command']
        while env.done == False:
            actions = agent.model.predict([ np.expand_dims(env.state['segmentation'],axis = 0),np.array([env.state['measurements'][0]])])
            #action = np.argmax(actions[high_level_command])
            #TODO Choose the right action according to our network
            action = exp_policy.choose_action(actions[high_level_command])
            #we need to map action to acc and steer
            steer, acc_brake = ACTION_NUMBER_TO_VALUES[action]
            if acc_brake >= 0 :
                acc = acc_brake
                brake  = 0

            else:
                brake = abs(acc_brake)
                acc = 0

            #steer, throttle, brake
            #env.step([0,1,0])
            env.step([steer,acc,brake])

            episode_data['states'].append(env.state)
            episode_data['actions'].append(actions)
            episode_data['reward'].append(env.reward)
            episode_data['done'].append(env.done)
            #episode_data.append((env.state,env.done))
        #save done == True data
        #call save_interaction_data
        save_interaction_data(episode_data,agent)


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
        command_dict = {0:'follow',1:'left',2:'right',3:'straight'}
        current_high_level_command = data['states'][i]['high_level_command']
        files_tracker = agent.rl_offline_buffers[command_dict[current_high_level_command]].files_tracker
        state = np.expand_dims(data['states'][i]['segmentation'],axis=0)
        velocity = np.array(data['states'][i]['measurements'][0])
        reward = np.array(data['reward'][i])
        done = np.array(data['done'][i])
        #targets = {'velocity':velocity,'reward':reward,'done':done}
        #follow data
        if current_high_level_command == 0:
            action = np.argmax(data['actions'][i][2])
            #targets['actions'] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory+'/'+command_dict[current_high_level_command]+'/'+'data_'+str(current_file_number)
            agent.rl_offline_buffers[command_dict[current_high_level_command]].files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(file_name, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                hdf.create_dataset('velocity', data=velocity)
                hdf.create_dataset('actions', data=action)
                hdf.create_dataset('reward', data=reward)
                hdf.create_dataset('done', data=done)
            pass
        #left data
        elif current_high_level_command == 1:
            action = np.argmax(data['actions'][i][0])
            targets['actions'] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory+'/'+command_dict[current_high_level_command]+'/'+'data_'+str(current_file_number)
            agent.rl_offline_buffers[command_dict[current_high_level_command]].files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(file_name, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                #hdf.create_dataset('targets', data=targets)
                hdf.create_dataset('velocity', data=velocity)
                hdf.create_dataset('actions', data=action)
                hdf.create_dataset('reward', data=reward)
                hdf.create_dataset('done', data=done)
            pass
        #right data
        elif current_high_level_command == 2:
            action = np.argmax(data['actions'][i][1])
            targets['actions'] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory+'/'+command_dict[current_high_level_command]+'/'+'data_'+str(current_file_number)
            agent.rl_offline_buffers[command_dict[current_high_level_command]].files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(file_name, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                #hdf.create_dataset('targets', data=targets)
                hdf.create_dataset('velocity', data=velocity)
                hdf.create_dataset('actions', data=action)
                hdf.create_dataset('reward', data=reward)
                hdf.create_dataset('done', data=done)
            pass
        # straight data
        elif current_high_level_command == 4:
            action = np.argmax(data['actions'][i][3])
            targets['actions'] = action
            current_file_number =  int(files_tracker.split("_")[1].split(".")[0])+1
            file_name = agent.rl_data_directory+'/'+command_dict[current_high_level_command]+'/'+'data_'+str(current_file_number)
            agent.rl_offline_buffers[command_dict[current_high_level_command]].files_tracker = 'data_'+str(current_file_number)+'.h5'
            with h5py.File(file_name, 'w') as hdf:
                hdf.create_dataset('rgb', data=state)
                #hdf.create_dataset('targets', data=targets)
                hdf.create_dataset('velocity', data=velocity)
                hdf.create_dataset('actions', data=action)
                hdf.create_dataset('reward', data=reward)
                hdf.create_dataset('done', data=done)
        else:
            continue

    
