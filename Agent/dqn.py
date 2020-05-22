import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Conv3D, LSTM, Conv1D, Multiply, TimeDistributed
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

# gpu_fraction = 0.9
# per_process_gpu_memory_fraction=gpu_fraction,
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# %matplotlib inline

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# K.set_session(sess)

from Memory.memory_buffer import MemoryBuffer,OfflineMemoryBuffer, OnlineMemoryBuffer
import matplotlib.pyplot as plt
import random
import os, os.path
import h5py
import numpy as np
import re
from enum import Enum
import re
import cv2
from PIL import Image
import IPython




class DDQN():

    def __init__(self, imitation_data_directory = '',validation_data_directory = '',rl_data_directory = '', output_directory = '', epochs = 1, number_minibatches = 16, save_every = 2, start_epoch = 1):

        self.epochs = epochs
        self.number_minibatches = number_minibatches
        self.start_epoch = save_every
        self.start_epoch = start_epoch
        self.save_every =save_every
        self.batch_size = 32
        self.dropout_count = 0
        self.conv_count = 0
        self.bn_count =  0
        self.pool_count = 0 
        self.fc_count = 0
        self.data_directory = imitation_data_directory
        self.imitation_data_directory = imitation_data_directory
        self.rl_data_directory = rl_data_directory
        self.validation_data_directory = validation_data_directory
        self.output_directory = output_directory   
        self.branch_names = ['left','right','follow','straight']

        self.initialize_buffers() # initializing offline buffers

        self.imitation_online_buffers['left'].reload(self.imitation_offline_buffers['left'])
        self.imitation_online_buffers['right'].reload(self.imitation_offline_buffers['right'])
        self.imitation_online_buffers['follow'].reload(self.imitation_offline_buffers['follow'])
        self.imitation_online_buffers['straight'].reload(self.imitation_offline_buffers['straight'])

        #batch = self.imitation_online_buffers['left'].sample_batch(4)
        # states, actions, rewards, dones, new_states, idxs = batch
        # print("shape of states ", states[0][0].shape)
        # print("shape of actions ", actions.shape)
        # print("shape of rewards ", rewards.shape)
        # print("shape of dones ", dones.shape)
        # print("shape of new_states ", new_states.shape)

        # for i in range(states.shape[0]):

        #     print("shape of idxs ", idxs.shape)
        #     print("action of sora ",actions[i])
        #     cv2.imshow("sora ",states[i][0][0])
        #     cv2.imshow("next sora ",new_states[i][0][0])
        #     cv2.waitKey()

        # for i in range(self.imitation_offline_buffers['left'].size()):
        #     print(self.imitation_offline_buffers['left'].buffer.get(i))
        ##print(imitation_offline_buffers['left'])
        #print(len(self.imitation_online_buffers['left'].buffer.get(0)))
        # idx , error , experience = self.imitation_online_buffers['left'].buffer.get(250)
        # state , action , reward , done , new_state  = experience
        # print('state ', state)
        # print('action ', action)
        # print('reward ', reward)
        # print('done ',done)
        # print('new states ', new_state)
        # print(state[0][0].shape)
        # cv2.imshow("sora",state[0][0])
        # cv2.imshow("next sora ",new_state[0][0])
        # cv2.waitKey()

        self.getter()
        self.model = self.get_model()
        self.compile_model(self.model)
        self.target_model = self.get_model()
        self.compile_model(self.target_model)
        print(self.model)
        print(self.target_model)
        #creating folder for weights
        if not (os.path.isdir(self.output_directory + 'Weights')):
            os.mkdir(self.output_directory + 'Weights')
        else:
            last_weight = [sorted(os.listdir(self.output_directory + 'Weights/'))][-1]
            print("Weights folder exist, continue from epoch:", self.start_epoch)
        if not (os.path.isdir(self.output_directory + 'TensorBoard')):
            os.mkdir(self.output_directory + 'TensorBoard')
        if not (os.path.isdir(self.output_directory + 'SteeringAngels')):
            os.mkdir(self.output_directory + 'SteeringAngels')

    # def __str__(self):

    #     return str(self.imitation_online_buffers['left'].buffer[0])

    def initialize_buffers(self):

        branches = ['left','right','follow', 'straight']
        self.rl_offline_buffers = {}
        self.rl_online_buffers = {}
        self.imitation_offline_buffers = {}
        self.imitation_online_buffers = {}
        #creating buffers offline and online for all branches
        for branch in branches:
            self.rl_online_buffers[branch] = OnlineMemoryBuffer(buffer_size = 1000, with_per = True,
                                                                         name = branch,train_data_directory = self.rl_data_directory,
                                                                         validation_data_directory = self.rl_data_directory)
            self.rl_offline_buffers[branch] = OfflineMemoryBuffer(buffer_size = 10000, with_per = True,
                                                                         name = branch,train_data_directory = self.rl_data_directory,
                                                                         validation_data_directory = self.rl_data_directory)
            #print(self.rl_offline_buffers[branch].files_tracker)
            self.imitation_online_buffers[branch] = OnlineMemoryBuffer(buffer_size = 83, with_per = True,
                                                                         name = branch,train_data_directory = self.imitation_data_directory,
                                                                         validation_data_directory = self.imitation_data_directory)
            self.imitation_offline_buffers[branch] = OfflineMemoryBuffer(buffer_size = 10000, with_per = True,
                                                                         name = branch,train_data_directory = self.imitation_data_directory,
                                                                         validation_data_directory = self.imitation_data_directory)

    def imshow(self,img):
        _,ret = cv2.imencode('.jpg',img)
        i = IPython.display.Image(data=ret)
        IPython.display.display(i)


    def getter(self):
        '''
        This function gets the shape of the input image as well as the scenario size
        self.images_per_h5_file
        self.imag_dim
        '''
        files_list = sorted(os.listdir(self.data_directory+'/'+'follow'))
        current_directory = self.data_directory + '/' + 'follow'+ '/' + files_list[0]
        with h5py.File(current_directory, 'r') as hdf:
            imgs = hdf.get('rgb') # ALL THE 200 IMAGES IN THE H5 FILE
            imgs = np.array(imgs[:,:,:], dtype = np.uint8)
            image_shape = imgs.shape
            targets = hdf.get('targets') # ALL THE 200 TARGETS IN THE H5 FILE
            targets = np.array(targets)

            self.image_dimension = image_shape[-3:]
            print("Input to model shape is ",self.image_dimension)
            self.scenario_length = image_shape[0]
        
            # print("INPUT SHAPE IS:", self.image_dimension)
            # print("scenrion length ",self.scenario_length)

        # if len(image_shape) == 4:
        #     #single frames
        #     print("Single frames data")
        #     batch_size = image_shape[0]
        #     image_dimension = image_shape[1:]
        #     scenario_length = 1
        # elif len(image_shape) == 5:
        #     #stacked frames
        #     print("stacked frames data")
        #     batch_size = image_shape[0]
        #     scenario_length = image_shape[1]
        #     image_dimension = image_shape[2:]
        
        # return files_list, batch_size, scenario_length, image_dimension    
    
    def conv(self,input_layer, kernel_size, stride, n_filters ,padding='same',data_format = "channels_last", activation ='relu', time_stride = 0, kernel_size_time = 0 ,name=None):
        self.conv_count +=1
        return Conv2D(filters= n_filters, kernel_size=(kernel_size,kernel_size), strides=(stride,stride),padding=padding, data_format=data_format, activation=activation,name =name)(input_layer)

    def bn(self, input_layer,name=None):
        self.bn_count += 1
        return  BatchNormalization(name=name)(input_layer)  

    def dropout(self, input_layer, drop_out,name=None):
        self.dropout_count += 1
        return Dropout(drop_out,name=name)(input_layer)


    def conv_block(self,input_layer, n_filters, kernel_size, stride, padding='same',data_format = "channels_last", activation ='relu', time_stride = 0, kernel_size_time = 0,batch_norm = True, drop_out = 0, name=None):
        layer = self.conv(input_layer, n_filters, kernel_size, stride, padding='same',data_format = "channels_last", activation ='relu', time_stride = 0, kernel_size_time = 0,name=name)
        if batch_norm:
            if name is None:
                layer = self.bn(layer)
            else: 
                layer = self.bn(layer, name=name+'_batch_norm')
        if drop_out:
            if name is None:
                layer = self.dropout(layer,drop_out)
            else:
                layer = self.dropout(layer,drop_out, name=name+'_dropout')
        return layer
        
    def flatten(self,input_layer,name=None):
        return Flatten(name=name)(input_layer)
    
    def fc(self, input_layer,n_neurons, activation = 'relu',kernel_initializer = 'glorot_normal',batch_norm= False, drop_out = 0,name=None):
        layer =  Dense(n_neurons, activation = 'relu',kernel_initializer = 'glorot_normal',name=name)(input_layer)
        if drop_out:
            if name is None:
                layer = self.dropout(layer,drop_out)
            else:
                layer = self.dropout(layer,drop_out,name=name+'_drop_out')
        if batch_norm:
            if name is None:
                layer = self.bn(layer)
            else: 
                layer = self.bn(layer, name=name+'_batch_norm')
        return layer
    
    def concat(self,input_layer_1 , input_layer_2,name=None):
        return Concatenate(name=name)([input_layer_1, input_layer_2])
        
    
    def get_branched_network(self,input_shape):
        image = Input(input_shape,name='image_input')
        print(image)
        layer = self.conv_block(image,5 , 3, 16, padding ='valid',batch_norm=False, drop_out = 0, name ='CONV1')       
        #print(layer)
        layer = self.conv_block(layer,3 , 3, 16, padding ='valid',batch_norm=False, drop_out = 0, name ='CONV2')
        #print(layer)
        layer = self.conv_block(layer,3 , 2, 32, padding ='valid',batch_norm=False, drop_out = 0, name ='CONV3')
        #print(layer)
        layer = self.conv_block(layer,3 , 2, 32, padding ='valid',batch_norm=False, drop_out = 0, name ='CONV4')
        # #print(layer)
        # layer = self.conv_block(layer,3 , 2, 64, padding ='valid',batch_norm=True, drop_out = 0, name ='CONV5')
        # # #print(layer)
        # # layer = self.conv_block(layer,3 , 1, 128, padding ='valid', drop_out = 0, name ='CONV6')
        # # #print(layer)
        # # layer = self.conv_block(layer,3 , 2, 256, padding ='valid', drop_out = 0, name ='CONV7')
        # # #print(layer)
        # # layer = self.conv_block(layer,3 , 1, 256, padding ='valid', drop_out = 0, name ='CONV8')
        # #print(layer)
        layer = self.flatten(layer)
        # #print(layer)
        # layer = self.fc(layer, 512,batch_norm=True, drop_out = 0, name ='CONV_FC1')
        # #print(layer)
        # #layer = self.fc(layer, 512, drop_out = 0, name ='CONV_FC2')     

        # mobile_net = tf.keras.applications.MobileNetV2(input_shape=input_shape,
        #                                                include_top=False,
        #                                                weights='imagenet')
        # mobile_net.trainable = False
        # layer = mobile_net(image)
        # layer = keras.layers.Flatten()(layer)

        # Speed sensory input
        speed=(1,) # input layer'
        speed_input = Input(speed,name='speed_input')
        
        layer_speed =  self.fc(speed_input, 128 ,batch_norm=True, drop_out = 0, name ='SPEED_FC1')
        layer_speed =  self.fc(layer_speed, 128 ,batch_norm=True, drop_out = 0, name ='SPEED_FC2')
        
        middle_layer = self.concat(layer,layer_speed, name ='CONCAT_FC1')
        #print(middle_layer)
        middle_layer = self.fc(middle_layer, 256,batch_norm=True, drop_out = 0, name ='MIDDLE_FC')

        branches_names = ['follow', 'left', 'right', 'straight' , 'speed']
        
        #TODO support for multi speed layers
        branches = {}
        output_branches_names = ['left_branch',
                                 'right_branch',
                               'follow_branch',
                                'str_branch',
                               'speed_branch_output']
        
        for i, branch_name in enumerate(branches_names):
            if branch_name != 'speed':
                #branche for control signals
                branch_output = self.fc(middle_layer, 128,batch_norm=True, drop_out = 0, name =branches_names[i]+'_FC1')
                #branch_output = self.fc(branch_output, 256,batch_norm=True, drop_out = 0, name =branches_names[i]+'_FC2')
                branch_output = self.fc(branch_output, 441,batch_norm=False, drop_out = 0, name =output_branches_names[(i)])
                branches[output_branches_names[i]] = branch_output
                
            else:
                #only used images feature vector for predicting speed
                #TODO try both speed and images feature vector
                branch_output = self.fc(layer, 128,batch_norm=True, drop_out = 0,name =branches_names[i]+'_FC1')
                #branch_output = self.fc(branch_output, 256,batch_norm=True, drop_out = 0,name =branches_names[i]+'_FC2')
                branch_output = self.fc(branch_output, 1,batch_norm=False, drop_out = 0, name =output_branches_names[-1])
                branches[output_branches_names[-1]] =  branch_output 
                
        


        model =  Model(inputs = [image, speed_input],outputs = [branches['left_branch'],
                                                                   branches['right_branch'],
                                                                   branches['follow_branch'],
                                                                   branches['str_branch'],
                                                                   branches['speed_branch_output']])

        print(model.summary()) 
        print("Building the model")

        return model

    def get_model(self):

        if (self.scenario_length == 1):    
            self.input_shape = self.image_dimension
        else:
            self.input_shape =  (self.scenario_length,) + self.image_dimension # Something like (32,5,200,200,3)
            
        print("Input shape to the network ", self.input_shape)   
        return self.get_branched_network(self.input_shape)


    def rl_loss(self, r_s_a, q_next_s_a_online, q_next_s_a_target, q_s_a, actions):
        """
        #reward + discount factor * aaction value for best action in the target network - action value for current action in the online network all squared        I/P : 
               r_s_a -> reward for current state and action
               q_next_s_next_a -> predicted state action value functions for next states and best action calculated from target network
               q_s_a  -> predicted state action value functions for demonestrated action
        O/P : rl loss
        """
        gamma = 0.99
        branched_batch_size = self.branched_batch_size
        #q_s_a_temp = np.zeros((len(q_s_a),np.array(q_s_a[0]).shape[0],np.array(q_s_a[0]).shape[1]))
        q_s_a_temp = q_s_a
        for i in range(4):
            for j in range(branched_batch_size):
                #q_next_s = np.max(q_next_s_a_online[i][i*branched_batch_size+j])
                #action_token = int(np.argmax(q_s_a_temp[i][i*branched_batch_size+j]))
                
                a_t_1_max = int(np.argmax(q_next_s_a_online[i][i*branched_batch_size+j]))
                
                #q_s_a_temp[i][i*batch_size+j][action_token] = (r_s_a[batch_size*i+j] + gamma*q_next_s - q_s_a[i][i*batch_size+j][action_token])**2 
                #q_s_a_temp[i][i*branched_batch_size+j] = (r_s_a[branched_batch_size*i+j] + gamma*q_next_s_a_target[i][i*branched_batch_size+j] - q_s_a[i][i*branched_batch_size+j])**2 
                q_s_a_temp[i][i*branched_batch_size+j][int(actions[i*branched_batch_size+j])] = (r_s_a[branched_batch_size*i+j] \
                                                                                            + gamma*q_next_s_a_target[i][i*branched_batch_size+j][a_t_1_max] \
                                                                                            - q_s_a[i][i*branched_batch_size+j][int(actions[i*branched_batch_size+j])])**2 

        return q_s_a_temp

    # def supervised_loss(self,q_s_a,ae):
    #     """
    #     #max of current ation value + 0.8 - action value of the right action from demonestrations
    #     I/P : 
    #            q_s_a -> predicted state action value functions,list of shape [5,128,25]
    #            ae - > numpy array of shape 128 = 16
    #     O/P : spervised loss
    #     """
    #     l_ae_a = 0.8
    #     batch_size = self.branched_batch_size
    #     q_s_a_temp = np.zeros((len(q_s_a),np.array(q_s_a[0]).shape[0],np.array(q_s_a[0]).shape[1]))
    #     error = np.zeros(ae.shape)
    #     q_s_a_temp = q_s_a
    #     true_actions = 0
    #     for i in range(4):
    #         for j in range(batch_size):
    #             demonestration_action = int(ae[i*batch_size+j]) # 0 -> 441
    #             action_token = int(np.argmax(q_s_a[i][i*batch_size+j]))
    #             error[i*batch_size+j] = np.abs(q_s_a[i][i*batch_size+j][action_token] + (l_ae_a - q_s_a[i][i*batch_size+j][demonestration_action]))
    #             #q_s_a_temp[i][i*batch_size+j][action_token] = q_s_a[i][i*batch_size+j][action_token] + (l_ae_a - q_s_a[i][i*batch_size+j][demonestration_action])
    #             q_s_a_temp[i][i*batch_size+j] = q_s_a[i][i*batch_size+j] + (l_ae_a - q_s_a[i][i*batch_size+j][demonestration_action])
    #             q_s_a_temp[i][i*batch_size+j][demonestration_action] -= l_ae_a 
    #             if (action_token == demonestration_action):
    #                 print("#################################################################################################")
    #                 true_actions += 1
    #     self.acc.append(true_actions *100 / (4*batch_size))
    #     return q_s_a_temp, error


    def supervised_loss(self,q_s_a,ae):
        """
        #max of current ation value + 0.8 - action value of the right action from demonestrations
        I/P : 
               q_s_a -> predicted state action value functions,list of shape [5,128,25]
               ae - > numpy array of shape 128 = 16
        O/P : spervised loss
        """
        l_ae_a = 0.8
        batch_size = self.branched_batch_size
        l_eq = np.zeros((5,1))

        q_s_a_temp = np.zeros((len(q_s_a),np.array(q_s_a[0]).shape[0],np.array(q_s_a[0]).shape[1]))
        error = np.zeros(ae.shape)
        q_s_a_temp = q_s_a
        true_actions = 0
        for i in range(4):
            j_e = 0
            for j in range(batch_size):
                
                demonestration_action = int(ae[i*batch_size+j]) # 0 -> 441
                
                action_token = int(np.argmax(q_s_a[i][i*batch_size+j]))
                
                error[i*batch_size+j] = np.abs(q_s_a[i][i*batch_size+j][action_token] + (l_ae_a - q_s_a[i][i*batch_size+j][demonestration_action]))
                
                #q_s_a_temp[i][i*batch_size+j][action_token] = q_s_a[i][i*batch_size+j][action_token] + (l_ae_a - q_s_a[i][i*batch_size+j][demonestration_action])
                if demonestration_action != action_token:
                    j_e += q_s_a[i][i*batch_size+j][action_token] + (l_ae_a - q_s_a[i][i*batch_size+j][demonestration_action])
                #q_s_a_temp[i][i*batch_size+j][demonestration_action] -= l_ae_a 

                if (action_token == demonestration_action):
                    print("#################################################################################################")
                    true_actions += 1

            l_eq[i] = j_e / batch_size
        self.acc.append(true_actions *100 / (4*batch_size))

        return l_eq, error

    def train_agent(self):
        """
            This model take care of all model's training stuff
            1- fetch mini batches for training
            2- calculate state action value function for current state and next states
            3- calculate td errors
            4- change priorities
            5- calculate supervised loss and rl loss 
        """

        loss = []
        self.acc = []
        self.batch_size = 32
        self.branched_batch_size = self.batch_size // 4
        
        for iteration in range(1000):

            print("*********************************Start Training iteration ******************************************* ")
            left_batch = self.imitation_online_buffers['left'].sample_batch(self.branched_batch_size)
            right_batch = self.imitation_online_buffers['right'].sample_batch(self.branched_batch_size)
            follow_batch = self.imitation_online_buffers['follow'].sample_batch(self.branched_batch_size)
            straight_batch = self.imitation_online_buffers['straight'].sample_batch(self.branched_batch_size)

            #states, actions, rewards, dones, new_states, idxs = batch
            batch = [left_batch, right_batch, follow_batch, straight_batch]
            training_states_batch = np.zeros(shape = (self.batch_size, 88, 200, 3))
            training_next_states_batch = np.zeros(shape = (self.batch_size, 88, 200, 3))
            actions = np.zeros(shape = (self.batch_size))
            speed = np.zeros(shape = (self.batch_size))
            next_speed = np.zeros(shape = (self.batch_size))
            reward = np.zeros(shape = (self.batch_size))
            idxs_left = left_batch[-1]
            idxs_right = right_batch[-1]
            idxs_follow = follow_batch[-1]
            idxs_straight = straight_batch[-1]
            idxs = [idxs_left,idxs_right,idxs_follow,idxs_straight]      
            for i in range(4):
                for j in range(self.branched_batch_size):
                    #print(batch[0][0].shape)
                    training_states_batch[i*self.branched_batch_size:i*self.branched_batch_size+j] = batch[i][0][:,0][j].squeeze(axis=0)
                    training_next_states_batch[i*self.branched_batch_size:i*self.branched_batch_size+j] = batch[i][4][:,0][j].squeeze(axis=0)
                    #training_next_states_batch[i*self.branched_batch_size:i*self.branched_batch_size+j] = batch[i][0][:,0][j].squeeze(axis=0)
                    actions[self.branched_batch_size*i+j] = batch[i][1][j]
                    speed[self.branched_batch_size*i+j] = batch[i][0][:,1][j]
                    next_speed[self.branched_batch_size*i+j] = batch[i][4][:,1][j]
                    reward[self.branched_batch_size*i+j] = batch[i][2][j]

            self.q_s_a = self.model.predict([training_states_batch,speed])
            self.q_next_s_a_online = self.model.predict([training_next_states_batch,speed])
            self.q_next_s_a_target = self.target_model.predict([training_next_states_batch,next_speed])          

            rl_loss = self.rl_loss(reward, self.q_next_s_a_online, self.q_next_s_a_target, self.q_s_a, actions)
            self.supervised_loss_, supervised_errors = self.supervised_loss(self.q_s_a, actions)
            modified_loss = rl_loss 
            
            self.updated_buffers(idxs,supervised_errors)
            hist = self.model.fit(x = [training_states_batch,speed], y = [modified_loss[0],modified_loss[1],modified_loss[2],modified_loss[3],modified_loss[4]])
            loss.append(hist.history["loss"])

            if (iteration % 100 == 0):
                print(" Training iteration ", iteration)
                self.update_target_model()

        #print(loss)    
        #plt.plot(loss[100:])
        plt.plot(loss)
        plt.show()
        plt.plot(self.acc)
        plt.show()


    def update_online_buffers(self,idxs,supervised_errors):
        self.imitation_online_buffers['left'].change_priorities(idxs[0],supervised_errors[:self.branched_batch_size])
        self.imitation_online_buffers['right'].change_priorities(idxs[1],supervised_errors[self.branched_batch_size:self.branched_batch_size*2])
        self.imitation_online_buffers['follow'].change_priorities(idxs[2],supervised_errors[self.branched_batch_size*2:self.branched_batch_size*3])
        self.imitation_online_buffers['straight'].change_priorities(idxs[3],supervised_errors[self.branched_batch_size*3:])

    def update_offline_buffer():
        """
            Updating priorites inside the offline buffer
            call buffer.get() on all buffer entires, save errors and then update all values, then reload
        """
        pass

    def masked_loss_function(self, y_true, y_pred):
        mask_value = 0
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())     
        return keras.losses.mean_squared_error(y_true*mask , y_pred * mask)


    def loss_function(self, y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred)

    def loss_function_left(self, y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred) 
        #+self.supervised_loss_[0]

    def loss_function_right(self, y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred) 

    def loss_function_follow(self, y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred) 

    def loss_function_straight(self, y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred) 

    def loss_function_speed(self, y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred) 


    def compile_model(self,model):
        opt = Adam(lr=0.01, beta_1=0.7, beta_2=0.85, decay=1e-6)
        self.supervised_loss_ = np.zeros((5,1))
        model.compile(optimizer = opt, loss ={'left_branch': self.loss_function_left,
                                             'right_branch': self.loss_function_right,
                                             'follow_branch': self.loss_function_follow,                                                                                      
                                             'str_branch': self.loss_function_straight,
                                             'speed_branch_output': self.loss_function_speed})
        print("Done compiling model!")
        return
  
    
       

    def train(self):
        train_directory = self.train_data_directory + 'follow'
        validation_directory = self.validation_data_directory + 'follow'
        number_of_files = len([name for name in os.listdir(train_directory) if os.path.isfile(os.path.join(train_directory, name))]) #Number of files
        print("Number of files is:", number_of_files, "In ", train_directory)
        validation_number_of_files = len([name for name in os.listdir(validation_directory) if os.path.isfile(os.path.join(validation_directory, name))]) #Number of files

        #xValidation, yValidation = self.fetch_validation(0,1)
        #validation_images, validation_labels = self.fetch_validation(0,1)
        #self.model.load_weights('/media/dell1/1.6TBVolume/RL/DeepLearningModel/LSTM10-2/Weights/' +   'weights00000032.h5')
        self.model.save(self.output_directory + '/Weights/' + 'BH1_Nvidia.h5')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.output_directory + 'TensorBoard', histogram_freq= 1, write_grads=True, write_images=True)
        mc = keras.callbacks.ModelCheckpoint(self.output_directory + '/Weights/' +  'weights{epoch:08d}.h5', 
                                                                         save_weights_only=True, period=self.save_every)
        print("Start Training...")
        self.generator_counter = 0
        self.validation_generator_counter = 0
        print("Start epoch {}".format(self.start_epoch))
        
#         hist1 = self.model.fit_generator(generator=self.fetch(self.number_minibatches,number_of_files), epochs = self.epochs+self.start_epoch, initial_epoch = self.start_epoch,
#                                                                      steps_per_epoch=number_of_files, verbose = 1, shuffle=True,
#                                                                      validation_data = (validation_images, validation_labels), validation_steps=1,
#                                                                      callbacks=[tensorboard, mc])
                
        hist1 = self.model.fit_generator(generator=self.fetch(self.number_minibatches,number_of_files), epochs = self.epochs+self.start_epoch, initial_epoch = self.start_epoch,
                                                                     steps_per_epoch=number_of_files, verbose = 1, shuffle=True, use_multiprocessing=False,
                                                                     validation_data = self.fetch_validation(self.number_minibatches,validation_number_of_files), validation_steps=int(validation_number_of_files/10),
                                                                     callbacks=[tensorboard, mc])




    def update_target_model(self):
        """
            This fucntion transfer online network weights to target network
        """
        self.target_model.set_weights(self.model.get_weights())


    def save_model(self):
        """
            This function save both target and online networks
            I/P : Models' Path
        """
        pass

    def load_model(self):
        """
            This function load both target and online networks
            I/P : Models' Path
        """
        pass
