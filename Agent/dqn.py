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
from Memory.memory_buffer import MemoryBuffer,OfflineMemoryBuffer

import matplotlib.pyplot as plt
import random
import os, os.path
# from google.colab.patches import cv2_imshow
import h5py
import numpy as np
import re
from enum import Enum

class DDQN():

    def __init__(self, imitation_data_directory = '',validation_data_directory = '', output_directory = '', epochs = 1, number_minibatches = 16, save_every = 2, start_epoch = 1):

        self.epochs = epochs
        self.number_minibatches = number_minibatches
        self.start_epoch = save_every
        self.start_epoch = start_epoch
        self.save_every =save_every
        self.batch_size = 32
        #self.current_folder_generator = 0
        #self.validation_current_folder_generator = 0
        self.dropout_count = 0
        self.conv_count = 0
        self.bn_count =  0
        self.pool_count = 0 
        self.fc_count = 0
        self.data_directory = imitation_data_directory
        self.imitation_data_directory = imitation_data_directory
        self.validation_data_directory = validation_data_directory
        self.output_directory = output_directory   
        self.branch_names = ['left','right','follow','straight']
        self.initialize_buffers() # initializing offline buffers
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

        #self.files_list, self.batch_size, self.scenario_length, self.image_dimension = self.getter()

    def initialize_buffers(self):
        branches = ['left','right','follow', 'straight']
        self.rl_offline_buffers = {}
        self.rl_online_buffers = {}
        self.imitation_offline_buffers = {}
        self.imitation_online_buffers = {}
        #creating buffers offline and online for all branches
        for branch in branches:
            #self.rl_online_buffers[branch] = MemoryBuffer(buffer_size = 1000, with_per = True , name = branch, directory = imitation_data_directory)
            #self.rl_offline_buffers[branch] = MemoryBuffer(buffer_size = 100000, with_per = True, name = branch, directory = imitation_data_directory)
            #self.imitation_online_buffers[branch] = MemoryBuffer(buffer_size = 1000, with_per = True, name = branch, directory = imitation_data_directory)
            self.imitation_offline_buffers[branch] = OfflineMemoryBuffer(buffer_size = 10000, with_per = True,
                                                                         name = branch,train_data_directory = self.imitation_data_directory,
                                                                         validation_data_directory = self.imitation_data_directory)



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
    
    def fc(self, input_layer,n_neurons, activation = 'relu',kernel_initializer = 'he_normal', drop_out = 0,name=None):
        layer =  Dense(n_neurons, activation = 'relu',kernel_initializer = 'he_normal',name=name)(input_layer)
        if drop_out:
            if name is None:
                layer = self.dropout(layer,drop_out)
            else:
                layer = self.dropout(layer,drop_out,name=name+'_drop_out')
        return layer
    
    def concat(self,input_layer_1 , input_layer_2,name=None):
        return Concatenate(name=name)([input_layer_1, input_layer_2])
        
    
    def get_branched_network(self,input_shape):
        image = Input(input_shape,name='image_input')
        #print(image)
        layer = self.conv_block(image,5 , 2, 32, padding ='valid', drop_out = 0.2, name ='CONV1')
        
        #print(layer)
        layer = self.conv_block(layer,3 , 1, 32, padding ='valid', drop_out = 0.2, name ='CONV2')
        
        #print(layer)
        layer = self.conv_block(layer,3 , 2, 64, padding ='valid', drop_out = 0.2, name ='CONV3')
        #print(layer)
        layer = self.conv_block(layer,3 , 1, 64, padding ='valid', drop_out = 0.2, name ='CONV4')
        #print(layer)
        layer = self.conv_block(layer,3 , 2, 128, padding ='valid', drop_out = 0.2, name ='CONV5')
        #print(layer)
        layer = self.conv_block(layer,3 , 1, 128, padding ='valid', drop_out = 0.2, name ='CONV6')
        #print(layer)
        layer = self.conv_block(layer,3 , 2, 256, padding ='valid', drop_out = 0.2, name ='CONV7')
        #print(layer)
        layer = self.conv_block(layer,3 , 1, 256, padding ='valid', drop_out = 0.2, name ='CONV8')
        #print(layer)
        layer = self.flatten(layer)
        #print(layer)
        layer = self.fc(layer, 512, drop_out = 0.5, name ='CONV_FC1')
        #print(layer)
        layer = self.fc(layer, 512, drop_out = 0.5, name ='CONV_FC2')
        
        # Speed sensory input
        speed=(1,) # input layer'
        speed_input = Input(speed,name='speed_input')
        
        layer_speed =  self.fc(speed_input, 128, drop_out = 0.5, name ='SPEED_FC1')
        layer_speed =  self.fc(layer_speed, 128, drop_out = 0.5, name ='SPEED_FC2')
        
        middle_layer = self.concat(layer,layer_speed, name ='CONCAT_FC1')
        #print(middle_layer)
        
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
                branch_output = self.fc(middle_layer, 256, drop_out = 0.5, name =branches_names[i]+'_FC1')
                branch_output = self.fc(branch_output, 256, drop_out = 0.5, name =branches_names[i]+'_FC2')
                branch_output = self.fc(branch_output, 25, drop_out = 0, name =output_branches_names[(i)])
                branches[output_branches_names[i]] = branch_output
                
            else:
                #only used images feature vector for predicting speed
                #TODO try both speed and images feature vector
                branch_output = self.fc(layer, 256, drop_out = 0.5,name =branches_names[i]+'_FC1')
                branch_output = self.fc(branch_output, 256, drop_out = 0.5,name =branches_names[i]+'_FC2')
                branch_output = self.fc(branch_output, 1, drop_out = 0, name =output_branches_names[-1])
                branches[output_branches_names[-1]] =  branch_output 
                
        return Model(inputs = [image, speed_input],outputs = [branches['left_branch'],
                                                                   branches['right_branch'],
                                                                   branches['follow_branch'],
                                                                   branches['str_branch'],
                                                                   branches['speed_branch_output']])

        #print(self.model.summary()) 
        print("Building the model")
    def get_model(self):

        if (self.scenario_length == 1):    
            self.input_shape = self.image_dimension
        else:
            self.input_shape =  (self.scenario_length,) + self.image_dimension # Something like (32,5,200,200,3)
            
        print("Input shape to the network ", self.input_shape)   
        return self.get_branched_network(self.input_shape)
        #self.compile_model()            


    def rl_loss(self, r_s_a, q_next_s_next_a, q_s_a):
        """
        #reward + discount factor * aaction value for best action in the target network - action value for current action in the online network all squared        I/P : 
               r_s_a -> reward for current state and action
               q_next_s_next_a -> predicted state action value functions for next states and best action calculated from target network
               q_s_a  -> predicted state action value functions for demonestrated action
        O/P : rl loss
        """
        gamma = 0.99
        #max over q_s_a
        #calculate q_s_ae
        batch_size = 32
        q_s_a_temp = q_s_a
        for i in range(4):
            for j in range(batch_size):
                q_next_s = np.max(q_next_s_next_a[i][i*batch_size+j])
                action_token = int(np.argmax(q_s_a_temp[i][i*batch_size+j]))
                #print(demonestration_action)
                #print(action_token)
                q_s_a_temp[i][i*batch_size+j][action_token] = (r_s_a + q_next_s - q_s_a_temp[i][i*batch_size+j][action_token]) 

        return q_s_a_temp
        
        
        pass
    def supervised_loss(self,q_s_a,ae):
        """
        #max of current ation value + 0.8 - action value of the right action from demonestrations
        I/P : 
               q_s_a -> predicted state action value functions,list of shape [5,128,25]
               ae - > numpy array of shape 128
        O/P : spervised loss
        """
        l_ae_a = 0.8
        #max over q_s_a
        #calculate q_s_ae
        batch_size = 32
        q_s_a_temp = q_s_a
        for i in range(4):
            for j in range(batch_size):
                demonestration_action = int(ae[i*batch_size+j])
                action_token = int(np.argmax(q_s_a[i][i*batch_size+j]))
                #print(demonestration_action)
                #print(action_token)
                q_s_a_temp[i][i*batch_size+j][action_token] += (l_ae_a - q_s_a[i][i*batch_size+j][demonestration_action]) 

        return q_s_a_temp

    def train_agent(self):
        """
            This model take care of all model's training stuff
            1- fetch mini batches for training
            2- calculate state action value function for current state and next states
            3- calculate td errors
            4- change priorities
            5- calculate supervised loss and rl loss 
        """
        training_states_batch = np.zeros(shape = (128, 88, 200, 3))
        training_next_states_batch = np.zeros(shape = (128, 88, 200, 3))
        actions = np.zeros(shape = (128))
        next_actions = np.zeros(shape = (128))
        speed = np.zeros(shape = (128))
        next_speed = np.zeros(shape = (128))

        for i in range(4):
            #training_states_batch[i*32:(i+1)*32], training_next_states_batch[i*32:(i+1)*32], actions[i*32:(i+1)*32], next_actions[i*32:(i+1)*32] = self.imitation_offline_buffers[self.branch_names[i]].data_handler.fetch_minibatch(branch_name = self.branch_names[i] , number_of_files =100)
            training_states_batch[i*32:(i+1)*32], training_next_states_batch[i*32:(i+1)*32], actions[i*32:(i+1)*32], next_actions[i*32:(i+1)*32], speed[i*32:(i+1)*32] , next_speed[i*32:(i+1)*32] = self.imitation_offline_buffers[self.branch_names[i]].fetch(32)

        # print(training_states_batch.shape)
        q_s_a = self.model.predict([training_states_batch,speed])   
        q_next_s_next_a = self.target_model.predict([training_next_states_batch,next_speed])
        
        y_true = self.rl_loss(0,q_next_s_next_a,q_s_a) + self.supervised_loss(q_s_a,actions)
        #model.fit([input,true_output])

    def masked_loss_function(self, y_true, y_pred):
        mask_value=-2
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())     
        return keras.losses.mean_squared_error(y_true * mask, y_pred * mask)
    
    def compile_model(self,model):
        opt = Adam(lr=0.0002, beta_1=0.7, beta_2=0.85, decay=1e-6)

        model.compile(optimizer = opt, loss ={'left_branch': self.masked_loss_function,
                                                                     'right_branch': self.masked_loss_function,
                                                                     'follow_branch': self.masked_loss_function,                                                                                      
                                                                     'str_branch': self.masked_loss_function,
                                                                     'speed_branch_output': self.masked_loss_function} )
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
        #self.visualizeTraining(validation_images, validation_labels)




    def update_target_model(self):
        """
            This fucntion transfer online network weights to target network
        """
        self.target_model.set_weights(self.model.get_weights())
        pass
        #self.target_model.set_weights(self.model.get_weights())

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
