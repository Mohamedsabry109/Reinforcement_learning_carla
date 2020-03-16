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

import matplotlib.pyplot as plt
import random
import os, os.path
# from google.colab.patches import cv2_imshow
import h5py
import numpy as np
import re
from enum import Enum

class DDQN():

    def __init__(self, data_directory = '', train_data_directory = '',validation_data_directory = '', output_directory = '', epochs = 1, number_minibatches = 16, save_every = 2, start_epoch = 1):

        self.epochs = epochs
        self.number_minibatches = number_minibatches
        self.start_epoch = save_every
        self.start_epoch = start_epoch
        self.save_every =save_every
        self.current_folder_generator = 0
        self.validation_current_folder_generator = 0
        self.dropout_count = 0
        self.conv_count = 0
        self.bn_count =  0
        self.pool_count = 0 
        self.fc_count = 0
        
        self.data_directory = data_directory
        self.train_data_directory = train_data_directory
        self.validation_data_directory = validation_data_directory
        self.output_directory = output_directory        
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


    def getter(self):
        '''
        This function gets the shape of the input image as well as the scenario size
        self.images_per_h5_file
        self.imag_dim
        '''
        files_list = sorted(os.listdir(self.data_directory))
        current_directory = self.data_directory +  files_list[0]
        with h5py.File(current_directory, 'r') as hdf:
            imgs = hdf.get('rgb') # ALL THE 200 IMAGES IN THE H5 FILE
            imgs = np.array(imgs[:,:,:], dtype = np.uint8)
            image_shape = imgs.shape
            targets = hdf.get('targets') # ALL THE 200 TARGETS IN THE H5 FILE
            targets = np.array(targets)
        
        print("INPUT SHAPE IS:", image_shape)
        if len(image_shape) == 4:
            #single frames
            print("Single frames data")
            batch_size = image_shape[0]
            image_dimension = image_shape[1:]
            scenario_length = 1
        elif len(image_shape) == 5:
            #stacked frames
            print("stacked frames data")
            batch_size = image_shape[0]
            scenario_length = image_shape[1]
            image_dimension = image_shape[2:]
        
        return files_list, batch_size, scenario_length, image_dimension    
    
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
        print(image)
        layer = self.conv_block(image,5 , 2, 32, padding ='valid', drop_out = 0.2, name ='CONV1')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 32, padding ='valid', drop_out = 0.2, name ='CONV2')
        print(layer)
        layer = self.conv_block(layer,3 , 2, 64, padding ='valid', drop_out = 0.2, name ='CONV3')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 64, padding ='valid', drop_out = 0.2, name ='CONV4')
        print(layer)
        layer = self.conv_block(layer,3 , 2, 128, padding ='valid', drop_out = 0.2, name ='CONV5')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 128, padding ='valid', drop_out = 0.2, name ='CONV6')
        print(layer)
        layer = self.conv_block(layer,3 , 2, 256, padding ='valid', drop_out = 0.2, name ='CONV7')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 256, padding ='valid', drop_out = 0.2, name ='CONV8')
        print(layer)
        layer = self.flatten(layer)
        print(layer)
        layer = self.fc(layer, 512, drop_out = 0.5, name ='CONV_FC1')
        print(layer)
        layer = self.fc(layer, 512, drop_out = 0.5, name ='CONV_FC2')
        
        # Speed sensory input
        speed=(1,) # input layer'
        speed_input = Input(speed,name='speed_input')
        
        layer_speed =  self.fc(speed_input, 128, drop_out = 0.5, name ='SPEED_FC1')
        layer_speed =  self.fc(layer_speed, 128, drop_out = 0.5, name ='SPEED_FC2')
        
        middle_layer = self.concat(layer,layer_speed, name ='CONCAT_FC1')
        print(middle_layer)
        
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
                branches[output_branches_names[i]]=branch_output
                
            else:
                #only used images feature vector for predicting speed
                #TODO try both speed and images feature vector
                branch_output = self.fc(layer, 256, drop_out = 0.5,name =branches_names[i]+'_FC1')
                branch_output = self.fc(branch_output, 256, drop_out = 0.5,name =branches_names[i]+'_FC2')
                branch_output = self.fc(branch_output, 1, drop_out = 0, name =output_branches_names[-1])
                branches[output_branches_names[-1]] =  branch_output 
                
        self.model = Model(inputs = [image, speed_input],outputs = [branches['left_branch'],
                                                               branches['right_branch'],
                                                               branches['follow_branch'],
                                                               branches['str_branch'],
                                                               branches['speed_branch_output']])

        print(self.model.summary()) 
        
    def get_unbranched_network(self,input_shape):
        image = Input(input_shape)
        print(image)
        layer = self.conv_block(image,5 , 2, 32, padding ='valid', drop_out = 0.2, name ='conv1')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 32, padding ='valid', drop_out = 0.2, name ='conv2')
        print(layer)
        layer = self.conv_block(layer,3 , 2, 64, padding ='valid', drop_out = 0.2, name ='conv3')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 64, padding ='valid', drop_out = 0.2, name ='conv4')
        print(layer)
        layer = self.conv_block(layer,3 , 2, 128, padding ='valid', drop_out = 0.2, name ='conv5')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 128, padding ='valid', drop_out = 0.2, name ='conv6')
        print(layer)
        layer = self.conv_block(layer,3 , 2, 256, padding ='valid', drop_out = 0.2, name ='conv7')
        print(layer)
        layer = self.conv_block(layer,3 , 1, 256, padding ='valid', drop_out = 0.2, name ='conv8')
        print(layer)
        layer = self.flatten(layer, name ='flatten')
        print(layer)
        layer = self.fc(layer, 512, drop_out = 0.5, name ='conv_fc1')
        print(layer)
        layer = self.fc(layer, 512, drop_out = 0.5, name ='conv_fc2')
        
        speed=(1,) # input layer'
        speed_input = Input(speed)
        layer_speed =  self.fc(layer, 128, drop_out = 0.5, name ='speed_fc1')
        layer_speed =  self.fc(layer_speed, 128, drop_out = 0.5, name ='speed_fc2')
        
        middle_layer = self.concat(layer,layer_speed, name ='concat')
        print(middle_layer)
        
        steer = self.fc(middle_layer, 256, drop_out = 0.5, name ='steer_fc_1')
        steer = self.fc(steer, 256, drop_out = 0.5, name ='steer_fc_2')
        steer = self.fc(steer, 1, activation = 'relu', drop_out = 0, name ='steer_output')
        
        throttle = self.fc(middle_layer, 256, drop_out = 0.5, name ='throttle_fc_1')
        throttle = self.fc(throttle, 256, drop_out = 0.5, name ='throttle_fc_2')
        throttle = self.fc(throttle, 1, activation = 'relu', drop_out = 0, name ='throttle_output')
        
        brake =  self.fc(middle_layer, 256, drop_out = 0.5, name ='brake_fc_1')
        brake =  self.fc(brake, 256, drop_out = 0.5, name ='brake_fc_2')
        brake =  self.fc(brake, 1, activation = 'relu', drop_out =0, name='brake_output')
        self.model = Model(inputs = [image, speed_input],outputs = [steer, throttle , brake ])
        print(self.model.summary())
        
    def get_model(self):

        if (self.scenario_length == 1):    
            self.input_shape = self.image_dimension
        else:
            self.input_shape =  (self.scenario_length,) + self.image_dimension # Something like (32,5,200,200,3)
            
        print("Input shape to the network ", self.input_shape)   
        self.get_branched_network(self.input_shape)
        #self.get_unbranched_network(self.input_shape)
            

    
    def masked_loss_function(self, y_true, y_pred):
        mask_value=-2
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())      
        return keras.losses.mean_absolute_error(y_true * mask, y_pred * mask)
    
    def compile_model(self):
        opt = Adam(lr=0.0002, beta_1=0.7, beta_2=0.85, decay=1e-6)
        self.model.compile(optimizer = opt, loss = {'left_branch_steering': self.masked_loss_function,
                                                                    'left_branch_gas': self.masked_loss_function,
                                                                    'left_branch_brake': self.masked_loss_function,                                                                                      
                                                                    'right_branch_steering': self.masked_loss_function,
                                                                    'right_branch_gas': self.masked_loss_function,
                                                                    'right_branch_brake': self.masked_loss_function,                                                                                                              
                                                                    'follow_branch_steering': self.masked_loss_function,
                                                                    'follow_branch_gas': self.masked_loss_function,
                                                                    'follow_branch_brake': self.masked_loss_function,
                                                                    'str_branch_steering': self.masked_loss_function,
                                                                    'str_branch_gas': self.masked_loss_function,
                                                                    'str_branch_brake': self.masked_loss_function,
                                                                    'speed_branch_output': self.masked_loss_function},
                                                                        loss_weights = {'left_branch_steering': 0.4275,
                                                                    'left_branch_gas': 0.4275,
                                                                    'left_branch_brake': 0.0475,
                                                                    'right_branch_steering': 0.4275,                                                                                                                
                                                                    'right_branch_gas': 0.4275,
                                                                    'right_branch_brake': 0.0475,                                                                                                           
                                                                    'follow_branch_steering': 0.4275,                                                                                                               
                                                                    'follow_branch_gas': 0.4275,
                                                                        'follow_branch_brake': 0.0475,                                                                                                           
                                                                    'str_branch_steering': 0.4275,                                                                                                              
                                                                    'str_branch_gas': 0.4275,
                                                                    'str_branch_brake': 0.0475,                                                                                                                     
                                                                    'speed_branch_output': 0.05})
        print("Done compiling model!")
        return
  
    
    
    
    
    def fetch(self, number_minibatches ,number_of_files):
        '''
        branche_name is one of {'Left', 'Right', ...}
        '''
        
        #size_minibatches_per_epoch=self.batch_size*step   #total size of batches in each epoch
        size_minibatches_per_epoch = self.batch_size*number_minibatches
        if self.scenario_length == 1:
            target_size = (4*size_minibatches_per_epoch,) +  self.image_dimension
        else:
            
            target_size = (4*size_minibatches_per_epoch,) + (self.scenario_length,) +   self.image_dimension # Something like (32,5,200,200,3)
        #print(target_size)
        images = np.zeros(target_size)
        # print("Images shape is:", images.shape)
        velocity = np.zeros((4*size_minibatches_per_epoch,))
        mask_steer = np.ones((self.batch_size,))*(-2)
        mask_throttle = np.ones((self.batch_size,))*(-2)
        mask_brake = np.ones((self.batch_size,))*(-2)
        output_labels = np.zeros((4*size_minibatches_per_epoch, 13)) # 13 = 4*3 +1 = 13 output
        branches = ['left', 'right', 'follow', 'straight']

        while True:
            for m, branche_name in enumerate(branches):
                for k in range(0, number_minibatches):
                        filename= self.train_data_directory + branche_name +'/data_' + str(self.current_folder_generator+k)+ '.h5'
                        #print(filename)
                        with h5py.File(filename, 'r') as hdf:
                                imgs = hdf.get('rgb')
                                imgs = np.array(imgs[:,:,:], dtype = np.uint8)
                                #print(imgs.shape)
                                targets = hdf.get('targets')
                                targets = np.array(targets)
                                # print(imgs.shape)
                                # print(targets.shape)
                                
                        #Preparing files and putting masks 
                        images[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size]=imgs/255.0
                        velocity[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size] = targets[:,10]
                        # print("Shaping is:", output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,:].shape)
                        output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,12] = targets[:,10]
                        # velocity[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,10]
                        

                        if branche_name== 'left':
                            # print("In Left...")
                            # print(output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,3].shape)
                            # print(mask_steer.shape)
                            # print(mask_throttle.shape)
                            # print(mask_brake.shape)
                            #self.get_left_output_labels(m,size_minibatches_per_epoch,k)
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = targets[:,2]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
                        elif branche_name== 'right':
                            #self.get_right_output_labels()
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = targets[:,2]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
                        elif branche_name== 'follow':
                            #self.get_follow_left_output_labels()
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = targets[:,2]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
                        elif branche_name== 'straight':
                            #self.get_straight_output_labels()
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = targets[:,2]
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]

            self.generator_counter += number_minibatches
            self.current_folder_generator = self.current_folder_generator + number_minibatches
            #print("train counter ",self.current_folder_generator,end=" ")
            if self.current_folder_generator == number_of_files:
                self.current_folder_generator = 0

            yield [images, velocity], [output_labels[:,0], output_labels[:,1], output_labels[:,2], output_labels[:,3], output_labels[:,4], output_labels[:,5], output_labels[:,6], output_labels[:,7], output_labels[:,8], output_labels[:,9], output_labels[:,10], output_labels[:,11], output_labels[:,12]]
    
       
    def fetch_validation(self, number_minibatches ,number_of_files):
        '''
        branche_name is one of {'Left', 'Right', ...}
        '''
        
        #size_minibatches_per_epoch=self.batch_size*step   #total size of batches in each epoch
        size_minibatches_per_epoch = self.batch_size*number_minibatches
        #print(size_minibatches_per_epoch)
        if self.scenario_length == 1:
            target_size = (4*size_minibatches_per_epoch,) +  self.image_dimension
        else:
            
            target_size = (4*size_minibatches_per_epoch,) + (self.scenario_length,) +   self.image_dimension # Something like (32,5,200,200,3)
        #print(target_size)
        images = np.zeros(target_size)
        # print("Images shape is:", images.shape)
        velocity = np.zeros((4*size_minibatches_per_epoch,))
        mask_steer = np.ones((self.batch_size,))*(-2)
        mask_throttle = np.ones((self.batch_size,))*(-2)
        mask_brake = np.ones((self.batch_size,))*(-2)
        output_labels = np.zeros((4*size_minibatches_per_epoch, 13)) # 13 = 4*3 +1 = 13 output
        branches = ['left', 'right', 'follow', 'straight']

        while True:
            for m, branche_name in enumerate(branches):
                for k in range(0, number_minibatches):
                        filename= self.validation_data_directory + '/' + branche_name +'/data_' + str(self.validation_current_folder_generator+k)+ '.h5'
                        with h5py.File(filename, 'r') as hdf:
                                imgs = hdf.get('rgb')
                                imgs = np.array(imgs[:,:,:], dtype = np.uint8)
                                targets = hdf.get('targets')
                                targets = np.array(targets)
                                #print(imgs.shape)
                                #print(targets.shape)
                                
                        #Preparing files and putting masks 
                        images[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size]=imgs/255.0
                        velocity[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size] = targets[:,10]
                        # print("Shaping is:", output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,:].shape)
                        output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,12] = targets[:,10]
                        # velocity[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,10]
                        

                        if branche_name== 'left':
                            # print("In Left...")
                            # print(output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,3].shape)
                            # print(mask_steer.shape)
                            # print(mask_throttle.shape)
                            # print(mask_brake.shape)
                            #self.get_left_output_labels(m,size_minibatches_per_epoch,k)
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = targets[:,2]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
                        elif branche_name== 'right':
                            #self.get_right_output_labels()
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = targets[:,2]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
                        elif branche_name== 'follow':
                            #self.get_follow_left_output_labels()
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = targets[:,2]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
                        elif branche_name== 'straight':
                            #self.get_straight_output_labels()
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = targets[:,0]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = targets[:,1]
                            output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = targets[:,2]
                            # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
                            # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
                            # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]

            self.validation_generator_counter += number_minibatches
            
            self.validation_current_folder_generator = self.validation_current_folder_generator + number_minibatches
            #print(self.validation_current_folder_generator,end=" ")
            if self.current_folder_generator == number_of_files:
                self.validation_current_folder_generator = 0

            yield [images, velocity], [output_labels[:,0], output_labels[:,1], output_labels[:,2], output_labels[:,3], output_labels[:,4], output_labels[:,5], output_labels[:,6], output_labels[:,7], output_labels[:,8], output_labels[:,9], output_labels[:,10], output_labels[:,11], output_labels[:,12]]
    
        

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
