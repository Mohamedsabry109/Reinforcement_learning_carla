import numpy as np
import h5py
from progressbar import *
import os, os.path
import matplotlib.pyplot as plt
import random
from Utils import config
import re
import cv2
from PIL import Image
import IPython

# import imgaug as ia
# from imgaug import augmenters as iaa



class handler(object):

    def __init__(self, train_data_directory, validation_data_directory, scenario_length = 1, batch_size = 32):
        self.data_directory = train_data_directory
        self.train_data_directory = train_data_directory
        self.validation_data_directory = validation_data_directory
        self.scenario_length = scenario_length
        self.batch_size = batch_size
        self.generator_counter = 0
        self.train_generator_counter = 0
        self.validation_generator_counter = 0
        self.current_folder_generator = 0
        self.validation_current_folder_generator = 0
        self.check_data()

    def check_data(self):
        '''
           This function checks for input data directory and size of images
        '''
        
        assert os.path.isdir(self.data_directory), "Given dataDirectory isn't a directory path!"
        files_list = sorted(os.listdir(self.data_directory + '/follow/'))
        assert files_list != [], "Given directory has no files!"
        # Reads the first .h5 file
        #print(files_list[0])
        with h5py.File(self.data_directory + '/follow/'+ files_list[0], 'r') as hdf:
            images = hdf.get('rgb') 
            images = np.array(images[:,:,:], dtype = np.uint8)
            assert images.size != 0, "H5 file has no images in it!"
            image_shape = images.shape
            targets = hdf.get('targets') # ALL THE 200 TARGETS IN THE H5 FILE
            targets = np.array(targets)
            assert targets.size != 0, "H5 file has no images in it!"
        images_per_H5_file = image_shape[0]
        self.image_dimension = image_shape[1:]
        #print("All Images Must Have The Same Length of: {}".format(self.image_dimension))
        #print("WARNING: All Files Have {} images".format(images_per_H5_file))

    def fetch_minibatch(self,branch_name,number_of_files):
        """
        Fetching a Mini-batch from one of the branches.

        Args:
            branch_name: str 
            number_of_files: int
        Returns:
            states, next_states, actions, next_actions, velocity, next_veloctiym reward
        """
        print("\n fetching mini batch from ",branch_name)    
        size_minibatches_per_epoch = self.batch_size
        target_size = (size_minibatches_per_epoch,) +  self.image_dimension[1:]
        images = np.zeros(target_size)
        next_images = np.zeros(target_size)
        action = np.zeros(self.batch_size)
        next_action = np.zeros(self.batch_size)
        velocity = np.zeros((size_minibatches_per_epoch,))
        next_velocity = np.zeros((size_minibatches_per_epoch,))
        mask_steer = np.ones((self.batch_size,))*(-2)
        mask_throttle = np.ones((self.batch_size,))*(-2)
        mask_brake = np.ones((self.batch_size,))*(-2)
        # output_labels = np.zeros((size_minibatches_per_epoch, 13)) # 13 = 4*3 +1 = 13 output
        # next_output_labels = np.zeros((size_minibatches_per_epoch, 13)) # 13 = 4*3 +1 = 13 output

        for i in range(self.batch_size):
            filename= self.train_data_directory + '/'+branch_name +'/data_' + str(self.current_folder_generator)+ '.h5'
            self.current_folder_generator += 1
            #print(filename)
            with h5py.File(filename, 'r') as hdf:
                    imgs = hdf.get('rgb')
                    imgs = np.array(imgs[:,:,:], dtype = np.uint8)
                    targets = hdf.get('targets')
                    targets = np.array(targets)

            images[i] = imgs[0][0]
            next_images[i] = imgs[0][1]
            velocity[i] = targets[0,10]
            next_velocity[i] = targets[0,10]

            steer = targets[:,0][0]
            throttle = targets[:,1][0]
            brake = targets[:,2][0]  

            next_steer = targets[:,0][0]
            next_throttle = targets[:,1][0]
            next_brake = targets[:,2][0]

            action[i] = self.map_outputs(throttle = throttle, steer = steer , brake = brake , one_output_for_throttle_brake = True)
            next_action[i] = self.map_outputs(throttle = next_throttle, steer = next_steer , brake = next_brake , one_output_for_throttle_brake = True)
        
        return images , next_images , action , next_action ,velocity , next_velocity


    def imshow(self,img):
        _,ret = cv2.imencode('.jpg',img)
        i = IPython.display.Image(data=ret)
        IPython.display.display(i)


    @staticmethod
    def fetch_single_image(directory, branch_name, observation_name):
        '''
            This Fucntion fetchs data for imitation learning
            Input: 
            Output:
        '''

        filename= directory + '/' + branch_name + '/' + observation_name 
        with h5py.File(filename, 'r') as hdf:
                imgs = hdf.get('rgb')
                imgs = np.array(imgs[:,:,:], dtype = np.uint8)
                #print(imgs.shape)
                targets = hdf.get('targets')
                targets = np.array(targets)
                steer = targets[:,0][0]
                throttle = targets[:,1][0]
                brake = targets[:,2][0]
                velocity = targets[:,10][0]
                reward = targets[:,28][0]
                done = targets[:,29][0]
                #print(self.map_outputs(throttle = throttle, steer = steer , brake = brake , one_output_for_throttle_brake = True))
                # self.imshow(imgs[0])
                action_number = handler.map_outputs(throttle = throttle, steer = steer , brake = brake , one_output_for_throttle_brake = True)

        return [imgs , velocity] , action_number , reward , done


    @staticmethod
    def fetch_single_interaction_image(directory, branch_name, observation_name):
        '''
            This Fucntion fetchs data for imitation learning
            Input: 
            Output:
        '''

        filename= directory + '/' + branch_name + '/' + observation_name 
        with h5py.File(filename, 'r') as hdf:
                imgs = hdf.get('rgb')
                imgs = np.array(imgs[:,:,:], dtype = np.uint8)
                #print(imgs.shape)
                targets = hdf.get('targets')
                targets = np.array(targets)
                steer = targets[:,0][0]
                throttle = targets[:,1][0]
                brake = targets[:,2][0]
                velocity = targets[:,10][0]
                reward = 0
                #print(self.map_outputs(throttle = throttle, steer = steer , brake = brake , one_output_for_throttle_brake = True))
                # self.imshow(imgs[0])
                action_number = handler.map_outputs(throttle = throttle, steer = steer , brake = brake , one_output_for_throttle_brake = True)


        return [imgs , velocity] , action_number , reward



    @staticmethod
    def closest(lst, K): 
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 
      
    @staticmethod
    def map_actions(action):
        action_dict = config.ACTION_DICT
        throttle_brake_steer = list(action_dict.keys())[list(action_dict.values()).index(action)]
        steer =  float(throttle_brake_steer[0].split("_")[0])
        throttle_brake = float(throttle_brake_steer[0].split("_")[1])

        if throttle_brake > 0:
            return throttle_brake , 0 , steer
        else:
            return 0, throttle_brake, steer


    @staticmethod
    def map_outputs( throttle, steer, brake = 0, disceretization_bins = config.QUANTIZATION_BINS, one_output_for_throttle_brake = True):
        '''
            This Fucntion Maps continous outputs from Demonstration data to disceret values
            Input : 
            Output : action number 
        '''
        # return action_number
        diff = throttle - brake
        steer_action = round(handler.closest(config.STEER_VALUES,steer),1)
        throttle_brake_action = round(handler.closest(config.STEER_VALUES,diff),1)

        if abs(throttle_brake_action) == 0:
            throttle_brake_action = 0.0
        if abs(steer_action) == 0:
            steer_action = 0.0


        # if steer_action > 0:
        #     steer_action = 1.0
        # elif steer_action < 0:
        #     steer_action = -1.0
        # else:
        #     0.0

        # if throttle_brake_action > 0:
        #     throttle_brake_action = 1.0
        # elif throttle_brake_action < 0:
        #     throttle_brake_action = -1.0
        # else:
        #     throttle_brake_action = 0.0

        action_num = config.ACTION_DICT[str(10*steer_action) + '_' + str(10*throttle_brake_action)]

        return action_num
