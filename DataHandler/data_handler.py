import numpy as np
import h5py
from progressbar import *
import os, os.path
import matplotlib.pyplot as plt
import random
# import imgaug as ia
# from imgaug import augmenters as iaa
import re
import cv2
from PIL import Image
import IPython

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

    
    
    # def fetch_minibatch(self, number_minibatches ,number_of_files):
    #     '''
    #     branche_name is one of {'Left', 'Right', ...}
    #     '''
        
    #     size_minibatches_per_epoch = self.batch_size*number_minibatches
    #     if self.scenario_length == 1:
    #         target_size = (4*size_minibatches_per_epoch,) +  self.image_dimension
    #     else:
            
    #         target_size = (4*size_minibatches_per_epoch,) + (self.scenario_length,) +   self.image_dimension 
    #     #print(target_size)
    #     images = np.zeros(target_size)
    #     # print("Images shape is:", images.shape)
    #     velocity = np.zeros((4*size_minibatches_per_epoch,))
    #     mask_steer = np.ones((self.batch_size,))*(-2)
    #     mask_throttle = np.ones((self.batch_size,))*(-2)
    #     mask_brake = np.ones((self.batch_size,))*(-2)
    #     output_labels = np.zeros((4*size_minibatches_per_epoch, 13)) # 13 = 4*3 +1 = 13 output
    #     branches = ['left', 'right', 'follow', 'straight']

    #     while True:
    #         for m, branche_name in enumerate(branches):
    #             for k in range(0, number_minibatches):


    #                     filename= self.train_data_directory + branche_name +'/data_' + str(self.current_folder_generator+k)+ '.h5'
    #                     #print(filename)
    #                     with h5py.File(filename, 'r') as hdf:
    #                             imgs = hdf.get('rgb')
    #                             imgs = np.array(imgs[:,:,:], dtype = np.uint8)
    #                             #print(imgs.shape)
    #                             targets = hdf.get('targets')
    #                             targets = np.array(targets)
    #                             # print(imgs.shape)
    #                             # print(targets.shape)
                                
    #                     #Preparing files and putting masks 
    #                     images[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size]=imgs/255.0
    #                     velocity[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size] = targets[:,10]
    #                     # print("Shaping is:", output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,:].shape)
    #                     output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,12] = targets[:,10]
    #                     # velocity[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,10]
                        

    #                     if branche_name== 'left':
    #                         # print("In Left...")
    #                         # print(output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,3].shape)
    #                         # print(mask_steer.shape)
    #                         # print(mask_throttle.shape)
    #                         # print(mask_brake.shape)
    #                         #self.get_left_output_labels(m,size_minibatches_per_epoch,k)
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = targets[:,0]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = targets[:,1]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = targets[:,2]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
    #                         # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
    #                         # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
    #                         # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]

    #                     elif branche_name== 'right':
    #                         #self.get_right_output_labels()
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = targets[:,0]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = targets[:,1]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = targets[:,2]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
    #                         # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
    #                         # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
    #                         # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
    #                     elif branche_name== 'follow':
    #                         #self.get_follow_left_output_labels()
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = targets[:,0]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = targets[:,1]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = targets[:,2]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
    #                         # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
    #                         # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
    #                         # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
    #                     elif branche_name== 'straight':
    #                         #self.get_straight_output_labels()
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = targets[:,0]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = targets[:,1]
    #                         output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = targets[:,2]
    #                         # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
    #                         # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
    #                         # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]

    #         self.generator_counter += number_minibatches
    #         self.current_folder_generator = self.current_folder_generator + number_minibatches
    #         #print("train counter ",self.current_folder_generator,end=" ")
    #         if self.current_folder_generator == number_of_files:
    #             self.current_folder_generator = 0

    #         yield [images, velocity], [output_labels[:,0], output_labels[:,1], output_labels[:,2], output_labels[:,3], output_labels[:,4], output_labels[:,5], output_labels[:,6], output_labels[:,7], output_labels[:,8], output_labels[:,9], output_labels[:,10], output_labels[:,11], output_labels[:,12]]


    def fetch_minibatch(self,branch_name,number_of_files):
        '''
        branche_name is one of {'Left', 'Right', ...}
        '''
        
        size_minibatches_per_epoch = self.batch_size
        target_size = (size_minibatches_per_epoch,) +  self.image_dimension[1:]
        images = np.zeros(target_size)
        next_images = np.zeros(target_size)
        action = np.zeros(self.batch_size)
        next_action = np.zeros(self.batch_size)
        #print('images shapes ',images.shape)
        velocity = np.zeros((size_minibatches_per_epoch,))
        next_velocity = np.zeros((size_minibatches_per_epoch,))
        mask_steer = np.ones((self.batch_size,))*(-2)
        mask_throttle = np.ones((self.batch_size,))*(-2)
        mask_brake = np.ones((self.batch_size,))*(-2)
        # output_labels = np.zeros((size_minibatches_per_epoch, 13)) # 13 = 4*3 +1 = 13 output
        # next_output_labels = np.zeros((size_minibatches_per_epoch, 13)) # 13 = 4*3 +1 = 13 output

        for i in range(self.batch_size):
            filename= self.train_data_directory + '/'+branch_name +'/data_' + str(self.current_folder_generator)+ '.h5'
            #print(filename)
            with h5py.File(filename, 'r') as hdf:
                    imgs = hdf.get('rgb')
                    imgs = np.array(imgs[:,:,:], dtype = np.uint8)
                    targets = hdf.get('targets')
                    targets = np.array(targets)
                    # if i == 0:
                    #     print(imgs.shape)
                    #     print(targets.shape)

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
        
        #print(images.shape)
        #print(action)

        return images , next_images , action , next_action ,velocity , next_velocity
        # if branche_name == 'left':
        # 	action[4:]
                    
        #             #Preparing files and putting masks 
        #             images[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size]=imgs/255.0
        #             velocity[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size] = targets[:,10]
        #             # print("Shaping is:", output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,:].shape)
        #             output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,12] = targets[:,10]
        #             # velocity[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,10]
                    

        #             if branche_name== 'left':
        #                 # print("In Left...")
        #                 # print(output_labels[m*size_minibatches_per_epoch+i*self.batch_size:m*size_minibatches_per_epoch+i*self.batch_size+self.batch_size,3].shape)
        #                 # print(mask_steer.shape)
        #                 # print(mask_throttle.shape)
        #                 # print(mask_brake.shape)
        #                 #self.get_left_output_labels(m,size_minibatches_per_epoch,k)
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = targets[:,0]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = targets[:,1]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = targets[:,2]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
        #                 # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
        #                 # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
        #                 # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]

        #             elif branche_name== 'right':
        #                 #self.get_right_output_labels()
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = targets[:,0]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = targets[:,1]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = targets[:,2]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
        #                 # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
        #                 # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
        #                 # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
        #             elif branche_name== 'follow':
        #                 #self.get_follow_left_output_labels()
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = targets[:,0]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = targets[:,1]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = targets[:,2]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = mask_brake
        #                 # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
        #                 # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
        #                 # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]
        #             elif branche_name== 'straight':
        #                 #self.get_straight_output_labels()
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,0] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,1] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,2] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,3] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,4] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,5] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,6] = mask_steer
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,7] = mask_throttle
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,8] = mask_brake
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,9] = targets[:,0]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,10] = targets[:,1]
        #                 output_labels[m*size_minibatches_per_epoch+k*self.batch_size:m*size_minibatches_per_epoch+k*self.batch_size+self.batch_size,11] = targets[:,2]
        #                 # steering_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,0]
        #                 # throttle_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,1]
        #                 # brake_vec[i*self.batch_size:i*self.batch_size+self.batch_size]=targets[:,2]

        # self.generator_counter += number_minibatches
        # self.current_folder_generator = self.current_folder_generator + number_minibatches
        # #print("train counter ",self.current_folder_generator,end=" ")
        # if self.current_folder_generator == number_of_files:
        #     self.current_folder_generator = 0

        # return [images, velocity], [output_labels[:,0], output_labels[:,1], output_labels[:,2], output_labels[:,3], output_labels[:,4], output_labels[:,5], output_labels[:,6], output_labels[:,7], output_labels[:,8], output_labels[:,9], output_labels[:,10], output_labels[:,11], output_labels[:,12]]



    def imshow(self,img):
        _,ret = cv2.imencode('.jpg',img)
        i = IPython.display.Image(data=ret)
        IPython.display.display(i)

    def fetch_single_image(self, image_number):
        '''
            This Fucntion fetchs data for imitation learning
            Input: 
            Output:
        '''

        filename= self.train_data_directory + '/follow' +'/data_' + str(image_number)+ '.h5'
        with h5py.File(filename, 'r') as hdf:
                imgs = hdf.get('rgb')
                imgs = np.array(imgs[:,:,:], dtype = np.uint8)
                #print(imgs.shape)
                targets = hdf.get('targets')
                targets = np.array(targets)
                #print(imgs.shape)
                #print(imgs)
                #print(targets)
                steer = targets[:,0][0]
                throttle = targets[:,1][0]
                brake = targets[:,2][0]
                print(self.map_outputs(throttle = throttle, steer = steer , brake = brake , one_output_for_throttle_brake = True))
                # self.imshow(imgs[0])

        pass

    def fetch_validation():
        '''
        '''
        pass

    def map_outputs(self, throttle, steer, brake = 0, disceretization_bins = 5, one_output_for_throttle_brake = True):
        '''
            This Fucntion Maps continous outputs from Demonstration data to disceret values
            Input : 
            Output : action number 
        '''
        epsilon = 0.1
        steer_range = [-1, 1]
        steer_outputs = np.linspace(steer_range[0], steer_range[1], num=disceretization_bins)
        throttle_brake_range = [-1 , 1]

        if one_output_for_throttle_brake:
        	#print("creating one array for both throttle and brake")
        	throttle_brake_outputs = np.linspace(throttle_brake_range[0], throttle_brake_range[1], num=disceretization_bins)

        	if brake > 0.01 and throttle < 0.01:
        		throttle = -brake

        num_actions = len(steer_outputs) * len(throttle_brake_outputs) 
        
        steer_action = int(steer_outputs[int(round(((steer - steer_range[0])/(steer_range[1]-steer_range[0]))*(disceretization_bins-1) + epsilon ))])
        throttle_brake_action = int(throttle_brake_outputs[int(round(((throttle - throttle_brake_range[0])/(throttle_brake_range[1]-throttle_brake_range[0]))*(disceretization_bins-1) + epsilon ))])
        
        action_number = int(((np.where(steer_outputs == steer_action)[0] ) * 5 ) + np.where(throttle_brake_outputs == throttle_brake_action)[0] )

        return action_number


# data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
# train_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
# validation_data_directory = '/home/mohamed/Desktop/Codes/rlfd_data/imitation_data'
# handler = handler(train_data_directory = train_data_directory, validation_data_directory = validation_data_directory )

# print(handler.map_outputs(throttle = 1, steer = 0.199))
# handler.fetch_single_image(1)