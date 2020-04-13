import h5py
import os
import time
import numpy as np

class data_expansion():
    def __init__(self, input_data_path, output_directory):
        self.input_data_path = input_data_path
        self.output_directory = output_directory
        self.output_directories_list = [ self.output_directory + "/left",
                                        self.output_directory + "/right",
                                        self.output_directory + "/follow",
                                        self.output_directory + "/straight"]
        
        self.input_directories_list = [ self.input_data_path + "/left",
                                        self.input_data_path + "/right",
                                        self.input_data_path + "/follow",
                                        self.input_data_path + "/straight"]

        self.folders_counters = [0,0,0,0] # Coubnters while saving
        self.create_subdirectories()

        

    def create_subdirectories(self):
        '''
        Creates subdirectories
        '''
        for subdir in self.output_directories_list:
            os.mkdir(subdir)


    def expand(self):
        '''
        Expands a 4 directories, e.g. Left
        '''
        for i,m in enumerate(self.input_directories_list):
            start = time.time()
            print("\nExpanding {} branch...".format(m))
            self.expand_folder(i,m)
            print("Done expanding {} branch, took {} seconds\n".format(m, time.time()- start))
        return

    def expand_folder(self, i, folder_path):
        '''
        Expands a self.input_directories_list instance folder into self.output_directories_list instance
        '''
        files_list = os.listdir(folder_path)
        for file in files_list:
            self.expand_file(i, file)
        return

    def expand_file(self, i, file):
        '''
        Expands file of 200 image into 200 seperate files
        '''
        with h5py.File(file, 'r') as hdf:
            imgs = hdf.get('rgb')
            imgs = np.array(imgs[:,:,:], dtype = np.uint8)
            targets = hdf.get('targets')
            targets = np.array(targets)

        for k in range(imgs.shape[0]):
            filename = self.output_directories_list[i] + "/data_" + str(self.folders_counters[i])
            self.folders_counters[i] += 1
            with h5py.File(filename, 'w') as hdf:
                hdf.create_dataset('rgb', data=imgs[k])
                hdf.create_dataset('targets', data=targets[k])

        return



input_data_path = "/home/mohamed/Desktop/SeqTrain"
output_directory = "/home/mohamed/Desktop/Expanded"
data_expansion = data_expansion(input_data_path, output_directory)
data_expansion.expand()