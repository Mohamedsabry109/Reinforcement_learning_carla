import h5py
import os
import time


class hdfs_test():
	def __init__(self, file_path, saving_directory):
		self.file_path = file_path
		self.file_counter = 0
		self.saving_directory = saving_directory


	def expand_file(self):
		'''
		Expands file of 200 image into 200 seperate files
		'''
		with h5py.File(self.file_path, 'r') as hdf:
            imgs = hdf.get('rgb')
            imgs = np.array(imgs[:,:,:], dtype = np.uint8)
            targets = hdf.get('targets')
            targets = np.array(targets)

        for i in range(imgs.shape[0]):
        	filename = self.saving_directory + "/data_" + str(self.file_counter)
        	self.file_counter += 1
        	with h5py.File(filename, 'w') as hdf:
        		hf.create_dataset('rgb', data=imgs[i])
        		hf.create_dataset('targets', data=targets[i])

    	return

	def test(self):
		'''
		Test the timing between two methodologies
		'''
		first_case_start = time.time()
		with h5py.File(self.file_path, 'r') as hdf:
            imgs = hdf.get('rgb')
            imgs = np.array(imgs[:,:,:], dtype = np.uint8)
            targets = hdf.get('targets')
            targets = np.array(targets)
        first_case_end = time.time()

        files_list = listdir(self.saving_directory)
        second_case_start = time.time()
        for file in files_list:
        	file_name = self.saving_directory + "/" + file
        	with h5py.File(file_name, 'r') as hdf:
	            imgs = hdf.get('rgb')
	            imgs = np.array(imgs[:,:,:], dtype = np.uint8)
	            targets = hdf.get('targets')
	            targets = np.array(targets)
        second_case_end = time.time()

        print("Test results:\nReading one file with 200 images timing: {}\nReading 200 files each has one image time: {}".format(first_case_end-first_case_start, second_case_end-second_case_start))
        return

file_path = ""
saving_directory = ""
hdfs_test = hdfs_test(file_path, saving_directory)
hdfs_test.expand_file()
hdfs_test.test()
        		
