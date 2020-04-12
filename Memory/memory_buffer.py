import random
import numpy as np
from DataHandler import data_handler
from collections import deque
from .sum_tree import SumTree
import os


"""
    buffers tasks: 
    OfflineBuffers:
        support fetching from disk :
            keep track of fetching
            files_counter keeps track of number of files in each folder by calling initialze_buffer
            buffer_pointer is pointer pointing to the next batch in case of fetching directly from offline buffer 
             and is used for changing the priorities after fetching offline                                   

        change priorities -> (upon fetching)
        save data to disk space -> keep track of files traker

    OnlineBufers: 
        reloading from offline buffer
        change priorities
        update priorities 
        fetching


"""


class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per = False, name = None, directory = None):
        """ Initialization
        """
        self.directory = directory
        self.name = name

        if(with_per):
            # Prioritized Experience Replay
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self.count = 0
        self.with_per = with_per
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state, error=None, path = None):
        """ Save an experience to memory, optionally with its TD-Error
        """

        experience = (state, action, reward, done, new_state)
        if(self.with_per):
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            self.count += 1
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
            a batch include all idxs
        """
        batch = []

        # Sample using prorities
        if(self.with_per):
            T = self.buffer.total() // batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                idx, error, data = self.buffer.get(s)
                batch.append((*data, idx))
            idx = np.array([i[5] for i in batch])
            #TD errors are only updated for transitions that are replayed
            
        # Sample randomly from Buffer
        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])

        return s_batch, a_batch, r_batch, d_batch, new_s_batch, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(buffer_size)
        else: self.buffer = deque()
        self.count = 0


class OnlineMemoryBuffer(MemoryBuffer):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per = False):
        super().__init__(buffer_size, with_per)
        pass
    
    def reload(self,buffer):

        """
            reload the onlinebuffer from offline buffer data according to some policy
            follow the same procedure of sampling high priority observations
            we can loop for a given offline buffer and sample from it
        Args:
            Instant of OfflineBuffer: 

        Returns:

        """
        '''
        sample from offline buffer -> name of sampled file
        store the sample in the online buffer (load the data with given name and put it the online buffer)
        buffer.get(name) -> data -> (state, next state, action, next action, reward)
        // handling fetching state and next state
        self.buffer.memorize(data)

        '''    
        # self.clear()
        # #buffer.sample(self.buffer_size)
        # # Sample using prorities
        # if(self.with_per):
        #     T = buffer.total() // batch_size
        #     for i in range(batch_size):
        #         a, b = T * i, T * (i + 1)
        #         s = random.uniform(a, b)
        #         idx, error, data = self.buffer.get(s)
                #fetch this file

        #         batch.append((*data, idx))
        #     idx = np.array([i[5] for i in batch])
        #     #TD errors are only updated for transitions that are replayed
            
        # # Sample randomly from Buffer
        # elif self.count < batch_size:
        #     idx = None
        #     batch = random.sample(self.buffer, self.count)
        # else:
        #     idx = None
        #     batch = random.sample(self.buffer, batch_size)
        pass

    def change_priorities():

        """
            change priorities of online buffer items
            it should iteratively calls update
        Args:

        Returns:

        """      
        pass


class OfflineMemoryBuffer(MemoryBuffer):
    """ 
        offline buffer inherits from MemoryBuffer but it only store names and priorites
    """
    def __init__(self, buffer_size, name, train_data_directory, validation_data_directory, with_per = False):
        super().__init__(buffer_size, with_per, name = name ,directory = train_data_directory)
        #buffer pointer is used to fetch minibatches in imitation learning
        self.buffer_pointer = 0
        print("Checking the data for initiallizing an offline buffer of the ", self.name, " branch")
        self.data_handler = data_handler.handler(train_data_directory = train_data_directory, validation_data_directory = validation_data_directory)
        print("tree is ",self.buffer.tree)

        self.initiallize_buffer()
        # print("buffer ",self.buffer.data[0])
        # print("buffer size ", self.buffer.total())
        # print("tree is ",self.buffer.tree)
        #self.data_handler.fetch_minibatch(name,100)

        pass

    def initiallize_buffer(self):
        """
        Initiallize offline buffer with all data of imitation learning to start track their priorities
            and get information about names and initiallize a tracker for updating

        """
        assert os.path.isdir(self.directory)
        files_list = sorted(os.listdir(self.directory + '/' + self.name + '/'))
        print("length of files ",len(files_list))
        assert files_list != []
        self.files_counter = 0
        for file_name in files_list:         
            self.memorize(name = file_name, error = 0)
            self.files_counter += 1

        #self.files_tracker = file_name

    def change_priorities(self,idxs,errors):
        """
        change priorities of offline buffer items, online buffers needs to store the idx in offline buffer of
            each element in it
        Args:
            idxs: array of int 
            errors: array of int
        """
        for i,idx in enumerate(idxs):
            self.update(idx,errors[i])

    def memorize(self,name= None, error=None):
        """
<<<<<<< HEAD
        Memorize data in the buffer.

        Args:
            name: str 
            error: int
        """

=======
        # data = (name, error)
>>>>>>> 28f4ac46783eb4b550d90adc152f662790a55df0
        data = name
        if(self.with_per):
            #priority = self.priority(error[0])
            priority = self.priority(error)
            self.buffer.add(priority, data)
            self.count += 1
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(data)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(data)

    def fetch(self, batch_size):
        """
        Fetching a Mini-batch from one of the branches.

        Args:
            batch_size: int 

        Returns:
            states, next_states, actions, next_actions, reward
            states inlcude the images and velocity
        """
        #handling case of arriving at the end of the file
        if self.buffer_pointer <= self.files_counter - 32:
            idxs = np.linspace(self.buffer_pointer, self.buffer_pointer + batch_size - 1 , batch_size)
        else:
            idxs1 = np.linspace(self.buffer_pointer, self.files_counter , self.files_counter - self.buffer_pointer + 1) 
            idxs2 = np.linspace(0,(batch_size - (self.files_counter - self.buffer_pointer) - 2) ,(batch_size - (self.files_counter - self.buffer_pointer)-1))
            idxs = np.concatenate(idxs1, idxs2, axis = 0)
        
        self.buffer_pointer += batch_size
        return self.data_handler.fetch_minibatch(self.name,self.files_counter)


    def save_episode_data(batch):
        """
            This function take the observations of some episode and store them on the disk
            and then updating offline buffer with these new data and their priorites
            should call change priorities, keep track of names of the data
        Args:
            batch: {state:,action,errors} 
        """

        pass