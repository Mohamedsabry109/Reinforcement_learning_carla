import random
import numpy as np
from DataHandler import data_handler
from collections import deque
from .sum_tree import SumTree

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
            I/P : offline buffer
            O/P : reloaded Online buffer 
        """
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
            I/P: batch of names and errors 
            O/P: changing priorities
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
        #self.data_handler.fetch_minibatch(name,100)

        pass

    def initiallize_buffer(self):
        """
            Initiallize offline buffer with all data of imitation learning to start track their priorities
            and get information about names and initiallize a tracker for updating
            I/P : read all files in a given folder and initiallize the buffer with names and priorities of zero
            O/P : no output
        """
        assert os.path.isdir(self.directory)
        files_list = sorted(os.listdir(self.directory + '/' + self.name + '/'))
        assert files_list != []
        #print(files_list[0])
        self.files_counter = 0
        for file_name in files_list:         
            self.memorize(name = file_name, error = 0)
            self.files_counter += 1

    def change_priorities(self):
        """
            change priorities of offline buffer items, online buffers needs to store the idx in offline buffer of
            each element in it
            we can then change priorities of idxs
            we can iterate on the online bufferidx and error and then update offline buffer, get 
            we can depend on just the buffer_pointer
        """ 

        pass

    def memorize(self,name= None, error=None, path = None):
        """ Save an experience to memory, optionally with its TD-Error
        """
        data = (name, error)
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

    def fetch(self, batch_size):
        """ Sample a batch, optionally with (PER)
            a batch include all idxs in the offline buffer
            change priorities should be called directly after one step learining
        """
        self.buffer_pointer += batch_size
        #TODO keep track of number of files
        return self.data_handler.fetch_minibatch(self.name,100)


    def save_episode_data(batch):
        """
            This function take the observations of some episode and store them on the disk
            and then updating offline buffer with these new data and their priorites
            should call change priorities, keep track of names of the data
            I/P: batch of observations
            O/P: saving states, next states, rewards in disk and update all tracker
            each state and it's next state is saved in one file
        """

        pass