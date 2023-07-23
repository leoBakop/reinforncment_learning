from collections import deque
import random

class ReplayBuffer():

    def __init__(self, batch_size, buffer_size, device):
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)

    def add(self, tuple):
        self.memory.append(tuple)
    
    def sample(self):
        sampled_elements = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = sampled_elements
        return state, action, reward, next_state, done
        