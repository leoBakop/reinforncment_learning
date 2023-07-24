from collections import deque
import random

class ReplayBuffer():

    def __init__(self, batch_size, buffer_size, device):
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)

    def add(self, tuple):
        for t in tuple:
            self.memory.append(t)
    
    def sample(self):
        sampled_elements = random.sample(self.memory, self.batch_size)

        state = list([s[0] for s in sampled_elements])
        action = list([s[1] for s in sampled_elements])
        reward = list([s[2] for s in sampled_elements])
        next_state = list([s[3] for s in sampled_elements])
        done = list([s[4] for s in sampled_elements])

        return state, action, reward, next_state, done


    def __len__(self):
        return len(self.memory)   