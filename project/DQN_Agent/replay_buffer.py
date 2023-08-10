from collections import deque
import random
import numpy as np

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
    
class PriorizedExperienceReplay():
    def __init__(self, batch_size, buffer_size, device, alpha, beta):

        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)
        self.priorities = np.zeros(buffer_size, dtype = np.float32)
        self.index = 0
        self.full = False
        self.alpha = alpha
        self.beta = beta
    
    def add(self, tuple):
        for t in tuple:
            self.memory.append(t)
            #The new tuple must be selected at least the first time
            self.priorities[self.index] = 1 if not self.full and self.index == 0 \
                                            else self.priorities.max()
            self.index = (self.index + 1) % self.buffer_size
            self.full = len(self.memory) == self.buffer_size

    def sample(self):
        if self.full:
            prios = self.priorities
        else:
            prios = self.priorities[:self.index]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(len(self.memory), self.batch_size, p=P) 
        sampled_elements = [self.memory[idx] for idx in indices]
        
        
                
        #Compute importance-sampling weight
        weights  = (len(self.memory) * P[indices]) ** (-self.beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        state = list([s[0] for s in sampled_elements])
        action = list([s[1] for s in sampled_elements])
        reward = list([s[2] for s in sampled_elements])
        next_state = list([s[3] for s in sampled_elements])
        done = list([s[4] for s in sampled_elements])

        return state, action, reward, next_state, done, indices, weights
    
    def __len__(self):
        return len(self.memory)
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_beta(self, beta):
        self.beta = beta