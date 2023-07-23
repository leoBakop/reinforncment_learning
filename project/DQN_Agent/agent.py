import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from network import DQN_Network
from replay_buffer import ReplayBuffer

class Agent():
    def __init__(   self,
                    input_size, 
                    hidden_size1, 
                    hidden_size2, 
                    num_actions,
                    device,
                    batch_size = 256,
                    buffer_size = 10_000,
                    gamma = .99,
                    horizon = 1_000_000,
                    lr = .001,
                    decrease = .99,
                    goal = .02
                ):
        self.device = device

        #networks
        self.num_actions= num_actions
        self.model = DQN_Network(input_size, hidden_size1, hidden_size2, num_actions).to(self.device)
        self.target_model = DQN_Network(input_size, hidden_size1, hidden_size2, num_actions).to(self.device)

        #memory staffs
        self.batch_size = batch_size,
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(batch_size= self.batch_size,buffer_size= self.buffer_size,device= self.device)

        #miscellaneous
        self.gamma = gamma
        self.horizon = horizon
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.use_raw = False #just for the env, False because the agent is not a human
        #epsilon greedy part
        self.eps = 1
        self.decrease = decrease
        self.goal = goal
    
    def push(self, tuples):
        self.replay_buffer.add(tuples)


    def predict(self,state):
        
        ''' Predict the masked Q-values

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value and sets -inf to the illegal 
        '''
        self.model.eval()
        input = torch.Tensor(np.expand_dims(state['obs'], 0)).to(self.device)
        #taking all the q-values
        with torch.no_grad():
            q_values = self.model(input)[0].cpu().numpy()
        self.model.train()
        #mask the illegal actions
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values
    
    def eval_step(self,state):
        qs = self.predict(state)
        action = np.argmax(qs)

        #code from rlcards dqn in order to send the best legal action
        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(qs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action, info

    def step(self, state):
        
        self.update_eps()
        p = random.random()
        legal_actions = list(state['legal_actions'].keys())

        if p < self.eps: #return random move
            return legal_actions[random.randint(a = 0, b =len(legal_actions)-1)]
        
        q_values = self.predict(state)
        for _ in q_values:
            best_action = np.argmax(q_values)
            if best_action in legal_actions: return np.argmax(q_values)
            q_values[best_action] = -1000 #just in order to choose the less better action
        print("error in step")
        return legal_actions[0] 

    def update_eps(self):
        if self.eps > self.goal:
            self.eps = max(self.eps*self.decrease, self.goal)
        if (self.eps == self.goal):
            print("----------training was ended--------------")
            self.eps = self.goal*self.decrease