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
        self.batch_size = batch_size
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


    def predict(self,state, network):
        
        ''' Predict the masked Q-values

        Args:
            state (numpy.array): current state,
            the network that is going to use

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value and sets -inf to the illegal 
        '''
        training = len(state) != 5 
        network.eval()
        if not training: #in case that there is just one tuple
            input = torch.Tensor(np.expand_dims(state['obs'], 0)).to(self.device)
        else:
            new_state_obs = []
            new_state_legal_action = []
            for s in state:
                new_state_obs.append(s['obs'])
                new_state_legal_action.append(list(s['legal_actions'].keys()))
            input = torch.Tensor(np.array(new_state_obs)).to(self.device)
        #taking all the q-values
        with torch.no_grad():
            q_values = network(input)[0].cpu().numpy() if not training else network(input).cpu().numpy()
        network.train()
        #mask the illegal actions
        rows = 1 
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float) if not training else -np.inf*np.ones((self.batch_size, self.num_actions), dtype = float)
        #I want the keys not the values and I have implement the values
        legal_actions = list(state['legal_actions'].keys()) if len(state) == 5 else list(new_state_legal_action)
        if training:
            for i,(m,q) in enumerate(zip(masked_q_values, q_values)):
                m[legal_actions[i]] = q[legal_actions[i]]
        else:
            masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values
    
    
    def eval_step(self,state):
        """ 
        method required from the rl-card environment.
        This method is called in env.run(is_training = False) 
        and returns a clear action.    
        """
        qs = self.predict(state, network = self.model)
        action = np.argmax(qs)

        #code from rlcards dqn in order to send the best legal action
        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(qs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action, info

    def step(self, state):
        """ 
        method required from the rl-card environment.
        This method is called in env.run(is_training = True) 
        and returns a noisy action.    
        """
        self.update_eps()
        p = random.random()
        legal_actions = list(state['legal_actions'].keys())

        if p < self.eps: #return random move
            return legal_actions[random.randint(a = 0, b =len(legal_actions)-1)]
        
        q_values = self.predict(state, network = self.model)
        for _ in q_values:
            best_action = np.argmax(q_values)
            if best_action in legal_actions: return np.argmax(q_values)
            q_values[best_action] = -1000 #just in order to choose the less better action
        print("error in step")
        return legal_actions[0] 

    def agents_step(self, tuples):
        """ 
        method responsible for storing the new experience in replay buffer,
        and train the agent. Basically the method must be called in every timestep
        of the training loop
        """
        #stores new experience in replay buffer
        self.push(tuples)
        if len(self.replay_buffer) < 2*self.batch_size: return 

        #enough experience was stored, so I can sample a minibatch
        experience = self.replay_buffer.sample()
        #now it is time for training
        self.train(experience)

    def update_eps(self):
        if self.eps > self.goal:
            self.eps = max(self.eps*self.decrease, self.goal)
        if (self.eps == self.goal):
            print("----------training was ended--------------")
            self.eps = self.goal*self.decrease

    def train(self,experience):
        state, action, reward, next_state, done = experience
        next_qs =self.predict(state = next_state, network=self.target_model)
        next_qs = np.argmax(next_qs, axis=1)
        #works fine until now      