import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from network import DQN_Network
from replay_buffer import ReplayBuffer, PriorizedExperienceReplay

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
                    goal = .02,
                    per = False,
                    a = 0,
                    b = 0
                ):
        self.device = device

        #networks
        self.num_actions= num_actions
        self.model = DQN_Network(input_size, hidden_size1, hidden_size2, num_actions).to(self.device)
        self.target_model = DQN_Network(input_size, hidden_size1, hidden_size2, num_actions).to(self.device)

        

        #memory staffs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.per = per
        if not per:
            self.replay_buffer = ReplayBuffer(batch_size= self.batch_size,buffer_size= self.buffer_size,device= self.device)
        else:
            self.a = a
            self.b = b
            self.offset = .05
            self.replay_buffer = PriorizedExperienceReplay(batch_size= self.batch_size,buffer_size= self.buffer_size,device= self.device, alpha = a, beta = b)
            

        #miscellaneous
        self.gamma = gamma
        self.horizon = horizon
        self.lr = lr
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.use_raw = False #just for the env, False because the agent is not a human
        self.counter = 0
        self.TAU = .005
        #epsilon greedy part
        self.eps = 1.0
        self.decrease = decrease
        self.goal = goal
        self.dt = .001
        self.epsilon_values = np.linspace(1.0, goal+self.dt, self.decrease)
        self.index = 0
    
    def push(self, tuples):
        self.replay_buffer.add(tuples)


    def no_grad_predict(self,state, network):
        
        ''' Predict the masked Q-values

        Args:
            state (numpy.array): current state,
            the network that is going to use,
            model (boolean): defines if the model or the model network is used (in case of training) in order to
            use or not grand, 

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value and sets -inf to the illegal 
        '''
        training = len(state) != 5 
        network.eval()
        #we should remember that state['obs'] is the 72 (or more in case of an extended environment) vector
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
            q_values = network(input)[0].cpu().detach().numpy() if not training else network(input).cpu().detach().numpy()
        
        network.train()
        #mask the illegal actions
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float) if not training else -np.inf*np.ones((self.batch_size, self.num_actions), dtype = float)
        #I want the keys not the values and I have implement the values
        legal_actions = list(state['legal_actions'].keys()) if len(state) == 5 else list(new_state_legal_action)
        if training:
            for i,(m,q) in enumerate(zip(masked_q_values, q_values)):
                m[legal_actions[i]] = q[legal_actions[i]] #replace the -infinity with the tru value when an action is legal
        else:
            masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values
    
    def eval_step(self,state):
        """ 
        method required from the rl-card environment.
        This method is called in env.run(is_training = False) 
        and returns a clear action.    
        """
        qs = self.no_grad_predict(state, network = self.model)
        action = np.argmax(qs)
        
        return action, None

    def step(self, state):
        """ 
        method required from the rl-card environment.
        This method is called in env.run(is_training = True) 
        and returns an action, selected by eps-greedy.    
        """
        self.update_eps()
        p = random.random()
        legal_actions = list(state['legal_actions'].keys())

        if p < self.eps: #return random move
            return legal_actions[random.randint(a = 0, b =len(legal_actions)-1)]
        
        q_values = self.no_grad_predict(state, network = self.model)
        
        return np.argmax(q_values)
            

    def agents_step(self, tuples):
        """ 
        method responsible for storing the new experience in replay buffer,
        and train the agent. Basically the method must be called in every timestep
        of the training loop
        """
        self.counter += 1
        #stores new experience in replay buffer
        self.push(tuples)
        if len(self.replay_buffer) < 2*self.batch_size: return 

        #enough experience was stored, so I can sample a minibatch
        experience = self.replay_buffer.sample()
        self.set_per_values()
        #now it is time for training
        if not self.per:self.train(experience)
        else: self.train_per(experience)
        


    def update_eps(self):
        if self.eps > self.goal+self.dt:
            self.index +=  1
            self.eps = self.epsilon_values[self.index]

        if (self.eps == self.goal + self.dt):
            print("\n----------exploration ended--------------")
            self.eps = self.goal - self.dt
            #self.optimizer.param_groups[0]['lr'] = self.lr*0.1

    def train_per(self,experience):
        self.model.train()
        self.optimizer.zero_grad()
        state, action, reward, next_state, done, idx, weights = experience
        action = torch.tensor(action).to(self.device)
        legal_actions_batch = list([ns['legal_actions'] for ns in next_state])

        #calulating the max(Q(s',a'))
        next_qs = self.no_grad_predict(state = next_state, network = self.target_model)
 
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])

        #masking the illegal moves for Q(s',a')
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = next_qs.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        #calculating the best action based in the Q(s', a')
        best_actions = np.argmax(masked_q_values, axis=1)
        
        #calulating the target
        done = list(map(float, done))
        ones= np.ones_like(done)
        y = reward + self.gamma* next_qs[np.arange(self.batch_size), best_actions]*(ones-done)
        y = torch.tensor(y, dtype = torch.float32).to(self.device)
        #so y = rewards + gamma* max(Q(s',a')) * done
        #calulating the Q(s,a) using the model network
        state = list([s['obs'] for s in state])
        state = torch.Tensor(np.array(state)).to(self.device)
        qs = self.model(state) #calulating the Q(s,a) for every a
        Q = torch.gather(qs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1).to(self.device) #filtering the selected a


        #It's time for training
        w = torch.Tensor((weights**(1-self.b))).to(self.device)
        loss = (self.criterion(Q, y)*w).mean()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        self.model.eval()

        #updating the propabillities
        td_error =  Q - y
        difference = td_error + self.offset
        self.replay_buffer.update_priorities(idx, abs(difference))
        
        return


    def train(self,experience):

        self.model.train()
        self.optimizer.zero_grad()

        state, action, reward, next_state, done = experience
        action = torch.tensor(action).to(self.device)
        legal_actions_batch = list([ns['legal_actions'] for ns in next_state])

        #calulating the max(Q(s',a'))                               
        next_qs = self.no_grad_predict(state = next_state, network = self.target_model)
 
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])

        #masking the illegal moves for Q(s',a')
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = next_qs.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        #calculating the best action based in the Q(s', a')
        best_actions = np.argmax(masked_q_values, axis=1)
        
        #calulating the target
        done = list(map(float, done))
        ones= np.ones_like(done)
        y = reward + self.gamma* next_qs[np.arange(self.batch_size), best_actions]*(ones-done)
        y = torch.tensor(y, dtype = torch.float32).to(self.device)
        #so y = rewards + gamma* max(Q(s',a')) * done
        #calulating the Q(s,a) using the model network
        state = list([s['obs'] for s in state])
        state = torch.Tensor(np.array(state)).to(self.device)
        qs = self.model(state) #calulating the Q(s,a) for every a
        Q = torch.gather(qs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1) #filtering the selected a


        #It's time for training
        loss = self.criterion(Q,y)
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        self.model.eval()

       

        return

    def soft_update(self):
        
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)
   
    def set_per_values(self):
        if not self.per: return
        self.replay_buffer.set_alpha(1-self.eps)
        self.replay_buffer.set_beta(1-self.eps)
    

    def load_model(self, weights):
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(weights)
        self.model.eval()
        self.target_model.eval()