from agent import Agent
import numpy as np
import utils


class PolicyIterationAgent(Agent):
    def __init__(self,P, epsilon = 10**(-4), gamma=.1):
        self.P = P
        self.epsilon = epsilon
        self.gamma = gamma
        self.pi = None
        self.V = None
        self.V, self.pi = self.policy_iteration()
        return 
    

    def send_action(self, state, useless_1, useless_2):
        return  self.pi(state)
    
    def policy_evaluation(self, pi = None):
        self.pi = pi if pi is not None else self.pi
        prev_V = np.zeros(len(self.P)) # use as "cost-to-go", i.e. for V(s'):
        while True:
            V = np.zeros(len(self.P)) # current value function to be learnerd
            for s in range(len(self.P)):  # do for every state
                
                for prob, next_state, reward, done in self.P[s][self.pi(s)]:  # calculate one Bellman step --> i.e., sum over all probabilities of transitions and reward for that state, the action suggested by the (fixed) policy, the reward earned (dictated by the model), and the cost-to-go from the next state (which is also decided by the model)
                    V[s] += prob * (reward + self.gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < self.epsilon: #check if the new V estimate is close enough to the previous one; 
                break # if yes, finish loop
            prev_V = V.copy() #freeze the new values (to be used as the next V(s'))

        self.V = V    
        return

    def policy_improvement(self, gamma=1.0):  # takes a value function (as the cost to go V(s')), a model, and a discount parameter
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64) #create a Q value array
        for s in range(len(self.P)):        # for every state in the environment/model
            for a in range(len(self.P[s])):  # and for every action in that state
                for prob, next_state, reward, done in self.P[s][a]:  #evaluate the action value based on the model and Value function given (which corresponds to the previous policy that we are trying to improve) 
                    Q[s][a] += prob * (reward + gamma * self.V[next_state] * (not done))
        new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]  # this basically creates the new (improved) policy by choosing at each state s the action a that has the highest Q value (based on the Q array we just calculated)
        # lambda is a "fancy" way of creating a function without formally defining it (e.g. simply to return, as here...or to use internally in another function)
        # you can implement this in a much simpler way, by using just a few more lines of code -- if this command is not clear, I suggest to try coding this yourself
    
        self.pi = new_pi
        return new_pi
    
    def policy_iteration(self):
        t = 0
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))     # start with random actions for each state  
        self.pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]     # and define your initial policy pi_0 based on these action (remember, we are passing policies around as python "functions", hence the need for this second line)
        
        while True:
            old_pi = {s: self.pi(s) for s in range(len(self.P))}  #keep the old policy to compare with new
            self.policy_evaluation()   #evaluate latest policy --> you receive its converged value function
            self.pi = self.policy_improvement()          #get a better policy using the value function of the previous one just calculated 
            
            t += 1
                                       #and the value function evolution (for the GUI)
        
            if old_pi == {s:self.pi(s) for s in range(len(self.P))}: # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
                break
        print('converged after %d iterations' %t) #keep track of the number of (outer) iterations to converge
        return self.V,self.pi


class Q_Learning_Agent(Agent):

    def __init__(self, state_size, action_size,policy, a = .2, gamma = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.a = a
        self.Q = np.random.rand(self.state_size, self.action_size)
        self.conv = 0
        self.comparison = policy

    def train(self, tuple):
        old_q = np.array([np.argmax(i) for i in self.Q])
        #and now the training section, based on the previous tuple (knowledge)
        state, prev_action, reward, next_state,  done = tuple
        if not prev_action is None: #it is None when the aent talks first
            target = reward +self.gamma*np.argmax(self.Q[next_state])*(not done)
            self.Q[state, prev_action]= (1-self.a)*self.Q[state, prev_action] + self.a*target
        policy = np.array([np.argmax(i) for i in self.Q])
        

    def send_action(self, state, t):
        k = self.action_size
        
        eps =(1/((t)**(1/3)))*((k*np.log(t))**(1/3)) if t > 10 else 1 
        p = np.random.rand()
        #select action
        if p < eps: #make a random move
            action = np.random.choice([0,1,2])
        else : #act as q suggests
            action = np.argmax(self.Q[state,:])

        if ((not (action in [0,1,2])) and ( action is not None)):
            print("bug needs to be fixed")
            return np.random.choice([0,1,2])
        return action
        

class Random_Agent(Agent):

    def send_action(self, state, useless_1, useless_2):
        return np.random.randint(0,3)
    


class Threshold_Agent(Agent):
    '''
    0 : A pre flop     4 : K pre flop   8 : Q pre flop   12 : J pre flop  16 : 10 pre flop   
    1 : A -A*          5 : K -K*        9 : Q - Q*       13 : J - J*      17 : 10 -10*
    2 : A - AA         6 : K-KK         10 : Q-QQ        14 : J - JJ      18 : 10 -10 10
    3 : A - **         7 : K - **       11 : Q - **      15 : J - **      19 : 10 - **

    ---------------  ---------------  ---------------  ---------------  -------------------
    -----------actions----------
    0: check
    1: fold
    2: raise
    '''
    def set_hand(self, hand):
        self.hand = hand
    def set_table(self, table):
        self.table = table
    def set_round(self, round):
        self.round = round   
    def send_action(self, state, useless_1, useless_2):
        
        cards_and_actions_round_0 = {
            "T": 1,
            "J": 0,
            "Q":0,
            "K":0,
            "A": 2
        }

        if self.round == 0:
            return cards_and_actions_round_0.get(self.hand.rank, 0) #by default check
        for card in self.table:
            if card.rank == self.hand.rank: return 2
        return 0