from agent import Agent
import numpy as np
class PolicyIterationAgent(Agent):
    def __init__(self,P, epsilon = 10**(-4), gamma=.1):
        self.P = P
        self.epsilon = epsilon
        self.gamma = gamma
        self.pi = None
        self.V = None
        return 
    

    def send_action(self, state):
        return super().send_action(state)
    
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