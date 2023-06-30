from agent import Agent
import numpy as np

class Random_Agent(Agent):

    def send_action(self, state):
        return np.random.randint(0,3)