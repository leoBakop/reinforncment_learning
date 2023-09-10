import rlcard 
import torch
import numpy as np
from rlcard.agents import LimitholdemHumanAgent as Human

from agent import Agent


from rlcard.utils import (
    set_seed,
    tournament,
    reorganize
)

def load_model(agent: Agent):
    w_t = .4
    w_b = .6
    tight = torch.load('data/final/models/threshold_True_per_True_loose_False_best_False_model.pth')
    best = torch.load('data/final/models/threshold_True_per_True_loose_True_best_True_model.pth')
    
    for (best_param_name, best_param_tensor), (_, tight_param_tensor) in zip(best.items(), tight.items()):
        best[best_param_name] = best_param_tensor * w_b + tight_param_tensor * w_t
    weights = best
    agent.load_model(weights)
    return agent

def main():
    seed = 42
    env = rlcard.make("limit-holdem", config={'seed': seed,})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    horizon = 2
    num_eval_games = 5 #how many hands will be played in every tournament
    


    agent = Agent(
        input_size= env.state_shape[0][0],
        hidden_size1= 512,
        hidden_size2=256,
        num_actions=env.num_actions,
        device=device,
        batch_size=64,
        buffer_size=100_000,
        gamma = .99,
        lr = 10**(-5), #good lr is .00003
        decrease= int(2*1.7*0.4*horizon), #exploration in the 40% of the horizon. because the agent's step is called almost 1.7 times ine evry game
        goal = .1,
        per = True
    )
    agent = load_model(agent)
    agents = [agent]

    for _ in range(1, env.num_players):
        agents.append(Human(env.num_actions))
    env.set_agents(agents)
    reward = np.zeros((horizon,2))
    for i in range(horizon):
        r = tournament(env,num_eval_games)
        reward[i][0] = r[0]
        reward[i][1] = r[1]
    means = np.mean(reward, axis=0)
    print('--------------------------------')
    print("average score is")
    print(f"Human player: {means[1]}")
    print(f"Agent : {means[0]}")
if __name__ == '__main__':
    main()