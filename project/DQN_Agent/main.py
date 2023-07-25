import rlcard 
import torch
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent
from agent import Agent
import pprint
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


if __name__ == '__main__':
    seed = 0
    env = rlcard.make("limit-holdem", config={'seed': seed,})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 50_000
    MAX_REWARD = 11.5 #just an approximation for the highest reward, in order to normillize the rewards
    #creating two random agents
    
    agent = Agent(
        input_size= 72,
        hidden_size1= 100,
        hidden_size2=100,
        num_actions=env.num_actions,
        device=device,
        batch_size=256,
        buffer_size=100_000,
        gamma = .98,
        lr = .001,
        decrease= .9999,
        goal = .01,
        update_every= 5

    )
    agents=[agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    rewards = np.zeros(horizon)

    for episode in tqdm(range(horizon), desc="Processing items", unit="item"):


            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)
            payoffs /= MAX_REWARD
            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)
            agent.agents_step(trajectories[0])
            #logistics
            lines = trajectories[0][:][:]
            r=list([i[2] for i in lines])
            avg = np.average(r)
            rewards[episode] = 0 if np.isnan(avg) else avg

    plt.figure(1)
    plt.title(f" Agent's Reward ") 
    plt.xlabel("Round T") 
    plt.ylabel("Average Score") 
    plt.plot(np.arange(1,horizon+1),rewards, label="Average reward per episode")   
    plt.grid()
    plt.legend()
    plt.show()
