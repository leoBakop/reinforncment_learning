import rlcard 
import torch
from rlcard.agents import RandomAgent
from other_agents import Threshold_Agent, Tight_Threshold_Agent
from agent import Agent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1

from rlcard.utils import (
    set_seed,
    tournament,
    reorganize
)


if __name__ == '__main__':
    seed = 42
    env = rlcard.make("limit-holdem", config={'seed': seed,})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"used device is {device}")
    horizon = 3_000_000
    num_eval_games = 2_000 #how many hands will be played in every tournament
    evaluate_every = 100_000
    index = 0

    threshold = True
    per = False
    loose = False
    best_threshold = False
    
    threshold = True if best_threshold else threshold
    loose =False if not threshold else loose
    
    
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
        decrease= int(2*1.7*0.4*horizon), #exploration in the 40% of the horizon. # because the agent's step is called almost 1.7 times ine evry game
        goal = .1,
        per = per
    )

    agents=[agent]
    for _ in range(1, env.num_players):
        if threshold:   
            opp = Threshold_Agent() if loose else Tight_Threshold_Agent()
            opp = LimitholdemRuleAgentV1() if best_threshold else opp
        else:
            opp = RandomAgent(num_actions=env.num_actions)
        agents.append(opp)
    print(f"the opponent of the agent is {type(opp)}")
    env.set_agents(agents)

    rewards = np.zeros(int(horizon/evaluate_every))
    for episode in tqdm(range(horizon), desc="Processing items", unit="item"):

        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)
        agent.agents_step(trajectories[0])

        #logistics/evaluation on clear data
        if episode%evaluate_every == 0 and index < len(rewards):
            rewards[index] = tournament(env,num_eval_games)[0]
            index+=1

    print(f"the buffer size at the end is {len(agent.replay_buffer)}")
    file_path = f"./data/final/"
    if not os.path.exists(file_path):
    # If it doesn't exist, create the directory
        os.makedirs(file_path)
        print(f"Directory '{file_path}' has been created.")
    else:
        print(f"Directory '{file_path}' already exists.")
    torch.save(agent.model.state_dict(), file_path+f'models/threshold_{threshold}_per_{per}_loose_{loose}_best_{best_threshold}_model.pth')
    np.save(file_path+f"threshold_{threshold}_per_{per}_loose_{loose}_best_{best_threshold}.npy", rewards, allow_pickle=True)

    plt.figure(1)
    plt.title(f" Agent's Reward ") 
    plt.xlabel("Round T") 
    plt.ylabel("Average Score") 
    plt.plot(np.linspace(0, horizon, int(horizon/evaluate_every)),rewards, label="Average reward per episode")   
    plt.grid()
    plt.legend()
    plt.savefig(f"./images/threshold_{threshold}_per_{per}_loose_{loose}_best_{best_threshold}")
    plt.show()
