import rlcard 
import torch
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent
from agent import Agent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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

    horizon = 50_000
    num_eval_games = 2_000 #how many hands will be played in every tournament
    evaluate_every = 1_500
    index = 0
    
    agent = Agent(
        input_size= env.state_shape[0][0],
        hidden_size1= 512,
        hidden_size2=256,
        num_actions=env.num_actions,
        device=device,
        batch_size=32,
        buffer_size=20_000,
        gamma = .99,
        lr = .00003,
        decrease= .99989,#.999999
        goal = .01,
        update_every= 1000
    )

    agents=[agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
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


    plt.figure(1)
    plt.title(f" Agent's Reward ") 
    plt.xlabel("Round T") 
    plt.ylabel("Average Score") 
    plt.plot(np.linspace(0, horizon, int(horizon/evaluate_every)),rewards, label="Average reward per episode")   
    plt.grid()
    plt.legend()
    plt.savefig("./DQN_Agent/")
    plt.show()
