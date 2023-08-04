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
    set_seed,
    tournament,
    reorganize
)


if __name__ == '__main__':
    seed = 42
    env = rlcard.make("limit-holdem", config={'seed': seed,})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 500_000
    num_eval_games = 1_000
    evaluate_every = 1_000
    index = 0
    
    agent = Agent(
        input_size= env.state_shape[0][0],
        hidden_size1= 128,
        hidden_size2=128,
        num_actions=env.num_actions,
        device=device,
        batch_size=32,
        buffer_size=20_000,
        gamma = .99,
        lr = .00004,
        decrease= .999999,#.99912,
        goal = .01,
        update_every= 1000

    )
    agents=[agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    rewards = np.zeros(int(horizon/evaluate_every)+1)
    for episode in tqdm(range(horizon), desc="Processing items", unit="item"):


        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)
        agent.agents_step(trajectories[0])

        #logistics/evaluation on clear data
        if episode%evaluate_every == 0:
            rewards[index] = tournament(
                        env,
                        num_eval_games,
                    )[0]
            index+=1

    print(f"the buffer size at the end is {len(agent.replay_buffer)}")


    plt.figure(1)
    plt.title(f" Agent's Reward ") 
    plt.xlabel("Round T") 
    plt.ylabel("Average Score") 
    plt.plot(np.linspace(0, horizon, int(horizon/evaluate_every)+1),rewards, label="Average reward per episode")   
    plt.grid()
    plt.legend()
    plt.show()
