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
    seed = 42
    env = rlcard.make("limit-holdem", config={'seed': seed,})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 7_000
    num_eval_games = 2000
    
    agent = Agent(
        input_size= env.state_shape[0][0],
        hidden_size1= 128,
        hidden_size2=128,
        num_actions=env.num_actions,
        device=device,
        batch_size=64,
        buffer_size=20_000,
        gamma = .99,
        lr = .00005,
        decrease= .9998,
        goal = .01,
        update_every= 1000

    )
    agents=[agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    rewards = np.zeros(horizon)
    with Logger("DQN_Agent/experiments/") as logger:
        for episode in range(horizon):


            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)
            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)
            agent.agents_step(trajectories[0])
            #logistics
            lines = trajectories[0][:][:]
            r=list([i[2] for i in lines])
            avg = np.mean(r) if r else 0
            rewards[episode] = avg

            # Evaluate the performance. Play with random agents.
            if episode % 100 == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        num_eval_games,
                    )[0]
                )
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'DQN')

    plt.figure(1)
    plt.title(f" Agent's Reward ") 
    plt.xlabel("Round T") 
    plt.ylabel("Average Score") 
    plt.plot(np.arange(1,horizon+1),rewards, label="Average reward per episode")   
    plt.grid()
    plt.legend()
    plt.show()
