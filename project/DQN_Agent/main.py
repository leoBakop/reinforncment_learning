import rlcard 
import torch
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent
from agent import Agent
import pprint
from tqdm import tqdm

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
    horizon = 10_000

    #creating two random agents
    
    agent = Agent(
        input_size= 72,
        hidden_size1= 100,
        hidden_size2=100,
        num_actions=env.num_actions,
        device=device,
        batch_size=256,
        buffer_size=10_000,
        gamma = .98,
        lr = .001,
        decrease= .999,
        goal = .01

    )
    agents=[agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    for episode in tqdm(range(horizon), desc="Processing items", unit="item"):


            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)
            agent.agents_step(trajectories[0])

    print("end of the code")       