import rlcard 
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent
import pprint

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
    

    #creating two random agents
    agents=[]
    for _ in range( env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

     # Generate data from the environment
    trajectories, player_wins = env.run(is_training=False)
    # Print out the trajectories
    print('\nTrajectories:')
    print(trajectories)
    print('\nSample raw observation:')
    pprint.pprint(trajectories[0][0]['raw_obs'])
    print('\nSample raw legal_actions:')
    pprint.pprint(trajectories[0][0]['raw_legal_actions'])
