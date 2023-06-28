
from agent import Agent
from implemented_agents import PolicyIterationAgent, Random_Agent, Threshold_Agent, Q_Learning_Agent
import utils
from env import Env
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import tkinter as tk 





def play_a_game(env: Env, agent: Agent, opponent:Agent, threshold=False, t = None):
    games = 0
    total_reward = 0

    if(t>20_000 ):
        agent.reduce_a()


    while True: #play as many hands until one player bankrupts
        state, *_ = env.reset()
        if(isinstance(opponent, Threshold_Agent)): opponent.set_hand(env.game.hand_of_player[1])
        games += 1
        if state == -1: return total_reward#means that the game has ended
        state = utils.convert_pre_flop_state_to_num(state[0:5]) if not threshold else utils.threshold_convert_state_to_num(state)
        preflop_state = state
        done = False
        mana = env.mana
        #for q_learning
        prev_state = state
        action = None
        reward = 0
        done = False
        
        
        round = 1
        while not done: #playing one hand
            #rememmber that state, reward an done are referring only in the agent
            round = (round + 1 )%2
            if (isinstance(opponent, Threshold_Agent)): 
                opponent.set_round(round)
                opponent.set_table(env.game.table)

            previous_tuple = [prev_state, action, reward, state, done]
            if mana == 0: #if our agent is the mana
                prev_state = state
                action = agent.send_action(state, t) if isinstance(agent, Q_Learning_Agent) else agent.send_action(state, None, None)
                
                state, reward, done=env.step(action, 0, t, previous_tuple, threshold=threshold, agent=agent)
                
                state = utils.return_state(state, threshold, agent,preflop_state)
                if isinstance(agent, Q_Learning_Agent):
                    agent.train([prev_state, action, reward, state, done])
                total_reward += reward
                if done: break
                state, reward, done = env.step(opponent.send_action(state, None, None), 1, t, previous_tuple, threshold=threshold, agent=agent)
                state = utils.return_state(state, threshold, agent,preflop_state)
                total_reward += reward 
                if done: break
                
            else:
                
                state, reward, done=env.step(opponent.send_action(state, None, None), 1, t, previous_tuple, threshold=threshold, agent=agent)
                state = utils.return_state(state, threshold, agent,preflop_state)
                prev_state = state
                total_reward += reward 
                if done: break
                action = agent.send_action(state, t) if isinstance(agent, Q_Learning_Agent) else agent.send_action(state, None, None)
                state, reward, done=env.step(action, 0, t, previous_tuple, threshold=threshold, agent=agent)
                state = utils.return_state(state, threshold, agent,preflop_state)
                if isinstance(agent, Q_Learning_Agent):
                    agent.train([prev_state, action, reward, state, done])
                total_reward += reward
                if done: break
                
            if round == 0:
                env.game.table = [env.game.dealer.deal_card(),env.game.dealer.deal_card()]
                #i have to update in this step the state
                state = env.form_state()
                state = utils.return_state(state, threshold, agent,preflop_state)


if __name__ == "__main__":


    threshold = True
    q_learning = False
    p = utils.P_THRESHOLD if threshold else utils.P
    seed = 15
    np.random.seed(seed)
    agent = Q_Learning_Agent(#state_size=2**10 if not threshold else 2**10, 
                             state_size=20 if not threshold else 33, 
                             action_size= 3,
                             a=.4 if threshold else .28,#.12
                             gamma=.9,
                             threshold=threshold,
                             ante= True) if q_learning else \
                             PolicyIterationAgent(P=p)
    opponent = Threshold_Agent() if threshold else Random_Agent(seed = seed)
    
    horizon = 80_000 if threshold else  25_000
    horizon = 10_000 if not q_learning else horizon
    
    env = Env(agent, opponent, number_of_cards=5, seed=np.random.randint(low=1, high = horizon))
    
    
    r = np.zeros(horizon)
    reward = np.zeros(horizon)
    dt = .000001
    for t in tqdm(range(horizon), desc="Processing items", unit="item"):
        reward[t] = play_a_game(env,agent,opponent, threshold=threshold, t=t+dt if t == 0 else t)
        r[t] = reward[t]+r[t-1]*(t>0)
        s = np.random.randint(low=1, high = horizon)
        env = Env(agent, opponent, number_of_cards=5, seed=s)
    
    print(f"mean of the total reward (for the last 1000 iterations) is {np.mean(reward[-1:-1000:-1])}")
    if not q_learning:
        decisions = list([agent.pi(i) for i in range(33 if threshold else 20)])
        np.savetxt(f"./data/q_learning_{ q_learning}_threshold_{threshold}.csv", decisions)
    else:
        decisions = list([np.argmax(agent.Q[i,:]) for i in range(33 if threshold else 20)])
        np.savetxt(f"./data/q_learning_{ q_learning}_threshold_{threshold}.csv", decisions)

    plt.figure(1)
    plt.title(f"Reward of the agent") 
    plt.xlabel("Round T") 
    plt.ylabel("Total score") 
    plt.plot(np.arange(1,horizon+1),r, label="cumulative reward")   
    plt.grid()
    plt.legend()
    plt.savefig(f'images/q_learning_{ q_learning}_threshold_{threshold}.jpg')

    plt.show()
