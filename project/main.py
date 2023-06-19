
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
                
                state, reward, done=env.step(action, 0, t, previous_tuple)
                
                if not threshold: 
                    state = state[0:10]
                    state = utils.convert_flop_state_to_num(preflop_state, state)
                else :
                    state = utils.threshold_convert_state_to_num(state)
                if isinstance(agent, Q_Learning_Agent):
                    agent.train([prev_state, action, reward, state, done])
                total_reward += reward
                if done: break
                state, reward, done = env.step(opponent.send_action(state, None, None), 1, t, previous_tuple)
                if not threshold: 
                    state = state[0:10]
                    state = utils.convert_flop_state_to_num(preflop_state, state)
                else :
                    state = utils.threshold_convert_state_to_num(state)
                total_reward += reward 
                if done: break
                
            else:
                prev_state = state
                state, reward, done=env.step(opponent.send_action(state, None, None), 1, t, previous_tuple)
                if not threshold: 
                    state = state[0:10]
                    state = utils.convert_flop_state_to_num(preflop_state, state)
                else :
                    state = utils.threshold_convert_state_to_num(state)
                total_reward += reward 
                if done: break
                action = agent.send_action(state, t) if isinstance(agent, Q_Learning_Agent) else agent.send_action(state, None, None)
                state, reward, done=env.step(action, 0, t, previous_tuple)
                
                if not threshold: 
                    state = state[0:10]
                    state = utils.convert_flop_state_to_num(preflop_state, state)
                else :
                    state = utils.threshold_convert_state_to_num(state)

                if isinstance(agent, Q_Learning_Agent):
                    agent.train([prev_state, action, reward, state, done])
                total_reward += reward
                if done: break
                
            if round == 0:
                env.game.table = [env.game.dealer.deal_card(),env.game.dealer.deal_card()]
                #i have to update in this step the state
                state = env.form_state()
                if not threshold: 
                    state = state[0:10]
                    state = utils.convert_flop_state_to_num(preflop_state, state)
                else :
                    state = utils.threshold_convert_state_to_num(state)


if __name__ == "__main__":


    threshold = True
    p = utils.P_THRESHOLD if threshold else utils.P
    agent_1 = PolicyIterationAgent(P=p)
    policy_policy_iteration = list([agent_1.pi(i) for i in range(33 if threshold else 20)])
    agent = Q_Learning_Agent(state_size=20 if not threshold else 33, 
                             action_size= 3,
                             a=.12,
                             gamma=1.0,
                             policy= np.array(policy_policy_iteration))
    opponent = Threshold_Agent() if threshold else Random_Agent()
    #agent = agent_1
    
    
    env = Env(agent, opponent, number_of_cards=5)
    horizon = 150_000
    r = np.zeros(horizon)
    dt = .000001
    for t in tqdm(range(horizon), desc="Processing items", unit="item"):
        
        r[t] = play_a_game(env,agent,opponent, threshold=threshold, t=t+dt)+r[t-1]*(t>0)
        env = Env(agent, opponent, number_of_cards=5)
    
    print(f"mean of the total reward is {np.mean(r)}")
    policy_q_learning = list([f"state {i}, action {np.argmax(j)}" for i,j in enumerate(agent.Q)])
    print(policy_q_learning)
    """ 
    sub = list([ int(i)-int(j) for i,j in zip(policy_q_learning,policy_policy_iteration)])
    print(sub.count(0)) """
    
    plt.figure(1)
    plt.title(f"Reward of the agent") 
    plt.xlabel("Round T") 
    plt.ylabel("Total score") 
    plt.plot(np.arange(1,horizon+1),r, label="cumulative reward")   
    plt.grid()
    plt.legend()
    plt.show()
