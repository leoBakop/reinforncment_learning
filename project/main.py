
from agent import Agent
from implemented_agents import PolicyIterationAgent, Random_Agent, Threshold_Agent_A,Threshold_Agent_D, Q_Learning_Agent, Human_Agent
import utils
from env import Env
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import tkinter as tk 





def play_a_game(env: Env, agent: Agent, opponent:Agent, threshold=False, t = None, disp = False):
    games = 0
    total_reward = 0

    if(t>20_000 ):
        agent.reduce_a()


    while True: #play as many hands until one player bankrupts
        state, *_ = env.reset(disp = disp)
        if(isinstance(opponent, Human_Agent)): 
            print("---------------------New Hand--------------------")
            print(f"Human_agent's total money{env.game.total_money_per_player[1]}")
            print(f"Q_learning_agent's total money{env.game.total_money_per_player[0]}")
        if(not isinstance(opponent, Random_Agent)): opponent.set_hand(env.game.hand_of_player[1])
        games += 1
        if state == -1: return total_reward #means that the game has ended
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
            #remember that state, reward an done are referring only in the agent
            round = (round + 1 )%2
            if (not isinstance(opponent, Random_Agent)): #instance of human or threshold agent
                opponent.set_round(round)
                opponent.set_table(env.game.table)
            previous_tuple = [prev_state, action, reward, state, done]

            if mana == 0: #if our agent is the mana
                prev_state = state
                action = agent.send_action(state, t) if isinstance(agent, Q_Learning_Agent) else agent.send_action(state, None, None)
                if (isinstance(opponent, Human_Agent)): print(f"q learning agent's action is {action}")
                state, reward, done=env.step(action, 0, t, previous_tuple, threshold=threshold, agent=agent)
                state = utils.return_state(state, threshold, agent,preflop_state)
                if isinstance(agent, Q_Learning_Agent):
                    if(agent.train([prev_state, action, reward, state, done])):
                        print("Training must end")
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
                if (isinstance(opponent, Human_Agent)): print(f"q learning agent's action is {action}")
                state, reward, done=env.step(action, 0, t, previous_tuple, threshold=threshold, agent=agent)
                state = utils.return_state(state, threshold, agent,preflop_state)
                if isinstance(agent, Q_Learning_Agent):
                    if(agent.train([prev_state, action, reward, state, done])):
                        print("Training must end")
                total_reward += reward
                if done: break
                
            if round == 0:
                env.game.table = [env.game.dealer.deal_card(),env.game.dealer.deal_card()]
                #i have to update in this step the state
                state = env.form_state()
                state = utils.return_state(state, threshold, agent,preflop_state)



def training_main(threshold, q_learning, aggressive):
    threshold = threshold
    q_learning = q_learning
    p = utils.P_THRESHOLD_A if threshold and aggressive else \
                (utils.P_THRESHOLD_D if threshold and not aggressive else utils.P)
    seed = 15
    np.random.seed(seed)
    gamma = .9
    agent = Q_Learning_Agent(#state_size=2**10, 
                                state_size=20 if not threshold else 33, 
                                action_size= 3,
                                a=.4 ,
                                gamma=gamma,
                                against_human = False) if q_learning else \
                                PolicyIterationAgent(P=p, gamma=gamma)
    if threshold :
        opponent = Threshold_Agent_A() if aggressive else Threshold_Agent_D()
    else:
        opponent = Random_Agent(seed = seed) 

    horizon = 80_000 if threshold else  80_000
    horizon = 80_000 if not q_learning else horizon
    
    env = Env(agent, opponent, number_of_cards=5, seed=np.random.randint(low=1, high = horizon))
    r = np.zeros(horizon)
    reward = np.zeros(horizon)
    
    for t in tqdm(range(horizon), desc="Processing items", unit="item"):

        reward[t] = play_a_game(env,agent,opponent, threshold=threshold, t=t, disp = False)
        r[t] = reward[t]+r[t-1]*(t>0)
        s = np.random.randint(low = 1, high = horizon)
        env = Env(agent, opponent, number_of_cards=5, seed=s)
    
    print(f"Total reward mean for the last 1000 iterations is {np.mean(reward[-1:-1000:-1])}")
    if not q_learning:
        decisions = list([agent.pi(i) for i in range(33 if threshold else 20)])
        np.savetxt(f"./data/q_learning_{ q_learning}_threshold_{threshold}_aggressive_{aggressive}.csv", decisions)
    else:
        decisions = list([np.argmax(agent.Q[i,:]) for i in range(33 if threshold else 20)])
        np.savetxt(f"./data/q_learning_{ q_learning}_threshold_{threshold}_aggressive_{aggressive}.csv", decisions)
        np.savetxt("./data/q_agent.csv", agent.Q, delimiter = ',')

    np.savetxt(f"./data/rewards/q_learning_{ q_learning}_threshold_{threshold}_aggressive_{aggressive}.csv", r)
    
    plt.figure(1)
    plt.title(f" Agent's Reward ") 
    plt.xlabel("Round T") 
    plt.ylabel("Total Score") 
    plt.plot(np.arange(1,horizon+1),r, label="Cumulative Reward")   
    plt.grid()
    plt.legend()
    plt.savefig(f'images/q_learning_{ q_learning}_threshold_{threshold}_aggressive_{aggressive}.jpg')

    plt.show()


def testing_main():
    
    horizon = 2
    q = np.loadtxt("./data/q_agent.csv", delimiter=",",dtype = float)
    agent = Q_Learning_Agent(state_size = 33, action_size = 2, Q = q, against_human=True )
    opponent = Human_Agent(action_size=2)
    env = Env(agent, opponent, number_of_cards=5, seed=np.random.randint(low=1, high = horizon))
    r = np.zeros(horizon)
    reward = np.zeros(horizon)
    for t in tqdm(range(horizon), desc= "Processing items", unit = "item"):
        
        reward[t] = play_a_game(env,agent,opponent, threshold=threshold, t=t, disp= True)
        r[t] = reward[t]+r[t-1]*(t>0)
        s = np.random.randint(low = 1, high = horizon)
        env = Env(agent, opponent, number_of_cards=5, seed=s)

if __name__ == "__main__":

    q_learning = True
    threshold = False
    aggressive = True
    train = True
    if(train):training_main(threshold = threshold, q_learning = q_learning, aggressive = aggressive)
    else: testing_main()
