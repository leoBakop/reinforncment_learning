
from agent import Agent
from policy_iteration_agent import PolicyIterationAgent
from random_agent import Random_Agent
from threshold_agent import Threshold_Agent
from env import Env
import numpy as np
import matplotlib.pyplot as plt
#import tkinter as tk 
#in case of Policy Iteratation 
'''
0 : A pre flop     4 : K pre flop   8 : Q pre flop   12 : J pre flop  16 : 10 pre flop   
1 : A -A*          5 : K -K*        9 : Q - Q*       13 : J - J*      17 : 10 -10*
2 : A - AA         6 : K-KK         10 : Q-QQ        14 : J - JJ      18 : 10 -10 10
3 : A - **         7 : K - **       11 : Q - **      15 : J - **      19 : 10 - **

---------------  ---------------  ---------------  ---------------  -------------------
-----------actions----------
0: check
1: fold
2: raise
'''
BEST_REWARD = 100
WORST_REWARD = -BEST_REWARD
MED_REWARD = BEST_REWARD/2
LOW_MED_REWARD = BEST_REWARD/4
LOW_BEST_REWARD = (3/4)*BEST_REWARD


P = {
    # A - pre flop
   0: {
        #action - check
        0: [(0.35, 1, 0.0,False),
            (0.3, 2, 0.0,False),
            (0.35,3,0.0,False)
        ],
       #action -fold
        1: [(1, 0, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.35, 1, 0.0,False),
            (0.3, 2, 0.0,False),
            (0.35, 3, 0.0,False)
        ]
    },
     #A- A*
    1: {
        #check
        0: [(0.5, 1, 0.0,False),
            (0.5, 1, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 1, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 1, BEST_REWARD,True),
            (0.5, 1, 0.0,False)

        ]
    },
     #A-AA
    2: {
        #check
        0: [(0.0, 2, 0.0,False),
            (1, 2, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 2, WORST_REWARD,True)
        ],
        #raise
        2: [(0.0, 1, BEST_REWARD,True),
            (1, 1, 0.0,False)
        ]
    },
    #A - **
    3: {
        #check
        0: [(0.5, 3, 0.0,False),
            (0.5, 3,LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 3, LOW_MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 3, MED_REWARD,True), #only for A-** since A is the highest card
            (0.5, 3, 0.0,False)
        ]
    },
    # K - pre flop
   4: {
        #check
        0: [(0.35, 5, 0.0,False),
            (0.3, 6, 0.0,False),
            (0.35,7,0.0,False)
        ],
       #fold
        1: [(1, 4, WORST_REWARD, True)
        ],
       #raise
        2: [(0.35, 5, 0.0,False),
            (0.3, 6, 0.0,False),
            (0.35, 7, 0.0,False)
        ]
    },
     #K- K*
    5: {
        #check
        0: [(0.5, 5, 0.0,False),
            (0.5, 5, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 5, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 5, BEST_REWARD,True),
            (0.5, 5, 0.0,False)

        ]
    },
     #K-KK
    6: {
        #check
        0: [(0.5, 6, 0.0,False),
            (0.5, 6, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 6, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 6, BEST_REWARD,True),
            (0.5, 6, 0.0,False)
        ]
    },
    #K - **
    7: {
        #check
        0: [(0.5, 7, 0.0,False),
            (0.5, 7, MED_REWARD,True),

        ],
        #fold
        1: [(1, 7, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 7, LOW_MED_REWARD,True),
            (0.5, 7, 0.0,False)
        ]
    },
    # Q - pre flop
    8: {
        #check
        0: [(0.35,9, 0.0,False),
            (0.3, 10, 0.0,False),
            (0.35,11,0.0,False)
        ],
       #fold
        1: [(1, 8, MED_REWARD, True)
        ],
       #raise
        2: [(0.35, 9, 0.0,False),
            (0.3, 10, 0.0,False),
            (0.35,11, 0.0,False)
        ]
    },
     #Q- Q*
    9: {
        #check
        0: [(0.5, 9, 0.0,False),
            (0.5, 9,LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 9, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 9, BEST_REWARD,True),
            (0.5, 9, 0.0,False)

        ]
    },
     #Q-QQ
    10: {
        #check
        0: [(0.5, 10, 0.0,False),
            (0.5, 10, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 10, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 10, BEST_REWARD,True),
            (0.5, 10, 0.0,False)
        ]
    },
    #Q - **
    11: {
        #check
        0: [(0.5, 11, 0.0,False),
            (0.5, 11, MED_REWARD,True),

        ],
        #fold
        1: [(1, 11, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 11, WORST_REWARD,True),
            (0.5, 11, 0.0,False)
        ]
    },     
    # J - pre flop
   12: {
        #check
        0: [(0.35,13, 0.0,False),
            (0.3, 14, 0.0,False),
            (0.35,15,0.0,False)
        ],
       #fold
        1: [(1, 12, WORST_REWARD, True)
        ],
       #raise
        2: [(0.35,13, 0.0,False),
            (0.3, 14, 0.0,False),
            (0.35,15, 0.0,False)
        ]
    },
     #J- J*
    13: {
        #check
        0: [(0.5, 13, 0.0,False),
            (0.5, 13, MED_REWARD,True),

        ],
        #fold
        1: [(1, 13, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 13, LOW_BEST_REWARD,True),
            (0.5, 13, 0.0,False)

        ]
    },
     #J-JJ
    14: {
        #check
        0: [(0.5, 14, 0.0,False),
            (0.5, 14, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 14, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 14, BEST_REWARD,True),
            (0.5, 14, 0.0,False)
        ]
    },
    #J - **
    15: {
        #check
        0: [(0.5, 15, 0.0,False),
            (0.5, 15, MED_REWARD,True),

        ],
        #fold
        1: [(1, 15, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 15, WORST_REWARD,True),
            (0.5, 15, 0.0,False)
        ]
    },
    # 10 - pre flop
   16: {
        #action - check
        0: [(0.35, 17, 0.0,False),
            (0.3, 18, 0.0,False),
            (0.35,19,0.0,False)
        ],
       #action -fold
        1: [(1, 16, LOW_MED_REWARD, True)
        ],
       #action -raise
        2: [(0.35, 17, 0.0,False),
            (0.3, 18, 0.0,False),
            (0.35, 19, 0.0,False)
        ]
    },
     #10-10*
    17: {
        #check
        0: [(0.5, 17, 0.0,False),
            (0.5, 17, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 17, MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 17, WORST_REWARD,True),
            (0.5, 17, 0.0,False)

        ]
    },
     #10-1010
    18: {
        #check
        0: [(0.5, 18, 0.0,False),
            (0.5, 18, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 18, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 18, BEST_REWARD,True),
            (0.5, 18, 0.0,False)
        ]
    },
    #10-**
    19: {
        #check
        0: [(0.5, 19, 0.0,False),
            (0.5, 19, MED_REWARD,True),

        ],
        #fold
        1: [(1, 19, LOW_BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 19, WORST_REWARD,True),
            (0.5, 19, 0.0,False)
        ]
    },
}

'''
The order of the each tuple is [card in hand, card on the table, phase, opponent last action]

0: A pre flop, raise	9: K pre flop, raise	18: Q pre flop, raise	27: J pre flop, raise	36: T pre flop, raise
1: A pre flop, check	10: K pre flop, check	19: Q pre flop, check	28: J pre flop, check	37: T pre flop, check
2: A pre flop, no info	11: K pre flop, no info	20: Q pre flop, no info	29: J pre flop, no info	38: T pre flop, no info
3: A -A*,  flop, raise	12: K-K* flop, raise	21: Q -Q*,  flop, raise	30: J-J*,  flop, raise	39: T -T*,  flop, raise
4: A -A*, flop, check	13: K-K* flop, check	22: Q-Q*, flop, check	31: J-J*, flop, check	40: T-T*, flop, check
5: A -AA, flop, raise	14: K-KK flop, raise	23: Q -QQ, flop, raise	32: J -JJ, flop, raise	41: T-TT, flop, raise
6: A - AA, flop, check	15: K-KK flop, check	24: Q- QQ, flop, check	33: J - JJ, flop, check	42: T- TT, flop, check
7:A- ** , flop, raise	16: K-** flop, raise	25:Q- ** , flop, raise	34:J- ** , flop, raise	43:T- ** , flop, raise
8: A- **, flop, check	17: K-** flop, check	26: Q- **, flop, check	35: J- **, flop, check	44: T- **, flop, check

---------------  ---------------  ---------------  ---------------  -------------------
-----------actions----------
0: check
1: fold
2: raise
'''

P_THRESHOLD= {
    # A - pre flop
   0: {
        #action - check
        0: [(0.35, 1, 0.0,False),
            (0.3, 2, 0.0,False),
            (0.35,3,0.0,False)
        ],
       #action -fold
        1: [(1, 0, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.25, 1, 0.0,False),
            (0.25, 2, 0.0,False),
            (0.25, 3, 0.0,False),
            (.25,0,MED_REWARD,True)        
            ]
    },
     #A- A*
    1: {
        #check
        0: [(0.5, 1, 0.0,False),
            (0.5, 1, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 1, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 1, BEST_REWARD,True),
            (0.5, 1, 0.0,False)

        ]
    },
     #A-AA
    2: {
        #check
        0: [(0.5, 2, 0.0,False),
            (0.5, 2, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 2, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 1, BEST_REWARD,True),
            (0.5, 1, 0.0,False)
        ]
    },
    #A - **
    3: {
        #check
        0: [(0.5, 3, 0.0,False),
            (0.5, 3,LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 3, LOW_MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 3, WORST_REWARD,True), #####################################
            (0.5, 3, 0.0,False)
        ]
    },
    # K - pre flop
   4: {
        #check
        0: [(0.35, 5, 0.0,False),
            (0.3, 6, 0.0,False),
            (0.35,7,0.0,False)
        ],
       #fold
        1: [(1, 4, WORST_REWARD, True)
        ],
       #raise
        2: [(0.35, 5, 0.0,False),
            (0.3, 6, 0.0,False),
            (0.35, 7, 0.0,False)
        ]
    },
     #K- K*
    5: {
        #check
        0: [(0.5, 5, 0.0,False),
            (0.5, 5, WORST_REWARD,True), ##########################33

        ],
        #fold
        1: [(1, 5, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 5, BEST_REWARD,True),
            (0.5, 5, 0.0,False)

        ]
    },
     #K-KK
    6: {
        #check
        0: [(0.5, 6, 0.0,False),
            (0.5, 6, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 6, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 6, BEST_REWARD,True),
            (0.5, 6, 0.0,False)
        ]
    },
    #K - **
    7: {
        #check
        0: [(0.5, 7, 0.0,False),
            (0.5, 7, MED_REWARD,True),

        ],
        #fold
        1: [(1, 7, LOW_MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 7, LOW_MED_REWARD,True),
            (0.5, 7, 0.0,False)
        ]
    },
    # Q - pre flop
    8: {
        #check
        0: [(0.35,9, 0.0,False),
            (0.3, 10, 0.0,False),
            (0.35,11,0.0,False)
        ],
       #fold
        1: [(1, 8, WORST_REWARD, True) #################################3
        ],
       #raise
        2: [(0.35, 9, 0.0,False),
            (0.3, 10, 0.0,False),
            (0.35,11, 0.0,False)
        ]
    },
     #Q- Q*
    9: {
        #check
        0: [(0.5, 9, 0.0,False),
            (0.5, 9,LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 9, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 9, BEST_REWARD,True),
            (0.5, 9, 0.0,False)

        ]
    },
     #Q-QQ
    10: {
        #check
        0: [(0.5, 10, 0.0,False),
            (0.5, 10, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 10, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 10, BEST_REWARD,True),
            (0.5, 10, 0.0,False)
        ]
    },
    #Q - **
    11: {
        #check
        0: [(0.5, 11, 0.0,False),
            (0.5, 11, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 11, LOW_BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 11, WORST_REWARD,True),
            (0.5, 11, 0.0,False)
        ]
    },     
    # J - pre flop
   12: {
        #check
        0: [(0.35,13, 0.0,False),
            (0.3, 14, 0.0,False),
            (0.35,15,0.0,False)
        ],
       #fold
        1: [(1, 12, MED_REWARD, True)
        ],
       #raise
        2: [(0.35,13, 0.0,False),
            (0.3, 14, 0.0,False),
            (0.35,15, 0.0,False)
        ]
    },
     #J- J*
    13: {
        #check
        0: [(0.5, 13, 0.0,False),
            (0.5, 13, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 13, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 13, BEST_REWARD,True),
            (0.5, 13, 0.0,False)

        ]
    },
     #J-JJ
    14: {
        #check
        0: [(0.5, 14, 0.0,False),
            (0.5, 14, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 14, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 14, BEST_REWARD,True),
            (0.5, 14, 0.0,False)
        ]
    },
    #J - **
    15: {
        #check
        0: [(0.5, 15, 0.0,False),
            (0.5, 15, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 15, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 15, WORST_REWARD,True),
            (0.5, 15, 0.0,False)
        ]
    },
    # 10 - pre flop
   16: {
        #action - check
        0: [(0.35, 17, 0.0,False),
            (0.3, 18, 0.0,False),
            (0.35,19,0.0,False)
        ],
       #action -fold
        1: [(1, 16,LOW_BEST_REWARD, True)
        ],
       #action -raise
        2: [(0.35, 17, 0.0,False),
            (0.3, 18, 0.0,False),
            (0.35, 19, 0.0,False)
        ]
    },
     #10-10*
    17: {
        #check
        0: [(0.5, 17, 0.0,False),
            (0.5, 17, MED_REWARD,True),

        ],
        #fold
        1: [(1, 17, LOW_BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 17, LOW_MED_REWARD,True),
            (0.5, 17, 0.0,False)

        ]
    },
     #10-1010
    18: {
        #check
        0: [(0.5, 18, 0.0,False),
            (0.5, 18, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 18, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 18, BEST_REWARD,True),
            (0.5, 18, 0.0,False)
        ]
    },
    #10-**
    19: {
        #check
        0: [(0.5, 19, 0.0,False),
            (0.5, 19, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 19, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 19, WORST_REWARD,True),
            (0.5, 19, 0.0,False)
        ]
    },
}
#creating a random policy 


def convert_pre_flop_state_to_num(state):
    card = np.where(np.array(state) == 1)
    state = (4-card[0][0])*4
    return state

def convert_flop_state_to_num(preflop_state,state):
    """ state is the full state (hand+table) =an array of 10 numbers/5 for preflop 5 for flop cards"""
    indices = np.where(np.array(state) == 1)[0]
    if(len(indices)==1): return preflop_state #if we find one 1 then we re at the preflop state
    if(len(indices) == 2): #we have a pair on the table, otherwise it would be 3
        if indices[0] + 5 == indices[1]: #if the pair is the same rank as the preflop
            return preflop_state + 2
        return preflop_state +3 # e.g. A-** 
    if ((indices[0]+5 == indices[1] or indices[0]+ 5 == indices[2] )and(len(indices) == 3)): return preflop_state + 1
    return preflop_state +3
    
    


def play_a_game(env: Env, agent: Agent, opponent:Agent, threshold):
    games = 0
    total_reward = 0
    while True: #play as many hands until one player bankrupts
        state, *_ = env.reset()
        games += 1
        if state == -1: return total_reward#means that the game has ended
        state = convert_pre_flop_state_to_num(state[0:5])
        preflop_state = state
        done = False
        mana = env.mana
        
        while not done: #playing one hand
            #rememmber that state, reward an done are referring only in the agent
            if mana == 0: #if our agent is the mana
                state, reward, done=env.step(agent.send_action(state), 0, state)
                if not threshold: state = state[0:10]
                state = convert_flop_state_to_num(preflop_state, state)
                total_reward += reward
                if done: break
                state, reward, done = env.step(opponent.send_action(state), 1, state)
                if not threshold: state = state[0:10]
                state = convert_flop_state_to_num(preflop_state, state)
                total_reward += reward 
                if done: break

            else:
                state, reward, done=env.step(opponent.send_action(state), 1, state)
                if not threshold: state = state[0:10]
                state = convert_flop_state_to_num(preflop_state, state)
                total_reward += reward 
                if done: break
                state, reward, done=env.step(agent.send_action(state), 0,state)
                if not threshold: state = state[0:10]
                state = convert_flop_state_to_num(preflop_state, state)
                total_reward += reward
                if done: break

def play_an_episode(env: Env, agent: Agent, opponent:Agent):
    pass


if __name__ == "__main__":
    
    threshold = False
    p = P_THRESHOLD if threshold else P
    agent = PolicyIterationAgent(P=p)
    
    opponent = Threshold_Agent() if threshold else Random_Agent()
    env = Env(agent, opponent, number_of_cards=5)
    horizon = 1000
    r = np.zeros(horizon)
    for i in range(horizon):
        r[i] = play_a_game(env,agent,opponent, threshold) + r[i-1] if i > 0 else play_a_game(env,agent,opponent, threshold)
        env = Env(agent, opponent, number_of_cards=5)
    
    plt.figure(1)
    plt.title(f"Reward of the agent") 
    plt.xlabel("Round T") 
    plt.ylabel("Total score") 
    plt.plot(np.arange(1,horizon+1),r, label="cumulative reward")   
    plt.grid()
    plt.legend()
    plt.show()
