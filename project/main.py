
from agent import Agent
from policy_iteration_agent import PolicyIterationAgent
from random_agent import Random_Agent
from env import Env
import numpy as np
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
BEST_REWARD = 10
WORST_REWARD = -BEST_REWARD
MED_REWARD = BEST_REWARD/2

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
            (0.5, 1, MED_REWARD,True),

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
            (0.5, 2, MED_REWARD,True),

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
            (0.5, 3, MED_REWARD,True),

        ],
        #fold
        1: [(1, 3, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 3, WORST_REWARD,True),
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
            (0.5, 5, MED_REWARD,True),

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
            (0.5, 6, MED_REWARD,True),

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
        1: [(1, 7, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 7, WORST_REWARD,True),
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
        1: [(1, 8, WORST_REWARD, True)
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
            (0.5, 9, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 9, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 9, MED_REWARD,True),
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
            (0.5, 13, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 13, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 13, MED_REWARD,True),
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
        1: [(1, 16, WORST_REWARD, True)
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
            (0.5, 18, WORST_REWARD,True),

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
    card = np.where(state == 1)
    state = (4-card)*4
    return state

def convert_flop_state_to_num(preflop_state,state):

    pass
def play_an_episode(env: Env, agent: Agent, opponent:Agent):
    state, *_ = env.reset()
    state = convert_pre_flop_state_to_num(state)
    preflop_state = state
    done = False
    mana = env.mana
    total_reward = 0
    while not done:
        #rememmber that state, reward an done are referring only in the agent
        if mana == 0: #if our agent is the mana
            state, reward, done=env.step(agent.send_action(state), 0)
            total_reward += reward
            if done: break
            state, reward, done = env.step(opponent.send_action(state), 1)
            total_reward += reward 
            
        else:
           state, reward, done=env.step(opponent.send_action(state), 1)
           total_reward += reward 
           if done: break
           state, reward, done=env.step(agent.send_action(state), 0)
           total_reward += reward


if __name__ == "__main__":
    
    
    agent = PolicyIterationAgent(P=P)
    opponent = Random_Agent()
    env = Env(agent, opponent, number_of_cards=5)
    for i in range(15):
        r = play_an_episode(env,agent,opponent)
        print(f"in episode {i}, reward is {r}")

    