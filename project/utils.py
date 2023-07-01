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
BEST_REWARD = 4.5
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
        2: [(0.0, 2, BEST_REWARD,True),
            (1, 2, 0.0,False)
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
        2: [(0.5, 3, LOW_MED_REWARD,True), #only for A-** since A is the highest card
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
            (0.5, 11, MED_REWARD,True),

        ],
        #fold
        1: [(1, 11, LOW_MED_REWARD,True)
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


---------------  ---------------  ---------------  ---------------  -------------------


0: A-AA or A-A*                     9: Q pre flop raise                18:J-J*, flop, raise         27:T-T*, flop, check
1:A- ** , flop, raise               10: Q pre flop check , no info     19:J-J*, flop, check         28:T-TT, flop, raise
2: A- **, flop, check               11: Q-Q* ,flop, raise              20:J -JJ, flop, raise        29:T- TT, flop, check
3:K pre flop raise                  12: Q-Q*, flop, check -na kn raise 21:J - JJ, flop, check       30:T- ** , flop, raise
4: K pre flop, check or no info     13: Q-QQ , flop raise or check     22:J- ** , flop, raise	    31:T- **, flop, check
5: K-K* flop, raise or check        14: Q- ** , flop, raise            23:J- **, flop, check	    32:A- pre flop any opp action
6: K-KK flop, raise or check        15: Q- **, flop, check             24:T pre flop, raise
7:  K-** flop, raise                16: J pre flop, raise              25: T pre flop, check/no info
8 : K-** flop, check                17: J pre flop, check - no info    26:T -T*,  flop, raise


-----------actions----------
0: check
1: fold
2: raise
'''

P_THRESHOLD_D= {    
    # A-AA or A-A* whatever the opp does, raise
   0: { 
        #action - check
        0: [(0.5, 0, 0.0,False),
            (0.5, 0, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 0, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 0, BEST_REWARD,True),
            (0.5, 0, 0.0,False)

        ]
    },
   1: { #A- ** , flop, raise  if the opp raised, he has sth. Better fold.
        #action - check
        0: [(0.5, 1, 0.0,False),
            (0.5, 1, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 1, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 1, WORST_REWARD,True),
            (0.5, 1, 0.0,False)

        ]
    },

   2: { #A- **, flop, check : The opponent doesnt have sth good. Raise.
        #action - check
        0: [(0.5, 2, 0.0,False),
            (0.5, 2, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 2, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 2, BEST_REWARD,True),
            (0.5, 2, 0.0,False)

        ]
    },

# K- pre flop, raise : low best is to raise 
   3: {
        #action - check
        0: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25,8,0.0,False)
        ],
       #action -fold
        1: [(1, 3, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25, 8, 0.0,False)
        ]
    },
#K pre flop, check or no info
   4: {
        #action - check
        0: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25,8,0.0,False)
        ],
       #action -fold
        1: [(1, 3, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25, 8, 0.0,False)
        ]
    },
#5: K-K* flop, raise or check : we probably have the winning card, lets raise 
   5: { 
        #action - check
        0: [(0.5, 5, 0.0,False),
            (0.5, 5, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 5, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 5, LOW_MED_REWARD,True),
            (0.5, 5, 0.0,False)

        ]
    },
#6: K-KK flop, raise or check : we have the winning hand unless the opp has A-AA -- best to raise 
   6: { 
        #action - check
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
#7:  K-** flop, raise -- the other has sth, best is check otherwise fold
   7: { 
        #action - check
        0: [(0.5, 7, 0.0,False),
            (0.5, 7, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 7, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 7, WORST_REWARD,True),
            (0.5, 7, 0.0,False)

        ]
    },
#8 : K-** flop, check -- they dont have anything, ur probably bettterrrr-- low best is raise
    8: { 
        #action - check
        0: [(0.5, 8, 0.0,False),
            (0.5, 8, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 8, LOW_MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 8, LOW_BEST_REWARD,True),
            (0.5, 8, 0.0,False)

        ]
    },
#9: Q pre flop raise 
   9: {
        #action - check
        0: [(0.2, 11, 0.0,False),
            (0.2, 12, 0.0,False),
            (0.2, 13, 0.0,False),
            (0.2,14, 0.0,False),
            (0.2,15, 0.0,False)
        ],
       #action -fold
        1: [(1, 9,MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 11, 0.0,False),
            (0.2, 12, 0.0,False),
            (0.2, 13, 0.0,False),
            (0.2,14, 0.0,False),
            (0.2,15, 0.0,False)
        ]
    },
#10: Q pre flop check , no info
    10: {
    #action - check
    0: [(0.2, 11, 0.0,False),
        (0.2, 12, 0.0,False),
        (0.2, 13, 0.0,False),
        (0.2,14, 0.0,False),
        (0.2,15, 0.0,False)
    ],
    #action -fold
    1: [(1, 9, WORST_REWARD, True)
    ],
    #action -raise
    2: [(0.2, 11, 0.0,False),
        (0.2, 12, 0.0,False),
        (0.2, 13, 0.0,False),
        (0.2,14, 0.0,False),
        (0.2,15, 0.0,False)
    ]
},
#11: Q-Q* ,flop, raise 
   11: { 
        #action - check
        0: [(0.5, 11, 0.0,False),
            (0.5, 11, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 11, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 11, MED_REWARD,True),
            (0.5, 11, 0.0,False)

        ]
    },
#12: Q-Q*, flop, check  --the opp has nothing , raise!
    12: { 
        #action - check
        0: [(0.5, 12, 0.0,False),
            (0.5, 12, MED_REWARD,True),

        ],
        #fold
        1: [(1, 12, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 12, BEST_REWARD,True),
            (0.5, 12, 0.0,False)

        ]
    },  
#13: Q-QQ , flop raise or check  
   13: { 
        #action - check
        0: [(0.5, 13, 0.0,False),
            (0.5, 13, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 13, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 13, BEST_REWARD,True),
            (0.5, 13, 0.0,False)
        ]
    },
#14: Q- ** , flop, raise -- we have nothing , they have sth--fold
   14: { 
        #action - check
        0: [(0.5, 14, 0.0,False),
            (0.5, 14, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 14, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 14, WORST_REWARD,True),
            (0.5, 14, 0.0,False)

        ]
    },
#15: Q- **, flop, check --you both have nothing its goona be highest card winning
     15: { 
        #action - check
        0: [(0.5, 8, 0.0,False),
            (0.5, 8, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 8, LOW_BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 8, BEST_REWARD,True),
            (0.5, 8, 0.0,False)

        ]
    },
#16: J pre flop, raise--fold
   16: {
        #action - check
        0: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)
        ],
       #action -fold
        1: [(1, 16, LOW_MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)]
    },
#17: J pre flop, check - no info
   17: {
        #action - check
        0: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)
        ],
       #action -fold
        1: [(1, 17, LOW_MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)]
    },

#18:J-J*, flop, raise  --probs has sth better,still better not fold
   18: { 
        #action - check
        0: [(0.5, 18, 0.0,False),
            (0.5, 18, MED_REWARD,True),

        ],
        #fold
        1: [(1, 18, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 18, WORST_REWARD,True),
            (0.5, 18, 0.0,False)

        ]
    },
#19: J-J*, flop, check --they have nothing. I win probs, raise 
   19: { 
        #action - check
        0: [(0.5, 19, 0.0,False),
            (0.5, 19, MED_REWARD,True),

        ],
        #fold
        1: [(1, 19, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 19, BEST_REWARD,True),
            (0.5, 19, 0.0,False)

        ]
    },
#20:J -JJ, flop, raise 
   20: { 
        #action - check
        0: [(0.5, 20, 0.0,False),
            (0.5, 20, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 20, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 20, LOW_BEST_REWARD,True),
            (0.5, 20, 0.0,False)
        ]
    },
#21:J - JJ, flop, check -- i win. raise only.
   21: { 
        #action - check
        0: [(0.5, 21, 0.0,False),
            (0.5, 21, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 21, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 21, BEST_REWARD,True),
            (0.5, 21, 0.0,False)

        ]
    },

#22:J- ** , flop, raise --fold asap
   22: { 
        #action - check
        0: [(0.5, 22, 0.0,False),
            (0.5, 22, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 22, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 22, WORST_REWARD,True),
            (0.5, 22, 0.0,False)

        ]
    },

#23:J- **, flop, check
   23: { 
        #action - check
        0: [(0.5, 23, 0.0,False),
            (0.5, 23, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 23, LOW_BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 23, WORST_REWARD,True),
            (0.5, 23, 0.0,False)

        ]
    },
#24:T pre flop, raise
   24: {
        #action - check
        0: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)
        ],
       #action -fold
        1: [(1, 24, LOW_MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)]
    },
#25: T pre flop, check/no info
   25: {
        #action - check
        0: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)
        ],
       #action -fold
        1: [(1, 24, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)]
    },
#26: T -T*,  flop, raise
   26: { 
        #action - check
        0: [(0.5, 26, 0.0,False),
            (0.5, 26, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 26, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 26, LOW_MED_REWARD,True),
            (0.5, 26, 0.0,False)

        ]
    },
#27:T-T*, flop, check - im winning 
   27: { 
        #action - check
        0: [(0.5, 27, 0.0,False),
            (0.5, 27, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 27, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 27, BEST_REWARD,True),
            (0.5, 27, 0.0,False)

        ]
    },
#28:T-TT, flop, raise
   28: { 
        #action - check
        0: [(0.5, 28, 0.0,False),
            (0.5, 28, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 28, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 28, LOW_BEST_REWARD,True),
            (0.5, 28, 0.0,False)

        ]
    },
#29:T- TT, flop, check - i win for sure.
   29: { 
        #action - check
        0: [(0.5, 29, 0.0,False),
            (0.5, 29, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 29, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 29, BEST_REWARD,True),
            (0.5, 29, 0.0,False)

        ]
    },
#30 :T- ** , flop, raise --fold.
   30: { 
        #action - check
        0: [(0.5, 30, 0.0,False),
            (0.5, 30, MED_REWARD,True),

        ],
        #fold
        1: [(1, 30, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 30, WORST_REWARD,True),
            (0.5, 30, 0.0,False)

        ]
    },
#31: T- **, flop, check
   31: { 
        #action - check
        0: [(0.5, 31, 0.0,False),
            (0.5, 31, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 31, LOW_MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 31, WORST_REWARD,True),
            (0.5, 31, 0.0,False)

        ]
    },
#32: A- pre flop any opp action -- better raise 
   32: {
        #action - check
        0: [(0.33, 0, 0.0,False),
            (0.33, 1, 0.0,False),
            (0.33, 2, 0.0,False)
            
        ],
       #action -fold
        1: [(1, 32, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.33, 0, 0.0,False),
            (0.33, 1, 0.0,False),
            (0.33, 2, 0.0,False)
            
        ]
    },

}

P_THRESHOLD_A= {
   # A-AA or A-A* whatever the opp does, raise
    0: { 
        #action - check
        0: [(0.5, 0, 0.0,False),
            (0.5, 0, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 0, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 0, BEST_REWARD,True),
            (0.5, 0, 0.0,False)

        ]
    },
    1: { #A- ** , flop, raise  if the opp raised, he has sth. Better fold.
        #action - check
        0: [(0.5, 1, 0.0,False),
            (0.5, 1, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 1, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 1, WORST_REWARD,True),
            (0.5, 1, 0.0,False)

        ]
    },

   2: { #A- **, flop, check : The opponent doesnt have sth good. Raise.
        #action - check
        0: [(0.5, 2, 0.0,False),
            (0.5, 2, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 2, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 2, BEST_REWARD,True),
            (0.5, 2, 0.0,False)

        ]
    },

# K- pre flop, raise : low best is to raise 
   3: {
        #action - check
        0: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25,8,0.0,False)
        ],
       #action -fold
        1: [(1, 3, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25, 8, 0.0,False)
        ]
    },
#K pre flop, check or no info
   4: {
        #action - check
        0: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25,8,0.0,False)
        ],
       #action -fold
        1: [(1, 3, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.25, 5, 0.0,False),
            (0.25, 6, 0.0,False),
            (0.25, 7, 0.0,False),
            (0.25, 8, 0.0,False)
        ]
    },
#5: K-K* flop, raise or check : we probably have the winning card, lets raise 
   5: { 
        #action - check
        0: [(0.5, 5, 0.0,False),
            (0.5, 5, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 5, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 5, BEST_REWARD,True),
            (0.5, 5, 0.0,False)

        ]
    },
#6: K-KK flop, raise or check : we have the winning hand unless the opp has A-AA -- best to raise 
   6: { 
        #action - check
        0: [(0.5, 6, 0.0,False),
            (0.5, 6, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 6, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 6, BEST_REWARD,True),
            (0.5, 6, 0.0,False)

        ]
    },
#7:  K-** flop, raise -- the other has sth, best is check otherwise fold
   7: { 
        #action - check
        0: [(0.5, 7, 0.0,False),
            (0.5, 7, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 7, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 7, WORST_REWARD,True),
            (0.5, 7, 0.0,False)

        ]
    },
#8 : K-** flop, check -- they dont have anything, ur probably bettterrrr-- low best is raise
    8: { 
        #action - check
        0: [(0.5, 8, 0.0,False),
            (0.5, 8, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 8, LOW_MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 8, LOW_BEST_REWARD,True),
            (0.5, 8, 0.0,False)

        ]
    },
#9: Q pre flop raise 
   9: {
        #action - check
        0: [(0.2, 11, 0.0,False),
            (0.2, 12, 0.0,False),
            (0.2, 13, 0.0,False),
            (0.2,14, 0.0,False),
            (0.2,15, 0.0,False)
        ],
       #action -fold
        1: [(1, 9,MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 11, 0.0,False),
            (0.2, 12, 0.0,False),
            (0.2, 13, 0.0,False),
            (0.2,14, 0.0,False),
            (0.2,15, 0.0,False)
        ]
    },
#10: Q pre flop check , no info
    10: {
    #action - check
    0: [(0.2, 11, 0.0,False),
        (0.2, 12, 0.0,False),
        (0.2, 13, 0.0,False),
        (0.2,14, 0.0,False),
        (0.2,15, 0.0,False)
    ],
    #action -fold
    1: [(1, 9, WORST_REWARD, True)
    ],
    #action -raise
    2: [(0.2, 11, 0.0,False),
        (0.2, 12, 0.0,False),
        (0.2, 13, 0.0,False),
        (0.2,14, 0.0,False),
        (0.2,15, 0.0,False)
    ]
},
#11: Q-Q* ,flop, raise 
   11: { 
        #action - check
        0: [(0.5, 11, 0.0,False),
            (0.5, 11, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 11, MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 11, BEST_REWARD,True),
            (0.5, 11, 0.0,False)

        ]
    },
#12: Q-Q*, flop, check  --
    12: { 
        #action - check
        0: [(0.5, 12, 0.0,False),
            (0.5, 12, MED_REWARD,True),

        ],
        #fold
        1: [(1, 12, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 12, BEST_REWARD,True),
            (0.5, 12, 0.0,False)

        ]
    },  
#13: Q-QQ , flop raise or check  
   13: { 
        #action - check
        0: [(0.5, 13, 0.0,False),
            (0.5, 13, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 13, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 13, BEST_REWARD,True),
            (0.5, 13, 0.0,False)
        ]
    },
#14: Q- ** , flop, raise -- we have nothing , they have sth--fold
   14: { 
        #action - check
        0: [(0.5, 14, 0.0,False),
            (0.5, 14, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 14, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 14, WORST_REWARD,True),
            (0.5, 14, 0.0,False)

        ]
    },
#15: Q- **, flop, check --you both have nothing its goona be highest card winning
     15: { 
        #action - check
        0: [(0.5, 8, 0.0,False),
            (0.5, 8, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 8, LOW_BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 8, BEST_REWARD,True),
            (0.5, 8, 0.0,False)

        ]
    },
#16: J pre flop, raise--fold
   16: {
        #action - check
        0: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)
        ],
       #action -fold
        1: [(1, 16, LOW_MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)]
    },
#17: J pre flop, check - no info
   17: {
        #action - check
        0: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)
        ],
       #action -fold
        1: [(1, 17, LOW_MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 18, 0.0,False),
            (0.2, 19, 0.0,False),
            (0.2, 20, 0.0,False),
            (0.2,21, 0.0,False),
            (0.2,22, 0.0,False),
            (0.2,23, 0.0,False)]
    },

#18:J-J*, flop, raise  --probs has sth better,still better not fold
   18: { 
        #action - check
        0: [(0.5, 18, 0.0,False),
            (0.5, 18, MED_REWARD,True),

        ],
        #fold
        1: [(1, 18, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 18, WORST_REWARD,True),
            (0.5, 18, 0.0,False)

        ]
    },
#19: J-J*, flop, check --they have nothing. I win probs, raise 
   19: { 
        #action - check
        0: [(0.5, 19, 0.0,False),
            (0.5, 19, MED_REWARD,True),

        ],
        #fold
        1: [(1, 19, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 19, BEST_REWARD,True),
            (0.5, 19, 0.0,False)

        ]
    },
#20:J -JJ, flop, raise 
   20: { 
        #action - check
        0: [(0.5, 20, 0.0,False),
            (0.5, 20, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 20, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 20, LOW_BEST_REWARD,True),
            (0.5, 20, 0.0,False)
        ]
    },
#21:J - JJ, flop, check -- i win. raise only.
   21: { 
        #action - check
        0: [(0.5, 21, 0.0,False),
            (0.5, 21, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 21, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 21, BEST_REWARD,True),
            (0.5, 21, 0.0,False)

        ]
    },

#22:J- ** , flop, raise --fold asap
   22: { 
        #action - check
        0: [(0.5, 22, 0.0,False),
            (0.5, 22, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 22, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 22, WORST_REWARD,True),
            (0.5, 22, 0.0,False)

        ]
    },

#23:J- **, flop, check
   23: { 
        #action - check
        0: [(0.5, 23, 0.0,False),
            (0.5, 23, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 23, LOW_BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 23, WORST_REWARD,True),
            (0.5, 23, 0.0,False)

        ]
    },
#24:T pre flop, raise
   24: {
        #action - check
        0: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)
        ],
       #action -fold
        1: [(1, 24, LOW_MED_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)]
    },
#25: T pre flop, check/no info
   25: {
        #action - check
        0: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)
        ],
       #action -fold
        1: [(1, 24, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.2, 26, 0.0,False),
            (0.2, 27, 0.0,False),
            (0.2, 28, 0.0,False),
            (0.2,29, 0.0,False),
            (0.2,30, 0.0,False),
            (0.2,31, 0.0,False)]
    },
#26: T -T*,  flop, raise
   26: { 
        #action - check
        0: [(0.5, 26, 0.0,False),
            (0.5, 26, LOW_BEST_REWARD,True),

        ],
        #fold
        1: [(1, 26, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 26, LOW_MED_REWARD,True),
            (0.5, 26, 0.0,False)

        ]
    },
#27:T-T*, flop, check - im winning 
   27: { 
        #action - check
        0: [(0.5, 27, 0.0,False),
            (0.5, 27, LOW_MED_REWARD,True),

        ],
        #fold
        1: [(1, 27, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 27, BEST_REWARD,True),
            (0.5, 27, 0.0,False)

        ]
    },
#28:T-TT, flop, raise
   28: { 
        #action - check
        0: [(0.5, 28, 0.0,False),
            (0.5, 28, MED_REWARD,True),

        ],
        #fold
        1: [(1, 28, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 28, BEST_REWARD,True),
            (0.5, 28, 0.0,False)

        ]
    },
#29:T- TT, flop, check - i win for sure.
   29: { 
        #action - check
        0: [(0.5, 29, 0.0,False),
            (0.5, 29, WORST_REWARD,True),

        ],
        #fold
        1: [(1, 29, WORST_REWARD,True)
        ],
        #raise
        2: [(0.5, 29, BEST_REWARD,True),
            (0.5, 29, 0.0,False)

        ]
    },
#30 :T- ** , flop, raise --fold.
   30: { 
        #action - check
        0: [(0.5, 30, 0.0,False),
            (0.5, 30, MED_REWARD,True),

        ],
        #fold
        1: [(1, 30, BEST_REWARD,True)
        ],
        #raise
        2: [(0.5, 30, WORST_REWARD,True),
            (0.5, 30, 0.0,False)

        ]
    },
#31: T- **, flop, check
   31: { 
        #action - check
        0: [(0.5, 31, 0.0,False),
            (0.5, 31, BEST_REWARD,True),

        ],
        #fold
        1: [(1, 31, LOW_MED_REWARD,True)
        ],
        #raise
        2: [(0.5, 31, WORST_REWARD,True),
            (0.5, 31, 0.0,False)

        ]
    },
#32: A- pre flop any opp action -- better raise 
   32: {
        #action - check
        0: [(0.33, 0, 0.0,False),
            (0.33, 1, 0.0,False),
            (0.33, 2, 0.0,False)
            
        ],
       #action -fold
        1: [(1, 32, WORST_REWARD, True)
        ],
       #action -raise
        2: [(0.33, 0, 0.0,False),
            (0.33, 1, 0.0,False),
            (0.33, 2, 0.0,False)
            
        ]
    },

}







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
    
def threshold_convert_state_to_num(state):
   # Map the card in hand to an index
   hand_card = np.where(np.array(state[0:5]) == 1)[0]

   # Map the card on the table to an index
   table_card = np.where(np.array(state[5:10]) == 1)[0]

   # Map the opponent's last action to an index
   opponent_action =  state[10:]
   if same_list(opponent_action, [0,0,0]): opponent_action[:]= [1,0,0] #same as check
##################pre flop phase############################
   if(len(table_card)==0): #we re at the pre-flop states
     #24:T pre flop, raise
     #25: T pre flop, check/no info
        if (state[0] == 1):
            if same_list(opponent_action, [0,0,0]) : return 25 #no info
            if same_list(opponent_action, [1,0,0]) : return 25 #check
            if same_list(opponent_action, [0,0,1]) : return 24 #raise

        #16: J pre flop, raise
        #17: J pre flop, check - no info
        if(state[1] ==1):
            if same_list(opponent_action, [0,0,0]) : return 17 #no info
            if same_list(opponent_action, [1,0,0]) : return 17 #check
            if same_list(opponent_action, [0,0,1]) : return 16 #raise  
        #9: Q pre flop raise
        #10: Q pre flop check , no info
        if(state[2]==1):
            if same_list(opponent_action, [0,0,0]) : return 10 #no info
            if same_list(opponent_action, [1,0,0]) : return 10 #check
            if same_list(opponent_action, [0,0,1]) : return 9 #raise
       #3:K pre flop raise                 
       #4: K pre flop, check or no info 
        if(state[3]==1):
            if same_list(opponent_action, [0,0,0]) : return 4 #no info
            if same_list(opponent_action, [1,0,0]) : return 4 #check
            if same_list(opponent_action, [0,0,1]) : return 3 #raise 
        #32:A- pre flop any opp action
        if(state[4]==1): return 32
        
###########################flop phase###############################
   if(len(table_card)==2): #we have 2 different (between them) cards on the table
        #30:T- ** , flop, raise
        #31:T- **, flop, check
        if (state[0] == 1 and state[5] == 0):
            if same_list(opponent_action,[1,0,0]) : return 31 #check
            if same_list(opponent_action,[0,0,1]) : return 30 #raise
            
        #26:T -T*,  flop, raise
        #27:T-T*, flop, check
        if(state[0] == 1 and state[5] == 1):
            if same_list(opponent_action,[1,0,0]) : return 27 #check
            if same_list(opponent_action,[0,0,1]) : return 26 #raise

        #22:J- ** , flop, raise	
        #23:J- **, flop, check
        if(state[1] ==1 and state[6]==0):
            if same_list(opponent_action,[1,0,0]) : return 23 #check
            if same_list(opponent_action,[0,0,1]) : return 22 #raise
        #18:J-J*, flop, raise 
        #19:J-J*, flop, check 
        if (state[1] ==1 and state[6]==1): 
            if same_list(opponent_action,[1,0,0]) : return 19 #check
            if same_list(opponent_action,[0,0,1]) : return 18 #raise            
        #14: Q- ** , flop, raise
        #15: Q- **, flop, check
        if(state[2]==1 and state[7]==0):
            if same_list(opponent_action,[1,0,0]) : return 15 #check
            if same_list(opponent_action,[0,0,1]) : return 14 #raise
            return 15
        #11: Q-Q* ,flop, raise 
        #12: Q-Q*, flop, check
        if(state[2]==1 and state[7]==1):
            if same_list(opponent_action,[1,0,0]) : return 12 #check
            if same_list(opponent_action,[0,0,1]) : return 11 #raise
        #7:  K-** flop, raise 
        #8 : K-** flop, check
        if(state[3]==1 and state[8]==0):
            if same_list(opponent_action,[1,0,0]) : return 8 #check
            if same_list(opponent_action,[0,0,1]) : return 7 #raise
        #5: K-K* flop, raise or check   
        if(state[3]==1 and state[8]==1):
            return 5
        #1:A- ** , flop, raise   
        #2: A- **, flop, check
        if(state[4]==1 and state[9]==0): 
            if same_list(opponent_action,[1,0,0]) : return 2 #check
            if same_list(opponent_action,[0,0,1]) : return 1 #raise 
        #0: A-AA or A-A*
        if(state[4]==1 and state[9]==1):
            return 0
        
#######################pair on the table already ##########################
   if(len(table_card)==1): #we have a pair already on the table
      #28:T-TT, flop, raise
      #29:T- TT, flop, check 
        if(state[0] == 1 and state[5] == 1):
            if same_list(opponent_action,[1,0,0]) : return 29 #check
            if same_list(opponent_action,[0,0,1]) : return 28 #raise 
        if (state[0] == 1 and state[5] ==0): #there is a pair in the table but we cannot make a 3 of a kind
            #act same as T-**
            if same_list(opponent_action,[1,0,0]) : return 31 
            if same_list(opponent_action,[0,0,1]) : return 30 
        #20:J -JJ, flop, raise
        #21:J - JJ, flop, check 
        if (state[1] ==1 and state[6]==1): 
            if same_list(opponent_action,[1,0,0]) : return 21 #check
            if same_list(opponent_action,[0,0,1]) : return 20 #raise
        if (state[1] == 1 and state[6] ==0): #there is a pair in the table but we cannot make a 3 of a kind
            #act same as J-**
            if same_list(opponent_action,[1,0,0]) : return 23 
            if same_list(opponent_action,[0,0,1]) : return 22   
        #13: Q-QQ , flop raise or check
        if(state[2]==1 and state[7]==1):return 13
        if(state[2]==1 and state[7] ==0):
            #act same as Q-**
            if same_list(opponent_action,[1,0,0]) : return 12
            if same_list(opponent_action,[0,0,1]) : return 11   
        #6: K-KK flop, raise or check 
        if(state[3]==1 and state[8]==1):return 6
        if (state[3] == 1 and state[8] ==0): #there is a pair in the table but we cannot make a 3 of a kind
            #act same as K-**
            if same_list(opponent_action,[1,0,0]) : return 8 
            if same_list(opponent_action,[0,0,1]) : return 7         
        #0: A-AA or A-A*      
        if(state[4]==1 and state[9]==1):return 0
        if (state[4] == 1 and state[9] ==0): #there is a pair in the table but we cannot make a 3 of a kind
            #act same as A-**
            if same_list(opponent_action,[1,0,0]) : return 2
            if same_list(opponent_action,[0,0,1]) : return 1      
        
   return state


def same_list(first, second):
    if len(first)!=len(second):
        return False
    same = True
    for i, j in zip(first, second):
        same = i==j
        if  not same:
            return False
    return True

def return_state(state_vector, threshold, agent, preflop_state, full_stages=False):
    state = 0
    
    if (agent.to_str() == "Q_Learning_Agent" and full_stages):
        state_vector = state_vector[0:10] #if not threshold else state_vector
        state_vector = state_vector[::-1]
        return int("".join(map(str, state_vector)), 2)

    if not threshold: 
        state_vector = state_vector[0:10]
        return convert_flop_state_to_num(preflop_state, state_vector)
    else :
        return threshold_convert_state_to_num(state_vector)

    return state




def differ(first, second):
    count = 0
    for i, j in zip(first, second):
        if (i != j): count+=1

    return count