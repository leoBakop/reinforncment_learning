from agent import Agent

class Threshold_Agent(Agent):
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
    def send_action(self, state):
        if state ==  0 or state == 1 or state == 2 or state == 4 or state == 5 or state == 6 or state == 9 or state == 10\
            or state == 13 or state == 14 or state == 17 or state == 18 :
            return 2 
        if state == 3 or state == 7 or state == 8 or state == 12 or state == 16: return 0
        return 1 