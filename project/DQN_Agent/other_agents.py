br = 0 #stands for bad reward
mr = 1 #stands for medium reward
gr = 2 #stands for good reward

#dictionary that represents the value of each card
cards = {
    "2": br,
    "3": br,
    "4": br,
    "5": br,
    "6": br,
    "7": mr,
    "8": mr,
    "9": mr,
    "10": mr,
    "J": gr,
    "Q": gr,
    "K": gr,
    "A": gr
}

action_hierarchy = {"raise":1, "call":0, "check":3, "fold":2}

def get_action(legal_moves, desired_move):
    """
        method that gets the -enumerated- desired action and returns the
        closest one(based on poker criteria and the so-called'legal' moves).
        returns 2 (fold) in case of an error
    """
    
    action_hierarchy_list = list(action_hierarchy.keys())
    index = action_hierarchy_list.index(desired_move)
    if desired_move in legal_moves: return action_hierarchy[desired_move]
    for i in range(3):
        next_move = action_hierarchy_list[(index+i+1)%len(action_hierarchy_list)]
        if next_move in legal_moves: return action_hierarchy.get(next_move, 2)
    return 2

class Threshold_Agent():
    #aka the offensive/loose agent
    def __init__(self):
        self.use_raw = False #just for the env, False because the agent is not a human
        self.pair = False

    def step(self, state):
        retval=self.eval_step(state)[0]
        return retval
    

    def eval_step(self, state):
        """
            method that sends/decides the action of the layer.
        """

        hand = self.get_hand(state)
        table = self.get_table(state)
        if table == -1: return self.play_preflop(state, hand), None
        return self.play_post_flop(state, hand, table), None

    

    def get_hand(self, state):
        hand = state["raw_obs"]["hand"]
        hand=list([list(h)[-1]for h in hand]) #hand variable contains the rank of the cards
        return hand
    
    def get_table(self, state):
        table = state["raw_obs"].get("table", False) #return false in case of preflop
        if(not table): return -1
        table=list([list(h)[-1]for h in table]) #table variable contains the rank of the cards
        return table
    
    def play_preflop(self, state,hand):
        """ 
        check if agent has strong preflop hand.
        ARGUMENTS: the ranks of the hand, state as the environment returns
        """
        pair = hand[0] == hand[1]
        self.pair = pair
        value = cards.get(hand[0], 0) + cards.get(hand[1], 0)
        legal_actions = state["raw_legal_actions"]
        if pair or value >= 3:
            return get_action(legal_moves=legal_actions, desired_move = "raise")
        return get_action(legal_moves=legal_actions, desired_move = "call")

    def play_post_flop(self, state, hand, table):
        """ 
        Decides either to raise in case of a match on the table,
        or to just call
        Arguments: ranked (hand and table)
        """
        match = False
        legal_actions = state["raw_legal_actions"]

        for h in hand:
            match = True if h in table else match
        #in case of a match within the hand and the table or just a pair in hand
        if match or self.pair: return get_action(legal_moves = legal_actions, desired_move="raise")
        return get_action(legal_moves = legal_actions, desired_move="call")
    

class Tight_Threshold_Agent(Threshold_Agent):
    
    def play_preflop(self, state,hand):
        """ 
        check if agent has strong preflop hand.
        ARGUMENTS: the ranks of the hand, state as the environment returns
        """
        pair = hand[0] == hand[1]
        self.pair = pair
        value = cards.get(hand[0], 0) + cards.get(hand[1], 0)
        legal_actions = state["raw_legal_actions"]
        if pair or value >= 3:
            return get_action(legal_moves=legal_actions, desired_move = "call")
        return get_action(legal_moves=legal_actions, desired_move = "check")


    def play_preflop(self, state,hand):
        """ 
        Decides either to raise in case of a match on the table,
        or to just call
        Arguments: ranked (hand and table)
        """
        match = False
        legal_actions = state["raw_legal_actions"]

        for h in hand:
            match = True if h in table else match
        #in case of a match within the hand and the table or just a pair in hand
        if match or self.pair: return get_action(legal_moves = legal_actions, desired_move="call")
        return get_action(legal_moves = legal_actions, desired_move="check")

if __name__ == "__main__":
    state = {
        "raw_legal_actions": ['raise','call','check'] 
    }
    hand = ['5', 'A']
    table = ['K', '7', '8']
    a = Tight_Threshold_Agent()
    a.pair = hand[0] == hand[1]
    ret = a.eval_step(state = state, hand = hand, table = table)
    print(ret)
