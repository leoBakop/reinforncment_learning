from game import Game
class Env:

    def __init__(self,agent, opponent):
        self.game = Game()
        self.agent = agent
        self.agents_hand = []
        self.opponent = opponent
        self.opponents_hand = []
        self.table = []
    def reset(self):
        self.mana = self.game.init_game()
        self.agent_hands = self.game.hand_of_player[0] #by assumption, trainable agent is player 0 (doesn't mean that plays first every time)
        self.opponents_hand = self.game.hand_of_player[1]
        done = False
        self.table = []
        state = self.form_state(self.agent_hands, self.table)
        return state, self.mana, done
    
    def step(self, action):
        """ Action is a (2*1), [action_player_0, action_player_1]
        """
        done = False
        p = self.mana
        for i in range(len(action)):
            done = self.game.step(action[p], p)
            p = p%2
            if(i == 1 and action[p] == 2): #in case that the last player raise
                next_player = self.agent if p == 1 else self.opponent
                #to do
                next_player.send_action() # a method that every agent should implememnt, taking a state, returning an action

    def form_state(self, hand, table):
        table == [0]*5 if len(table) == 0 else table
        return [hand, table]
        