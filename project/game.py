import numpy as np
from dealer import Dealer
from judger import Judger
from card import Card

INITIAL_TOKENS = 4.5
ACTIONS = {
    0:"check",
    1:"fold",
    2:"raise"
}

class Game:


    def __init__(self, num_players=2, num_of_cards_in_hand=1, num_of_cards_on_table=2, num_of_rounds=2, done = False):
        #logistics
        self.small_blind = 0.5
        self.num_players = num_players
        self.big_blind = self.small_blind
        self.num_of_cards_in_hand = num_of_cards_in_hand
        self.num_of_cards_on_table = num_of_cards_on_table
        self.num_of_rounds = num_of_rounds
        self.done = False

        #info for every player
        self.total_money_per_player = [INITIAL_TOKENS] * num_players
        self.hand_of_player = [0]*num_players
        self.active_players = [True] * num_players

        #game state
        self.current_round = 0 #how many steps have been played
        self.opponent_last_action = None #the opponents last action before this betting round
        self.current_phase = 0 #if we are in flop, or pre flop etc
        self.consecutive_raises = 0
        self.terminate_phase = 2 #show how many betting we will have in just one phase (ex. the flop etc)
        self.table = []

    def init_game(self):
        """
            Method that is called in the beginning of every game/hand of the "tournament"
        """
        if self.done or self.check_if_game_end: return False#if at least one player is bancrupt the the tournament is over
        self.dealer = Dealer()
        mana = np.random.choice([0,1])
        #hand in the cards
        for i in range(self.num_players):
            self.hand_of_player[i] = self.dealer.deal_card()
        self.total_money_per_player = list([i-self.small_blind for i in self.total_money_per_player]) #every player bets initially 0.5 tokens
        self.pot = 2*self.small_blind
        

        return mana

    def step(self,action,player):
        """
            it is called every time that a player talks,
            returns True if the hand is over , else False
        """
        self.current_round+=1
        opponent = np.abs(player - 1)
        #the only available option is "fold".You dont have the money to continue the game.
        action = 1 if self.total_money_per_player[player] <1 and self.opponent_last_action ==2 else action 
        #if 
        action = 0 if self.consecutive_raises == 2 and action == 2 else action 
                
        action = 1 if action == 2 and self.opponent_last_action == 2 else action
        if action ==  1: #player folds
                #the opponent wins
            return self.win(player,opponent), action
        if action == 2: #player raises 
            self.consecutive_raises +=1
            if self.opponent_last_action == 2: #if opponent was raise 
                self.terminate_phase = 3 #the two players must talk exactly 3 times
                if self.total_money_per_player[player] >= 2: #if I have the money I should bet 2 tokens
                    self.pot +=2
                    self.total_money_per_player[player]-=2
                    
                else: #else I lose
                    return self.win(player,opponent), action
        
            elif self.total_money_per_player[player] >= 1:# if opponent didn't raise and i have enough money
                self.pot +=1
                self.total_money_per_player[player]-=1
            else:# else you just lose
                #so the opponent winds
                return self.win(player,opponent), action
        if action == 0:
            if self.opponent_last_action ==2 : #if opponents raised in the last round
                if self.total_money_per_player[player] >= 1:#(and has the money to do it)
                    self.pot +=1
                    self.total_money_per_player[player]-=1
                else:#you dont have money to call
                    return self.win(player,opponent), action
                
        if self.terminate_phase == self.current_round: #phase must terminate
            if self.current_phase == 2: #at the end of the flop
                judger = Judger(2)
                r = judger.split_pot(self.pot,self.table, self.hand_of_player) 

                for i, reward in enumerate(r):
                    self.total_money_per_player[i]+=reward
                return True, action
            else:
                self.current_phase +=1
                self.consecutive_raises = 0
                self.opponent_last_action = 0
                self.current_round = 0
                self.terminate_phase = 2
                self.table = [self.dealer.deal_card(),self.dealer.deal_card()]
                
                return False , action
        return False, action


    def win(self,player, opponent):
        """ split the pot,
            terminates the hand
        """

        self.total_money_per_player[opponent]+=self.pot
        self.pot = 0
        self.done =  self.total_money_per_player[player] <= 0
        self.opponent_last_action = None
        return True
    

    def check_if_game_end(self):
        return  np.min(self.total_money_per_player) <.5

if __name__ == "__main__":
    g= Game()
    mana = g.init_game()
    player_hand = g.hand_of_player
    done = False
    i=mana
    while not done:
        done = g.step(action = 0, player = (i)%2)
        i+=1
        table = g.table

    print("end")