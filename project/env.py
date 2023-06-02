import game
from game import Game
import numpy as np
from agent import Agent
from card import Card


class Env:

    def __init__(self,agent, opponent, number_of_cards=5):
        self.game = Game()
        self.agent = agent
        self.agents_hand = [0]*number_of_cards
        self.agents_chips = []
        self.chip_index = 0
        self.opponent = opponent
        self.opponents_hand = []
        self.table = [Card('S', '-1')]*2
        #chips of the agent
        total_chips = int(game.INITIAL_TOKENS * 4 + 1)
        self.agents_chips = [0]*total_chips
        self.total_chips = total_chips
        self.chip_index = int((total_chips - 1)/2)
        self.agents_chips[self.chip_index] = 1
        #back account in the begining of every hand
        self.bank = []
        #storing the opponents move in order to return it on state
        self.last_opponent_move = -1

    def reset(self):

        
        self.mana = self.game.init_game()
        if self.mana == -1 : return [-1] * 3
        agents_hand = self.game.hand_of_player[0] #by assumption, trainable agent is player 0 (doesn't mean that plays first every time)
        self.agents_hand = ["T", "J", "Q", "K", "A"] #important line in order to create the correct state vector
        self.agents_hand = list([1 if agents_hand.rank == i else 0 for i in self.agents_hand])
        self.opponents_hand = self.game.hand_of_player[1]
        self.bank = list([ i +.5 for i in self.game.total_money_per_player]) #blind/ante
        self.last_opponent_move = None
        done = False
        self.table = [Card('S', '-1')]*2
        state = self.form_state()
        return state, self.mana, done
    
    def step(self, action, player: Agent,enumarated_state):
        """ Action is a int, and player is either 0(agent) or 1(opponent)
        """
        done = False
        p = self.mana
        
        done, a = self.game.step(action, player) #a is the action aw it was translated from the game
        #if opponent is playing, then store its move 
        self.last_opponent_move = a if player == 1 else  self.last_opponent_move
            
            
        if (player == 0 and a == 2):#in case that the agent raises
            #then I should reduce the agents chips
            self.agents_chips[self.chip_index] = 0
            self.agents_chips[self.chip_index-2] = 1
            self.chip_index-=2
        
        if(player != p and action == 2): #in case that the last player raise (first player is always the mana)
            next_player = self.agent if player == 1 else self.opponent
            #to do
            new_action = next_player.send_action(enumarated_state) # a method that every agent should implememnt, taking a state, returning an action
            done, a= self.game.step(new_action, np.abs(player-1))
        
        
        self.table = self.game.table
        
        
        state = self.form_state()
        
        if done: #the episode is terminated so I have to send reward
            
            bank_after_episode = self.game.total_money_per_player
            reward = bank_after_episode[0] - self.bank[0] #reward is how much money did the agent win (or lose)
            deb = bank_after_episode[1] - self.bank[1]
            #changing the chips of the agent based on the result of the game
            #in case of a negative reward this will work too
            index_shift =  int(reward * 2)
            
            self.agents_chips[self.chip_index] = 0
            if self.chip_index + index_shift  < self.total_chips -1 and self.chip_index + index_shift  > -1:
                self.agents_chips[self.chip_index + index_shift] = 1
                self.chip_index += index_shift
            
            return state, reward, done
        else:
            return state, 0, done


    def form_state(self):
        """
            state: [0:4 is agents hand (T to A), 5:9 cards on the table (T to A), 
            10:28 available chips of agent (every index is .5 chips), 
            29:31 is the laste move of the opponent]
        """
        table_state = [0]*5
        table = ["T", "J", "Q", "K", "A"] #important line in order to create the correct state vector
        for card in self.table:
            tmp = list([1 if card.rank == i else 0 for i in table])
            table_state = np.bitwise_or(np.array(table_state), np.array(tmp))
        last_op_move = list([0 if i != self.last_opponent_move else 1 for i in range(3)])
        
        return self.agents_hand + table_state.tolist() + last_op_move
        #return [self.agents_hand, self.table, self.agents_chips,last_op_move] in case of Q-learning
        
