import game
from game import Game
import numpy as np
from agent import Agent
from implemented_agents import Q_Learning_Agent, PolicyIterationAgent, Random_Agent
from card import Card
import utils
class Env:

    def __init__(self,agent, opponent, seed = 0, number_of_cards=5):
        self.game = Game(seed = seed)
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

    def reset(self, disp = False):

        
        self.mana = self.game.init_game()
        if self.mana == -1 : return [-1] * 3
       
        agents_hand = self.game.hand_of_player[0] #by assumption, trainable agent is player 0 (doesn't mean that plays first every time)
        self.agents_hand = ["T", "J", "Q", "K", "A"] #important line in order to create the correct state vector
        self.agents_hand = list([1 if agents_hand.rank == i else 0 for i in self.agents_hand])
        self.opponents_hand = self.game.hand_of_player[1]
        #setting up the ante
        self.bank = list([ i+self.game.small_blind for i in self.game.total_money_per_player]) #blind/ante, +self.game.small_blind is right
        if self.game.small_blind !=0:
            if self.chip_index == 0:return [-1] * 3
            self.agents_chips[self.chip_index] = 0
            self.agents_chips[self.chip_index-1] = 1
            self.chip_index-=1
        
        self.last_opponent_move = None
        done = False
        self.table = [Card('S', '-1')]*2
        state = self.form_state()
        self.disp = disp
        return state, self.mana, done
    
    def calulate_chips(self):
        bank = self.game.total_money_per_player
        self.chip_index = int(bank[0]*2)
        self.agents_chips=[0]*self.total_chips
        self.agents_chips[self.chip_index] = 1
        return

    def step(self, action, player, t, previous_tuple, threshold, agent:Agent):
        """ Action is a int, and player is either 0(agent) or 1(opponent)
        """
        s = self.get_enumerate_states(self.form_state(), threshold, agent)
        

        self.calulate_chips()
        if (not 1 in self.agents_chips): #if agent has no chips available
            return self.form_state(), 0, True  #0 is a magic number, for a bad reward
        done = False
        p = self.mana
        
        done, a = self.game.step(action, player) #a is the action aw it was translated from the game
        #if opponent is playing, then store its move 
        self.last_opponent_move = a if player == 1 else  self.last_opponent_move
        
            
        if (player == 0 and a == 2):#in case that the agent raises
            #then I should reduce the agents chips
            self.agents_chips[self.chip_index] = 0
            if self.chip_index-2 >= 0:
                self.calulate_chips()
            else:
                return self.form_state(), 0, True  #0 is a magic number, for a bad reward
        if(player != p and action == 2): #in case that the last player raise (first player is always the mana)
            next_player = self.agent if player == 1 else self.opponent
            #to do
            if isinstance(self.opponent, Random_Agent): enumarated_state=utils.convert_flop_state_to_num(utils.convert_pre_flop_state_to_num(self.form_state()[:5]),self.form_state())
            else: enumarated_state=utils.threshold_convert_state_to_num(self.form_state())
            if (isinstance(next_player, Q_Learning_Agent)): #if our agent is playing
                new_action = next_player.send_action(enumarated_state, t)
                #next_player.train(previous_tuple)
            else:
                if(self.disp):print("Re-raise")
                new_action = next_player.send_action(enumarated_state, None, None) # a method that every agent should implememnt, taking a state, returning an action
            #done, a= self.game.step(new_action, np.abs(player-1))
            return self.step(new_action, np.abs(player - 1), t, previous_tuple, threshold=threshold, agent=agent)
        
        self.table = self.game.table
        state = self.form_state()
        
        if done: #the episode is terminated so I have to send reward
            
            bank_after_episode = self.game.total_money_per_player
            reward = bank_after_episode[0] - self.bank[0] #reward is how much money did the agent win (or lose)
            self.bank = bank_after_episode
            #changing the chips of the agent based on the result of the game
            self.calulate_chips()
            
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
        self.table = self.game.table
        for card in self.table:
            tmp = list([1 if card.rank == i else 0 for i in table])
            table_state = np.bitwise_or(np.array(table_state), np.array(tmp))
        last_op_move = [0]*3
        if self.game.opponent_last_action is not None:
            last_op_move[self.game.opponent_last_action] = 1
        
        return self.agents_hand + table_state.tolist() + last_op_move
        #return [self.agents_hand, self.table, self.agents_chips,last_op_move] in case of Q-learning

    """method only for debugging reasons"""
    def get_enumerate_states(self, state, threshold, agent):
        state = self.form_state()

        s = utils.return_state( state_vector=state, 
                                threshold=threshold,
                                agent=agent,
                                preflop_state = utils.convert_pre_flop_state_to_num(state[0:5])
                                ) 
        return s

if __name__ == "__main__":

    agent = PolicyIterationAgent(utils.P_THRESHOLD)
    opponent = Random_Agent()
    env = Env(agent, opponent, number_of_cards=5)
    for i in range(2):
        reward = 0
        env.reset()
        state, reward, done = env.step(2,0,0, [])
        state = utils.threshold_convert_state_to_num(state)
        state, reward, done = env.step(0,1,1, [])
        state = utils.threshold_convert_state_to_num(state)
        state, reward, done =  env.step(2,0,0, [])
        state = utils.threshold_convert_state_to_num(state)
        state, reward, done = ret = env.step(0,1,1, [])
    print("last line`")