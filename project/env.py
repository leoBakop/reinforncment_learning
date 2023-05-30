import game
from game import Game

class Env:

    def __init__(self,agent, opponent):
        self.game = Game()
        self.agent = agent
        self.agents_hand = []
        self.agents_chips = []
        self.chip_index = 0
        self.opponent = opponent
        self.opponents_hand = []
        self.table = 0
        #chips of the agent
        total_chips = int(game.INITIAL_TOKENS * 4 + 1)
        self.agents_chips = [0]*total_chips
        self.chip_index = int((total_chips - 1)/2)
        self.agents_chips[self.chip_index] = 1
        #back account in the begining of every hand
        self.bank = []

    def reset(self):

        
        self.mana = self.game.init_game()
        self.agents_hand = self.game.hand_of_player[0] #by assumption, trainable agent is player 0 (doesn't mean that plays first every time)
        self.opponents_hand = self.game.hand_of_player[1]
        self.bank = list([ i +.5 for i in self.game.total_money_per_player]) #blind/ante

        done = False
        self.table = 0
        state = self.form_state()
        return state, self.mana, done
    
    def step(self, action):
        """ Action is a (2*1), [action_player_0, action_player_1]
        """
        done = False
        p = self.mana
        for i in range(len(action)):
            done, a = self.game.step(action[p], p) #a is the action aw it was translated from the game 
            if (p == 0 and a == 2):#in case that the agent raises
                #then I should reduce the agents chips
                self.agents_chips[self.chip_index] = 0
                self.agents_chips[self.chip_index-2] = 1
                self.chip_index-=2
            
            if(i == 1 and action[p] == 2): #in case that the last player raise
                next_player = self.agent if p == 1 else self.opponent
                #to do
                new_action = next_player.send_action(self.form_state()) # a method that every agent should implememnt, taking a state, returning an action
                done, a = self.game.step(new_action, p)
            p = (p+1)%2
            if done:
                break
        self.table = self.game.table
        state = self.form_state()
        
        if done: #the episode is terminated so I have to send reward
            
            bank_after_episode = self.game.total_money_per_player
            reward = bank_after_episode[0] - self.bank[0] #reward is how much money did the agent win (or lose)
            #changing the chips of the agent based on the result of the game
            #in case of a negative reward this will work too
            index_shift =  int(reward * 2)
            self.agents_chips[self.chip_index] = 0
            self.agents_chips[self.chip_index + index_shift] = 1
            self.chip_index += index_shift

            return state, reward, done
        else:
            return state, 0, done

    def form_state(self):
        """
            state: [0:4 is agents hand, 5:9 cards on the table, 10:28 available chips of agent (every index is .5 chips)]
        """
        self.table == [0]*5 if isinstance(self.table, int) else self.table
        return [self.agents_hand, self.table, self.agents_chips]
        


if __name__ == "__main__":
    from agent import Agent
    agent = Agent()
    opponent = Agent()

    env = Env(agent, opponent)
    
    state = env.reset()
    done = False
    while not done:
        state, reward, done=env.step([agent.send_action(state), opponent.send_action(state)])

    print("end")