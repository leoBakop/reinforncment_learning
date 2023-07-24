
import numpy as np


class Judger:
    def __init__(self, num_of_active_players):
        self.rewards = {
            "T" : 10,
            "J" : 11,
            "Q" : 12,
            "K" : 13,
            "A" : 14
        }
        self.reward_per_player = [0]*num_of_active_players


    def compare_hands(self, hands, table):
        """
        hands: An array containing the cards that player i has been hanted
        table: An array containing the cards on the table
        """
        for player, hand in enumerate(hands): #for every hand
            
            counts = table.count(hand)
            counts = counts * 100 if counts == 2 else (counts * 10 if counts == 1 else counts)
            self.reward_per_player[player] += (counts+1)* self.rewards.get(hand.rank)

            

        return  -1 if self.reward_per_player[0] == self.reward_per_player[1] else np.argmax(self.reward_per_player)

    def split_pot(self, pot, hands, table):
        
        winner = self.compare_hands(hands, table)
        if winner == -1:
            return [pot/2, pot/2]
        elif winner == 0:
            return [pot, 0]
        else:
            return [0, pot]



if __name__ == "__main__":
    from card import Card
    j = Judger(2)
    hands= [Card( 'S', 'Q'), Card('S', 'A')]
    table = [Card('S', 'T'), Card('S', 'T')]
    print(j.split_pot(12, hands,table))