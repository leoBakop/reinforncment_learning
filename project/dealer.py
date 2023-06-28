from card import Card
import numpy as np

class Dealer():

    def __init__(self, seed = 1):
        ''' Initialize a leducholdem dealer class
        '''
        np.random.seed(seed)
        #self.np_random = np.random.RandomState()
        self.deck = [ 
                      Card('S', 'T'), Card('H', 'T'),Card('D', 'T'),Card('C', 'T'),
                      Card('S', 'J'), Card('H', 'J'),Card('D', 'J'),Card('C', 'J'),
                      Card('S', 'Q'), Card('H', 'Q'),Card('D', 'Q'),Card('C', 'Q'),
                      Card('S', 'K'), Card('H', 'K'),Card('D', 'K'),Card('C', 'K'),
                      Card('S', 'A'), Card('H', 'A'),Card('D', 'A'),Card('C', 'A')
                      ]
        self.shuffle()
        self.pot = 0

    def shuffle(self):
        np.random.shuffle(self.deck)

    def deal_card(self):
        """
        Deal one card from the deck

        Returns:
            (Card): The drawn card from the deck
        """
        return self.deck.pop()


if __name__ == '__main__':
    d = Dealer()
    card_1=d.deal_card()
    card_2 =d.deal_card()
    print(card_1)
    print(card_2)