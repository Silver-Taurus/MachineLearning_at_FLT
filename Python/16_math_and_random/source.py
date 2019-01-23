''' Few uses of math and random module's functions '''

# Calling a math module and setting the alias as m
import math as m
# for random integer
from random import randint
from random import randrange
from random import choice
from random import shuffle

# absolute function
X = -5
print(m.fabs(X))

# pow function
print(m.pow(X, 2))

# sqrt function
A = 3
B = 4
print(m.sqrt(A**2 + B**2))

# other math function
VAL = 23.54
print(m.ceil(VAL))
print(m.floor(VAL))
print(m.trunc(VAL))



# Printing random integer
print(randint(1, 10))
print(randrange(1, 11))

# odd random integers
print(randrange(1, 102, 2))

# Select Rock, Paper or Scissors
OPTIONS = ['Rock', ' Paper', 'Scissors']
def rps():
    '''  function to return a random choice '''
    return choice(OPTIONS)
print(rps())

# Shuffling the elements of a list
print(OPTIONS)
shuffle(OPTIONS)
print(OPTIONS)

# Picking a random playing card
def pick_card():
    ''' function to pick a card of a given number '''

    card_type = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    card_number = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']

    choosen_type = choice(card_type)
    choosen_number = choice(card_number)

    return "".join(f"{choosen_number} of {choosen_type}.")
print(pick_card())
