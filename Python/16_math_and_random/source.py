# Calling a math module and setting the alias as m
import math as m

X = -5
print(m.fabs(X))        # absolute function 

print(m.pow(X, 2))      # pow function

A = 3
B = 4
print(m.sqrt(A**2 + B**2))   # sqrt function

VAL = 23.54
print(m.ceil(VAL))
print(m.floor(VAL))
print(m.trunc(VAL))



# for random integer
from random import randint
print(randint(1, 10))

from random import randrange
print(randrange(1, 11))

# odd random integers
print(randrange(1,102,2))