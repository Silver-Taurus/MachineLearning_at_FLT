''' Enumerate '''

# Enumerate is a built-in function in python. Its usefulness can not be summarized in a single line.
# It allows us to loop over something and have an automatic counter.

MY_LIST = ['apple', 'banana', 'grapes', 'pear']
for c, val in enumerate(MY_LIST,1):
    print(c,val)


LIST = ['apple', 'banana', 'grapes', 'pear']
COUNTER_LIST = list(enumerate(LIST,1))
print(COUNTER_LIST)