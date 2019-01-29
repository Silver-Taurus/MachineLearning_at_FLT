# Comprehensions

# Comprehensions are a feature of Python which I woul reallly miss if I ever have to leave it. Comprehensions are constructs that allow sequences to be built
# from other sequences. Following are the types available:
#   --> list comprehension
#   --> dictionary comprehension
#   --> set comprehension
#   --> generator comprehension


# List Comprehensions
#   - it provides a short and concise way to create lists. It consists of square brackets containing an expression followed by a for clause, the nzero or more for or if 
#   clauses. The expressions can be anything, meaning you can put in all kinds of objects in lists.
multiples3 = [i for i in range(1, 31) if i%3 == 0]
print(multiples3)
squared = [x**2 for x in range(10)]
print(squared)


# Dictionary Comprehension
#   - They are used in a same way.
mcase = {'a': 10, 'b': 34, 'A': 7, 'Z': 3}
mcase_rev = {v: k for k, v in mcase.items()}
print(mcase_rev)


# Set Comprehensions
#   - They are also similar to the list comprehensions
squared_no_rep = {x**2 for x in [1,1,2]}
print(squared_no_rep)


# Generator Comprehensions
#   - They are also similar to the list comprehensions. The only differnece is that
# they don't allocate momory for the whole list but generate one item at a time, thus
# memory efficient
multiples_gen = (i for i in range(1, 31) if i%3 == 0)
for x in multiples_gen:
    print(x)
    