# Generators


# First we should understand about the iterators. Iterator is an object that enables a programmer to traverse a container
# particularly list. However an iterator performs traversal and gives access to data elements in a container, but does not perform
# iteration.

# There are mainly three parts:
#   --> Iterable
#   --> Iterator
#   --> Iteration

# An interable is any object in Python which has an __iter__ or a __getitem__ method defined which returns an iterator or can take indexes.
# In short, an iterable is any object which can provide us with iterator.

# An iterator is any object in Python which has a __next__ method defined.

# When we use loop to loop over something, this process itself is calle as iteration.

# Generators are iterators, but you can only iterate over them once. It's because they do not store all the values in memory, they generate the
# values on the fly. You use them by iterating over them, either with a for loop or by passing them to any function or construct that iterates. 
# Most of the time generators are implemented as functions. However they do not return they yield it.

# Fiboonacci generator example
def fib_gen(n):
    a = b = 1
    for i in range(n):
        yield a
        a, b = b, a+b
    
def test():
    num = int(input("Enter the number: "))
    for val in fib_gen(num):
        print(val)

test()