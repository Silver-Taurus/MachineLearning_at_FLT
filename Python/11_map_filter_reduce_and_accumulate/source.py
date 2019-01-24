# Map, Filter, Reduce and Accumulate
# These are three functions whicih facilitate a functional approach to programming.


# Map applies a function to all the items in an input_list

#   items = [1,2,3,4,5]
#   squared = []
#   for val in items:
#       squared.append(i**2)

# Map allows us to implement this in a much simpler and nicer way.
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
print(squared)

def mul(x):
    return x*x
def add(x):
    return x+x

funcs = [mul, add]
for i in range(5):
    val = list(map(lambda x: x(i), funcs))
    print(val)

    
# Filter creates a list of elements for which a function returns true.
num_list = range(-5,5)
less_than_zero = list(filter(lambda x: x < 0, num_list))
print(less_than_zero)


# Reduce is a really useful function for performing some computation on a list and returning the result.
# It applies a rolling computation to sequential pairs of values in a list.

#   product = 1
#   list = [1, 2, 3, 4]
#   for num in list:
#       product = product * num

from functools import reduce
product = reduce((lambda x, y: x * y), [1,2,3,4])
print(product)

#Working : 
#   At first step, first two elements of sequence are picked and the result is obtained.
#   Next step is to apply the same function to the previously attained result and the number just succeeding the second element and the result is again stored.
#   This process continues till no more elements are left in the container.
#   The final returned result is returned and printed on console.

# Python code to demonstrate the working of reduce() using opeartor functions
import operator
print("The sum of the first five numbers is = ", reduce(operator.add,[1,2,3,4,5],))


# reduce() vs accumulate() 

#   Both reduce() and accumulate() can be used to calculate the summation of a sequence elements. But there are differences in the implementation aspects in both of these.
#   reduce() is defined in “functools” module, accumulate() in “itertools” module.
#   reduce() stores the intermediate result and only returns the final summation value. Whereas, accumulate() returns a list containing the intermediate results. The last number of the list returned is summation value of the list.
#   reduce(fun,seq) takes function as 1st and sequence as 2nd argument. In contrast accumulate(seq,fun) takes sequence as 1st argument and function as 2nd argument.
from itertools import accumulate
print(list(accumulate([1,2,3,4], lambda x,y: x*y)))