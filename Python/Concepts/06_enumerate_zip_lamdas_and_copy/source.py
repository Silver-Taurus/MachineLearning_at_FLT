#! /usr/bin/env python3

import copy 

# Enumerate is a built-in function in python. Its usefulness can not be summarized in a single line.
# It allows us to loop over something and have an automatic counter.

MY_LIST = ['apple', 'banana', 'grapes', 'pear']
for c, val in enumerate(MY_LIST,1):
    print(c,val)

LIST = ['apple', 'banana', 'grapes', 'pear']
COUNTER_LIST = list(enumerate(LIST,1))
print(COUNTER_LIST)



# zip
# The purpose of zip() is to map the similar index of multiple containers so that they can be used
# just using as single entity.

# initializing lists 
name = [ "Manjeet", "Nikhil", "Shambhavi", "Astha" ] 
roll_no = [ 4, 1, 3, 2 ] 
marks = [ 40, 50, 60, 70 ] 

mapped = set(zip(name, roll_no, marks))
print(mapped)

# How to unzip?
# Unzipping means converting the zipped values back to the individual self as they were. This is done
# with the help of “*” operator.

# unzipping values 
namz, roll_noz, marksz = zip(*mapped)
print(namz)
print(roll_noz)
print(marksz)

# Practical Applications : There are many possible applications that can be said to be exected using zip,
# be it student database or scorecard or any other utility that requires mapping of groups.



# Lambdas
# Lambdas are one line functions. They are also known as anonymous functions in some other languages.
# You might want to use lambdas when you don’t want to use a function twice in a program. They are
# just like normal functions and even behave like them.
add = lambda x, y: x + y
print(add(3, 5))

# list sorting
lis = [(1, 2), (4, 1), (9, 10), (13, -3)]
lis.sort(key=lambda x: x[1])
print(lis)

list1 = [2, 6, 8, 4]
list2 = [1, 5, 7, 3]
print(list1, list2)
data = zip(list1, list2)
temp_data = copy.deepcopy(data)
print(list(tsemp_data))
list1, list2 = map(lambda x: list(x), zip(*data))
print(list1, list2)



# In Python, Assignment statements do not copy objects, they create bindings between a target and an object.
# When we use = operator user thinks that this creates a new object; well, it doesn’t. It only creates a new
# variable that shares the reference of the original object. Sometimes a user wants to work with mutable
# objects, in order to do that user looks for a way to create “real copies” or “clones” of these objects.
# Or, sometimes a user wants copies that user can modify without automatically modifying the original at
# the same time, in order to do that we create copies of objects.

# A copy is sometimes needed so one can change one copy without changing the other. In Python, there are two
# ways to create copies :
#   Deep copy
#   Shallow copy
# In order to make these copy, we use copy module. We use copy module for shallow and deep copy operations.

# Deep copy is a process in which the copying process occurs recursively. It means first constructing a new
# collection object and then recursively populating it with copies of the child objects found in the original.
# In case of deep copy, a copy of object is copied in other object. It means that any changes made to a copy
# of object do not reflect in the original object. In python, this is implemented using “deepcopy()” function.

# A shallow copy means constructing a new collection object and then populating it with references to the
# child objects found in the original. The copying process does not recurse and therefore won’t create copies
# of the child objects themselves. In case of shallow copy, a reference of object is copied in other object.
# It means that any changes made to a copy of object do reflect in the original object. In python, this is
# implemented using “copy()” function.