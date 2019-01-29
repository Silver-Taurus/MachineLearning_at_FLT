# Object introspection

# In computer programming, introspection is the ability to determine the type of  an object at runtime. It is one of python's strengths. Everything in 
# python is an object and we can examine those objects.

# dir
my_list = [1,2,3]
dir(my_list)

print(type(' '))

print(type([]))

print(type(()))

print(type({}))

print(type(dict))

print(type(3))

print(type(3.5))

name = "Silver"
print(id(name))



# inspect module
#   - The inspect module also provides several useful functions to get information about live projects.
import inspect
print(inspect.getmembers(str))