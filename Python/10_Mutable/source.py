# Mutation


l1 = ["hi"]
print(l1)
l2 =  l1
l2 += ["bye"]
print(l2)
print(l1)
# It's Mutability in action. Whenever you assign a variable to another variable of mutable datatype, 
# any changes to the data are reflected by both variables. The new variable is just alias for the old variable.
# This is true for only mutable datatypes.

def add_to(num, target=[]):
    target.append(num)
    return target
print(add_to(1))
print(add_to(2))
print(add_to(3))
# Expected result should be [1] then [2] and then [3] rather than the real output i.e [1] then [1, 2] and then [1, 2, 3].
# Again this is the result of the mutability of lists which causes the change in the output. In Python the default arguments are evaluated
# once when the function is defined, not each time the function is called (equivalent to static). So never define default arguments of mutable type
# unless you know what you are doing.


# Edit in the above to code to avoid static property
def add(ele, target=None):
    if target is None:
        target = []
    target.append(ele)
    return target
print(add(1))
print(add(2))
