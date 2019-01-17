# set Data Structure
# set is a really useful data structure. sets behave mostly like lists with the distinction that they can not contain
# duplicate values. It is really useful in a lot of cases. For instance you might want to check whether there are duplicates in a
# list or not.

# First option
lis = ['a','b','c','b','d','m','n','n']
duplicates = []
for val in lis:
    if lis.count(val) > 1:
        if val not in duplicates:
            duplicates.append(val)
print(duplicates)
    
# Second and more elegant solution
duplicates = set([x for x in lis if lis.count(x) > 1])
print(duplicates)


# Sets also have few other methods
valid = set(["yellow","red","blue","green","black"])
input_set = set(["red","brown"])
# Intersection
print(input_set.intersection(valid))
# Difference
print(input_set.difference(valid))