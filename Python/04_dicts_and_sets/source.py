''' Dictionaries and Set Data Structure '''

# Dictionary is a useful data structure which stores the key and its value and access it in
# constant time

STUDENT = {'name': 'John', 'age': 25, 'courses': ['Math', 'Os', 'ToC', 'CG', 'Java']}
print(STUDENT)
print(STUDENT['courses'])
# print(STUDENT['phone']) --> will give an key error

print(STUDENT.get('name'))
print(STUDENT.get('phone'))     # If the key doesn't exists it will simply return None
print(STUDENT.get('phone', 'Not Found'))
# If the key doesn't exists it will return the specified statement

# del STUDENT['age']    --> This will delete the age key (del keyword)
# Another way is pop function
AGE = STUDENT.pop('age')
print(STUDENT)
print(AGE)

# Traversing through keys and values
print(STUDENT.keys())
print(STUDENT.values())
print(STUDENT.items())
# By default it loops over the key of the dictionary
for key in STUDENT:
    print(key)
for key, val in STUDENT.items():
    print((key, val))


print()


# Set is a really useful data structure. sets behave mostly like lists with the distinction that
# they can not contain duplicate values. It is really useful in a lot of cases. For instance you
# might want to check whether there are duplicates in a list or not.

# First option
LIST = ['a', 'b', 'c', 'b', 'd', 'm', 'n', 'n']
DUPLICATES = []
for val in LIST:
    if LIST.count(val) > 1:
        if val not in DUPLICATES:
            DUPLICATES.append(val)
print(DUPLICATES)

# Second and more elegant solution
DUPLICATES = set([x for x in LIST if LIST.count(x) > 1])
print(DUPLICATES)


# Sets also have few other methods
VALID = set(['yellow', 'red', 'blue', 'green', 'black'])
INPUT_SET = set(['red', 'brown'])
# Intersection
print(INPUT_SET.intersection(VALID))
# Difference
print(INPUT_SET.difference(VALID))
