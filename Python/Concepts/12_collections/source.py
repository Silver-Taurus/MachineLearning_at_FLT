# Collections

# Python ships with a amodule that contains a number of container data types called Collections.
# This includes:
#   --> defaultdict
#   --> OrderDict
#   --> counter
#   --> deque
#   --> namedtuple
#   --> enum.Enum



# defaultdict
#   - Unlike dict, with defaultdict we do not need to check whether a key is present or not.
from collections import defaultdict

colours = (
        ("Silver","Blue"),
        ("Silver","Black"),
        ("Rahul","Red"),
        ("Rohan","Brown"),
        )

fav_colours = defaultdict(list)
for name, colour in colours:
    fav_colours[name].append(colour)
print(fav_colours)
# One other very important use case is when you are appending to nested lists inside a dictionary. If a key is not already present in the dictionary
# then you are greeted with the KeyError.
# defaultdict allows us to circumvent the issue in a clever way.

tree = lambda: defaultdict(tree)
dict_of_dict = tree()
dict_of_dict["colours"]["fav"] = ["Black","Blue"]
print(dict_of_dict)

# You can print dict_of_dict using json.dumps.
import json
print(json.dumps(dict_of_dict))



# OrderdDict
#   - Ordered Dict keeps its entries sorted as they are initially inserted. Overwritting a value of an existing a key doesn't change the position of that
#     key. However, deleting and reinserting an entry moves the key to the nend of the dictionary.

data = {"Red":100, "Green":101, "Blue":102, "Black":103, "White":104}
for key,val in data.items():
    print(key,val)
# Entries may or may not be retrieved in predictable order in a normal dict

print()

from collections import OrderedDict
new_data = OrderedDict([("Red",100),("Green",101),("Blue",102),("Black",103),("White",104)])
for key,val in new_data.items():
    print(key,val)
# Entries will always be retrieved in order.



# Counter
#   - counter allows us to count the occurences of a particular item. For instance it can be used to count the number of individual favourite colours:
from collections import Counter
new_colours = (
        ("Silver","Blue"),
        ("Silver","Black"),
        ("Rahul","Red"),
        ("Rohan","Brown"),
        )
favs = Counter(name for name, colour in colours)
print(favs)

# We can also count the most common line in a file using it.
#   with open("filename","rb") as file:
#       line_count = Counter(file)
#   print(line_count)



# Deque
#   - Deque provides you with a double ended queue which means that you can append and delete elements from either side of the queue. First of all 
#   you ahve to import the deque module form the collections library:
from collections import deque
deq = deque()
deq.append('1')
deq.append('2')
deq.append('3')
print(len(deq))
print(deq[0])
print(deq[-1])
print(deq)
deq.appendleft('4')
print(deq)
deq.pop()
print(deq)
deq.popleft()
print(deq)

# We can also limit the amount of items a deque can hold. By doing this when we achieve the maximum limit of our deque it will simply pop the elements 
# from the other end.
new_deq = deque([0,1,2,3,4], maxlen=5)
print(new_deq)
new_deq.extend([5])
print(new_deq)



# namedtuple
#   - A tuple is basically a immutable list which allows you to store a sequence of values separatd by commas. They are just like lists but have
#   but have a few key differences. The major one is that unlike lists, you can not reassign an item in tuple.
#   - With Namedtuples we don't have to use integer indexes for accessing the members of tuple. They can be considered similar to dictionary but 
#   but unlike them they are immutable.
from collections import namedtuple
animal = namedtuple("Animal","name age type")
perry = animal(name="perry", age=31, type="cat")
print(perry)
print(perry.name)
print(perry.age)

# A namedtuple has two required arguments. They are tuple name and the tuple field_names. Morewver, as 'nametuple' instances do not have per-instance
# dictionaries, they are lightweight an require no more memory that regular tuples. This makes them faster than dictionaries. Also if you want you can
# indexes also to access the data in the namedtuples.



# enum.Enum
#   - Another useful collection is the enum object. It is available in the enum module. Enums are basically a way to organise various things.
from enum import Enum

class species(Enum):
    cat = 1
    dog = 2
    horse = 3
    aardvark = 4
    butterfly = 5
    owl = 6
    dragon = 7
    unicorn = 8
    kitten = 1
    puppy = 2

neko = animal(name="neko",age=4,type=species.kitten)
tom = animal(name="tom",age=25,type=species.cat)
print(neko.type == tom.type)
print(neko.type)
print(tom.type)