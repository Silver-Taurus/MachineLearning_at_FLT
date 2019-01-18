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

from collections import OrderedDict
    