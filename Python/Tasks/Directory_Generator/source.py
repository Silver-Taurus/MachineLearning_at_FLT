''' To make a Directory Generator '''

# The following program is designed to generate a number of directories.
# The directory names follow the pattern (MM_DD_YY_randnum), where:
#     - MM_DD_YY: is today's date as month/day/year
#     - randnum: is a random integer between 10000 and 50000
# For example, if today is May 12th, 2016, then the following would be valid names: 05_12_16_11050
# or 05_12_16_15001
#
# For this task, you should complete the functions:
# 1) `directory_count()`
# 2) `name_generator()`
# 3) `directory_creator(name)`
# 4) `create()`

import os
from datetime import datetime
from random import randint

LVAL = 100000000

def directory_count():
    """ Calculate the number of directories to be generated.
            I) Get the current minute using appropriate functionality from `datetime`
            II) Take the square root of the current minute + 15
            III) Round the square root to an integer
            VI) return the rounded number as the number of directories to be created
        args:
            NONE
        returns: 
            `dir_count`: number of directories to be created 
    """
    return round((datetime.today().minute + 15) ** 0.5)

def directory_creator(name=datetime.today().strftime('%d_%m_%y')):
    """ Create a single directory called `name` in the current working directory.
    
        args: 
            name: directory to be created
        
        returns:
            NONE
    """
    while True:
        rand_int = randint(1, LVAL+1)
        dir_name = name + str(rand_int)
        
        for _, dirnames, _ in os.walk('./'):
            if dir_name == dirnames:
                continue
        break
    os.mkdir(dir_name)

def create():
    """ Generate the necessary directories.
        Use `directory_count` to calculate the number of directories, then use
        `directory_creator` and `name_generator`.

        args:
            NONE
        returns:
            NONE
    """
    for _ in range(directory_count()):
        directory_creator()

# Change working directory to `parent_dir` or `create`
if 'parent_dir' not in os.getcwd():
    if not os.path.exists('./parent_dir'):
        os.mkdir(os.path.join(os.getcwd(), 'parent_dir'))
    os.chdir('parent_dir')
    print('Changing working dir to parent_dir')

print('Current directory content: ', os.listdir())
create()
print('Current directory content', os.listdir())
