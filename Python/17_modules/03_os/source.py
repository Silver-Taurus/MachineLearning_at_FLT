''' Importing and working on os module '''

import os
from datetime import datetime

print(dir(os))
print(os.getcwd())      # gives the current working directory

os.chdir('./')          # changes the current working directory - in this case that will remain same
print(os.getcwd())

print(os.listdir('../../'))    # prints all the folders in the directory - by default be cur-dir

print(os.listdir())

os.mkdir('Demo-1')
# os.mkdir('Demo-2/Sub-Dir')    --> This will give error as mkdir function only creats one dir and
# in this case the upper level directory Demo-2 not exist so it cannot create the Sub-Dir

os.makedirs('Demo-2/Sub-Dir')   # This can make multiple level directories

print(os.listdir())

# This deletes the directory or directories and work in similar way as that of mkdir and makedirs
os.rmdir('Demo-1')
os.removedirs('Demo-2/Sub-Dir')
print(os.listdir())

# os.rename('test.txt', 'demo.txt')     # rename the file in the directory

print(os.stat('source.py'))     # gives all the info or attributes assigned to a file
print('size of source file =', format(os.stat('source.py').st_size))  # give the specified attribute
MOD_TIME = os.stat('source.py').st_mtime    # gives the time of last modification of the file
print('timestamp:', datetime.fromtimestamp(MOD_TIME))

# os. walk() function walks thorugh all the directories from the initial directory while
# giving three yield values (since the walk() is a generator )
for dirpath, dirnames, filenames in os.walk('../../'):
    print('Directory Path:', dirpath)
    print('Directories:', dirnames)
    print('Files:', filenames)
    print()

# to get the environment variable
print(os.environ.get('HOME'))   # in this case we are getting specifically the home variable

# os.path.join() function keep track for any /, \ or //
# so that two different path can join correctly.
ML_DIR_PATH = os.path.join(os.environ.get('HOME'), 'Projects\MachineLearning_at_FLT')
print('Path for Machine Leanring directory is:', ML_DIR_PATH)

print(os.path.basename(ML_DIR_PATH))    # gives the name of base(last) directory
print(os.path.dirname(ML_DIR_PATH))     # gives the full path of the directory in which the
                                        # base directory exists
print(os.path.split(ML_DIR_PATH))       # gives both the dirname and base name splitted in two
                                        # string and stored in a tuple 
print(os.path.exists('../test.txt'))    # gives true or false on the basis of whether the file
                                        # exists or not in the specified directory
print(os.path.isdir('source.py'))       # checks if the basename is a dir or not
print(os.path.isfile('source.py'))      # checks if the basename is a file or not

print(os.path.splitext('source.py'))    # separates the root name and the extension of the file

print(dir(os.path))
