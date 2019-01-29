''' File handling techniques '''

# File Objects

# Following are some of the ways in which we can open the file:
# r --> read
# w --> write
# a --> append
# r+ --> read and write
# rb --> read in binary
# wb --> write in binary

# 1st way
F = open('test.txt', 'r')    # By default - reading(r)
print(F.mode)
F.close()
# In this case we have to explicitly close the file

# 2nd way
# using context manager
with open('test.txt', 'r') as f:
    print(f.mode)
# In this we don't need to close the file explicitly
# In the 2nd way, even after closing of the file we still have the access to the f variable
print(f.closed)
print(f.mode)

# Reading the file
with open('test.txt', 'r') as f:
    F_CONTENTS = f.readlines()  # read all the lines
    print(F_CONTENTS)

# Reading all the lines at once may cause few memory issues if the size of the file is very large
# So, line by line reading is more preferable, also in this way we have more control over what to
# read and what to not.
with open('test.txt', 'r') as f:
    F_CONTENTS = []
    while True:
        line = f.readline()
        if line:
            F_CONTENTS.append(line)
        else:
            break
    print(F_CONTENTS)

# In order to get more control over the data we read we can use read() function
with open('test.txt', 'r') as f:
    SIZE = 10
    while True:
        F_CONTENTS = f.read(SIZE)
        if F_CONTENTS:
            print('pos:', f.tell())     # f.tell gives the position pointer in the file
            print(F_CONTENTS)
            print()
        else:
            break
# similarly the seek() function is use to change the position pointer in the file


# Writing in the file
# In this case, unlike the read if the file not exist then it will get it created for us
# and if the file that name already exists then it will overwrite it.
with open('test2.txt', 'w') as f:
    f.write('Test')
    f.seek(0)
    f.write('R')


# Using Reading and Writing on Multiple files
# Copying one file to another file
with open('test.txt', 'r') as rf:
    with open('copy_text.txt', 'w') as wf:
        for line in rf:
            wf.write(line)


# Copying a image file
# For this we have to copy the byte code of the file
with open('God-Tatsuya.jpg', 'rb') as rf:
    with open('copy_God-Tatsuya.jpg', 'wb') as wf:
        for line in rf:
            wf.write(line)
    