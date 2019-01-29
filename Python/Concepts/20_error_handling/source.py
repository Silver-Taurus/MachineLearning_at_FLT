'''  Error handling '''


# General Example
try:    # put the block where error may occur
    f = open('test_file.txt')
    var = bad_var
except Exception:
    print('Error Found')


# Treating multiple errors simultaneously
try:
    f = open('test.txt', 'rb')
except (IOError, EOFError) as E:
    print("An error occurred. {}".format(E.args[-1]))


# Little Particular Example
# Treating multiple error individually
try:
    f = open('test_file.txt')
    var = bad_var
except FileNotFoundError as FNFE:      # put the error to be handled
    print(FNFE)
except Exception as E:      # general category
    print(E)
else:   # if no error is found then the else block will execute
    print(f.read())
    f.close()


# Particular Example
print('Performing division (x / y): ')
X = int(input('Enter x: '))
Y = int(input('Enter y: '))

try:
    ANS = X / Y
except ZeroDivisionError:
    print('Y cannot be 0!!!')
else:
    print(ANS)
finally:
    print('Executing Anyway!!!')    # finally block executes anyway
