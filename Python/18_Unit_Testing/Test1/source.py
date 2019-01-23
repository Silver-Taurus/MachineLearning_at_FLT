''' A simple source file having the code for a simple calculator '''

def add(x, y):
    ''' add function '''
    return x + y

def sub(x, y):
    ''' subtract function '''
    return x - y

def mul(x, y):
    ''' multiply function '''
    return x * y

def div(x, y):
    ''' divide function '''
    if y == 0:
        raise(ValueError('Can not divide by zero!!!'))
    return x / y
    