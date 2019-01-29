''' For understanding the scope of various variables in the python source file'''

# LEGB
# Local, Enclosing, Global, Built-in


# Case-1
X = 'global x'  # global var

def test1():
    Y = 'local y'   # local var
    print(Y)    
    print(X)    # global X will print

test1()
print()
# print(y)  --> will give error as y is local to test1


# Case-2
def test2():
    X = 'local x' 
    print(X)    # local X will print

test2()
print(X)    # global X will print
print()


# Case-3
def test3():
    global X    # value overwritten
    X = 'local x'
    print(X)

test3()
print(X)
print()


# Case-4
def min():
    pass
# MIN_VAL = min([3,2,14,4,6]) --> will give error
# as global min function will be called not the built-in


# Case-5
def outer():
    X = 'outer x'
    Y = 'outer y'   # enclosing var for inner() function

    def inner():
        X = 'inner x'
        print(X)    # prints local x
        print(Y)    # prints enclosing y
    
    inner()
    print(X)    # print enclosing x
    print(Y)    # print enclosing y

outer()
print(X)    # global x i.e local x of test3()
print()


# Case-6
def outer2():
    X = 'outer x'
    Y = 'outer y'   # enclosing var for inner() function

    def inner():
        nonlocal X  # enclosing x wiil be overwritten with local x
        X = 'inner x'
        print(X)    # prints local x
        print(Y)    # prints enclosing y
    
    inner()
    print(X)    # print local x i.e. inner x
    print(Y)    # print enclosing y

outer()
print(X)    # global x i.e local x of test3()
