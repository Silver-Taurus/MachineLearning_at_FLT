# Decorators
# They are significant part of the python. In simple words, they are functions which modify the functionality of other functions.
# They help to make our code shorter and more Pytthonic.

# Everything in Python is an object
def hi(name="Silver"):
    return "Hi "+name
print(hi())

greet = hi
print(greet())

del hi
# print(hi()) --> outputs: NameError
print(greet())



# Defining functions within functions
def hi(name="Silver"):
    print("now you are inside the hi() function")
    
    def greet():
        return "now you are in the greet() function"
    
    def welcome():
        return "now you are in the welcome() function"
    
    print(greet())
    print(welcome())
    print("now you are back in the hi() function")

hi()

# This shows that whenever you call hi(), greet() and welcome()
# are also called. However the greet() and welcome() functions
# are not available outside the hi() function e.g:

# greet()
# outputs: NameError: name 'greet' is not defined

# So now we know that we can define functions in other functions. 
# In other words: we can make nested functions. Now we need to learn one more thing, that functions can return functions too.



# Returning functions from within functions
def hi(name="Silver"):
    def greet():
        return "now you are in the greet() function"
    def welcome():
        return "now you are in the welcome() function"
    
    if name == "Silver":
        return greet
    else:
        return welcome
    
a = hi()
print(a)
print(a())



# Giving a function as an argument to another function
def hi():
    return "Hi Silver"
def do(func):
    print(func())
do(hi)



# Writing a decorator
def decorator(func):
    def wrapTheFunction():
        print("I am doing some boring work before executing func()")
        func()
        print("I am doing some boring work after executing func()")
    return wrapTheFunction

def func_req_dec1():
    print("I am the function which needs some decoration")

func_req_dec1()
func_req_dec1 = decorator(func_req_dec1)
func_req_dec1()

@decorator
def func_req_dec2():
    print("I am the function which needs some decoration")
func_req_dec2()
# This is just a short way of making up a decorated function. Here is how we could have run the previous code sample using @.

# Now there is one problem with our code.
print(func_req_dec1.__name__)
# That’s not what we expected! Its name is “a_function_requiring_decoration”. Well our function was replaced by wrapTheFunction. 
# It overrode the name and docstring of our function. Luckily Python provides us a simple function to solve this problem and that is functools.wraps.
from functools import wraps

def new_decorator(func):
    @wraps(func)
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
        func()
        print("I am doing some boring work after executing a_func()")
    return wrapTheFunction
 
@new_decorator
def func_req_dec3():
    print("I am the function which needs some decoration")

print(func_req_dec3.__name__)



# Blueprint:

# from functools import wraps
# def decorator_name(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         if not can_run:
#             return "Function will not run"
#         return f(*args, **kwargs)
#     return decorated
#
# @decorator_name
# def func():
#     return("Function is running")
#
# can_run = True
# print(func())

# Output: Function is running

# can_run = False
# print(func())

# Output: Function will not run

#Note: @wraps takes a function to be decorated and adds the functionality of copying over the function name, docstring, 
# arguments list, etc. This allows to access the pre-decorated function’s properties in the decorator.



# Nesting decorator within a function
def logit(logfile="out.log"):
    def logging_decorator(func):
        @wraps(func)
        def wraped_function(*args,**kwargs):
            log_string = func.__name__ + " was called"
            print(log_string)
            with open(logfile,'a') as file:
                file.write(log_string + '\n')
        return wraped_function
    return logging_decorator

@logit()
def myfunc1():
    pass
myfunc1()
# A file called out.log now exists, with the above string

@logit(logfile="func2.log")
def myfunc2():
    pass
myfunc2()
# A file calles func2.log now exists, with the same above string



# Decorator Classes
class log(object):
    _logfile = "out.log"
    
    def __init__(self,func):
        self.func = func
        
    def __call__(self,*args):
        log_string = self.func.__name__ + " was called"
        print(log_string)
        
        with open(self._logfile,'a') as file:
            file.write(log_string + '\n')
        
        self.notfiy()
        return self.func(*args)
    
    def notify(self):
        pass

log._logfile = "out.log"
@log
def myfunc():
    pass
myfunc()

class email_log(log):
    def __init__(self,email="admin@myproject.com",*args,**kwargs):
        self.email = email
        super(email_log,self).__init__(*args,**kwargs)
        
    def notify(self):
        pass
# From here, @email_log works just like @logit but sends an email to the admin in addition to logging.