''' It is not necessary to write *args or **kwargs. Only the * (asterisk) is necessary.
     You could have also written *var and **vars. Writing *args and **kwargs is just a convention.
	So now lets take a look at *args first. '''


# ---- Usage of *args -----
# *args is used to send a non-keyworded variable length argument list to the function. Here’s an
# example to help you get a clear idea:
def test_var_args(*args):
    ''' function to take multiple arguments and lopp throgh them '''
    for arg in args:
        print("Arg through *argv: ", arg)

test_var_args("python", "eggs", "test")


# ----- Usage of **kwargs -----
# **kwargs allows you to pass keyworded variable length of arguments to a function. You should use
# **kwargs if you want to handle named arguments in a function. Here is an example to get you going
# with it:
def test_var_kwargs(**kwargs):
    ''' function to take multiple arguments along with their keywords and loop through them '''
    for key, value in kwargs.items():
        print("{} = {}".format(key, value))
test_var_kwargs(name="Silver")


# Using *args and **kwargs to call a function
def test_args_kwargs(arg1, arg2, arg3):
    ''' function to utilise both the features of args and kwargs in one functuion '''
    print("arg1: ", arg1)
    print("arg2: ", arg2)
    print("arg3: ", arg3)

ARGS = ("two", 3, 5)
test_args_kwargs(*ARGS)
KWARGS = {"arg3":5, "arg1":2, "arg2":3}
test_args_kwargs(**KWARGS)


# Order of using *args **kwargs and formal args:
# So if you want to use all three of these in functions then the order is
#   --> some_func(fargs, *args, **kwargs)


# *args store the positional arguments in form of a tuple
# **kwargs store the positional arguments in form of dictionary
def student_info(*args, **kwargs):
    ''' function to print the *args and **kwargs '''
    print(args)
    print(kwargs)

student_info('Math', 'ToC', Name='Silver', age=19)
COURSES = ['Math', 'ToC', 'Os', 'CG']
INFO = {'fname': 'Silver', 'lname': 'Taurus', 'age': 19}
student_info(COURSES, INFO)     # In this case both COURSES and INFO will be treated as positonal
                                # arguments and are packed inside the *args
student_info(*COURSES, **INFO)  # In this case both wil be unpacked according to the passed *
                                # i.e COURSES as *args and INFO as **kwargs
