'''
it is not necessary to write *args or **kwargs. Only the * (asterisk) is necessary. 
You could have also written *var and **vars. Writing *args and **kwargs is just a convention. 
So now lets take a look at *args first.
'''

# ---- Usage of *args -----
# *args is used to send a non-keyworded variable length argument list to the function. Here’s an example to 
# help you get a clear idea:
def test_var_args(*args):
    for arg in args:
        print("Arg through *argv: ", arg)

test_var_args("python","eggs","test")

# ----- Usage of **kwargs -----
# **kwargs allows you to pass keyworded variable length of arguments to a function. You should use **kwargs 
# if you want to handle named arguments in a function. Here is an example to get you going with it:
def test_var_kwargs(**kwargs):
    for key, value in kwargs.items():
        print("{} = {}".format(key,value))
    
test_var_kwargs(name = "Silver")