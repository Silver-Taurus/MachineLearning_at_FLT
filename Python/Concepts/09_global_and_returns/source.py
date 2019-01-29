# Global & Return


# Return example
def add(val1, val2):
    return val1 + val2
print(add(2,3))

# Global example
def sub(val1, val2):
    global res
    res = val1 - val2
sub(3,2)
print(res)

# So first lets talk about the first code which involves the return keyword. That function is doing is that it is assigning the value 
# to the variable which is calling the function in our case is in the print function.
# However in the another code we are using the global keyword whcih makes the res variable a global variable and thus can be accessed 
# outside the sub function and thus is directly used in the print function.



# Multiple return values
def profile1():
    global name
    global age
    name = "Silver"
    age = 19

profile1()
print(name)
print(age)
# This is not hte good way to implement the multiple return values

# Multiple return values using lists, tuples, dicts
def profile2():
    name = "Silver"
    age = 19
    return (name,age)
profile_data = profile2()
print(profile_data)

# Or by Convention
def profile3():
    name = "Silver"
    age = 19
    return name, age
profile_name, profile_age = profile3()
print(profile_name,' ',profile_age)