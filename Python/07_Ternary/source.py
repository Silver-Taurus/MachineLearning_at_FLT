# Ternary Operators
# Ternary operators are more commonly known as conditional expressions in Python.
# These operators evaluate something based on condition being true or not.

# first way
# statement_if_true if condition else statement_if_false
is_nice = True
state = "nice" if is_nice else "not nice"
print(state)

# second way
# (if_test_is_false, if_test_is_true)[test]
nice = True
personality = ("mean","nice")[nice]
print("The cat is",personality)
# This works simply because True == 1 and False == 0, and so can be done with lists in addition to tuples.

# Reason to avoid using a tupled ternery is that it results in both elements of the tuple being evaluated, 
# whereas the if-else ternary operator does not.

#Example:
#   condition = True
#   print(2 if condition else 1/0)

#   Output is 2

#   print((1/0, 2)[condition])
#   ZeroDivisionError is raised

# This happens because with the tupled ternary technique, the tuple is first built, then an index is found. 
# For the if-else ternary operator, it follows the normal if-else logic tree. Thus, if one case could raise an 
# exception based on the condition, or if either case is a computation-heavy method, using tuples is best avoided.


# ShortHand Ternary
# True or "Some" --> True
# False or "Some" --> Some
output1 = None
output2 = "Hello"
msg1 = output1 or "No data returned"
msg2 = output2 or "No data returned"
print(msg1)
print(msg2)