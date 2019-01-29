''' Conditionals and Booleans '''

# is Conditonal
A = [1, 2, 3]
B = [1, 2, 3]
C = A
print(A is B)
print(A is C)
print(id(A))
print(id(B))
print(id(C))


# False Values in Python:
#   --> False
#   --> None
#   --> Zero of any numberic type
#   --> Any empty sequence. For ex: '', [], ().
#   --> Any empty mapping. For ex: {}.
# Everything else is considered True in Python.


# Ternary Operators
# Ternary operators are more commonly known as conditional expressions in Python.
# These operators evaluate something based on condition being true or not.

# first way
# statement_if_true if condition else statement_if_false
IS_NICE = True
STATE = "nice" if IS_NICE else "not nice"
print(STATE)

# second way
# (if_test_is_false, if_test_is_true)[test]
NICE = True
PERSONALITY = ('mean', 'nice')[NICE]
print("The cat is", PERSONALITY)
# This works simply because True == 1 and False == 0, and so can be done with lists in addition to
# tuples.

# Reason to avoid using a tupled ternery is that it results in both elements of the tuple being
# evaluated, whereas the if-else ternary operator does not.

#Example:
#   condition = True
#   print(2 if condition else 1/0)

#   Output is 2

#   print((1/0, 2)[condition])
#   ZeroDivisionError is raised

# This happens because with the tupled ternary technique, the tuple is first built, then an index
# is found. For the if-else ternary operator, it follows the normal if-else logic tree. Thus, if
# one case could raise an exception based on the condition, or if either case is a
# computation-heavy method, using tuples is best avoided.


# ShortHand Ternary
# True or "Some" --> True
# False or "Some" --> Some
OUTPUT1 = None
OUTPUT2 = "Hello"
MSG1 = OUTPUT1 or "No data returned"
MSG2 = OUTPUT2 or "No data returned"
print(MSG1)
print(MSG2)
