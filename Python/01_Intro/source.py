''' This file provides the general introduction about the pyrhon '''

# Variable declaration
# VAR = 20        # Assigning value
A = B = C = 10  # Multiple declaration



# Python Tokens
#   --> Keywords
#   --> Identifiers
#   --> Literals
#   --> Operators


# Keywords
#    --> Python Keywords are special reserved words
#    --> Convey a special meaning to the compiler/interpreter
#    --> Each keyword have a special meaning and a specific operation
#    --> Keyword cannot be used as a variable
# False	 class	     finally	 is	        return
# None	    continue   for	    lambda	 try
# True  	 def	     from	    nonlocal  while
# and	    del        global 	 not	    with
# as	    elif	     if	    or	       yield
# assert   else	     import	 pass
# break	 except	   in	    raise


# Identifiers
#   --> Identifiers are the name used to identify a variable, function,
#        class or an object
#   --> Rules defined for naming an Identifiers:
#           - No special character except underscore can be used as an identifier
#           - Keyword should not be used as an idetifier name
#           - Python is case insensitive, i.e Var and var are two different identifier
#           - First character of an identifier can be character, underscore but not digit


# Literals
#   --> Literals are data given in a variable or constant
#   --> There are four types of literals in python:
#           - String literal
#           - Numeric literal
#           - Boolean literal
#           - Special literal

# String Literals
FNAME = "Silver"
LNAME = "Taurus"
print(FNAME + ' ' + LNAME)
MULTILINE = ''' First Line
            Second Line
            Third Line'''
print(MULTILINE)

# Numeric Literals
#   --> Int: +ve and -ve nubmers with no fractional part
#   --> Long: Unlimited integer size followed by upper or lowercase L.
#   --> Float: Real numbers with both integer and fractional part.
#   --> Complex: In the form of a + bj, where a is the real part and the b is the immaginary part.
# In python, the value of the integer is not restricted by the number of bits and can expand to the
# limit of the available memory. No special arrangement is required for storing large numbers.

# Boolean Literals
# They can have two values: True or False

# Special Literals
# Python has a special literal: None that is used to specify the field that is not created


# Operators
#   --> Arithmetic Operator: Takes two operand to perform operations on them (+,-,/,*,%,**,//)
#   --> Assignment Operator: Use to assign a value to a variable
#   --> Comparison Operator: Use to compare two entities
#   --> Logical Operator: Python Boolean operators return the last value evaluated
#   --> Bitwise Operator: Used to perform Bitwise operator
#   --> Identity Operator: These operators test if the two operands share an identity (is, is not)
#   --> Memebership Operator: Test whether a value is a member of a sequence (in, in not)



# DataTypes in Python
#   --> There are two types of data types in python:
#           - Immutables
#           - Mutables
# Immutabels
#   - Numbers
#   - Strings
#   - Tuples
# Mutables
#   - Lists
#   - Dictionary
#   - Sets
