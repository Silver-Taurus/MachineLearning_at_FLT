# ----- ASSIGNMENT -----
# You can assign a value using = or <-
myNum1 <- 5
myNum2 = 6



# ----- VARIABLES -----
# Variable names start with a letter and can contain
# numbers, underscores and dots

# Most languages use data types to define how much
# space to set asside in memory
# Variables in R are assigned R Objects

# Types are dynamic which means a variable names data
# type changes based on the data assigned to it

# Here are the Vector types
# numeric
print(class(4))

# integer
print(class(4L))

# logical (TRUE, FALSE, T, F)
print(class(TRUE))

# complex
print(class(1 + 4i))

# character
print(class("Sample"))

# raw when converted into raw bytes
print(class(charToRaw("Sample")))



# You can check the objects class with
print(is.integer(myNum1))

# You can convert to different classes if possible
var1 = "String"
var2 = 7
print(as.character(var2))
print(as.numeric((var1)))     # will give you a warning message and NA conversion



# ----- ARITHMETIC OPERATORS -----
sprintf("4 + 5 = %d", 4 + 5)
sprintf("4 - 5 = %d", 4 - 5)
sprintf("4 * 5 = %d", 4 * 5)
sprintf("4 / 5 = %1.3f", 4 / 5)

# Modulus or remainder of division
sprintf("5 %% 4 = %d", 5 %% 4)

# Value raised to the exponent of the next
sprintf("4^2 = %d", 4^2)



# ----- VECTORS -----
# Vectos store multiple values
# Create a vector
numbers = c(3,2,0,1,8)

# Get value by index
print(numbers[1])    # here index starts from 1

#  Get the number of items
print(length(numbers))

# Get the last value
print(numbers[length(numbers)])

# Get everything but an index
print(numbers[-1])