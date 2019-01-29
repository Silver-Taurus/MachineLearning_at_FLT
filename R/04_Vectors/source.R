# ----- Vectors -----
# Vectors store multiple values
# Create a vector
numbers = c(3,2,0,1,8)

# Get value by index
print(numbers[1])

# Get the number of items
print(lengt(numbers))

# Get the last value
print(numbers[length(numbers)])

# Get everything but an index
print(numbers[-1])

# Get the 1st 2 values
print(numbers[c(1,2)])

# Get the specific values
print(numbers[c(1,3)])

# Get the series of a range
print(numbers[1:3])

# Replace a value
numbers[5] = 1
print(numbers)

# Replace the 2nd and 3rd with 2
numbers[c(2,3)] = 2
print(numbers)

# Replace the 4th and 5th with 3 and 4
numbers[c(4,5)] = c(3,4)
print(numbers)

# Sort values
sort(numbers, decreasing = TRUE)

# Generate a sequence from 1 to 10
oneToTen = 1:10
print(oneToTen)

# Sequence from 3 to 27 adding 3 each time
add3 = seq(from = 3, to = 27, by = 3)
print(add3)

# Create 10 evens from 2
evens = seq(from = 2, by = 2, length.out = 10)
print(evens)

# Find out if a value is in vector
sprintf("4 is even %s", 4%in%evens)

# rep() repeats a value or values
rep(x = 2, times = 5, each = 2)   # each defines how many times to repeat each item
rep(x = c(1,2,3), times = 3, each = 2)

