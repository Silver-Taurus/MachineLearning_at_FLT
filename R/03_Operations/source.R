# ----- ARITHMETIC OPERATORS -----
sprintf("4 + 5 = %d", 4 + 5)
sprintf("4 - 5 = %d", 4 - 5)
sprintf("4 * 5 = %d", 4 * 5)
sprintf("4 / 5 = %1.3f", 4 / 5)

# Modulus or remainder of division
sprintf("5 %% 4 = %d", 5 %% 4)

# Value raised to the exponent of the next
sprintf("4^2 = %d", 4^2)



# ----- RELATIONAL OPERATORS -----
iAmTrue = TRUE
iAmFalse = FALSE

sprintf("4 == 5 : %s", 4 == 5)
sprintf("4 != 5 : %s", 4 != 5)
sprintf("4 > 5 : %s", 4 > 5)
sprintf("4 < 5 : %s", 4 < 5)
sprintf("4 >= 5 : %s", 4 >= 5)
sprintf("4 <= 5 : %s", 4 <= 5)

# Relational operator vector tricks
oneTo20 = c(1:20)

# Create vector of Ts and Fs depending on condition
isEven = oneTo20 %% 2 == 0
print(isEven)

# Create array of evens
justEvens = oneTo20[oneTo20 %% 2 == 0]
print(justEvens)



# ----- LOGICAL OPERATORS -----
cat("TRUE && FALSE = ", T && F, "\n")
cat("TRUE || FALSE = ", T || F, "\n")
cat("!TRUE = ", !T, "\n")



# ----- DECISION MAKING -----
age = 18

# if, else and else if works like other languages
if(age >= 18) {
  print("Drive and Vote")
} else if (age >= 16){
  print("Drive")
} else {
  print("Wait")
}



# ----- SWITCH -----
# Used when you have a limited set of possible values
grade = "Z"

switch(grade,
       "A" = print("Great"),
       "B" = print("Good"),
       "C" = print("Ok"),
       "D" = print("Bad"),
       "F" = print("Terrible"),
       print("No Such Grade"))
