# Enumerate
# It is a built-in function in python. Its usefulness can not be summarized in a single line. It allows us to loop over something and have an automatic 
# counter.

my_list = ["apple","banana","grapes","pear"]
for c, val in enumerate(my_list,1):
    print(c,val)


lis = ['apple','banana','grapes','pear']
counter_list = list(enumerate(lis,1))
print(counter_list)