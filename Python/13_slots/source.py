# __slots__ Magic

# In python, every class can have instance attributes. By default Python uses a dict to store an object's instance attributes. This is really helpful
# as it allows setting arbitary new attributes at runtime.
# However, for small classes with known attributes it might be a bottleneck. The dict wastes a lot of RAM. Python can't just allocate a static amount
# of memory at object creation to store all the attributes. Therefore it sucks a lot of RAM if you create a lot of objects. Still there is a way to circumvent this issue.
# it involves the usage of __slots__ to tell Python not to use a dict,and only allocate space for fixed set of attributes.

class MyClass(object):
    __slots__ = ["name","identifier"]
    
    def __init__(self,name,identifier):
        self.name = name
        self.identifier = identifier
        self.set_up()
# This code reduces the burden on RAM.
        

# On a side note, in PyPy, all of these optimizations is done by deafult.
        