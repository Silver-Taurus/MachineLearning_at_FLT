''' A Simple source file having the code for storing of employee data '''

class Employee(object):
    ''' Class for storing the data of employee '''

    RAISE_AMT = 1.05

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
    
    @property
    def email(self):
        ''' function to return the email of the employee '''
        return f'{self.first}.{self.last}@email.com'

    @property
    def fullname(self):
        ''' function to return the fullname of the employee '''
        return f'{self.first} {self.last}'

    def apply_raise(self):
        ''' function to increase the pay by raise_amt '''
        self.pay = int(self.pay * self.RAISE_AMT)
        