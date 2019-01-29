''' Few info on datetime module '''

from datetime import time
from datetime import date
from datetime import datetime
from datetime import timedelta



# time objects

INIT_TIME = time(23, 23, 45, 200)
print(INIT_TIME)

END_TIME = time(minute=45, hour=23, second=50, microsecond=500)
print(END_TIME)
# once keyword arguments are passed, further to that all the arguments passes should have keyword

# modify hour
INIT_TIME = INIT_TIME.replace(hour=21)
print(INIT_TIME)



# date objects

DATE1 = date(2019, 1, 23)
print("DATE1 is:", DATE1)

# other functions and working behaviour is same as that of the time object

# get current local date
CUR_LOC_DATE = date.today()
print(CUR_LOC_DATE)



# datetime objects
DT = datetime(2022, 7, 4, 16, 30)
print(DT)

# other functions and working behaviour is same as that of the time or date object

# get current local date and time
print(datetime.today())



# formatting dates and times
T = time(hour=10, minute=15)
F_STRING = T.strftime('%I:%M %p')
print('First Format: ', F_STRING)
F_STRING = T.strftime('%H:%M:%S')
print('Second Format: ', F_STRING)

D = date(year=1999, month=8, day=6)
F_STRING = D.strftime('%d-%b-%Y')
print(F_STRING)


'''
Date formatting directives
Directive   Meaning	                        Example
%a	        Abbreviated weekday name        Sun, Mon, ..., Fri
%A	        Full weekday name           	Sunday, Monday, ..., Friday
%d	        Day of the month as a
            zero-padded decimal	            01, 02, 03, ... 31
%b         	Abbreviated month name	        Jan, Feb, ..., Dec
%B	        Full month name	                January, February, ..., December
%m          Month as a zero-padded decimal	01, 02,..., 12
%y          2 decimal year number 
            (without century)	            00, 01, ..., 99
%Y          4 decimal year number 
            (with century)	                1900, 1989, ..., 2015

Time formatting directives
Directive	Meaning	                        Example
%H      	Hour in 24-hour clock
            (zero-padded)	                00, 01, ..., 23
%I      	Hour in 12-hour clock
            (zero-padded)                  	00, 01, ..., 12
%p      	AM or PM	                    AM, PM
%M      	Minutes as zero-padded decimal	00, 01, ..., 59
%S	        Seconds as zero-padded decimal	00, 01, ...,59
'''



# defining a time delta object
DELTA1 = timedelta(days = 2, seconds=0, minutes=15, hours=1, weeks=4)
print(DELTA1, ' is stored in DELTA1')
print('Total number of seconds =', DELTA1.total_seconds())

# Unlike date, time and datetime objects we can perform artithmetic operations on the
# timedelta objects
