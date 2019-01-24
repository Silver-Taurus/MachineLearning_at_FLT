''' Few info on datetime module '''

from datetime import time
from datetime import date
from datetime import datetime



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
