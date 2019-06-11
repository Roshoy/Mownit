import datetime
import random
import matplotlib.pyplot as plt
import numpy as np

t = datetime.datetime.time(datetime.datetime.now())
curr_s = t.microsecond
tab = []
i = 0
#(curr_s+1 - first)%20 + 10
# first = curr_s
# while i <= 2048:
#     i+=1
#     print((curr_s, (curr_s-first) * random.randint(1, curr_s+10 - first)))
#     tab.append((curr_s - first, ((curr_s+1)%800 + 1) * random.randint(5, 10)))
#     while curr_s + 20 > t.microsecond:
#         t = datetime.datetime.time(datetime.datetime.now())
#     curr_s = t.microsecond

for i in range(128):
    tab.append((i , ((i+1)%10 + 1)*random.randint(8, 10)))

f = open("new_data.csv", "w+")

for i in tab:
    f.write(str(i[0])+","+ str(i[1]) + "\n")

x = [i[0] for i in tab]
y = [i[1] for i in tab]
tab = np.asarray(tab)


plt.plot(x,y)
plt.show()