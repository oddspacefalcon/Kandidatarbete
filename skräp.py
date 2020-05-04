import numpy as np
import matplotlib.pyplot as plt
dygn = 7
h = 24*3*dygn

t = np.linspace(0,h,h)
y = []
N_0 = 0
Tid = 0
First = True

for j in range(1,dygn+1):
    for i in range(1,25*3-2):
        if First:
            print('haje')
            a = [3,6,9,12]
        else:
            a = [3,6,9,12,15,21,23,26,29,32,35,38,43,48]*2

        if i in a:
            Tid=0
            last = y[len(y)-1]
            N_0 = 40 + last
            N = N_0*np.exp(np.log(1/2)/(6)*Tid)
        else:
            N = N_0*np.exp(np.log(1/2)/(6)*Tid)
        First = False  
        y.append(N)
        Tid += 1

t = t/(24*3)
fig, ax = plt.subplots()
plt.plot(t,y)
plt.xlabel('Dygn', fontsize=24)
plt.ylabel('Koffein i kroppen[mg]', fontsize=24)
plt.title('En kopp kaffe har 40 mg koffein i sig', fontsize=24)
plt.tick_params(axis='both', labelsize=24)
plt.show()
