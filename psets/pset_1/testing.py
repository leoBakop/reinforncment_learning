import numpy as np
import matplotlib.pyplot as plt 

T=10**5
k= 10
dt = .0001

y = np.zeros(T)
for t in range(T):
    y[t]= (1/((t+dt)**(1/3)))*((k*np.log(t+dt))**(1/3)) if t ==0 else (1/((t)**(1/3)))*((k*np.log(t))**(1/3)) #dt was used only for t==0

plt.figure(1)

plt.plot(np.arange(1,T+1),y) 
plt.legend()
plt.show()