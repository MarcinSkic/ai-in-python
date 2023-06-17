# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:09:24 2022

@author: marci
"""

import math
import numpy as np
import matplotlib.pyplot as plt
e = math.e

fig, ax = plt.subplots(3,2, figsize = (10,15))
x = np.arange(-5, 5, 0.1)

#TASK 1
y1 = np.tanh(x)

ax[0,0].plot(x,y1)
ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('y')

#TASK 2
y2 = (e**x - e**(-x))/(e**x+e**(-x))

ax[0,1].plot(x,y2)
ax[0,1].set_xlabel('x')
ax[0,1].set_ylabel('y')

#TASK 3
y3 = 1/1+e**(-x)

ax[1,0].plot(x,y3)
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('y')

#TASK 4
y4_1 = x[x <= 0]*0
y4_2 = x[x > 0]
y4 = np.concatenate((y4_1,y4_2))

ax[1,1].plot(x,y4)
ax[1,1].set_xlabel('x')
ax[1,1].set_ylabel('y')

#TASK 5
y5_1 = e**(x[x <= 0])-1
y5_2 = x[x > 0]
y5 = np.concatenate((y5_1,y5_2))

ax[2,0].plot(x,y5)
ax[2,0].set_xlabel('x')
ax[2,0].set_ylabel('y')
fig.tight_layout()