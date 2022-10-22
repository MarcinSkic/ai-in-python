#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:58:53 2022

@author: marcinskic
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.arange(-3,3, 0.1).reshape((-1,1))
y = np.tanh(x) + np.random.randn(*x.shape)*0.2
ypred = LinearRegression().fit(x,y).predict(x)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, ypred)
plt.legend([ 'F(x) - aproksymująca',
 'f(x) - aproksymowana zaszumiona'])