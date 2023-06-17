#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:23:50 2022

@author: marcinskic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_frame = pd.read_csv("practice_lab_2.csv",sep=";")
data_frame = data_frame.iloc[200:300,:7]
corr = data_frame.corr()
#Najsilniej skorelowane są ze sobą przestępczość i tlenki azotu

data = data_frame.values
data_columns = list(data_frame.columns)
X = data[:,:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,random_state=5)
linReg = LinearRegression()
linReg.fit(X_train,y_train)
weights = linReg.coef_

names = data_columns[:-1]

plt.bar(names,weights)
plt.xticks(names,)