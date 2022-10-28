#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:17:09 2022

@author: marcinskic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

#TASK 1
data_frame = pd.read_csv("surgical.csv")
data = data_frame.values
data_columns = np.array(data_frame.columns)

change_factors = data.mean(axis=0)/data.std(axis=0)
columnWithBiggestChangeFactor = data_columns[change_factors==change_factors.max()][0]
print("Kolumna z najwiekszym współczynnikiem zmienności: {}".format(columnWithBiggestChangeFactor))

#TASK 2
fig, ax = plt.subplots(1,2,figsize=(10,5))

X = data[:,:-1]
y = data[:,-1]
ax[0].hist(y)
ax[0].set_title("Histogram")
ax[1].boxplot(y)
ax[1].set_title("Boxplot")
plt.show()

#TASK 3
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2022,shuffle=False)
linReg = LinearRegression()
linReg.fit(X_train,y_train)
weights = linReg.coef_
weights_names = data_columns[:-1]

mask_of_highest_weight = weights==weights.max()

print("Największy wpływ ma zmienna: {} i jej waga wynosi: {}".format(weights_names[mask_of_highest_weight][0],weights[mask_of_highest_weight][0]))

#TASK 4
y_pred = linReg.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)

print("Średni błąd absolutny w procentach: {}".format(mape))
print("Średni błąd absolutny: {}".format(mae))
print("Średni błąd kwadratów: {}".format(mse))

#EXTRA
minval = min(y_test.min(),y_pred.min())
maxval = max(y_test.max(),y_pred.max())

plt.scatter(y_test,y_pred)
plt.plot([minval,maxval],[minval,maxval])
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()