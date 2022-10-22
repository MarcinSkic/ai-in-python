#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:44:31 2022

@author: marcinskic
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error

data_excel = pd.read_excel("practice_lab_2.xlsx",engine="openpyxl")
data = data_excel.values
data_columns_names = list(data_excel.columns)

# EX 1
def ex1():
    correlation = data_excel.corr()
    fig, ax = plt.subplots(13,1,figsize=(10,35))
    
    for i in range(13):
        y1 = data[:,13]
        y2 = data[:,i]
        
        ax[i].scatter(y1,y2)
        ax[i].set_title(data_columns_names[i])
    fig.tight_layout()    

# EX 2
def test(repeats):
    total_mape = 0
    for i in range(repeats):
    
        X, y = data[:,:-1], data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        
        minval = min(y_test.min(), y_pred.min())
        maxval = max(y_test.max(),y_pred.max())
        plt.scatter(y_test,y_pred)
        plt.plot([minval,maxval],[minval,maxval])
        plt.xlabel('y_test')
        plt.ylabel('y_pred')
        plt.show()
        
        mape = mean_absolute_percentage_error (y_test, y_pred)
        total_mape += mape
        print(mape)
        
    return (total_mape/repeats,repeats)

meanMape, repeats = test(20)
print("Mean mape: {} for {} tests".format(meanMape,repeats))

#EX 3
def removeOutliers(repeats):
    
def replaceOutliers(repeats):