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

def createBoxChart():
    global X
    X = data[:,:-1]
    global y
    y = data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    plt.boxplot(y_train)
    plt.title("Medianowa wartość mieszkania")
    global igrek
    igrek = y_train
    global outliers
    outliers = np.abs((y_train - y_train.mean())/y_train.std())>2
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    global y_mean 
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()
    
    ax[0].boxplot(y_train)
    ax[1].boxplot(y_train_mean)

def removeOutliers(repeats):
    totalMape = 0
    X = data[:,:-1]
    y = data[:,-1]
    
    for i in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        
        outliers = np.abs((y_train - y_train.mean())/y_train.std())>2
        X_train_no_outliers = X_train[~outliers,:]
        y_train_no_outliers = y_train[~outliers]
        
        linReg = LinearRegression()
        linReg.fit(X_train_no_outliers,y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        
        minval = min(y_test.min(),y_pred.min())
        maxval = max(y_test.max(),y_pred.max())
        plt.scatter(y_test,y_pred)
        plt.plot([minval,maxval],[minval,maxval])
        plt.xlabel("y_test")
        plt.ylabel("y_pred")
        plt.show()
        
        totalMape += mean_absolute_percentage_error(y_test, y_pred)
    
    meanMape = totalMape/repeats
    return (meanMape,repeats)
        
        

#EX 3
def replaceOutliers(repeats):
    total_mape = 0
    X = data[:,:-1]
    y = data[:,-1]
    weights_names = data_columns_names[:-1]
    
    for i in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        
        outliers = np.abs((y_train - y_train.mean())/y_train.std())>2
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean()
        
        linReg = LinearRegression()
        linReg.fit(X_train,y_train_mean)
        y_pred = linReg.predict(X_test)
        
        minval = min(y_test.min(),y_pred.min())
        maxval = max(y_test.max(),y_pred.max())
        #plt.scatter(y_test, y_pred)
        #plt.plot([minval,maxval],[minval,maxval])
        #plt.xlabel("y_test")
        #plt.ylabel("y_pred")
        #plt.show()
        
        
        x = np.arange(len(weights_names))
        weights = linReg.coef_
        plt.bar(weights_names,weights)
        plt.xticks(weights_names,rotation='vertical')
        plt.show()
        
        
        total_mape += mean_absolute_percentage_error(y_test, y_pred)
    mean_mape = total_mape/repeats
    return (mean_mape,repeats)

def addExtraData(repeats):
    total_mape = 0
    X = data[:,:-1]
    y = data[:,-1]
    weights_names = data_columns_names[:-1]
    
    global extra_data
    extra_data = np.stack([
        X[:,4] * X[:,3],
        X[:,4] / X[:,7]
        ],axis=-1)
    
    global X_additional
    X = np.concatenate([X,extra_data],axis=-1)
    
    extra_names = np.stack([
        "TlenkiAzotu * PrzyRzece",
        "TlenkiAzotu / OdlOdCentrum"
        ])
    global weights_names_additional
    weights_names = np.concatenate([weights_names,extra_names],axis=0)
    
    for i in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        
        
        linReg = LinearRegression()
        linReg.fit(X_train,y_train)
        y_pred = linReg.predict(X_test)
        
        minval = min(y_test.min(),y_pred.min())
        maxval = max(y_test.max(),y_pred.max())
        plt.scatter(y_test, y_pred)
        plt.plot([minval,maxval],[minval,maxval])
        plt.xlabel("y_test")
        plt.ylabel("y_pred")
        plt.show()
        
        x = np.arange(len(weights_names))
        weights = linReg.coef_
        plt.bar(weights_names,weights)
        plt.xticks(weights_names,rotation='vertical')
        plt.show()
        
        
        total_mape += mean_absolute_percentage_error(y_test, y_pred)
    mean_mape = total_mape/repeats
    return (mean_mape,repeats)
        

#meanMape, repeats1 = test(20)
#print("Normal testing: Mean mape: {} for {} tests".format(meanMape,repeats1))
#meanMape2, repeats2 = replaceOutliers(20)
#print("Replacing outliers: Mean mape: {} for {} tests".format(meanMape2, repeats2))
#meanMape3, repeats3 = removeOutliers(20)
#print("Removing outliers: Mean mape: {} for {} tests".format(meanMape3, repeats3))

meanMape4, repeats4 = addExtraData(20)
print("Adding extra attributes: Mean mape: {} for {} tests".format(meanMape4, repeats4))


#createBoxChart()
