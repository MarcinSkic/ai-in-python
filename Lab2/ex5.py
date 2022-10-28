# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:58:50 2022

@author: marci
"""
#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#READING DATA
raw_data = load_diabetes()
data_excel = pd.DataFrame(raw_data.data,columns=raw_data.feature_names)
data_excel['target'] = raw_data.target

data = np.array(data_excel.values)
data_columns = list(data_excel.columns)

#CORRELATION ANALIZE
corr = data_excel.corr()

fig, ax = plt.subplots(2,5,figsize=(25,10))
for i in range(10):
    attr = data[:,i]
    target = data[:,-1]
    
    ax[int(i/5)][i%5].scatter(target,attr)
    ax[int(i/5)][i%5].set_xlabel("target")
    ax[int(i/5)][i%5].set_ylabel(data_columns[i])
    fig.tight_layout()

#TRAINING LINEAR REGRESSION
#Method to generate chart of regression
def generateTestVsPredPlot(index,y_test,y_pred,title):
    minval = min(y_test.min(),y_pred.min())
    maxval = max(y_test.max(),y_pred.max())

    ax[index].plot([minval,maxval],[minval,maxval])
    ax[index].scatter(y_test,y_pred)
    ax[index].set_xlabel('y_test')
    ax[index].set_ylabel('y_pred')
    ax[index].set_title(title)
    
#Method to train and test values with different treatment of outliers
def runModel(X,y,repeats):
    #Export this variables to skip returning all this values as big tuple
    global y_test
    global y_pred_base
    global y_pred_mean
    global y_pred_no_outliers
    global mean_mape_base
    global mean_mape_mean
    global mean_mape_no_outliers
    
    mean_mape_base = 0
    mean_mape_mean = 0
    mean_mape_no_outliers = 0
    for i in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        
        outliers = np.abs((y_train - y_train.mean())/y_train.std())>2   #Mask defining outliers
        
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean() #Replacing outliers with mean value
        
        X_train_no_outliers = X_train[~outliers,:]  #Removing outliers
        y_train_no_outliers = y_train[~outliers] #Removing outliers
        
        linReg_base = LinearRegression()
        linReg_mean = LinearRegression()
        linReg_no_outliers = LinearRegression()
        
        linReg_base.fit(X_train,y_train)
        linReg_mean.fit(X_train,y_train_mean)
        linReg_no_outliers.fit(X_train_no_outliers,y_train_no_outliers)
        
        y_pred_base = linReg_base.predict(X_test)
        y_pred_mean = linReg_mean.predict(X_test)
        y_pred_no_outliers = linReg_no_outliers.predict(X_test)
        
        mean_mape_base += mean_absolute_percentage_error(y_test, y_pred_base)
        mean_mape_mean += mean_absolute_percentage_error(y_test, y_pred_mean)
        mean_mape_no_outliers += mean_absolute_percentage_error(y_test, y_pred_no_outliers)
    
    mean_mape_base /= repeats
    mean_mape_mean /= repeats
    mean_mape_no_outliers /= repeats

#Variables used through all testing
X = data[:,:-1]
y = data[:,-1]
repeats = 20

#Running and visualizing model
fig, ax = plt.subplots(3,1,figsize=(5,15)) 
runModel(X,y,repeats)

generateTestVsPredPlot(0,y_test, y_pred_base, 'Base prediction, mape: {}'.format(mean_mape_base))
generateTestVsPredPlot(1,y_test, y_pred_mean, 'Outliers replaced, mape: {}'.format(mean_mape_mean))
generateTestVsPredPlot(2,y_test, y_pred_no_outliers, 'Outliers removed, mape: {}'.format(mean_mape_no_outliers))

fig.tight_layout()
plt.show()

#ADDING EXTRA ATTRIBUTES
#Seeing weights
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
linReg = LinearRegression()
linReg.fit(X_train,y_train)
weights_names = data_columns[:-1]
weights = linReg.coef_
plt.bar(weights_names,weights)
plt.xticks(weights_names,rotation='vertical')
plt.show()

#Generating data
extra_data = np.stack([
    X[:,4] / X[:,5],
    X[:,4] / X[:,3],
    X[:,4] / X[:,2],
    X[:,4] / X[:,8]
    ],axis=-1)
X_extra = np.concatenate([X,extra_data],axis=-1)

#Running and visualizing model
fig, ax = plt.subplots(3,1,figsize=(5,15))  
runModel(X_extra,y,repeats)
generateTestVsPredPlot(0,y_test, y_pred_base, 'Base prediction, mape: {}'.format(mean_mape_base))
generateTestVsPredPlot(1,y_test, y_pred_mean, 'Outliers replaced, mape: {}'.format(mean_mape_mean))
generateTestVsPredPlot(2,y_test, y_pred_no_outliers, 'Outliers removed, mape: {}'.format(mean_mape_no_outliers))

fig.tight_layout()
plt.show()

