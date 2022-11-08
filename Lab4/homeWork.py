#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:20:12 2022

@author: marcinskic
"""
#%% Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.pipeline import Pipeline

#%% Read and manipulate data
data_frame = pd.read_csv('ionosphere_data.csv', header=None)
data_frame.columns = ["C" + str(i) for i in range(36)]
data_frame.drop("C0",axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data_frame.iloc[:,:-1],data_frame.iloc[:,-1],test_size=0.2,shuffle=True)
# %% Train and test models
transformers = [("PCA",PCA(0.95)),("FastICA",FastICA(20,random_state=2022))]
scalers = [("None",None),("Standard",StandardScaler()),("MinMax",MinMaxScaler()),("Robust",RobustScaler())]
classifiers = [("kNN",kNN(weights='distance')),("SVC",SVC()),("DT",DT(max_depth=3)),("Random Forest",RandomForest(max_depth=3))]

for transformer in transformers:
    for scaler in scalers:
        for classifier in classifiers:
            pipe = None
            tr_name, tr = transformer
            sc_name, sc = scaler
            cl_name, cl = classifier
            
            if(scaler == None):
                pipe = Pipeline([
                    ['transformer',tr],
                    ['classifier',cl]
                    ])
            else:
                pipe = Pipeline([
                    ['transformer',tr],
                    ['scaler',sc],
                    ['classifier',cl]
                    ])
            pipe.fit(X_train,y_train)
            y_pred = pipe.predict(X_test)
            print("Transformer: {} Scaler: {} Classifier: {}".format(tr_name,sc_name,cl_name));
            print(confusion_matrix(y_test, y_pred))
            print(accuracy_score(y_test,y_pred))
            print('\n\n')