#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:20:12 2022

@author: marcinskic
"""
#%% Importy
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

transformers = [("PCA",PCA(0.95)),("FastICA",FastICA(20,random_state=2022))]
scalers = [None,StandardScaler(),MinMaxScaler(),RobustScaler()]
classifiers = [kNN(weights='distance'),SVC(),DT(max_depth=3),RandomForest(max_depth=3)]

for transformer in transformers:
    for scaler in scalers:
        for classifier in classifiers:
            if(scaler == None):
                print("Lol")
            else:
                print("Not lol")
            tr_name, tr = transformer
            print(tr_name)