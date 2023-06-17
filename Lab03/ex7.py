#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:23:41 2022

@author: marcinskic
"""
#%%
import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SVM
#%% Podział
X, y = load_breast_cancer(return_X_y=True,as_frame=True)

columns = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
#%% Skalowanie
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train) #Teaching scaler only with train data
X_train = scaler.transform(X_train) #Transforming train and...
X_test = scaler.transform(X_test) #...test data
#%% Testowanie kNN i SVM
for weight in ('uniform','distance'):   #Test for every weight
    for neighbours in range(2,10,1):        #and neighbour combination
        kNN_model = kNN(neighbours,weights=weight)
        kNN_model.fit(X_train,y_train)
        y_pred = kNN_model.predict(X_test)
        print("kNN results for: {} neighbours and '{}' weight function".format(neighbours,weight))
        print("accuracy: {}, confusion matrix: ".format(accuracy_score(y_test,y_pred)))
        print(confusion_matrix(y_test, y_pred))

for kernel in ('linear', 'poly', 'rbf', 'sigmoid'): #Test for every type of kernel (precoumputed requires square matrix)
    
    SVM_model = SVM(kernel=kernel)
    SVM_model.fit(X_train,y_train)
    y_pred = SVM_model.predict(X_test)
    print("SVM results for: {} kernel".format(kernel))
    print("accuracy: {}, confusion matrix: ".format(accuracy_score(y_test,y_pred)))
    print(confusion_matrix(y_test, y_pred))
#%% Testowanie DT
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree

for depth in range(1,10,1):
    
    model = DT(max_depth=depth)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("DT results for: {} max depth".format(depth))
    print("accuracy: {}, confusion matrix: ".format(accuracy_score(y_test,y_pred)))
    print(confusion_matrix(y_test, y_pred))

    #%% Rysowanie drzew
    from matplotlib import pyplot as plt
    plt.figure(figsize=(50,20))
    tree_vis = plot_tree(model,feature_names=columns[:-1],class_names=['N', 'Y'], fontsize = 20)