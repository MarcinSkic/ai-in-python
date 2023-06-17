#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:55:20 2022

@author: marcinskic
"""

import pandas as pd
import numpy as np
data = pd.read_excel('practice_lab_3.xlsx')
originalData = data.copy()
columns = list(data.columns)
mask = data['Gender'].values == 'Female'
data.loc[mask, 'Gender'] = 1
data.loc[~mask, 'Gender'] = 0
#%%
cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ['Property_Area'])
# zadanie 3.2
def qualitative_to_0_1(data, column,value_to_be_1):
    mask = data [column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data
data=qualitative_to_0_1(data, 'Married', 'Yes')
data=qualitative_to_0_1(data, 'Education', 'Graduate')
data=qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data=qualitative_to_0_1(data, 'Loan_Status', 'Y')
#%%
from sklearn.model_selection import train_test_split
data=data.astype(np.float64)
X=data.drop('Loan_Status', axis=1)
y=data['Loan_Status']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
#%%
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC as SVM
models = [kNN(), SVM()]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%%
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
models = [kNN(), SVC()]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
#%%
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
model = DT(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
tree_vis = plot_tree(model,feature_names=data.columns[:-1],class_names=['N', 'Y'], fontsize = 20)