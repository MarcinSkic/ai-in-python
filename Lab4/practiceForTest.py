# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:51:02 2022

@author: marci
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True,as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2022,stratify=y)
model_dt = DT(max_depth=5,random_state = 2022)
model_dt.fit(X_train,y_train)
y_pred = model_dt.predict(X_test)

temp = PCA()
temp.fit(X_train)
variance = temp.explained_variance_ratio_
cumulated_variance = variance.cumsum()
num = (cumulated_variance<0.99).sum()+1

tr = PCA(num)

tr.fit(X_train)
var1 = tr.n_components_
X_train = tr.transform(X_train)
X_test = tr.transform(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))