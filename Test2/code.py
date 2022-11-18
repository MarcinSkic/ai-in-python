#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:19:06 2022

@author: marcinskic
"""

#%%Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

#%% Ex1 - Data manipulation
data_frame = pd.read_csv('gender_classification.csv')
data_frame = pd.get_dummies(data_frame,
   columns=['Favorite Color','Favorite Music Genre','Favorite Beverage', 'Favorite Soft Drink'],
   prefix=['FavCol','FavMusicGenre','FavBev','FavSoftDrink'], 
   drop_first=True)

#%% Ex2 - Decision Tree
X=data_frame.iloc[:,1:]
y=data_frame['Gender']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2022)


model = DT(max_depth=3, random_state=2022)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

plot_tree(model,feature_names=data_frame.columns,class_names=['N','Y'])
print("Największy wpływ miały: FavMusicGenre Folk/Traditional,")
print(confusion_matrix(y_test,y_pred))
print("Czułość: {}".format(8/(8+2)))

#%%Ex3 - Pipeline
pip1 = Pipeline([
    ['scaler',MinMaxScaler()],
    ['clasificator',SVC(kernel='poly')]
    ])
pip2 = Pipeline([
    ['scaler',MinMaxScaler()],
    ['clasificator',kNN(7,weights='distance')]
    ])

pip1.fit(X_train,y_train)
y_pred = pip1.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred)

pip2.fit(X_train,y_train)
y_pred = pip2.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred)
print("Lepiej zadziałał SVM z wynikiem: {}".format(svm_acc))
#%%Ex4 - PCA
data_frame2 = pd.read_csv('drift_database.csv')
X2=data_frame2.iloc[:,:-1]
y2=data_frame2['target']

X_train, X_test, y_train, y_test = train_test_split(X2,y2,random_state=2022)
pca = PCA()
pca.fit(X_train)
variance = pca.explained_variance_ratio_
cumulated_variance = variance.cumsum()
target_components = (cumulated_variance<0.98).sum()+1

print("Aby wyjaśnić 98% wariancji wystarczą {} składowe główne".format(target_components))
