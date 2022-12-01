#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:29:05 2022

@author: marcinskic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
data = load_digits()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]


from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import reciprocal
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
num_epochs=10

def build_model(n_hidden, n_neurons, learning_rate, activation,optimizer):
    model = Sequential()
    model.add(Dense(n_neurons, input_shape = (X.shape[1],), activation = activation))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(optimizer=optimizer(learning_rate), loss='categorical_crossentropy', metrics=('accuracy'))
    return model    
#%%
# RANDOMIZED SEARCH
keras_classifier=KerasClassifier(build_model)
# w randomized search mozna dac wiecej kombinacji bo sa sprawdzane losowe z nich
param_distribs={
    'n_hidden': [0,1,2,3],
    'n_neurons': np.arange(1,100),
    'learning_rate': reciprocal(3e-4, 3e-2),
    'activation': ['relu','selu','softmax'],
    'optimizer': [SGD,Adam,RMSprop]
    }
rnd_search_cv=RandomizedSearchCV(keras_classifier, param_distribs, n_iter=10, cv=5)
rnd_search_cv.fit(X_train, y_train, epochs=num_epochs)

best_params_from_random=rnd_search_cv.best_params_ 
best_model_from_random=rnd_search_cv.best_estimator_ 

print(best_params_from_random,best_model_from_random)
