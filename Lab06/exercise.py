#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:41:33 2022

@author: marcinskic
"""

#%% DATA LOADING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

#%%MODELS IMPORTS
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import reciprocal
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, GaussianNoise
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.regularizers import l2, l1

#%%DATA SPLIT AND MANIPULATION
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)
scaler=RobustScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#%%BUILDING NEURAL NETWORK
num_epochs_for_search=13
num_epochs_for_final_training = 70
def build_model( n_neurons, noise, optimizer):
    model = Sequential()
    model.add(Dense(n_neurons, input_shape = (X.shape[1],), activation = 'tanh', name='Input'))
    
    model.add(Dense(n_neurons,activation = 'tanh',kernel_regularizer = l1(0.001),name='First'))
    model.add(GaussianNoise(noise,name='Second'))
    model.add(Dense(n_neurons,activation = 'tanh',name='Third'))
    model.add(Dense(class_num,activation='softmax'))
        
    model.compile(optimizer=optimizer(0.001), loss='binary_crossentropy', metrics=('accuracy'))
    return model    
#%%RANDOMIZED SEARCH -looking for best parameters
keras_classifier=KerasClassifier(build_model)
param_distribs={
    'n_neurons': [40,60],
    'noise': [0.12,0.31],
    'optimizer': [Adam,RMSprop]
    }
grid_search_cv=GridSearchCV(keras_classifier, param_distribs)
grid_search_cv.fit(X_train, y_train, epochs=num_epochs_for_search)

best_params=grid_search_cv.best_params_ 
#best_model_from_random=rnd_search_cv.best_estimator_ 

print("Najlepsze parametry: ",best_params)

choice = input("Kontynować? [y/N]")

if(choice != "y" or choice != "Y"):
    exit(0)

#%% CREATION OF BEST NETWORK - based on parameters search
best_model = build_model(**best_params)

best_model.fit(X_train,y_train,batch_size=32,epochs=num_epochs_for_final_training,validation_data=(X_test,y_test),verbose=2)

#%% CHARTS OF MODEL TRAINING HISTORY
history = best_model.history.history
floss_train = history['loss']
floss_test = history['val_loss']
acc_train = history['accuracy']
acc_test = history['val_accuracy']
fig,ax = plt.subplots(1,2, figsize=(20,10))

epochs = np.arange(0, num_epochs_for_final_training)
ax[0].plot(epochs, floss_train, label = 'floss_train')
ax[0].plot(epochs, floss_test, label = 'floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokładność')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()