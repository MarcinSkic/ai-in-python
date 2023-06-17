# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:29:44 2022

@author: marci
"""

#%% GOOGLE COLLAB
from google.colab import drive
drive.mount('/content/drive')
#%% DATA LOADING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('/content/drive/MyDrive/Studia/AI/practice_lab_3.csv',";")
originalData = data.copy()
columns = list(data.columns)
mask = data['Gender'].values == 'Female'
data.loc[mask, 'Gender'] = 1
data.loc[~mask, 'Gender'] = 0

cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ['Property_Area'])

def qualitative_to_0_1(data, column,value_to_be_1):
    mask = data [column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data
data=qualitative_to_0_1(data, 'Married', 'Yes')
data=qualitative_to_0_1(data, 'Education', 'Graduate')
data=qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data=qualitative_to_0_1(data, 'Loan_Status', 'Y')

data=data.astype(np.float64)
X=data.drop('Loan_Status', axis=1)
y=data['Loan_Status']
#%%MODELS IMPORTS
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import reciprocal
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, GaussianNoise, LayerNormalization, BatchNormalization
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
#%%LAB6EX2
num_epochs_for_search=30
num_epochs_for_final_training=70

n_neurons = 64
#do_rate = 0.5
noise = 0.1
learning_rate = 0.001
optimizer = Adam

def build_model(do_rate):
    model = Sequential()
    model.add(Dense(n_neurons, input_shape = (X.shape[1],), activation = 'relu', name='Input'))
    
    model.add(Dense(n_neurons,activation = 'selu',name='Dense1'))
    model.add(Dropout(do_rate,name='Dropout1'))
    model.add(Dense(n_neurons,activation = 'selu',name='Dense2'))
    model.add(Dropout(do_rate,name='Dropout2'))
        
    model.add(Dense(1,activation = 'sigmoid',name='Output'))
    model.compile(optimizer=optimizer(learning_rate), loss='binary_crossentropy', metrics=('accuracy'))
    return model    

#GRID SEARCH -looking for best parameters
keras_classifier=KerasClassifier(build_model)
param_distribs={
    'do_rate': [0,0.2,0.3,0.5],
    }
grid_search_cv=GridSearchCV(keras_classifier, param_distribs)
grid_search_cv.fit(X_train, y_train, epochs=num_epochs_for_search)

best_params=grid_search_cv.best_params_ 

print("Najlepsze parametry: ",best_params)

input("Naciśnij ENTER żeby kontynuować: ")

#CREATION OF BEST NETWORK - based on parameters search
best_model = build_model(**best_params)

best_model.fit(X_train,y_train,batch_size=32,epochs=num_epochs_for_final_training,validation_data=(X_test,y_test),verbose=2)

#CHARTS OF MODEL TRAINING HISTORY
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
#%%LAB6EX3
num_epochs_for_search=30
num_epochs_for_final_training=70

n_neurons = 64
do_rate = 0.5
#noise = 0.1
learning_rate = 0.001
optimizer = Adam

def build_model(noise):
    model = Sequential()
    model.add(Dense(n_neurons, input_shape = (X.shape[1],), activation = 'relu', name='Input'))
    
    model.add(Dense(n_neurons,activation = 'selu',name='Dense1'))
    model.add(GaussianNoise(noise,name='GaussianNoise1'))
    model.add(Dense(n_neurons,activation = 'selu',name='Dense2'))
    model.add(GaussianNoise(noise,name='GaussianNoise2'))
        
    model.add(Dense(1,activation = 'sigmoid',name='Output'))
    model.compile(optimizer=optimizer(learning_rate), loss='binary_crossentropy', metrics=('accuracy'))
    return model    

#GRID SEARCH -looking for best parameters
keras_classifier=KerasClassifier(build_model)
param_distribs={
    'noise': [0,0.1,0.2,0.3],
    }
grid_search_cv=GridSearchCV(keras_classifier, param_distribs)
grid_search_cv.fit(X_train, y_train, epochs=num_epochs_for_search)

best_params=grid_search_cv.best_params_ 

print("Najlepsze parametry: ",best_params)

input("Naciśnij ENTER żeby kontynuować: ")

#CREATION OF BEST NETWORK - based on parameters search
best_model = build_model(**best_params)

best_model.fit(X_train,y_train,batch_size=32,epochs=num_epochs_for_final_training,validation_data=(X_test,y_test),verbose=2)

#CHARTS OF MODEL TRAINING HISTORY
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
#%%LAB6EX4
num_epochs_for_final_training=70

n_neurons = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001
optimizer = Adam

block = [
  Dense,
  LayerNormalization,
  Dropout,
  GaussianNoise]
args = [
  (n_neurons,'selu'),(),(do_rate,),(noise,)]

def build_model(repeat_num):
  model = Sequential()
  model.add(Dense(n_neurons, input_shape = (X.shape[1],), activation = 'relu', name='Input'))

  for i in range(repeat_num):
    for layer,arg in zip(block, args):
      model.add(layer(*arg))
      
  model.add(Dense(1,activation = 'sigmoid',name='Output'))
  model.compile(optimizer=optimizer(learning_rate), loss='binary_crossentropy', metrics=('accuracy', 'Recall','Precision'))
  return model    


for repeats in [1,5]:
  model = build_model(repeats)
  model.summary()
  #input("Naciśnij ENTER żeby kontynuować: ")

  model.fit(X_train,y_train,batch_size=32,epochs=num_epochs_for_final_training,validation_data=(X_test,y_test),verbose=2)

  #CHARTS OF MODEL TRAINING HISTORY
  history = model.history.history
  floss_train = history['loss']
  floss_test = history['val_loss']
  acc_train = history['accuracy']
  acc_test = history['val_accuracy']
  fig,ax = plt.subplots(1,2, figsize=(20,10))

  average_floss_train = np.asarray(floss_train).astype('float32').sum()/num_epochs_for_final_training
  average_floss_test = np.asarray(floss_test).astype('float32').sum()/num_epochs_for_final_training
  average_acc_train = np.asarray(acc_train).astype('float32').sum()/num_epochs_for_final_training
  average_acc_test = np.asarray(acc_test).astype('float32').sum()/num_epochs_for_final_training

  epochs = np.arange(0, num_epochs_for_final_training)
  ax[0].plot(epochs, floss_train, label = 'floss_train')
  ax[0].plot(epochs, floss_test, label = 'floss_test')
  ax[0].set_title('Funkcje strat')
  ax[0].legend()
  ax[1].set_title('Dokładność')
  ax[1].plot(epochs, acc_train, label = 'acc_train')
  ax[1].plot(epochs, acc_test, label = 'acc_test')
  ax[1].legend()

  fig.show()

  print("Średnia strata na zbiorze uczącym: {}".format(average_floss_train))
  print("Średnia strata na zbiorze testowym: {}".format(average_floss_test))
  print("Średnia dokładność na zbiorze uczącym: {}".format(average_acc_train))
  print("Średnia dokładność na zbiorze testowym: {}".format(average_acc_test))

  #input("Naciśnij ENTER żeby kontynuować: ")