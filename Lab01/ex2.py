#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:06:52 2022

@author: marcinskic
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

data_excel = pd.read_excel("practice_lab_1.xlsx",engine="openpyxl")
data_excel = data_excel.iloc[:100,:7]

columnsNames =np.array(list(data_excel.columns))
data = data_excel.values


# TASK 1
even = data[::2,:]
odd = data[1::2,:]
subtracted = even-odd

# TASK 2
mean_task2 = data.mean()
std_task2 = data.std()

arr_task2 = (data - mean_task2)/std_task2

# TASK 3
arr_task3 = (data-data.mean(axis=0))/(np.spacing(data.std(axis=0))+data.std(axis=0))

#TASK 4
changeFactor = data.mean(axis=0) / (np.spacing(data.std(axis=0))+data.std(axis=0))

#TASK 5
mask_task5 = changeFactor == changeFactor.max()
arr_task5 = data[:,mask_task5]
print("Kolumna z największym współczynnikiem zmiennosci:",columnsNames[mask_task5][0])

#TASK 6
mask_task6 = data>data.mean(axis=0)
arr_task6 = np.count_nonzero(mask_task6,axis=0) #Można też mask_task6.sum(axis=0)

#TASK 7
max_values = data.max(axis=0)
mask_task7 = max_values == max_values.max()
print("Kolumy z wartosciami maksymalnymi:",columnsNames[mask_task7])

#TASK 8
mask_task8_1 = (data==0).sum(axis=0)
mask_task8_2 = mask_task8_1 == mask_task8_1.max()
print("Koluma z największą liczbą zer:",columnsNames[mask_task8_2][0])

#TASK 9
even_sums = data[::2,:].sum(axis=0)
odd_sums = data[1::2,:].sum(axis=0)
mask_task9 = even_sums>odd_sums
print("Kolumny gdzie suma elementów na pozycjach parzystych jest większa:",columnsNames[mask_task9])