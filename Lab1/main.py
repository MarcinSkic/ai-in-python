#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:06:52 2022

@author: marcinskic
"""

# TASK 1
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

data_excel = pd.read_excel("practice_lab_1.xlsx",engine="openpyxl")
data_excel = data_excel.iloc[:100,:7]

columnsNames = list(data_excel.columns)
data = data_excel.values

even = data[::2,:]
odd = data[1::2,:]
subtracted = even-odd

# TASK 2
mean_task2 = data.mean()
std_task2 = data.std()

arr_task2 = (data - mean_task2)/std_task2

# TASK 3
arr_task3 = (data-data.mean(axis=0))/(np.spacing(data.std(axis=0))+data.std(axis=0))

changeFactor = data.mean(axis=0) / (np.spacing(data.std(axis=0))+data.std(axis=0))