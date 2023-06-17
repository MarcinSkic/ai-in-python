#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:54:48 2022

@author: marcinskic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_excel("practice_lab_1.xlsx",engine="openpyxl")
corr_arr = data_frame.corr()

data = data_frame.values

fig, ax = plt.subplots(7,7, figsize=(35,35))
x = np.arange(0,100,1)

for i in range(7):
    for z in range(7): #W celu uniknięcia powtórzeń można użyć range(i,7,1)
        y1 = data[:,i]
        y2 = data[:,z]
        ax[i,z].scatter(x,y1)
        ax[i,z].scatter(x,y2)
        ax[i,z].set_title("Kolumna {} vs Kolumna {}".format(i+1,z+1))
        #ax[i,z].set_ylim([-1,100]) opcjonalnie w celu wzrokowego 
        #potwierdzenia poprawnosci algorytmu
