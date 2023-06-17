#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:07:20 2022

@author: marcinskic
"""
#%%Obliczanie metryk klasyfikacji
TN = 7
FP = 26
FN = 17
TP = 73

sensitivity = TP/(TP+FN)
precision = TP/(TP+FP)
specificity = TN/(FP+TN)
accuracy = (TP+TN)/(TP+FN+FP+TN)
F1 = 2*(sensitivity*precision)/(sensitivity+precision)