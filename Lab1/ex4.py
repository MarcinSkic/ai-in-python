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
test = data_frame.corr()

for i in test:
    print (test[i])