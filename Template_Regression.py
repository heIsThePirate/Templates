# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:40:28 2019

@author: mohit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'D:\\documents\\ML\\Udemy Machine Learning A-Z Hands-On Python & R In Data Science\\Udemy Machine Learning A-Z Hands-On Python & R In Data Science\\Files\\Machine-Learning-A-Z-Template-Folder\\Machine Learning A-Z Template Folder\\'
os.chdir(path)

# Import the dataset
dataset = pd.read_csv('.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , -1].values

# Split into test and train sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Apply Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)