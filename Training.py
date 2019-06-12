import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fileNames = ['train_bag.csv', 'train_hips.csv', 'train_torso.csv']
X = []
Y = []
X_train = []
X_test = []
Y_train = []
Y_test = []

for fileName in fileNames:
    data = pd.read_csv(fileName)
    data = data[['XY_accel_mean','XY_accel_variance','Z_accel_mean','Z_accel_variance','XYZ_mag_mean', 'XYZ_mag_variance','pressure_difference','XYZ_gyro_mean','XYZ_gyro_variance']]
    print(data)


