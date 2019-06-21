import pandas as pd
import os
import math
from statistics import mean, median, variance, stdev
import numpy as np

files = os.listdir("SVM_Data")

matrix = []
for file in files:
    df = pd.read_csv("SVM_Data/" + file)

    Label = df.Label.values
    X_Acc = df.Acc_x.values
    Y_Acc = df.Acc_y.values
    Z_Acc = df.Acc_z.values

    xy_accel = []
    for x_acc, y_acc in zip(X_Acc, Y_Acc):
        mixedAccel = math.sqrt(x_acc * x_acc + y_acc * y_acc)
        xy_accel.append(mixedAccel)

    xy_mean = mean(xy_accel)
    xy_variance = variance(xy_accel)
    z_mean = mean(Z_Acc)
    z_variance = variance(Z_Acc)

    mat = [Label[0], xy_mean, xy_variance, z_mean, z_variance]
    matrix.append(mat)
    print(file + "終わった")

matrix = np.array(matrix)
np.save("train_torso", matrix)