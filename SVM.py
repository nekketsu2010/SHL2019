import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import decimal

fileNames = ['train_hips.npy', 'train_torso.npy']

data = np.loadtxt('train_bag.npy', delimiter=',', skiprows=1)
label = data[:, 0]
# features xy_accel_mean variance
# z_accel_mean variance
# xyz_gra_mean variance
# xyz_gyr_mean ariance
# xyz_mag_mean variance
# pressure_difference
feautures = data[:, [1, 2, 3, 4, 9, 10]] #加速度と地磁気だけを取り出した
X = feautures
Y = label

for fileName in fileNames:
    data = np.loadtxt(fileName, delimiter=',', skiprows=1)
    label = data[:, 0]
    feautures = data[:, [1, 2, 3, 4, 9, 10]]  # 加速度と地磁気だけを取り出した
    X = np.vstack((X, feautures))
    Y = np.hstack((Y, label))


#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))

#標準化のみ（画像のRGBに正規化はいいらしいので）
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
print(X_std[0:5])
with open('data_0615/stdFile.binaryfile', 'wb') as file:
    pickle.dump(stdsc, file)

#ここでnanをなおす
# X_std[np.isnan(X_std)] = 0
X_std[np.isnan(X_std)] = -1

#学習開始！
clf2 = svm.SVC(verbose=True)
clf2.fit(X_std, Y)
with open('model_next.binaryfile', 'wb') as file:
    pickle.dump(clf2, file)