import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import decimal

fileNames = ['train_hips.npy', 'train_torso.npy']

data = np.load('train_bag.npy')
label = data[:, 0]
# features xy_accel_mean variance
# z_accel_mean variance
# xyz_gra_mean variance
# xyz_gyr_mean ariance
# xyz_mag_mean variance
# pressure_difference
feautures = data[:, 1:]
X = feautures
Y = label

for fileName in fileNames:
    data = np.load(fileName)
    label = data[:, 0]
    feautures = data[:, 1:]
    X = np.vstack((X, feautures))
    Y = np.hstack((Y, label))


#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))

#標準化のみ（画像のRGBに正規化はいいらしいので）
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
print(X_std[0:5])
with open('stdFile.binaryfile', 'wb') as file:
    pickle.dump(stdsc, file)

#ここでnanの行は消す
deleteIndex = np.isnan(X_std)
X_std = np.delete(X_std, deleteIndex, 0)
Y = np.delete(Y, deleteIndex, 0)

print(X_std.shape)
print(Y.shape)


learn_label = [5, 6, 7, 8] #still,car,bus,train,subway
index = np.where(Y == 1)
X_2 = X_std[index]
Y_2 = Y[index]
for label in learn_label:
    index = np.where(Y == label)
    X_2 = np.vstack((X_2, X_std[index]))
    Y_2 = np.hstack((Y_2, Y[index]))

#carとbusのラベルを2にする
#同様にtrainとsubwayのラベルを3にする
Y_2[Y_2==5] = 2
Y_2[Y_2==6] = 2
Y_2[Y_2==7] = 3
Y_2[Y_2==8] = 3

#センサは加速度と地磁気のみ使う
X_2 = X_2[:, [0,1,2,3,8,9]]

print(X_2.shape)
print(Y_2.shape)
#学習開始！
clf2 = svm.SVC(verbose=True)
clf2.fit(X_2, Y_2)
with open('model2.binaryfile', 'wb') as file:
    pickle.dump(clf2, file)