import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import decimal

fileNames = ['train_hips.csv', 'train_torso.csv']

data = np.loadtxt('train_bag.csv', delimiter=',', skiprows=1)
label = np.array(data[:, 0:1], dtype=int)
feautures = np.array(data[:, 1:])
X = feautures
Y = label

for fileName in fileNames:
    # data = pd.read_csv(fileName)
    # data = data[['XY_accel_mean','XY_accel_variance','Z_accel_mean','Z_accel_variance','XYZ_mag_mean', 'XYZ_mag_variance','pressure_difference','XYZ_gyro_mean','XYZ_gyro_variance']]
    data = np.loadtxt(fileName, delimiter=',', skiprows=1)
    label = np.array(data[:, 0:1], dtype=int)
    feautures = np.array(data[:, 1:])
    X = np.vstack((X, feautures))
    Y = np.vstack((Y, label))

#Yは横行列になおす
Y = Y.reshape((Y.shape[0], ))

#ここで乗り物（車、バス、電車、バス）のラベルの値を１とする
Y_1 = Y.copy()
Y_1[Y_1==5] = 1
Y_1[Y_1==6] = 1
Y_1[Y_1==7] = 1
Y_1[Y_1==8] = 1

#標準化のみ（画像のRGBに正規化はいいらしいので）
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
print(X_std[0:5])
with open('stdFile.binaryfile', 'wb') as file:
    pickle.dump(stdsc, file)

#ここでnanをなおす
# X_std[np.isnan(X_std)] = 0
X_std[np.isnan(X_std)] = 0

#X_1で使うパラメータは加速度とジャイロ
X_1 = X_std[:, [0, 1, 2, 3, 7, 8]]


# #学習開始！
# clf1 = svm.SVC()
# clf1.fit(X_1, Y_1)
# with open('model1.binaryfile', 'wb') as file:
#     pickle.dump(clf1, file)

print("１つ目のモデルの学習終わった")

#２つ目のモデルを学習する
#２つ目のモデルのパラメータは地磁気だけ
#２つ目のモデルはstill、car、bus、train、subwayを分けるので、そのサンプルだけ学習に使う
print(type(Y))
index1 = np.where(Y == 1)
index5 = np.where(Y == 5)
index6 = np.where(Y == 6)
index7 = np.where(Y == 7)
index8 = np.where(Y == 8)
X_2 = X_std[index1]
X_2 = np.vstack((X_2, X_std[index5]))
X_2 = np.vstack((X_2, X_std[index6]))
X_2 = np.vstack((X_2, X_std[index7]))
X_2 = np.vstack((X_2, X_std[index8]))
X_2 = X_2[:, [0, 1, 2, 3, 4, 5]]
Y_2 = Y[index1]
Y_2 = np.hstack((Y_2, Y[index5]))
Y_2 = np.hstack((Y_2, Y[index6]))
Y_2 = np.hstack((Y_2, Y[index7]))
Y_2 = np.hstack((Y_2, Y[index8]))
#学習開始！
clf2 = svm.SVC()
clf2.fit(X_2, Y_2)
with open('model2.binaryfile', 'wb') as file:
    pickle.dump(clf2, file)
