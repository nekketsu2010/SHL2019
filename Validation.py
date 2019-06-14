import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

fileNames = ['val_Hips.csv', 'val_Torso.csv']#, 'val_Hand.csv']

data = np.loadtxt('val_Bag.csv', delimiter=',', skiprows=1)
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

#標準化のみ（画像のRGBに正規化はいいらしいので）
with open('stdFile.binaryFile', 'rb') as file:
    stdsc = pickle.load(file)
X_std = stdsc.transform(X)
print(X_std[0:5])

#ここでnanをなおす
# X_std[np.isnan(X_std)] = 0
X_std[np.isnan(X_std)] = 0

#X_1で使うパラメータは加速度とジャイロ
X_1 = X_std[:, [0, 1, 2, 3, 7, 8]]

#１つ目のモデルで予測開始！
# with open('model1.binaryFile', 'rb') as file:
#     clf = pickle.load(file)
# print("predict1するよ")
# predict1 = clf.predict(X_1)
# with open('predict1.binaryFile', 'wb') as file:
#     pickle.dump(predict1, file)

with open('predict1.binaryFile', 'rb') as file:
    predict1 = pickle.load(file)

#１つ目のモデルで１と識別したサンプルインデックスを取り出す
indexes = np.where(predict1 == 1)
X_2 = X_std[indexes]
X_2 = X_2[:, [0, 1, 2, 3, 4, 5]]
# ２つ目のモデルで予測開始！
with open('model2.binaryFile', 'rb') as file:
    clf = pickle.load(file)
print("predict2するよ")
predict2 = clf.predict(X_2)
with open('predict2.binaryFile', 'wb') as file:
    pickle.dump(predict2, file)

i = 0
for index in indexes:
    for ind in index:
        predict1[ind] = predict2[i]
        i += 1

print(accuracy_score(Y, predict1))
print(confusion_matrix(Y, predict1))